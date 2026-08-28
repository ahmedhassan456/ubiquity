"""Tests for the terminal client.

The CLI owns no agent logic, so what is worth testing is the translation at
its two edges: arguments into `Options`, and the message stream into terminal
output or JSON. The permission handler is the third: it is the only place the
CLI can change what a run is allowed to do.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ubiquity import Options
from ubiquity.cli.commands import ReplState, dispatch, is_command
from ubiquity.cli.main import _read_prompt, _run_turn, build_parser, options_from
from ubiquity.cli.prompts import terminal_handler
from ubiquity.cli.render import Renderer
from ubiquity.tool import PermissionContext, ToolContext


def scripted(*turns: list[Any]) -> FunctionModel:
    """Build a model that replays `turns`, one response per model request."""
    calls = {"n": 0}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        index = min(calls["n"], len(turns) - 1)
        calls["n"] += 1
        return ModelResponse(parts=list(turns[index]))

    return FunctionModel(respond)


def renderer(output_format: str = "text", **kwargs: Any) -> tuple[Renderer, io.StringIO]:
    buffer = io.StringIO()
    return Renderer(output_format, color=False, stream=buffer, **kwargs), buffer


def state_for(model: FunctionModel, cwd: Path, **kwargs: Any) -> ReplState:
    return ReplState(
        options=Options(
            model=model,
            cwd=cwd,
            persist_session=False,
            persist_todos=False,
            **kwargs,
        )
    )


def context_for(options: Options) -> ToolContext:
    return ToolContext(
        cwd=options.resolved_cwd(),
        options=options,
        permissions=PermissionContext(mode=options.permission_mode),
        session_id="test",
    )


class TestArguments:
    def test_rules_split_on_commas(self) -> None:
        args = build_parser().parse_args(
            ["--allowed-tools", "Read, Bash(git:*)", "hello"]
        )
        assert options_from(args).allowed_tools == ["Read", "Bash(git:*)"]

    def test_json_output_disables_partial_messages(self) -> None:
        args = build_parser().parse_args(["--output-format", "json", "hi"])
        assert options_from(args).include_partial_messages is False

    def test_sources_reach_every_discovery_root(self) -> None:
        args = build_parser().parse_args(["--sources", "project,local", "hi"])
        options = options_from(args)
        assert list(options.setting_sources) == ["project", "local"]
        assert list(options.skill_sources) == ["project", "local"]
        assert list(options.agent_sources) == ["project", "local"]
        assert list(options.memory_sources) == ["project", "local"]

    def test_prompt_joins_arguments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        args = build_parser().parse_args(["fix", "the", "bug"])
        assert _read_prompt(args) == "fix the bug"


class TestRendering:
    async def test_stream_json_emits_every_message(self, tmp_path: Path) -> None:
        model = scripted([TextPart("hello")])
        view, buffer = renderer("stream-json")
        await _run_turn("hi", state_for(model, tmp_path), view, interactive=False)

        kinds = [json.loads(line)["type"] for line in buffer.getvalue().splitlines()]
        assert kinds[0] == "system"
        assert kinds[-1] == "result"
        assert "assistant" in kinds

    async def test_json_prints_only_the_result(self, tmp_path: Path) -> None:
        model = scripted([TextPart("hello")])
        view, buffer = renderer("json")
        await _run_turn("hi", state_for(model, tmp_path), view, interactive=False)

        lines = buffer.getvalue().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["type"] == "result"

    async def test_text_prints_assistant_and_tool_call(self, tmp_path: Path) -> None:
        (tmp_path / "note.txt").write_text("contents")
        model = scripted(
            [ToolCallPart("Read", {"file_path": str(tmp_path / "note.txt")})],
            [TextPart("read it")],
        )
        view, buffer = renderer("text")
        state = state_for(model, tmp_path, permission_mode="bypassPermissions")
        code = await _run_turn("read", state, view, interactive=False)

        output = buffer.getvalue()
        assert code == 0
        assert "● Read" in output
        assert "read it" in output

    async def test_result_carries_session_and_turns(self, tmp_path: Path) -> None:
        model = scripted([TextPart("done")])
        view, _ = renderer("text")
        state = state_for(model, tmp_path)
        await _run_turn("hi", state, view, interactive=False)

        assert state.session_id
        assert state.turns == 1


class TestPermissionHandler:
    async def test_deny_is_reported_to_the_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda *a: "n")
        view, _ = renderer("text")
        options = Options(cwd=tmp_path)
        result = await terminal_handler(view)(
            "Write", {"file_path": "x", "content": "y"}, context_for(options)
        )
        assert result.behavior == "deny"

    async def test_always_writes_a_rule_into_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda *a: "a")
        view, _ = renderer("text")
        ctx = context_for(Options(cwd=tmp_path))
        result = await terminal_handler(view)("Write", {"file_path": "x"}, ctx)

        assert result.behavior == "allow"
        assert "Write" in ctx.permissions.allow_rules

    async def test_answers_are_written_into_the_tool_input(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda *a: "2")
        view, _ = renderer("text")
        tool_input = {
            "questions": [
                {
                    "question": "Which format?",
                    "header": "format",
                    "options": [
                        {"label": "JSON", "description": "machine"},
                        {"label": "Text", "description": "human"},
                    ],
                    "multi_select": False,
                }
            ]
        }
        result = await terminal_handler(view)(
            "AskUserQuestion", tool_input, context_for(Options(cwd=tmp_path))
        )
        assert result.behavior == "allow"
        assert result.updated_input["answers"] == {"Which format?": "Text"}

    async def test_free_text_answers_pass_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda *a: "neither, use yaml")
        view, _ = renderer("text")
        tool_input = {
            "questions": [
                {
                    "question": "Which format?",
                    "header": "format",
                    "options": [
                        {"label": "JSON", "description": "machine"},
                        {"label": "Text", "description": "human"},
                    ],
                }
            ]
        }
        result = await terminal_handler(view)(
            "AskUserQuestion", tool_input, context_for(Options(cwd=tmp_path))
        )
        assert result.updated_input["answers"] == {"Which format?": "neither, use yaml"}


class TestCommands:
    def test_slash_prefix_marks_a_command(self) -> None:
        assert is_command("/help")
        assert not is_command("help me")

    def test_mode_and_model_change_the_next_turn(self, tmp_path: Path) -> None:
        view, _ = renderer("text")
        state = state_for(scripted([TextPart("x")]), tmp_path)

        assert dispatch("/mode plan", state, view) == "continue"
        assert dispatch("/model openai:gpt-5", state, view) == "continue"
        assert state.options.permission_mode == "plan"
        assert state.options.model == "openai:gpt-5"

    def test_invalid_mode_is_rejected(self, tmp_path: Path) -> None:
        view, buffer = renderer("text")
        state = state_for(scripted([TextPart("x")]), tmp_path)

        dispatch("/mode nonsense", state, view)
        assert state.options.permission_mode == "default"
        assert "must be one of" in buffer.getvalue()

    def test_new_forgets_the_session(self, tmp_path: Path) -> None:
        view, _ = renderer("text")
        state = state_for(scripted([TextPart("x")]), tmp_path)
        state.session_id = "abc"

        dispatch("/new", state, view)
        assert state.session_id is None

    def test_exit_stops_the_loop(self, tmp_path: Path) -> None:
        view, _ = renderer("text")
        state = state_for(scripted([TextPart("x")]), tmp_path)
        assert dispatch("/exit", state, view) == "exit"

    def test_unknown_command_is_not_sent_to_the_model(self, tmp_path: Path) -> None:
        view, buffer = renderer("text")
        state = state_for(scripted([TextPart("x")]), tmp_path)

        assert dispatch("/bogus", state, view) == "continue"
        assert "unknown command" in buffer.getvalue()
