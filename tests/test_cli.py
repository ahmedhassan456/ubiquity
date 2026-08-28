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
from ubiquity.cli import ui
from ubiquity.cli.commands import COMMANDS, MODES, ReplState, dispatch, is_command
from ubiquity.cli.main import _read_prompt, _run_turn, build_parser, options_from
from ubiquity.cli.prompts import terminal_handler
from ubiquity.cli.render import Renderer
from ubiquity.tool import PermissionContext, ToolContext


@pytest.fixture(autouse=True)
def captured_console() -> io.StringIO:
    """Point the shared console at a buffer, so tests read what was drawn."""
    buffer = io.StringIO()
    ui.set_console(file=buffer, color=False, width=120)
    return buffer


def scripted(*turns: list[Any]) -> FunctionModel:
    """Build a model that replays `turns`, one response per model request."""
    calls = {"n": 0}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        index = min(calls["n"], len(turns) - 1)
        calls["n"] += 1
        return ModelResponse(parts=list(turns[index]))

    return FunctionModel(respond)


def renderer(output_format: str = "text", **kwargs: Any) -> Renderer:
    return Renderer(output_format, **kwargs)


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
    async def test_stream_json_emits_every_message(
        self, tmp_path: Path, captured_console: io.StringIO
    ) -> None:
        model = scripted([TextPart("hello")])
        view = renderer("stream-json")
        await _run_turn("hi", state_for(model, tmp_path), view, interactive=False)

        kinds = [
            json.loads(line)["type"]
            for line in captured_console.getvalue().splitlines()
        ]
        assert kinds[0] == "system"
        assert kinds[-1] == "result"
        assert "assistant" in kinds

    async def test_json_prints_only_the_result(
        self, tmp_path: Path, captured_console: io.StringIO
    ) -> None:
        model = scripted([TextPart("hello")])
        view = renderer("json")
        await _run_turn("hi", state_for(model, tmp_path), view, interactive=False)

        lines = captured_console.getvalue().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["type"] == "result"

    async def test_text_prints_assistant_and_tool_call(
        self, tmp_path: Path, captured_console: io.StringIO
    ) -> None:
        (tmp_path / "note.txt").write_text("contents")
        model = scripted(
            [ToolCallPart("Read", {"file_path": str(tmp_path / "note.txt")})],
            [TextPart("read it")],
        )
        view = renderer("text")
        state = state_for(model, tmp_path, permission_mode="bypassPermissions")
        code = await _run_turn("read", state, view, interactive=False)

        output = captured_console.getvalue()
        assert code == 0
        assert "Read" in output
        assert "read it" in output

    async def test_result_carries_session_and_turns(self, tmp_path: Path) -> None:
        model = scripted([TextPart("done")])
        view = renderer("text")
        state = state_for(model, tmp_path)
        await _run_turn("hi", state, view, interactive=False)

        assert state.session_id
        assert state.turns == 1


class TestPermissionHandler:
    async def test_deny_is_reported_to_the_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda *a: "n")
        view = renderer("text")
        options = Options(cwd=tmp_path)
        result = await terminal_handler(view)(
            "Write", {"file_path": "x", "content": "y"}, context_for(options)
        )
        assert result.behavior == "deny"

    async def test_always_writes_a_rule_into_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda *a: "a")
        view = renderer("text")
        ctx = context_for(Options(cwd=tmp_path))
        result = await terminal_handler(view)("Write", {"file_path": "x"}, ctx)

        assert result.behavior == "allow"
        assert "Write" in ctx.permissions.allow_rules

    async def test_answers_are_written_into_the_tool_input(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda *a: "2")
        view = renderer("text")
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
        view = renderer("text")
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

    async def test_mode_and_model_change_the_next_turn(self, tmp_path: Path) -> None:
        view = renderer("text")
        state = state_for(scripted([TextPart("x")]), tmp_path)

        assert await dispatch("/mode plan", state, view) == "continue"
        assert await dispatch("/model openai:gpt-5", state, view) == "continue"
        assert state.options.permission_mode == "plan"
        assert state.options.model == "openai:gpt-5"

    async def test_invalid_mode_is_rejected(
        self, tmp_path: Path, captured_console: io.StringIO
    ) -> None:
        view = renderer("text")
        state = state_for(scripted([TextPart("x")]), tmp_path)

        await dispatch("/mode nonsense", state, view)
        assert state.options.permission_mode == "default"
        assert "must be one of" in captured_console.getvalue()

    async def test_new_forgets_the_session(self, tmp_path: Path) -> None:
        view = renderer("text")
        state = state_for(scripted([TextPart("x")]), tmp_path)
        state.session_id = "abc"

        await dispatch("/new", state, view)
        assert state.session_id is None

    async def test_exit_stops_the_loop(self, tmp_path: Path) -> None:
        view = renderer("text")
        state = state_for(scripted([TextPart("x")]), tmp_path)
        assert await dispatch("/exit", state, view) == "exit"

    async def test_unknown_command_is_not_sent_to_the_model(
        self, tmp_path: Path, captured_console: io.StringIO
    ) -> None:
        view = renderer("text")
        state = state_for(scripted([TextPart("x")]), tmp_path)

        assert await dispatch("/bogus", state, view) == "continue"
        assert "unknown command" in captured_console.getvalue()


class TestSetupWizard:
    @pytest.fixture(autouse=True)
    def home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Point the wizard at a throwaway home directory."""
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        monkeypatch.delenv("UBIQUITY_MODEL", raising=False)
        return tmp_path

    def test_unconfigured_until_a_model_exists(self) -> None:
        from ubiquity.cli.setup import is_configured, save_config

        assert is_configured() is False
        save_config({"model": "groq:openai/gpt-oss-120b"})
        assert is_configured() is True

    def test_env_var_alone_counts_as_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ubiquity.cli.setup import is_configured

        monkeypatch.setenv("UBIQUITY_MODEL", "openai:gpt-5")
        assert is_configured() is True

    def test_saving_preserves_hand_written_keys(self) -> None:
        from ubiquity.cli.setup import config_path, load_config, save_config

        config_path().parent.mkdir(parents=True, exist_ok=True)
        config_path().write_text(
            json.dumps({"permissions": {"deny": ["Bash(rm:*)"]}, "env": {"TZ": "UTC"}})
        )
        save_config({"model": "openai:gpt-5"})

        stored = load_config()
        assert stored["model"] == "openai:gpt-5"
        assert stored["permissions"]["deny"] == ["Bash(rm:*)"]
        assert stored["env"] == {"TZ": "UTC"}

    def test_credential_check_names_the_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ubiquity.cli.setup import credential_missing, env_var_for

        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        assert env_var_for("groq:openai/gpt-oss-120b") == "GROQ_API_KEY"
        assert credential_missing("groq:openai/gpt-oss-120b") == "GROQ_API_KEY"

        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        assert credential_missing("groq:openai/gpt-oss-120b") is None

    def test_unknown_provider_asserts_nothing_about_credentials(self) -> None:
        from ubiquity.cli.setup import credential_missing, env_var_for

        assert env_var_for("some-local-thing:model") is None
        assert credential_missing("some-local-thing:model") is None

    async def test_wizard_writes_the_answers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ubiquity.cli import setup as wizard
        from ubiquity.cli.setup import load_config, run_wizard

        choices = iter(["groq", "groq:openai/gpt-oss-120b", "acceptEdits"])
        monkeypatch.setattr(wizard.ui, "ask_choice", lambda *a, **k: _reply(choices))
        monkeypatch.setattr(wizard.ui, "ask_text", lambda *a, **k: _reply(iter([""])))
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")

        changes = await run_wizard(first_run=True)

        assert changes["model"] == "groq:openai/gpt-oss-120b"
        stored = load_config()
        assert stored["model"] == "groq:openai/gpt-oss-120b"
        assert stored["permissions"]["defaultMode"] == "acceptEdits"

    async def test_abandoned_wizard_writes_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ubiquity.cli import setup as wizard
        from ubiquity.cli.setup import config_path, run_wizard

        monkeypatch.setattr(wizard.ui, "ask_choice", lambda *a, **k: _reply(iter([""])))

        assert await run_wizard() == {}
        assert not config_path().exists()


async def _reply(values: Any) -> str:
    """Stand in for a `ui.ask_*` coroutine, returning the next scripted answer."""
    return next(values)


class TestKeyReader:
    def test_control_characters_get_names(self) -> None:
        from ubiquity.cli import keys

        assert keys._name("\r") == keys.ENTER
        assert keys._name("\n") == keys.ENTER
        assert keys._name(" ") == keys.SPACE
        assert keys._name("\x7f") == keys.BACKSPACE
        assert keys._name("\x03") == keys.INTERRUPT
        assert keys._name("k") == "k"

    def test_both_arrow_encodings_are_recognized(self) -> None:
        from ubiquity.cli import keys

        assert keys._SEQUENCES["[A"] == keys.UP
        assert keys._SEQUENCES["OA"] == keys.UP
        assert keys._SEQUENCES["[B"] == keys.DOWN
        assert keys._SEQUENCES["OB"] == keys.DOWN

    def test_a_pipe_is_not_a_keyboard(self) -> None:
        from ubiquity.cli import keys

        assert keys.supported(io.StringIO()) is False


class TestMenu:
    def test_typing_narrows_the_list(self) -> None:
        options = [("openai:gpt-5", "flagship"), ("groq:kimi", "strong at tools")]

        visible, cursor = ui._filter(options, "groq", 1)
        assert [row[0] for row in visible] == ["groq:kimi"]
        assert cursor == 1

        visible, _ = ui._filter(options, "flagship", 0)
        assert [row[0] for row in visible] == ["openai:gpt-5"]

        visible, _ = ui._filter(options, "", 0)
        assert len(visible) == 2

    async def test_without_a_tty_the_menu_falls_back_to_numbers(
        self, monkeypatch: pytest.MonkeyPatch, captured_console: io.StringIO
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda *a: "2")

        picked = await ui.select(
            "Which model?", [("openai:gpt-5", "flagship"), ("groq:kimi", "fast")]
        )
        assert picked == ["groq:kimi"]
        assert "flagship" in captured_console.getvalue()

    async def test_hotkey_falls_back_to_a_line_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda *a: "a")
        assert await ui.hotkey("allow?", accepted="yan", default="y") == "a"

    async def test_multi_select_answers_join_with_commas(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from ubiquity.cli import prompts

        async def picked(*args: Any, **kwargs: Any) -> list[str]:
            return ["Markdown", "HTML"]

        monkeypatch.setattr(prompts.ui, "select", picked)
        tool_input = {
            "questions": [
                {
                    "question": "Which formats?",
                    "header": "formats",
                    "options": [
                        {"label": "JSON", "description": "machine"},
                        {"label": "Markdown", "description": "readme"},
                        {"label": "HTML", "description": "docs"},
                    ],
                    "multi_select": True,
                }
            ]
        }
        result = await terminal_handler(renderer("text"))(
            "AskUserQuestion", tool_input, context_for(Options(cwd=tmp_path))
        )
        assert result.updated_input["answers"] == {"Which formats?": "Markdown, HTML"}


class TestCompletion:
    """Slash-command completion in the REPL input line."""

    def complete(self, state: ReplState, text: str) -> list[tuple[str, str]]:
        from prompt_toolkit.document import Document

        from ubiquity.cli.completion import build_completer

        completer = build_completer(state)
        found = completer.get_completions(Document(text, len(text)), None)
        return [(c.text, c.display_meta_text) for c in found]

    def test_a_bare_slash_offers_every_command(self, tmp_path: Path) -> None:
        state = state_for(scripted([TextPart(content="hi")]), tmp_path)
        offered = [name for name, _ in self.complete(state, "/")]
        assert "/help" in offered
        assert "/model" in offered
        assert len(offered) == len(COMMANDS)

    def test_completions_carry_what_the_command_does(self, tmp_path: Path) -> None:
        """A list of bare names would be no more discoverable than /help."""
        state = state_for(scripted([TextPart(content="hi")]), tmp_path)
        meta = dict(self.complete(state, "/"))
        assert meta["/help"] == "show this list"

    def test_the_offered_name_excludes_the_argument_placeholder(
        self, tmp_path: Path
    ) -> None:
        """`/model <name>` is help text; completing it would insert junk."""
        state = state_for(scripted([TextPart(content="hi")]), tmp_path)
        assert "/model" in [name for name, _ in self.complete(state, "/mod")]

    def test_typing_narrows_the_list(self, tmp_path: Path) -> None:
        state = state_for(scripted([TextPart(content="hi")]), tmp_path)
        offered = [name for name, _ in self.complete(state, "/mo")]
        assert set(offered) == {"/model", "/mode"}

    def test_plain_text_completes_to_nothing(self, tmp_path: Path) -> None:
        """The completer must stay out of the way of ordinary prompts."""
        state = state_for(scripted([TextPart(content="hi")]), tmp_path)
        assert self.complete(state, "what does this repo do") == []

    def test_mode_completes_its_permission_modes(self, tmp_path: Path) -> None:
        state = state_for(scripted([TextPart(content="hi")]), tmp_path)
        offered = [name for name, _ in self.complete(state, "/mode ")]
        assert set(offered) == set(MODES)

    def test_a_mode_argument_is_filtered_by_what_is_typed(
        self, tmp_path: Path
    ) -> None:
        state = state_for(scripted([TextPart(content="hi")]), tmp_path)
        assert [n for n, _ in self.complete(state, "/mode acc")] == ["acceptEdits"]

    def test_model_completes_from_the_wizard_suggestions(self, tmp_path: Path) -> None:
        from ubiquity.cli.setup import SUGGESTED

        state = state_for(scripted([TextPart(content="hi")]), tmp_path)
        offered = [name for name, _ in self.complete(state, "/model openai:")]
        assert offered
        assert all(name.startswith("openai:") for name in offered)
        assert set(offered) <= {name for name, _ in SUGGESTED["openai"]}

    def test_resume_completes_stored_sessions(self, tmp_path: Path) -> None:
        """The ids are unguessable, so completion is the only usable path."""
        from ubiquity import SessionStore

        store = SessionStore(tmp_path / "sessions")
        store.append("sess-abc", tmp_path, TextPart(content="hi"))
        state = state_for(scripted([TextPart(content="hi")]), tmp_path)
        state.options.session_dir = tmp_path / "sessions"
        assert "sess-abc" in [n for n, _ in self.complete(state, "/resume ")]

    def test_a_command_with_no_argument_completes_nothing(
        self, tmp_path: Path
    ) -> None:
        state = state_for(scripted([TextPart(content="hi")]), tmp_path)
        assert self.complete(state, "/cost ") == []

    def test_the_reader_falls_back_when_there_is_no_terminal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Piped input and CI have no line editor; the REPL still has to run."""
        from ubiquity.cli import completion

        state = state_for(scripted([TextPart(content="hi")]), tmp_path)
        monkeypatch.setattr(
            completion.LineReader, "_build", lambda self: setattr(self, "session", None)
        )
        reader = completion.LineReader(state)
        assert reader.session is None

    def test_history_lives_beside_the_settings(self) -> None:
        from ubiquity.cli.completion import history_path

        assert history_path().parent.name == ".ubiquity"
