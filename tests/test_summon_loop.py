"""End-to-end tests for the agent loop.

Uses pydantic-ai's `FunctionModel` to script model behavior, so the loop, the
toolset bridge, permissions, and hooks are exercised together without a
network call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ubiquity import Options, summon
from ubiquity.types import PermissionResultAllow, PermissionResultDeny


def scripted(*turns: list[Any]) -> FunctionModel:
    """Build a model that replays `turns`, one response per model request."""
    calls = {"n": 0}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        index = min(calls["n"], len(turns) - 1)
        calls["n"] += 1
        return ModelResponse(parts=list(turns[index]))

    return FunctionModel(respond)


async def collect(prompt: str, options: Options) -> list[Any]:
    """Drain `summon` into a list."""
    return [m async for m in summon(prompt, options)]


async def allow_all(name: str, args: dict, ctx) -> PermissionResultAllow:
    return PermissionResultAllow()


async def deny_all(name: str, args: dict, ctx) -> PermissionResultDeny:
    return PermissionResultDeny(message="denied by policy")


async def test_text_only_run_emits_system_user_and_result(tmp_path: Path) -> None:
    messages = await collect(
        "hello",
        Options(model=scripted([TextPart(content="hi there")]), cwd=tmp_path),
    )
    kinds = [m.type for m in messages]

    assert kinds[0] == "system"
    assert kinds[-1] == "result"
    assert messages[-1].subtype == "success"
    assert messages[-1].is_error is False


async def test_system_message_describes_configuration(tmp_path: Path) -> None:
    messages = await collect(
        "hello",
        Options(
            model=scripted([TextPart(content="ok")]),
            cwd=tmp_path,
            permission_mode="acceptEdits",
        ),
    )
    system = messages[0]
    assert system.cwd == str(tmp_path.resolve())
    assert system.permission_mode == "acceptEdits"
    assert "Read" in system.tools


async def test_tool_call_runs_and_streams_use_and_result(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("file contents here")

    model = scripted(
        [ToolCallPart(tool_name="Read", args={"file_path": str(tmp_path / "hello.txt")})],
        [TextPart(content="done")],
    )
    messages = await collect(
        "read it",
        Options(model=model, cwd=tmp_path, can_use_tool=allow_all),
    )

    uses = [m for m in messages if m.type == "tool_use"]
    results = [m for m in messages if m.type == "tool_result"]

    assert len(uses) == 1
    assert uses[0].tool_name == "Read"
    assert len(results) == 1
    assert "file contents here" in results[0].output.content
    assert messages[-1].subtype == "success"


async def test_denied_tool_is_reported_not_fatal(tmp_path: Path) -> None:
    """A denial returns to the model as a tool result; the run still succeeds.

    Uses Bash because it always escalates. Read inside the working directory
    auto-allows in its own `check_permissions` and never reaches `can_use_tool`.
    """
    model = scripted(
        [ToolCallPart(tool_name="Bash", args={"command": "cat secrets.txt"})],
        [TextPart(content="understood, I was denied")],
    )
    messages = await collect(
        "read it",
        Options(model=model, cwd=tmp_path, can_use_tool=deny_all),
    )

    result = messages[-1]
    assert result.subtype == "success"
    assert [d.tool_name for d in result.permission_denials] == ["Bash"]
    assert "denied by policy" in result.permission_denials[0].message


async def test_read_inside_cwd_never_reaches_can_use_tool(tmp_path: Path) -> None:
    """Reads within the working directory are auto-allowed by the tool itself."""
    (tmp_path / "hello.txt").write_text("contents")
    consulted: list[str] = []

    async def recording(name: str, args: dict, ctx):
        consulted.append(name)
        return PermissionResultDeny(message="should not be reached")

    model = scripted(
        [ToolCallPart(tool_name="Read", args={"file_path": str(tmp_path / "hello.txt")})],
        [TextPart(content="done")],
    )
    messages = await collect(
        "read it", Options(model=model, cwd=tmp_path, can_use_tool=recording)
    )

    assert consulted == []
    assert not messages[-1].permission_denials


async def test_missing_can_use_tool_denies_rather_than_hanging(tmp_path: Path) -> None:
    """With nothing to prompt with, an `ask` must deny rather than block."""
    model = scripted(
        [ToolCallPart(tool_name="Bash", args={"command": "echo hi"})],
        [TextPart(content="ok")],
    )
    messages = await collect("run it", Options(model=model, cwd=tmp_path))

    result = messages[-1]
    assert result.permission_denials
    assert "no `can_use_tool`" in result.permission_denials[0].message


async def test_allow_rule_avoids_prompting(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    model = scripted(
        [ToolCallPart(tool_name="Read", args={"file_path": str(tmp_path / "a.txt")})],
        [TextPart(content="ok")],
    )
    messages = await collect("read", Options(model=model, cwd=tmp_path))

    assert not messages[-1].permission_denials
    assert any(m.type == "tool_result" for m in messages)


async def test_plan_mode_blocks_mutating_tool(tmp_path: Path) -> None:
    model = scripted(
        [
            ToolCallPart(
                tool_name="Write",
                args={"file_path": str(tmp_path / "new.txt"), "content": "x"},
            )
        ],
        [TextPart(content="cannot write in plan mode")],
    )
    messages = await collect(
        "write a file",
        Options(model=model, cwd=tmp_path, permission_mode="plan", can_use_tool=allow_all),
    )

    assert (tmp_path / "new.txt").exists() is False
    assert messages[-1].permission_denials


async def test_disallowed_tool_is_not_offered(tmp_path: Path) -> None:
    messages = await collect(
        "hello",
        Options(
            model=scripted([TextPart(content="ok")]),
            cwd=tmp_path,
            disallowed_tools=["Bash"],
        ),
    )
    assert "Bash" not in messages[0].tools
    assert "Read" in messages[0].tools


async def test_allowed_tools_restricts_suite(tmp_path: Path) -> None:
    messages = await collect(
        "hello",
        Options(
            model=scripted([TextPart(content="ok")]),
            cwd=tmp_path,
            allowed_tools=["Read", "Glob"],
        ),
    )
    assert set(messages[0].tools) == {"Read", "Glob"}


async def test_max_turns_is_enforced(tmp_path: Path) -> None:
    """A model that never stops calling tools must terminate the run."""
    model = scripted(
        [ToolCallPart(tool_name="Glob", args={"pattern": "*"})],
    )
    messages = await collect(
        "loop forever",
        Options(model=model, cwd=tmp_path, max_turns=3, can_use_tool=allow_all),
    )

    result = messages[-1]
    assert result.subtype == "error_max_turns"
    assert result.is_error is True
    assert result.num_turns <= 4


async def test_invalid_tool_arguments_are_returned_to_model(tmp_path: Path) -> None:
    model = scripted(
        [ToolCallPart(tool_name="Read", args={"file_path": str(tmp_path / "ghost.txt")})],
        [TextPart(content="that file does not exist")],
    )
    messages = await collect(
        "read a missing file",
        Options(model=model, cwd=tmp_path, can_use_tool=allow_all),
    )
    assert messages[-1].subtype == "success"


async def test_write_then_result_reaches_disk(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    model = scripted(
        [ToolCallPart(tool_name="Write", args={"file_path": str(target), "content": "written"})],
        [TextPart(content="wrote it")],
    )
    await collect(
        "write",
        Options(model=model, cwd=tmp_path, permission_mode="acceptEdits"),
    )
    assert target.read_text() == "written"


async def test_an_allow_decision_can_rewrite_the_tool_input(tmp_path: Path) -> None:
    """The tool must receive what was authorized, not what was asked for.

    A decision that rewrites the input is the only way a host can narrow a call
    instead of refusing it outright, so a rewrite that shows up in the stream
    but not in the execution would be worse than no rewrite at all.
    """
    target = tmp_path / "out.txt"

    async def redirect(name: str, args: dict, ctx) -> PermissionResultAllow:
        return PermissionResultAllow(
            updated_input={"file_path": str(target), "content": "authorized"}
        )

    model = scripted(
        [
            ToolCallPart(
                tool_name="Write",
                args={"file_path": str(tmp_path / "asked.txt"), "content": "asked"},
            )
        ],
        [TextPart(content="done")],
    )
    messages = await collect(
        "write",
        Options(model=model, cwd=tmp_path, can_use_tool=redirect),
    )

    assert target.read_text() == "authorized"
    assert not (tmp_path / "asked.txt").exists()
    uses = [m for m in messages if m.type == "tool_use"]
    assert uses[0].tool_input["content"] == "authorized"
