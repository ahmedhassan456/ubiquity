"""Resolution of `allowed_tools` and `disallowed_tools` into tools and rules.

These lists are permission rules, not plain tool names, so they feed two
different mechanisms: a bare ``Bash`` decides whether the tool exists at all,
while ``Bash(rm:*)`` leaves it in place and speaks to the permission engine
about one call. Getting that split wrong is silent — a scoped deny rule that
is treated as a name matches no tool and blocks nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ubiquity import Options, summon
from ubiquity.client import _build_context
from ubiquity.hooks.registry import HookRegistry
from ubiquity.tools import names_tool, resolve_tools, rule_names


def test_rule_names_splits_bare_from_scoped() -> None:
    bare, scoped = rule_names(["Read", "Bash(git:*)", "Write"])
    assert bare == {"Read", "Write"}
    assert scoped == {"Bash"}


def test_scoped_deny_rule_keeps_the_tool_available() -> None:
    names = [t.name for t in resolve_tools(disallowed=["Bash(rm:*)"])]
    assert "Bash" in names


def test_bare_deny_rule_removes_the_tool() -> None:
    names = [t.name for t in resolve_tools(disallowed=["Bash"])]
    assert "Bash" not in names


def test_scoped_allow_rule_keeps_the_tool_it_names() -> None:
    names = [t.name for t in resolve_tools(allowed=["Bash(echo:*)"])]
    assert names == ["Bash"]


def test_allow_list_limits_the_run_to_the_tools_it_names() -> None:
    names = sorted(t.name for t in resolve_tools(allowed=["Bash", "Read"]))
    assert names == ["Bash", "Read"]


def test_bare_deny_beats_allow_for_the_same_tool() -> None:
    names = [t.name for t in resolve_tools(allowed=["Bash"], disallowed=["Bash"])]
    assert names == []


def test_names_tool_honors_aliases() -> None:
    read = next(t for t in resolve_tools() if t.name == "Read")
    assert names_tool(read, {"Read"})
    assert not names_tool(read, {"Bash"})


def test_options_rules_reach_the_permission_context(tmp_path: Path) -> None:
    options = Options(
        model="test",
        cwd=tmp_path,
        allowed_tools=["Bash(git:*)"],
        disallowed_tools=["Bash(rm:*)"],
        ask_tools=["Bash(git push:*)"],
    )
    ctx = _build_context(options, "s", HookRegistry(()))
    assert ctx.permissions.allow_rules == {"Bash(git:*)"}
    assert ctx.permissions.deny_rules == {"Bash(rm:*)"}
    assert ctx.permissions.ask_rules == {"Bash(git push:*)"}


def _one_call(tool: str, args: dict[str, Any]) -> FunctionModel:
    """A model that calls `tool` once, then answers."""
    calls = {"n": 0}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name=tool, args=args)])
        return ModelResponse(parts=[TextPart(content="done")])

    return FunctionModel(respond)


async def test_scoped_deny_rule_blocks_the_matching_command(tmp_path: Path) -> None:
    target = tmp_path / "secret.txt"
    target.write_text("classified\n")
    messages = [
        m
        async for m in summon(
            "delete it",
            Options(
                model=_one_call("Bash", {"command": f"rm {target}"}),
                cwd=tmp_path,
                permission_mode="bypassPermissions",
                disallowed_tools=["Bash(rm:*)"],
                persist_session=False,
                persist_todos=False,
                max_turns=4,
            ),
        )
    ]
    assert target.exists()
    denials = messages[-1].permission_denials
    assert denials and "rm:*" in denials[0].message


async def test_scoped_allow_rule_authorizes_the_matching_command(tmp_path: Path) -> None:
    messages = [
        m
        async for m in summon(
            "run it",
            Options(
                model=_one_call("Bash", {"command": "echo permitted"}),
                cwd=tmp_path,
                permission_mode="default",
                allowed_tools=["Bash(echo:*)"],
                persist_session=False,
                persist_todos=False,
                max_turns=4,
            ),
        )
    ]
    results = [m for m in messages if m.type == "tool_result"]
    assert results and not results[0].output.is_error
    assert "permitted" in results[0].output.content


async def test_scoped_allow_rule_does_not_authorize_other_commands(tmp_path: Path) -> None:
    messages = [
        m
        async for m in summon(
            "run it",
            Options(
                model=_one_call("Bash", {"command": "cat /etc/passwd"}),
                cwd=tmp_path,
                permission_mode="default",
                allowed_tools=["Bash(echo:*)"],
                persist_session=False,
                persist_todos=False,
                max_turns=4,
            ),
        )
    ]
    assert messages[-1].permission_denials


async def test_bare_allow_rule_authorizes_the_tool(tmp_path: Path) -> None:
    messages = [
        m
        async for m in summon(
            "run it",
            Options(
                model=_one_call("Bash", {"command": "echo permitted"}),
                cwd=tmp_path,
                permission_mode="default",
                allowed_tools=["Bash"],
                persist_session=False,
                persist_todos=False,
                max_turns=4,
            ),
        )
    ]
    results = [m for m in messages if m.type == "tool_result"]
    assert results and not results[0].output.is_error


async def test_ask_rule_denies_when_there_is_nobody_to_prompt(tmp_path: Path) -> None:
    messages = [
        m
        async for m in summon(
            "push it",
            Options(
                model=_one_call("Bash", {"command": "git push origin main"}),
                cwd=tmp_path,
                permission_mode="bypassPermissions",
                ask_tools=["Bash(git push:*)"],
                persist_session=False,
                persist_todos=False,
                max_turns=4,
            ),
        )
    ]
    assert messages[-1].permission_denials
