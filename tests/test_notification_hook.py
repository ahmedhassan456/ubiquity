"""Dispatch of the `Notification` event.

`Notification` is the one event a host can subscribe to in order to learn that
a run has stopped making progress on its own, so what matters is that it
reaches every such point: a call waiting on approval, and a run that ended by
any route other than answering the prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ubiquity import HookMatcher, HookOutput, Options, PermissionResultAllow, summon
from ubiquity.hooks import HookInput, HookRegistry


def _recorder() -> tuple[list[HookInput], Any]:
    """A callback that records every payload it is handed."""
    seen: list[HookInput] = []

    async def record(payload: HookInput) -> None:
        seen.append(payload)
        return None

    return seen, record


def _reasons(seen: list[HookInput]) -> list[str]:
    """The `reason` of each recorded notification."""
    return [str(p.extra.get("reason")) for p in seen]


def _hooks(seen_for: Any) -> list[HookMatcher]:
    """Subscribe `seen_for` to Notification."""
    return [HookMatcher("Notification", [seen_for])]


async def test_notify_is_a_no_op_without_a_subscriber() -> None:
    """The common case must not build a payload nobody reads."""
    registry = HookRegistry(())
    await registry.notify("x", reason="error", session_id="s", cwd="/tmp")


async def test_notify_carries_the_reason_in_extra() -> None:
    seen, record = _recorder()
    registry = HookRegistry(_hooks(record))
    await registry.notify(
        "Bash needs permission", reason="permission_required", session_id="s", cwd="/tmp"
    )
    assert len(seen) == 1
    assert seen[0].hook_event_name == "Notification"
    assert seen[0].message == "Bash needs permission"
    assert seen[0].extra["reason"] == "permission_required"


async def test_a_tool_matcher_filters_notifications() -> None:
    """A notification carrying a tool name is matched like a tool event."""
    seen, record = _recorder()
    registry = HookRegistry([HookMatcher("Notification", [record], matcher="Write")])
    await registry.notify(
        "Bash needs permission",
        reason="permission_required",
        session_id="s",
        cwd="/tmp",
        tool_name="Bash",
    )
    assert seen == []


def _one_call(tool: str, args: dict[str, Any]) -> FunctionModel:
    """A model that calls `tool` once, then answers."""
    calls = {"n": 0}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name=tool, args=args)])
        return ModelResponse(parts=[TextPart(content="done")])

    return FunctionModel(respond)


def _loops(tool: str, args: dict[str, Any]) -> FunctionModel:
    """A model that never stops calling `tool`."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(tool_name=tool, args=args)])

    return FunctionModel(respond)


async def _run(prompt: str, options: Options) -> list[Any]:
    return [m async for m in summon(prompt, options)]


async def test_a_call_awaiting_approval_notifies(tmp_path: Path) -> None:
    seen, record = _recorder()

    async def approve(name: str, args: dict[str, Any], ctx: Any) -> Any:
        return PermissionResultAllow()

    await _run(
        "run it",
        Options(
            model=_one_call("Bash", {"command": "echo hi"}),
            cwd=tmp_path,
            permission_mode="default",
            can_use_tool=approve,
            hooks=_hooks(record),
            persist_session=False,
            persist_todos=False,
            max_turns=4,
        ),
    )
    assert _reasons(seen) == ["permission_required"]
    assert seen[0].tool_name == "Bash"
    assert seen[0].tool_input == {"command": "echo hi"}


async def test_a_call_that_cannot_be_approved_still_notifies(tmp_path: Path) -> None:
    """A run stalled for want of a handler is exactly what a host must hear."""
    seen, record = _recorder()

    await _run(
        "run it",
        Options(
            model=_one_call("Bash", {"command": "echo hi"}),
            cwd=tmp_path,
            permission_mode="default",
            hooks=_hooks(record),
            persist_session=False,
            persist_todos=False,
            max_turns=4,
        ),
    )
    assert "permission_required" in _reasons(seen)
    assert "no can_use_tool handler" in str(seen[0].message)


async def test_exhausting_the_turn_budget_notifies(tmp_path: Path) -> None:
    seen, record = _recorder()

    await _run(
        "run it",
        Options(
            model=_loops("Bash", {"command": "echo hi"}),
            cwd=tmp_path,
            permission_mode="bypassPermissions",
            hooks=_hooks(record),
            persist_session=False,
            persist_todos=False,
            max_turns=2,
        ),
    )
    assert "max_turns" in _reasons(seen)


async def test_a_stop_hook_holding_the_run_open_notifies(tmp_path: Path) -> None:
    seen, record = _recorder()

    async def keep_going(payload: HookInput) -> HookOutput:
        return HookOutput(decision="block", reason="not finished yet")

    await _run(
        "answer",
        Options(
            model=FunctionModel(
                lambda messages, info: ModelResponse(parts=[TextPart(content="hi")])
            ),
            cwd=tmp_path,
            tools=[],
            hooks=[*_hooks(record), HookMatcher("Stop", [keep_going])],
            persist_session=False,
            persist_todos=False,
            max_turns=4,
        ),
    )
    assert _reasons(seen) == ["stopped"]
    assert seen[0].message == "not finished yet"


async def test_a_successful_run_notifies_nothing(tmp_path: Path) -> None:
    """Notification means attention is needed; an ordinary run needs none."""
    seen, record = _recorder()

    await _run(
        "answer",
        Options(
            model=FunctionModel(
                lambda messages, info: ModelResponse(parts=[TextPart(content="hi")])
            ),
            cwd=tmp_path,
            tools=[],
            hooks=_hooks(record),
            persist_session=False,
            persist_todos=False,
            max_turns=4,
        ),
    )
    assert seen == []
