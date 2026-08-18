"""Tests for hook matching and dispatch."""

from __future__ import annotations

from ubiquity.hooks import HookInput, HookMatcher, HookOutput, HookRegistry


def _payload(**kwargs) -> HookInput:
    defaults = {
        "hook_event_name": "PreToolUse",
        "session_id": "s",
        "cwd": "/tmp",
    }
    return HookInput(**{**defaults, **kwargs})


def test_matcher_wildcard_and_none_match_everything() -> None:
    assert HookMatcher("PreToolUse", [], matcher=None).matches("Bash") is True
    assert HookMatcher("PreToolUse", [], matcher="*").matches("Bash") is True


def test_matcher_alternation() -> None:
    m = HookMatcher("PreToolUse", [], matcher="Write|Edit")
    assert m.matches("Write") is True
    assert m.matches("Edit") is True
    assert m.matches("Bash") is False


def test_matcher_with_tool_name_none_does_not_match() -> None:
    """Non-tool events carry no tool name, so a tool matcher must not fire."""
    assert HookMatcher("Stop", [], matcher="Bash").matches(None) is False


async def test_callbacks_run_in_registration_order() -> None:
    order: list[int] = []

    async def first(_):
        order.append(1)
        return None

    async def second(_):
        order.append(2)
        return None

    registry = HookRegistry([HookMatcher("PreToolUse", [first, second])])
    await registry.run(_payload(tool_name="Bash"))
    assert order == [1, 2]


async def test_block_short_circuits_remaining_hooks() -> None:
    """A veto must not be overturned by a later hook."""
    ran: list[str] = []

    async def blocker(_):
        ran.append("blocker")
        return HookOutput(decision="block", reason="nope")

    async def after(_):
        ran.append("after")
        return HookOutput(decision="approve")

    registry = HookRegistry([HookMatcher("PreToolUse", [blocker, after])])
    result = await registry.run(_payload(tool_name="Bash"))

    assert result.decision == "block"
    assert result.reason == "nope"
    assert ran == ["blocker"]


async def test_failing_hook_is_skipped_not_fatal() -> None:
    """A broken observability hook must not take down the run."""

    async def broken(_):
        raise RuntimeError("boom")

    async def healthy(_):
        return HookOutput(additional_context="still here")

    registry = HookRegistry([HookMatcher("PreToolUse", [broken, healthy])])
    result = await registry.run(_payload(tool_name="Bash"))
    assert result.additional_context == "still here"


async def test_additional_context_accumulates() -> None:
    async def one(_):
        return HookOutput(additional_context="alpha")

    async def two(_):
        return HookOutput(additional_context="beta")

    registry = HookRegistry([HookMatcher("PreToolUse", [one, two])])
    result = await registry.run(_payload(tool_name="Bash"))
    assert result.additional_context == "alpha\nbeta"


async def test_updated_input_is_visible_to_later_hooks() -> None:
    """A later hook sees the rewrite performed by an earlier one."""
    seen: list[dict] = []

    async def rewrite(_):
        return HookOutput(updated_input={"command": "safe"})

    async def observe(payload):
        seen.append(payload.tool_input)
        return None

    registry = HookRegistry([HookMatcher("PreToolUse", [rewrite, observe])])
    result = await registry.run(
        _payload(tool_name="Bash", tool_input={"command": "unsafe"})
    )
    assert seen == [{"command": "safe"}]
    assert result.updated_input == {"command": "safe"}


async def test_non_matching_tool_is_not_dispatched() -> None:
    ran: list[str] = []

    async def cb(_):
        ran.append("x")
        return None

    registry = HookRegistry([HookMatcher("PreToolUse", [cb], matcher="Write")])
    await registry.run(_payload(tool_name="Bash"))
    assert ran == []


async def test_events_are_isolated() -> None:
    ran: list[str] = []

    async def cb(_):
        ran.append("pre")
        return None

    registry = HookRegistry([HookMatcher("PreToolUse", [cb])])
    await registry.run(_payload(hook_event_name="PostToolUse", tool_name="Bash"))
    assert ran == []


async def test_has_reports_registration() -> None:
    registry = HookRegistry([HookMatcher("Stop", [])])
    assert registry.has("Stop") is True
    assert registry.has("PreToolUse") is False


async def test_continue_false_short_circuits() -> None:
    ran: list[str] = []

    async def stopper(_):
        return HookOutput(continue_=False, reason="halt")

    async def after(_):
        ran.append("after")
        return None

    registry = HookRegistry([HookMatcher("Stop", [stopper, after])])
    result = await registry.run(_payload(hook_event_name="Stop"))
    assert result.continue_ is False
    assert ran == []


async def test_empty_registry_returns_neutral_output() -> None:
    result = await HookRegistry().run(_payload(tool_name="Bash"))
    assert result.decision is None
    assert result.continue_ is True
    assert result.additional_context is None
