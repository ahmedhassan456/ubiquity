"""Hook event payloads and outputs.

Hooks observe and can veto the loop at fixed points.

Events and when they fire:
    PreToolUse          before a tool runs, may block or rewrite its input
    PostToolUse         after a tool succeeds
    PostToolUseFailure  after a tool raises or returns an error
    PermissionRequest   when a call is about to prompt the user
    PermissionDenied    after a call is denied
    UserPromptSubmit    when a user turn enters the loop, may inject context
    SessionStart        once, before the first model request
    SessionEnd          once, after the run finishes
    Stop                when the model stops without calling a tool
    SubagentStart       before a subagent begins
    SubagentStop        after a subagent finishes
    PreCompact          before the conversation is compacted
    PostCompact         after the conversation is compacted
    Notification        when a run needs the host's attention

`Notification` is the informational channel: a call waiting on approval, or a
run that ended by exhausting its turns, raising, or being held open by a `Stop`
hook. Its payload carries `extra["reason"]` — one of ``permission_required``,
``max_turns``, ``error``, or ``stopped`` — so a callback can route without
parsing the message. Nothing it returns is acted on, because it reports rather
than gates.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

HookEvent: TypeAlias = Literal[
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "PermissionRequest",
    "PermissionDenied",
    "UserPromptSubmit",
    "SessionStart",
    "SessionEnd",
    "Stop",
    "SubagentStart",
    "SubagentStop",
    "PreCompact",
    "PostCompact",
    "Notification",
]

HOOK_EVENTS: tuple[HookEvent, ...] = (
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "PermissionRequest",
    "PermissionDenied",
    "UserPromptSubmit",
    "SessionStart",
    "SessionEnd",
    "Stop",
    "SubagentStart",
    "SubagentStop",
    "PreCompact",
    "PostCompact",
    "Notification",
)


@dataclass(slots=True)
class HookInput:
    """The payload handed to a hook callback.

    `tool_name`, `tool_input`, `tool_use_id`, and `tool_output` are populated
    only for the tool-scoped events. `agent_id` is set only when the hook fires
    inside a subagent, which is how a callback distinguishes subagent activity
    from the main thread.
    """

    hook_event_name: HookEvent
    session_id: str
    cwd: str
    permission_mode: str = "default"
    agent_id: str | None = None
    agent_type: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_use_id: str | None = None
    tool_output: Any = None
    prompt: str | None = None
    message: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HookOutput:
    """What a hook callback may return.

    All fields are optional; returning None from a callback is equivalent to
    returning an empty `HookOutput`.

    `decision` set to ``block`` vetoes the action. For `PreToolUse` that means
    the tool does not run and `reason` is returned to the model in its place.
    For `Stop` it means the model is told to keep working.

    `updated_input` rewrites the tool input, and is honored only for
    `PreToolUse`. `additional_context` is appended to the conversation and is
    honored for `UserPromptSubmit`, `SessionStart`, and `PostToolUse`.
    """

    decision: Literal["block", "approve"] | None = None
    reason: str | None = None
    updated_input: dict[str, Any] | None = None
    additional_context: str | None = None
    system_message: str | None = None
    suppress_output: bool = False
    continue_: bool = True


HookCallback: TypeAlias = Callable[[HookInput], Awaitable[HookOutput | None]]


@dataclass(slots=True)
class HookMatcher:
    """Binds a callback to an event, optionally filtered by tool name.

    `matcher` is matched against the tool name for tool-scoped events. It
    accepts an exact name, a ``|``-separated alternation such as
    ``Write|Edit``, or ``*`` for every tool. When None the hook fires for every
    invocation of its event.
    """

    event: HookEvent
    hooks: list[HookCallback]
    matcher: str | None = None

    def matches(self, tool_name: str | None) -> bool:
        """True when this matcher applies to the given tool name."""
        if self.matcher is None or self.matcher == "*":
            return True
        if tool_name is None:
            return False
        return tool_name in {p.strip() for p in self.matcher.split("|")}


__all__ = [
    "HookEvent",
    "HOOK_EVENTS",
    "HookInput",
    "HookOutput",
    "HookCallback",
    "HookMatcher",
]
