"""Core serializable types for the ubiquity SDK.

These are the types that cross the SDK boundary: permission decisions, hook payloads, and the
`SDKMessage` union that `summon()` streams to callers.

Permission modes:
    default            prompt for anything not pre-approved
    acceptEdits        auto-accept file edits, prompt for the rest
    bypassPermissions  allow everything, requires explicit opt-in
    plan               read-only, no mutating tool may run
    dontAsk            never prompt, deny anything not pre-approved
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias
from uuid import uuid4

PermissionMode: TypeAlias = Literal[
    "default",
    "acceptEdits",
    "bypassPermissions",
    "plan",
    "dontAsk",
]

PermissionBehavior: TypeAlias = Literal["allow", "deny", "ask"]

PermissionUpdateDestination: TypeAlias = Literal[
    "userSettings", "projectSettings", "localSettings", "session"
]

PermissionDecisionClassification: TypeAlias = Literal[
    "user_temporary", "user_permanent", "user_reject"
]


@dataclass(frozen=True, slots=True)
class PermissionRuleValue:
    """A single permission rule, e.g. ``Bash(git status:*)``.

    `tool_name` is the tool the rule applies to; `rule_content` is the
    tool-specific matcher, the part inside the parentheses. A rule with no
    content matches every invocation of that tool.
    """

    tool_name: str
    rule_content: str | None = None

    def __str__(self) -> str:
        if self.rule_content is None:
            return self.tool_name
        return f"{self.tool_name}({self.rule_content})"

    @classmethod
    def parse(cls, raw: str) -> PermissionRuleValue:
        """Parse ``Tool(content)`` or bare ``Tool`` into a rule."""
        raw = raw.strip()
        if raw.endswith(")") and "(" in raw:
            name, _, content = raw.partition("(")
            return cls(tool_name=name.strip(), rule_content=content[:-1])
        return cls(tool_name=raw)


@dataclass(frozen=True, slots=True)
class PermissionUpdate:
    """A mutation to the permission context, requested by a tool or a hook."""

    type: Literal[
        "addRules",
        "replaceRules",
        "removeRules",
        "setMode",
        "addDirectories",
        "removeDirectories",
    ]
    destination: PermissionUpdateDestination = "session"
    rules: tuple[PermissionRuleValue, ...] = ()
    behavior: PermissionBehavior | None = None
    mode: PermissionMode | None = None
    directories: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PermissionResultAllow:
    """Allow the call.

    `updated_input` rewrites the tool input; when None the original input is
    used unchanged. `reason` is surfaced in the transcript and telemetry.
    """

    behavior: Literal["allow"] = "allow"
    updated_input: dict[str, Any] | None = None
    updated_permissions: tuple[PermissionUpdate, ...] = ()
    decision_classification: PermissionDecisionClassification | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class PermissionResultDeny:
    """Deny the call.

    When `interrupt` is set the whole turn aborts, rather than returning an
    error to the model and letting it continue.
    """

    message: str
    behavior: Literal["deny"] = "deny"
    interrupt: bool = False
    decision_classification: PermissionDecisionClassification | None = None


@dataclass(frozen=True, slots=True)
class PermissionResultAsk:
    """Actively request a prompt; escalates to `can_use_tool`.

    `suggestions` are rules to offer the user as "always allow" options.
    `bypass_immune` marks decisions that must prompt even in
    `bypassPermissions` mode, such as a user-configured content-specific ask
    rule or a safety check on a sensitive path.
    """

    message: str
    behavior: Literal["ask"] = "ask"
    suggestions: tuple[PermissionUpdate, ...] = ()
    bypass_immune: bool = False


@dataclass(frozen=True, slots=True)
class PermissionResultPassthrough:
    """The tool has no opinion; defer to the rule engine and the mode.

    Distinct from `ask`: a passthrough that survives every later step is
    converted to `ask`, but unlike a real `ask` it does not survive
    `bypassPermissions` mode or a tool-level allow rule.
    """

    message: str | None = None
    behavior: Literal["passthrough"] = "passthrough"


PermissionResult: TypeAlias = (
    PermissionResultAllow
    | PermissionResultDeny
    | PermissionResultAsk
    | PermissionResultPassthrough
)


@dataclass(slots=True)
class ToolOutput:
    """What a tool hands back to the loop.

    `content` is the model-facing payload. `display` is an optional
    human-facing rendering; when omitted, hosts fall back to `content`.
    `metadata` carries structured data for programmatic consumers and is never
    shown to the model.
    """

    content: str
    display: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False


def _new_uuid() -> str:
    return str(uuid4())


@dataclass(slots=True)
class SDKSystemMessage:
    """Emitted once at the start of a run, describing the resolved config."""

    session_id: str
    model: str
    cwd: str
    tools: list[str]
    permission_mode: PermissionMode
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    slash_commands: list[str] = field(default_factory=list)
    subtype: Literal["init"] = "init"
    type: Literal["system"] = "system"
    uuid: str = field(default_factory=_new_uuid)


@dataclass(slots=True)
class SDKUserMessage:
    """A user turn entering the loop."""

    content: str | list[dict[str, Any]]
    session_id: str = ""
    parent_tool_use_id: str | None = None
    type: Literal["user"] = "user"
    uuid: str = field(default_factory=_new_uuid)


@dataclass(slots=True)
class SDKAssistantMessage:
    """A complete assistant turn, including any tool_use blocks it issued."""

    content: list[dict[str, Any]]
    session_id: str = ""
    model: str = ""
    parent_tool_use_id: str | None = None
    type: Literal["assistant"] = "assistant"
    uuid: str = field(default_factory=_new_uuid)

    @property
    def text(self) -> str:
        """Concatenated text blocks, ignoring thinking and tool_use blocks."""
        return "".join(
            b.get("text", "") for b in self.content if b.get("type") == "text"
        )


@dataclass(slots=True)
class SDKPartialAssistantMessage:
    """A streaming delta, only emitted when `include_partial_messages` is set."""

    delta: str
    block_type: Literal["text", "thinking"] = "text"
    session_id: str = ""
    parent_tool_use_id: str | None = None
    type: Literal["stream_event"] = "stream_event"
    uuid: str = field(default_factory=_new_uuid)


@dataclass(slots=True)
class SDKToolUseMessage:
    """A tool is about to run, emitted after permissions resolve to allow."""

    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str
    session_id: str = ""
    parent_tool_use_id: str | None = None
    type: Literal["tool_use"] = "tool_use"
    uuid: str = field(default_factory=_new_uuid)


@dataclass(slots=True)
class SDKToolResultMessage:
    """The outcome of a tool call."""

    tool_name: str
    tool_use_id: str
    output: ToolOutput
    duration_ms: float = 0.0
    session_id: str = ""
    parent_tool_use_id: str | None = None
    type: Literal["tool_result"] = "tool_result"
    uuid: str = field(default_factory=_new_uuid)


@dataclass(slots=True)
class SDKPermissionDenial:
    """A record of one denied tool call, collected into the result message."""

    tool_name: str
    tool_use_id: str
    message: str
    tool_input: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SDKResultMessage:
    """Terminal message of every run, emitted on success and failure alike."""

    subtype: Literal["success", "error_max_turns", "error_during_execution"]
    result: str
    session_id: str = ""
    duration_ms: float = 0.0
    num_turns: int = 0
    total_cost_usd: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    permission_denials: list[SDKPermissionDenial] = field(default_factory=list)
    is_error: bool = False
    type: Literal["result"] = "result"
    uuid: str = field(default_factory=_new_uuid)


@dataclass(slots=True)
class SDKCompactBoundaryMessage:
    """Marks where the conversation was compacted to fit the context window."""

    pre_tokens: int
    trigger: Literal["auto", "manual"] = "auto"
    session_id: str = ""
    subtype: Literal["compact_boundary"] = "compact_boundary"
    type: Literal["system"] = "system"
    uuid: str = field(default_factory=_new_uuid)


@dataclass(slots=True)
class SDKMicrocompactMessage:
    """Marks tool results cleared in place to defer a full compaction."""

    tokens_saved: int
    cleared: int
    kept: int
    session_id: str = ""
    subtype: Literal["microcompact"] = "microcompact"
    type: Literal["system"] = "system"
    uuid: str = field(default_factory=_new_uuid)


SDKMessage: TypeAlias = (
    SDKSystemMessage
    | SDKUserMessage
    | SDKAssistantMessage
    | SDKPartialAssistantMessage
    | SDKToolUseMessage
    | SDKToolResultMessage
    | SDKResultMessage
    | SDKCompactBoundaryMessage
    | SDKMicrocompactMessage
)


__all__ = [
    "PermissionMode",
    "PermissionBehavior",
    "PermissionUpdateDestination",
    "PermissionDecisionClassification",
    "PermissionRuleValue",
    "PermissionUpdate",
    "PermissionResultAllow",
    "PermissionResultDeny",
    "PermissionResultAsk",
    "PermissionResultPassthrough",
    "PermissionResult",
    "ToolOutput",
    "SDKSystemMessage",
    "SDKUserMessage",
    "SDKAssistantMessage",
    "SDKPartialAssistantMessage",
    "SDKToolUseMessage",
    "SDKToolResultMessage",
    "SDKPermissionDenial",
    "SDKResultMessage",
    "SDKCompactBoundaryMessage",
    "SDKMicrocompactMessage",
    "SDKMessage",
]
