"""The tool protocol.

A `Tool` is a Pydantic-validated input schema plus a `call` coroutine, with
metadata the permission engine and the loop consult to decide how the call may
run. Nothing here is tied to a particular front end: a tool describes what it
is and what it does, and the caller decides how any of it is displayed.

The metadata hooks and what they control:
    is_read_only          exempts the call from plan mode and edit gating
    is_concurrency_safe   allows the call to run in parallel with its siblings
    is_destructive        marks irreversible operations for stricter prompting
    check_permissions     tool-specific permission logic, run before the rules
    validate_input        input rejection that reaches the model, not the user
"""

from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel

from .types import (
    PermissionMode,
    PermissionResult,
    PermissionResultPassthrough,
    PermissionRuleValue,
    ToolOutput,
)

if TYPE_CHECKING:
    from .hooks.registry import HookRegistry
    from .options import Options

InputT = TypeVar("InputT", bound=BaseModel)


@dataclass(slots=True)
class ValidationError:
    """A tool-input rejection that is reported to the model, not the user."""

    message: str
    error_code: int = 1


@dataclass
class PermissionContext:
    """The mutable permission state for one run.

    Rules are keyed by behavior. `additional_directories` widens the set of
    paths tools may touch beyond `cwd`.
    """

    mode: PermissionMode = "default"
    allow_rules: set[str] = field(default_factory=set)
    deny_rules: set[str] = field(default_factory=set)
    ask_rules: set[str] = field(default_factory=set)
    additional_directories: set[Path] = field(default_factory=set)
    bypass_available: bool = False

    def rules_for(self, behavior: str) -> set[str]:
        """Return the mutable rule set backing the given behavior."""
        return {
            "allow": self.allow_rules,
            "deny": self.deny_rules,
            "ask": self.ask_rules,
        }[behavior]

    def parsed_rules(self, behavior: str) -> list[PermissionRuleValue]:
        """Return the rules for a behavior, parsed into structured form."""
        return [PermissionRuleValue.parse(r) for r in self.rules_for(behavior)]


@dataclass
class ToolContext:
    """Everything a tool needs from its surroundings.

    `file_state` records the mtime and content hash of every file read this
    run, so `Edit` and `Write` can refuse to clobber a file the agent has
    not seen.

    `abort` is the caller's cancellation event, carried from `Options.abort`.
    It is checked before every tool call, and again by tools that can wait
    indefinitely: cooperative cancellation is the only kind available to a tool
    that owns a subprocess, since killing the coroutine would leave the process
    behind. A tool that returns promptly needs no check.
    """

    cwd: Path
    options: Options
    permissions: PermissionContext
    session_id: str
    file_state: dict[Path, FileState] = field(default_factory=dict)
    hooks: HookRegistry | None = None
    agent_id: str | None = None
    agent_type: str | None = None
    tool_use_id: str | None = None
    abort: asyncio.Event | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def is_path_allowed(self, path: Path) -> bool:
        """True when `path` is inside cwd or an explicitly added directory."""
        resolved = path.resolve()
        roots = [self.cwd.resolve(), *(d.resolve() for d in self.additional_dirs)]
        return any(resolved == r or r in resolved.parents for r in roots)

    @property
    def additional_dirs(self) -> set[Path]:
        """Directories tools may access in addition to `cwd`."""
        return self.permissions.additional_directories


@dataclass(slots=True)
class FileState:
    """A snapshot of a file at the moment a tool last read it.

    `is_partial_view` is set when the read was bounded by offset or limit. A
    partial view does not authorize a write, because the writer never saw the
    whole file and would silently discard the part it did not read.
    """

    mtime: float
    size: int
    content_hash: str
    is_partial_view: bool = False


class Tool(abc.ABC, Generic[InputT]):
    """Base class for every tool the agent can call.

    Subclasses declare `name`, `input_model`, and implement `call`. Everything
    else has a safe default: not read-only, not concurrency-safe, not
    destructive, and permitted by tool-specific logic so that the rule engine
    makes the final call.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    input_model: ClassVar[type[BaseModel]]

    search_hint: ClassVar[str | None] = None
    defer_loading: ClassVar[bool] = False
    max_result_chars: ClassVar[int] = 30_000
    aliases: ClassVar[tuple[str, ...]] = ()

    @abc.abstractmethod
    async def call(self, args: InputT, ctx: ToolContext) -> ToolOutput:
        """Execute the tool and return its model-facing output."""

    async def prompt(self, ctx: ToolContext) -> str:
        """Return the tool description shown to the model.

        Defaults to the class-level `description`. Override when the text must
        vary by run, for example to list the directories the tool may access.
        """
        return self.description

    def is_read_only(self, args: InputT) -> bool:
        """True when the call cannot mutate anything outside the process."""
        return False

    def is_concurrency_safe(self, args: InputT) -> bool:
        """True when the call may run in parallel with sibling tool calls.

        Defaults to False so that a tool must opt in to parallelism. Read-only
        tools generally override this to return True.
        """
        return False

    def is_destructive(self, args: InputT) -> bool:
        """True when the call is irreversible, such as a delete or overwrite."""
        return False

    async def validate_input(self, args: InputT, ctx: ToolContext) -> ValidationError | None:
        """Reject malformed input before permissions are consulted.

        The returned message goes back to the model as a tool error, so it
        should explain how to fix the call. Return None to accept.
        """
        return None

    async def check_permissions(self, args: InputT, ctx: ToolContext) -> PermissionResult:
        """Apply tool-specific permission logic.

        Runs after the tool-level deny and ask rules but before mode bypass
        and the tool-level allow rules. Returning `deny` is final and survives
        `bypassPermissions`. The default defers to the rule engine.
        """
        return PermissionResultPassthrough()

    def requires_user_interaction(self) -> bool:
        """True when this tool must prompt even in `bypassPermissions` mode."""
        return False

    def permission_rule_content(self, args: InputT) -> list[str]:
        """Return the rule strings this call should be matched against.

        `Bash` returns prefixes like ``git status:*``; path-based tools return
        the path. Each string is tested against the rules registered for this
        tool name. An empty list means only bare ``ToolName`` rules can match.
        """
        return []

    def get_path(self, args: InputT) -> Path | None:
        """Return the filesystem path this call targets, when it has one."""
        return None

    def user_facing_name(self, args: InputT | None = None) -> str:
        """Return the display name for this call."""
        return self.name

    def describe_call(self, args: InputT) -> str:
        """Return a one-line summary of the call for logs and transcripts."""
        return self.name

    def matches_name(self, name: str) -> bool:
        """True when `name` is this tool's name or one of its aliases."""
        return name == self.name or name in self.aliases


__all__ = [
    "Tool",
    "ToolContext",
    "PermissionContext",
    "FileState",
    "ValidationError",
    "InputT",
]
