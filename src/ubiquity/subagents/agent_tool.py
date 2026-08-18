"""The Agent tool, which dispatches work to a subagent.

A subagent is a nested agent run with its own message history, its own tool subset, and its own turn budget. Only the
subagent's final text returns to the parent, which is the point: the parent's
context stays clean while the subagent does verbose exploratory work.

Isolation is deliberate and partial. The subagent gets a fresh `ToolContext`
so its file-read bookkeeping cannot satisfy a parent's read-before-write
check, but it shares the parent's permission context, because a subagent that
could widen its own permissions would be an escalation path.

The todo list is not inherited either. A subagent keeps its own, keyed by
agent id, so a delegated side task cannot edit the plan its parent is still
working through.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

from ..hooks.types import HookInput
from ..todos import key_for, store_for
from ..tool import Tool, ToolContext, ValidationError
from ..types import PermissionResult, PermissionResultAllow, ToolOutput

if TYPE_CHECKING:
    from ..options import AgentDefinition

MAX_DEPTH = 3


class AgentInput(BaseModel):
    description: str = Field(description="A short 3-5 word description of the task.")
    prompt: str = Field(description="The full task for the subagent to perform.")
    subagent_type: str = Field(
        default="general-purpose", description="Which agent definition to use."
    )


class AgentTool(Tool[AgentInput]):
    """Dispatch a task to a subagent and return only its final answer."""

    name = "Agent"
    description = (
        "Launch a subagent to handle a multi-step task. The subagent runs with "
        "its own context and returns only its final report, which keeps "
        "verbose intermediate work out of this conversation. Give it a "
        "complete, self-contained prompt naming exactly what it should report "
        "back: it cannot ask you follow-up questions and does not see this "
        "conversation. Its report reaches you and not the user, so summarize "
        "what matters in your own reply."
    )
    input_model = AgentInput
    search_hint = "subagent delegate spawn task parallel"

    def __init__(self, definitions: dict[str, AgentDefinition] | None = None) -> None:
        self._definitions = definitions or {}

    async def prompt(self, ctx: ToolContext) -> str:
        """List the available agent types alongside the base description.

        The types are sorted rather than left in the order the caller happened
        to build the dictionary in. Tool descriptions are part of the cached
        prefix, so a set of agents that renders in a different order between
        runs costs a full cache miss for a difference the model cannot see.
        """
        if not self._definitions:
            return self.description
        listed = "\n".join(
            f"  - {name}: {self._definitions[name].description}"
            for name in sorted(self._definitions)
        )
        return f"{self.description}\n\nAvailable subagent types:\n{listed}"

    def is_concurrency_safe(self, args: AgentInput) -> bool:
        """Subagents are independent runs and may proceed in parallel."""
        return True

    def describe_call(self, args: AgentInput) -> str:
        return f"Agent({args.subagent_type}): {args.description}"

    def permission_rule_content(self, args: AgentInput) -> list[str]:
        return [args.subagent_type]

    async def validate_input(
        self, args: AgentInput, ctx: ToolContext
    ) -> ValidationError | None:
        if self._definitions and args.subagent_type not in self._definitions:
            available = ", ".join(sorted(self._definitions)) or "none"
            return ValidationError(
                message=(
                    f"Unknown subagent_type {args.subagent_type!r}. "
                    f"Available types: {available}."
                )
            )
        if ctx.extra.get("agent_depth", 0) >= MAX_DEPTH:
            return ValidationError(
                message=(
                    f"Subagent nesting limit of {MAX_DEPTH} reached. Complete "
                    "this work directly rather than delegating further."
                )
            )
        return None

    async def check_permissions(
        self, args: AgentInput, ctx: ToolContext
    ) -> PermissionResult:
        """Spawning a subagent is allowed; its own tool calls are still checked.

        The subagent inherits the parent's permission context, so every tool it
        calls goes through the same pipeline. Gating the spawn itself would
        prompt twice for the same underlying work.
        """
        return PermissionResultAllow(reason="subagent inherits parent permissions")

    async def call(self, args: AgentInput, ctx: ToolContext) -> ToolOutput:
        from ..client import run_subagent

        definition = self._definitions.get(args.subagent_type)
        depth = ctx.extra.get("agent_depth", 0) + 1
        inherited = {k: v for k, v in ctx.extra.items() if k != "todos"}

        child_ctx = replace(
            ctx,
            file_state={},
            agent_id=f"{args.subagent_type}-{depth}-{uuid4().hex[:6]}",
            agent_type=args.subagent_type,
            extra={**inherited, "agent_depth": depth},
        )

        if ctx.hooks is not None and ctx.hooks.has("SubagentStart"):
            await ctx.hooks.run(
                HookInput(
                    hook_event_name="SubagentStart",
                    session_id=ctx.session_id,
                    cwd=str(ctx.cwd),
                    permission_mode=ctx.permissions.mode,
                    agent_id=child_ctx.agent_id,
                    agent_type=args.subagent_type,
                    prompt=args.prompt,
                )
            )

        report, denials = await run_subagent(args.prompt, child_ctx, definition)
        self.discard_todos(child_ctx)

        if ctx.hooks is not None and ctx.hooks.has("SubagentStop"):
            await ctx.hooks.run(
                HookInput(
                    hook_event_name="SubagentStop",
                    session_id=ctx.session_id,
                    cwd=str(ctx.cwd),
                    permission_mode=ctx.permissions.mode,
                    agent_id=child_ctx.agent_id,
                    agent_type=args.subagent_type,
                    message=report,
                )
            )

        return ToolOutput(
            content=report,
            metadata={
                "subagent_type": args.subagent_type,
                "depth": depth,
                "denials": len(denials),
            },
        )

    def discard_todos(self, child_ctx: ToolContext) -> None:
        """Drop the subagent's todo list now that it has reported.

        An agent id names one delegated task and never recurs, so the list has
        no future reader, and a session that spawns hundreds of agents would
        otherwise accumulate one dead list per agent forever.
        """
        store = store_for(child_ctx.options)
        if store is None or child_ctx.agent_id is None:
            return
        store.clear(
            key_for(
                child_ctx.options,
                child_ctx.session_id,
                child_ctx.cwd,
                child_ctx.agent_id,
            ),
            child_ctx.cwd,
        )


__all__ = ["AgentTool", "AgentInput", "MAX_DEPTH"]
