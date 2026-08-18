"""The bridge between ubiquity tools and pydantic-ai.

`UbiquityToolset` presents the tool suite to pydantic-ai and owns the
execution pipeline for every call:

    1. validate the raw arguments against the tool's Pydantic model
    2. run the tool's own `validate_input`
    3. fire the PreToolUse hook, which may block or rewrite the input
    4. resolve permissions, escalating to `can_use_tool` on an `ask`
    5. execute the tool
    6. fire PostToolUse, or PostToolUseFailure when the call raised

Failures at steps 1-4 are returned to the model as `ModelRetry` so it can
correct itself, rather than aborting the run. A denial is terminal for that
call but not for the turn.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from typing import Any

from pydantic import TypeAdapter, ValidationError as PydanticValidationError
from pydantic_ai import ModelRetry, RunContext, ToolDefinition, ToolsetTool
from pydantic_ai.toolsets import AbstractToolset

from .hooks.types import HookInput
from .permissions.engine import apply_mode_transformations, check_permissions
from .tool import Tool, ToolContext
from .types import (
    PermissionResult,
    PermissionResultAllow,
    PermissionResultDeny,
    SDKPermissionDenial,
    SDKToolResultMessage,
    SDKToolUseMessage,
    ToolOutput,
)

logger = logging.getLogger("ubiquity.toolset")

_ARGS_ADAPTER = TypeAdapter(dict[str, Any])


class ToolDenied(Exception):
    """Raised when a tool call is refused by a rule, a hook, or the user."""

    def __init__(self, message: str, *, interrupt: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.interrupt = interrupt


class UbiquityToolset(AbstractToolset[Any]):
    """Exposes ubiquity tools to pydantic-ai and enforces the call pipeline.

    `emit` receives SDK messages for tool use and tool results so the loop can
    stream them to the caller. `denials` accumulates refused calls for the
    terminal result message.
    """

    def __init__(
        self,
        tools: list[Tool[Any]],
        ctx: ToolContext,
        *,
        emit: Any = None,
        parent_tool_use_id: str | None = None,
    ) -> None:
        self._tools = {t.name: t for t in tools}
        self._ctx = ctx
        self._emit = emit
        self._parent_tool_use_id = parent_tool_use_id
        self.denials: list[SDKPermissionDenial] = []

    @property
    def id(self) -> str:
        return "ubiquity"

    @property
    def label(self) -> str:
        return "ubiquity tools"

    @property
    def tool_context(self) -> ToolContext:
        """The shared context handed to every tool in this toolset."""
        return self._ctx

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        """Describe every tool to the model."""
        out: dict[str, ToolsetTool[Any]] = {}
        for name, tool in self._tools.items():
            schema = tool.input_model.model_json_schema()
            schema.pop("title", None)
            tool_def = ToolDefinition(
                name=name,
                description=await tool.prompt(self._ctx),
                parameters_json_schema=schema,
                defer_loading=tool.defer_loading,
            )
            out[name] = ToolsetTool(
                toolset=self,
                tool_def=tool_def,
                max_retries=ctx.max_retries,
                args_validator=_ARGS_ADAPTER.validator,
            )
        return out

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> Any:
        """Run one tool call through the full pipeline."""
        impl = self._tools.get(name)
        if impl is None:
            raise ModelRetry(f"Unknown tool {name!r}.")
        return await self.call_with(impl, tool_args, ctx.tool_call_id)

    async def call_with(
        self,
        impl: Tool[Any],
        tool_args: dict[str, Any],
        tool_call_id: str | None = None,
    ) -> Any:
        """Run `impl` through the full pipeline and return its model-facing output.

        Public because MCP tools are executed by their own pydantic-ai toolset
        rather than by this one, and they must still be held to the same
        permissions, hooks, and message stream. Anything that reaches the model
        as a tool goes through here.

        An aborted run is refused here, before hooks and permissions run. This
        is the choke point every tool passes through, so one check covers the
        built-in suite, MCP servers, and anything a caller registers, rather
        than each tool being trusted to remember.
        """
        name = impl.name
        if self._ctx.abort is not None and self._ctx.abort.is_set():
            raise ToolDenied(f"Run aborted before {name}.", interrupt=True)
        tool_use_id = tool_call_id or f"{name}-{time.monotonic_ns()}"
        call_ctx = replace(self._ctx, tool_use_id=tool_use_id)

        args_dict = dict(tool_args)

        try:
            hook_result = await self._run_pre_tool_hook(name, args_dict, tool_use_id)
            if hook_result is not None:
                args_dict = hook_result

            parsed = self._parse(impl, args_dict)

            validation = await impl.validate_input(parsed, call_ctx)
            if validation is not None:
                raise ModelRetry(validation.message)

            await self._authorize(impl, parsed, call_ctx, args_dict, tool_use_id)
            parsed = self._parse(impl, args_dict)
        except ToolDenied as denied:
            if denied.interrupt:
                raise
            return f"Tool call denied: {denied.message}"

        self._send(
            SDKToolUseMessage(
                tool_name=name,
                tool_input=args_dict,
                tool_use_id=tool_use_id,
                session_id=call_ctx.session_id,
                parent_tool_use_id=self._parent_tool_use_id,
            )
        )

        started = time.monotonic()
        try:
            output = await impl.call(parsed, call_ctx)
        except ToolDenied:
            raise
        except Exception as exc:
            await self._run_post_tool_hook(
                "PostToolUseFailure", name, args_dict, tool_use_id, str(exc)
            )
            raise ModelRetry(f"{name} failed: {exc}") from exc

        duration_ms = (time.monotonic() - started) * 1000
        self._send(
            SDKToolResultMessage(
                tool_name=name,
                tool_use_id=tool_use_id,
                output=output,
                duration_ms=duration_ms,
                session_id=call_ctx.session_id,
                parent_tool_use_id=self._parent_tool_use_id,
            )
        )

        extra = await self._run_post_tool_hook(
            "PostToolUse", name, args_dict, tool_use_id, output.content
        )

        content = self._truncate(impl, output)
        if extra:
            content = f"{content}\n\n{extra}"
        if output.is_error:
            raise ModelRetry(content)
        return content

    def _parse(self, impl: Tool[Any], args: dict[str, Any]) -> Any:
        """Validate raw arguments against a tool's input model.

        Called again after permissions resolve, because an `allow` decision may
        rewrite the input and the tool must receive what was authorized rather
        than what was asked for.
        """
        try:
            return impl.input_model(**args)
        except PydanticValidationError as exc:
            raise ModelRetry(f"Invalid arguments for {impl.name}: {exc}") from exc

    async def _authorize(
        self,
        impl: Tool[Any],
        parsed: Any,
        call_ctx: ToolContext,
        args_dict: dict[str, Any],
        tool_use_id: str,
    ) -> None:
        """Resolve permissions, prompting via `can_use_tool` on an `ask`.

        Mutates `args_dict` in place when a decision rewrites the tool input.
        Raises `ToolDenied` when the call may not proceed.
        """
        decision = await check_permissions(impl, parsed, call_ctx)
        decision = apply_mode_transformations(decision, call_ctx)

        if decision.behavior == "ask":
            can_use_tool = call_ctx.options.can_use_tool
            await self._notify_approval_needed(
                impl.name, args_dict, tool_use_id, can_use_tool is not None
            )
            if can_use_tool is None:
                decision = PermissionResultDeny(
                    message=(
                        f"{impl.name} requires approval but no `can_use_tool` "
                        "handler is configured, so it was denied."
                    )
                )
            else:
                await self._run_permission_hook(impl.name, args_dict, tool_use_id)
                decision = await self._await_decision(
                    can_use_tool, impl, args_dict, call_ctx
                )

        if decision.behavior == "deny":
            self.denials.append(
                SDKPermissionDenial(
                    tool_name=impl.name,
                    tool_use_id=tool_use_id,
                    message=decision.message,
                    tool_input=args_dict,
                )
            )
            await self._run_permission_denied_hook(
                impl.name, args_dict, tool_use_id, decision.message
            )
            raise ToolDenied(decision.message, interrupt=decision.interrupt)

        if isinstance(decision, PermissionResultAllow) and decision.updated_input:
            args_dict.clear()
            args_dict.update(decision.updated_input)

    def _truncate(self, impl: Tool[Any], output: ToolOutput) -> str:
        """Cap a tool result at the tool's declared maximum size."""
        content = output.content
        limit = impl.max_result_chars
        if len(content) <= limit:
            return content
        return (
            content[:limit]
            + f"\n\n[Output truncated at {limit} characters "
            f"({len(content) - limit} more).]"
        )

    def _send(self, message: Any) -> None:
        """Forward an SDK message to the loop, if one is listening."""
        if self._emit is not None:
            self._emit(message)

    def _hook_payload(self, event: str, name: str, args: dict[str, Any], tid: str):
        """Build the common hook payload for a tool-scoped event."""
        return HookInput(
            hook_event_name=event,  # type: ignore[arg-type]
            session_id=self._ctx.session_id,
            cwd=str(self._ctx.cwd),
            permission_mode=self._ctx.permissions.mode,
            agent_id=self._ctx.agent_id,
            agent_type=self._ctx.agent_type,
            tool_name=name,
            tool_input=args,
            tool_use_id=tid,
        )

    async def _await_decision(
        self,
        can_use_tool: Any,
        impl: Tool[Any],
        args_dict: dict[str, Any],
        call_ctx: ToolContext,
    ) -> PermissionResult:
        """Await the host's approval without making the run unabortable.

        A prompt is the one step of the pipeline with no bound of its own: it
        ends when a person answers, and a person may never answer. Awaiting it
        bare would let a pending prompt outlive an aborted run, since an abort
        event nobody is watching cannot interrupt anything. So the wait is
        raced against `ToolContext.abort`, and against
        `Options.permission_prompt_timeout_s` when one is configured.

        That timeout is unset by default, and expiry denies rather than
        allows. A tool that ran because nobody objected in time is a tool that
        was never approved, and for `AskUserQuestion` an invented answer is
        worse still than an unanswered question.
        """
        timeout = call_ctx.options.permission_prompt_timeout_s
        abort = call_ctx.abort

        decision = asyncio.ensure_future(
            can_use_tool(impl.name, args_dict, call_ctx)
        )
        if abort is None and timeout is None:
            return await decision

        cancelled = asyncio.ensure_future(abort.wait()) if abort is not None else None
        waiters = {decision} | ({cancelled} if cancelled is not None else set())
        try:
            done, _ = await asyncio.wait(
                waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if decision in done:
                return decision.result()
            if cancelled is not None and cancelled in done:
                raise ToolDenied(
                    f"Run aborted while {impl.name} was waiting for the user.",
                    interrupt=True,
                )
            return PermissionResultDeny(
                message=(
                    f"{impl.name} was still waiting for the user after "
                    f"{timeout:g}s, so it was denied unanswered."
                )
            )
        finally:
            for task in (decision, cancelled):
                if task is not None and not task.done():
                    task.cancel()

    async def _run_pre_tool_hook(
        self, name: str, args: dict[str, Any], tid: str
    ) -> dict[str, Any] | None:
        """Fire PreToolUse; raise `ToolDenied` if a hook blocks the call."""
        if self._ctx.hooks is None or not self._ctx.hooks.has("PreToolUse"):
            return None
        result = await self._ctx.hooks.run(self._hook_payload("PreToolUse", name, args, tid))
        if result.decision == "block":
            reason = result.reason or f"{name} was blocked by a PreToolUse hook."
            self.denials.append(
                SDKPermissionDenial(
                    tool_name=name, tool_use_id=tid, message=reason, tool_input=args
                )
            )
            raise ToolDenied(reason)
        return result.updated_input

    async def _run_post_tool_hook(
        self, event: str, name: str, args: dict[str, Any], tid: str, output: Any
    ) -> str | None:
        """Fire a post-execution hook and return any context it contributed."""
        if self._ctx.hooks is None or not self._ctx.hooks.has(event):  # type: ignore[arg-type]
            return None
        payload = self._hook_payload(event, name, args, tid)
        payload.tool_output = output
        result = await self._ctx.hooks.run(payload)
        return result.additional_context

    async def _run_permission_hook(
        self, name: str, args: dict[str, Any], tid: str
    ) -> None:
        """Notify hooks that a call is about to prompt the user."""
        if self._ctx.hooks is None or not self._ctx.hooks.has("PermissionRequest"):
            return
        await self._ctx.hooks.run(self._hook_payload("PermissionRequest", name, args, tid))

    async def _notify_approval_needed(
        self, name: str, args: dict[str, Any], tid: str, prompting: bool
    ) -> None:
        """Tell the host that a call is waiting on a human decision.

        This is the moment a run stops making progress on its own, so it is
        reported whether or not a `can_use_tool` handler exists to resolve it:
        without one the call is about to be denied for want of an answer, which
        is precisely what a host watching for stalled runs needs to hear.
        """
        if self._ctx.hooks is None:
            return
        detail = (
            "waiting for approval"
            if prompting
            else "no can_use_tool handler is configured, so it will be denied"
        )
        await self._ctx.hooks.notify(
            f"{name} needs permission to run: {detail}.",
            reason="permission_required",
            session_id=self._ctx.session_id,
            cwd=str(self._ctx.cwd),
            permission_mode=self._ctx.permissions.mode,
            tool_name=name,
            tool_input=args,
            tool_use_id=tid,
        )

    async def _run_permission_denied_hook(
        self, name: str, args: dict[str, Any], tid: str, message: str
    ) -> None:
        """Notify hooks that a call was denied."""
        if self._ctx.hooks is None or not self._ctx.hooks.has("PermissionDenied"):
            return
        payload = self._hook_payload("PermissionDenied", name, args, tid)
        payload.message = message
        await self._ctx.hooks.run(payload)


__all__ = ["UbiquityToolset", "ToolDenied"]
