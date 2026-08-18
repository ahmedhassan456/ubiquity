"""Holding MCP tools to the same rules as built-in ones.

pydantic-ai executes an MCP tool through the toolset that owns it, so a server's
tools would otherwise reach the model without passing the permission engine,
without firing a hook, and without appearing in the message stream — invisible
and unrestricted, which is the opposite of what a remote tool deserves.

`GatedMCPToolset` wraps a server's toolset and routes every call back through
`UbiquityToolset.call_with`, the same path a built-in tool takes. Schemas and
execution still come from the server; only the decision to run does not.

Tools are addressable as ``mcp__<server>__<tool>``, and a whole server can be
named with a trailing wildcard such as ``mcp__github__*``.

The gate is also where a server's tool list is put in a canonical order. Tool
definitions sit at the very front of the cached prompt prefix, ahead of the
system prompt and every message, so two runs whose tools differ only in order
share no cached prefix at all. A server is free to return `list_tools` in any
order it likes and to change its mind between calls, which would invalidate the
whole conversation on every request for a difference the model cannot perceive.
Sorting by name here costs nothing and makes the prefix depend on what the tools
are rather than on what order a remote process happened to list them in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict
from pydantic_ai import RunContext, ToolsetTool
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset

from ..tool import Tool, ToolContext
from ..types import ToolOutput

if TYPE_CHECKING:
    from ..toolset import UbiquityToolset

WILDCARD = "*"


class McpArgs(BaseModel):
    """The arguments of an MCP call.

    Deliberately permissive: the schema that matters is the server's own, which
    pydantic-ai has already validated the call against by the time it arrives.
    """

    model_config = ConfigDict(extra="allow")


class McpTool(Tool[McpArgs]):
    """One remote MCP tool, presented to the permission engine as a tool.

    `is_read_only` follows the server's `readOnlyHint` annotation when it sends
    one and is False otherwise, so a server that says nothing about a tool is
    treated as able to mutate — the safe reading, and the one that keeps plan
    mode meaningful against an unknown server.
    """

    input_model = McpArgs
    description = "An MCP tool provided by a configured server."

    def __init__(self, name: str, invoke: Any, *, read_only: bool = False) -> None:
        self.name = name
        self._invoke = invoke
        self._read_only = read_only

    async def call(self, args: McpArgs, ctx: ToolContext) -> ToolOutput:
        """Execute the call against the server."""
        result = await self._invoke(args.model_dump())
        if isinstance(result, ToolOutput):
            return result
        return ToolOutput(content=result if isinstance(result, str) else str(result))

    def is_read_only(self, args: McpArgs) -> bool:
        return self._read_only

    def matches_name(self, name: str) -> bool:
        """True for this tool's own name, or a wildcard covering its server."""
        if name == self.name:
            return True
        return name.endswith(WILDCARD) and self.name.startswith(name[: -len(WILDCARD)])


def _read_only_hint(tool: ToolsetTool[Any]) -> bool:
    """Whether the server annotated this tool as read-only."""
    metadata = getattr(tool.tool_def, "metadata", None) or {}
    annotations = metadata.get("annotations") or metadata
    return bool(annotations.get("readOnlyHint", False))


@dataclass
class GatedMCPToolset(WrapperToolset[Any]):
    """An MCP toolset whose calls go through the ubiquity pipeline."""

    wrapped: AbstractToolset[Any]
    gate: UbiquityToolset = field(default=None)  # type: ignore[assignment]

    @property
    def label(self) -> str:
        return f"gated {self.wrapped.label}"

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        """Describe the server's tools to the model, in name order.

        The sort is the point: `list_tools` order is the server's business and
        may differ between two calls to the same server, and an order that
        changes invalidates the cached prefix that every tool definition sits
        at the front of. Sorting by name makes the prefix a function of the
        tools themselves.
        """
        tools = await self.wrapped.get_tools(ctx)
        return {name: tools[name] for name in sorted(tools)}

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> Any:
        """Authorize the call, then let the server perform it."""

        async def invoke(args: dict[str, Any]) -> Any:
            return await self.wrapped.call_tool(name, args, ctx, tool)

        impl = McpTool(name, invoke, read_only=_read_only_hint(tool))
        return await self.gate.call_with(impl, tool_args, ctx.tool_call_id)


def gated(
    toolsets: list[AbstractToolset[Any]], gate: UbiquityToolset
) -> list[AbstractToolset[Any]]:
    """Wrap every MCP toolset so its calls pass through `gate`."""
    return [GatedMCPToolset(wrapped=toolset, gate=gate) for toolset in toolsets]


__all__ = ["GatedMCPToolset", "McpTool", "McpArgs", "gated"]
