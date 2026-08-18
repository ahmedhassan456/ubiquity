"""MCP server configuration.

Three transports are supported — stdio, SSE, and streamable HTTP — and each is
translated into pydantic-ai's `MCPToolset`.

MCP tools arrive already namespaced as ``mcp__<server>__<tool>`` so they cannot
collide with built-in tool names, and so permission rules can target a whole
server with a prefix rule such as ``mcp__github__*``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPToolset

MCP_PREFIX = "mcp__"


@dataclass(slots=True)
class McpStdioServerConfig:
    """An MCP server launched as a subprocess and spoken to over stdio."""

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    type: str = "stdio"


@dataclass(slots=True)
class McpSSEServerConfig:
    """A remote MCP server reached over server-sent events."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    type: str = "sse"


@dataclass(slots=True)
class McpHttpServerConfig:
    """A remote MCP server reached over streamable HTTP."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    type: str = "http"


McpServerConfig: TypeAlias = (
    McpStdioServerConfig | McpSSEServerConfig | McpHttpServerConfig
)


def tool_prefix(server_name: str) -> str:
    """Return the prefix to hand to pydantic-ai's `prefixed()` for a server.

    `PrefixedToolset` renders names as ``f"{prefix}_{name}"``, so the prefix
    ends with a single underscore to produce the conventional
    ``mcp__<server>__<tool>`` convention rather than a single-underscore
    variant.
    """
    return f"{MCP_PREFIX}{server_name}_"


def qualified_tool_name(server_name: str, tool_name: str) -> str:
    """Return the fully namespaced name an MCP tool is exposed under."""
    return f"{MCP_PREFIX}{server_name}__{tool_name}"


def parse_config(raw: dict[str, Any]) -> McpServerConfig:
    """Build a server config from a plain dict.

    Accepts the shapes used by a standard `.mcp.json`. The type is inferred
    when absent: a `command` key means stdio, a `url` key means HTTP.
    """
    kind = raw.get("type")
    if kind == "stdio" or (kind is None and "command" in raw):
        return McpStdioServerConfig(
            command=raw["command"],
            args=list(raw.get("args", [])),
            env=dict(raw.get("env", {})),
            cwd=raw.get("cwd"),
        )
    if kind == "sse":
        return McpSSEServerConfig(url=raw["url"], headers=dict(raw.get("headers", {})))
    if kind in ("http", "streamable-http") or (kind is None and "url" in raw):
        return McpHttpServerConfig(url=raw["url"], headers=dict(raw.get("headers", {})))
    raise ValueError(f"Unrecognized MCP server config: {raw!r}")


def build_transport(config: McpServerConfig) -> Any:
    """Translate a server config into the pydantic-ai transport it describes.

    `MCPToolset` is constructed from a transport rather than from connection
    keywords, and the transport class is what distinguishes SSE from
    streamable HTTP — both carry a URL, so the config's `type` is the only
    thing that says which protocol to speak.
    """
    from pydantic_ai.mcp import SSETransport, StdioTransport, StreamableHttpTransport

    if isinstance(config, McpStdioServerConfig):
        return StdioTransport(
            command=config.command,
            args=list(config.args),
            env=config.env or None,
            cwd=config.cwd,
        )
    if isinstance(config, McpSSEServerConfig):
        return SSETransport(url=config.url, headers=config.headers or None)
    if isinstance(config, McpHttpServerConfig):
        return StreamableHttpTransport(url=config.url, headers=config.headers or None)
    raise TypeError(f"Unsupported MCP config type: {type(config).__name__}")


def build_toolset(name: str, config: McpServerConfig) -> MCPToolset:
    """Create a pydantic-ai `MCPToolset` for one configured server.

    Tools are prefixed with ``mcp__<server>__`` so they are addressable by
    permission rules and cannot shadow a built-in tool.
    """
    from pydantic_ai.mcp import MCPToolset

    return MCPToolset(build_transport(config)).prefixed(tool_prefix(name))


def build_toolsets(servers: dict[str, McpServerConfig]) -> list[MCPToolset]:
    """Build a toolset for every configured server, in server-name order.

    Sorted for the same reason the gate sorts a server's own tools: tool
    definitions are the front of the cached prompt prefix, so their order has
    to depend on the configuration rather than on how the caller's dictionary
    was assembled. Server names are prefixed onto every tool they own, so no
    two servers can collide and the order carries no other meaning.
    """
    return [build_toolset(name, servers[name]) for name in sorted(servers)]


__all__ = [
    "McpServerConfig",
    "McpStdioServerConfig",
    "McpSSEServerConfig",
    "McpHttpServerConfig",
    "parse_config",
    "build_transport",
    "build_toolset",
    "build_toolsets",
    "tool_prefix",
    "qualified_tool_name",
    "MCP_PREFIX",
]
