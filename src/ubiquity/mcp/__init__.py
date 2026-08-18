"""MCP server configuration and toolset construction."""

from .gateway import GatedMCPToolset, McpTool, gated
from .config import (
    MCP_PREFIX,
    McpHttpServerConfig,
    McpServerConfig,
    McpSSEServerConfig,
    McpStdioServerConfig,
    build_toolset,
    build_toolsets,
    build_transport,
    parse_config,
    qualified_tool_name,
    tool_prefix,
)

__all__ = [
    "GatedMCPToolset",
    "McpTool",
    "gated",
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
