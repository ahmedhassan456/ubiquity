"""MCP transport construction, and the gate every MCP call passes through.

pydantic-ai executes an MCP tool through the toolset that owns it, so without
a gate a server's tools reach the model unrestricted and invisible: no
permission check, no hook, and nothing in the message stream. These tests pin
both halves — that a server can be built at all, and that what it exposes is
governed like a built-in tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_ai.mcp import SSETransport, StdioTransport, StreamableHttpTransport

from ubiquity import Options, PermissionContext, ToolContext, parse_config
from ubiquity.mcp.config import build_toolset, build_transport
from ubiquity.mcp.gateway import GatedMCPToolset, McpArgs, McpTool
from ubiquity.toolset import UbiquityToolset


def test_stdio_config_builds_a_stdio_transport() -> None:
    transport = build_transport(
        parse_config({"command": "npx", "args": ["-y", "server"], "env": {"A": "1"}})
    )
    assert isinstance(transport, StdioTransport)


def test_sse_and_http_are_distinguished_by_type() -> None:
    """Both carry a URL, so only the declared type says which protocol to use."""
    sse = build_transport(parse_config({"type": "sse", "url": "https://x.test/sse"}))
    http = build_transport(parse_config({"url": "https://x.test/mcp"}))
    assert isinstance(sse, SSETransport)
    assert isinstance(http, StreamableHttpTransport)


def test_every_transport_builds_a_prefixed_toolset() -> None:
    for raw in (
        {"command": "python", "args": ["-c", "pass"]},
        {"type": "sse", "url": "https://x.test/sse"},
        {"url": "https://x.test/mcp"},
    ):
        toolset = build_toolset("demo", parse_config(raw))
        assert toolset is not None


def test_a_server_wildcard_names_every_tool_on_that_server() -> None:
    tool = McpTool("mcp__github__search", None)
    assert tool.matches_name("mcp__github__*")
    assert tool.matches_name("mcp__github__search")
    assert not tool.matches_name("mcp__gitlab__*")
    assert not tool.matches_name("Bash")


def test_an_unannotated_mcp_tool_is_treated_as_mutating() -> None:
    """A server that says nothing must not be assumed harmless in plan mode."""
    assert McpTool("mcp__x__y", None).is_read_only(McpArgs()) is False
    assert McpTool("mcp__x__y", None, read_only=True).is_read_only(McpArgs()) is True


def _ctx(tmp_path: Path, mode: str = "bypassPermissions", **rules: Any) -> ToolContext:
    """A tool context with explicit permission rules."""
    return ToolContext(
        cwd=tmp_path,
        options=Options(
            model="test", cwd=tmp_path, persist_session=False, persist_todos=False
        ),
        permissions=PermissionContext(
            mode=mode,
            deny_rules=set(rules.get("deny", ())),
            allow_rules=set(rules.get("allow", ())),
            ask_rules=set(rules.get("ask", ())),
            bypass_available=mode == "bypassPermissions",
        ),
        session_id="mcp-test",
    )


class _FakeMcpToolset:
    """Stands in for a server's toolset, recording what reached it."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def label(self) -> str:
        return "fake mcp"

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: Any, tool: Any
    ) -> Any:
        self.calls.append(tool_args)
        return "server answered"


class _FakeToolDef:
    metadata: dict[str, Any] = {}


class _FakeTool:
    tool_def = _FakeToolDef()


async def _call(
    gate_ctx: ToolContext, name: str, args: dict[str, Any]
) -> tuple[Any, _FakeMcpToolset, UbiquityToolset]:
    """Drive one MCP call through the gate the way pydantic-ai would."""
    from unittest.mock import MagicMock

    server = _FakeMcpToolset()
    emitted: list[Any] = []
    gate = UbiquityToolset([], gate_ctx, emit=emitted.append)
    gate.emitted = emitted  # type: ignore[attr-defined]

    toolset = GatedMCPToolset(wrapped=server, gate=gate)
    run_ctx = MagicMock()
    run_ctx.tool_call_id = "call-1"
    result = await toolset.call_tool(name, args, run_ctx, _FakeTool())
    return result, server, gate


async def test_an_allowed_mcp_call_reaches_the_server(tmp_path: Path) -> None:
    result, server, gate = await _call(
        _ctx(tmp_path), "mcp__demo__add", {"a": 1, "b": 2}
    )
    assert result == "server answered"
    assert server.calls == [{"a": 1, "b": 2}]


async def test_an_allowed_mcp_call_appears_in_the_message_stream(
    tmp_path: Path,
) -> None:
    _, _, gate = await _call(_ctx(tmp_path), "mcp__demo__add", {"a": 1})
    kinds = [m.type for m in gate.emitted]  # type: ignore[attr-defined]
    assert kinds == ["tool_use", "tool_result"]
    assert gate.emitted[0].tool_name == "mcp__demo__add"  # type: ignore[attr-defined]


async def test_a_deny_rule_stops_the_call_before_the_server_sees_it(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path, deny=["mcp__demo__add"])
    result, server, gate = await _call(ctx, "mcp__demo__add", {"a": 1})
    assert server.calls == []
    assert "denied" in str(result).lower()
    assert gate.denials and gate.denials[0].tool_name == "mcp__demo__add"


async def test_a_server_wildcard_deny_covers_its_tools(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, deny=["mcp__demo__*"])
    _, server, gate = await _call(ctx, "mcp__demo__add", {"a": 1})
    assert server.calls == []
    assert "mcp__demo__*" in gate.denials[0].message


async def test_a_wildcard_for_another_server_does_not_apply(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, deny=["mcp__other__*"])
    _, server, _ = await _call(ctx, "mcp__demo__add", {"a": 1})
    assert server.calls == [{"a": 1}]


async def test_plan_mode_blocks_an_unannotated_mcp_tool(tmp_path: Path) -> None:
    _, server, gate = await _call(_ctx(tmp_path, "plan"), "mcp__demo__write", {})
    assert server.calls == []
    assert "plan mode" in gate.denials[0].message


async def test_default_mode_asks_and_denies_without_a_handler(
    tmp_path: Path,
) -> None:
    _, server, gate = await _call(_ctx(tmp_path, "default"), "mcp__demo__add", {})
    assert server.calls == []
    assert "can_use_tool" in gate.denials[0].message


async def test_a_pre_tool_use_hook_can_rewrite_an_mcp_call(tmp_path: Path) -> None:
    from ubiquity import HookMatcher, HookOutput
    from ubiquity.hooks.registry import HookRegistry

    async def rewrite(payload: Any) -> HookOutput:
        return HookOutput(updated_input={"a": 99})

    ctx = _ctx(tmp_path)
    ctx.hooks = HookRegistry([HookMatcher("PreToolUse", [rewrite])])
    _, server, _ = await _call(ctx, "mcp__demo__add", {"a": 1})
    assert server.calls == [{"a": 99}]


async def test_a_pre_tool_use_hook_can_block_an_mcp_call(tmp_path: Path) -> None:
    from ubiquity import HookMatcher, HookOutput
    from ubiquity.hooks.registry import HookRegistry

    async def block(payload: Any) -> HookOutput:
        return HookOutput(decision="block", reason="no remote tools here")

    ctx = _ctx(tmp_path)
    ctx.hooks = HookRegistry([HookMatcher("PreToolUse", [block])])
    result, server, gate = await _call(ctx, "mcp__demo__add", {"a": 1})
    assert server.calls == []
    assert "no remote tools here" in str(result)


class TestCanonicalOrder:
    """Tool order is the front of the cached prefix, so it cannot be a server's.

    A server may return `list_tools` in any order and may change that order
    between calls. Because tool definitions precede the system prompt and every
    message in the cached prefix, a reordering the model cannot even perceive
    invalidates the entire conversation on every request. These tests pin the
    order to the tool names instead.
    """

    class _OrderedToolset:
        """A toolset that lists its tools in whatever order it was given."""

        label = "fake mcp"

        def __init__(self, names: list[str]) -> None:
            self.names = names

        async def get_tools(self, ctx: Any) -> dict[str, Any]:
            return {name: object() for name in self.names}

    async def _listed(self, names: list[str]) -> list[str]:
        from unittest.mock import MagicMock

        from ubiquity.mcp.gateway import gated

        toolset, = gated([self._OrderedToolset(names)], MagicMock())
        return list(await toolset.get_tools(MagicMock()))

    async def test_a_servers_tools_are_listed_in_name_order(self) -> None:
        assert await self._listed(["zeta", "alpha", "mid"]) == ["alpha", "mid", "zeta"]

    async def test_two_orders_of_the_same_tools_list_identically(self) -> None:
        """The property that matters: the prefix depends on what, not on order."""
        assert await self._listed(["zeta", "alpha", "mid"]) == await self._listed(
            ["alpha", "mid", "zeta"]
        )

    async def test_no_tool_is_lost_or_invented_by_the_sort(self) -> None:
        names = ["zeta", "alpha", "mid"]
        assert sorted(await self._listed(names)) == sorted(names)

    async def test_a_server_that_lists_nothing_stays_empty(self) -> None:
        assert await self._listed([]) == []

    async def test_the_order_reaching_the_model_is_sorted(self) -> None:
        """End to end. Sorting the dict is only worth anything if it survives."""
        from unittest.mock import MagicMock

        from pydantic_ai import Agent
        from pydantic_ai.messages import ModelResponse, TextPart
        from pydantic_ai.models.function import AgentInfo, FunctionModel
        from pydantic_ai.toolsets import FunctionToolset

        from ubiquity.mcp.gateway import gated

        seen: dict[str, list[str]] = {}

        def respond(messages: Any, info: AgentInfo) -> ModelResponse:
            seen["order"] = [t.name for t in info.function_tools]
            return ModelResponse(parts=[TextPart(content="ok")])

        inner: FunctionToolset[Any] = FunctionToolset()
        for name in ("zeta", "alpha", "mid"):
            inner.add_function(lambda: "x", name=name)
        toolset, = gated([inner], MagicMock())
        await Agent(FunctionModel(respond), toolsets=[toolset]).run("go")
        assert seen["order"] == ["alpha", "mid", "zeta"]


class TestServerOrder:
    def test_servers_are_built_in_name_order(self) -> None:
        """Same exposure one level up: a caller's dict order is not a promise."""
        from ubiquity.mcp.config import build_toolsets

        servers = {
            name: parse_config({"url": f"https://{name}.test/mcp"})
            for name in ("zeta", "alpha", "mid")
        }
        built = build_toolsets(servers)
        assert [t.prefix for t in built] == [
            "mcp__alpha_",
            "mcp__mid_",
            "mcp__zeta_",
        ]

    def test_the_same_servers_in_a_different_order_build_identically(self) -> None:
        from ubiquity.mcp.config import build_toolsets

        def prefixes(names: tuple[str, ...]) -> list[str]:
            servers = {
                name: parse_config({"url": f"https://{name}.test/mcp"})
                for name in names
            }
            return [t.prefix for t in build_toolsets(servers)]

        assert prefixes(("zeta", "alpha")) == prefixes(("alpha", "zeta"))

    def test_every_configured_server_is_still_built(self) -> None:
        from ubiquity.mcp.config import build_toolsets

        servers = {
            name: parse_config({"url": f"https://{name}.test/mcp"})
            for name in ("zeta", "alpha", "mid")
        }
        assert len(build_toolsets(servers)) == 3
