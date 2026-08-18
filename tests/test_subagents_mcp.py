"""Tests for the Agent tool, subagent isolation, and MCP configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ubiquity import AgentDefinition, Options, summon
from ubiquity.mcp import (
    McpHttpServerConfig,
    McpSSEServerConfig,
    McpStdioServerConfig,
    parse_config,
    qualified_tool_name,
    tool_prefix,
)
from ubiquity.subagents.agent_tool import MAX_DEPTH, AgentInput, AgentTool
from ubiquity.types import PermissionResultAllow


def scripted(*turns: list[Any]) -> FunctionModel:
    """Build a model that replays `turns`, one response per model request."""
    calls = {"n": 0}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        index = min(calls["n"], len(turns) - 1)
        calls["n"] += 1
        return ModelResponse(parts=list(turns[index]))

    return FunctionModel(respond)


async def allow_all(name: str, args: dict, ctx) -> PermissionResultAllow:
    return PermissionResultAllow()


def test_parse_stdio_config_from_dict() -> None:
    config = parse_config({"command": "npx", "args": ["-y", "server"]})
    assert isinstance(config, McpStdioServerConfig)
    assert config.command == "npx"
    assert config.args == ["-y", "server"]


def test_parse_infers_stdio_without_explicit_type() -> None:
    assert isinstance(parse_config({"command": "run-me"}), McpStdioServerConfig)


def test_parse_infers_http_from_url() -> None:
    assert isinstance(parse_config({"url": "https://x.test/mcp"}), McpHttpServerConfig)


def test_parse_sse_config() -> None:
    config = parse_config({"type": "sse", "url": "https://x.test/sse"})
    assert isinstance(config, McpSSEServerConfig)


def test_parse_rejects_unrecognized_config() -> None:
    with pytest.raises(ValueError):
        parse_config({"nonsense": True})


def test_tool_prefix_yields_double_underscore_convention() -> None:
    """pydantic-ai joins with one underscore, so the prefix supplies the other."""
    prefix = tool_prefix("github")
    assert f"{prefix}_create_issue" == qualified_tool_name("github", "create_issue")
    assert qualified_tool_name("github", "x") == "mcp__github__x"


async def test_agent_tool_lists_available_types(make_ctx) -> None:
    tool = AgentTool(
        {"reviewer": AgentDefinition(description="Reviews code", prompt="review")}
    )
    text = await tool.prompt(make_ctx())
    assert "reviewer" in text
    assert "Reviews code" in text


async def test_agent_tool_rejects_unknown_type(make_ctx) -> None:
    tool = AgentTool({"reviewer": AgentDefinition(description="d", prompt="p")})
    error = await tool.validate_input(
        AgentInput(description="x", prompt="y", subagent_type="ghost"), make_ctx()
    )
    assert error is not None
    assert "Unknown subagent_type" in error.message


async def test_agent_tool_enforces_nesting_limit(make_ctx) -> None:
    """Unbounded nesting would let one prompt fan out without limit."""
    tool = AgentTool({"general-purpose": AgentDefinition(description="d", prompt="p")})
    ctx = make_ctx()
    ctx.extra["agent_depth"] = MAX_DEPTH

    error = await tool.validate_input(
        AgentInput(description="x", prompt="y", subagent_type="general-purpose"), ctx
    )
    assert error is not None
    assert "nesting limit" in error.message


async def test_agent_tool_allows_spawn_without_prompting(make_ctx) -> None:
    """The spawn itself is free; the subagent's own tool calls are still checked."""
    tool = AgentTool()
    result = await tool.check_permissions(
        AgentInput(description="x", prompt="y"), make_ctx()
    )
    assert result.behavior == "allow"


async def test_agent_tool_is_offered_when_agents_configured(tmp_path: Path) -> None:
    messages = [
        m
        async for m in summon(
            "hello",
            Options(
                model=scripted([TextPart(content="ok")]),
                cwd=tmp_path,
                agents={"reviewer": AgentDefinition(description="d", prompt="p")},
            ),
        )
    ]
    assert "Agent" in messages[0].tools
    assert messages[0].agents == ["reviewer"]


async def test_agent_tool_absent_without_agents(tmp_path: Path) -> None:
    messages = [
        m
        async for m in summon(
            "hello",
            Options(model=scripted([TextPart(content="ok")]), cwd=tmp_path),
        )
    ]
    assert "Agent" not in messages[0].tools


async def test_subagent_runs_and_returns_only_final_text(tmp_path: Path) -> None:
    """The parent sees the subagent's report, not its intermediate work."""
    (tmp_path / "target.txt").write_text("subagent should read this")

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        names = {t.name for t in info.function_tools}
        if "Agent" in names:
            if not any(
                isinstance(p, ToolCallPart)
                for m in messages
                for p in getattr(m, "parts", [])
            ):
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="Agent",
                            args={
                                "description": "read file",
                                "prompt": "read target.txt",
                                "subagent_type": "reader",
                            },
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart(content="parent done")])
        return ModelResponse(parts=[TextPart(content="subagent report")])

    messages = [
        m
        async for m in summon(
            "delegate this",
            Options(
                model=FunctionModel(respond),
                cwd=tmp_path,
                can_use_tool=allow_all,
                agents={"reader": AgentDefinition(description="Reads", prompt="Read it")},
            ),
        )
    ]

    results = [m for m in messages if m.type == "tool_result" and m.tool_name == "Agent"]
    assert len(results) == 1
    assert results[0].output.content == "subagent report"
    assert results[0].output.metadata["subagent_type"] == "reader"
    assert messages[-1].subtype == "success"


async def test_subagent_cannot_spawn_further_agents(tmp_path: Path) -> None:
    """The Agent tool is stripped from a subagent's own tool set."""
    from ubiquity.client import run_subagent
    from ubiquity.hooks.registry import HookRegistry
    from ubiquity.client import _build_context

    seen: dict[str, set[str]] = {}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen["tools"] = {t.name for t in info.function_tools}
        return ModelResponse(parts=[TextPart(content="done")])

    options = Options(model=FunctionModel(respond), cwd=tmp_path)
    ctx = _build_context(options, "s", HookRegistry())

    report, _ = await run_subagent(
        "do a thing", ctx, AgentDefinition(description="d", prompt="p")
    )
    assert report == "done"
    assert "Agent" not in seen["tools"]


async def test_subagent_tool_restriction_is_honored(tmp_path: Path) -> None:
    from ubiquity.client import _build_context, run_subagent
    from ubiquity.hooks.registry import HookRegistry

    seen: dict[str, set[str]] = {}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen["tools"] = {t.name for t in info.function_tools}
        return ModelResponse(parts=[TextPart(content="done")])

    options = Options(model=FunctionModel(respond), cwd=tmp_path)
    ctx = _build_context(options, "s", HookRegistry())

    await run_subagent(
        "x", ctx, AgentDefinition(description="d", prompt="p", tools=["Read", "Glob"])
    )
    assert seen["tools"] == {"Read", "Glob"}


async def test_subagent_failure_is_reported_not_raised(tmp_path: Path) -> None:
    """A broken subagent returns an error string rather than killing the parent."""
    from ubiquity.client import _build_context, run_subagent
    from ubiquity.hooks.registry import HookRegistry

    def explode(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("subagent exploded")

    options = Options(model=FunctionModel(explode), cwd=tmp_path)
    ctx = _build_context(options, "s", HookRegistry())

    report, _ = await run_subagent("x", ctx, None)
    assert "Subagent failed" in report
    assert "subagent exploded" in report
