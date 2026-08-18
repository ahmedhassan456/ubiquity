"""Tests pinning the permission decision ordering.

The ordering is load-bearing for safety: deny rules and content-specific ask
rules must survive `bypassPermissions`, and plan mode must block every
mutating tool. These tests encode the step order documented in
`ubiquity.permissions.engine`.
"""

from __future__ import annotations

import pytest

from ubiquity.permissions.engine import apply_mode_transformations, check_permissions
from ubiquity.types import (
    PermissionResultAllow,
    PermissionResultAsk,
    PermissionResultDeny,
)

from doubles import EchoInput, EchoTool


async def test_plan_mode_blocks_mutating_tools(make_ctx) -> None:
    result = await check_permissions(
        EchoTool(read_only=False), EchoInput(command="rm -rf /"), make_ctx(mode="plan")
    )
    assert result.behavior == "deny"
    assert "plan mode" in result.message


async def test_plan_mode_allows_read_only_tools(make_ctx) -> None:
    result = await check_permissions(
        EchoTool(read_only=True, permission=PermissionResultAllow()),
        EchoInput(command="ls"),
        make_ctx(mode="plan"),
    )
    assert result.behavior == "allow"


async def test_bare_deny_rule_blocks_tool(make_ctx) -> None:
    result = await check_permissions(
        EchoTool(), EchoInput(command="ls"), make_ctx(deny={"Echo"})
    )
    assert result.behavior == "deny"


async def test_content_deny_rule_blocks_matching_call(make_ctx) -> None:
    ctx = make_ctx(deny={"Echo(rm:*)"})
    denied = await check_permissions(EchoTool(), EchoInput(command="rm -rf ."), ctx)
    assert denied.behavior == "deny"

    allowed = await check_permissions(EchoTool(), EchoInput(command="ls -la"), ctx)
    assert allowed.behavior != "deny"


async def test_deny_rule_survives_bypass_permissions(make_ctx) -> None:
    """A deny rule must win over `bypassPermissions`."""
    result = await check_permissions(
        EchoTool(),
        EchoInput(command="ls"),
        make_ctx(mode="bypassPermissions", deny={"Echo"}),
    )
    assert result.behavior == "deny"


async def test_content_ask_rule_survives_bypass_permissions(make_ctx) -> None:
    """A user-configured content ask rule must prompt even in bypass mode."""
    result = await check_permissions(
        EchoTool(),
        EchoInput(command="npm publish"),
        make_ctx(mode="bypassPermissions", ask={"Echo(npm publish:*)"}),
    )
    assert result.behavior == "ask"


async def test_tool_deny_survives_bypass_permissions(make_ctx) -> None:
    result = await check_permissions(
        EchoTool(permission=PermissionResultDeny(message="tool said no")),
        EchoInput(command="ls"),
        make_ctx(mode="bypassPermissions"),
    )
    assert result.behavior == "deny"
    assert result.message == "tool said no"


async def test_bypass_permissions_allows_otherwise_unapproved_call(make_ctx) -> None:
    result = await check_permissions(
        EchoTool(), EchoInput(command="ls"), make_ctx(mode="bypassPermissions")
    )
    assert result.behavior == "allow"


async def test_tool_requiring_interaction_asks_even_in_bypass(make_ctx) -> None:
    result = await check_permissions(
        EchoTool(
            permission=PermissionResultAsk(message="confirm?"), needs_interaction=True
        ),
        EchoInput(command="ls"),
        make_ctx(mode="bypassPermissions"),
    )
    assert result.behavior == "ask"


async def test_bare_allow_rule_permits_call(make_ctx) -> None:
    result = await check_permissions(
        EchoTool(), EchoInput(command="ls"), make_ctx(allow={"Echo"})
    )
    assert result.behavior == "allow"


async def test_content_allow_rule_permits_matching_call_only(make_ctx) -> None:
    ctx = make_ctx(allow={"Echo(git:*)"})
    allowed = await check_permissions(EchoTool(), EchoInput(command="git status"), ctx)
    assert allowed.behavior == "allow"

    not_allowed = await check_permissions(EchoTool(), EchoInput(command="rm -rf ."), ctx)
    assert not_allowed.behavior == "ask"


async def test_passthrough_becomes_ask_by_default(make_ctx) -> None:
    """With no rules and no tool opinion, default mode prompts."""
    result = await check_permissions(EchoTool(), EchoInput(command="ls"), make_ctx())
    assert result.behavior == "ask"


async def test_tool_allow_is_honored_without_rules(make_ctx) -> None:
    result = await check_permissions(
        EchoTool(permission=PermissionResultAllow()),
        EchoInput(command="ls"),
        make_ctx(),
    )
    assert result.behavior == "allow"


async def test_allow_rule_carries_tool_input_rewrite(make_ctx) -> None:
    """An input rewrite from the tool survives a later rule-based allow."""
    result = await check_permissions(
        EchoTool(permission=PermissionResultAllow(updated_input={"command": "safe"})),
        EchoInput(command="unsafe"),
        make_ctx(allow={"Echo"}),
    )
    assert result.behavior == "allow"
    assert result.updated_input == {"command": "safe"}


async def test_deny_precedes_ask_when_both_rules_match(make_ctx) -> None:
    result = await check_permissions(
        EchoTool(), EchoInput(command="ls"), make_ctx(deny={"Echo"}, ask={"Echo"})
    )
    assert result.behavior == "deny"


async def test_raising_tool_check_is_denied_not_crashed(make_ctx) -> None:
    """A tool whose permission check raises must fail closed."""

    class Exploding(EchoTool):
        async def check_permissions(self, args, ctx):
            raise RuntimeError("boom")

    result = await check_permissions(Exploding(), EchoInput(command="ls"), make_ctx())
    assert result.behavior == "deny"
    assert "boom" in result.message


async def test_dont_ask_mode_converts_ask_to_deny(make_ctx) -> None:
    ctx = make_ctx(mode="dontAsk")
    asked = await check_permissions(EchoTool(), EchoInput(command="ls"), ctx)
    assert asked.behavior == "ask"

    transformed = apply_mode_transformations(asked, ctx)
    assert transformed.behavior == "deny"
    assert "dontAsk" in transformed.message


async def test_dont_ask_mode_leaves_allow_untouched(make_ctx) -> None:
    ctx = make_ctx(mode="dontAsk", allow={"Echo"})
    result = apply_mode_transformations(
        await check_permissions(EchoTool(), EchoInput(command="ls"), ctx), ctx
    )
    assert result.behavior == "allow"


@pytest.mark.parametrize("mode", ["default", "acceptEdits", "dontAsk", "plan"])
async def test_deny_rule_wins_in_every_mode(make_ctx, mode: str) -> None:
    result = await check_permissions(
        EchoTool(read_only=True), EchoInput(command="ls"), make_ctx(mode=mode, deny={"Echo"})
    )
    assert result.behavior == "deny"
