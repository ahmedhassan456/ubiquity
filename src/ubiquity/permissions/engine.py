"""The permission decision pipeline.

Order of evaluation:
    1a  tool-level deny rule                    deny, bypass-immune
    1b  tool-level ask rule                     ask
    1c  tool.check_permissions
    1d  tool returned deny                      deny, bypass-immune
    1e  tool needs interaction and returned ask ask, bypass-immune
    1f  tool returned a bypass-immune ask       ask, bypass-immune
    2a  bypassPermissions mode                  allow
    2b  tool-level allow rule                   allow
    3   passthrough becomes ask

Mode transformations are applied around this core: `plan` blocks every
mutating tool up front, and `dontAsk` rewrites a trailing `ask` into a `deny`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..types import (
    PermissionResult,
    PermissionResultAllow,
    PermissionResultAsk,
    PermissionResultDeny,
    PermissionResultPassthrough,
    PermissionRuleValue,
)
from .rules import matches_all, matches_any

if TYPE_CHECKING:
    from ..tool import Tool, ToolContext


def _rule_strings_for_tool(
    ctx: ToolContext, tool: Tool[Any], behavior: str
) -> tuple[list[str], list[str]]:
    """Split a behavior's rules into bare and content-bearing rule strings.

    Returns `(bare, contents)` where `bare` holds rules naming only this tool
    and `contents` holds the matcher strings from parenthesized rules.
    """
    bare: list[str] = []
    contents: list[str] = []
    for raw in ctx.permissions.rules_for(behavior):
        rule = PermissionRuleValue.parse(raw)
        if not tool.matches_name(rule.tool_name):
            continue
        if rule.rule_content is None:
            bare.append(raw)
        else:
            contents.append(rule.rule_content)
    return bare, contents


def _match_content_rule(
    ctx: ToolContext, tool: Tool[Any], args: Any, behavior: str
) -> str | None:
    """Return a content rule of `behavior` matching any candidate, if any.

    Used for deny and ask, where a single offending candidate is decisive.
    """
    _, contents = _rule_strings_for_tool(ctx, tool, behavior)
    if not contents:
        return None
    candidates = tool.permission_rule_content(args)
    if not candidates:
        return None
    return matches_any(contents, candidates)


def _match_allow_content_rules(
    ctx: ToolContext, tool: Tool[Any], args: Any
) -> list[str] | None:
    """Return the allow rules covering every candidate, or None.

    Allow requires full coverage. A tool may present several candidates for a
    single call, and authorizing on a partial match would let a rule for one
    command authorize a chain containing others.
    """
    _, contents = _rule_strings_for_tool(ctx, tool, "allow")
    if not contents:
        return None
    return matches_all(contents, tool.permission_rule_content(args))


async def check_permissions(
    tool: Tool[Any],
    args: Any,
    ctx: ToolContext,
) -> PermissionResult:
    """Resolve whether `tool` may run with `args`, without prompting.

    Returns `allow`, `deny`, or `ask`. An `ask` means the host must consult
    `can_use_tool`; this function never blocks on user input.
    """
    mode = ctx.permissions.mode

    if mode == "plan" and not tool.is_read_only(args):
        return PermissionResultDeny(
            message=(
                f"{tool.name} is not permitted in plan mode. Plan mode is "
                "read-only; present the plan instead of acting on it."
            ),
        )

    deny_bare, _ = _rule_strings_for_tool(ctx, tool, "deny")
    if deny_bare:
        return PermissionResultDeny(
            message=f"{tool.name} is blocked by permission rule {deny_bare[0]}.",
        )

    denied_content = _match_content_rule(ctx, tool, args, "deny")
    if denied_content is not None:
        return PermissionResultDeny(
            message=(
                f"{tool.name} call is blocked by permission rule "
                f"{tool.name}({denied_content})."
            ),
        )

    ask_bare, _ = _rule_strings_for_tool(ctx, tool, "ask")
    if ask_bare:
        return PermissionResultAsk(
            message=f"{tool.name} requires approval (rule {ask_bare[0]}).",
            bypass_immune=True,
        )

    asked_content = _match_content_rule(ctx, tool, args, "ask")
    if asked_content is not None:
        return PermissionResultAsk(
            message=(
                f"{tool.name} call requires approval (rule "
                f"{tool.name}({asked_content}))."
            ),
            bypass_immune=True,
        )

    try:
        tool_result: PermissionResult = await tool.check_permissions(args, ctx)
    except Exception as exc:
        tool_result = PermissionResultDeny(
            message=f"{tool.name} permission check failed: {exc}",
        )

    if tool_result.behavior == "deny":
        return tool_result

    if tool_result.behavior == "ask" and (
        tool.requires_user_interaction() or tool_result.bypass_immune
    ):
        return tool_result

    if mode == "bypassPermissions":
        return PermissionResultAllow(
            updated_input=_updated_input(tool_result),
            reason="bypassPermissions mode",
        )

    allow_bare, _ = _rule_strings_for_tool(ctx, tool, "allow")
    if allow_bare:
        return PermissionResultAllow(
            updated_input=_updated_input(tool_result),
            reason=f"permission rule {allow_bare[0]}",
        )

    if tool_result.behavior == "allow":
        return tool_result

    allowed_content = _match_allow_content_rules(ctx, tool, args)
    if allowed_content is not None:
        covering = ", ".join(f"{tool.name}({c})" for c in dict.fromkeys(allowed_content))
        return PermissionResultAllow(
            updated_input=_updated_input(tool_result),
            reason=f"permission rules {covering}",
        )

    if tool_result.behavior == "ask":
        return tool_result

    return PermissionResultAsk(
        message=tool_result.message or f"{tool.name} requires approval.",
    )


def _updated_input(result: PermissionResult) -> dict[str, Any] | None:
    """Carry a tool's input rewrite through a later allow decision."""
    if result.behavior == "allow":
        return result.updated_input
    return None


def apply_mode_transformations(
    result: PermissionResult, ctx: ToolContext
) -> PermissionResult:
    """Rewrite a decision according to the permission mode.

    In `dontAsk` mode an `ask` becomes a `deny`, since there is nobody to
    prompt. Other modes pass the decision through unchanged.
    """
    if ctx.permissions.mode == "dontAsk" and result.behavior == "ask":
        return PermissionResultDeny(
            message=(
                f"{result.message} Permission mode is 'dontAsk', so this call "
                "was denied rather than prompted."
            ),
        )
    return result


__all__ = ["check_permissions", "apply_mode_transformations"]
