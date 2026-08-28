"""The terminal side of `can_use_tool`.

One handler covers both jobs the callback has. A permission prompt asks
whether a call may run and returns allow, deny, or allow plus a session rule
that stops the same question coming back. An "always" answer is written
straight into `ctx.permissions`, which is the run's live rule set, so the
permission engine resolves the next call without coming back here. An `AskUserQuestion` call is not an
approval at all: the questions are the tool's effect, so the handler renders
the form and returns allow with the answers written into `updated_input`,
which is where the tool reads them from.

Reads happen on a worker thread because `summon()` owns the event loop and
`input()` would otherwise block the run -- including the abort event that a
Ctrl-C sets.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..tool import ToolContext
from ..types import (
    PermissionResult,
    PermissionResultAllow,
    PermissionResultDeny,
    PermissionRuleValue,
    PermissionUpdate,
)
from .render import Renderer, paint, _summarize

_ALWAYS_SUFFIX = "and don't ask again for this tool"


async def _ask_line(renderer: Renderer, prompt: str) -> str:
    """Read one line from the terminal without blocking the event loop.

    The prompt goes out through the renderer so questions and answers share
    one stream with everything else the run prints.
    """
    renderer.write(prompt)
    try:
        return (await asyncio.to_thread(input)).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def _allow_rule(tool_name: str) -> PermissionUpdate:
    """A session-scoped rule that allows every later call of this tool."""
    return PermissionUpdate(
        type="addRules",
        destination="session",
        behavior="allow",
        rules=(PermissionRuleValue(tool_name=tool_name),),
    )


async def _answer_question(question: dict[str, Any], renderer: Renderer) -> str:
    """Render one question and return the user's reply as text.

    The reply is free text on purpose. The tool's contract says the user may
    answer with something that was never offered, so a number selects a label
    as a shorthand and anything else is passed through verbatim.
    """
    color = renderer.color
    options = question.get("options", [])
    multi = question.get("multi_select", False)
    renderer.write(paint(f"\n{question['question']}\n", "bold", enabled=color))
    for index, option in enumerate(options, start=1):
        label = paint(option["label"], "cyan", enabled=color)
        detail = paint(option["description"], "dim", enabled=color)
        renderer.write(f"  {index}. {label} — {detail}\n")
    hint = "numbers separated by commas, or free text" if multi else "number or free text"
    reply = await _ask_line(
        renderer, f"  {question.get('header', 'answer')} ({hint}): "
    )
    if not reply:
        return ""

    picks = [part.strip() for part in reply.split(",")] if multi else [reply]
    labels = []
    for pick in picks:
        if pick.isdigit() and 1 <= int(pick) <= len(options):
            labels.append(options[int(pick) - 1]["label"])
        else:
            return reply
    return ", ".join(labels)


async def _handle_ask_tool(
    tool_input: dict[str, Any], renderer: Renderer
) -> PermissionResult:
    """Collect answers for an `AskUserQuestion` call."""
    answers: dict[str, str] = {}
    for question in tool_input.get("questions", []):
        reply = await _answer_question(question, renderer)
        if reply:
            answers[question["question"]] = reply
    return PermissionResultAllow(
        updated_input={**tool_input, "answers": answers},
        decision_classification="user_temporary",
    )


def terminal_handler(renderer: Renderer) -> Any:
    """Build a `can_use_tool` callback bound to this terminal.

    The renderer is passed in rather than printing directly so a prompt closes
    any half-written assistant line before it draws.
    """

    async def can_use_tool(
        tool_name: str, tool_input: dict[str, Any], ctx: ToolContext
    ) -> PermissionResult:
        renderer.close_text()
        renderer.prompted = tool_name
        color = renderer.color
        if tool_name == "AskUserQuestion":
            return await _handle_ask_tool(tool_input, renderer)

        renderer.write(paint(f"\n● {tool_name}\n", "yellow", "bold", enabled=color))
        renderer.write(f"  {paint(_summarize(tool_input), 'dim', enabled=color)}\n")
        renderer.write(f"  [y] allow  [a] allow {_ALWAYS_SUFFIX}  [n] deny\n")
        reply = (await _ask_line(renderer, "  > ")).lower()

        if reply in ("a", "always"):
            ctx.permissions.allow_rules.add(tool_name)
            return PermissionResultAllow(
                updated_permissions=(_allow_rule(tool_name),),
                decision_classification="user_permanent",
            )
        if reply in ("", "y", "yes"):
            return PermissionResultAllow(decision_classification="user_temporary")
        return PermissionResultDeny(
            message="The user declined this call.",
            decision_classification="user_reject",
        )

    return can_use_tool
