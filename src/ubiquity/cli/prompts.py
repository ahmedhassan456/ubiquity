"""The terminal side of `can_use_tool`.

One handler covers both jobs the callback has. A permission prompt asks
whether a call may run and returns allow, deny, or allow plus a rule that stops
the same question coming back. An `AskUserQuestion` call is not an approval at
all: the questions are the tool's effect, so the handler renders the form and
returns allow with the answers written into `updated_input`, which is where the
tool reads them from.

An "always" answer is written straight into `ctx.permissions`, the run's live
rule set, so the permission engine resolves the next call without coming back
here. It is also returned as `updated_permissions` for hosts that persist it.

Reads happen on a worker thread because `summon()` owns the event loop, and a
blocking `input()` would freeze the run -- including the abort event a Ctrl-C
sets.
"""

from __future__ import annotations

from typing import Any

from rich.syntax import Syntax
from rich.text import Text

from ..tool import ToolContext
from ..types import (
    PermissionResult,
    PermissionResultAllow,
    PermissionResultDeny,
    PermissionRuleValue,
    PermissionUpdate,
)
from . import ui
from .render import Renderer, _summarize

_DIFFABLE = {"Write": "content", "Edit": "new_string", "Bash": "command"}
"""Tools whose input is worth showing in full, and the field that carries it."""

_LANGUAGE = {"Bash": "bash", "Write": "text", "Edit": "text"}


def _allow_rule(tool_name: str) -> PermissionUpdate:
    """A session-scoped rule that allows every later call of this tool."""
    return PermissionUpdate(
        type="addRules",
        destination="session",
        behavior="allow",
        rules=(PermissionRuleValue(tool_name=tool_name),),
    )


def _preview(tool_name: str, tool_input: dict[str, Any]) -> Any:
    """Build the body of the approval panel.

    A path alone is not enough to approve a `Write` on, so the tools that carry
    a payload show it, capped at a screenful. Everything else shows the one-line
    summary the transcript would.
    """
    field = _DIFFABLE.get(tool_name)
    if field and isinstance(tool_input.get(field), str):
        body = tool_input[field]
        lines = body.splitlines()
        if len(lines) > 20:
            body = "\n".join([*lines[:20], f"… +{len(lines) - 20} more lines"])
        return Syntax(
            body,
            _LANGUAGE.get(tool_name, "text"),
            theme="ansi_dark",
            background_color="default",
            word_wrap=True,
        )
    return Text(_summarize(tool_input, tool_name), style="tool.args")


async def _answer_question(question: dict[str, Any]) -> str:
    """Render one question and return the user's reply as text.

    The reply is free text on purpose. The tool's contract says the user may
    answer with something that was never offered, so the menu's "type something
    else" row is passed through verbatim rather than matched to a label.
    """
    options = [
        (option["label"], option.get("description", ""))
        for option in question.get("options", [])
    ]
    picked = await ui.select(
        question["question"],
        options,
        multi=bool(question.get("multi_select")),
        allow_other=True,
    )
    return ", ".join(picked)


async def _handle_ask_tool(tool_input: dict[str, Any]) -> PermissionResult:
    """Collect answers for an `AskUserQuestion` call."""
    answers: dict[str, str] = {}
    for question in tool_input.get("questions", []):
        reply = await _answer_question(question)
        if reply:
            answers[question["question"]] = reply
    return PermissionResultAllow(
        updated_input={**tool_input, "answers": answers},
        decision_classification="user_temporary",
    )


def terminal_handler(renderer: Renderer) -> Any:
    """Build a `can_use_tool` callback bound to this terminal.

    The renderer is passed in so a prompt can take down the spinner and close
    any half-written line before it draws.
    """

    async def can_use_tool(
        tool_name: str, tool_input: dict[str, Any], ctx: ToolContext
    ) -> PermissionResult:
        renderer.close_text()
        renderer.prompted = tool_name

        if tool_name == "AskUserQuestion":
            result = await _handle_ask_tool(tool_input)
            renderer.start_status()
            return result

        ui.panel(
            _preview(tool_name, tool_input),
            title=f"{tool_name} — allow?",
            style="warn",
        )
        reply = await ui.hotkey(
            "  [y] allow   [a] always   [n] deny", accepted="yan", default="y"
        )

        if reply in ("a", "always"):
            ctx.permissions.allow_rules.add(tool_name)
            renderer.note(f"  ✓ allowing {tool_name} for the rest of this session", "ok")
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
