"""The built-in tool suite and the registry that assembles it."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..tool import Tool
from ..types import PermissionRuleValue
from .ask import AskUserQuestionInput, AskUserQuestionTool, Question, QuestionOption
from .bash import BashInput, BashTool
from .edit import EditInput, EditTool
from .glob import GlobInput, GlobTool
from .grep import GrepInput, GrepTool
from .read import ReadInput, ReadTool
from .skill import SkillInput, SkillTool
from .todo import TodoInput, TodoItem, TodoPatch, TodoWriteTool
from .write import WriteInput, WriteTool


def builtin_tools() -> list[Tool[Any]]:
    """Return a fresh instance of every built-in tool.

    Instances are created per call because tools may carry per-run state.
    """
    return [
        ReadTool(),
        WriteTool(),
        EditTool(),
        BashTool(),
        GlobTool(),
        GrepTool(),
        TodoWriteTool(),
    ]


def rule_names(rules: Sequence[str]) -> tuple[set[str], set[str]]:
    """Split rule strings into the tools named by bare and by scoped rules.

    Returns `(bare, scoped)`. ``Bash`` names Bash in `bare`; ``Bash(git:*)``
    names it in `scoped`. The distinction decides availability: a bare rule is
    about the tool itself, a scoped one is about particular calls and so leaves
    the tool in place for the permission engine to judge.
    """
    bare: set[str] = set()
    scoped: set[str] = set()
    for raw in rules:
        rule = PermissionRuleValue.parse(raw)
        (bare if rule.rule_content is None else scoped).add(rule.tool_name)
    return bare, scoped


def names_tool(tool: Tool[Any], names: set[str]) -> bool:
    """True when any name in `names` refers to `tool`."""
    return any(tool.matches_name(name) for name in names)


def resolve_tools(
    tools: Sequence[Tool[Any]] | None = None,
    *,
    allowed: Sequence[str] | None = None,
    disallowed: Sequence[str] = (),
) -> list[Tool[Any]]:
    """Filter a tool set by the allow and disallow lists.

    Starts from `tools`, or the built-in suite when None. Both lists hold
    permission rules, not plain names, so only the bare rules take a tool away
    or keep it: ``disallowed=["Bash(rm:*)"]`` must leave `Bash` available for
    the engine to block `rm` specifically, and ``allowed=["Bash(git:*)"]`` must
    keep `Bash` rather than looking for a tool by that literal name.

    `disallowed` is applied first so that a tool named by bare rules in both
    lists is excluded, which keeps the deny-wins rule consistent with the
    permission engine.
    """
    available = list(tools) if tools is not None else builtin_tools()
    removed, _ = rule_names(disallowed)
    available = [t for t in available if not names_tool(t, removed)]
    if allowed is not None:
        bare, scoped = rule_names(allowed)
        permitted = bare | scoped
        available = [t for t in available if names_tool(t, permitted)]
    return available


__all__ = [
    "builtin_tools",
    "resolve_tools",
    "rule_names",
    "names_tool",
    "ReadTool",
    "WriteTool",
    "EditTool",
    "BashTool",
    "GlobTool",
    "GrepTool",
    "TodoWriteTool",
    "SkillTool",
    "AskUserQuestionTool",
    "ReadInput",
    "WriteInput",
    "EditInput",
    "BashInput",
    "GlobInput",
    "GrepInput",
    "TodoInput",
    "TodoItem",
    "TodoPatch",
    "SkillInput",
    "AskUserQuestionInput",
    "Question",
    "QuestionOption",
]
