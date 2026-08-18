"""Tests that the tool descriptions describe the tools we actually shipped.

A tool description is read by the model as fact and is paid for on every
request, so a wrong one is worse than a missing one: it sends the model after
a tool that does not exist, or promises a limit the code does not enforce.
These are properties over the whole shipped set rather than assertions about
wording, so they keep holding as the text is rewritten.
"""

from __future__ import annotations

import re
from pathlib import Path

import ubiquity
from ubiquity.tools import builtin_tools
from ubiquity.tools.ask import AskUserQuestionTool
from ubiquity.tools.bash import BANNED_COMMANDS, MAX_OUTPUT_CHARS
from ubiquity.tools.glob import MAX_RESULTS
from ubiquity.tools.grep import MAX_MATCHES
from ubiquity.tools._files import MAX_LINES


def shipped_tools() -> list:
    """Every tool a run can expose, not only the unconditional ones.

    `AskUserQuestion` is added by `summon` only when a `can_use_tool`
    handler exists, so it is absent from `builtin_tools` and would escape
    a check written against that list alone.
    """
    return [*builtin_tools(), AskUserQuestionTool()]


KNOWN_TOOLS = {tool.name for tool in shipped_tools()} | {"Agent", "Skill"}

ABSENT = {"WebFetch", "WebSearch", "NotebookEdit", "ExitPlanMode"}


def descriptions() -> dict[str, str]:
    """Every shipped tool's description, keyed by tool name."""
    return {tool.name: tool.description for tool in shipped_tools()}


def test_nothing_the_model_reads_names_a_tool_we_do_not_have() -> None:
    """Bash once told the model to use WebFetch, which never existed here.

    The whole package is scanned rather than just the descriptions, because
    that mistake was in a validation message: every string the model can read
    is a place to send it after a tool that is not there.
    """
    package = Path(ubiquity.__file__).parent
    for source in package.rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        for absent in ABSENT:
            assert absent not in text, f"{source.name} names the absent tool {absent}"


def test_every_capitalized_tool_reference_resolves() -> None:
    """Catches a tool renamed in code but still named in a sibling's text."""
    referenced = re.compile(r"\b(?:use|Use|Prefer|prefer) (?:the )?([A-Z][a-zA-Z]+)")
    for name, text in descriptions().items():
        for candidate in referenced.findall(text):
            assert candidate in KNOWN_TOOLS, f"{name} points at unknown {candidate}"


def test_the_stated_limits_are_the_enforced_ones() -> None:
    """A limit in the text and a constant in the code must not drift apart."""
    text = descriptions()
    assert str(MAX_LINES) in text["Read"]
    assert str(MAX_RESULTS) in text["Glob"]
    assert str(MAX_MATCHES) in text["Grep"]
    assert str(MAX_OUTPUT_CHARS) in text["Bash"]


def test_bash_names_every_command_it_refuses() -> None:
    """The model should not have to discover the ban list by tripping over it."""
    for command in BANNED_COMMANDS:
        assert command in descriptions()["Bash"]


def test_the_ban_list_renders_in_a_stable_order() -> None:
    """It comes from a set, whose iteration order is not guaranteed to persist.

    Descriptions sit in the cached prefix, so a list that renders in a
    different order between runs costs a full cache miss for a difference the
    model cannot see.
    """
    text = descriptions()["Bash"]
    positions = [text.index(command) for command in sorted(BANNED_COMMANDS)]
    assert positions == sorted(positions)
