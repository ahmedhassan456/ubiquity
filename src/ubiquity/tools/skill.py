"""The Skill tool, which loads one skill's instructions on demand.

This is the second of the three loading steps described in `skills`. This tool's
own description carries every skill's name and description, and a call turns one
of those names into the full procedure. It exists so the body is a cost the run
pays only when the model has decided the task matches.

The listing lives here and only here. Tool descriptions and the system prompt
both sit in the cached prefix, so a listing repeated in both would be paid for
twice on every request; this is the half the model is already reading when it
decides whether to call.

The tool is not part of the built-in suite. It is added to a run only when
skills were actually loaded, because a tool that lists nothing is a tool the
model can only misuse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ..tool import Tool, ToolContext, ValidationError
from ..types import PermissionResult, PermissionResultAllow, ToolOutput

if TYPE_CHECKING:
    from ..skills import Skill

MAX_DESCRIPTION_CHARS = 250

MAX_LISTING_CHARS = 8_000

ELLIPSIS = "…"


def format_listing(skills: dict[str, Skill]) -> str:
    """Render the skill listing, within a budget.

    The listing is for recognition only: the model needs enough to tell whether
    a skill covers the task in front of it, and the body it gets on invoking
    supplies everything else. A description long enough to explain the whole
    procedure therefore buys nothing and costs the run a cache-creation charge
    on turn one and a cache-read charge on every turn after.

    So each description is clamped, and if the whole listing still exceeds the
    budget it degrades to names alone rather than growing without limit. Names
    are never dropped, because a skill the model cannot name is a skill it
    cannot call; a run with enough skills to blow the budget on names alone has
    a problem this function cannot fix.

    Sorted by name rather than left in load order, since an order that shifts
    between runs costs a full cache miss for a difference the model cannot see.
    """
    names = sorted(skills)
    entries = [f"  - {name}: {_clamp(skills[name].description)}" for name in names]
    if sum(len(entry) + 1 for entry in entries) <= MAX_LISTING_CHARS:
        return "\n".join(entries)
    return "\n".join(f"  - {name}" for name in names)


def _clamp(description: str) -> str:
    """Shorten one description to the per-entry cap."""
    if len(description) <= MAX_DESCRIPTION_CHARS:
        return description
    return description[: MAX_DESCRIPTION_CHARS - 1].rstrip() + ELLIPSIS


class SkillInput(BaseModel):
    name: str = Field(description="The name of the skill to load.")


class SkillTool(Tool[SkillInput]):
    """Return the instructions for one named skill."""

    name = "Skill"
    description = (
        "Load the full instructions for a skill. Call this before starting a "
        "task a skill covers, and follow what it returns in place of your "
        "default approach. The skills available to this run are listed below "
        "with the situations they are for."
    )
    input_model = SkillInput
    search_hint = "skill instructions procedure playbook workflow"

    def __init__(self, skills: dict[str, Skill] | None = None) -> None:
        self._skills = skills or {}

    async def prompt(self, ctx: ToolContext) -> str:
        """List the loaded skills alongside the base description."""
        if not self._skills:
            return self.description
        return f"{self.description}\n\nAvailable skills:\n{format_listing(self._skills)}"

    def is_read_only(self, args: SkillInput) -> bool:
        return True

    def is_concurrency_safe(self, args: SkillInput) -> bool:
        return True

    def describe_call(self, args: SkillInput) -> str:
        return f"Skill({args.name})"

    def permission_rule_content(self, args: SkillInput) -> list[str]:
        return [args.name]

    async def validate_input(
        self, args: SkillInput, ctx: ToolContext
    ) -> ValidationError | None:
        if args.name not in self._skills:
            available = ", ".join(sorted(self._skills)) or "none"
            return ValidationError(
                message=(
                    f"Unknown skill {args.name!r}. Available skills: {available}."
                )
            )
        return None

    async def check_permissions(
        self, args: SkillInput, ctx: ToolContext
    ) -> PermissionResult:
        """Reading a loaded skill is allowed; what it then asks for is not.

        The caller chose the directories these were loaded from, so the
        instructions are already theirs. Every tool call the skill goes on to
        recommend still goes through the ordinary pipeline, which is where a
        skill that asks for too much gets stopped.
        """
        return PermissionResultAllow(reason="skill loaded from a configured directory")

    async def call(self, args: SkillInput, ctx: ToolContext) -> ToolOutput:
        skill = self._skills.get(args.name)
        if skill is None:
            available = ", ".join(sorted(self._skills)) or "none"
            return ToolOutput(
                content=f"No skill named {args.name!r}. Available: {available}.",
                is_error=True,
            )
        return ToolOutput(
            content=skill.render(),
            metadata={"skill": skill.name, "path": str(skill.path)},
        )


__all__ = [
    "SkillTool",
    "SkillInput",
    "format_listing",
    "MAX_DESCRIPTION_CHARS",
    "MAX_LISTING_CHARS",
]
