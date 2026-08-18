"""Test doubles shared across the suite."""

from __future__ import annotations

from pydantic import BaseModel

from ubiquity.tool import Tool, ToolContext
from ubiquity.types import PermissionResult, PermissionResultPassthrough, ToolOutput


class EchoInput(BaseModel):
    command: str = ""
    path: str = ""


class EchoTool(Tool[EchoInput]):
    """A configurable stand-in used to exercise the permission pipeline."""

    name = "Echo"
    description = "Echoes its input."
    input_model = EchoInput

    def __init__(
        self,
        *,
        read_only: bool = False,
        permission: PermissionResult | None = None,
        needs_interaction: bool = False,
        rule_content: list[str] | None = None,
    ) -> None:
        self._read_only = read_only
        self._permission = permission or PermissionResultPassthrough()
        self._needs_interaction = needs_interaction
        self._rule_content = rule_content

    async def call(self, args: EchoInput, ctx: ToolContext) -> ToolOutput:
        return ToolOutput(content=args.command or args.path)

    def is_read_only(self, args: EchoInput) -> bool:
        return self._read_only

    async def check_permissions(
        self, args: EchoInput, ctx: ToolContext
    ) -> PermissionResult:
        return self._permission

    def requires_user_interaction(self) -> bool:
        return self._needs_interaction

    def permission_rule_content(self, args: EchoInput) -> list[str]:
        if self._rule_content is not None:
            return self._rule_content
        return [args.command] if args.command else []
