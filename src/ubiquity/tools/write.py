"""The Write tool.

Creates a file or fully replaces an existing one, refusing to overwrite a file the agent has not read.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ..tool import Tool, ToolContext, ValidationError
from ..types import (
    PermissionResult,
    PermissionResultAllow,
    PermissionResultAsk,
    ToolOutput,
)
from ._files import check_path_access, check_write_allowed, normalize, record_read

SENSITIVE_NAMES = {".env", ".npmrc", ".pypirc", "id_rsa", "id_ed25519", ".netrc"}
SENSITIVE_DIRS = {".git", ".ssh", ".aws", ".config"}


class WriteInput(BaseModel):
    file_path: str = Field(description="Absolute path to the file to write.")
    content: str = Field(description="The full contents to write to the file.")


def is_sensitive(path: Path) -> bool:
    """True when the path names a credential file or lives in a config dir.

    These paths prompt even in modes that would otherwise auto-accept edits.
    A mode is a statement about how much routine work the user wants to review,
    not consent to have their credentials rewritten unseen.
    """
    if path.name in SENSITIVE_NAMES:
        return True
    return any(part in SENSITIVE_DIRS for part in path.parts)


class WriteTool(Tool[WriteInput]):
    """Write a file to the filesystem."""

    name = "Write"
    description = (
        "Write a file to the local filesystem, overwriting it if it exists. "
        "You must Read an existing file before overwriting it. Missing parent "
        "directories are created, so there is no need to mkdir first. Prefer "
        "Edit for partial changes: it sends only the part that changes."
    )
    input_model = WriteInput
    search_hint = "create save new file overwrite"

    def is_destructive(self, args: WriteInput) -> bool:
        return Path(args.file_path).exists()

    def get_path(self, args: WriteInput) -> Path:
        return Path(args.file_path)

    def permission_rule_content(self, args: WriteInput) -> list[str]:
        return [args.file_path]

    def describe_call(self, args: WriteInput) -> str:
        return f"Write {args.file_path}"

    def _resolve(self, args: WriteInput, ctx: ToolContext) -> Path:
        """Resolve the target path against cwd when it is relative."""
        path = Path(args.file_path)
        return path if path.is_absolute() else (ctx.cwd / path).resolve()

    async def validate_input(
        self, args: WriteInput, ctx: ToolContext
    ) -> ValidationError | None:
        path = self._resolve(args, ctx)
        if (access_error := check_path_access(ctx, path)) is not None:
            return access_error
        if path.is_dir():
            return ValidationError(message=f"{path} is a directory.")
        return check_write_allowed(ctx, path)

    async def check_permissions(
        self, args: WriteInput, ctx: ToolContext
    ) -> PermissionResult:
        """Auto-accept edits in `acceptEdits` mode unless the path is sensitive."""
        path = self._resolve(args, ctx)
        if is_sensitive(path):
            return PermissionResultAsk(
                message=f"{path} is a sensitive file. Approve writing to it?",
                bypass_immune=True,
            )
        if ctx.permissions.mode == "acceptEdits" and ctx.is_path_allowed(path):
            return PermissionResultAllow(reason="acceptEdits mode")
        return await super().check_permissions(args, ctx)

    async def call(self, args: WriteInput, ctx: ToolContext) -> ToolOutput:
        path = self._resolve(args, ctx)
        existed = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)

        content = normalize(args.content)
        path.write_text(content, encoding="utf-8")
        record_read(ctx, path, content)

        line_count = len(content.split("\n"))
        verb = "Updated" if existed else "Created"
        return ToolOutput(
            content=f"{verb} {path} ({line_count} lines).",
            metadata={"path": str(path), "created": not existed, "lines": line_count},
        )


__all__ = ["WriteTool", "WriteInput", "is_sensitive"]
