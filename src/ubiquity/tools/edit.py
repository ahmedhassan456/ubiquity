"""The Edit tool.

Performs exact string replacement, requiring the target string to be unique unless `replace_all` is set — an
ambiguous match is rejected rather than guessed at.
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
from ._files import (
    check_path_access,
    check_write_allowed,
    normalize,
    read_text,
    record_read,
)
from .write import is_sensitive


class EditInput(BaseModel):
    file_path: str = Field(description="Absolute path to the file to modify.")
    old_string: str = Field(description="The exact text to replace.")
    new_string: str = Field(description="The text to replace it with.")
    replace_all: bool = Field(
        default=False, description="Replace every occurrence instead of requiring one."
    )


class EditTool(Tool[EditInput]):
    """Perform an exact string replacement in a file."""

    name = "Edit"
    description = (
        "Perform exact string replacement in a file. You must Read the file "
        "first. `old_string` must match exactly, including indentation, and "
        "must be unique in the file unless `replace_all` is set. Read returns "
        "each line prefixed with its line number and a tab; that prefix is not "
        "part of the file, so strip it and keep the indentation that follows "
        "it. Prefer the shortest `old_string` that is unique, usually two to "
        "four adjacent lines."
    )
    input_model = EditInput
    search_hint = "modify change replace patch file"

    def get_path(self, args: EditInput) -> Path:
        return Path(args.file_path)

    def permission_rule_content(self, args: EditInput) -> list[str]:
        return [args.file_path]

    def describe_call(self, args: EditInput) -> str:
        return f"Edit {args.file_path}"

    def _resolve(self, args: EditInput, ctx: ToolContext) -> Path:
        """Resolve the target path against cwd when it is relative."""
        path = Path(args.file_path)
        return path if path.is_absolute() else (ctx.cwd / path).resolve()

    async def validate_input(
        self, args: EditInput, ctx: ToolContext
    ) -> ValidationError | None:
        path = self._resolve(args, ctx)

        if (access_error := check_path_access(ctx, path)) is not None:
            return access_error
        if not path.exists():
            return ValidationError(message=f"File does not exist: {path}")
        if args.old_string == args.new_string:
            return ValidationError(
                message="`old_string` and `new_string` are identical; nothing to do."
            )
        if (write_error := check_write_allowed(ctx, path)) is not None:
            return write_error

        text = normalize(read_text(path))
        old = normalize(args.old_string)

        if old == "":
            return ValidationError(
                message="`old_string` is empty. Use Write to create a new file."
            )

        occurrences = text.count(old)
        if occurrences == 0:
            return ValidationError(
                message=(
                    "`old_string` was not found in the file. It must match "
                    "exactly, including whitespace and indentation."
                )
            )
        if occurrences > 1 and not args.replace_all:
            return ValidationError(
                message=(
                    f"`old_string` appears {occurrences} times in the file. "
                    "Add more surrounding context to make it unique, or set "
                    "`replace_all` to replace every occurrence."
                )
            )
        return None

    async def check_permissions(
        self, args: EditInput, ctx: ToolContext
    ) -> PermissionResult:
        """Auto-accept edits in `acceptEdits` mode unless the path is sensitive."""
        path = self._resolve(args, ctx)
        if is_sensitive(path):
            return PermissionResultAsk(
                message=f"{path} is a sensitive file. Approve editing it?",
                bypass_immune=True,
            )
        if ctx.permissions.mode == "acceptEdits" and ctx.is_path_allowed(path):
            return PermissionResultAllow(reason="acceptEdits mode")
        return await super().check_permissions(args, ctx)

    async def call(self, args: EditInput, ctx: ToolContext) -> ToolOutput:
        path = self._resolve(args, ctx)
        text = normalize(read_text(path))
        old = normalize(args.old_string)
        new = normalize(args.new_string)

        count = text.count(old)
        updated = text.replace(old, new) if args.replace_all else text.replace(old, new, 1)
        replaced = count if args.replace_all else 1

        path.write_text(updated, encoding="utf-8")
        record_read(ctx, path, updated)

        plural = "occurrence" if replaced == 1 else "occurrences"
        return ToolOutput(
            content=f"Replaced {replaced} {plural} in {path}.",
            metadata={"path": str(path), "replacements": replaced},
        )


__all__ = ["EditTool", "EditInput"]
