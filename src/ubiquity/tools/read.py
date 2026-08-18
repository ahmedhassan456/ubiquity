"""The Read tool.

Reads a file and returns it in ``cat -n`` format, recording a `FileState` so that later writes can enforce
read-before-write.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ..tool import Tool, ToolContext, ValidationError
from ..types import PermissionResultAllow, PermissionResult, ToolOutput
from ._files import (
    MAX_FILE_BYTES,
    MAX_LINES,
    check_path_access,
    format_with_line_numbers,
    read_text,
    record_read,
    slice_lines,
)


class ReadInput(BaseModel):
    file_path: str = Field(description="Absolute path to the file to read.")
    offset: int | None = Field(
        default=None, description="1-indexed line to start reading from."
    )
    limit: int | None = Field(default=None, description="Number of lines to read.")


class ReadTool(Tool[ReadInput]):
    """Read a file from the filesystem."""

    name = "Read"
    description = (
        "Read a file from the local filesystem. Returns the contents in "
        "cat -n format with line numbers starting at 1. Use `offset` and "
        f"`limit` to read a slice of a large file; without them the first "
        f"{MAX_LINES} lines are returned. You must read a file before you may "
        "write to or edit it, and a slice does not count: to edit a file "
        f"longer than {MAX_LINES} lines, read it with a `limit` past its last "
        "line. Directories cannot be read; use Glob or Bash to list one."
    )
    input_model = ReadInput
    search_hint = "open view inspect file contents"
    max_result_chars = 100_000

    def is_read_only(self, args: ReadInput) -> bool:
        return True

    def is_concurrency_safe(self, args: ReadInput) -> bool:
        return True

    def get_path(self, args: ReadInput) -> Path:
        return Path(args.file_path)

    def permission_rule_content(self, args: ReadInput) -> list[str]:
        return [args.file_path]

    def describe_call(self, args: ReadInput) -> str:
        return f"Read {args.file_path}"

    async def validate_input(
        self, args: ReadInput, ctx: ToolContext
    ) -> ValidationError | None:
        path = Path(args.file_path)
        if not path.is_absolute():
            path = (ctx.cwd / path).resolve()

        if (access_error := check_path_access(ctx, path)) is not None:
            return access_error
        if not path.exists():
            return ValidationError(message=f"File does not exist: {args.file_path}")
        if path.is_dir():
            return ValidationError(
                message=f"{args.file_path} is a directory. Use Glob or Bash to list it."
            )
        if path.stat().st_size > MAX_FILE_BYTES:
            return ValidationError(
                message=(
                    f"File is larger than {MAX_FILE_BYTES // 1024 // 1024}MB. "
                    "Use offset and limit to read it in slices."
                )
            )
        return None

    async def check_permissions(
        self, args: ReadInput, ctx: ToolContext
    ) -> PermissionResult:
        """Reading inside a permitted directory needs no approval."""
        path = Path(args.file_path)
        if not path.is_absolute():
            path = (ctx.cwd / path).resolve()
        if ctx.is_path_allowed(path):
            return PermissionResultAllow(reason="read within working directory")
        return await super().check_permissions(args, ctx)

    async def call(self, args: ReadInput, ctx: ToolContext) -> ToolOutput:
        path = Path(args.file_path)
        if not path.is_absolute():
            path = (ctx.cwd / path).resolve()

        text = read_text(path)
        sliced, start_line, is_partial = slice_lines(text, args.offset, args.limit)
        record_read(ctx, path, text, is_partial_view=is_partial)

        if not sliced.strip():
            return ToolOutput(
                content=f"{path} exists but is empty.",
                metadata={"path": str(path), "empty": True},
            )

        body = format_with_line_numbers(sliced, start_line)
        total_lines = len(text.split("\n"))
        if is_partial:
            shown = len(sliced.split("\n"))
            body += (
                f"\n\n[Showing lines {start_line}-{start_line + shown - 1} of "
                f"{total_lines}. Use offset and limit to read more.]"
            )
        elif total_lines > MAX_LINES:
            body += f"\n\n[File truncated at {MAX_LINES} lines.]"

        return ToolOutput(
            content=body,
            metadata={
                "path": str(path),
                "total_lines": total_lines,
                "is_partial": is_partial,
            },
        )


__all__ = ["ReadTool", "ReadInput"]
