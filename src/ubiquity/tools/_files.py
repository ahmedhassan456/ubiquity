"""Shared filesystem helpers for the file tools.

Centralizes the read-before-write bookkeeping shared by `Read`, `Write`, and
`Edit`: every read records a `FileState`, and every write consults it to refuse
edits that would silently clobber changes the agent has not seen.

Line endings are normalized to ``\\n`` before hashing so that a CRLF checkout
does not read as a modification.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..tool import FileState, ToolContext, ValidationError

MAX_LINES = 2000
MAX_LINE_WIDTH = 2000
MAX_FILE_BYTES = 10 * 1024 * 1024


def normalize(text: str) -> str:
    """Normalize line endings to ``\\n``."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def content_hash(text: str) -> str:
    """Return a stable hash of `text` after line-ending normalization."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def record_read(
    ctx: ToolContext, path: Path, text: str, *, is_partial_view: bool = False
) -> None:
    """Record that `path` was read, so later writes can detect staleness."""
    try:
        stat = path.stat()
    except OSError:
        return
    ctx.file_state[path.resolve()] = FileState(
        mtime=stat.st_mtime,
        size=stat.st_size,
        content_hash=content_hash(text),
        is_partial_view=is_partial_view,
    )


def check_write_allowed(ctx: ToolContext, path: Path) -> ValidationError | None:
    """Verify the agent has read `path` and that it has not changed since.

    Returns None when the write may proceed. A file that does not exist yet is
    always writable, since there is nothing to clobber.
    """
    resolved = path.resolve()
    if not resolved.exists():
        return None

    state = ctx.file_state.get(resolved)
    if state is None:
        return ValidationError(
            message="File has not been read yet. Read it first before writing to it.",
            error_code=2,
        )
    if state.is_partial_view:
        return ValidationError(
            message=(
                "Only part of this file has been read, so a write could clobber "
                "the rest. Read it again with `limit` set past its last line."
            ),
            error_code=2,
        )

    try:
        current_mtime = resolved.stat().st_mtime
    except OSError as exc:
        return ValidationError(message=f"Cannot stat {path}: {exc}", error_code=4)

    if current_mtime > state.mtime:
        return ValidationError(
            message=(
                "File has been modified since read, either by the user or by a "
                "linter. Read it again before attempting to write it."
            ),
            error_code=3,
        )
    return None


def check_path_access(ctx: ToolContext, path: Path) -> ValidationError | None:
    """Verify `path` is inside cwd or an explicitly permitted directory."""
    if ctx.is_path_allowed(path):
        return None
    return ValidationError(
        message=(
            f"{path} is outside the working directory. Permitted roots: "
            f"{ctx.cwd} plus {sorted(str(d) for d in ctx.additional_dirs) or 'none'}."
        ),
        error_code=5,
    )


def read_text(path: Path) -> str:
    """Read a file as UTF-8, replacing undecodable bytes."""
    return path.read_text(encoding="utf-8", errors="replace")


def format_with_line_numbers(text: str, start_line: int = 1) -> str:
    """Render `text` in ``cat -n`` style, the format the model expects.

    Lines wider than `MAX_LINE_WIDTH` are truncated so that a minified bundle
    cannot flood the context window.
    """
    out: list[str] = []
    for i, line in enumerate(text.split("\n"), start=start_line):
        if len(line) > MAX_LINE_WIDTH:
            line = line[:MAX_LINE_WIDTH] + "... [line truncated]"
        out.append(f"{i:6d}\t{line}")
    return "\n".join(out)


def slice_lines(
    text: str, offset: int | None, limit: int | None
) -> tuple[str, int, bool]:
    """Return `(sliced_text, start_line, is_partial)` for an offset/limit read.

    `offset` is 1-indexed to match the line numbers shown to the model. When
    neither bound is given, the file is capped at `MAX_LINES`.
    """
    lines = text.split("\n")
    total = len(lines)
    start = max(1, offset or 1)
    count = limit if limit is not None else MAX_LINES
    end = min(total, start - 1 + count)
    sliced = "\n".join(lines[start - 1 : end])
    is_partial = start > 1 or end < total
    return sliced, start, is_partial


__all__ = [
    "MAX_LINES",
    "MAX_LINE_WIDTH",
    "MAX_FILE_BYTES",
    "normalize",
    "content_hash",
    "record_read",
    "check_write_allowed",
    "check_path_access",
    "read_text",
    "format_with_line_numbers",
    "slice_lines",
]
