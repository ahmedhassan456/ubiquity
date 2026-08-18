"""The Glob tool.

Results are sorted by modification time,
newest first, so the most recently touched files appear before stale ones.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ..tool import Tool, ToolContext, ValidationError
from ..types import PermissionResult, PermissionResultAllow, ToolOutput
from ._files import check_path_access

MAX_RESULTS = 500

IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    ".next", ".tox", ".idea", ".DS_Store",
}


class GlobInput(BaseModel):
    pattern: str = Field(description="Glob pattern, for example `**/*.py`.")
    path: str | None = Field(
        default=None, description="Directory to search in. Defaults to cwd."
    )


def is_ignored(path: Path, root: Path) -> bool:
    """True when any path component below `root` is a conventional build dir."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return any(part in IGNORED_DIRS for part in relative.parts)


class GlobTool(Tool[GlobInput]):
    """Find files by glob pattern."""

    name = "Glob"
    description = (
        "Fast file pattern matching. Patterns are Python glob syntax, so "
        "`**/*.py` works but brace alternation like `*.{ts,tsx}` does not: run "
        "one call per extension. Matches files only, never directories, sorted "
        "by modification time, newest first, and capped at "
        f"{MAX_RESULTS}. Common build and vendor directories are skipped."
    )
    input_model = GlobInput
    search_hint = "find files by name pattern wildcard"

    def is_read_only(self, args: GlobInput) -> bool:
        return True

    def is_concurrency_safe(self, args: GlobInput) -> bool:
        return True

    def describe_call(self, args: GlobInput) -> str:
        return f"Glob {args.pattern}"

    def _root(self, args: GlobInput, ctx: ToolContext) -> Path:
        """Resolve the directory the search is rooted at."""
        if args.path is None:
            return ctx.cwd
        path = Path(args.path)
        return path if path.is_absolute() else (ctx.cwd / path).resolve()

    async def validate_input(
        self, args: GlobInput, ctx: ToolContext
    ) -> ValidationError | None:
        root = self._root(args, ctx)
        if (access_error := check_path_access(ctx, root)) is not None:
            return access_error
        if not root.is_dir():
            return ValidationError(message=f"{root} is not a directory.")
        return None

    async def check_permissions(
        self, args: GlobInput, ctx: ToolContext
    ) -> PermissionResult:
        """Searching inside a permitted directory needs no approval."""
        if ctx.is_path_allowed(self._root(args, ctx)):
            return PermissionResultAllow(reason="search within working directory")
        return await super().check_permissions(args, ctx)

    async def call(self, args: GlobInput, ctx: ToolContext) -> ToolOutput:
        root = self._root(args, ctx)

        matches = [
            p
            for p in root.glob(args.pattern)
            if p.is_file() and not is_ignored(p, root)
        ]
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        truncated = len(matches) > MAX_RESULTS
        shown = matches[:MAX_RESULTS]

        if not shown:
            return ToolOutput(
                content=f"No files matching {args.pattern} under {root}.",
                metadata={"count": 0},
            )

        body = "\n".join(str(p) for p in shown)
        if truncated:
            body += f"\n\n[Showing {MAX_RESULTS} of {len(matches)} matches.]"

        return ToolOutput(
            content=body,
            metadata={"count": len(matches), "truncated": truncated},
        )


__all__ = ["GlobTool", "GlobInput", "is_ignored"]
