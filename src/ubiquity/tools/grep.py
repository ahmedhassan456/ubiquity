"""The Grep tool.

Prefers ripgrep when it is on PATH and falls back to a pure-Python scan, so the tool works on machines without `rg`
installed rather than failing.
"""

from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from ..tool import Tool, ToolContext, ValidationError
from ..types import PermissionResult, PermissionResultAllow, ToolOutput
from ._files import check_path_access
from .glob import IGNORED_DIRS, is_ignored

MAX_MATCHES = 200


def _render_hits(
    path: Path, body: list[str], hits: list[int], context_lines: int
) -> list[str]:
    """Render one file's matches, with `context_lines` around each.

    Follows ripgrep's convention so both backends read alike: a matching line
    is joined by ``:`` and a context line by ``-``. Overlapping windows are
    merged rather than emitted twice, which is what makes a wide context on a
    dense pattern cost the same as a narrow one.
    """
    if context_lines <= 0:
        return [f"{path}:{n}:{body[n - 1]}" for n in hits]

    matched = set(hits)
    shown = sorted(
        {
            n
            for hit in hits
            for n in range(
                max(1, hit - context_lines), min(len(body), hit + context_lines) + 1
            )
        }
    )
    return [
        f"{path}:{n}:{body[n - 1]}" if n in matched else f"{path}-{n}-{body[n - 1]}"
        for n in shown
    ]


class GrepInput(BaseModel):
    pattern: str = Field(description="Regular expression to search for.")
    path: str | None = Field(
        default=None, description="File or directory to search. Defaults to cwd."
    )
    glob: str | None = Field(
        default=None, description="Restrict to files matching this glob."
    )
    output_mode: str = Field(
        default="content",
        description="One of `content`, `files_with_matches`, or `count`.",
    )
    case_insensitive: bool = Field(default=False, description="Ignore case.")
    context_lines: int = Field(
        default=0, description="Lines of context to show around each match."
    )


class GrepTool(Tool[GrepInput]):
    """Search file contents with a regular expression."""

    name = "Grep"
    description = (
        "Search file contents using a regular expression. Use `output_mode` "
        "to choose between matching lines, which is the default, matching "
        f"filenames, or per-file counts. Results are capped at {MAX_MATCHES}, "
        "so narrow the pattern or the path rather than reading a truncated "
        "list. Patterns match within a single line. Prefer this over "
        "`bash grep`."
    )
    input_model = GrepInput
    search_hint = "search text regex ripgrep find in files"

    def is_read_only(self, args: GrepInput) -> bool:
        return True

    def is_concurrency_safe(self, args: GrepInput) -> bool:
        return True

    def describe_call(self, args: GrepInput) -> str:
        return f"Grep {args.pattern}"

    def _root(self, args: GrepInput, ctx: ToolContext) -> Path:
        """Resolve the file or directory the search is rooted at."""
        if args.path is None:
            return ctx.cwd
        path = Path(args.path)
        return path if path.is_absolute() else (ctx.cwd / path).resolve()

    async def validate_input(
        self, args: GrepInput, ctx: ToolContext
    ) -> ValidationError | None:
        try:
            re.compile(args.pattern)
        except re.error as exc:
            return ValidationError(message=f"Invalid regular expression: {exc}")

        if args.output_mode not in {"content", "files_with_matches", "count"}:
            return ValidationError(
                message=(
                    f"Unknown output_mode {args.output_mode!r}. Use `content`, "
                    "`files_with_matches`, or `count`."
                )
            )

        root = self._root(args, ctx)
        if (access_error := check_path_access(ctx, root)) is not None:
            return access_error
        if not root.exists():
            return ValidationError(message=f"{root} does not exist.")
        return None

    async def check_permissions(
        self, args: GrepInput, ctx: ToolContext
    ) -> PermissionResult:
        """Searching inside a permitted directory needs no approval."""
        if ctx.is_path_allowed(self._root(args, ctx)):
            return PermissionResultAllow(reason="search within working directory")
        return await super().check_permissions(args, ctx)

    async def call(self, args: GrepInput, ctx: ToolContext) -> ToolOutput:
        root = self._root(args, ctx)
        if shutil.which("rg"):
            result = await self._ripgrep(args, root)
            if result is not None:
                return result
        return self._python_scan(args, root)

    async def _ripgrep(self, args: GrepInput, root: Path) -> ToolOutput | None:
        """Run ripgrep, returning None if it is unusable so the fallback runs."""
        cmd = ["rg", "--no-heading", "--line-number", "--color", "never"]
        if args.case_insensitive:
            cmd.append("-i")
        if args.glob:
            cmd += ["--glob", args.glob]
        if args.output_mode == "files_with_matches":
            cmd.append("--files-with-matches")
        elif args.output_mode == "count":
            cmd.append("--count")
        elif args.context_lines:
            cmd += ["--context", str(args.context_lines)]
        cmd += ["--regexp", args.pattern, str(root)]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, _ = await proc.communicate()

        if proc.returncode not in (0, 1):
            return None

        lines = stdout_b.decode("utf-8", errors="replace").splitlines()
        return self._format(lines, args)

    def _python_scan(self, args: GrepInput, root: Path) -> ToolOutput:
        """Scan files line by line without ripgrep."""
        flags = re.IGNORECASE if args.case_insensitive else 0
        regex = re.compile(args.pattern, flags)

        if root.is_file():
            files = [root]
        else:
            files = [
                p
                for p in root.rglob(args.glob or "*")
                if p.is_file() and not is_ignored(p, root)
            ]

        lines: list[str] = []
        per_file: dict[str, int] = {}
        for path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            body = text.split("\n")
            hits = [n for n, line in enumerate(body, start=1) if regex.search(line)]
            if not hits:
                continue
            per_file[str(path)] = len(hits)
            lines += _render_hits(path, body, hits, args.context_lines)

        if args.output_mode == "files_with_matches":
            lines = list(per_file)
        elif args.output_mode == "count":
            lines = [f"{p}:{n}" for p, n in per_file.items()]

        return self._format(lines, args)

    def _format(self, lines: list[str], args: GrepInput) -> ToolOutput:
        """Truncate and render match lines."""
        if not lines:
            return ToolOutput(
                content=f"No matches for {args.pattern!r}.",
                metadata={"count": 0},
            )

        truncated = len(lines) > MAX_MATCHES
        shown = lines[:MAX_MATCHES]
        body = "\n".join(shown)
        if truncated:
            body += f"\n\n[Showing {MAX_MATCHES} of {len(lines)} results.]"

        return ToolOutput(
            content=body,
            metadata={"count": len(lines), "truncated": truncated},
        )


__all__ = ["GrepTool", "GrepInput"]
