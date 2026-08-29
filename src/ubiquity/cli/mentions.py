"""`@path` mentions in a prompt.

Naming a file is the most common thing a prompt does, and making the agent
find it costs a tool call and a round-trip for something the user already
knew. A mention resolves the path here and puts the file's text in front of
the model with the prompt, so the first response is already informed by it.

The syntax is the one memory files already use -- `memory.INCLUDE_RE` -- so
`@src/thing.py` means the same thing wherever it is written. A mention that
does not resolve is left in the prompt untouched rather than erased: the model
can still read it as the reference the user meant, and `Read` remains
available for anything not attached here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..memory import INCLUDE_RE, TRAILING_PUNCTUATION

MAX_FILE_BYTES = 100_000
MAX_TOTAL_BYTES = 400_000
SKIP_DIRECTORIES = frozenset({".git", "__pycache__", "node_modules", ".venv", ".mypy_cache"})


@dataclass(frozen=True, slots=True)
class Mention:
    """One `@path` in a prompt, and what it turned out to be."""

    reference: str
    path: Path
    kind: str


def find(text: str, base: Path) -> list[Mention]:
    """Return the mentions in `text`, in order, each one only once.

    `kind` is `file`, `directory`, or `missing`; the caller decides what to do
    with each, since a missing path is a typo worth reporting rather than an
    error worth refusing the prompt over.
    """
    found: list[Mention] = []
    seen: set[Path] = set()
    for raw in INCLUDE_RE.findall(text):
        reference = raw.split("#", 1)[0].replace("\\ ", " ")
        reference = reference.rstrip(TRAILING_PUNCTUATION)
        if not reference or reference.startswith("@"):
            continue
        candidate = Path(reference).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        kind = (
            "directory"
            if resolved.is_dir()
            else "file"
            if resolved.is_file()
            else "missing"
        )
        found.append(Mention(reference=reference, path=resolved, kind=kind))
    return found


def _read(path: Path, budget: int) -> str | None:
    """Read `path` as text, truncated to `budget` bytes, or None if binary.

    Truncation is by bytes rather than lines because the point of the budget
    is the context window, and a partial file with a note saying so is more
    use to the model than nothing at all.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None
    clipped = data[:budget]
    if b"\x00" in clipped:
        return None
    text = clipped.decode("utf-8", errors="replace")
    if len(data) > len(clipped):
        text += f"\n… truncated at {budget} bytes of {len(data)}\n"
    return text


def _listing(path: Path, base: Path, limit: int = 200) -> str:
    """Render a directory as the names it holds, one per line."""
    try:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    except OSError:
        return "(unreadable)"
    names = [
        f"{entry.name}/" if entry.is_dir() else entry.name
        for entry in entries
        if entry.name not in SKIP_DIRECTORIES
    ]
    shown = names[:limit]
    if len(names) > limit:
        shown.append(f"… {len(names) - limit} more")
    return "\n".join(shown)


def expand(text: str, base: Path) -> tuple[str, list[Mention]]:
    """Return `text` with its mentioned files attached, and what was attached.

    The prompt itself is never rewritten. The attachments follow it, so the
    model reads the user's words first and the file contents as context for
    them -- and so a prompt that reads naturally on screen still reads
    naturally in the transcript.
    """
    mentions = find(text, base)
    if not mentions:
        return text, []

    blocks: list[str] = []
    attached: list[Mention] = []
    budget = MAX_TOTAL_BYTES
    for mention in mentions:
        if budget <= 0:
            break
        if mention.kind == "directory":
            body = _listing(mention.path, base)
            label = "Directory"
        elif mention.kind == "file":
            body_or_none = _read(mention.path, min(MAX_FILE_BYTES, budget))
            if body_or_none is None:
                continue
            body = body_or_none
            label = "Contents of"
        else:
            continue
        budget -= len(body.encode("utf-8", errors="ignore"))
        blocks.append(f"{label} {display(mention, base)}:\n\n{body.rstrip()}")
        attached.append(mention)

    if not blocks:
        return text, []
    joined = "\n\n".join(blocks)
    return f"{text}\n\n<attached-files>\n{joined}\n</attached-files>", attached


def display(mention: Mention, base: Path) -> str:
    """Render a mention's path relative to `base` when it lives under it."""
    try:
        return str(mention.path.relative_to(base))
    except ValueError:
        return str(mention.path)


__all__ = [
    "Mention",
    "find",
    "expand",
    "display",
    "MAX_FILE_BYTES",
    "MAX_TOTAL_BYTES",
    "SKIP_DIRECTORIES",
]
