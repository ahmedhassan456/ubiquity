"""Memory files: standing instructions loaded from `UBIQUITY.md`.

Three sources, read only when `Options.memory_sources` names them:

    user      ``~/.ubiquity/UBIQUITY.md``
    project   ``UBIQUITY.md`` and ``.ubiquity/UBIQUITY.md`` in every directory
              from the filesystem root down to the working directory
    local     ``UBIQUITY.local.md`` in those same directories

Files are ordered weakest first, so the nearest one is read last and the model
weighs it most. Nothing is read unless `Options.memory_sources` names the
source, for the same reason skills are opt-in: a memory file is instructions,
and instructions picked up from the filesystem without being asked for would
make everything else in `Options` a suggestion. `Options.memory` names files
outright and is read last of all, so a caller's own file outranks a discovered
one.

A file may pull in another with an ``@path`` reference: ``@notes.md``,
``@./notes.md``, ``@~/notes.md``, or an absolute path. The included file is
loaded straight after the file that named it. Three limits keep that from
becoming a way to read the filesystem: an include is followed only within the
root its source is anchored to, only for a text extension, and only to
`MAX_INCLUDE_DEPTH` with each file loaded once however many times it is named.

References are found by scanning rather than by parsing the markdown, so
fenced blocks and inline code spans are skipped but an ``@path`` inside an HTML
block or a link title is not seen. That is a narrower contract than a full
parse would give, and it is stated rather than implied: what is documented here
is what happens.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .options import SettingSource

logger = logging.getLogger("ubiquity.memory")

MEMORY_FILE = "UBIQUITY.md"
LOCAL_MEMORY_FILE = "UBIQUITY.local.md"

MAX_MEMORY_CHARS = 40_000
MAX_INCLUDE_DEPTH = 5

TEXT_SUFFIXES = frozenset(
    {".md", ".markdown", ".txt", ".text", ".rst", ".json", ".toml", ".yaml", ".yml"}
)

INCLUDE_RE = re.compile(r"(?:^|\s)@((?:[^\s\\]|\\ )+)")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE_RE = re.compile(r"`[^`]*`")

SOURCE_DESCRIPTIONS = {
    "user": "your private instructions for every project",
    "project": "project instructions, checked into the codebase",
    "local": "the user's private project instructions, not checked in",
    "explicit": "instructions this run was started with",
}

INSTRUCTION_HEADER = (
    "The user's standing instructions are below, weakest first. They override "
    "your defaults and the guidance above, and you must follow them as "
    "written. Where two of them conflict, the later one wins."
)

TRUNCATION_MARK = (
    "\n\n[This file is longer than {limit} characters and was cut off here. "
    "Read {path} if you need the rest.]"
)


@dataclass(frozen=True, slots=True)
class MemoryFile:
    """One loaded instruction file.

    `source` is the origin it was found through, which decides how it is
    described to the model. `parent` is the file whose ``@path`` pulled it in,
    or None when it was discovered directly.
    """

    path: Path
    source: str
    content: str
    truncated: bool = False
    parent: Path | None = None


def memory_paths(source: SettingSource, cwd: Path) -> list[Path]:
    """Return the candidate files for one source, weakest first.

    The project and local sources walk from the filesystem root down to `cwd`
    rather than only looking in `cwd`, because a run is usually started in a
    subdirectory of the checkout whose instructions live at its root.
    """
    if source == "user":
        from .settings import SETTINGS_DIR

        return [Path.home() / SETTINGS_DIR / MEMORY_FILE]

    from .settings import SETTINGS_DIR

    cwd = Path(cwd).resolve()
    branch = [cwd, *cwd.parents]
    paths: list[Path] = []
    for directory in reversed(branch):
        if source == "project":
            paths.append(directory / MEMORY_FILE)
            paths.append(directory / SETTINGS_DIR / MEMORY_FILE)
        else:
            paths.append(directory / LOCAL_MEMORY_FILE)
    return paths


def _root_for(source: str, cwd: Path) -> Path:
    """The directory an include from this source may not escape.

    A user file is anchored at the home directory and a project or local file
    at the working directory. This is what stops a checked-in `UBIQUITY.md`
    from reading ``@~/.ssh/config`` into the prompt of everyone who clones the
    repository.
    """
    return Path.home() if source in ("user", "explicit") else Path(cwd).resolve()


def _within(path: Path, root: Path) -> bool:
    """True when `path` is `root` or sits underneath it."""
    return path == root or root in path.parents


def _read(path: Path) -> str | None:
    """Read one memory file, treating an unreadable one as absent.

    A malformed or unreadable file is skipped rather than raised. It is
    ambient configuration the caller may not know exists, so it must not be
    able to stop a run that would otherwise work.
    """
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        logger.warning("ignoring unreadable memory file %s", path)
        return None


def _prose_lines(content: str) -> list[str]:
    """Return the lines an ``@path`` may be found on.

    Fenced blocks are dropped whole and inline code spans are blanked, so that
    documenting the syntax in a memory file does not trigger it.
    """
    lines: list[str] = []
    fence: str | None = None
    for line in content.split("\n"):
        marker = FENCE_RE.match(line)
        if fence is not None:
            if marker is not None and marker.group(1) == fence:
                fence = None
            continue
        if marker is not None:
            fence = marker.group(1)
            continue
        lines.append(INLINE_CODE_RE.sub(" ", line))
    return lines


def include_paths(content: str, base: Path) -> list[Path]:
    """Return the files an ``@path`` in `content` refers to, in order of mention.

    Order is first mention rather than sorted, and duplicates collapse to the
    first. Both matter beyond tidiness: this text ends up in the cached prefix,
    and a listing that renders in a different order between runs costs a full
    cache miss for a difference the model cannot see.
    """
    found: list[Path] = []
    seen: set[Path] = set()
    for line in _prose_lines(content):
        for raw in INCLUDE_RE.findall(line):
            reference = raw.split("#", 1)[0].replace("\\ ", " ")
            if not reference or reference.startswith("@"):
                continue
            if not (
                reference.startswith(("./", "~/", "/"))
                or reference[0].isalnum()
                or reference[0] in "._"
            ):
                continue
            candidate = Path(reference).expanduser()
            if not candidate.is_absolute():
                candidate = base / candidate
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(resolved)
    return found


def _load_file(
    path: Path,
    source: str,
    root: Path,
    seen: set[Path],
    depth: int = 0,
    parent: Path | None = None,
) -> list[MemoryFile]:
    """Load one file and everything it includes, the file itself first."""
    resolved = path.resolve()
    if resolved in seen or depth >= MAX_INCLUDE_DEPTH:
        return []
    if not resolved.is_file():
        return []
    seen.add(resolved)

    content = _read(resolved)
    if content is None or not content.strip():
        return []

    truncated = len(content) > MAX_MEMORY_CHARS
    if truncated:
        logger.warning(
            "memory file %s is over %d characters and was truncated",
            resolved,
            MAX_MEMORY_CHARS,
        )
        content = content[:MAX_MEMORY_CHARS] + TRUNCATION_MARK.format(
            limit=MAX_MEMORY_CHARS, path=resolved
        )

    loaded = [
        MemoryFile(
            path=resolved,
            source=source,
            content=content.strip(),
            truncated=truncated,
            parent=parent,
        )
    ]

    for target in include_paths(content, resolved.parent):
        if target.suffix.lower() not in TEXT_SUFFIXES:
            logger.warning("skipping @%s: not a text file", target)
            continue
        if not _within(target, root):
            logger.warning("skipping @%s: outside %s", target, root)
            continue
        loaded.extend(_load_file(target, source, root, seen, depth + 1, resolved))
    return loaded


def load_memory(
    sources: Sequence[SettingSource],
    cwd: Path,
    explicit: Sequence[Path] = (),
) -> list[MemoryFile]:
    """Load every memory file this run is configured to read, weakest first.

    A no-op when nothing is configured, which is the default. A file reached
    twice -- named by two sources, or included after being discovered -- is
    kept only at its first, weakest position, so that adding a source can add
    instructions but never reorder the ones already there.
    """
    from .settings import ORDER

    named = set(sources)
    seen: set[Path] = set()
    loaded: list[MemoryFile] = []

    for source in ORDER:
        if source not in named:
            continue
        root = _root_for(source, cwd)
        for path in memory_paths(source, cwd):  # type: ignore[arg-type]
            loaded.extend(_load_file(path, source, root, seen))

    for path in explicit:
        loaded.extend(
            _load_file(path, "explicit", _root_for("explicit", cwd), seen)
        )
    return loaded


def render_memory(files: Sequence[MemoryFile]) -> str:
    """Render loaded memory as the system prompt section, or "" when there is none."""
    if not files:
        return ""

    blocks = [INSTRUCTION_HEADER]
    for entry in files:
        description = SOURCE_DESCRIPTIONS.get(entry.source, entry.source)
        origin = (
            f"{entry.path} ({description}, included by {entry.parent})"
            if entry.parent is not None
            else f"{entry.path} ({description})"
        )
        blocks.append(f"Contents of {origin}:\n\n{entry.content}")
    return "\n\n".join(blocks)


__all__ = [
    "MemoryFile",
    "load_memory",
    "memory_paths",
    "render_memory",
    "include_paths",
    "MEMORY_FILE",
    "LOCAL_MEMORY_FILE",
    "MAX_MEMORY_CHARS",
    "MAX_INCLUDE_DEPTH",
]
