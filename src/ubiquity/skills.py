"""Skills: instructions the agent loads only when it needs them.

A skill is a directory holding a `SKILL.md` whose YAML-style frontmatter names
it and says when to use it, and whose body is the procedure to follow. Anything
else in the directory -- checklists, scripts, reference tables -- is a bundled
file the body can point at.

The reason a skill is not simply appended to the system prompt is cost. A
useful procedure runs to hundreds of lines, and a run that carries a dozen of
them pays for all twelve on every request while using at most one. So loading
happens in three steps, each paid for only when it earns its place: every
skill's name and description sit in the `Skill` tool's description, which is a
line or two each and is budgeted; that tool returns one skill's full body when
the model decides the task matches; and the body points at bundled files the
model reads with the ordinary file tools.

Nothing is discovered unless the caller asks for it, for the same reason
settings are not: instructions picked up from the filesystem would make an
explicit `Options` a suggestion, and a skill has more say over a run than any
setting does. `Options.skills` names directories outright and
`Options.skill_sources` opts into the conventional ones.

Only `name` and `description` are read from the frontmatter. Other keys parse
and are kept on the skill for a caller to inspect, but nothing in this SDK acts
on them -- in particular there is no per-skill tool gating, so a frontmatter
`allowed-tools` restricts nothing here.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .options import SettingSource

logger = logging.getLogger("ubiquity.skills")

SKILL_FILE = "SKILL.md"
SKILLS_DIR = "skills"
LOCAL_SKILLS_DIR = "skills.local"

NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
"""A skill name is an identifier the model types back, not a path or a phrase."""

MAX_BUNDLED_LISTED = 40

IGNORED_NAMES = {".DS_Store", "__pycache__", ".git"}


@dataclass(slots=True)
class Skill:
    """One loaded skill.

    `path` is the `SKILL.md` itself, so `root` is the directory that owns the
    bundled files. Both are absolute, because they are handed to the model to
    read back and a relative path would be resolved against the wrong place.
    """

    name: str
    description: str
    body: str
    path: Path
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def root(self) -> Path:
        """The directory holding this skill and its bundled files."""
        return self.path.parent

    def bundled_files(self) -> list[Path]:
        """Return the other files in this skill's directory, sorted.

        The listing is capped: a skill that ships a large tree should describe
        it in its body, and pasting hundreds of paths into a tool result would
        spend the context the three-step loading exists to save.
        """
        found: list[Path] = []
        for entry in sorted(self.root.rglob("*")):
            if len(found) >= MAX_BUNDLED_LISTED:
                break
            if not entry.is_file() or entry == self.path:
                continue
            if any(part in IGNORED_NAMES for part in entry.relative_to(self.root).parts):
                continue
            found.append(entry)
        return found

    def render(self) -> str:
        """Return the text the `Skill` tool hands back to the model."""
        sections = [f"# Skill: {self.name}\n\n{self.description}", self.body.strip()]
        bundled = self.bundled_files()
        if bundled:
            listed = "\n".join(f"  {p}" for p in bundled)
            sections.append(
                "Files bundled with this skill, readable with Read or Bash:\n"
                f"{listed}"
            )
        return "\n\n".join(s for s in sections if s)


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split leading `---` frontmatter from the body.

    Deliberately not a YAML parser. Skill frontmatter is a handful of scalars,
    and taking a YAML dependency to read them would pull a package into every
    install for four lines of text. Values are read as strings with surrounding
    quotes stripped; a key whose value spans lines is not supported and is
    reported rather than half-parsed.

    Returns `({}, text)` when there is no frontmatter, which is what makes a
    plain markdown file a body with no name -- rejected later, by the loader.
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    closing = next(
        (i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if closing is None:
        return {}, text

    meta: dict[str, str] = {}
    for line in lines[1:closing]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, sep, value = stripped.partition(":")
        if not sep:
            logger.debug("skipping unparsable frontmatter line: %s", stripped)
            continue
        meta[key.strip()] = value.strip().strip("'\"")
    return meta, "\n".join(lines[closing + 1 :]).lstrip("\n")


def load_skill(path: Path) -> Skill | None:
    """Load one `SKILL.md`, or return None when it cannot be a skill.

    A skill missing a name or a description is skipped with a warning rather
    than raised. These are files the caller pointed at a directory of, often
    written by someone else, and one malformed file should not take down a run
    that the other skills would have served.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("cannot read skill file %s", path)
        return None

    meta, body = split_frontmatter(text)
    name = meta.pop("name", "") or path.parent.name
    description = meta.pop("description", "")

    if not NAME_PATTERN.match(name):
        logger.warning("skipping skill at %s: unusable name %r", path, name)
        return None
    if not description:
        logger.warning(
            "skipping skill %r at %s: no description, so the model has no way "
            "to know when to use it",
            name,
            path,
        )
        return None

    return Skill(
        name=name, description=description, body=body, path=path.resolve(), metadata=meta
    )


def discover(root: Path) -> list[Path]:
    """Return the `SKILL.md` files under `root`.

    A root is either one skill directory or a directory of them, because both
    are natural things for a caller to point at and telling them apart costs
    one `exists` check.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    direct = root / SKILL_FILE
    if direct.is_file():
        return [direct]
    return [
        child / SKILL_FILE
        for child in sorted(root.iterdir())
        if child.is_dir()
        and child.name not in IGNORED_NAMES
        and (child / SKILL_FILE).is_file()
    ]


def load_skills(roots: Iterable[Path]) -> dict[str, Skill]:
    """Load every skill under `roots`, keyed by name.

    Later roots win a name collision, so the ordering `Options` builds -- the
    conventional directories first, the caller's explicit ones after -- lets a
    project override a skill it inherited rather than being stuck with it.
    """
    loaded: dict[str, Skill] = {}
    for root in roots:
        for path in discover(Path(root)):
            skill = load_skill(path)
            if skill is None:
                continue
            if skill.name in loaded:
                logger.debug(
                    "skill %r at %s overrides the one at %s",
                    skill.name,
                    skill.path,
                    loaded[skill.name].path,
                )
            loaded[skill.name] = skill
    return loaded


def skills_path(source: SettingSource, cwd: Path) -> Path:
    """Return the conventional skills directory for a setting source."""
    from .settings import SETTINGS_DIR

    if source == "user":
        return Path.home() / SETTINGS_DIR / SKILLS_DIR
    if source == "project":
        return Path(cwd) / SETTINGS_DIR / SKILLS_DIR
    return Path(cwd) / SETTINGS_DIR / LOCAL_SKILLS_DIR


def select(skills: dict[str, Skill], names: Sequence[str] | None) -> dict[str, Skill]:
    """Restrict `skills` to `names`, or return all of them when None.

    None and an empty sequence mean different things, the same way they do for
    a subagent's tools: None inherits everything, `()` grants nothing. A name
    that matches no loaded skill is dropped with a warning, since the
    alternative is a subagent silently missing the one skill it was defined to
    use.
    """
    if names is None:
        return dict(skills)
    chosen: dict[str, Skill] = {}
    for name in names:
        if name in skills:
            chosen[name] = skills[name]
        else:
            logger.warning("no skill named %r is loaded", name)
    return chosen


__all__ = [
    "Skill",
    "load_skill",
    "load_skills",
    "discover",
    "select",
    "skills_path",
    "split_frontmatter",
    "SKILL_FILE",
]
