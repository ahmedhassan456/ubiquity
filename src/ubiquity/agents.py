"""Agent files: subagent definitions written as markdown rather than code.

An agent file is a single markdown file whose frontmatter configures a subagent
and whose body is that subagent's system prompt:

    ---
    name: reviewer
    description: Reviews a diff and reports what would break
    tools: Read, Grep, Glob
    model: inherit
    ---

    You review code. Report what would break, not what you would prefer.

This is the same definition `AgentDefinition` carries in code, and it produces
exactly that -- there is no second notion of an agent here. Writing one as a
file is worth supporting because a subagent is mostly a prompt, and a prompt is
the part of a program most worth editing without editing the program.

Files are found under `.ubiquity/agents`, on the same three sources as skills
and by the same rule: nothing is discovered unless `Options.agent_sources` asks
for it, because a definition picked up from the filesystem decides what a
delegated run is told to do. `Options.agents` is applied last, so a definition
written in code overrides a discovered one of the same name rather than
colliding with it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

from .skills import IGNORED_NAMES, NAME_PATTERN, split_frontmatter
from .types import PermissionMode

if TYPE_CHECKING:
    from .options import AgentDefinition, SettingSource

logger = logging.getLogger("ubiquity.agents")

AGENTS_DIR = "agents"
LOCAL_AGENTS_DIR = "agents.local"

AGENT_SUFFIX = ".md"

PERMISSION_MODES = frozenset(get_args(PermissionMode))

ALIASES = {
    "disallowed-tools": "disallowed_tools",
    "disallowedtools": "disallowed_tools",
    "permission-mode": "permission_mode",
    "permissionmode": "permission_mode",
    "max-turns": "max_turns",
    "maxturns": "max_turns",
}

LIST_FIELDS = ("tools", "skills", "disallowed_tools")


def normalize_keys(meta: dict[str, str]) -> dict[str, str]:
    """Fold frontmatter keys to their `AgentDefinition` field names.

    Both ``disallowed-tools`` and ``disallowedTools`` reach the same field.
    Hyphens read better in frontmatter and camelCase is what a caller coming
    from JSON settings will type; rejecting either would be a spelling test
    rather than a check on anything that matters.
    """
    folded: dict[str, str] = {}
    for key, value in meta.items():
        plain = key.strip().lower()
        folded[ALIASES.get(plain, plain.replace("-", "_"))] = value
    return folded


def parse_list(value: str) -> Sequence[str]:
    """Parse a frontmatter list, accepting ``a, b`` and ``[a, b]``.

    An empty value parses to an empty sequence rather than to None, and the
    difference is the whole point: an absent key inherits everything, while
    ``tools:`` with nothing after it grants nothing. Both are things a person
    might mean, so neither is silently turned into the other.
    """
    stripped = value.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1]
    return tuple(
        item for part in stripped.split(",") if (item := part.strip().strip("'\""))
    )


def parse_agent(path: Path) -> tuple[str, AgentDefinition] | None:
    """Load one agent file as `(name, definition)`, or None when it is neither.

    A malformed file is skipped with a warning rather than raised, the same way
    a malformed skill is. These are files a caller pointed a directory at, and
    one bad file should not take down a run the other definitions would have
    served.

    A field that cannot be parsed -- a `max_turns` that is not a number, a
    `permission_mode` that is not a mode -- is dropped with a warning rather
    than guessed at. Dropping it falls back to the run's own setting, which is
    a value someone chose; guessing would invent one.
    """
    from .options import AgentDefinition

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        logger.warning("cannot read agent file %s", path)
        return None

    raw, body = split_frontmatter(text)
    meta = normalize_keys(raw)

    name = meta.pop("name", "") or path.stem
    description = meta.pop("description", "")
    prompt = body.strip()

    if not NAME_PATTERN.match(name):
        logger.warning("skipping agent at %s: unusable name %r", path, name)
        return None
    if not description:
        logger.warning(
            "skipping agent %r at %s: no description, so the model has no way "
            "to know when to delegate to it",
            name,
            path,
        )
        return None
    if not prompt:
        logger.warning(
            "skipping agent %r at %s: no body, so the subagent would be given "
            "no instructions at all",
            name,
            path,
        )
        return None

    fields: dict[str, Any] = {"description": description, "prompt": prompt}

    for field_name in LIST_FIELDS:
        if field_name in meta:
            fields[field_name] = parse_list(meta.pop(field_name))

    if model := meta.pop("model", "").strip():
        fields["model"] = model

    if mode := meta.pop("permission_mode", "").strip():
        if mode in PERMISSION_MODES:
            fields["permission_mode"] = mode
        else:
            logger.warning(
                "agent %r at %s names an unknown permission mode %r, ignoring it",
                name,
                path,
                mode,
            )

    if turns := meta.pop("max_turns", "").strip():
        try:
            fields["max_turns"] = int(turns)
        except ValueError:
            logger.warning(
                "agent %r at %s has a non-numeric max_turns %r, ignoring it",
                name,
                path,
                turns,
            )

    if meta:
        logger.debug("agent %r at %s has unused frontmatter keys: %s", name, path, meta)

    return name, AgentDefinition(**fields)


def discover(root: Path) -> list[Path]:
    """Return the agent files under `root`, sorted and depth-first.

    Subdirectories are walked so a project can group its agents, and the sort
    keeps the load order identical between runs. That matters beyond tidiness:
    the definitions end up listed in the `Agent` tool's description, which sits
    in the cached prefix.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    return [
        path
        for path in sorted(root.rglob(f"*{AGENT_SUFFIX}"))
        if path.is_file()
        and not any(part in IGNORED_NAMES for part in path.relative_to(root).parts)
    ]


def load_agents(roots: Iterable[Path]) -> dict[str, AgentDefinition]:
    """Load every agent under `roots`, keyed by name.

    Later roots win a name collision, so the ordering `Options` builds -- user,
    then project, then local -- lets a checkout override an agent it inherited
    from a home directory rather than being stuck with it.
    """
    loaded: dict[str, AgentDefinition] = {}
    for root in roots:
        for path in discover(Path(root)):
            parsed = parse_agent(path)
            if parsed is None:
                continue
            name, definition = parsed
            if name in loaded:
                logger.debug("agent %r at %s overrides an earlier definition", name, path)
            loaded[name] = definition
    return loaded


def agents_path(source: SettingSource, cwd: Path) -> Path:
    """Return the conventional agents directory for a setting source."""
    from .settings import SETTINGS_DIR

    if source == "user":
        return Path.home() / SETTINGS_DIR / AGENTS_DIR
    if source == "project":
        return Path(cwd) / SETTINGS_DIR / AGENTS_DIR
    return Path(cwd) / SETTINGS_DIR / LOCAL_AGENTS_DIR


__all__ = [
    "load_agents",
    "parse_agent",
    "discover",
    "agents_path",
    "parse_list",
    "normalize_keys",
    "AGENTS_DIR",
    "LOCAL_AGENTS_DIR",
]
