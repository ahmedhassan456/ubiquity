"""Settings files.

Three sources, read only when `Options.setting_sources` names them:

    user      ``~/.ubiquity/settings.json``
    project   ``<cwd>/.ubiquity/settings.json``
    local     ``<cwd>/.ubiquity/settings.local.json``

Nothing is read by default. A library that silently picks up configuration
from the filesystem makes a caller's explicit `Options` a suggestion, so
opting in is the whole point of the field.

Precedence runs local over project over user, and explicit `Options` over all
of them — with one exception. Permission rules are unioned rather than
overridden, because a rule in a settings file is a restriction the caller did
not write and must not be able to drop by passing a list of their own.

`permissions.allow` carries the same dual meaning as `Options.allowed_tools`:
it authorizes the calls it names, and a bare tool name in it also limits the
run to the tools listed. That is the field's documented behavior either way,
and for a file whose subject is permissions it errs toward less authority
rather than more.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .options import Options, SettingSource

logger = logging.getLogger("ubiquity.settings")

SETTINGS_DIR = ".ubiquity"
SETTINGS_FILE = "settings.json"
LOCAL_SETTINGS_FILE = "settings.local.json"

ORDER: tuple[str, ...] = ("user", "project", "local")
"""Sources from weakest to strongest, whatever order the caller lists them in."""


def settings_path(source: SettingSource, cwd: Path) -> Path:
    """Return the file a source is read from."""
    if source == "user":
        return Path.home() / SETTINGS_DIR / SETTINGS_FILE
    if source == "project":
        return Path(cwd) / SETTINGS_DIR / SETTINGS_FILE
    return Path(cwd) / SETTINGS_DIR / LOCAL_SETTINGS_FILE


def _read(path: Path) -> dict[str, Any]:
    """Parse one settings file, treating an unreadable one as absent.

    A malformed settings file is logged and skipped rather than raised: it is
    ambient configuration the caller may not know exists, so it should not be
    able to stop a run that would otherwise work.
    """
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("ignoring unreadable settings file %s", path)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def load_settings(
    sources: Sequence[SettingSource], cwd: Path
) -> dict[str, Any]:
    """Merge the named sources into one settings mapping.

    Later sources win for scalar keys; `env` is merged key by key and the
    permission lists are concatenated.
    """
    merged: dict[str, Any] = {}
    permissions: dict[str, list[str]] = {}
    env: dict[str, str] = {}

    for source in ORDER:
        if source not in sources:
            continue
        data = _read(settings_path(source, cwd))  # type: ignore[arg-type]
        for key, value in data.items():
            if key == "permissions" and isinstance(value, dict):
                for rule_key, rules in value.items():
                    if isinstance(rules, list):
                        permissions.setdefault(rule_key, []).extend(
                            str(r) for r in rules
                        )
                    else:
                        merged.setdefault("permissions", {})[rule_key] = rules
            elif key == "env" and isinstance(value, dict):
                env.update({str(k): str(v) for k, v in value.items()})
            else:
                merged[key] = value

    if permissions:
        merged["permissions"] = {**merged.get("permissions", {}), **permissions}
    if env:
        merged["env"] = env
    return merged


def apply_settings(options: Options) -> Options:
    """Return `options` with any configured settings files folded in.

    A no-op when `setting_sources` is empty, which is the default.
    """
    if not options.setting_sources:
        return options

    cwd = options.resolved_cwd()
    settings = load_settings(options.setting_sources, cwd)
    if not settings:
        return options

    changes: dict[str, Any] = {}
    permissions = settings.get("permissions") or {}

    if options.model is None and settings.get("model"):
        changes["model"] = settings["model"]
    if settings.get("env"):
        changes["env"] = {**settings["env"], **options.env}
    if permissions.get("defaultMode") and options.permission_mode == "default":
        changes["permission_mode"] = permissions["defaultMode"]

    allow = [str(r) for r in permissions.get("allow", [])]
    deny = [str(r) for r in permissions.get("deny", [])]
    ask = [str(r) for r in permissions.get("ask", [])]

    if allow:
        changes["allowed_tools"] = [*(options.allowed_tools or ()), *allow]
    if deny:
        changes["disallowed_tools"] = [*options.disallowed_tools, *deny]
    if ask:
        changes["ask_tools"] = [*options.ask_tools, *ask]

    extra_dirs = [str(d) for d in permissions.get("additionalDirectories", [])]
    if extra_dirs:
        changes["add_dirs"] = [*options.add_dirs, *extra_dirs]

    return replace(options, **changes) if changes else options


__all__ = [
    "apply_settings",
    "load_settings",
    "settings_path",
    "SETTINGS_DIR",
    "SETTINGS_FILE",
    "LOCAL_SETTINGS_FILE",
]
