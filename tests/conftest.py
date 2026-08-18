"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ubiquity.options import Options
from ubiquity.tool import PermissionContext, ToolContext


@pytest.fixture(autouse=True)
def isolated_session_root(tmp_path_factory, monkeypatch):
    """Redirect session persistence away from the real home directory.

    `Options.persist_session` defaults to True, so without this every test
    that calls `summon` would write transcripts under `~/.ubiquity`.
    """
    root = tmp_path_factory.mktemp("sessions")
    monkeypatch.setattr("ubiquity.sessions.store.DEFAULT_SESSION_ROOT", root)
    return root


@pytest.fixture(autouse=True)
def isolated_todo_root(tmp_path_factory, monkeypatch):
    """Redirect todo persistence away from the real home directory.

    Todos are stored per project directory, so without this a test run would
    both read and overwrite the developer's own list for this repository.
    """
    root = tmp_path_factory.mktemp("todos")
    monkeypatch.setattr("ubiquity.todos.DEFAULT_TODO_ROOT", root)
    return root


@pytest.fixture
def make_ctx(tmp_path: Path):
    """Return a factory building a `ToolContext` with the given permissions."""

    def _make(
        *,
        mode: str = "default",
        allow: set[str] | None = None,
        deny: set[str] | None = None,
        ask: set[str] | None = None,
        cwd: Path | None = None,
        **extra: Any,
    ) -> ToolContext:
        root = cwd or tmp_path
        return ToolContext(
            cwd=root,
            options=Options(cwd=root, permission_mode=mode),  # type: ignore[arg-type]
            permissions=PermissionContext(
                mode=mode,  # type: ignore[arg-type]
                allow_rules=allow or set(),
                deny_rules=deny or set(),
                ask_rules=ask or set(),
            ),
            session_id="test-session",
            **extra,
        )

    return _make
