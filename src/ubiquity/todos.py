"""Todo list state: the schema, the merge rules, and on-disk persistence.

The list itself is small, but where it lives determines what the agent can do
with it. Holding it only in the message history makes it ephemeral and forces
every change to restate the whole list. Holding it on disk lets a later run
pick up unfinished work, and gives incremental edits something stable to edit.

Layout:

    <todo_dir>/<project-slug>/<key>/<task-id>.json

One file per task, not one file per list. A shared list stored as a single
document has to be rewritten whole on every change, so two runs editing
different tasks from their own stale copies overwrite each other's work
entirely. Disjoint tasks in separate files never contend, which removes the
problem rather than locking around it.

The remaining shared state is the ordering, kept as a `position` on each task.
Two runs appending at once can land on the same position, which leaves their
relative order undefined but loses nothing; ties break on id so a list always
reads back in a stable order.

Individual files are still written through a temporary file and an atomic
rename, so a crash mid-write cannot leave a half-parsed task behind.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from .sessions.store import project_slug

logger = logging.getLogger("ubiquity")

DEFAULT_TODO_ROOT = Path.home() / ".ubiquity" / "todos"

TodoStatus = Literal["pending", "in_progress", "completed"]

TodoScope = Literal["session", "project"]

STATUS_MARK = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def safe_component(value: str) -> str:
    """Reduce a string to something safe to use as one path component.

    Keys and task ids reach the filesystem, and an id carrying `..` or a
    separator would let a write escape the todo directory.
    """
    return _UNSAFE.sub("-", value).strip("-") or "unnamed"


class TodoItem(BaseModel):
    """One task on the list."""

    content: str = Field(description="Imperative description of the task.")
    status: TodoStatus = Field(default="pending")
    active_form: str = Field(
        default="", description="Present-continuous form shown while in progress."
    )
    id: str = Field(
        default="", description="Stable identifier; assigned on write if omitted."
    )
    position: int = Field(default=0, description="Sort key within the list.")

    def with_id(self) -> TodoItem:
        """Return this item with an identifier, minting one if it has none."""
        if self.id:
            return self
        return self.model_copy(update={"id": uuid4().hex[:8]})

    @property
    def sort_key(self) -> tuple[int, str]:
        """Order within a list, with ties broken deterministically."""
        return (self.position, self.id)


class TodoPatch(BaseModel):
    """A change to one existing task."""

    task: str = Field(description="The id or exact content of the task to change.")
    status: TodoStatus | None = None
    content: str | None = None
    active_form: str | None = None


@dataclass(slots=True)
class TodoChange:
    """The outcome of one write: the new list, and what has to reach disk.

    `written` and `removed` are what make per-task storage work. Persisting the
    whole of `todos` would reintroduce the shared document that the layout
    exists to avoid, so only the tasks this write actually touched are stored.
    """

    todos: list[TodoItem]
    written: list[TodoItem] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    replaced: bool = False


def render(todos: Sequence[TodoItem]) -> str:
    """Render the list the way the model sees it after a write."""
    return "\n".join(f"{STATUS_MARK[t.status]} ({t.id}) {t.content}" for t in todos)


def locate(todos: Sequence[TodoItem], reference: str) -> int | None:
    """Return the index of the task `reference` names, or None.

    A reference is matched against ids first and contents second, so the model
    may name a task either way. Content matching is what makes the tool usable
    without the model having to track generated ids.
    """
    for index, todo in enumerate(todos):
        if todo.id == reference:
            return index
    for index, todo in enumerate(todos):
        if todo.content == reference:
            return index
    return None


class MergeError(Exception):
    """Raised when an incremental change names something that is not there."""


def merge(
    current: Sequence[TodoItem],
    *,
    todos: Sequence[TodoItem] | None = None,
    add: Iterable[TodoItem] = (),
    remove: Iterable[str] = (),
    update: Iterable[TodoPatch] = (),
) -> TodoChange:
    """Apply one write to the current list and report what changed.

    `todos` replaces the list outright. The incremental operations apply in the
    order add, update, remove, so a single call may append a task and complete
    an earlier one without the two interfering.

    Raises `MergeError` when a removal or update names a task that is not on
    the list. Silently ignoring it would leave the model believing it had made
    a change it did not make, which is worse than a retry.
    """
    if todos is not None:
        fresh = [
            t.with_id().model_copy(update={"position": index + 1})
            for index, t in enumerate(todos)
        ]
        return TodoChange(todos=fresh, written=list(fresh), replaced=True)

    merged = [t.model_copy() for t in current]
    written: dict[str, TodoItem] = {}
    removed: list[str] = []
    next_position = max((t.position for t in merged), default=0)

    for item in add:
        if locate(merged, item.content) is not None:
            continue
        next_position += 1
        fresh = item.with_id().model_copy(update={"position": next_position})
        merged.append(fresh)
        written[fresh.id] = fresh

    for patch in update:
        index = locate(merged, patch.task)
        if index is None:
            raise MergeError(_unknown(patch.task, merged))
        changes = patch.model_dump(exclude={"task"}, exclude_none=True)
        merged[index] = merged[index].model_copy(update=changes)
        written[merged[index].id] = merged[index]

    for reference in remove:
        index = locate(merged, reference)
        if index is None:
            raise MergeError(_unknown(reference, merged))
        dropped = merged.pop(index)
        written.pop(dropped.id, None)
        removed.append(dropped.id)

    return TodoChange(todos=merged, written=list(written.values()), removed=removed)


def _unknown(reference: str, todos: Sequence[TodoItem]) -> str:
    """Describe a failed lookup in terms the model can act on."""
    if not todos:
        return f"No task matches {reference!r}: the todo list is empty."
    known = ", ".join(f"{t.id} ({t.content!r})" for t in todos)
    return f"No task matches {reference!r}. The list holds: {known}."


class TodoStore:
    """Reads and writes todo lists under a root directory.

    Every method addresses one task, or reads the whole directory. There is
    deliberately no write-the-list operation: that is the shared document the
    per-task layout exists to avoid.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_TODO_ROOT

    def dir_for(self, key: str, cwd: Path) -> Path:
        """Return the directory holding one todo list."""
        return self.root / project_slug(cwd) / safe_component(key)

    def path_for(self, key: str, cwd: Path, task_id: str) -> Path:
        """Return the file backing one task."""
        return self.dir_for(key, cwd) / f"{safe_component(task_id)}.json"

    def load(self, key: str, cwd: Path) -> list[TodoItem]:
        """Return every stored task, in list order.

        An unreadable file is skipped rather than raising: one corrupt task is
        a nuisance, not a reason to lose the rest of the list or refuse to run.
        """
        directory = self.dir_for(key, cwd)
        if not directory.is_dir():
            return []
        found: list[TodoItem] = []
        for path in directory.glob("*.json"):
            item = self._read(path)
            if item is not None:
                found.append(item)
        found.sort(key=lambda t: t.sort_key)
        return found

    def _read(self, path: Path) -> TodoItem | None:
        """Parse one task file, or None when it cannot be read."""
        try:
            return TodoItem(**json.loads(path.read_text(encoding="utf-8"))["task"])
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
            logger.warning("ignoring unreadable task file %s", path)
            return None

    def write(self, key: str, cwd: Path, task: TodoItem) -> Path:
        """Store one task, replacing whatever was under its id."""
        path = self.path_for(key, cwd, task.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"updated_at": time.time(), "cwd": str(cwd), "task": task.model_dump()}
        tmp = path.with_suffix(f".{uuid4().hex[:8]}.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return path

    def delete(self, key: str, cwd: Path, task_id: str) -> bool:
        """Remove one task. Returns True when a file was removed."""
        path = self.path_for(key, cwd, task_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def clear(self, key: str, cwd: Path) -> int:
        """Remove every task in a list. Returns how many were removed.

        The directory goes too when nothing else is left in it. An empty
        directory per discarded list is the same slow accumulation as an empty
        list per discarded list, which is what clearing is here to avoid.
        """
        directory = self.dir_for(key, cwd)
        if not directory.is_dir():
            return 0
        removed = 0
        for path in directory.glob("*.json"):
            path.unlink()
            removed += 1
        if not any(directory.iterdir()):
            directory.rmdir()
        return removed

    def apply(self, key: str, cwd: Path, change: TodoChange) -> None:
        """Persist one merge.

        A whole-list replacement is the only destructive case, and it is
        destructive because the model asked for it: the stored list is cleared
        before the new one lands, so tasks dropped by the replacement do not
        linger as files nobody references.
        """
        if change.replaced:
            self.clear(key, cwd)
        for task in change.written:
            self.write(key, cwd, task)
        for task_id in change.removed:
            self.delete(key, cwd, task_id)


def store_for(options: object) -> TodoStore | None:
    """Return the store for a run, or None when persistence is off."""
    if not getattr(options, "persist_todos", False):
        return None
    return TodoStore(getattr(options, "todo_dir", None))


def key_for(
    options: object,
    session_id: str,
    cwd: Path,
    agent_id: str | None = None,
) -> str:
    """Return the storage key for a list, most specific scope first.

    A subagent's list is its own, keyed by agent id. Without that a delegated
    side task would edit the plan its parent is still working through.
    """
    if agent_id is not None:
        return agent_id
    if getattr(options, "todo_scope", "project") == "session":
        return session_id
    return project_slug(cwd)


CARRIED_OVER_TEMPLATE = """\
An unfinished todo list is stored for this directory:
{rendered}

Continue from it rather than starting a new list. Use TodoWrite with `add`, \
`update`, or `remove` to change individual tasks, and remove anything that is \
no longer relevant to the current request.\
"""


def carried_over_context(todos: Sequence[TodoItem]) -> str | None:
    """Describe a restored list for injection into the first prompt.

    Returns None when nothing is worth mentioning. A list whose every task is
    already completed is treated as nothing: surfacing it would invite the
    model to reason about finished work from an unrelated run.
    """
    if not todos or all(t.status == "completed" for t in todos):
        return None
    return CARRIED_OVER_TEMPLATE.format(rendered=render(todos))


__all__ = [
    "TodoStore",
    "TodoItem",
    "TodoPatch",
    "TodoChange",
    "TodoStatus",
    "TodoScope",
    "MergeError",
    "merge",
    "locate",
    "render",
    "safe_component",
    "carried_over_context",
    "store_for",
    "key_for",
    "STATUS_MARK",
    "DEFAULT_TODO_ROOT",
]
