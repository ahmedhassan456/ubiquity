"""The TodoWrite tool.

The obvious design for this tool is a whole-list replacement, and it is safe
but wasteful: to mark one task done the model restates every task, and a slip
in the restatement silently rewrites the plan. So whole-list writes are kept
and `add`, `update`, and `remove` are added alongside them, and a one-task
change costs one task.

The list also outlives the run. It is read from and written back to a
`TodoStore`, so a later run in the same directory starts from the unfinished
work instead of an empty list. See `todos` for the storage layout.

The one-in-progress invariant is enforced rather than merely suggested: an
agent that marks several items in progress at once has stopped tracking what
it is actually doing, which is the failure mode the tool exists to prevent.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..tool import Tool, ToolContext, ValidationError
from ..todos import (
    STATUS_MARK,
    MergeError,
    TodoChange,
    TodoItem,
    TodoPatch,
    TodoStatus,
    TodoStore,
    key_for,
    merge,
    render,
    store_for,
)
from ..types import PermissionResult, PermissionResultAllow, ToolOutput


class TodoInput(BaseModel):
    todos: list[TodoItem] | None = Field(
        default=None,
        description="The complete list, replacing the current one entirely.",
    )
    add: list[TodoItem] = Field(
        default_factory=list, description="Tasks to append to the current list."
    )
    remove: list[str] = Field(
        default_factory=list,
        description="Ids or exact contents of tasks to drop from the list.",
    )
    update: list[TodoPatch] = Field(
        default_factory=list, description="Changes to tasks already on the list."
    )

    @property
    def is_incremental(self) -> bool:
        """True when this write edits the list rather than replacing it."""
        return bool(self.add or self.remove or self.update)


class TodoWriteTool(Tool[TodoInput]):
    """Create and maintain a structured task list."""

    name = "TodoWrite"
    description = (
        "Create and manage a structured task list for the current work. "
        "Pass `add`, `update`, or `remove` to change individual tasks, "
        "referring to a task by its id or its exact content; pass `todos` "
        "with the complete list only when starting a plan from scratch. "
        "Exactly one task may be in_progress at a time. Mark a task completed "
        "as soon as it is done rather than batching updates. The list is "
        "stored, so it may already hold work carried over from earlier."
    )
    input_model = TodoInput
    search_hint = "task list todo plan track progress add remove"

    def is_read_only(self, args: TodoInput) -> bool:
        return True

    def describe_call(self, args: TodoInput) -> str:
        if args.is_incremental:
            parts = [
                f"+{len(args.add)}" if args.add else "",
                f"~{len(args.update)}" if args.update else "",
                f"-{len(args.remove)}" if args.remove else "",
            ]
            return f"TodoWrite ({' '.join(p for p in parts if p)})"
        count = len(args.todos or [])
        done = sum(1 for t in args.todos or [] if t.status == "completed")
        return f"TodoWrite ({done}/{count} done)"

    def key(self, ctx: ToolContext) -> str:
        """Return the storage key for the list this context owns."""
        return key_for(ctx.options, ctx.session_id, ctx.cwd, ctx.agent_id)

    def store(self, ctx: ToolContext) -> TodoStore | None:
        """Return the store backing this context's list, or None."""
        return store_for(ctx.options)

    def current(self, ctx: ToolContext) -> list[TodoItem]:
        """Return the list this write applies to.

        The store is read on every call rather than trusting the copy in
        memory. Another run in the same directory may have added or finished a
        task since this one started, and editing a stale base is how a shared
        list loses work. A subagent reads its own list here, since the key
        already separates it from its parent's. The in-memory copy is the
        fallback for a run with persistence turned off.
        """
        store = self.store(ctx)
        if store is not None:
            return store.load(self.key(ctx), ctx.cwd)
        return [TodoItem(**item) for item in ctx.extra.get("todos", [])]

    def merged(self, args: TodoInput, ctx: ToolContext) -> TodoChange:
        """Apply `args` to the stored list, raising `MergeError` on a bad ref."""
        return merge(
            self.current(ctx),
            todos=args.todos,
            add=args.add,
            remove=args.remove,
            update=args.update,
        )

    async def validate_input(
        self, args: TodoInput, ctx: ToolContext
    ) -> ValidationError | None:
        if args.todos is None and not args.is_incremental:
            return ValidationError(
                message=(
                    "Nothing to write. Pass `todos` with the complete list to "
                    "replace it, or `add`/`update`/`remove` to change it."
                )
            )
        if args.todos is not None and args.is_incremental:
            return ValidationError(
                message=(
                    "Pass either `todos` to replace the whole list or "
                    "`add`/`update`/`remove` to change it, not both."
                )
            )

        try:
            change = self.merged(args, ctx)
        except MergeError as exc:
            return ValidationError(message=str(exc))

        in_progress = [t for t in change.todos if t.status == "in_progress"]
        if len(in_progress) > 1:
            names = ", ".join(repr(t.content) for t in in_progress)
            return ValidationError(
                message=(
                    f"This would leave {len(in_progress)} tasks in_progress "
                    f"({names}). Exactly one task may be in progress at a time."
                )
            )
        return None

    async def check_permissions(
        self, args: TodoInput, ctx: ToolContext
    ) -> PermissionResult:
        """Bookkeeping the agent does about itself needs no approval."""
        return PermissionResultAllow(reason="own task list")

    async def call(self, args: TodoInput, ctx: ToolContext) -> ToolOutput:
        try:
            change = self.merged(args, ctx)
        except MergeError as exc:
            return ToolOutput(content=str(exc), is_error=True)

        ctx.extra["todos"] = [t.model_dump() for t in change.todos]
        self.persist(change, ctx)

        if not change.todos:
            return ToolOutput(content="Todo list cleared.", metadata={"todos": []})

        todos = change.todos
        done = sum(1 for t in todos if t.status == "completed")
        return ToolOutput(
            content=f"Todo list updated ({done}/{len(todos)} completed):\n{render(todos)}",
            metadata={"todos": ctx.extra["todos"], "completed": done},
        )

    def persist(self, change: TodoChange, ctx: ToolContext) -> None:
        """Store the tasks this write touched, if this context has a store."""
        store = self.store(ctx)
        if store is None:
            return
        store.apply(self.key(ctx), ctx.cwd, change)


__all__ = [
    "TodoWriteTool",
    "TodoInput",
    "TodoItem",
    "TodoPatch",
    "TodoStatus",
    "STATUS_MARK",
]
