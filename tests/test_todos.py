"""Tests for the todo list: incremental edits and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ubiquity import AgentDefinition, HookMatcher, Options, summon
from ubiquity.todos import (
    MergeError,
    TodoItem,
    TodoPatch,
    TodoStore,
    carried_over_context,
    key_for,
    locate,
    merge,
    store_for,
)
from ubiquity.tools import TodoWriteTool
from ubiquity.tools.todo import TodoInput


def items(*specs: tuple[str, str]) -> list[TodoItem]:
    return [TodoItem(content=c, status=s, id=c[:2]) for c, s in specs]


def test_a_whole_list_write_replaces_everything() -> None:
    current = items(("old", "completed"))
    result = merge(current, todos=[TodoItem(content="new")]).todos
    assert [t.content for t in result] == ["new"]


def test_every_task_gets_an_id() -> None:
    result = merge([], todos=[TodoItem(content="a"), TodoItem(content="b")]).todos
    assert all(t.id for t in result)
    assert result[0].id != result[1].id


def test_add_appends_without_disturbing_the_rest() -> None:
    current = items(("first", "completed"), ("second", "in_progress"))
    result = merge(current, add=[TodoItem(content="third")]).todos
    assert [t.content for t in result] == ["first", "second", "third"]
    assert result[1].status == "in_progress"


def test_adding_a_task_already_on_the_list_is_a_no_op() -> None:
    """Re-adding must not silently duplicate a task the model already tracks."""
    current = items(("first", "pending"))
    result = merge(current, add=[TodoItem(content="first")]).todos
    assert len(result) == 1


def test_remove_drops_a_task_by_id_or_by_content() -> None:
    current = items(("first", "pending"), ("second", "pending"))
    assert [t.content for t in merge(current, remove=["fi"]).todos] == ["second"]
    assert [t.content for t in merge(current, remove=["second"]).todos] == ["first"]


def test_update_changes_only_the_named_task() -> None:
    current = items(("first", "pending"), ("second", "pending"))
    result = merge(current, update=[TodoPatch(task="first", status="completed")]).todos
    assert result[0].status == "completed"
    assert result[1].status == "pending"


def test_update_leaves_fields_it_does_not_mention_alone() -> None:
    current = [TodoItem(content="first", active_form="Doing first", id="fi")]
    result = merge(current, update=[TodoPatch(task="fi", status="in_progress")]).todos
    assert result[0].active_form == "Doing first"
    assert result[0].content == "first"


def test_operations_apply_add_then_update_then_remove() -> None:
    """One call may append a task and finish an earlier one."""
    current = items(("first", "in_progress"))
    result = merge(
        current,
        add=[TodoItem(content="second")],
        update=[TodoPatch(task="first", status="completed")],
    ).todos
    assert [(t.content, t.status) for t in result] == [
        ("first", "completed"),
        ("second", "pending"),
    ]


def test_an_unknown_reference_raises_rather_than_doing_nothing() -> None:
    """A silent no-op leaves the model believing it made a change it did not."""
    current = items(("first", "pending"))
    for kwargs in ({"remove": ["ghost"]}, {"update": [TodoPatch(task="ghost")]}):
        try:
            merge(current, **kwargs)
        except MergeError as exc:
            assert "ghost" in str(exc)
            assert "first" in str(exc)
        else:
            raise AssertionError(f"{kwargs} should have raised")


def test_locate_prefers_an_id_over_a_content_match() -> None:
    todos = [TodoItem(content="b", id="a"), TodoItem(content="a", id="b")]
    assert locate(todos, "a") == 0


async def test_the_tool_rejects_a_write_with_nothing_in_it(make_ctx) -> None:
    error = await TodoWriteTool().validate_input(TodoInput(), make_ctx())
    assert error is not None
    assert "Nothing to write" in error.message


async def test_the_tool_rejects_mixing_replacement_with_edits(make_ctx) -> None:
    error = await TodoWriteTool().validate_input(
        TodoInput(todos=[TodoItem(content="a")], remove=["b"]),
        make_ctx(),
    )
    assert error is not None
    assert "not both" in error.message


async def test_an_added_task_cannot_smuggle_in_a_second_in_progress(make_ctx) -> None:
    """The invariant holds against the merged list, not just the input."""
    ctx = make_ctx()
    await TodoWriteTool().call(
        TodoInput(todos=[TodoItem(content="first", status="in_progress")]), ctx
    )
    error = await TodoWriteTool().validate_input(
        TodoInput(add=[TodoItem(content="second", status="in_progress")]), ctx
    )
    assert error is not None
    assert "in_progress" in error.message


async def test_an_unknown_reference_reaching_call_is_an_error_not_a_crash(
    make_ctx,
) -> None:
    out = await TodoWriteTool().call(TodoInput(remove=["ghost"]), make_ctx())
    assert out.is_error
    assert "ghost" in out.content


async def test_edits_accumulate_across_calls_in_one_run(make_ctx) -> None:
    ctx = make_ctx()
    tool = TodoWriteTool()
    await tool.call(TodoInput(todos=[TodoItem(content="first")]), ctx)
    await tool.call(TodoInput(add=[TodoItem(content="second")]), ctx)
    out = await tool.call(
        TodoInput(update=[TodoPatch(task="first", status="completed")]), ctx
    )
    assert "1/2 completed" in out.content


def storage(ctx) -> tuple[TodoStore, str]:
    store = store_for(ctx.options)
    assert store is not None
    return store, key_for(ctx.options, ctx.session_id, ctx.cwd, ctx.agent_id)


async def test_each_task_gets_its_own_file(make_ctx) -> None:
    """One file per task is what keeps two runs from overwriting each other."""
    ctx = make_ctx()
    await TodoWriteTool().call(
        TodoInput(todos=[TodoItem(content="first"), TodoItem(content="second")]), ctx
    )

    store, key = storage(ctx)
    files = sorted(store.dir_for(key, ctx.cwd).glob("*.json"))
    assert len(files) == 2
    stored = [json.loads(f.read_text())["task"] for f in files]
    assert {t["content"] for t in stored} == {"first", "second"}


async def test_an_edit_only_rewrites_the_tasks_it_touched(make_ctx) -> None:
    ctx = make_ctx()
    tool = TodoWriteTool()
    await tool.call(
        TodoInput(todos=[TodoItem(content="first"), TodoItem(content="second")]), ctx
    )
    store, key = storage(ctx)
    stamps = {
        f.name: f.stat().st_mtime_ns for f in store.dir_for(key, ctx.cwd).glob("*.json")
    }

    before = {t.content: t.id for t in tool.current(ctx)}
    await tool.call(TodoInput(update=[TodoPatch(task="first", status="completed")]), ctx)

    untouched = store.path_for(key, ctx.cwd, before["second"])
    assert untouched.stat().st_mtime_ns == stamps[untouched.name]


async def test_removing_a_task_deletes_its_file(make_ctx) -> None:
    ctx = make_ctx()
    tool = TodoWriteTool()
    await tool.call(
        TodoInput(todos=[TodoItem(content="first"), TodoItem(content="second")]), ctx
    )
    await tool.call(TodoInput(remove=["first"]), ctx)

    store, key = storage(ctx)
    assert len(list(store.dir_for(key, ctx.cwd).glob("*.json"))) == 1
    assert [t.content for t in tool.current(ctx)] == ["second"]


async def test_a_replacement_clears_tasks_it_dropped(make_ctx) -> None:
    """A whole-list write is destructive by intent; no orphans may survive."""
    ctx = make_ctx()
    tool = TodoWriteTool()
    await tool.call(
        TodoInput(todos=[TodoItem(content="first"), TodoItem(content="second")]), ctx
    )
    await tool.call(TodoInput(todos=[TodoItem(content="only")]), ctx)

    store, key = storage(ctx)
    assert len(list(store.dir_for(key, ctx.cwd).glob("*.json"))) == 1
    assert [t.content for t in tool.current(ctx)] == ["only"]


async def test_a_concurrent_run_does_not_lose_the_others_tasks(make_ctx) -> None:
    """Two contexts hold stale bases; neither may erase the other's work."""
    first = make_ctx()
    second = make_ctx()
    tool = TodoWriteTool()

    await tool.call(TodoInput(todos=[TodoItem(content="shared")]), first)
    await tool.call(TodoInput(add=[TodoItem(content="from first")]), first)
    await tool.call(TodoInput(add=[TodoItem(content="from second")]), second)

    assert {t.content for t in tool.current(first)} == {
        "shared",
        "from first",
        "from second",
    }


async def test_the_list_reads_back_in_the_order_it_was_built(make_ctx) -> None:
    """Ordering is the one piece of shared state a per-task layout must carry."""
    ctx = make_ctx()
    tool = TodoWriteTool()
    await tool.call(TodoInput(todos=[TodoItem(content="first")]), ctx)
    for content in ("second", "third", "fourth"):
        await tool.call(TodoInput(add=[TodoItem(content=content)]), ctx)

    assert [t.content for t in tool.current(ctx)] == [
        "first",
        "second",
        "third",
        "fourth",
    ]


def test_a_task_id_cannot_escape_the_todo_directory(tmp_path: Path) -> None:
    store = TodoStore(tmp_path / "root")
    path = store.path_for("k", tmp_path, "../../escaped")
    assert store.dir_for("k", tmp_path) == path.parent


async def test_a_fresh_context_in_the_same_directory_sees_the_stored_list(
    make_ctx,
) -> None:
    """This is the point of persistence: a later run continues the plan."""
    await TodoWriteTool().call(
        TodoInput(todos=[TodoItem(content="ship it")]), make_ctx()
    )
    later = make_ctx()
    assert [t.content for t in TodoWriteTool().current(later)] == ["ship it"]

    out = await TodoWriteTool().call(
        TodoInput(update=[TodoPatch(task="ship it", status="completed")]), later
    )
    assert "1/1 completed" in out.content


async def test_a_different_directory_gets_its_own_list(make_ctx, tmp_path: Path) -> None:
    other = tmp_path / "elsewhere"
    other.mkdir()
    await TodoWriteTool().call(
        TodoInput(todos=[TodoItem(content="ship it")]), make_ctx()
    )
    assert TodoWriteTool().current(make_ctx(cwd=other)) == []


async def test_session_scope_does_not_leak_between_sessions(make_ctx) -> None:
    first = make_ctx()
    first.options.todo_scope = "session"
    await TodoWriteTool().call(TodoInput(todos=[TodoItem(content="ship it")]), first)

    second = make_ctx()
    second.options.todo_scope = "session"
    second.session_id = "another-session"
    assert TodoWriteTool().current(second) == []


async def test_persistence_can_be_disabled(make_ctx) -> None:
    ctx = make_ctx()
    ctx.options.persist_todos = False
    await TodoWriteTool().call(TodoInput(todos=[TodoItem(content="ship it")]), ctx)
    assert store_for(ctx.options) is None
    assert TodoWriteTool().current(make_ctx()) == []


async def test_a_subagent_does_not_overwrite_its_parents_list(make_ctx) -> None:
    """A delegated side task shares the cwd but not the plan."""
    parent = make_ctx()
    await TodoWriteTool().call(TodoInput(todos=[TodoItem(content="parent plan")]), parent)

    child = make_ctx(agent_id="sub-1")
    out = await TodoWriteTool().call(
        TodoInput(todos=[TodoItem(content="child task")]), child
    )
    assert "child task" in out.content
    assert [t.content for t in TodoWriteTool().current(make_ctx())] == ["parent plan"]


async def test_a_subagent_starts_from_an_empty_list(make_ctx) -> None:
    """Keyed by agent id, so the parent's plan is not even visible."""
    await TodoWriteTool().call(
        TodoInput(todos=[TodoItem(content="parent plan")]), make_ctx()
    )
    assert TodoWriteTool().current(make_ctx(agent_id="sub-1")) == []


async def test_a_subagent_keeps_its_own_list_across_calls(make_ctx) -> None:
    child = make_ctx(agent_id="sub-1")
    tool = TodoWriteTool()
    await tool.call(TodoInput(todos=[TodoItem(content="child task")]), child)
    out = await tool.call(TodoInput(add=[TodoItem(content="another")]), child)
    assert "0/2 completed" in out.content


async def test_a_subagent_list_is_stored_under_its_agent_id(make_ctx) -> None:
    """The key is what separates the two lists, so it has to reach the path."""
    child = make_ctx(agent_id="sub-1")
    await TodoWriteTool().call(TodoInput(todos=[TodoItem(content="child task")]), child)

    store, key = storage(child)
    assert key == "sub-1"
    assert [t.content for t in store.load(key, child.cwd)] == ["child task"]

    parent = make_ctx()
    assert store.dir_for(storage(parent)[1], parent.cwd) != store.dir_for(key, child.cwd)


def delegating_model() -> FunctionModel:
    """A parent that plans, delegates, and stops; the child reports its list."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        names = {t.name for t in info.function_tools}
        calls = [
            part
            for message in messages
            for part in getattr(message, "parts", [])
            if isinstance(part, ToolCallPart)
        ]
        if "Agent" in names:
            if not calls:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="TodoWrite",
                            args={"todos": [{"content": "parent plan"}]},
                        )
                    ]
                )
            if len(calls) == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="Agent",
                            args={
                                "description": "side task",
                                "prompt": "do the side task",
                                "subagent_type": "helper",
                            },
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart(content="parent done")])

        if not calls:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="TodoWrite",
                        args={"add": [{"content": "child task"}]},
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content=str(messages[-1].parts[-1].content))])

    return FunctionModel(respond)


async def test_a_subagent_does_not_inherit_the_parents_list_in_memory(
    tmp_path: Path,
) -> None:
    """The parent's list must not ride along in the shared tool context."""
    messages = [
        m
        async for m in summon(
            "delegate it",
            todo_options(
                tmp_path,
                delegating_model(),
                allowed_tools=["TodoWrite", "Agent"],
                agents={"helper": AgentDefinition(description="Helps", prompt="help")},
            ),
        )
    ]

    reports = [m for m in messages if m.type == "tool_result" and m.tool_name == "Agent"]
    assert len(reports) == 1
    assert "child task" in reports[0].output.content
    assert "parent plan" not in reports[0].output.content


async def test_a_subagent_leaves_no_list_behind(tmp_path: Path) -> None:
    """One dead list per delegated task would accumulate without limit."""
    async for _ in summon(
        "delegate it",
        todo_options(
            tmp_path,
            delegating_model(),
            allowed_tools=["TodoWrite", "Agent"],
            agents={"helper": AgentDefinition(description="Helps", prompt="help")},
        ),
    ):
        pass

    store = TodoStore()
    project = store.dir_for(key_for(Options(), "unused", tmp_path), tmp_path)
    lists = [d for d in project.parent.iterdir() if d.is_dir()]
    assert lists == [project]
    assert [t.content for t in store.load(project.name, tmp_path)] == ["parent plan"]


def twice_delegating_model() -> FunctionModel:
    """A parent that delegates two side tasks of the same type."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        names = {t.name for t in info.function_tools}
        calls = [
            part
            for message in messages
            for part in getattr(message, "parts", [])
            if isinstance(part, ToolCallPart)
        ]
        if "Agent" not in names:
            return ModelResponse(parts=[TextPart(content="child done")])
        if len(calls) < 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="Agent",
                        args={
                            "description": "side task",
                            "prompt": f"task {len(calls)}",
                            "subagent_type": "helper",
                        },
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="parent done")])

    return FunctionModel(respond)


async def test_two_subagents_of_one_type_get_separate_lists(tmp_path: Path) -> None:
    """Subagents may run in parallel, so their ids cannot collide."""
    seen: list[str | None] = []

    async def record(payload: Any) -> None:
        seen.append(payload.agent_id)
        return None

    async for _ in summon(
        "delegate twice",
        todo_options(
            tmp_path,
            twice_delegating_model(),
            allowed_tools=["TodoWrite", "Agent"],
            agents={"helper": AgentDefinition(description="Helps", prompt="help")},
            hooks=[HookMatcher("SubagentStart", [record])],
        ),
    ):
        pass

    assert len(seen) == 2
    assert seen[0] != seen[1]


def test_a_corrupt_task_file_does_not_lose_the_rest(tmp_path: Path) -> None:
    store = TodoStore(tmp_path / "root")
    store.write("k", tmp_path, TodoItem(content="intact", id="good", position=1))
    (store.dir_for("k", tmp_path) / "bad.json").write_text("{not json")
    assert [t.content for t in store.load("k", tmp_path)] == ["intact"]


def test_a_finished_list_is_not_carried_over() -> None:
    """Completed work from an unrelated run is noise, not context."""
    assert carried_over_context(items(("done", "completed"))) is None
    assert carried_over_context([]) is None
    assert "unfinished" in (carried_over_context(items(("todo", "pending"))) or "")


def planning_model() -> FunctionModel:
    """A model that adds one task, then finishes."""
    calls: list[int] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        if len(calls) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="TodoWrite",
                        args={"add": [{"content": "write the parser"}]},
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="done")])

    return FunctionModel(respond)


def echoing_model() -> FunctionModel:
    """A model that reports back the prompt it was given."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        text = str(messages[-1].parts[-1].content)
        return ModelResponse(parts=[TextPart(content=text)])

    return FunctionModel(respond)


def todo_options(tmp_path: Path, model: Any, **kwargs: Any) -> Options:
    kwargs.setdefault("allowed_tools", ["TodoWrite"])
    return Options(
        model=model,
        cwd=tmp_path,
        permission_mode="bypassPermissions",
        auto_compact=False,
        **kwargs,
    )


async def test_a_run_leaves_its_todos_behind(tmp_path: Path) -> None:
    async for _ in summon("plan it", todo_options(tmp_path, planning_model())):
        pass

    store = TodoStore()
    stored = store.load(key_for(Options(), "unused", tmp_path), tmp_path)
    assert [t.content for t in stored] == ["write the parser"]


async def test_the_next_run_is_told_about_the_carried_over_list(
    tmp_path: Path,
) -> None:
    async for _ in summon("plan it", todo_options(tmp_path, planning_model())):
        pass

    messages = [
        m async for m in summon("carry on", todo_options(tmp_path, echoing_model()))
    ]
    final = [m for m in messages if m.type == "result"][0]
    assert "write the parser" in final.result
    assert "carry on" in final.result


async def test_a_run_with_persistence_off_is_told_nothing(tmp_path: Path) -> None:
    async for _ in summon("plan it", todo_options(tmp_path, planning_model())):
        pass

    messages = [
        m
        async for m in summon(
            "carry on",
            todo_options(tmp_path, echoing_model(), persist_todos=False),
        )
    ]
    final = [m for m in messages if m.type == "result"][0]
    assert "write the parser" not in final.result
