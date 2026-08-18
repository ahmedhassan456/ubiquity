"""The `Options` fields that used to be declared but read by nothing.

Each of these was a promise the SDK did not keep: a caller could set it, see
no error, and get a run configured as though they had not. The tests here pin
the wiring at the point it would silently come loose again — the settings
handed to the model, the history handed to the run, the messages the loop
emits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.function import AgentInfo, DeltaToolCalls, FunctionModel

from ubiquity import (
    Options,
    SessionStore,
    apply_settings,
    history_from,
    load_settings,
    summon,
    settings_path,
    with_fallback,
)
from ubiquity.sessions.store import SessionRecord
from ubiquity.types import (
    SDKAssistantMessage,
    SDKToolResultMessage,
    SDKUserMessage,
    ToolOutput,
)


def test_no_settings_means_no_model_settings() -> None:
    """A run that configures nothing must send nothing.

    Prompt caching is the one default that is on, so switching it off is what
    "nothing configured" now means.
    """
    assert Options(model="test", cache_prompt=False).model_settings() is None


def test_caching_is_the_only_setting_a_bare_run_sends() -> None:
    assert Options(model="test").model_settings() == {"anthropic_cache": True}


def test_temperature_and_max_tokens_reach_model_settings() -> None:
    settings = Options(
        model="test", cache_prompt=False, temperature=0.2, max_tokens=512
    ).model_settings()
    assert settings == {"temperature": 0.2, "max_tokens": 512}


def test_thinking_budget_is_named_for_each_provider_that_takes_one() -> None:
    """The portable setting is an effort level, so a budget needs both keys."""
    settings = Options(model="test", max_thinking_tokens=2048).model_settings()
    assert settings is not None
    assert settings["thinking"] is True
    assert settings["anthropic_thinking"]["budget_tokens"] == 2048
    assert settings["google_thinking_config"]["thinking_budget"] == 2048


def test_user_is_carried_as_an_end_user_identifier() -> None:
    settings = Options(model="test", user="tenant-7").model_settings()
    assert settings is not None
    assert settings["openai_user"] == "tenant-7"
    assert settings["anthropic_metadata"] == {"user_id": "tenant-7"}


def test_fallback_model_wraps_the_primary() -> None:
    primary, fallback = _echo_model("primary"), _echo_model("fallback")
    model = with_fallback(primary, fallback)
    assert list(model.models) == [primary, fallback]


async def test_a_failing_primary_fails_over(tmp_path: Path) -> None:
    """Failover is per request, so the run continues rather than restarting."""

    def explode(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=503, model_name="primary", body=None)

    messages = [
        m
        async for m in summon(
            "hello",
            Options(
                model=FunctionModel(explode),
                fallback_model=_echo_model("answered by the fallback"),
                cwd=tmp_path,
                tools=[],
                persist_session=False,
                persist_todos=False,
            ),
        )
    ]
    assert messages[-1].subtype == "success"
    assert messages[-1].result == "answered by the fallback"


def _echo_model(reply: str = "done") -> FunctionModel:
    """A model that answers with `reply` and calls nothing."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=reply)])

    return FunctionModel(respond)


def _streaming_model(*chunks: str) -> FunctionModel:
    """A model that emits `chunks` as separate deltas."""

    async def stream(messages: list[ModelMessage], info: AgentInfo):
        for chunk in chunks:
            yield chunk

    return FunctionModel(stream_function=stream)


async def test_include_partial_messages_emits_deltas(tmp_path: Path) -> None:
    messages = [
        m
        async for m in summon(
            "hello",
            Options(
                model=_streaming_model("streamed ", "answer"),
                cwd=tmp_path,
                tools=[],
                include_partial_messages=True,
                persist_session=False,
                persist_todos=False,
            ),
        )
    ]
    partials = [m for m in messages if m.type == "stream_event"]
    assert partials
    assert "".join(p.delta for p in partials) == "streamed answer"
    assert [m.type for m in messages].index("stream_event") < [
        m.type for m in messages
    ].index("assistant")


async def test_partial_messages_are_off_by_default(tmp_path: Path) -> None:
    messages = [
        m
        async for m in summon(
            "hello",
            Options(
                model=_echo_model(),
                cwd=tmp_path,
                tools=[],
                persist_session=False,
                persist_todos=False,
            ),
        )
    ]
    assert not [m for m in messages if m.type == "stream_event"]


def _seed_session(root: Path, cwd: Path) -> str:
    """Write a two-turn session with one tool call, and return its id."""
    store = SessionStore(root)
    session_id = "seed-session"
    store.append(session_id, cwd, SDKUserMessage(content="what is in a.txt?"))
    store.append(
        session_id,
        cwd,
        SDKAssistantMessage(
            content=[
                {"type": "text", "text": "reading it"},
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "Read",
                    "input": {"file_path": "a.txt"},
                },
            ]
        ),
    )
    store.append(
        session_id,
        cwd,
        SDKToolResultMessage(
            tool_name="Read",
            tool_use_id="call-1",
            output=ToolOutput(content="the codeword is ZEPHYR42"),
        ),
    )
    store.append(
        session_id,
        cwd,
        SDKAssistantMessage(content=[{"type": "text", "text": "it says ZEPHYR42"}]),
    )
    return session_id


def test_history_from_rebuilds_the_conversation(tmp_path: Path) -> None:
    session_id = _seed_session(tmp_path / "sessions", tmp_path)
    records = SessionStore(tmp_path / "sessions").read(session_id, tmp_path)
    history = history_from(records)

    kinds = [type(m).__name__ for m in history]
    assert kinds == ["ModelRequest", "ModelResponse", "ModelRequest", "ModelResponse"]
    assert isinstance(history[0].parts[0], UserPromptPart)
    assert any(isinstance(p, ToolCallPart) for p in history[1].parts)
    assert isinstance(history[2].parts[0], ToolReturnPart)


def test_a_tool_call_without_a_result_is_dropped() -> None:
    """A dangling tool use makes a provider reject the whole conversation."""
    records = [
        SessionRecord(uuid="1", type="user", payload={"content": "go"}),
        SessionRecord(
            uuid="2",
            type="assistant",
            payload={
                "content": [
                    {"type": "text", "text": "trying"},
                    {"type": "tool_use", "id": "denied-1", "name": "Bash", "input": {}},
                ]
            },
        ),
    ]
    history = history_from(records)
    assert not any(
        isinstance(part, ToolCallPart) for m in history for part in m.parts
    )
    assert any(isinstance(part, TextPart) for m in history for part in m.parts)


def test_a_result_without_its_call_is_dropped() -> None:
    records = [
        SessionRecord(
            uuid="1",
            type="tool_result",
            payload={
                "tool_name": "Bash",
                "tool_use_id": "orphan",
                "output": {"content": "hi"},
            },
        )
    ]
    assert history_from(records) == []


def test_system_and_result_records_are_not_replayed(tmp_path: Path) -> None:
    records = [
        SessionRecord(uuid="1", type="system", payload={"model": "x"}),
        SessionRecord(uuid="2", type="user", payload={"content": "go"}),
        SessionRecord(uuid="3", type="result", payload={"result": "done"}),
    ]
    history = history_from(records)
    assert len(history) == 1
    assert isinstance(history[0], ModelRequest)


def _seen_history() -> tuple[list[list[ModelMessage]], FunctionModel]:
    """A model that records the history it was called with."""
    seen: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        return ModelResponse(parts=[TextPart(content="ok")])

    return seen, FunctionModel(respond)


async def test_resume_replays_the_stored_conversation(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    session_id = _seed_session(sessions, tmp_path)
    seen, model = _seen_history()

    messages = [
        m
        async for m in summon(
            "and what was the codeword?",
            Options(
                model=model,
                cwd=tmp_path,
                tools=[],
                resume=session_id,
                session_dir=sessions,
                persist_todos=False,
            ),
        )
    ]
    replayed = "\n".join(str(m) for m in seen[0])
    assert "ZEPHYR42" in replayed
    assert messages[0].session_id == session_id


async def test_resume_appends_to_the_same_transcript(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    session_id = _seed_session(sessions, tmp_path)
    before = len(SessionStore(sessions).read(session_id, tmp_path))
    _, model = _seen_history()

    async for _ in summon(
        "more",
        Options(
            model=model,
            cwd=tmp_path,
            tools=[],
            resume=session_id,
            session_dir=sessions,
            persist_todos=False,
        ),
    ):
        pass
    assert len(SessionStore(sessions).read(session_id, tmp_path)) > before


async def test_fork_session_leaves_the_original_untouched(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    session_id = _seed_session(sessions, tmp_path)
    store = SessionStore(sessions)
    before = len(store.read(session_id, tmp_path))
    _, model = _seen_history()

    messages = [
        m
        async for m in summon(
            "branch from here",
            Options(
                model=model,
                cwd=tmp_path,
                tools=[],
                resume=session_id,
                fork_session=True,
                session_dir=sessions,
                persist_todos=False,
            ),
        )
    ]
    forked = messages[0].session_id
    assert forked != session_id
    assert len(store.read(session_id, tmp_path)) == before
    assert len(store.read(forked, tmp_path)) > before


async def test_continue_conversation_picks_the_latest_session(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    _seed_session(sessions, tmp_path)
    seen, model = _seen_history()

    messages = [
        m
        async for m in summon(
            "keep going",
            Options(
                model=model,
                cwd=tmp_path,
                tools=[],
                continue_conversation=True,
                session_dir=sessions,
                persist_todos=False,
            ),
        )
    ]
    assert messages[0].session_id == "seed-session"
    assert "ZEPHYR42" in "\n".join(str(m) for m in seen[0])


async def test_continue_with_no_prior_session_starts_a_fresh_one(tmp_path: Path) -> None:
    _, model = _seen_history()
    messages = [
        m
        async for m in summon(
            "hello",
            Options(
                model=model,
                cwd=tmp_path,
                tools=[],
                continue_conversation=True,
                session_dir=tmp_path / "sessions",
                persist_todos=False,
            ),
        )
    ]
    assert messages[-1].subtype == "success"


async def test_resuming_a_missing_session_is_an_error(tmp_path: Path) -> None:
    """Starting over silently would lose the work the caller asked to continue."""
    messages = [
        m
        async for m in summon(
            "hello",
            Options(
                model=_echo_model(),
                cwd=tmp_path,
                tools=[],
                resume="does-not-exist",
                session_dir=tmp_path / "sessions",
                persist_todos=False,
            ),
        )
    ]
    assert messages[-1].is_error
    assert "does-not-exist" in str(messages[-1].result)


def _write_settings(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_no_sources_reads_nothing(tmp_path: Path) -> None:
    """The default must not let a stray file reconfigure a caller's run."""
    _write_settings(
        settings_path("project", tmp_path), {"permissions": {"deny": ["Bash"]}}
    )
    options = Options(model="test", cwd=tmp_path)
    assert apply_settings(options) is options


def test_project_settings_add_permission_rules(tmp_path: Path) -> None:
    _write_settings(
        settings_path("project", tmp_path),
        {"permissions": {"deny": ["Bash(rm:*)"], "ask": ["Bash(git push:*)"]}},
    )
    applied = apply_settings(
        Options(
            model="test",
            cwd=tmp_path,
            disallowed_tools=["Write"],
            setting_sources=["project"],
        )
    )
    assert list(applied.disallowed_tools) == ["Write", "Bash(rm:*)"]
    assert list(applied.ask_tools) == ["Bash(git push:*)"]


def test_explicit_options_win_over_a_settings_file(tmp_path: Path) -> None:
    _write_settings(
        settings_path("project", tmp_path),
        {"model": "from-file", "permissions": {"defaultMode": "bypassPermissions"}},
    )
    applied = apply_settings(
        Options(
            model="explicit",
            cwd=tmp_path,
            permission_mode="plan",
            setting_sources=["project"],
        )
    )
    assert applied.model == "explicit"
    assert applied.permission_mode == "plan"


def test_a_settings_file_fills_what_options_left_unset(tmp_path: Path) -> None:
    _write_settings(
        settings_path("project", tmp_path),
        {"model": "from-file", "env": {"TOKEN": "abc"}},
    )
    applied = apply_settings(
        Options(cwd=tmp_path, setting_sources=["project"])
    )
    assert applied.model == "from-file"
    assert applied.env == {"TOKEN": "abc"}


def test_local_settings_override_project_settings(tmp_path: Path) -> None:
    _write_settings(settings_path("project", tmp_path), {"model": "project-model"})
    _write_settings(settings_path("local", tmp_path), {"model": "local-model"})
    merged = load_settings(["project", "local"], tmp_path)
    assert merged["model"] == "local-model"


def test_permission_rules_union_across_sources(tmp_path: Path) -> None:
    """A rule in one file must not be dropped by another file's list."""
    _write_settings(
        settings_path("project", tmp_path), {"permissions": {"deny": ["Bash(rm:*)"]}}
    )
    _write_settings(
        settings_path("local", tmp_path), {"permissions": {"deny": ["Write"]}}
    )
    merged = load_settings(["project", "local"], tmp_path)
    assert merged["permissions"]["deny"] == ["Bash(rm:*)", "Write"]


def test_a_malformed_settings_file_is_skipped(tmp_path: Path) -> None:
    """Ambient config the caller may not know about must not break the run."""
    path = settings_path("project", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert load_settings(["project"], tmp_path) == {}


def test_every_options_field_is_read_somewhere() -> None:
    """The survey that failed this suite, run offline as a regression guard."""
    from audit_options import unread_option_fields

    unread = unread_option_fields()
    assert not unread, f"declared but never read: {unread}"


def test_the_audit_covers_the_classes_that_hid_dead_fields() -> None:
    """An audit that passes because it looks nowhere proves nothing."""
    from audit_options import audited_classes

    covered = {cls.__name__ for cls, _ in audited_classes()}
    assert covered >= {"Options", "AgentDefinition", "ToolContext"}


def test_the_audit_still_finds_a_field_nothing_reads(tmp_path) -> None:
    """The guard has to fail on a dead field, not merely pass on a live one."""
    from dataclasses import dataclass

    from audit_options import unread_fields

    module = tmp_path / "planted.py"
    module.write_text(
        "from dataclasses import dataclass\n\n\n"
        "@dataclass\nclass Planted:\n    live: int = 0\n    dead: int = 0\n",
        encoding="utf-8",
    )

    @dataclass
    class Planted:
        live: int = 0
        dead: int = 0

    assert unread_fields(Planted, module, tmp_path) == ["dead", "live"]

    (tmp_path / "reader.py").write_text("print(config.live)\n", encoding="utf-8")
    assert unread_fields(Planted, module, tmp_path) == ["dead"]


def test_a_field_read_elsewhere_in_its_own_module_counts(tmp_path) -> None:
    """Only the class body is excluded; the rest of its module is calling code."""
    from dataclasses import dataclass

    from audit_options import unread_fields

    module = tmp_path / "planted.py"
    module.write_text(
        "from dataclasses import dataclass\n\n\n"
        "@dataclass\nclass Planted:\n    live: int = 0\n\n\n"
        "def use(p: Planted) -> int:\n    return p.live\n",
        encoding="utf-8",
    )

    @dataclass
    class Planted:
        live: int = 0

    assert unread_fields(Planted, module, tmp_path) == []


def test_a_sibling_class_does_not_keep_a_field_alive(tmp_path) -> None:
    """One class's `self.x` must not vouch for another class's `x`."""
    from dataclasses import dataclass

    from audit_options import unread_fields

    module = tmp_path / "planted.py"
    module.write_text(
        "from dataclasses import dataclass\n\n\n"
        "@dataclass\nclass Planted:\n    dead: int = 0\n\n\n"
        "class Other:\n    def go(self):\n        return self.dead\n",
        encoding="utf-8",
    )

    @dataclass
    class Planted:
        dead: int = 0

    assert unread_fields(Planted, module, tmp_path) == ["dead"]


def test_every_caller_facing_field_is_read_somewhere() -> None:
    """`Options` was never the only class making the promise.

    `AgentDefinition.skills` and `ToolContext.abort` both went unread for as
    long as the audit looked at `Options` alone, which is the argument for
    naming the classes it covers rather than the one it started with.
    """
    from audit_options import audited_classes, unread_fields

    unread = {
        cls.__name__: found
        for cls, module in audited_classes()
        if (found := unread_fields(cls, module))
    }
    assert not unread, f"declared but never read: {unread}"
