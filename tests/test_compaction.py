"""Tests for conversation compaction and provider-neutral model resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from ubiquity import HookMatcher, HookOutput, Options, summon
from ubiquity.compaction import (
    CONTEXT_WINDOW_ENV_VAR,
    CONTEXT_WINDOWS,
    DEFAULT_CONTEXT_WINDOW,
    HEADROOM_TOKENS,
    MAX_CONSECUTIVE_FAILURES,
    OUTPUT_RESERVE_TOKENS,
    compact,
    compaction_boundary,
    compaction_threshold,
    estimate_tokens,
    infer_context_window,
    measure_context,
    register_context_window,
    render_transcript,
    should_compact,
    usage_tokens,
)
from ubiquity.models import (
    MODEL_ENV_VAR,
    clear_aliases,
    default_model_spec,
    expand_alias,
    model_name_of,
    register_alias,
    registered_aliases,
)


@pytest.fixture(autouse=True)
def clean_registries():
    """Keep the process-wide registries from leaking between tests."""
    clear_aliases()
    CONTEXT_WINDOWS.clear()
    yield
    clear_aliases()
    CONTEXT_WINDOWS.clear()


def user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def assistant(text: str, usage: RequestUsage | None = None) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)], usage=usage or RequestUsage())


def tool_call(name: str, call_id: str) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name=name, args={}, tool_call_id=call_id)])


def tool_return(name: str, call_id: str, content: str = "ok") -> ModelRequest:
    return ModelRequest(
        parts=[ToolReturnPart(tool_name=name, content=content, tool_call_id=call_id)]
    )


def conversation(turns: int) -> list[ModelMessage]:
    """Build a history of `turns` tool-calling exchanges."""
    messages: list[ModelMessage] = [user("start")]
    for i in range(turns):
        messages.append(tool_call("ping", f"c{i}"))
        messages.append(tool_return("ping", f"c{i}"))
    messages.append(assistant("finished"))
    return messages


def test_no_model_windows_are_shipped() -> None:
    """Asserting window sizes for models we cannot verify would go stale."""
    assert CONTEXT_WINDOWS == {}
    assert infer_context_window("openai:gpt-5") == DEFAULT_CONTEXT_WINDOW


def test_registered_windows_match_by_substring() -> None:
    register_context_window("gemini-3", 1_048_576)
    assert infer_context_window("google:gemini-3-pro") == 1_048_576


def test_registered_windows_are_case_insensitive() -> None:
    register_context_window("gemini-3", 1_048_576)
    assert infer_context_window("Google:Gemini-3-Pro") == 1_048_576


def test_the_most_specific_registration_wins() -> None:
    register_context_window("llama", 8_192)
    register_context_window("llama-3.3", 128_000)
    assert infer_context_window("groq:llama-3.3-70b-versatile") == 128_000
    assert infer_context_window("ollama:llama-2") == 8_192


def test_unregistered_models_fall_back(monkeypatch) -> None:
    monkeypatch.delenv(CONTEXT_WINDOW_ENV_VAR, raising=False)
    assert infer_context_window("someones-private-finetune") == DEFAULT_CONTEXT_WINDOW


def test_environment_can_lower_the_default(monkeypatch) -> None:
    monkeypatch.setenv(CONTEXT_WINDOW_ENV_VAR, "8000")
    assert infer_context_window("anything") == 8_000


def test_garbage_in_the_environment_is_ignored(monkeypatch) -> None:
    monkeypatch.setenv(CONTEXT_WINDOW_ENV_VAR, "lots")
    assert infer_context_window("anything") == DEFAULT_CONTEXT_WINDOW


def test_threshold_reserves_a_fixed_budget_not_a_percentage() -> None:
    """A percentage would waste 200k tokens on a million-token window."""
    assert compaction_threshold(1_000_000) == 1_000_000 - OUTPUT_RESERVE_TOKENS - HEADROOM_TOKENS
    assert compaction_threshold(1_000_000) > 1_000_000 * 0.95


def test_threshold_never_collapses_on_a_small_window() -> None:
    """A local 8k model would otherwise get a negative threshold."""
    assert compaction_threshold(8_000) == 4_000


def test_threshold_fraction_overrides_the_reserve() -> None:
    assert compaction_threshold(100_000, fraction=0.5) == 50_000


def test_usage_tokens_counts_cached_input() -> None:
    """Providers that bill cache hits separately would otherwise look tiny."""
    usage = RequestUsage(input_tokens=100, cache_read_tokens=9_000, output_tokens=50)
    assert usage_tokens(usage) == 9_150


def test_usage_tokens_of_nothing_is_zero() -> None:
    assert usage_tokens(None) == 0


def test_measure_context_prefers_provider_accounting() -> None:
    history = [user("hi"), assistant("yo", RequestUsage(input_tokens=5_000))]
    assert measure_context(history) == 5_000


def test_measure_context_falls_back_when_provider_reports_nothing() -> None:
    """Self-hosted OpenAI-compatible servers frequently report no usage."""
    history = [user("x" * 4_000), assistant("y" * 4_000)]
    assert measure_context(history) == pytest.approx(estimate_tokens(history))
    assert measure_context(history) > 0


def test_measure_context_skips_responses_with_no_usage() -> None:
    """A retry or synthetic response must not mask real numbers behind it."""
    history = [
        user("hi"),
        assistant("real", RequestUsage(input_tokens=5_000)),
        assistant("no usage at all"),
    ]
    assert measure_context(history) == 5_000 + estimate_tokens(history[2:])


def test_measure_context_counts_messages_after_the_last_usage() -> None:
    """The tool results that just landed are what push a run over the limit.

    Compaction is checked between a response and the request answering it, so
    a large tool result always sits past the last usage record. Stopping at
    that record would make the measurement blind to it.
    """
    landed = tool_return("Read", "c0", "x" * 40_000)
    history = [
        user("read the file"),
        assistant("reading", RequestUsage(input_tokens=5_000)),
        landed,
    ]
    assert measure_context(history) > 9_000
    assert measure_context(history) == 5_000 + estimate_tokens([landed])


def test_measure_context_ignores_a_trailing_response_that_reports_usage() -> None:
    """Later accounting supersedes earlier accounting rather than adding to it."""
    history = [
        user("hi"),
        assistant("first", RequestUsage(input_tokens=5_000)),
        user("more"),
        assistant("second", RequestUsage(input_tokens=9_000)),
    ]
    assert measure_context(history) == 9_000


def test_output_reserve_lowers_the_threshold_cap() -> None:
    """Reserving 20k for a model that can emit 4k wastes 16k every turn."""
    capped = compaction_threshold(200_000, output_reserve=4_000)
    assert capped == 200_000 - 4_000 - HEADROOM_TOKENS
    assert capped > compaction_threshold(200_000)


def test_output_reserve_never_raises_the_reserve() -> None:
    """A generous max_tokens must not shrink the summary's own budget."""
    assert compaction_threshold(200_000, output_reserve=64_000) == compaction_threshold(200_000)
    assert compaction_threshold(200_000, output_reserve=0) == compaction_threshold(200_000)


def test_should_compact_respects_the_threshold() -> None:
    history = [user("hi"), assistant("yo", RequestUsage(input_tokens=800))]
    assert should_compact(history, context_window=1_000, fraction=0.5)[0] is True
    assert should_compact(history, context_window=1_000, fraction=0.9)[0] is False


def test_boundary_always_lands_on_a_response() -> None:
    """Cutting before a request would orphan the tool returns it carries."""
    history = conversation(6)
    boundary = compaction_boundary(history, keep_recent=6)
    assert boundary is not None
    assert isinstance(history[boundary], ModelResponse)


def test_boundary_keeps_every_tool_return_paired() -> None:
    history = conversation(6)
    kept = history[compaction_boundary(history, keep_recent=6) :]

    called = {
        p.tool_call_id
        for m in kept
        if isinstance(m, ModelResponse)
        for p in m.parts
        if isinstance(p, ToolCallPart)
    }
    returned = {
        p.tool_call_id
        for m in kept
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, ToolReturnPart)
    }
    assert returned <= called


def test_boundary_retains_the_final_response() -> None:
    """The in-flight request answers the last response's calls, so it must stay."""
    history = conversation(6)
    boundary = compaction_boundary(history, keep_recent=6)
    assert boundary is not None
    assert history[-1] in history[boundary:]


def test_boundary_declines_a_short_history() -> None:
    assert compaction_boundary([user("hi"), assistant("yo")], keep_recent=6) is None


def test_boundary_declines_when_nothing_would_be_dropped() -> None:
    assert compaction_boundary(conversation(1), keep_recent=100) is None


def test_render_transcript_truncates_the_middle_not_the_tail() -> None:
    """A command's verdict is usually in the last lines of its output."""
    history = [user("HEAD" + "x" * 10_000 + "TAIL")]
    rendered = render_transcript(history)
    assert "HEAD" in rendered
    assert "TAIL" in rendered
    assert "characters omitted" in rendered
    assert len(rendered) < 3_000


def test_render_transcript_labels_roles() -> None:
    rendered = render_transcript([user("question"), assistant("answer")])
    assert "[user]" in rendered
    assert "[assistant]" in rendered


async def test_compact_declines_a_history_it_cannot_split() -> None:
    summarizer = FunctionModel(lambda m, i: ModelResponse(parts=[TextPart(content="s")]))
    assert await compact([user("hi"), assistant("yo")], summarizer) is None


async def test_compact_replaces_the_prefix_with_a_summary() -> None:
    summarizer = FunctionModel(
        lambda m, i: ModelResponse(parts=[TextPart(content="THE SUMMARY")])
    )
    history = conversation(6)
    result = await compact(history, summarizer, keep_recent=4)

    assert result is not None
    assert result.summary == "THE SUMMARY"
    assert len(result.messages) < len(history)
    assert isinstance(result.messages[0], ModelRequest)
    assert "THE SUMMARY" in str(result.messages[0].parts[0].content)
    assert isinstance(result.messages[1], ModelResponse)


async def test_compact_leaves_the_input_history_untouched() -> None:
    """`compact` returns a new list; the caller decides when to swap it in."""
    summarizer = FunctionModel(lambda m, i: ModelResponse(parts=[TextPart(content="s")]))
    history = conversation(6)
    before = list(history)
    await compact(history, summarizer, keep_recent=4)
    assert history == before


async def test_compact_sees_the_dropped_messages_only() -> None:
    seen: dict[str, str] = {}

    def summarize_it(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen["prompt"] = str(messages[-1].parts[-1].content)
        return ModelResponse(parts=[TextPart(content="s")])

    history = [user("THE ORIGINAL ASK"), *conversation(6)[1:]]
    await compact(history, FunctionModel(summarize_it), keep_recent=2)
    assert "THE ORIGINAL ASK" in seen["prompt"]


def looping_model(stop_after: int = 6) -> tuple[FunctionModel, list[int]]:
    """A model that calls a tool `stop_after` times, reporting heavy usage.

    Returns the model and the list it records each request's history length
    into, so a test can prove compaction actually shrank the context rather
    than merely emitting a boundary message. The reported usage sits above
    every threshold the tests configure, so compaction is always due.
    """
    lengths: list[int] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        lengths.append(len(messages))
        usage = RequestUsage(input_tokens=900)
        if len(lengths) < stop_after:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="Glob", args={"pattern": "*"})], usage=usage
            )
        return ModelResponse(parts=[TextPart(content="done")], usage=usage)

    return FunctionModel(respond), lengths


def compacting_options(tmp_path: Path, model: Any, **kwargs: Any) -> Options:
    """Options tuned so a run of more than two turns always compacts."""
    kwargs.setdefault(
        "compact_model",
        FunctionModel(lambda m, i: ModelResponse(parts=[TextPart(content="COMPACTED")])),
    )
    return Options(
        model=model,
        cwd=tmp_path,
        allowed_tools=["Glob"],
        permission_mode="bypassPermissions",
        max_context_tokens=1_000,
        compact_threshold=0.5,
        compact_keep_recent=2,
        **kwargs,
    )


async def test_auto_compaction_emits_a_boundary_and_shrinks_the_history(
    tmp_path: Path,
) -> None:
    """The proof of compaction is the history getting smaller, not the message."""
    model, lengths = looping_model()
    messages = [m async for m in summon("go", compacting_options(tmp_path, model))]

    boundaries = [
        m for m in messages if m.type == "system" and m.subtype == "compact_boundary"
    ]
    assert boundaries
    assert boundaries[0].pre_tokens == 900
    assert boundaries[0].trigger == "auto"

    assert len(lengths) >= 4
    assert max(lengths[2:]) <= 4
    assert messages[-1].subtype == "success"


async def test_the_summary_reaches_the_next_request(tmp_path: Path) -> None:
    """Compaction is worthless if the replacement text is never actually sent."""
    prompts: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompts.extend(
            str(p.content)
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, UserPromptPart)
        )
        usage = RequestUsage(input_tokens=900)
        if sum(1 for m in messages if isinstance(m, ModelResponse)) >= 4:
            return ModelResponse(parts=[TextPart(content="done")], usage=usage)
        return ModelResponse(
            parts=[ToolCallPart(tool_name="Glob", args={"pattern": "*"})], usage=usage
        )

    [
        m
        async for m in summon(
            "go", compacting_options(tmp_path, FunctionModel(respond))
        )
    ]
    assert any("COMPACTED" in p for p in prompts)


async def test_compaction_preserves_tool_call_pairing(tmp_path: Path) -> None:
    """An orphaned tool result is rejected outright by most providers."""
    orphans: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        called = {
            p.tool_call_id
            for m in messages
            if isinstance(m, ModelResponse)
            for p in m.parts
            if isinstance(p, ToolCallPart)
        }
        orphans.extend(
            p.tool_call_id
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, ToolReturnPart) and p.tool_call_id not in called
        )
        usage = RequestUsage(input_tokens=900)
        if len(orphans) or sum(1 for m in messages if isinstance(m, ModelResponse)) >= 5:
            return ModelResponse(parts=[TextPart(content="done")], usage=usage)
        return ModelResponse(
            parts=[ToolCallPart(tool_name="Glob", args={"pattern": "*"})], usage=usage
        )

    messages = [
        m
        async for m in summon(
            "go", compacting_options(tmp_path, FunctionModel(respond))
        )
    ]
    assert [m for m in messages if getattr(m, "subtype", None) == "compact_boundary"]
    assert orphans == []


async def test_auto_compact_can_be_disabled(tmp_path: Path) -> None:
    model, _ = looping_model()
    options = compacting_options(tmp_path, model, auto_compact=False)
    messages = [m async for m in summon("go", options)]
    assert not [m for m in messages if getattr(m, "subtype", None) == "compact_boundary"]


async def test_precompact_hook_can_veto(tmp_path: Path) -> None:
    fired: list[str] = []

    async def veto(payload):
        fired.append(payload.hook_event_name)
        return HookOutput(decision="block", reason="not now")

    model, _ = looping_model()
    options = compacting_options(
        tmp_path, model, hooks=[HookMatcher("PreCompact", [veto])]
    )
    messages = [m async for m in summon("go", options)]

    assert fired == ["PreCompact"] * len(fired) and fired
    assert not [m for m in messages if getattr(m, "subtype", None) == "compact_boundary"]


async def test_postcompact_hook_receives_the_summary(tmp_path: Path) -> None:
    seen: list[Any] = []

    async def observe(payload):
        seen.append(payload)
        return None

    model, _ = looping_model()
    options = compacting_options(
        tmp_path, model, hooks=[HookMatcher("PostCompact", [observe])]
    )
    [m async for m in summon("go", options)]

    assert seen
    assert seen[0].message == "COMPACTED"
    assert seen[0].extra["trigger"] == "auto"


async def test_failed_compaction_does_not_kill_the_run(tmp_path: Path, caplog) -> None:
    """Losing the run outright is worse than one oversized request."""

    def explode(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("summarizer is down")

    model, _ = looping_model()
    options = compacting_options(tmp_path, model, compact_model=FunctionModel(explode))

    with caplog.at_level("ERROR", logger="ubiquity"):
        messages = [m async for m in summon("go", options)]

    assert "compaction failed" in caplog.text
    assert messages[-1].type == "result"
    assert messages[-1].subtype == "success"


async def test_repeated_failures_trip_the_circuit_breaker(tmp_path: Path, caplog) -> None:
    """A doomed context must not retry compaction on every remaining turn."""
    attempts = {"n": 0}

    def explode(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        attempts["n"] += 1
        raise RuntimeError("summarizer is down")

    model, _ = looping_model(stop_after=20)
    options = compacting_options(tmp_path, model, compact_model=FunctionModel(explode))

    with caplog.at_level("ERROR", logger="ubiquity"):
        [m async for m in summon("go", options)]

    assert attempts["n"] == MAX_CONSECUTIVE_FAILURES


def test_no_model_is_configured_by_default(monkeypatch) -> None:
    """A default model would be an arbitrary vendor endorsement."""
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    assert Options().model is None
    with pytest.raises(ValueError, match="No model configured"):
        Options().resolved_model()


def test_model_falls_back_to_the_environment(monkeypatch) -> None:
    monkeypatch.setenv(MODEL_ENV_VAR, "openai:gpt-5")
    assert default_model_spec() == "openai:gpt-5"
    assert Options().resolved_model() == "openai:gpt-5"


def test_explicit_model_beats_the_environment(monkeypatch) -> None:
    monkeypatch.setenv(MODEL_ENV_VAR, "openai:gpt-5")
    assert Options(model="groq:llama-3.3-70b-versatile").resolved_model() == (
        "groq:llama-3.3-70b-versatile"
    )


def test_aliases_are_registrable_for_any_provider() -> None:
    register_alias("fast", "groq:llama-3.3-70b-versatile")
    register_alias("local", "ollama:qwen3")
    assert expand_alias("fast") == "groq:llama-3.3-70b-versatile"
    assert expand_alias("local") == "ollama:qwen3"
    assert set(registered_aliases()) == {"fast", "local"}


def test_unknown_names_pass_through_unchanged() -> None:
    assert expand_alias("mistral:mistral-large-latest") == "mistral:mistral-large-latest"


def test_run_scoped_aliases_win_over_registered_ones() -> None:
    register_alias("main", "openai:gpt-5")
    assert expand_alias("main", {"main": "google:gemini-3-pro"}) == "google:gemini-3-pro"


def test_aliases_resolve_transitively() -> None:
    register_alias("a", "b")
    register_alias("b", "cohere:command-a")
    assert expand_alias("a") == "cohere:command-a"


def test_alias_cycles_terminate() -> None:
    """A misconfigured registry must not hang the process."""
    register_alias("a", "b")
    register_alias("b", "a")
    assert expand_alias("a") in {"a", "b"}


def test_model_name_of_expands_aliases() -> None:
    register_alias("fast", "cerebras:llama-3.3-70b")
    assert model_name_of("fast") == "cerebras:llama-3.3-70b"


def test_compact_model_defaults_to_the_run_model(monkeypatch) -> None:
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    options = Options(model="xai:grok-4")
    assert options.resolved_compact_model() == "xai:grok-4"


def test_explicit_context_window_overrides_the_heuristic() -> None:
    options = Options(model="openai:gpt-5", max_context_tokens=42)
    assert options.resolved_context_window("openai:gpt-5") == 42
