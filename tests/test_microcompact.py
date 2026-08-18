"""Tests for microcompaction: reclaiming context without a model call."""

from __future__ import annotations

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
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from ubiquity import Options, summon
from ubiquity.compaction import should_compact
from ubiquity.microcompact import (
    CLEARED_CONTENT,
    COMPACTABLE_TOOLS,
    compactable_call_ids,
    microcompact,
)


def user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def call(name: str, call_id: str) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name=name, args={}, tool_call_id=call_id)])


def returned(name: str, call_id: str, content: str) -> ModelRequest:
    return ModelRequest(
        parts=[ToolReturnPart(tool_name=name, content=content, tool_call_id=call_id)]
    )


def transcript(turns: int, tool: str = "Read", size: int = 4_000) -> list[ModelMessage]:
    """Build a history of `turns` tool exchanges with bulky results."""
    messages: list[ModelMessage] = [user("start")]
    for i in range(turns):
        messages.append(call(tool, f"c{i}"))
        messages.append(returned(tool, f"c{i}", "x" * size))
    return messages


def contents(messages: list[ModelMessage]) -> list[Any]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


def test_nothing_to_reclaim_returns_none() -> None:
    """Early in a run there is no stale output, and that is not a failure."""
    assert microcompact(transcript(2), keep_recent=5) is None
    assert microcompact([user("hi")], keep_recent=5) is None


def test_recent_results_survive_and_older_ones_are_cleared() -> None:
    result = microcompact(transcript(6), keep_recent=2)
    assert result is not None
    assert result.cleared == 4
    assert result.kept == 2
    assert contents(result.messages) == [CLEARED_CONTENT] * 4 + ["x" * 4_000] * 2


def test_clearing_preserves_call_and_return_pairing() -> None:
    """A cleared result is still a result; orphaning one would break the request."""
    history = transcript(6)
    result = microcompact(history, keep_recent=2)
    assert result is not None

    def call_ids(messages: list[ModelMessage]) -> list[str]:
        return [
            part.tool_call_id
            for message in messages
            for part in message.parts
            if isinstance(part, (ToolCallPart, ToolReturnPart))
        ]

    assert len(result.messages) == len(history)
    assert call_ids(result.messages) == call_ids(history)


def test_the_input_history_is_never_mutated() -> None:
    """Callers opt into the saving by adopting the result, not by side effect."""
    history = transcript(6)
    before = contents(history)
    microcompact(history, keep_recent=2)
    assert contents(history) == before


def test_only_observational_tools_are_eligible() -> None:
    """Clearing a todo list or a subagent report rewrites what the model believes."""
    history: list[ModelMessage] = [user("start")]
    for i, tool in enumerate(["TodoWrite", "Agent", "mcp__github__list_issues"]):
        history.append(call(tool, f"stateful{i}"))
        history.append(returned(tool, f"stateful{i}", "y" * 4_000))
    history.extend(transcript(6)[1:])

    assert compactable_call_ids(history) == [f"c{i}" for i in range(6)]
    result = microcompact(history, keep_recent=1)
    assert result is not None
    assert contents(result.messages)[:3] == ["y" * 4_000] * 3
    assert result.cleared == 5


def test_an_ineligible_result_survives_a_colliding_call_id() -> None:
    """Eligibility is a property of the tool, not of whichever id it was given."""
    history: list[ModelMessage] = [
        user("start"),
        call("Read", "shared"),
        returned("TodoWrite", "shared", "the todo list"),
        *transcript(6)[1:],
    ]
    result = microcompact(history, keep_recent=1)
    assert result is not None
    assert contents(result.messages)[0] == "the todo list"


def test_every_builtin_observational_tool_is_eligible() -> None:
    for tool in ("Read", "Write", "Edit", "Bash", "Glob", "Grep"):
        assert tool in COMPACTABLE_TOOLS
    assert "TodoWrite" not in COMPACTABLE_TOOLS
    assert "Agent" not in COMPACTABLE_TOOLS


def test_the_custom_tool_set_is_respected() -> None:
    history = transcript(6, tool="Search")
    assert microcompact(history, keep_recent=1) is None
    assert microcompact(history, keep_recent=1, tools={"Search"}) is not None


def test_at_least_one_result_always_survives() -> None:
    """Clearing everything leaves the model with no observation to act on."""
    result = microcompact(transcript(6), keep_recent=0)
    assert result is not None
    assert result.kept == 1
    assert contents(result.messages)[-1] == "x" * 4_000


def test_a_second_pass_finds_nothing_left_to_clear() -> None:
    """Re-clearing would report savings that were already banked."""
    first = microcompact(transcript(6), keep_recent=2)
    assert first is not None
    assert microcompact(first.messages, keep_recent=2) is None


def test_savings_are_net_of_the_marker() -> None:
    result = microcompact(transcript(6, size=4_000), keep_recent=2)
    assert result is not None
    expected = (4 * 4_000 - 4 * len(CLEARED_CONTENT)) // 4
    assert result.tokens_saved == expected


def test_should_compact_discounts_what_was_already_reclaimed() -> None:
    """Provider usage reports what was sent, so the saving is invisible to it."""
    history = [user("hi"), ModelResponse(parts=[TextPart(content="yo")], usage=RequestUsage(input_tokens=800))]
    assert should_compact(history, context_window=1_000, fraction=0.5)[0] is True
    assert should_compact(history, context_window=1_000, fraction=0.5, freed=500)[0] is False


def test_the_discount_cannot_drive_the_measurement_negative() -> None:
    history = [user("hi"), ModelResponse(parts=[TextPart(content="yo")], usage=RequestUsage(input_tokens=800))]
    assert should_compact(history, context_window=1_000, fraction=0.5, freed=10_000)[1] == 0


def reading_model(big: Path, stop_after: int = 8) -> FunctionModel:
    """A model that re-reads one large file until it is told to stop."""
    calls: list[int] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        usage = RequestUsage(input_tokens=900)
        if len(calls) < stop_after:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="Read", args={"file_path": str(big)})],
                usage=usage,
            )
        return ModelResponse(parts=[TextPart(content="done")], usage=usage)

    return FunctionModel(respond)


def microcompacting_options(tmp_path: Path, model: Any, **kwargs: Any) -> Options:
    """Options tuned so a run of more than two turns is always under pressure."""
    kwargs.setdefault(
        "compact_model",
        FunctionModel(lambda m, i: ModelResponse(parts=[TextPart(content="COMPACTED")])),
    )
    return Options(
        model=model,
        cwd=tmp_path,
        allowed_tools=["Read"],
        permission_mode="bypassPermissions",
        max_context_tokens=1_000,
        compact_threshold=0.5,
        compact_keep_recent=2,
        microcompact_keep_recent=1,
        **kwargs,
    )


async def test_a_pressured_run_emits_microcompact_messages(tmp_path: Path) -> None:
    big = tmp_path / "big.txt"
    big.write_text("lorem ipsum dolor sit amet\n" * 400)

    options = microcompacting_options(tmp_path, reading_model(big))
    messages = [m async for m in summon("read it", options)]

    reclaimed = [m for m in messages if getattr(m, "subtype", None) == "microcompact"]
    assert reclaimed
    assert all(m.tokens_saved > 0 for m in reclaimed)
    assert all(m.cleared >= 1 for m in reclaimed)


async def test_microcompaction_defers_full_compaction(tmp_path: Path) -> None:
    """The point of the cheap tier is that the expensive one runs less often."""
    big = tmp_path / "big.txt"
    big.write_text("lorem ipsum dolor sit amet\n" * 400)

    with_micro = [
        m
        async for m in summon(
            "read it", microcompacting_options(tmp_path, reading_model(big))
        )
    ]
    without_micro = [
        m
        async for m in summon(
            "read it",
            microcompacting_options(tmp_path, reading_model(big), auto_microcompact=False),
        )
    ]

    def boundaries(messages: list[Any]) -> int:
        return len([m for m in messages if getattr(m, "subtype", None) == "compact_boundary"])

    assert boundaries(with_micro) < boundaries(without_micro)


async def test_microcompaction_can_be_disabled(tmp_path: Path) -> None:
    big = tmp_path / "big.txt"
    big.write_text("lorem ipsum dolor sit amet\n" * 400)

    options = microcompacting_options(tmp_path, reading_model(big), auto_microcompact=False)
    messages = [m async for m in summon("read it", options)]

    assert not [m for m in messages if getattr(m, "subtype", None) == "microcompact"]
