"""A blocking `Stop` hook sends the agent back for another turn.

`Stop` fires when the agent is ready to finish, and blocking it means the work
is not actually done. The only useful reading of that is another turn with the
hook's reason as the prompt: a veto that could do nothing but rewrite the
result text would report a failure the run never had, and would never give the
model the turn the veto was asking for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ubiquity import HookMatcher, HookOutput, Options, summon
from ubiquity.hooks import HookInput


def blocking(times: int, reason: str = "keep going") -> Any:
    """A Stop hook that blocks `times` times and then lets the run end."""
    seen = {"n": 0}

    async def hook(payload: HookInput) -> HookOutput:
        seen["n"] += 1
        if seen["n"] > times:
            return HookOutput()
        return HookOutput(decision="block", reason=reason)

    return hook


def transcribing(prompts: list[str]) -> FunctionModel:
    """A model that records the last user text it saw and answers with a count."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        texts = [
            part.content
            for message in messages
            for part in message.parts
            if getattr(part, "part_kind", "") == "user-prompt"
        ]
        prompts.append(str(texts[-1]) if texts else "")
        return ModelResponse(parts=[TextPart(content=f"turn {len(prompts)}")])

    return FunctionModel(respond)


async def run(model: Any, tmp_path: Path, hook: Any, **options: Any) -> list[Any]:
    """Drive one run with `hook` subscribed to Stop and return the stream."""
    return [
        m
        async for m in summon(
            "do the thing",
            Options(
                model=model,
                cwd=tmp_path,
                tools=[],
                hooks=[HookMatcher("Stop", [hook])],
                persist_session=False,
                persist_todos=False,
                **options,
            ),
        )
    ]


async def test_a_block_asks_the_model_for_another_turn(tmp_path: Path) -> None:
    prompts: list[str] = []
    messages = await run(transcribing(prompts), tmp_path, blocking(1), max_turns=5)

    assert len(prompts) == 2
    assert messages[-1].result == "turn 2"


async def test_the_reason_becomes_the_next_prompt(tmp_path: Path) -> None:
    """The reason is the instruction, so it has to reach the model verbatim."""
    prompts: list[str] = []
    await run(
        transcribing(prompts),
        tmp_path,
        blocking(1, reason="the tests are still failing"),
        max_turns=5,
    )

    assert prompts[1] == "the tests are still failing"


async def test_a_continued_run_still_succeeds(tmp_path: Path) -> None:
    """The veto extends the work; it does not relabel it as a failure.

    A hook meant to keep the agent working that instead reported an error was
    the sharpest edge of the old behavior, because the run had in fact done
    everything asked of it.
    """
    prompts: list[str] = []
    messages = await run(transcribing(prompts), tmp_path, blocking(1), max_turns=5)
    result = messages[-1]

    assert result.subtype == "success"
    assert result.is_error is False


async def test_the_continuation_is_visible_in_the_stream(tmp_path: Path) -> None:
    """A host has to be able to show why the agent carried on."""
    prompts: list[str] = []
    messages = await run(
        transcribing(prompts), tmp_path, blocking(1, reason="not done"), max_turns=5
    )
    users = [m.content for m in messages if m.type == "user"]

    assert users[-1] == "not done"


async def test_a_hook_that_never_relents_is_bounded_by_max_turns(
    tmp_path: Path,
) -> None:
    """Blocking without limit is asking without limit; the budget is the limit."""
    prompts: list[str] = []
    messages = await run(transcribing(prompts), tmp_path, blocking(99), max_turns=3)
    result = messages[-1]

    assert result.subtype == "error_max_turns"
    assert result.is_error is True
    assert len(prompts) <= 3


async def test_a_veto_at_the_budget_does_not_prompt_a_turn_that_cannot_run(
    tmp_path: Path,
) -> None:
    """The last turn is spent, so asking for another would be asking for nothing.

    Emitting the continuation anyway would show the user an instruction the
    agent was never given a turn to act on, which reads as work in progress
    that silently never happened.
    """
    prompts: list[str] = []
    messages = await run(transcribing(prompts), tmp_path, blocking(99), max_turns=1)
    users = [m.content for m in messages if m.type == "user"]

    assert users == ["do the thing"]
    assert messages[-1].subtype == "error_max_turns"


async def test_a_hook_that_does_not_block_ends_the_run(tmp_path: Path) -> None:
    prompts: list[str] = []
    messages = await run(transcribing(prompts), tmp_path, blocking(0), max_turns=5)

    assert len(prompts) == 1
    assert messages[-1].subtype == "success"


async def test_a_block_without_a_reason_still_asks_for_a_turn(
    tmp_path: Path,
) -> None:
    """A hook may veto without explaining, and the veto still has to mean it."""

    async def mute(payload: HookInput) -> HookOutput:
        return HookOutput(decision="block")

    prompts: list[str] = []
    hook = blocking(0)
    calls = {"n": 0}

    async def once(payload: HookInput) -> HookOutput:
        calls["n"] += 1
        return await (mute(payload) if calls["n"] == 1 else hook(payload))

    messages = await run(transcribing(prompts), tmp_path, once, max_turns=5)

    assert len(prompts) == 2
    assert "Stop hook" in prompts[1]
    assert messages[-1].subtype == "success"


async def test_the_model_sees_the_first_turn_in_its_history(tmp_path: Path) -> None:
    """A continuation continues; it does not restart with a blank slate.

    Re-entering with an empty history would make the second turn redo the work
    the first turn already did, which is the opposite of what a hook asking for
    more work wants.
    """
    lengths: list[int] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        lengths.append(len(messages))
        return ModelResponse(parts=[TextPart(content="ok")])

    await run(FunctionModel(respond), tmp_path, blocking(1), max_turns=5)

    assert lengths[1] > lengths[0]


async def test_usage_covers_every_leg_of_a_continued_run(tmp_path: Path) -> None:
    """Each `agent.iter` reports only its own leg, so the totals must add up.

    Replacing rather than accumulating would bill the caller for the final
    continuation alone: a number that looks plausible and understates every run
    a hook extended.
    """
    prompts: list[str] = []
    one = await run(transcribing(prompts), tmp_path, blocking(0), max_turns=5)
    prompts.clear()
    two = await run(transcribing(prompts), tmp_path, blocking(1), max_turns=5)

    assert two[-1].usage["requests"] > one[-1].usage["requests"]
    assert two[-1].usage["output_tokens"] > one[-1].usage["output_tokens"]


async def test_a_failed_run_is_not_offered_to_the_hook(tmp_path: Path) -> None:
    """Stop asks whether finished work is done, not whether a crash should retry.

    Re-entering after an error would loop the failure until the turn budget
    ran out and report `max_turns` for something that was never about turns.
    """
    calls: list[str] = []

    async def watch(payload: HookInput) -> HookOutput:
        calls.append("stop")
        return HookOutput(decision="block", reason="again")

    def explode(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("model is down")

    messages = await run(FunctionModel(explode), tmp_path, watch, max_turns=5)

    assert calls == []
    assert messages[-1].subtype == "error_during_execution"
