"""Every tool call is answered in the message stream, including refused ones.

A host renders the stream as it arrives, so a `tool_use` with no matching
`tool_result` is a call left displayed as running. These tests pin the pairing
across every way a call can end: it ran, a rule refused it, a hook blocked it,
its arguments were rejected, or the tool itself raised.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_ai.messages import TextPart, ToolCallPart

from test_summon_loop import collect, deny_all, scripted
from ubiquity import Options
from ubiquity.hooks import HookMatcher, HookOutput
from ubiquity.tool import Tool, ToolContext, ValidationError
from ubiquity.types import ToolOutput


class BoomInput(BaseModel):
    """Input for the tools defined here; deliberately has a required field."""

    text: str


class BoomTool(Tool[BoomInput]):
    """Raises when called, to exercise the failure path."""

    name = "Boom"
    description = "Always raises."
    input_model = BoomInput

    async def call(self, args: BoomInput, ctx: ToolContext) -> ToolOutput:
        raise RuntimeError("nope")


class PickyTool(Tool[BoomInput]):
    """Rejects every call in `validate_input`."""

    name = "Picky"
    description = "Rejects every call."
    input_model = BoomInput

    async def call(self, args: BoomInput, ctx: ToolContext) -> ToolOutput:
        return ToolOutput(content=args.text)

    async def validate_input(
        self, args: BoomInput, ctx: ToolContext
    ) -> ValidationError | None:
        return ValidationError(message="text must name a file that exists")


class QuietTool(Tool[BoomInput]):
    """Succeeds, as the baseline for the pairing check."""

    name = "Quiet"
    description = "Returns its input."
    input_model = BoomInput

    async def call(self, args: BoomInput, ctx: ToolContext) -> ToolOutput:
        return ToolOutput(content=args.text)


def pairs(messages: list[Any]) -> tuple[list[str], list[str]]:
    """Return the tool_use ids and the tool_result ids, in stream order."""
    return (
        [m.tool_use_id for m in messages if m.type == "tool_use"],
        [m.tool_use_id for m in messages if m.type == "tool_result"],
    )


def assert_paired(messages: list[Any]) -> None:
    """Assert every tool_use is answered by exactly one tool_result."""
    uses, results = pairs(messages)
    assert uses, "no tool_use message was emitted at all"
    assert uses == results


def results(messages: list[Any]) -> list[Any]:
    """Return the tool_result messages in stream order."""
    return [m for m in messages if m.type == "tool_result"]


async def run(tool: Tool[Any], args: dict[str, Any], **options: Any) -> list[Any]:
    """Drive one scripted tool call through the loop and return the stream."""
    cwd = options.pop("cwd")
    model = scripted(
        [ToolCallPart(tool_name=tool.name, args=args)],
        [TextPart(content="done")],
    )
    return await collect("go", Options(model=model, cwd=cwd, tools=[tool], **options))


async def test_a_successful_call_is_paired(tmp_path: Path) -> None:
    messages = await run(
        QuietTool(), {"text": "hello"}, cwd=tmp_path, allowed_tools=["Quiet"]
    )

    assert_paired(messages)
    assert results(messages)[0].output.is_error is False


async def test_a_denied_call_is_paired_and_marked_an_error(tmp_path: Path) -> None:
    messages = await run(
        QuietTool(),
        {"text": "hello"},
        cwd=tmp_path,
        ask_tools=["Quiet"],
        can_use_tool=deny_all,
    )

    assert_paired(messages)
    result = results(messages)[0]
    assert result.output.is_error is True
    assert "denied by policy" in result.output.content


async def test_a_denial_is_reported_before_the_run_ends(tmp_path: Path) -> None:
    """The point of the pair: the host hears about it at the moment it happens.

    Reporting the denial only in the terminal result message would leave the
    call displayed as outstanding for the rest of the run, which on a long run
    is the whole of the time the user needed to know.
    """
    messages = await run(
        QuietTool(),
        {"text": "hello"},
        cwd=tmp_path,
        ask_tools=["Quiet"],
        can_use_tool=deny_all,
    )
    kinds = [m.type for m in messages]

    assert kinds.index("tool_result") < kinds.index("result")


async def test_a_call_blocked_by_a_hook_is_paired(tmp_path: Path) -> None:
    async def block(payload):
        return HookOutput(decision="block", reason="not on my watch")

    messages = await run(
        QuietTool(),
        {"text": "hello"},
        cwd=tmp_path,
        allowed_tools=["Quiet"],
        hooks=[HookMatcher("PreToolUse", [block])],
    )

    assert_paired(messages)
    assert "not on my watch" in results(messages)[0].output.content


async def test_a_call_rejected_by_validate_input_is_paired(tmp_path: Path) -> None:
    messages = await run(
        PickyTool(), {"text": "hello"}, cwd=tmp_path, allowed_tools=["Picky"]
    )

    assert_paired(messages)
    result = results(messages)[0]
    assert result.output.is_error is True
    assert "must name a file that exists" in result.output.content


async def test_a_call_with_unparseable_arguments_is_paired(tmp_path: Path) -> None:
    """A call the model malformed is still a call the host drew on screen."""
    messages = await run(
        QuietTool(), {"wrong": "field"}, cwd=tmp_path, allowed_tools=["Quiet"]
    )

    assert_paired(messages)
    assert results(messages)[0].output.is_error is True


async def test_a_tool_that_raises_is_paired(tmp_path: Path) -> None:
    messages = await run(
        BoomTool(), {"text": "hello"}, cwd=tmp_path, allowed_tools=["Boom"]
    )

    assert_paired(messages)
    result = results(messages)[0]
    assert result.output.is_error is True
    assert "nope" in result.output.content


async def test_a_refusal_reports_the_input_it_judged(tmp_path: Path) -> None:
    """A hook may rewrite the input before a later step refuses the call.

    The emitted `tool_use` carries the rewritten form, because that is what was
    refused. Showing the model's original would describe a decision nobody
    made.
    """

    async def rewrite(payload):
        return HookOutput(updated_input={"text": "rewritten"})

    messages = await run(
        QuietTool(),
        {"text": "original"},
        cwd=tmp_path,
        hooks=[HookMatcher("PreToolUse", [rewrite])],
        ask_tools=["Quiet"],
        can_use_tool=deny_all,
    )

    assert_paired(messages)
    use = next(m for m in messages if m.type == "tool_use")
    assert use.tool_input == {"text": "rewritten"}


async def test_the_result_carries_the_id_the_model_used(tmp_path: Path) -> None:
    """The pair has to be joinable to the tool_use block in the model response.

    A host matches the stream against the assistant message by id, so a
    refusal emitted under a freshly minted id would read as an unrelated call
    rather than as the answer to the one on screen.
    """
    messages = await run(
        QuietTool(),
        {"text": "hello"},
        cwd=tmp_path,
        ask_tools=["Quiet"],
        can_use_tool=deny_all,
    )

    assistant = next(
        m
        for m in messages
        if m.type == "assistant"
        and any(block.get("type") == "tool_use" for block in m.content)
    )
    called_id = next(
        block["id"] for block in assistant.content if block["type"] == "tool_use"
    )
    uses, _ = pairs(messages)

    assert uses == [called_id]


async def test_a_denial_is_still_collected_for_the_result_message(
    tmp_path: Path,
) -> None:
    """The stream pair is an addition, not a replacement.

    A caller that drains the run and reads only the terminal message must keep
    seeing denials there, since that is the shape the non-streaming path has
    always had.
    """
    messages = await run(
        QuietTool(),
        {"text": "hello"},
        cwd=tmp_path,
        ask_tools=["Quiet"],
        can_use_tool=deny_all,
    )

    denials = messages[-1].permission_denials
    assert [d.tool_name for d in denials] == ["Quiet"]
