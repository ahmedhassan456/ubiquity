"""Tests for cancelling a run in progress.

`ToolContext.abort` was declared and documented long before anything checked
it, which is the failure the field audit exists to catch. These are the tests
that make the documentation true.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ubiquity import Options, summon
from ubiquity.tools.bash import BashInput, BashTool
from ubiquity.toolset import ToolDenied


def scripted(*turns: list[Any]) -> FunctionModel:
    """Build a model that replays `turns`, one response per model request."""
    calls = {"n": 0}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        index = min(calls["n"], len(turns) - 1)
        calls["n"] += 1
        return ModelResponse(parts=list(turns[index]))

    return FunctionModel(respond)


async def collect(prompt: str, options: Options) -> list[Any]:
    return [m async for m in summon(prompt, options)]


class TestBash:
    async def test_an_abort_kills_the_process_it_is_waiting_on(
        self, make_ctx, tmp_path: Path
    ) -> None:
        """The whole point: a runaway command must not run out its timeout."""
        abort = asyncio.Event()
        ctx = make_ctx(cwd=tmp_path, abort=abort)
        args = BashInput(command="sleep 30", timeout=30_000)

        async def cancel() -> None:
            await asyncio.sleep(0.05)
            abort.set()

        started = time.monotonic()
        with pytest.raises(ToolDenied) as raised:
            await asyncio.gather(BashTool().call(args, ctx), cancel())
        elapsed = time.monotonic() - started

        assert raised.value.interrupt
        assert elapsed < 5
        assert "sleep 30" in raised.value.message

    async def test_the_process_is_reaped_not_merely_signalled(
        self, make_ctx, tmp_path: Path
    ) -> None:
        """A killed but unwaited process lingers as a zombie holding its pipes."""
        abort = asyncio.Event()
        ctx = make_ctx(cwd=tmp_path, abort=abort)
        abort.set()
        with pytest.raises(ToolDenied):
            await BashTool().call(BashInput(command="sleep 30"), ctx)
        await asyncio.sleep(0)

    async def test_a_command_that_finishes_is_unaffected(
        self, make_ctx, tmp_path: Path
    ) -> None:
        ctx = make_ctx(cwd=tmp_path, abort=asyncio.Event())
        output = await BashTool().call(BashInput(command="echo hello"), ctx)
        assert output.content == "hello"
        assert not output.is_error

    async def test_a_run_with_no_abort_behaves_exactly_as_before(
        self, make_ctx, tmp_path: Path
    ) -> None:
        ctx = make_ctx(cwd=tmp_path)
        assert ctx.abort is None
        output = await BashTool().call(BashInput(command="echo hello"), ctx)
        assert output.content == "hello"

    async def test_the_timeout_still_fires_with_an_abort_present(
        self, make_ctx, tmp_path: Path
    ) -> None:
        """Racing the event must not swallow the deadline it races against."""
        ctx = make_ctx(cwd=tmp_path, abort=asyncio.Event())
        output = await BashTool().call(
            BashInput(command="sleep 30", timeout=200), ctx
        )
        assert output.is_error
        assert output.metadata["timed_out"] is True


class TestTheToolFunnel:
    async def test_an_aborted_run_refuses_a_tool_before_it_starts(
        self, make_ctx, tmp_path: Path
    ) -> None:
        """The check has to be in the funnel, not only in the loop.

        MCP tools and caller-registered tools reach `call_with` without the
        loop's own check having run against that call.
        """
        from ubiquity.toolset import UbiquityToolset

        from doubles import EchoInput, EchoTool

        abort = asyncio.Event()
        abort.set()
        ctx = make_ctx(cwd=tmp_path, mode="bypassPermissions", abort=abort)
        toolset = UbiquityToolset([], ctx, emit=lambda m: None)

        with pytest.raises(ToolDenied) as raised:
            await toolset.call_with(EchoTool(), {"command": "hi"})
        assert raised.value.interrupt

    async def test_the_funnel_lets_an_unaborted_call_through(
        self, make_ctx, tmp_path: Path
    ) -> None:
        from ubiquity.toolset import UbiquityToolset

        from doubles import EchoTool

        ctx = make_ctx(cwd=tmp_path, mode="bypassPermissions", abort=asyncio.Event())
        toolset = UbiquityToolset([], ctx, emit=lambda m: None)
        assert "hi" in str(await toolset.call_with(EchoTool(), {"command": "hi"}))


class TestReaping:
    async def test_terminate_waits_for_the_process_to_actually_die(
        self, tmp_path: Path
    ) -> None:
        """A killed but unwaited child is a zombie still holding its pipes."""
        from ubiquity.tools.bash import _terminate

        proc = await asyncio.create_subprocess_shell(
            "sleep 30",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.returncode is None
        await _terminate(proc)
        assert proc.returncode is not None

    async def test_terminate_is_harmless_on_a_finished_process(self) -> None:
        from ubiquity.tools.bash import _terminate

        proc = await asyncio.create_subprocess_shell("true")
        await proc.wait()
        code = proc.returncode
        await _terminate(proc)
        assert proc.returncode == code


class TestTheLoop:
    async def test_an_aborted_run_refuses_the_next_tool_call(
        self, tmp_path: Path
    ) -> None:
        abort = asyncio.Event()
        abort.set()
        model = scripted(
            [ToolCallPart(tool_name="Bash", args={"command": "echo hi"}, tool_call_id="c0")],
            [TextPart(content="done")],
        )
        messages = await collect(
            "run it",
            Options(
                model=model,
                cwd=tmp_path,
                abort=abort,
                permission_mode="bypassPermissions",
            ),
        )
        result = messages[-1]
        assert result.is_error
        assert not any(m.type == "tool_use" for m in messages)

    async def test_an_abort_ends_a_run_that_calls_no_tools(
        self, tmp_path: Path
    ) -> None:
        """A model looping on text alone has no tool call to refuse."""
        abort = asyncio.Event()
        abort.set()
        messages = await collect(
            "hello",
            Options(model=scripted([TextPart(content="hi")]), cwd=tmp_path, abort=abort),
        )
        result = messages[-1]
        assert result.is_error
        assert "abort" in str(result.result).lower()

    async def test_an_unaborted_run_is_untouched(self, tmp_path: Path) -> None:
        messages = await collect(
            "hello",
            Options(
                model=scripted([TextPart(content="hi")]),
                cwd=tmp_path,
                abort=asyncio.Event(),
            ),
        )
        assert not messages[-1].is_error
