"""Tests for AskUserQuestion, and for what happens when nobody answers.

The tool has almost no logic of its own; its behavior is the permission
pipeline's. So most of these are about the seam: that the prompt survives the
modes and rules that skip other prompts, that the model cannot write the
user's answer for them, and that a question left open ends in a denial rather
than a hung run or an invented reply.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import TextPart

from ubiquity import Options, summon
from ubiquity.permissions.engine import check_permissions
from ubiquity.tool import ToolContext
from ubiquity.tools.ask import (
    HEADER_WIDTH,
    AskUserQuestionInput,
    AskUserQuestionTool,
)
from ubiquity.toolset import ToolDenied, UbiquityToolset
from ubiquity.types import PermissionResultAllow

QUESTION = {
    "question": "Which database should the service use?",
    "header": "Database",
    "options": [
        {"label": "Postgres (Recommended)", "description": "Relational, familiar."},
        {"label": "SQLite", "description": "Embedded, no server to run."},
    ],
}

SECOND = {
    "question": "Should migrations run on boot?",
    "header": "Migrations",
    "options": [
        {"label": "Yes", "description": "Convenient in development."},
        {"label": "No", "description": "Safer in production."},
    ],
}


def call_args(*questions: dict[str, Any]) -> dict[str, Any]:
    """The raw tool arguments for one or more questions."""
    return {"questions": [dict(q) for q in questions]}


def with_handler(ctx: ToolContext, handler: Any, **option_overrides: Any) -> ToolContext:
    """Return `ctx` with a `can_use_tool` handler installed."""
    return replace(
        ctx, options=replace(ctx.options, can_use_tool=handler, **option_overrides)
    )


def answering(answers: dict[str, str]) -> Any:
    """A handler that stands in for the user, returning `answers`."""

    async def handler(name: str, args: dict[str, Any], ctx: ToolContext):
        return PermissionResultAllow(updated_input={**args, "answers": answers})

    return handler


class TestThePromptCannotBeSkipped:
    async def test_the_tool_asks_rather_than_allowing_itself(self, make_ctx) -> None:
        tool = AskUserQuestionTool()
        args = AskUserQuestionInput(**call_args(QUESTION))
        result = await tool.check_permissions(args, make_ctx())
        assert result.behavior == "ask"

    async def test_bypass_permissions_still_prompts(self, make_ctx) -> None:
        """The ask is the tool's whole effect, so bypassing it bypasses the tool."""
        tool = AskUserQuestionTool()
        args = AskUserQuestionInput(**call_args(QUESTION))
        ctx = make_ctx(mode="bypassPermissions")
        assert (await check_permissions(tool, args, ctx)).behavior == "ask"

    async def test_a_bare_allow_rule_still_prompts(self, make_ctx) -> None:
        tool = AskUserQuestionTool()
        args = AskUserQuestionInput(**call_args(QUESTION))
        ctx = make_ctx(allow={"AskUserQuestion"})
        assert (await check_permissions(tool, args, ctx)).behavior == "ask"

    async def test_a_deny_rule_takes_it_away(self, make_ctx) -> None:
        tool = AskUserQuestionTool()
        args = AskUserQuestionInput(**call_args(QUESTION))
        ctx = make_ctx(deny={"AskUserQuestion"})
        assert (await check_permissions(tool, args, ctx)).behavior == "deny"


class TestTheModelCannotAnswerItself:
    async def test_prefilled_answers_are_rejected(self, make_ctx) -> None:
        """Otherwise the model could report a decision the user never made."""
        tool = AskUserQuestionTool()
        args = AskUserQuestionInput(
            **call_args(QUESTION), answers={QUESTION["question"]: "SQLite"}
        )
        error = await tool.validate_input(args, make_ctx())
        assert error is not None
        assert "answers" in error.message

    async def test_duplicate_questions_are_rejected(self, make_ctx) -> None:
        tool = AskUserQuestionTool()
        args = AskUserQuestionInput(**call_args(QUESTION, QUESTION))
        error = await tool.validate_input(args, make_ctx())
        assert error is not None

    async def test_duplicate_option_labels_are_rejected(self, make_ctx) -> None:
        """Two identical labels give the user a choice with no distinct answer."""
        same = dict(QUESTION)
        same["options"] = [
            {"label": "Postgres", "description": "One."},
            {"label": "Postgres", "description": "Another."},
        ]
        tool = AskUserQuestionTool()
        error = await tool.validate_input(
            AskUserQuestionInput(**call_args(same)), make_ctx()
        )
        assert error is not None
        assert "distinct labels" in error.message

    async def test_an_over_long_header_is_rejected(self, make_ctx) -> None:
        """The description states a maximum, so the code has to enforce one."""
        wide = dict(QUESTION, header="D" * (HEADER_WIDTH + 1))
        tool = AskUserQuestionTool()
        error = await tool.validate_input(
            AskUserQuestionInput(**call_args(wide)), make_ctx()
        )
        assert error is not None
        assert str(HEADER_WIDTH) in error.message

    def test_a_question_needs_at_least_two_options(self) -> None:
        lonely = dict(QUESTION, options=[{"label": "Only", "description": "One."}])
        with pytest.raises(Exception):
            AskUserQuestionInput(**call_args(lonely))


class TestTheAnswersComeBack:
    async def test_the_handler_supplies_answers_the_tool_reports(
        self, make_ctx
    ) -> None:
        tool = AskUserQuestionTool()
        ctx = with_handler(make_ctx(), answering({QUESTION["question"]: "SQLite"}))
        toolset = UbiquityToolset([tool], ctx)

        output = await toolset.call_with(tool, call_args(QUESTION))

        assert "SQLite" in output
        assert QUESTION["question"] in output

    async def test_an_unanswered_question_is_named_not_hidden(self, make_ctx) -> None:
        tool = AskUserQuestionTool()
        ctx = with_handler(make_ctx(), answering({QUESTION["question"]: "SQLite"}))
        toolset = UbiquityToolset([tool], ctx)

        output = await toolset.call_with(tool, call_args(QUESTION, SECOND))

        assert "SQLite" in output
        assert "Left unanswered" in output
        assert SECOND["question"] in output

    async def test_an_approval_with_no_answers_is_an_error(self, make_ctx) -> None:
        """A generic approval dialog allows without collecting anything.

        Reporting that as a successful empty form would tell the model the
        user had nothing to say, when in fact nobody was asked.
        """
        tool = AskUserQuestionTool()
        ctx = with_handler(make_ctx(), answering({}))
        toolset = UbiquityToolset([tool], ctx)

        with pytest.raises(Exception) as caught:
            await toolset.call_with(tool, call_args(QUESTION))
        assert "No answers came back" in str(caught.value)


class TestAQuestionNobodyAnswers:
    async def test_an_abort_ends_a_pending_question(self, make_ctx) -> None:
        """Without racing the abort, a pending prompt outlives the aborted run."""
        abort = asyncio.Event()

        async def never_answers(name: str, args: dict[str, Any], ctx: ToolContext):
            await asyncio.Event().wait()

        tool = AskUserQuestionTool()
        ctx = replace(with_handler(make_ctx(), never_answers), abort=abort)
        toolset = UbiquityToolset([tool], ctx)

        async def abort_soon() -> None:
            await asyncio.sleep(0.05)
            abort.set()

        task = asyncio.gather(
            toolset.call_with(tool, call_args(QUESTION)), abort_soon()
        )
        with pytest.raises(ToolDenied) as caught:
            await task
        assert caught.value.interrupt is True

    async def test_a_timeout_denies_rather_than_proceeding(self, make_ctx) -> None:
        """Expiry must not be read as consent, and must not invent an answer."""

        async def never_answers(name: str, args: dict[str, Any], ctx: ToolContext):
            await asyncio.Event().wait()

        tool = AskUserQuestionTool()
        ctx = with_handler(
            make_ctx(), never_answers, permission_prompt_timeout_s=0.05
        )
        toolset = UbiquityToolset([tool], ctx)

        output = await toolset.call_with(tool, call_args(QUESTION))

        assert "denied" in output
        assert "unanswered" in output

    async def test_no_timeout_means_no_deadline(self, make_ctx) -> None:
        """By default a prompt waits until it is resolved, however long that takes."""
        released = asyncio.Event()

        async def answers_eventually(
            name: str, args: dict[str, Any], ctx: ToolContext
        ):
            await released.wait()
            return PermissionResultAllow(
                updated_input={**args, "answers": {QUESTION["question"]: "Postgres"}}
            )

        tool = AskUserQuestionTool()
        ctx = with_handler(make_ctx(), answers_eventually)
        toolset = UbiquityToolset([tool], ctx)

        async def release_late() -> None:
            await asyncio.sleep(0.1)
            released.set()

        output, _ = await asyncio.gather(
            toolset.call_with(tool, call_args(QUESTION)), release_late()
        )
        assert "Postgres" in output


class TestAvailability:
    async def test_a_run_with_no_handler_does_not_offer_the_tool(
        self, tmp_path: Path
    ) -> None:
        """Nobody to ask means the tool could only ever be denied."""
        from test_summon_loop import collect, scripted

        messages = await collect(
            "hello",
            Options(model=scripted([TextPart(content="ok")]), cwd=tmp_path),
        )
        assert "AskUserQuestion" not in messages[0].tools

    async def test_a_run_with_a_handler_offers_it(self, tmp_path: Path) -> None:
        from test_summon_loop import allow_all, collect, scripted

        messages = await collect(
            "hello",
            Options(
                model=scripted([TextPart(content="ok")]),
                cwd=tmp_path,
                can_use_tool=allow_all,
            ),
        )
        assert "AskUserQuestion" in messages[0].tools

    async def test_a_deny_rule_keeps_it_out_of_the_suite(self, tmp_path: Path) -> None:
        from test_summon_loop import allow_all, collect, scripted

        messages = await collect(
            "hello",
            Options(
                model=scripted([TextPart(content="ok")]),
                cwd=tmp_path,
                can_use_tool=allow_all,
                disallowed_tools=["AskUserQuestion"],
            ),
        )
        assert "AskUserQuestion" not in messages[0].tools


async def test_a_subagent_is_not_given_the_tool(tmp_path: Path) -> None:
    """Its report goes to the parent, so a dialog it opened would have no context."""
    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    from ubiquity.client import _build_context, run_subagent
    from ubiquity.hooks.registry import HookRegistry

    seen: dict[str, set[str]] = {}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen["tools"] = {t.name for t in info.function_tools}
        return ModelResponse(parts=[TextPart(content="done")])

    async def allow(name: str, args: dict[str, Any], ctx: ToolContext):
        return PermissionResultAllow()

    options = Options(
        model=FunctionModel(respond),
        cwd=tmp_path,
        can_use_tool=allow,
        tools=[AskUserQuestionTool()],
    )
    ctx = _build_context(options, "s", HookRegistry())

    await run_subagent("x", ctx, None)
    assert "AskUserQuestion" not in seen["tools"]
