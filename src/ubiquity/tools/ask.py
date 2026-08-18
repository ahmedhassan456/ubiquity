"""The AskUserQuestion tool.

Lets the model put a small multiple-choice form in front of the user and
continue with the answer.

The mechanism is the part worth understanding, because the tool itself does
almost nothing. `call` does not prompt anybody: it formats answers that are
already present in its input. The prompting happens one layer up, in the
permission pipeline. `check_permissions` returns `ask`, `requires_user_interaction`
keeps that `ask` alive through `bypassPermissions` and through any allow rule,
and the host's `can_use_tool` handler is what actually renders the questions and
returns `allow` with the answers written into `updated_input`. The toolset
re-parses the authorized input before calling the tool, so the answers arrive as
ordinary validated fields.

Reusing the permission prompt rather than adding a second waiting mechanism is
what makes the unanswered case behave sensibly. A question that nobody answers
is a permission prompt that nobody answers, so it is already covered by
everything that resolves one: an abort ends it, `dontAsk` mode declines it
without prompting, and a run with no `can_use_tool` handler cannot offer the
tool in the first place. There is deliberately no timer that invents an answer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..tool import Tool, ToolContext, ValidationError
from ..types import PermissionResult, PermissionResultAsk, ToolOutput

MAX_QUESTIONS = 4
MIN_OPTIONS = 2
MAX_OPTIONS = 4
HEADER_WIDTH = 12


class QuestionOption(BaseModel):
    label: str = Field(
        description="The choice as the user sees it, a few words at most."
    )
    description: str = Field(
        description="What choosing this option means or leads to."
    )


class Question(BaseModel):
    question: str = Field(description="The question, phrased so it can be answered.")
    header: str = Field(
        description=f"Short label for the question, at most {HEADER_WIDTH} characters."
    )
    options: list[QuestionOption] = Field(
        min_length=MIN_OPTIONS,
        max_length=MAX_OPTIONS,
        description="The choices offered for this question.",
    )
    multi_select: bool = Field(
        default=False, description="Allow more than one option to be chosen."
    )


class AskUserQuestionInput(BaseModel):
    questions: list[Question] = Field(
        min_length=1,
        max_length=MAX_QUESTIONS,
        description="The questions to put to the user.",
    )
    answers: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Filled in with the user's replies once they answer. Leave it out: "
            "a call that already carries answers is rejected."
        ),
    )


class AskUserQuestionTool(Tool[AskUserQuestionInput]):
    """Put multiple-choice questions to the user and wait for the answers."""

    name = "AskUserQuestion"
    description = (
        f"Ask the user up to {MAX_QUESTIONS} multiple-choice questions and "
        "continue with their answers. Use it for a decision that is genuinely "
        "the user's to make and that you cannot settle from the request, the "
        "code, or a sensible default, not for a choice with an obvious answer. "
        f"Each question needs {MIN_OPTIONS} to {MAX_OPTIONS} options; if you "
        "recommend one, put it first and end its label with `(Recommended)`. "
        "Set `multi_select` when several options may be chosen together. The "
        "user can always answer with something you did not offer, so read the "
        "reply as text rather than assuming it is one of your labels. The run "
        "waits for the reply, so ask everything you need in one call. Leave "
        "`answers` unset: it is filled in when the user responds."
    )
    input_model = AskUserQuestionInput
    search_hint = "ask user question clarify choice preference decide"

    def is_read_only(self, args: AskUserQuestionInput) -> bool:
        return True

    def is_concurrency_safe(self, args: AskUserQuestionInput) -> bool:
        return True

    def requires_user_interaction(self) -> bool:
        """The user is the whole point, so no mode or rule may skip the prompt."""
        return True

    def describe_call(self, args: AskUserQuestionInput) -> str:
        return "Ask: " + ", ".join(q.header for q in args.questions)

    async def validate_input(
        self, args: AskUserQuestionInput, ctx: ToolContext
    ) -> ValidationError | None:
        """Reject a call the user could not answer, or one that answers itself."""
        if args.answers:
            return ValidationError(
                message=(
                    "`answers` is filled in by the user, not by you. Send the "
                    "questions with `answers` left out."
                )
            )

        texts = [q.question for q in args.questions]
        if len(set(texts)) != len(texts):
            return ValidationError(
                message="Each question must be different; two of them are identical."
            )

        for question in args.questions:
            if len(question.header) > HEADER_WIDTH:
                return ValidationError(
                    message=(
                        f"Header {question.header!r} is longer than "
                        f"{HEADER_WIDTH} characters."
                    )
                )
            labels = [option.label for option in question.options]
            if len(set(labels)) != len(labels):
                return ValidationError(
                    message=(
                        f"The options for {question.question!r} must have "
                        "distinct labels."
                    )
                )
        return None

    async def check_permissions(
        self, args: AskUserQuestionInput, ctx: ToolContext
    ) -> PermissionResult:
        """Ask, which is what puts the questions in front of the user.

        The `ask` is not a request to approve a side effect; it *is* the tool's
        effect. The handler that resolves it is expected to return `allow` with
        `answers` written into `updated_input`.
        """
        count = len(args.questions)
        noun = "question" if count == 1 else "questions"
        return PermissionResultAsk(message=f"Answer {count} {noun}?")

    async def call(
        self, args: AskUserQuestionInput, ctx: ToolContext
    ) -> ToolOutput:
        """Report the answers, and say plainly which questions came back empty.

        A handler may allow the call without collecting anything, which is what
        a generic remote approval dialog does. Rendering that as a successful
        empty form would tell the model the user had nothing to say. It is
        reported as an error instead, so the model falls back to its own
        judgment knowing it never heard from anybody.
        """
        answered = {
            question.question: args.answers[question.question]
            for question in args.questions
            if args.answers.get(question.question)
        }
        unanswered = [q.question for q in args.questions if q.question not in answered]

        if not answered:
            return ToolOutput(
                content=(
                    "No answers came back, so the user has not chosen. Continue "
                    "with your own best judgment and say what you assumed."
                ),
                is_error=True,
                metadata={"answers": {}, "unanswered": unanswered},
            )

        body = "\n".join(f"{text} -> {answer}" for text, answer in answered.items())
        content = f"The user answered:\n{body}"
        if unanswered:
            skipped = "\n".join(unanswered)
            content += f"\n\nLeft unanswered, so decide these yourself:\n{skipped}"

        return ToolOutput(
            content=content,
            metadata={"answers": answered, "unanswered": unanswered},
        )


__all__ = [
    "AskUserQuestionTool",
    "AskUserQuestionInput",
    "Question",
    "QuestionOption",
]
