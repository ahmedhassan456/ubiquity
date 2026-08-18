"""Conversation compaction.

A long agent run eventually outgrows its model's context window. Compaction
replaces the older part of the message history with a model-written summary,
so the run can continue with the same goal in view but a fraction of the
tokens.

Nothing here is provider-specific. Pressure is measured from the token counts
that pydantic-ai already collects from whichever backend answered, and the
summary is produced by an ordinary model call, so the mechanism works the same
against a hosted frontier model and a local one.

The trigger is a token reserve rather than a percentage of the window, because
a percentage does not survive the range of window sizes this SDK has to cover.
The window itself is never guessed from the model name by default; see
`register_context_window`.

The compaction boundary is chosen carefully. History is cut immediately before
a `ModelResponse`, never before a `ModelRequest`, because a request may carry
`ToolReturnPart`s answering tool calls made in the response above it. Cutting
between those two would leave tool results with no matching call, which most
providers reject outright. Cutting before a response keeps every call/return
pair whole.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
)

if TYPE_CHECKING:
    from pydantic_ai.models import Model

logger = logging.getLogger("ubiquity")

DEFAULT_CONTEXT_WINDOW = 128_000
CONTEXT_WINDOW_ENV_VAR = "UBIQUITY_MAX_CONTEXT_TOKENS"
CHARS_PER_TOKEN = 4
MAX_RENDERED_PART_CHARS = 2_000

OUTPUT_RESERVE_TOKENS = 20_000
HEADROOM_TOKENS = 13_000
MIN_THRESHOLD_FRACTION = 0.5
MAX_CONSECUTIVE_FAILURES = 3

CONTEXT_WINDOWS: dict[str, int] = {}

COMPACTION_PROMPT = """\
You are compacting a software engineering agent's conversation so it can keep \
working with a smaller context. Write a summary that a fresh instance of the \
agent could read and continue from without asking the user to repeat anything.

Cover, in this order and only where the transcript supports it:

1. What the user asked for, in their own terms, including any constraints or \
preferences they stated.
2. What has been done so far, and what the outcome was.
3. Files that were read, created, or modified, with their paths, and what \
changed in each.
4. Technical facts established along the way that would be expensive to \
rediscover: signatures, schemas, commands that worked, error messages, \
version numbers, decisions and their reasoning.
5. Anything that failed or was tried and rejected, so it is not retried.
6. What remains to be done, and the immediate next step.

Be specific. Preserve exact identifiers, paths, and commands verbatim rather \
than describing them. Do not add information that is not in the transcript, \
and do not offer to help; produce only the summary.\
"""

SUMMARY_TEMPLATE = """\
The conversation so far has been compacted to fit the context window. This \
summary replaces the earlier messages; the most recent exchanges follow it \
unchanged.

<summary>
{summary}
</summary>

Continue from here.\
"""


@dataclass(slots=True)
class CompactionResult:
    """The outcome of one compaction pass."""

    messages: list[ModelMessage]
    summary: str
    pre_tokens: int
    dropped: int
    kept: int
    trigger: str = "auto"
    metadata: dict[str, Any] = field(default_factory=dict)


def register_context_window(pattern: str, tokens: int) -> None:
    """Declare the context window for models whose name contains `pattern`.

    The registry ships empty on purpose. Shipping a table of per-model window
    sizes would mean asserting numbers for hundreds of models across every
    provider, with no way to keep them true as models are released and
    revised; a stale entry that overstates a window causes exactly the hard
    failure compaction exists to prevent. Declare the models you actually use,
    or set `Options.max_context_tokens` per run.
    """
    CONTEXT_WINDOWS[pattern.lower()] = tokens


def default_context_window() -> int:
    """Return the fallback window, honoring `UBIQUITY_MAX_CONTEXT_TOKENS`."""
    raw = os.environ.get(CONTEXT_WINDOW_ENV_VAR, "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_CONTEXT_WINDOW


def infer_context_window(model_name: str) -> int:
    """Return the context window to assume for `model_name`, in tokens.

    Consults the registry by substring, longest pattern first so a specific
    entry beats a general one, then falls back to a single conservative
    default. The default deliberately errs low: underestimating costs one
    unnecessary compaction, while overestimating means the provider rejects
    the request outright.
    """
    lowered = model_name.lower()
    matches = [key for key in CONTEXT_WINDOWS if key in lowered]
    if not matches:
        return default_context_window()
    return CONTEXT_WINDOWS[max(matches, key=len)]


def compaction_threshold(
    context_window: int,
    *,
    fraction: float | None = None,
    output_reserve: int | None = None,
) -> int:
    """Return the token count at which a run should compact.

    By default the threshold is the window minus a reserve for the summary
    the compaction itself must generate, minus headroom for the request that
    follows. A fixed reserve rather than a fixed percentage is what keeps the
    trigger sane across a two-order-of-magnitude range of window sizes: 80% of
    a one-million-token window wastes 200k tokens of usable context, while 80%
    of a small local window may not leave room to write the summary at all.

    `output_reserve` caps that reserve at what the model can actually emit,
    which matters when the cap is small: reserving 20k for a summary from a
    model that can only produce 4k tokens throws away 16k of usable context on
    every turn. It only ever lowers the reserve, never raises it.

    `fraction` overrides the whole calculation with a plain proportion of the
    window. A floor of half the window applies to the default path so a small
    window cannot produce a threshold at or below zero.
    """
    if fraction is not None:
        return int(context_window * fraction)
    reserve = OUTPUT_RESERVE_TOKENS
    if output_reserve is not None and output_reserve > 0:
        reserve = min(output_reserve, OUTPUT_RESERVE_TOKENS)
    reserved = context_window - reserve - HEADROOM_TOKENS
    return max(reserved, int(context_window * MIN_THRESHOLD_FRACTION))


def usage_tokens(usage: Any) -> int:
    """Return the context occupied according to a pydantic-ai usage record.

    Cached tokens are counted. Some providers report a cache hit outside
    `input_tokens`, so summing only the uncached fields would understate a
    long conversation badly enough to miss the compaction threshold entirely.
    """
    if usage is None:
        return 0
    return sum(
        int(getattr(usage, name, 0) or 0)
        for name in (
            "input_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "output_tokens",
        )
    )


def estimate_tokens(messages: list[ModelMessage]) -> int:
    """Estimate the token cost of `messages` from their rendered length.

    A fallback for backends that report no usage at all, which is common for
    self-hosted OpenAI-compatible servers. Deliberately crude.
    """
    return len(render_transcript(messages, truncate=False)) // CHARS_PER_TOKEN


def measure_context(messages: list[ModelMessage]) -> int:
    """Return the best available measure of how much context `messages` use.

    Prefers the provider's own accounting from the most recent response that
    carries any, since that reflects the real tokenizer. The scan continues
    past responses that report nothing rather than giving up at the first one,
    because a retry or a synthetic response can leave an empty usage record in
    front of perfectly good numbers.

    Anything after that response is estimated and added on. The check runs
    between a response and the request that answers it, so the tool results
    that just landed are always in that tail — and a single large file read or
    command output is exactly the event that pushes a run over the limit. A
    measurement that stopped at the last usage record would be blind to the
    tokens it exists to catch.
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, ModelResponse) and (measured := usage_tokens(message.usage)):
            return measured + estimate_tokens(messages[index + 1 :])
    return estimate_tokens(messages)


def should_compact(
    messages: list[ModelMessage],
    *,
    context_window: int,
    fraction: float | None = None,
    output_reserve: int | None = None,
    freed: int = 0,
) -> tuple[bool, int]:
    """Return whether to compact, and the token measurement behind it.

    `freed` discounts context already reclaimed since the last response, which
    the provider's accounting cannot see: usage reports what was sent, not
    what will be sent next. Microcompaction shrinks the history without any
    round-trip, so without this the very next check would still read the
    pre-microcompaction number and compact a conversation that no longer
    needs it.
    """
    used = max(0, measure_context(messages) - freed)
    threshold = compaction_threshold(
        context_window, fraction=fraction, output_reserve=output_reserve
    )
    return used >= threshold, used


def compaction_boundary(messages: list[ModelMessage], keep_recent: int) -> int | None:
    """Return the index to cut at, or None when no safe cut exists.

    The cut aims to retain roughly `keep_recent` messages, then moves forward
    to the next `ModelResponse` so the retained tail can never open with tool
    results whose calls were just discarded. Returns None when that search
    finds nothing worth dropping, which is the correct outcome for a history
    that is short or entirely one exchange.
    """
    target = max(1, len(messages) - keep_recent)
    for index in range(target, len(messages)):
        if isinstance(messages[index], ModelResponse):
            return index if index >= 2 else None
    return None


def _render_part(part: Any) -> str:
    """Render a single message part as plain text for the summarizer."""
    kind = type(part).__name__
    if (content := getattr(part, "content", None)) is not None:
        body = content if isinstance(content, str) else repr(content)
    elif (args := getattr(part, "args", None)) is not None:
        body = args if isinstance(args, str) else repr(args)
    else:
        body = ""

    name = getattr(part, "tool_name", None)
    label = f"{kind}({name})" if name else kind
    return f"{label}: {body}"


def render_transcript(messages: list[ModelMessage], *, truncate: bool = True) -> str:
    """Render messages as plain text for the summarizing model.

    The transcript is passed as a single prompt rather than as real message
    history, which keeps the summarizer free of the agent's tool definitions
    and sidesteps the role-alternation rules that differ between providers.
    Long tool outputs are truncated in the middle, keeping both the start and
    the end, because the tail of a command's output is usually where its
    verdict is.
    """
    lines: list[str] = []
    for message in messages:
        role = "assistant" if isinstance(message, ModelResponse) else "user"
        for part in message.parts:
            text = _render_part(part)
            if truncate and len(text) > MAX_RENDERED_PART_CHARS:
                half = MAX_RENDERED_PART_CHARS // 2
                omitted = len(text) - MAX_RENDERED_PART_CHARS
                text = f"{text[:half]}\n... [{omitted} characters omitted] ...\n{text[-half:]}"
            lines.append(f"[{role}] {text}")
    return "\n".join(lines)


def build_summary_message(summary: str) -> ModelRequest:
    """Wrap a summary in the user-role message that replaces the dropped history."""
    return ModelRequest(parts=[UserPromptPart(content=SUMMARY_TEMPLATE.format(summary=summary))])


async def summarize(
    messages: list[ModelMessage],
    model: str | Model,
    *,
    instructions: str | None = None,
    aliases: dict[str, str] | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    meter: Any = None,
) -> str:
    """Summarize `messages` with a toolless model call.

    Raises whatever the underlying model raises; callers decide whether a
    failed compaction should abort the run or leave the history untouched.

    `meter` is an optional `pricing.CostMeter`. Summarizing is a real request
    against a real model, often a different and cheaper one than the run's, so
    a run that reported its cost without it would omit a charge it had made.
    """
    from pydantic_ai import Agent

    from .models import resolve_model

    agent = Agent(
        resolve_model(model, aliases, provider_kwargs),
        instructions=instructions or COMPACTION_PROMPT,
    )
    result = await agent.run(render_transcript(messages))
    if meter is not None:
        response = result.all_messages()[-1]
        meter.add(
            getattr(response, "model_name", None),
            getattr(response, "provider_name", None),
            getattr(response, "usage", None),
        )
    return str(result.output).strip()


async def compact(
    messages: list[ModelMessage],
    model: str | Model,
    *,
    keep_recent: int = 6,
    instructions: str | None = None,
    aliases: dict[str, str] | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    trigger: str = "auto",
    meter: Any = None,
) -> CompactionResult | None:
    """Compact `messages`, returning the new history, or None if not possible.

    None means the history could not be safely divided — too short, or with no
    `ModelResponse` late enough to cut before. That is a normal outcome and
    leaves the caller's history untouched.
    """
    pre_tokens = measure_context(messages)
    boundary = compaction_boundary(messages, keep_recent)
    if boundary is None:
        return None

    dropped = messages[:boundary]
    kept = messages[boundary:]
    summary = await summarize(
        dropped,
        model,
        instructions=instructions,
        aliases=aliases,
        provider_kwargs=provider_kwargs,
        meter=meter,
    )

    return CompactionResult(
        messages=[build_summary_message(summary), *kept],
        summary=summary,
        pre_tokens=pre_tokens,
        dropped=len(dropped),
        kept=len(kept),
        trigger=trigger,
    )


__all__ = [
    "CompactionResult",
    "compact",
    "summarize",
    "should_compact",
    "measure_context",
    "estimate_tokens",
    "usage_tokens",
    "compaction_boundary",
    "compaction_threshold",
    "infer_context_window",
    "register_context_window",
    "default_context_window",
    "MAX_CONSECUTIVE_FAILURES",
    "CONTEXT_WINDOW_ENV_VAR",
    "render_transcript",
    "build_summary_message",
    "COMPACTION_PROMPT",
    "CONTEXT_WINDOWS",
    "DEFAULT_CONTEXT_WINDOW",
]
