"""Microcompaction: reclaiming context without a model call.

Full compaction is expensive. It costs an extra model round-trip, it discards
the transcript in favor of a summary, and it puts the run at the mercy of the
summarizer. Most of the context it reclaims, though, is stale tool output —
the contents of a file read forty turns ago, the output of a command whose
result has long since been acted on.

Microcompaction takes that back for free. The content of older tool results is
replaced with a marker in place; the call and its return stay paired, the
transcript keeps its shape, and nothing is summarized. No model is involved,
so there is no failure mode beyond reclaiming less than hoped.

Only tools whose results are pure observation are eligible. A tool that
carries state the model is expected to still be tracking — a todo list, a
subagent's report — is left alone, because clearing it silently rewrites what
the model believes about the task. MCP tools are excluded for the same reason:
their semantics are unknown to this SDK.

The saving is invisible to the provider's own token accounting, which reports
what was already sent rather than what will be sent next. Callers therefore
carry `tokens_saved` forward and subtract it from the next measurement, or
they will compact a conversation that no longer needs it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

from .compaction import CHARS_PER_TOKEN

logger = logging.getLogger("ubiquity")

CLEARED_CONTENT = "[Old tool result content cleared]"

COMPACTABLE_TOOLS = frozenset(
    {
        "Read",
        "Write",
        "Edit",
        "Bash",
        "Glob",
        "Grep",
    }
)

DEFAULT_KEEP_RECENT = 5


@dataclass(slots=True)
class MicrocompactResult:
    """The outcome of one microcompaction pass."""

    messages: list[ModelMessage]
    cleared: int
    kept: int
    tokens_saved: int


def _content_length(content: object) -> int:
    """Return the rendered length of a tool result's content."""
    if isinstance(content, str):
        return len(content)
    return len(repr(content))


def compactable_call_ids(
    messages: list[ModelMessage],
    tools: frozenset[str] | set[str] | None = None,
) -> list[str]:
    """Return the ids of eligible tool calls, oldest first.

    Ordering comes from the calls rather than the returns so that "the most
    recent N" means the same thing whether or not a call has been answered
    yet, and so a provider that batches several returns into one request
    cannot reorder them.
    """
    eligible = COMPACTABLE_TOOLS if tools is None else tools
    ids: list[str] = []
    for message in messages:
        if not isinstance(message, ModelResponse):
            continue
        for part in message.parts:
            if isinstance(part, ToolCallPart) and part.tool_name in eligible:
                ids.append(part.tool_call_id)
    return ids


def microcompact(
    messages: list[ModelMessage],
    *,
    keep_recent: int = DEFAULT_KEEP_RECENT,
    tools: frozenset[str] | set[str] | None = None,
) -> MicrocompactResult | None:
    """Clear the content of all but the most recent eligible tool results.

    Returns None when there is nothing to reclaim, which is the common case
    early in a run. The input list is never modified; callers that want the
    saving must adopt `result.messages`.

    `keep_recent` is floored at one. Clearing every result would leave the
    model with no observation at all to act on, which is worse than the
    context pressure being relieved.
    """
    eligible = COMPACTABLE_TOOLS if tools is None else tools
    order = compactable_call_ids(messages, eligible)
    if not order:
        return None

    keep = set(order[-max(1, keep_recent) :])
    clear = {call_id for call_id in order if call_id not in keep}
    if not clear:
        return None

    saved = 0
    cleared = 0
    rebuilt: list[ModelMessage] = []
    for message in messages:
        if not isinstance(message, ModelRequest):
            rebuilt.append(message)
            continue

        parts = list(message.parts)
        touched = False
        for index, part in enumerate(parts):
            if (
                isinstance(part, ToolReturnPart)
                and part.tool_name in eligible
                and part.tool_call_id in clear
                and part.content != CLEARED_CONTENT
            ):
                saved += _content_length(part.content)
                cleared += 1
                parts[index] = replace(part, content=CLEARED_CONTENT)
                touched = True
        rebuilt.append(replace(message, parts=parts) if touched else message)

    if not cleared:
        return None

    tokens_saved = max(0, (saved - cleared * len(CLEARED_CONTENT)) // CHARS_PER_TOKEN)
    logger.debug(
        "microcompacted %d tool results (~%d tokens), kept %d",
        cleared,
        tokens_saved,
        len(keep),
    )
    return MicrocompactResult(
        messages=rebuilt,
        cleared=cleared,
        kept=len(keep),
        tokens_saved=tokens_saved,
    )


__all__ = [
    "MicrocompactResult",
    "microcompact",
    "compactable_call_ids",
    "COMPACTABLE_TOOLS",
    "CLEARED_CONTENT",
    "DEFAULT_KEEP_RECENT",
]
