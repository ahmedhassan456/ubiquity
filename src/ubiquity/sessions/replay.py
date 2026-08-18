"""Rebuilding a model conversation from a stored transcript.

A session file records SDK messages, which is the right shape for a host but
not what a provider expects on the wire. Resuming means translating those
records back into pydantic-ai `ModelMessage` objects so the model sees the
earlier turns as conversation rather than as a summary of one.

Tool calls and their results are paired by id, and anything left unpaired is
dropped: a call whose result is missing — because it was denied, or because
the run died between the two — is rejected by most providers as a dangling
tool use, so replaying it faithfully would make the session unresumable.
"""

from __future__ import annotations

from collections.abc import Sequence
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

from .store import SessionRecord


def _assistant_parts(payload: dict[str, Any]) -> list[Any]:
    """Translate an assistant record's content blocks into response parts.

    Thinking blocks are left out: they are provider-specific, often signed,
    and a replayed one is more likely to be rejected than to help.
    """
    parts: list[Any] = []
    for block in payload.get("content") or ():
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and block.get("text"):
            parts.append(TextPart(content=block["text"]))
        elif block.get("type") == "tool_use":
            parts.append(
                ToolCallPart(
                    tool_name=str(block.get("name", "")),
                    args=block.get("input") or {},
                    tool_call_id=str(block.get("id", "")),
                )
            )
    return parts


def _user_content(payload: dict[str, Any]) -> str:
    """Render a user record's content as text."""
    content = payload.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict)
        )
    return str(content or "")


def _result_content(output: Any) -> str:
    """Render a stored tool result as the text the model originally saw.

    Transcripts are serialized with a string fallback, so a record written by
    an older or unusual writer may hold the rendering rather than the object.
    """
    if isinstance(output, dict):
        return str(output.get("content", ""))
    return str(output or "")


def history_from(records: Sequence[SessionRecord]) -> list[ModelMessage]:
    """Rebuild the model conversation recorded by `records`.

    System and result records are skipped — the first is rebuilt from the
    current `Options` on every run, and the second describes the run rather
    than taking part in the conversation.
    """
    messages: list[ModelMessage] = []
    returns: list[ToolReturnPart] = []

    def flush() -> None:
        """Emit any buffered tool results as one request."""
        if returns:
            messages.append(ModelRequest(parts=list(returns)))
            returns.clear()

    for record in records:
        payload = record.payload or {}
        if record.type == "user":
            flush()
            text = _user_content(payload)
            if text:
                messages.append(ModelRequest(parts=[UserPromptPart(content=text)]))
        elif record.type == "assistant":
            flush()
            parts = _assistant_parts(payload)
            if parts:
                messages.append(ModelResponse(parts=parts))
        elif record.type == "tool_result":
            returns.append(
                ToolReturnPart(
                    tool_name=str(payload.get("tool_name", "")),
                    content=_result_content(payload.get("output")),
                    tool_call_id=str(payload.get("tool_use_id", "")),
                )
            )

    flush()
    return _drop_unpaired(messages)


def _drop_unpaired(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Remove tool calls with no result and results with no call."""
    called = {
        part.tool_call_id
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    }
    returned = {
        part.tool_call_id
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }
    paired = called & returned
    if called == paired and returned == paired:
        return messages

    kept: list[ModelMessage] = []
    for message in messages:
        parts = [
            part
            for part in message.parts
            if not (
                isinstance(part, (ToolCallPart, ToolReturnPart))
                and part.tool_call_id not in paired
            )
        ]
        if not parts:
            continue
        kept.append(
            ModelResponse(parts=parts)
            if isinstance(message, ModelResponse)
            else ModelRequest(parts=parts)
        )
    return kept


__all__ = ["history_from"]
