"""Turning the `SDKMessage` stream into terminal output.

Three output formats share one entry point. `text` is the human rendering:
assistant text streams in as it arrives, tool calls get a one-line header, and
the run ends with a footer carrying turns, tokens, and cost. `stream-json`
writes one JSON object per message, which is what makes the CLI scriptable.
`json` stays silent until the terminal `SDKResultMessage` and prints that.

`prompted` is set by the permission handler so an approved call is announced
once rather than twice: the prompt already showed the tool and its input, and
the header that normally introduces the call would only repeat it.

The renderer tracks whether a text block is open so it can close it before
anything else is written. Without that, a tool header lands mid-sentence: the
deltas of an assistant turn arrive first and the turn's own message only
follows once the model has finished speaking.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from typing import Any, TextIO

from ..types import SDKMessage

OutputFormat = ("text", "json", "stream-json")

_STYLES = {
    "dim": "\033[2m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "reset": "\033[0m",
}


def paint(text: str, *styles: str, enabled: bool = True) -> str:
    """Wrap `text` in ANSI styles, or return it untouched when disabled."""
    if not enabled or not styles:
        return text
    prefix = "".join(_STYLES.get(style, "") for style in styles)
    return f"{prefix}{text}{_STYLES['reset']}"


def as_payload(message: SDKMessage) -> dict[str, Any]:
    """Convert a message to a JSON-serializable dict."""
    return dataclasses.asdict(message)


def _summarize(tool_input: dict[str, Any]) -> str:
    """Render tool input as a short single line.

    The model-facing input can be a whole file's contents, so every value is
    truncated and newlines are folded away. This is a header, not a transcript.
    """
    parts = []
    for key, value in tool_input.items():
        text = str(value).replace("\n", " ")
        if len(text) > 60:
            text = text[:57] + "..."
        parts.append(f"{key}={text}")
    joined = ", ".join(parts)
    return joined if len(joined) <= 120 else joined[:117] + "..."


def _usage_line(message: Any) -> str:
    """Format the result footer: turns, duration, tokens, and cost."""
    bits = [f"{message.num_turns} turns", f"{message.duration_ms / 1000:.1f}s"]
    usage = message.usage or {}
    tokens = usage.get("total_tokens") or usage.get("input_tokens")
    if tokens:
        bits.append(f"{tokens} tokens")
    if message.total_cost_usd:
        bits.append(f"${message.total_cost_usd:.4f}")
    return " · ".join(bits)


class Renderer:
    """Writes one run's messages to a stream in the chosen format."""

    def __init__(
        self,
        output_format: str = "text",
        *,
        color: bool = True,
        verbose: bool = False,
        stream: TextIO | None = None,
    ) -> None:
        self.output_format = output_format
        self.color = color
        self.verbose = verbose
        self.out = stream or sys.stdout
        self.session_id = ""
        self.prompted: str | None = None
        self._text_open = False

    def write(self, text: str) -> None:
        """Write raw text to the renderer's stream, flushing as it goes."""
        self.out.write(text)
        self.out.flush()

    def _paint(self, text: str, *styles: str) -> str:
        return paint(text, *styles, enabled=self.color)

    def close_text(self) -> None:
        """End an open assistant text block, so the next line starts clean."""
        if self._text_open:
            self.write("\n")
            self._text_open = False

    def note(self, text: str, *styles: str) -> None:
        """Write a line of CLI chrome, outside the message stream."""
        if self.output_format != "text":
            return
        self.close_text()
        self.write(self._paint(text, *styles) + "\n")

    def handle(self, message: SDKMessage) -> None:
        """Render one message according to the active output format."""
        if getattr(message, "session_id", ""):
            self.session_id = message.session_id
        if self.output_format == "stream-json":
            self.write(json.dumps(as_payload(message), default=str) + "\n")
            return
        if self.output_format == "json":
            if message.type == "result":
                self.write(json.dumps(as_payload(message), default=str) + "\n")
            return
        self._text(message)

    def _text(self, message: SDKMessage) -> None:
        kind = message.type
        subtype = getattr(message, "subtype", None)
        if kind == "system" and subtype == "init":
            self._init(message)
        elif kind == "system" and subtype == "compact_boundary":
            self.note(
                f"  compacted at {message.pre_tokens} tokens ({message.trigger})",
                "dim",
            )
        elif kind == "system" and subtype == "microcompact":
            self.note(
                f"  cleared {message.cleared} tool results "
                f"(~{message.tokens_saved} tokens)",
                "dim",
            )
        elif kind == "stream_event":
            self._delta(message)
        elif kind == "assistant":
            self._assistant(message)
        elif kind == "tool_use":
            self._tool_use(message)
        elif kind == "tool_result":
            self._tool_result(message)
        elif kind == "result":
            self._result(message)

    def _init(self, message: Any) -> None:
        if not self.verbose:
            return
        self.note(f"  model     {message.model}", "dim")
        self.note(f"  cwd       {message.cwd}", "dim")
        self.note(f"  mode      {message.permission_mode}", "dim")
        self.note(f"  tools     {len(message.tools)}", "dim")
        if message.agents:
            self.note(f"  agents    {', '.join(message.agents)}", "dim")

    def _delta(self, message: Any) -> None:
        if message.block_type == "thinking":
            if self.verbose:
                self.write(self._paint(message.delta, "dim"))
                self._text_open = True
            return
        self.write(message.delta)
        self._text_open = True

    def _assistant(self, message: Any) -> None:
        """Print a completed turn, unless its deltas already went out."""
        if self._text_open:
            self.close_text()
            return
        text = message.text
        if text.strip():
            self.write(text.rstrip() + "\n")

    def _tool_use(self, message: Any) -> None:
        """Announce a call, unless a permission prompt just showed the same one."""
        self.close_text()
        if self.prompted == message.tool_name:
            self.prompted = None
            return
        head = self._paint(f"● {message.tool_name}", "cyan", "bold")
        self.write(f"{head} {self._paint(_summarize(message.tool_input), 'dim')}\n")

    def _tool_result(self, message: Any) -> None:
        self.close_text()
        output = message.output
        if output.is_error:
            body = (output.display or output.content).strip().splitlines()
            first = body[0] if body else "failed"
            self.write("  " + self._paint(f"✗ {first}", "red") + "\n")
            return
        if not self.verbose:
            return
        body = (output.display or output.content).strip().splitlines()
        for line in body[:6]:
            self.write("  " + self._paint(line, "dim") + "\n")
        if len(body) > 6:
            self.write("  " + self._paint(f"… +{len(body) - 6} lines", "dim") + "\n")

    def _result(self, message: Any) -> None:
        self.close_text()
        for denial in message.permission_denials:
            self.note(f"  denied {denial.tool_name}: {denial.message}", "yellow")
        if message.is_error:
            self.note(f"✗ {message.result}", "red")
        if self.verbose or message.is_error:
            self.note(f"  {_usage_line(message)}", "dim")
