"""Turning the `SDKMessage` stream into terminal output.

Three output formats share one entry point. `text` is the human rendering:
a spinner while the model works, tool calls as one labelled line each, and the
reply as rendered markdown. `stream-json` writes one JSON object per message,
which is what makes the CLI scriptable, and `json` stays quiet until the
terminal `SDKResultMessage`.

Two details are worth knowing. The spinner is stopped by anything that prints,
including a permission prompt, because a live region and a cursor waiting for
input cannot share a terminal. And `prompted` is set by the permission handler
so an approved call is announced once rather than twice: the prompt already
showed the tool and its input, and the header would only repeat it.

Assistant text is rendered as markdown once the turn is complete rather than
streamed token by token. Markdown cannot be re-flowed after it is printed, so
the choice is between live plain text and formatted output; `--stream` picks
the former for anyone who prefers watching it arrive.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from typing import Any

from rich.markdown import Markdown
from rich.text import Text

from ..types import SDKMessage
from . import ui

OUTPUT_FORMATS = ("text", "json", "stream-json")

_GLYPH = {
    "Read": "📖",
    "Write": "✎",
    "Edit": "✎",
    "Bash": "❯_",
    "Glob": "🔎",
    "Grep": "🔎",
    "TodoWrite": "☑",
    "Agent": "⚑",
    "Skill": "★",
}
"""A mark per tool, so a long transcript can be skimmed by shape."""

_TOOL_ARG = {
    "Read": "file_path",
    "Write": "file_path",
    "Edit": "file_path",
    "Bash": "command",
    "Glob": "pattern",
    "Grep": "pattern",
    "TodoWrite": "todos",
    "Agent": "description",
    "Skill": "skill",
}
"""The one input worth showing per built-in tool, when it has one."""


def as_payload(message: SDKMessage) -> dict[str, Any]:
    """Convert a message to a JSON-serializable dict."""
    return dataclasses.asdict(message)


def _summarize(tool_input: dict[str, Any], tool_name: str = "") -> str:
    """Render tool input as one short line.

    Tools with an obvious subject show only that: a `Write` is identified by
    its path, and printing the file's whole contents next to it would bury the
    thing the user is being asked to approve.
    """
    key = _TOOL_ARG.get(tool_name)
    if key and key in tool_input:
        return _clip(str(tool_input[key]), 110)
    parts = [f"{k}={_clip(str(v), 60)}" for k, v in tool_input.items()]
    return _clip(", ".join(parts), 110)


def _clip(text: str, width: int) -> str:
    """Fold a value onto one line, truncating in the middle of long paths."""
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    if "/" in text and len(text) - text.rfind("/") < width - 4:
        tail = text[text.rfind("/") :]
        return text[: width - len(tail) - 1] + "…" + tail
    return text[: width - 1] + "…"


def _usage_line(message: Any) -> str:
    """Format the result footer: turns, duration, tokens, and cost."""
    bits = [f"{message.num_turns} turns", f"{message.duration_ms / 1000:.1f}s"]
    usage = message.usage or {}
    tokens = usage.get("total_tokens") or usage.get("input_tokens")
    if tokens:
        bits.append(f"{tokens:,} tokens")
    if message.total_cost_usd:
        bits.append(f"${message.total_cost_usd:.4f}")
    return " · ".join(bits)


class Renderer:
    """Writes one run's messages to the shared console in the chosen format."""

    def __init__(
        self,
        output_format: str = "text",
        *,
        verbose: bool = False,
        stream_text: bool = False,
    ) -> None:
        self.output_format = output_format
        self.verbose = verbose
        self.stream_text = stream_text
        self.session_id = ""
        self.prompted: str | None = None
        self._status: Any = None
        self._ticker: Any = None
        self._label = "thinking"
        self._started = 0.0
        self._text_open = False

    @property
    def console(self) -> Any:
        return ui.console()

    @property
    def color(self) -> bool:
        return not self.console.no_color

    def write(self, text: str) -> None:
        """Write raw text, bypassing markup and wrapping."""
        self.console.file.write(text)
        self.console.file.flush()

    def start_status(self, label: str = "thinking") -> None:
        """Show the working spinner, if this format has one.

        A ticker task counts the seconds up beside it. Waiting on a model with
        no idea how long it has been waiting is the difference between a slow
        run and an apparently hung one, and the elapsed time is the cheapest
        way to tell them apart.
        """
        if self.output_format != "text" or self._status is not None:
            return
        self._status = self.console.status(f"[muted]{label}…[/muted]", spinner="dots")
        self._status.start()
        self._label = label
        self._started = time.monotonic()
        try:
            self._ticker = asyncio.get_running_loop().create_task(self._tick())
        except RuntimeError:
            self._ticker = None

    async def _tick(self) -> None:
        """Refresh the spinner's label once a second while it is up."""
        try:
            while self._status is not None:
                await asyncio.sleep(1.0)
                if self._status is None:
                    return
                elapsed = int(time.monotonic() - self._started)
                if elapsed:
                    self._status.update(f"[muted]{self._label}… {elapsed}s[/muted]")
        except asyncio.CancelledError:
            pass

    def stop_status(self) -> None:
        """Take the spinner down before anything else touches the terminal."""
        if self._ticker is not None:
            self._ticker.cancel()
            self._ticker = None
        if self._status is not None:
            self._status.stop()
            self._status = None

    def close_text(self) -> None:
        """End an open streamed line, and clear the spinner."""
        self.stop_status()
        if self._text_open:
            self.write("\n")
            self._text_open = False

    def note(self, message: str, style: str = "muted") -> None:
        """Write a line of CLI chrome, outside the message stream."""
        if self.output_format != "text":
            return
        self.close_text()
        ui.note(message, style)

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
                f"  ⤺ compacted at {message.pre_tokens:,} tokens ({message.trigger})"
            )
        elif kind == "system" and subtype == "microcompact":
            self.note(
                f"  ⤺ cleared {message.cleared} tool results "
                f"(~{message.tokens_saved:,} tokens)"
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
        self.start_status()
        if not self.verbose:
            return
        self.stop_status()
        ui.key_values(
            [
                ("model", message.model),
                ("cwd", message.cwd),
                ("mode", message.permission_mode),
                ("tools", str(len(message.tools))),
                *([("agents", ", ".join(message.agents))] if message.agents else []),
            ],
            title="run",
        )
        self.start_status()

    def _delta(self, message: Any) -> None:
        if message.block_type == "thinking":
            if self.verbose:
                self.stop_status()
                self.console.print(Text(message.delta, style="muted"), end="")
                self._text_open = True
            return
        self.stop_status()
        self.write(message.delta)
        self._text_open = True

    def _assistant(self, message: Any) -> None:
        """Print a completed turn as markdown, unless its deltas already went out."""
        if self._text_open:
            self.close_text()
            return
        self.stop_status()
        text = message.text.strip()
        if text:
            self.console.print(Markdown(text))
            self.console.print()
        self.start_status()

    def _tool_use(self, message: Any) -> None:
        self.close_text()
        if self.prompted == message.tool_name:
            self.prompted = None
            self.start_status("working")
            return
        glyph = _GLYPH.get(message.tool_name, "●")
        line = Text(f"  {glyph} ", style="tool")
        line.append(message.tool_name, style="tool")
        summary = _summarize(message.tool_input, message.tool_name)
        if summary:
            line.append(f"  {summary}", style="tool.args")
        self.console.print(line)
        self.start_status("working")

    def _tool_result(self, message: Any) -> None:
        self.close_text()
        output = message.output
        body = (output.display or output.content).strip().splitlines()
        if output.is_error:
            first = body[0] if body else "failed"
            self.console.print(Text(f"    ✗ {_clip(first, 100)}", style="bad"))
        elif self.verbose:
            for line in body[:8]:
                self.console.print(Text(f"    {_clip(line, 110)}", style="muted"))
            if len(body) > 8:
                self.console.print(
                    Text(f"    … +{len(body) - 8} lines", style="muted")
                )
        self.start_status()

    def _result(self, message: Any) -> None:
        self.close_text()
        for denial in message.permission_denials:
            self.note(f"  denied {denial.tool_name}: {denial.message}", "warn")
        if message.is_error:
            self.note(f"  ✗ {message.result}", "bad")
        if self.verbose or message.is_error:
            self.note(f"  {_usage_line(message)}")
