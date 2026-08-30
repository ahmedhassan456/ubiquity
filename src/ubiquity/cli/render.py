"""Turning the `SDKMessage` stream into terminal output.

Three output formats share one entry point. `text` is the human rendering:
a spinner while the model works, tool calls as one labelled line each, and the
reply as rendered markdown. `stream-json` writes one JSON object per message,
which is what makes the CLI scriptable, and `json` stays quiet until the
terminal `SDKResultMessage`.

Two details are worth knowing. The animation is stopped by anything that prints,
including a permission prompt, because a live region and a cursor waiting for
input cannot share a terminal. And `prompted` is set by the permission handler
so an approved call is announced once rather than twice: the prompt already
showed the tool and its input, and the header would only repeat it.

Reasoning is drawn by `thinking.Stream`: dim indented prose under one header,
streamed as it arrives when the model exposes it. It is chrome rather than
answer, so it is written straight through and never re-rendered, and turning it
off is a matter of not asking the SDK for partial messages at all.

Assistant text is rendered as markdown once the turn is complete rather than
streamed token by token. Markdown cannot be re-flowed after it is printed, so
the choice is between live plain text and formatted output; `--stream` picks
the former for anyone who prefers watching it arrive.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from rich.markdown import Markdown
from rich.text import Text

from ..types import SDKMessage
from . import spinner, thinking, ui

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
        show_thinking: bool = True,
    ) -> None:
        self.output_format = output_format
        self.verbose = verbose
        self.stream_text = stream_text
        self.show_thinking = show_thinking
        self.session_id = ""
        self.prompted: str | None = None
        self._status: spinner.Working | None = None
        self._text_open = False
        self._thinking: thinking.Stream | None = None
        self._streamed = False

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

    def start_status(self, label: str = "") -> None:
        """Show the working animation, if this format has one.

        The animation counts the seconds up beside its label. Waiting on a
        model with no idea how long it has been waiting is the difference
        between a slow run and an apparently hung one, and the elapsed time is
        the cheapest way to tell them apart -- and the rest of the caption says
        which key ends the wait.
        """
        if self.output_format != "text" or self._status is not None:
            return
        self._status = spinner.Working(self.console, label or spinner.word())
        self._status.start()

    def stop_status(self) -> None:
        """Take the animation down before anything else touches the terminal."""
        if self._status is not None:
            self._status.stop()
            self._status = None

    def thinking(self) -> thinking.Stream:
        """The reasoning writer for this run, built on first thought."""
        if self._thinking is None:
            self._thinking = thinking.Stream(self.console)
        return self._thinking

    def close_thinking(self) -> None:
        """End an open reasoning block, wherever the next output came from."""
        if self._thinking is not None:
            self._thinking.close()

    def close_text(self) -> None:
        """End an open streamed line or thought, and clear the spinner."""
        self.stop_status()
        self.close_thinking()
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
        """Route one delta: reasoning to its own block, text to the line."""
        if message.block_type == "thinking":
            if not self.show_thinking:
                return
            self.stop_status()
            self.thinking().write(message.delta)
            self._streamed = True
            return
        if not self.stream_text:
            return
        self.stop_status()
        self.close_thinking()
        self.write(message.delta)
        self._text_open = True

    def _thoughts(self, message: Any) -> None:
        """Print the thinking of a turn that arrived whole rather than in deltas."""
        if not self.show_thinking or self._streamed:
            self._streamed = False
            return
        for block in message.content:
            if block.get("type") == "thinking" and block.get("thinking"):
                self.stop_status()
                self.thinking().feed(block["thinking"])

    def _assistant(self, message: Any) -> None:
        """Print a completed turn as markdown, unless its deltas already went out."""
        self._thoughts(message)
        self.close_thinking()
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
