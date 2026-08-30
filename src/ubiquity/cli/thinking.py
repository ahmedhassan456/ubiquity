"""Rendering a model's reasoning as it arrives.

Thinking is not the answer, and it should not look like one. It arrives as a
stream of tokens with no markdown worth rendering and no structure beyond the
occasional summary heading, so it is drawn as dim italic prose under one
header, indented far enough that the eye can skip the whole block.

The wrapping is done here rather than left to rich because the text arrives in
fragments: a console asked to wrap `"abou"` and then `"t"` has already
committed to a line break in the middle of a word. `Stream` therefore holds a
partial word until a space proves it complete, tracks the column itself, and
breaks where a word will not fit.

A line that starts as a markdown heading -- `**Checking the config**`, which is
how several models introduce a summary -- is held back until the newline
arrives, so it can be printed bold and undecorated on a line of its own rather
than word by word with its asterisks showing.
"""

from __future__ import annotations

import re
from typing import Any, Iterator

from rich.text import Text

HEADER = "✻ Thinking…"

HEADING = re.compile(r"^\s*(#{1,6}\s+\S|\*\*\S)")
"""What a line looks like when it opens with a heading rather than prose."""

DECORATION = re.compile(r"^\s*#{0,6}\s*|\*+|_+|:\s*$")

INDENT = "  "

MIN_WIDTH = 24


def undecorate(line: str) -> str:
    """Strip the markdown a heading is wrapped in."""
    return DECORATION.sub("", line).strip()


def split_lines(text: str) -> Iterator[tuple[str, bool]]:
    """Yield `(chunk, ended)` pairs, one per line the text touches."""
    parts = text.split("\n")
    for part in parts[:-1]:
        yield part, True
    yield parts[-1], False


class Stream:
    """Writes thinking text to a console, wrapped, indented, and styled."""

    def __init__(self, console: Any, indent: str = INDENT) -> None:
        self.console = console
        self.indent = indent
        self.open = False
        self._pending = ""
        self._column = 0
        self._on_line = False
        self._blank = False

    @property
    def width(self) -> int:
        """The column a word may not cross."""
        return max(self.console.width - len(self.indent) - 1, MIN_WIDTH)

    def write(self, delta: str) -> None:
        """Add a fragment of thinking, printing whatever it completes."""
        self._begin()
        for chunk, ended in split_lines(delta):
            self._pending += chunk
            if ended:
                self._flush(final=True)
                self._break()
            else:
                self._flush(final=False)

    def feed(self, text: str) -> None:
        """Write a whole thought that was never streamed, then close it."""
        if text.strip():
            self.write(text)
            self.close()

    def close(self) -> None:
        """Finish an open block, leaving the cursor at column zero."""
        if not self.open:
            return
        self._flush(final=True)
        if self._on_line:
            self.console.print()
        self.console.print()
        self.open = False
        self._on_line = False
        self._column = 0
        self._blank = False
        self._pending = ""

    def _begin(self) -> None:
        if self.open:
            return
        self.console.print()
        self.console.print(Text(HEADER, style="thinking.head"))
        self.open = True

    def _flush(self, final: bool) -> None:
        """Print the words the buffer has proven complete."""
        heading = not self._on_line and HEADING.match(self._pending) is not None
        if heading and not final:
            return
        if final:
            body, self._pending = self._pending, ""
        else:
            body, _, self._pending = self._pending.rpartition(" ")
            if not body:
                return
        style = "thinking.title" if heading else "thinking"
        for word in (undecorate(body) if heading else body).split():
            self._word(word, style)

    def _word(self, word: str, style: str) -> None:
        """Place one word, breaking the line first if it will not fit."""
        if self._column and self._column + 1 + len(word) > self.width:
            self._break()
        if self._blank:
            self.console.print()
            self._blank = False
        lead = self.indent if not self._on_line else " " * bool(self._column)
        self.console.print(Text(lead + word, style=style), end="", soft_wrap=True)
        self._column += len(lead) + len(word)
        self._on_line = True

    def _break(self) -> None:
        """End the current line, or remember that a blank one was asked for."""
        if self._on_line:
            self.console.print()
            self._on_line = False
            self._column = 0
        else:
            self._blank = self.open
