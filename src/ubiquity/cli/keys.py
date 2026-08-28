"""Reading single keypresses from the terminal.

An arrow-key menu needs the key the moment it is pressed, which line-buffered
input cannot give: `input()` returns nothing until Enter. So the terminal is
put into raw mode for the duration of one keystroke and restored immediately
afterwards, rather than for the life of the menu -- a raw terminal that outlives
an exception is a shell the user has to reset by hand.

Escape sequences arrive as several bytes with no length prefix. The reader
takes the first byte, then drains whatever is already waiting behind it, which
distinguishes a real Escape keypress from the start of an arrow key without
waiting on a timer.

Bytes are read from the file descriptor with `os.read` rather than through the
stream object. `sys.stdin` is buffered, and it will happily pull an entire
escape sequence into its own buffer on a one-character read -- after which the
descriptor looks empty, every arrow key reads as a bare Escape, and the rest of
the sequence surfaces later as stray keystrokes.
"""

from __future__ import annotations

import sys
from typing import TextIO

UP = "up"
DOWN = "down"
LEFT = "left"
RIGHT = "right"
ENTER = "enter"
SPACE = "space"
ESCAPE = "escape"
BACKSPACE = "backspace"
INTERRUPT = "interrupt"
EOF = "eof"

_SEQUENCES = {
    "[A": UP,
    "[B": DOWN,
    "[C": RIGHT,
    "[D": LEFT,
    "OA": UP,
    "OB": DOWN,
    "OC": RIGHT,
    "OD": LEFT,
}


def supported(stream: TextIO | None = None) -> bool:
    """True when this terminal can be read one keypress at a time."""
    stream = stream or sys.stdin
    try:
        import termios  # noqa: F401
    except ImportError:
        return False
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


def read_key(stream: TextIO | None = None) -> str:
    """Block until one key is pressed and return its name or character."""
    import os
    import termios
    import tty

    stream = stream or sys.stdin
    fd = stream.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        first = os.read(fd, 1).decode(errors="replace")
        if not first:
            return EOF
        if first == "\x1b":
            rest = _drain(fd)
            return _SEQUENCES.get(rest, ESCAPE)
        return _name(first)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _drain(fd: int) -> str:
    """Read whatever is already waiting after an escape byte."""
    import os
    import select

    out = ""
    while len(out) < 2 and select.select([fd], [], [], 0.02)[0]:
        chunk = os.read(fd, 1).decode(errors="replace")
        if not chunk:
            break
        out += chunk
    return out


def _name(char: str) -> str:
    """Map a control character to its name, or pass the character through."""
    if char in ("\r", "\n"):
        return ENTER
    if char == " ":
        return SPACE
    if char in ("\x7f", "\b"):
        return BACKSPACE
    if char == "\x03":
        return INTERRUPT
    if char == "\x04":
        return EOF
    return char


__all__ = [
    "read_key",
    "supported",
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "ENTER",
    "SPACE",
    "ESCAPE",
    "BACKSPACE",
    "INTERRUPT",
    "EOF",
]
