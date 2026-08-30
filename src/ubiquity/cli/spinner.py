"""The waiting animation.

A spinner is the only thing on screen while a model thinks, so it is worth
drawing rather than borrowing. This one is built from the same two marks the
rest of the CLI uses: the star that heads a thinking block, twinkling through a
set of glyphs, and the brand gradient, sweeping left to right across the label
like a shine crossing a surface. The wait is the same wait either way; the
point is that it looks like this program and not like every other one.

`Pulse` is a pure function of a tick number, so a test can ask for frame 7 and
get the same text the terminal would have drawn. `Working` wraps it in a rich
`Live` and is the only part that touches the clock -- which also means the
elapsed seconds need no ticker task of their own: the animation is already
repainting several times a second, and it reads the clock when it does.
"""

from __future__ import annotations

import random
import time
from typing import Any, Iterator

from rich.live import Live
from rich.text import Text

from . import ui

STARS = ("✶", "✸", "✹", "✺", "✹", "✷")
"""One twinkle, drawn as a star gaining and losing its points."""

WORDS = (
    "thinking",
    "pondering",
    "puzzling",
    "considering",
    "mulling",
    "weighing",
    "reasoning",
    "deliberating",
    "untangling",
    "chewing",
)
"""Labels for the spinner. Which one is showing marks where a turn began."""

INTERRUPT = "ctrl-c to interrupt"

FPS = 12.5

SWEEP = 5
"""How many characters of the label the shine covers at once."""

TRAVEL = 3
"""Ticks the shine takes per character, so it drifts rather than races."""


def word() -> str:
    """Pick the word for one waiting spell."""
    return random.choice(WORDS)


def shimmer(label: str, tick: int) -> Text:
    """Draw `label` with a gradient highlight passing through it."""
    text = Text()
    span = len(label) + SWEEP * 2
    head = (tick // TRAVEL) % span - SWEEP
    for index, character in enumerate(label):
        distance = abs(index - head)
        if distance >= SWEEP:
            text.append(character, style="muted")
        else:
            text.append(character, style=ui.shade(1 - distance / SWEEP))
    return text


class Pulse:
    """One frame of the animation, given a tick and how long the wait has run."""

    def __init__(self, label: str = "", hint: str = INTERRUPT) -> None:
        self.label = label
        self.hint = hint

    def caption(self, elapsed: int = 0) -> str:
        """The text after the star: what is happening, and the way out of it."""
        return f"{self.label}…" + (f" ({elapsed}s · {self.hint})" if elapsed else "")

    def frame(self, tick: int, elapsed: int = 0) -> Text:
        """Render the animation at `tick`, counted in refreshes."""
        text = Text()
        text.append(STARS[tick % len(STARS)] + " ", style=ui.shade((tick % 24) / 23))
        text.append_text(shimmer(self.label, tick))
        tail = self.caption(elapsed)[len(self.label) :]
        text.append(tail, style="muted")
        return text


class Working:
    """A `Pulse` on the terminal, started and stopped like a rich status."""

    def __init__(self, console: Any, label: str = "", hint: str = INTERRUPT) -> None:
        self.pulse = Pulse(label, hint)
        self.started = time.monotonic()
        self._live = Live(
            self, console=console, refresh_per_second=FPS, transient=True
        )

    @property
    def elapsed(self) -> int:
        return int(time.monotonic() - self.started)

    def __rich_console__(self, console: Any, options: Any) -> Iterator[Text]:
        tick = int((time.monotonic() - self.started) * FPS)
        yield self.pulse.frame(tick, self.elapsed)

    def update(self, label: str) -> None:
        """Say something else while the animation keeps running."""
        self.pulse.label = label

    def start(self) -> None:
        self._live.start()

    def stop(self) -> None:
        self._live.stop()
