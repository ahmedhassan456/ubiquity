"""Shared terminal furniture: one console, one theme, one set of prompts.

Everything the CLI draws goes through the `Console` built here, so output
ordering is never in question -- a spinner, a permission prompt, and a tool
header all contend for the same cursor, and rich only guarantees they behave
when they share a console.

`Console` is created lazily rather than at import, because constructing one
probes the terminal, and importing `ubiquity.cli` should not depend on there
being a terminal to probe.
"""

from __future__ import annotations

import asyncio
from typing import Any, Sequence, TextIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from . import keys

THEME = Theme(
    {
        "brand": "bold magenta",
        "tool": "bold cyan",
        "tool.args": "dim",
        "ok": "green",
        "warn": "yellow",
        "bad": "bold red",
        "muted": "dim",
        "rule": "grey37",
        "prompt.head": "bold yellow",
        "banner": "magenta",
        "key": "bold",
    }
)

_console: Console | None = None


def console(**kwargs: Any) -> Console:
    """Return the shared console, building it on first use."""
    global _console
    if _console is None:
        _console = Console(theme=THEME, **kwargs)
    return _console


def set_console(
    *, file: TextIO | None = None, color: bool = True, width: int | None = None
) -> Console:
    """Replace the shared console, which is how tests capture output."""
    global _console
    _console = Console(
        theme=THEME,
        file=file,
        no_color=not color,
        force_terminal=False if file is not None else None,
        width=width,
        highlight=False,
        soft_wrap=file is not None,
    )
    return _console


WORDMARK = """\
██╗   ██╗██████╗ ██╗ ██████╗ ██╗   ██╗██╗████████╗██╗   ██╗
██║   ██║██╔══██╗██║██╔═══██╗██║   ██║██║╚══██╔══╝╚██╗ ██╔╝
██║   ██║██████╔╝██║██║   ██║██║   ██║██║   ██║    ╚████╔╝
██║   ██║██╔══██╗██║██║▄▄ ██║██║   ██║██║   ██║     ╚██╔╝
╚██████╔╝██████╔╝██║╚██████╔╝╚██████╔╝██║   ██║      ██║
 ╚═════╝ ╚═════╝ ╚═╝ ╚══▀▀═╝  ╚═════╝ ╚═╝   ╚═╝      ╚═╝"""

COMPACT = "∞ ubiquity"

GRADIENT = ("#22d3ee", "#6366f1", "#a855f7", "#e879f9")

BANNER_MIN_WIDTH = 62


def _blend(start: str, end: str, ratio: float) -> str:
    """Mix two ``#rrggbb`` colors, `ratio` of the way from `start` to `end`."""
    pairs = zip(
        (int(start[i : i + 2], 16) for i in (1, 3, 5)),
        (int(end[i : i + 2], 16) for i in (1, 3, 5)),
    )
    return "#" + "".join(f"{round(a + (b - a) * ratio):02x}" for a, b in pairs)


def shade(ratio: float, stops: Sequence[str] = GRADIENT) -> str:
    """The gradient color at `ratio`, 0 at its left end and 1 at its right."""
    ratio = min(max(ratio, 0.0), 1.0)
    span = 1 / (len(stops) - 1)
    index = min(int(ratio / span), len(stops) - 2)
    return _blend(stops[index], stops[index + 1], (ratio - index * span) / span)


def gradient(art: str, stops: Sequence[str] = GRADIENT) -> Text:
    """Paint `art` left to right through `stops`, one color per column.

    The whole block shares one horizontal ramp rather than each line carrying
    its own, so the mark and the wordmark under it read as a single object --
    which is the point of the logo they are drawn from.
    """
    lines = art.split("\n")
    span = max((len(line) for line in lines), default=1) - 1 or 1
    text = Text()
    for row, line in enumerate(lines):
        for column, character in enumerate(line):
            text.append(character, style=shade(column / span, stops))
        if row < len(lines) - 1:
            text.append("\n")
    return text


def banner(subtitle: str = "") -> None:
    """Draw the mark and wordmark, used by the wizard and the top of the REPL.

    A terminal too narrow for the wordmark gets the one-line form instead of a
    wrapped one: art that wraps is worse than no art.
    """
    view = console()
    view.print(gradient(COMPACT if view.width < BANNER_MIN_WIDTH else WORDMARK))
    if subtitle:
        view.print(Text(f"  {subtitle}", style="muted"))
    view.print()


def rule(title: str) -> None:
    """Draw a labelled horizontal rule."""
    console().rule(Text(title, style="muted"), style="muted")


def line(style: str = "rule") -> None:
    """Draw a plain full-width horizontal line.

    The input is fenced by one of these above and below, so a prompt and its
    answer read as a block rather than running into the turn before it.
    """
    console().print(Text("─" * console().width, style=style))


def note(message: str, style: str = "muted") -> None:
    """Print one line of CLI chrome."""
    console().print(Text(message, style=style))


def panel(body: Any, title: str = "", style: str = "muted") -> None:
    """Print `body` inside a titled box."""
    console().print(Panel(body, title=title, border_style=style, title_align="left"))


def key_values(rows: Sequence[tuple[str, str]], title: str = "") -> None:
    """Print aligned label/value pairs, used for summaries and session lists."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="muted", justify="right")
    table.add_column()
    for label, value in rows:
        table.add_row(label, value)
    if title:
        panel(table, title=title)
    else:
        console().print(table)


async def ask_text(prompt: str, default: str = "", password: bool = False) -> str:
    """Ask for a line of input without blocking the event loop.

    The prompt is escaped before it is styled. Prompts carry user-facing text
    -- `[y] allow`, a model identifier, a question the model wrote -- and rich
    would read the brackets in any of them as markup and swallow them.
    """
    from rich.markup import escape
    from rich.prompt import Prompt

    def read() -> str:
        return Prompt.ask(
            f"[key]{escape(prompt)}[/key]",
            console=console(),
            default=default or None,
            password=password,
            show_default=bool(default),
        ) or ""

    try:
        return (await asyncio.to_thread(read)).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


MENU_ROWS = 10
"""How many options a menu shows at once before it starts scrolling."""


def menu_height() -> int:
    """How many rows this terminal can spare for a menu's options."""
    return max(4, min(MENU_ROWS, console().height - 8))


def window(total: int, cursor: int, height: int) -> tuple[int, int]:
    """Return the half-open range of rows to draw, keeping `cursor` inside it.

    A list longer than the terminal has to move under a stationary cursor at
    some point; doing it around the middle means the rows either side of the
    selection stay visible, which is what makes a long list navigable rather
    than merely scrollable.
    """
    if total <= height:
        return 0, total
    start = min(max(cursor - height // 2, 0), total - height)
    return start, start + height


def _menu(
    prompt: str,
    options: Sequence[tuple[str, str]],
    cursor: int,
    chosen: set[int],
    *,
    multi: bool,
    allow_other: bool,
    typed: str | None,
) -> Any:
    """Render the menu as it currently stands, for one Live refresh."""
    body = Table.grid(padding=(0, 1))
    body.add_column(width=2)
    body.add_column(width=3)
    body.add_column(style="key")
    body.add_column(style="muted")

    total = len(options) + (1 if typed is not None else 0)
    height = menu_height()
    start, end = window(total, cursor, height)

    if start:
        body.add_row(Text(""), Text(""), Text(f"⋯ {start} more", style="muted"), Text(""))
    for index in range(start, min(end, len(options))):
        value, description = options[index]
        active = index == cursor
        mark = ""
        if multi:
            mark = "[x]" if index in chosen else "[ ]"
        body.add_row(
            Text("❯" if active else " ", style="brand"),
            Text(mark, style="ok" if index in chosen else "muted"),
            Text(value, style="brand" if active else "key"),
            Text(description, style="muted"),
        )

    if typed is not None and end > len(options):
        body.add_row(
            Text("❯" if cursor == len(options) else " ", style="brand"),
            Text(""),
            Text(typed or "type something else…", style="brand" if cursor == len(options) else "muted"),
            Text("", style="muted"),
        )

    if end < total:
        body.add_row(
            Text(""), Text(""), Text(f"⋯ {total - end} more", style="muted"), Text("")
        )

    keys_help = "↑↓ move · enter select"
    if multi:
        keys_help = "↑↓ move · space toggle · enter confirm"
    if allow_other:
        keys_help += " · type to search"

    group = Table.grid()
    group.add_column()
    group.add_row(Text(prompt, style="key"))
    group.add_row(body)
    group.add_row(Text(keys_help, style="muted"))
    return group


def _select_blocking(
    prompt: str,
    options: Sequence[tuple[str, str]],
    *,
    multi: bool,
    allow_other: bool,
    default: int,
) -> list[str]:
    """Run the arrow-key menu, returning the chosen values.

    Typing filters the list rather than replacing it, and the typed text is
    itself selectable when `allow_other` is set -- which is how a model
    identifier the CLI has never heard of gets entered without a second prompt.
    """
    from rich.live import Live

    view = console()
    cursor = min(default, len(options) - 1) if options else 0
    chosen: set[int] = set()
    typed = "" if allow_other else None
    visible = list(options)

    with Live(console=view, auto_refresh=False, transient=True) as live:
        while True:
            live.update(
                _menu(
                    prompt, visible, cursor, chosen,
                    multi=multi, allow_other=allow_other, typed=typed,
                ),
                refresh=True,
            )
            key = keys.read_key()
            limit = len(visible) + (1 if typed is not None else 0)

            if key == keys.INTERRUPT:
                raise KeyboardInterrupt
            if key in (keys.EOF, keys.ESCAPE):
                raise _Abandoned
            if key == keys.UP:
                cursor = (cursor - 1) % max(limit, 1)
            elif key == keys.DOWN:
                cursor = (cursor + 1) % max(limit, 1)
            elif key == keys.SPACE and multi and cursor < len(visible):
                chosen.symmetric_difference_update({cursor})
            elif key == keys.ENTER:
                if typed is not None and cursor == len(visible):
                    if typed:
                        return [typed]
                    continue
                if multi:
                    picked = sorted(chosen or {cursor})
                    return [visible[i][0] for i in picked if i < len(visible)]
                if visible:
                    return [visible[cursor][0]]
            elif key == keys.BACKSPACE and typed:
                typed = typed[:-1]
                visible, cursor = _filter(options, typed, cursor)
            elif len(key) == 1 and key.isprintable():
                if typed is None:
                    if key.isdigit() and 1 <= int(key) <= len(visible):
                        return [visible[int(key) - 1][0]]
                else:
                    typed += key
                    visible, cursor = _filter(options, typed, cursor)

    return []


def _filter(
    options: Sequence[tuple[str, str]], typed: str, cursor: int
) -> tuple[list[tuple[str, str]], int]:
    """Narrow the list to what the typed text matches, keeping the cursor sane."""
    if not typed:
        return list(options), min(cursor, max(len(options) - 1, 0))
    needle = typed.lower()
    visible = [row for row in options if needle in row[0].lower() or needle in row[1].lower()]
    return visible, min(cursor, len(visible))


class _Abandoned(Exception):
    """Raised inside the menu when the user presses Escape."""


async def select(
    prompt: str,
    options: Sequence[tuple[str, str]],
    *,
    multi: bool = False,
    allow_other: bool = False,
    default: int = 0,
) -> list[str]:
    """Ask the user to pick from `options`, returning the chosen values.

    Uses an arrow-key menu on a real terminal and falls back to a numbered
    prompt everywhere else, so the same call works over a pipe, in CI, and in
    an editor's terminal pane that does not report itself as a tty.
    """
    if not keys.supported():
        value = await _ask_numbered(prompt, options, allow_other=allow_other)
        return [value] if value else []
    try:
        return await asyncio.to_thread(
            _select_blocking,
            prompt,
            options,
            multi=multi,
            allow_other=allow_other,
            default=default,
        )
    except _Abandoned:
        return []


async def ask_choice(
    prompt: str,
    options: Sequence[tuple[str, str]],
    *,
    allow_other: bool = False,
    default: int | None = None,
) -> str:
    """Single-select convenience wrapper returning one value, or ``""``."""
    picked = await select(
        prompt,
        options,
        allow_other=allow_other,
        default=(default - 1) if default else 0,
    )
    return picked[0] if picked else ""


async def _ask_numbered(
    prompt: str,
    options: Sequence[tuple[str, str]],
    *,
    allow_other: bool = False,
) -> str:
    """The fallback menu: print the list and read a number."""
    view = console()
    view.print()
    view.print(Text(prompt, style="key"))
    table = Table.grid(padding=(0, 2))
    table.add_column(style="brand", justify="right")
    table.add_column(style="key")
    table.add_column(style="muted")
    for index, (value, description) in enumerate(options, start=1):
        table.add_row(str(index), value, description)
    view.print(table)

    hint = "number, or type your own" if allow_other else "number"
    while True:
        reply = await ask_text(f"  choice ({hint})", default="1")
        if reply.isdigit() and 1 <= int(reply) <= len(options):
            return options[int(reply) - 1][0]
        if reply and allow_other:
            return reply
        if not reply:
            return ""
        note("  pick a number from the list", "warn")


async def confirm(prompt: str, default: bool = True) -> bool:
    """Ask a yes/no question."""
    suffix = "Y/n" if default else "y/N"
    reply = (await ask_text(f"{prompt} ({suffix})")).lower()
    if not reply:
        return default
    return reply in ("y", "yes")


async def hotkey(prompt: str, accepted: str, default: str = "") -> str:
    """Wait for one of `accepted` to be pressed, without needing Enter.

    Falls back to a line read when the terminal cannot be put in raw mode, so
    the same prompt works over a pipe.
    """
    view = console()
    if not keys.supported():
        reply = (await ask_text(prompt, default=default)).lower()
        return reply[:1] if reply else default

    view.print(Text(prompt, style="prompt.head"), end=" ")
    while True:
        try:
            key = await asyncio.to_thread(keys.read_key)
        except (EOFError, KeyboardInterrupt):
            view.print()
            return default
        if key == keys.ENTER and default:
            view.print(Text(default, style="brand"))
            return default
        if key in (keys.INTERRUPT, keys.EOF, keys.ESCAPE):
            view.print()
            return default
        if len(key) == 1 and key.lower() in accepted:
            view.print(Text(key.lower(), style="brand"))
            return key.lower()


__all__ = [
    "console",
    "set_console",
    "banner",
    "gradient",
    "shade",
    "rule",
    "note",
    "panel",
    "key_values",
    "ask_text",
    "ask_choice",
    "select",
    "hotkey",
    "confirm",
    "THEME",
]
