"""Slash commands for the interactive session.

A command mutates the `ReplState` in place and reports what the loop should do
next. Nothing here talks to the model: every command is a change to the
options the *next* `summon()` call is given, which is what keeps the REPL a
thin wrapper over the same one-shot path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..options import Options
from ..sessions.store import SessionStore

if TYPE_CHECKING:
    from .render import Renderer

_MODES: tuple[str, ...] = (
    "default",
    "acceptEdits",
    "bypassPermissions",
    "plan",
    "dontAsk",
)


@dataclass(slots=True)
class ReplState:
    """What carries across turns of one interactive session."""

    options: Options
    session_id: str | None = None
    total_cost: float = 0.0
    turns: int = 0
    history: list[str] = field(default_factory=list)


HELP = """\
  /help              show this list
  /model <name>      switch model for the next turn
  /mode <mode>       switch permission mode ({modes})
  /new               start a fresh session, forgetting the transcript
  /resume <id>       continue a stored session
  /sessions          list recent sessions for this directory
  /cost              total cost of this session so far
  /cwd <path>        change the working directory
  /exit              leave (Ctrl-D works too)\
""".format(modes=", ".join(_MODES))


def is_command(line: str) -> bool:
    """True when a line should be handled here rather than sent to the model."""
    return line.startswith("/")


def dispatch(line: str, state: ReplState, renderer: Renderer) -> str:
    """Run one slash command and return ``continue`` or ``exit``.

    Unknown commands are reported rather than forwarded, so a typo does not
    silently become a prompt and cost a turn.
    """
    name, _, rest = line[1:].partition(" ")
    rest = rest.strip()

    if name in ("exit", "quit", "q"):
        return "exit"

    if name in ("help", "?", ""):
        renderer.note(HELP, "dim")
    elif name == "model":
        if not rest:
            renderer.note(f"  model {state.options.resolved_model()}", "dim")
        else:
            state.options.model = rest
            renderer.note(f"  model set to {rest}", "green")
    elif name == "mode":
        if rest not in _MODES:
            renderer.note(f"  mode must be one of: {', '.join(_MODES)}", "yellow")
        else:
            state.options.permission_mode = rest
            renderer.note(f"  permission mode set to {rest}", "green")
    elif name in ("new", "clear"):
        state.session_id = None
        state.turns = 0
        renderer.note("  started a new session", "green")
    elif name == "resume":
        if not rest:
            renderer.note("  usage: /resume <session-id>", "yellow")
        else:
            state.session_id = rest
            renderer.note(f"  resuming {rest}", "green")
    elif name == "sessions":
        _list_sessions(state, renderer)
    elif name == "cost":
        renderer.note(
            f"  ${state.total_cost:.4f} over {state.turns} turns", "dim"
        )
    elif name == "cwd":
        if not rest:
            renderer.note(f"  cwd {state.options.resolved_cwd()}", "dim")
        else:
            target = Path(rest).expanduser()
            if not target.is_dir():
                renderer.note(f"  no such directory: {target}", "yellow")
            else:
                state.options.cwd = target
                state.session_id = None
                renderer.note(f"  cwd set to {target}, session reset", "green")
    else:
        renderer.note(f"  unknown command: /{name} (try /help)", "yellow")

    return "continue"


def _list_sessions(state: ReplState, renderer: Renderer) -> None:
    """Print recent sessions for the current directory."""
    store = SessionStore(state.options.session_dir)
    entries = store.list(state.options.resolved_cwd(), limit=10)
    if not entries:
        renderer.note("  no stored sessions here", "dim")
        return
    now = time.time()
    for entry in entries:
        age = _age(now - entry.updated_at)
        label = entry.title or entry.summary or f"{entry.message_count} messages"
        renderer.note(f"  {entry.session_id[:8]}  {age:>6}  {label[:60]}", "dim")


def _age(seconds: float) -> str:
    """Render an age as a compact relative string."""
    if seconds < 60:
        return "now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"
