"""Slash commands for the interactive session.

A command mutates the `ReplState` in place and reports what the loop should do
next. Almost none of them talk to the model: a command is a change to the
options the *next* `summon()` call is given, which is what keeps the REPL a
thin wrapper over the same one-shot path. `/compact` is the exception, and it
makes a summarizing call of its own rather than a run.

`dispatch` is async because several commands -- `/setup`, `/key`, `/compact` --
put a question to the user or a request to a provider, and everything that
reads the terminal does so off the event loop thread.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich.markdown import Markdown
from rich.table import Table

from ..compaction import COMPACTION_PROMPT, SUMMARY_TEMPLATE
from ..options import Options
from ..sessions.store import SessionStore
from . import setup as setup_module
from . import ui

if TYPE_CHECKING:
    from .render import Renderer

MODES: tuple[str, ...] = (
    "default",
    "acceptEdits",
    "bypassPermissions",
    "plan",
    "dontAsk",
)

COMMANDS: tuple[tuple[str, str], ...] = (
    ("/help", "show this list"),
    ("/model <name>", "switch model for the next turn"),
    ("/mode <mode>", f"permission mode: {', '.join(MODES)}"),
    ("/setup", "reconfigure provider, model, and default mode"),
    ("/key", "set the provider credential for this shell"),
    ("/compact [focus]", "summarize the session so far and continue from it"),
    ("/new", "start a fresh session, forgetting the transcript"),
    ("/resume <id>", "continue a stored session"),
    ("/sessions", "list recent sessions for this directory"),
    ("/cost", "what this session has cost so far"),
    ("/cwd <path>", "change the working directory"),
    ("/verbose", "toggle tool output and usage lines"),
    ("/exit", "leave (Ctrl-D works too)"),
)


@dataclass(slots=True)
class ReplState:
    """What carries across turns of one interactive session."""

    options: Options
    session_id: str | None = None
    total_cost: float = 0.0
    turns: int = 0
    history: list[str] = field(default_factory=list)


def is_command(line: str) -> bool:
    """True when a line should be handled here rather than sent to the model."""
    return line.startswith("/")


async def dispatch(line: str, state: ReplState, renderer: Renderer) -> str:
    """Run one slash command and return ``continue`` or ``exit``.

    Unknown commands are reported rather than forwarded, so a typo does not
    silently become a prompt and cost a turn.
    """
    name, _, rest = line[1:].partition(" ")
    rest = rest.strip()

    if name in ("exit", "quit", "q"):
        return "exit"

    if name in ("help", "?", ""):
        _help()
    elif name == "model":
        await _model(rest, state)
    elif name == "mode":
        await _mode(rest, state)
    elif name == "setup":
        changes = await setup_module.run_wizard()
        if changes.get("model"):
            state.options.model = changes["model"]
            state.session_id = None
    elif name == "compact":
        await _compact(rest, state, renderer)
    elif name == "key":
        await _key(state)
    elif name in ("new", "clear"):
        state.session_id = None
        state.turns = 0
        ui.note("  started a new session", "ok")
    elif name == "resume":
        if not rest:
            ui.note("  usage: /resume <session-id>", "warn")
        else:
            state.session_id = rest
            ui.note(f"  resuming {rest}", "ok")
    elif name == "sessions":
        _sessions(state)
    elif name == "cost":
        ui.note(f"  ${state.total_cost:.4f} over {state.turns} turns")
    elif name == "cwd":
        _cwd(rest, state)
    elif name == "verbose":
        renderer.verbose = not renderer.verbose
        ui.note(f"  verbose {'on' if renderer.verbose else 'off'}", "ok")
    else:
        ui.note(f"  unknown command: /{name} (try /help)", "warn")

    return "continue"


def _help() -> None:
    table = Table.grid(padding=(0, 3))
    table.add_column(style="brand")
    table.add_column(style="muted")
    for command, description in COMMANDS:
        table.add_row(command, description)
    ui.panel(table, title="commands")


async def _model(rest: str, state: ReplState) -> None:
    """Switch models, opening the picker when no name is given."""
    if not rest:
        rest = await _pick_model(state)
        if not rest:
            return
    state.options.model = rest
    ui.note(f"  model set to {rest}", "ok")
    variable = setup_module.credential_missing(rest)
    if variable:
        ui.note(f"  {variable} is not set — /key to fix it", "warn")


async def _mode(rest: str, state: ReplState) -> None:
    """Switch permission mode, opening the picker when none is given."""
    if not rest:
        rest = await ui.ask_choice(
            "Permission mode", list(setup_module.MODES)
        )
        if not rest:
            return
    if rest not in MODES:
        ui.note(f"  mode must be one of: {', '.join(MODES)}", "warn")
        return
    state.options.permission_mode = rest
    ui.note(f"  permission mode set to {rest}", "ok")


def _ollama_configured() -> bool:
    """True when the user has pointed the CLI at a local Ollama server."""
    return bool(os.environ.get(setup_module.OLLAMA_ENV, "").strip())


async def _pick_model(state: ReplState) -> str:
    """Offer this provider's models, plus a row for typing any other."""
    current = str(state.options.model or "")
    provider = current.split(":", 1)[0] if ":" in current else ""
    options = list(setup_module.SUGGESTED.get(provider, ()))
    if provider == "ollama" or _ollama_configured():
        options = [*setup_module.local_models(), *options]
    for other, models in setup_module.SUGGESTED.items():
        if other != provider:
            options.extend(models)
    return await ui.ask_choice("Which model?", options, allow_other=True)


async def _key(state: ReplState) -> None:
    """Set the provider credential for this process, and show how to persist it."""
    model = state.options.model
    variable = setup_module.env_var_for(str(model)) if model else None
    if not variable:
        variable = await ui.ask_text("  which environment variable")
    if not variable:
        return
    value = await ui.ask_text(
        f"  {variable}", password=variable != setup_module.OLLAMA_ENV
    )
    if not value:
        return
    os.environ[variable] = value
    ui.note(f"  {variable} set for this session", "ok")
    setup_module.credential_help(variable, value)


async def _compact(rest: str, state: ReplState, renderer: Renderer) -> None:
    """Replace the session transcript with a summary and continue from it.

    The REPL keeps no history of its own -- each turn resumes the stored
    session -- so compacting means writing a *new* session whose whole
    transcript is the summary, and pointing the next turn at it. The old
    session file is left alone: it is the only remaining copy of what was
    summarized, and `/resume` can still reach it.

    `rest` is passed to the summarizer as extra instruction, which is how you
    say what the summary must not lose.
    """
    from uuid import uuid4

    from ..compaction import summarize
    from ..sessions.replay import history_from
    from ..types import SDKUserMessage

    if not state.session_id:
        ui.note("  nothing to compact yet — this session has no turns", "warn")
        return
    if not state.options.persist_session:
        ui.note("  nothing to compact: this run is not persisting a session", "warn")
        return

    cwd = state.options.resolved_cwd()
    store = SessionStore(state.options.session_dir)
    history = history_from(store.read(state.session_id, cwd))
    if len(history) < 2:
        ui.note("  nothing to compact yet — this session has no turns", "warn")
        return

    instructions = None
    if rest:
        instructions = (
            f"{COMPACTION_PROMPT}\n\nThe user asks that the summary focus on: {rest}"
        )

    renderer.start_status("compacting")
    try:
        summary = await summarize(
            history,
            state.options.resolved_compact_model(),
            instructions=instructions,
            aliases=state.options.model_aliases,
            provider_kwargs=state.options.provider_kwargs,
        )
    except Exception as error:
        ui.note(f"  compaction failed: {error}", "bad")
        return
    finally:
        renderer.stop_status()

    compacted = str(uuid4())
    store.append(
        compacted,
        cwd,
        SDKUserMessage(
            content=SUMMARY_TEMPLATE.format(summary=summary),
            session_id=compacted,
        ),
    )
    state.session_id = compacted
    ui.panel(Markdown(summary), title="compacted")
    ui.note(f"  {len(history)} messages summarized; continuing as {compacted}", "ok")


def _cwd(rest: str, state: ReplState) -> None:
    if not rest:
        ui.note(f"  cwd {state.options.resolved_cwd()}")
        return
    target = Path(rest).expanduser()
    if not target.is_dir():
        ui.note(f"  no such directory: {target}", "warn")
        return
    state.options.cwd = target
    state.session_id = None
    ui.note(f"  cwd set to {target}, session reset", "ok")


def _sessions(state: ReplState) -> None:
    """Print recent sessions for the current directory."""
    store = SessionStore(state.options.session_dir)
    entries = store.list(state.options.resolved_cwd(), limit=10)
    if not entries:
        ui.note("  no stored sessions here")
        return
    now = time.time()
    table = Table.grid(padding=(0, 3))
    table.add_column(style="brand")
    table.add_column(style="muted", justify="right")
    table.add_column()
    for entry in entries:
        label = entry.title or entry.summary or f"{entry.message_count} messages"
        table.add_row(entry.session_id[:8], _age(now - entry.updated_at), label[:60])
    ui.panel(table, title="sessions")


def _age(seconds: float) -> str:
    """Render an age as a compact relative string."""
    if seconds < 60:
        return "now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"
