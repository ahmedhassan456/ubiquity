"""Entry point for the `ubiquity` command.

There is one path through the SDK: `_run_turn()` calls `summon()` once and
renders what comes back. Everything else decides what to feed it. A prompt on
the command line or on stdin runs a single turn and exits; a terminal with no
prompt opens the REPL, which is the same call in a loop with the session id
carried forward.

Configuration is resolved before any of that. A first run with no model
anywhere -- no flag, no `UBIQUITY_MODEL`, no settings file -- opens the wizard
rather than failing with a `ValueError` about a field the user has never heard
of. The wizard writes the SDK's own user settings file, so `--sources` defaults
to reading it and the choice sticks.

Partial messages are asked for whenever reasoning is being shown, which is the
default: a thinking block is only worth watching while it is being written.
`--stream` is a separate question about the reply itself -- reasoning streams
either way, and the answer is still rendered as markdown when the turn ends.

Interrupt handling belongs here rather than in the SDK: `Options.abort` is the
caller's event, and this is the caller. The first Ctrl-C sets it, which ends
the run at the next tool call and still yields a result message; a second one
during the same turn gives up on the graceful path.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from dataclasses import replace
from pathlib import Path

from pydantic_ai.exceptions import UserError

from .. import __version__
from ..client import summon
from ..options import Options
from ..settings import apply_settings
from . import setup as setup_module
from . import ui
from .commands import ReplState, dispatch, is_command
from . import mentions
from .completion import LineReader
from .prompts import terminal_handler
from .render import OUTPUT_FORMATS, Renderer

_MODES = ("default", "acceptEdits", "bypassPermissions", "plan", "dontAsk")
_SOURCES = ("user", "project", "local")


def build_parser() -> argparse.ArgumentParser:
    """Define the command line."""
    parser = argparse.ArgumentParser(
        prog="ubiquity",
        description="Run a ubiquity coding agent from the terminal.",
    )
    parser.add_argument("prompt", nargs="*", help="the prompt; omit for a REPL")
    parser.add_argument("--version", action="version", version=f"ubiquity {__version__}")
    parser.add_argument(
        "--setup", action="store_true", help="reconfigure model and provider"
    )

    parser.add_argument("-m", "--model", help="model id or alias, e.g. openai:gpt-5")
    parser.add_argument("--fallback-model", help="model to fall back to on failure")
    parser.add_argument(
        "--permission-mode", choices=_MODES, help="permission mode for this run"
    )
    parser.add_argument(
        "--allowed-tools", help="comma-separated allow rules, e.g. 'Read,Bash(git:*)'"
    )
    parser.add_argument("--disallowed-tools", help="comma-separated deny rules")
    parser.add_argument("--cwd", help="working directory for the run")
    parser.add_argument(
        "--add-dir", action="append", default=[], help="extra readable directory"
    )
    parser.add_argument("--max-turns", type=int, default=40, help="turn budget")
    parser.add_argument("--system-prompt", help="replace the system prompt")
    parser.add_argument("--append-system-prompt", help="append to the system prompt")
    parser.add_argument(
        "--sources",
        default="user,project,local",
        help=f"comma-separated setting sources ({', '.join(_SOURCES)}); '' for none",
    )

    parser.add_argument(
        "-c", "--continue", dest="continue_conversation", action="store_true",
        help="continue the most recent session",
    )
    parser.add_argument("-r", "--resume", help="resume a session by id")
    parser.add_argument(
        "--no-persist", action="store_true", help="do not write a session transcript"
    )

    parser.add_argument(
        "-p", "--print", dest="print_mode", action="store_true",
        help="non-interactive: never prompt, print the result and exit",
    )
    parser.add_argument("--output-format", choices=OUTPUT_FORMATS, default="text")
    parser.add_argument(
        "--stream", action="store_true",
        help="stream replies as plain text instead of rendering markdown",
    )
    parser.add_argument(
        "--no-thinking", action="store_true", help="hide the model's reasoning"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show tool output")
    parser.add_argument("--no-color", action="store_true", help="disable color")
    return parser


def _rules(raw: str | None) -> list[str]:
    """Split a comma-separated rule list."""
    return [part.strip() for part in raw.split(",") if part.strip()] if raw else []


def options_from(args: argparse.Namespace) -> Options:
    """Translate parsed arguments into `Options`.

    `permission_mode` is left at its default unless the flag is given, so a
    `defaultMode` in a settings file is not overridden by an argument the user
    did not type.
    """
    sources = _rules(args.sources)
    return Options(
        model=args.model,
        fallback_model=args.fallback_model,
        cwd=Path(args.cwd).expanduser() if args.cwd else Path.cwd(),
        add_dirs=[Path(d).expanduser() for d in args.add_dir],
        permission_mode=args.permission_mode or "default",
        allowed_tools=_rules(args.allowed_tools) or None,
        disallowed_tools=_rules(args.disallowed_tools),
        system_prompt=args.system_prompt,
        append_system_prompt=args.append_system_prompt,
        setting_sources=sources,
        agent_sources=sources,
        skill_sources=sources,
        memory_sources=sources,
        max_turns=args.max_turns,
        continue_conversation=args.continue_conversation,
        resume=args.resume,
        persist_session=not args.no_persist,
        include_partial_messages=(
            args.output_format == "text" and (args.stream or not args.no_thinking)
        ),
    )


async def _run_turn(
    prompt: str, state: ReplState, renderer: Renderer, *, interactive: bool
) -> int:
    """Run one `summon()` call and render it. Returns a process exit code.

    `can_use_tool` is only installed for an interactive run. Without a handler
    the SDK leaves `AskUserQuestion` out of the suite and falls back to the
    permission rules, which is the right behaviour for `--print`: there is
    nobody at the terminal to answer.
    """
    abort = asyncio.Event()
    options = replace(
        state.options,
        resume=state.session_id or state.options.resume,
        continue_conversation=(
            state.options.continue_conversation and state.session_id is None
        ),
        can_use_tool=terminal_handler(renderer) if interactive else None,
        abort=abort,
    )

    exit_code = 0
    with _interrupts(abort, renderer):
        try:
            async for message in summon(prompt, options):
                renderer.handle(message)
                if message.type == "system" and getattr(message, "subtype", "") == "init":
                    state.session_id = message.session_id
                elif message.type == "result":
                    state.turns += message.num_turns
                    state.total_cost += message.total_cost_usd or 0.0
                    exit_code = 1 if message.is_error else 0
        finally:
            renderer.close_text()
    return exit_code


class _interrupts:
    """Route SIGINT to the run's abort event for the duration of a turn."""

    def __init__(self, abort: asyncio.Event, renderer: Renderer) -> None:
        self.abort = abort
        self.renderer = renderer
        self.installed = False

    def __enter__(self) -> _interrupts:
        try:
            asyncio.get_running_loop().add_signal_handler(signal.SIGINT, self._fire)
            self.installed = True
        except NotImplementedError:
            self.installed = False
        return self

    def _fire(self) -> None:
        if self.abort.is_set():
            raise KeyboardInterrupt
        self.abort.set()
        self.renderer.note("  interrupting… (Ctrl-C again to force)", "warn")

    def __exit__(self, *exc: object) -> None:
        if self.installed:
            asyncio.get_running_loop().remove_signal_handler(signal.SIGINT)


async def _repl(state: ReplState, renderer: Renderer) -> int:
    """Read prompts until the user leaves, running one turn each."""
    ui.banner(f"v{__version__} — /help for commands, /exit to leave")
    ui.key_values(
        [
            ("model", str(state.options.resolved_model())),
            ("cwd", str(state.options.resolved_cwd())),
            ("mode", state.options.permission_mode),
        ],
        title="session",
    )

    exit_code = 0
    reader = LineReader(state)
    while True:
        try:
            line = await reader.read()
        except EOFError:
            return exit_code
        if not line:
            if not sys.stdin.isatty():
                return exit_code
            continue
        if is_command(line):
            outcome = await dispatch(line, state, renderer)
            if outcome == "exit":
                return exit_code
            continue

        state.history.append(line)
        prompt, attached = mentions.expand(line, state.options.resolved_cwd())
        for mention in attached:
            renderer.note(f"  ⎘ {mentions.display(mention, state.options.resolved_cwd())}", "muted")
        try:
            exit_code = await _run_turn(prompt, state, renderer, interactive=True)
        except KeyboardInterrupt:
            renderer.note("  interrupted", "warn")
    return exit_code


def _read_prompt(args: argparse.Namespace) -> str:
    """Assemble the prompt from arguments and any piped stdin.

    Reading stdin is deferred until configuration is settled. The wizard reads
    the same descriptor, and a run that drained it first would leave the
    prompts with nothing to read.
    """
    parts = [" ".join(args.prompt).strip()] if args.prompt else []
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            parts.append(piped)
    return "\n\n".join(part for part in parts if part)


async def _ensure_configured(args: argparse.Namespace, interactive: bool) -> bool:
    """Make sure a model is configured, running the wizard when it is not.

    Returns False when the CLI should stop: either the user abandoned the
    wizard, or there is no terminal to run it in, which is the case a script
    hits and where a one-line error beats a prompt nobody will see.
    """
    if args.setup:
        return bool(await setup_module.run_wizard())
    if args.model or setup_module.is_configured():
        return True
    if not interactive:
        ui.note(
            "ubiquity: no model configured. Run `ubiquity --setup`, "
            "or set UBIQUITY_MODEL.",
            "bad",
        )
        return False
    return bool(await setup_module.run_wizard(first_run=True))


def _warn_missing_credential(options: Options) -> None:
    """Point at an unset provider variable before the request fails on it."""
    try:
        model = options.resolved_model()
    except ValueError:
        return
    if not isinstance(model, str):
        return
    variable = setup_module.credential_missing(model)
    if variable:
        ui.note(f"  {variable} is not set — this run will fail to authenticate", "warn")
        setup_module.credential_help(variable)


async def _main(argv: list[str] | None = None) -> int:
    """Parse, configure, then decide between one-shot and REPL."""
    args = build_parser().parse_args(argv)
    ui.set_console(color=not args.no_color)

    interactive = sys.stdin.isatty() and not args.print_mode
    if not await _ensure_configured(args, interactive):
        return 0 if args.setup else 2

    prompt = "" if args.setup else _read_prompt(args)
    if args.setup and not prompt:
        ui.note("\n  ready — run `ubiquity` to start a session\n", "ok")
        return 0

    options = apply_settings(options_from(args))
    renderer = Renderer(
        args.output_format,
        verbose=args.verbose,
        stream_text=args.stream,
        show_thinking=not args.no_thinking,
    )
    state = ReplState(options=options)
    if args.output_format == "text":
        _warn_missing_credential(options)

    if prompt:
        prompt, _ = mentions.expand(prompt, options.resolved_cwd())
        return await _run_turn(prompt, state, renderer, interactive=interactive)
    if not interactive:
        ui.note("ubiquity: no prompt given", "bad")
        return 2
    return await _repl(state, renderer)


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point.

    A misconfigured run -- a missing credential, an unknown provider, a
    session id that does not exist -- is a mistake at the command line, not a
    crash. It is reported as one line rather than a traceback, which is also
    what keeps `--output-format json` machine-readable when a run cannot start.
    `UserError` is pydantic-ai's name for the same category, raised when it
    builds a provider it has no key for.
    """
    try:
        return asyncio.run(_main(argv))
    except (ValueError, UserError) as exc:
        ui.note(f"ubiquity: {exc}", "bad")
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
