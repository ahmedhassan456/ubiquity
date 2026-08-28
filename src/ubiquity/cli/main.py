"""Entry point for the `ubiquity` command.

The CLI has exactly one path through the SDK: `_run_turn()` calls `summon()`
once and renders what comes back. Everything else is about deciding what to
feed it. A prompt on the command line or on stdin runs a single turn and
exits; a terminal with no prompt opens the REPL, which is the same call in a
loop with the session id carried forward.

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

from .. import __version__
from ..client import summon
from ..options import Options
from ..settings import apply_settings
from .commands import ReplState, dispatch, is_command
from .prompts import terminal_handler
from .render import Renderer, paint

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

    parser.add_argument("-m", "--model", help="model id or alias, e.g. openai:gpt-5")
    parser.add_argument("--fallback-model", help="model to fall back to on failure")
    parser.add_argument(
        "--permission-mode", choices=_MODES, default="default", help="permission mode"
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
        help=f"comma-separated setting sources ({', '.join(_SOURCES)})",
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
    parser.add_argument(
        "--output-format", choices=("text", "json", "stream-json"), default="text"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show tool output")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color")
    return parser


def _rules(raw: str | None) -> list[str]:
    """Split a comma-separated rule list."""
    return [part.strip() for part in raw.split(",") if part.strip()] if raw else []


def options_from(args: argparse.Namespace) -> Options:
    """Translate parsed arguments into `Options`."""
    sources = _rules(args.sources)
    return Options(
        model=args.model,
        fallback_model=args.fallback_model,
        cwd=Path(args.cwd).expanduser() if args.cwd else Path.cwd(),
        add_dirs=[Path(d).expanduser() for d in args.add_dir],
        permission_mode=args.permission_mode,
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
        include_partial_messages=args.output_format == "text",
    )


async def _run_turn(
    prompt: str, state: ReplState, renderer: Renderer, *, interactive: bool
) -> int:
    """Run one `summon()` call and render it. Returns a process exit code.

    `can_use_tool` is only installed for an interactive run. Without a handler
    the SDK leaves `AskUserQuestion` out of the suite and falls back to the
    permission rules, which is the correct behaviour for `--print`: there is
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
        async for message in summon(prompt, options):
            renderer.handle(message)
            if message.type == "system" and getattr(message, "subtype", "") == "init":
                state.session_id = message.session_id
            elif message.type == "result":
                state.turns += message.num_turns
                state.total_cost += message.total_cost_usd or 0.0
                exit_code = 1 if message.is_error else 0
    renderer.close_text()
    return exit_code


class _interrupts:
    """Route SIGINT to the run's abort event for the duration of a turn."""

    def __init__(self, abort: asyncio.Event, renderer: Renderer) -> None:
        self.abort = abort
        self.renderer = renderer
        self.previous = None

    def __enter__(self) -> _interrupts:
        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, self._fire)
            self.previous = True
        except NotImplementedError:
            self.previous = None
        return self

    def _fire(self) -> None:
        if self.abort.is_set():
            raise KeyboardInterrupt
        self.abort.set()
        self.renderer.note("  interrupting… (Ctrl-C again to force)", "yellow")

    def __exit__(self, *exc: object) -> None:
        if self.previous:
            asyncio.get_running_loop().remove_signal_handler(signal.SIGINT)


async def _repl(state: ReplState, renderer: Renderer) -> int:
    """Read prompts until the user leaves, running one turn each."""
    color = renderer.color
    renderer.note(f"ubiquity {__version__} — /help for commands, /exit to leave", "dim")
    renderer.note(f"  {state.options.resolved_model()} in {state.options.resolved_cwd()}", "dim")

    exit_code = 0
    while True:
        try:
            line = (await asyncio.to_thread(input, paint("\n› ", "bold", enabled=color))).strip()
        except (EOFError, KeyboardInterrupt):
            renderer.note("", "dim")
            return exit_code

        if not line:
            continue
        if is_command(line):
            if dispatch(line, state, renderer) == "exit":
                return exit_code
            continue

        state.history.append(line)
        try:
            exit_code = await _run_turn(line, state, renderer, interactive=True)
        except KeyboardInterrupt:
            renderer.note("  interrupted", "yellow")
    return exit_code


def _read_prompt(args: argparse.Namespace) -> str:
    """Assemble the prompt from arguments and any piped stdin."""
    parts = [" ".join(args.prompt).strip()] if args.prompt else []
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            parts.append(piped)
    return "\n\n".join(part for part in parts if part)


async def _main(argv: list[str] | None = None) -> int:
    """Parse, decide between one-shot and REPL, and run."""
    args = build_parser().parse_args(argv)
    options = apply_settings(options_from(args))
    renderer = Renderer(
        args.output_format,
        color=not args.no_color and sys.stdout.isatty(),
        verbose=args.verbose,
    )
    state = ReplState(options=options)

    prompt = _read_prompt(args)
    if prompt:
        return await _run_turn(
            prompt, state, renderer, interactive=not args.print_mode
        )
    if args.print_mode or not sys.stdin.isatty():
        print("ubiquity: no prompt given", file=sys.stderr)
        return 2
    return await _repl(state, renderer)


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point.

    A misconfigured run -- no model, an unknown provider, a session id that
    does not exist -- is a mistake at the command line, not a crash. It is
    reported as one line on stderr rather than a traceback, which is also what
    keeps `--output-format json` machine-readable when a run cannot start.
    """
    try:
        return asyncio.run(_main(argv))
    except ValueError as exc:
        print(f"ubiquity: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
