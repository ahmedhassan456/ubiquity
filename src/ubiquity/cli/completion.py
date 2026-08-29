"""The REPL input line: completion, history, and the prompt itself.

Slash commands are only discoverable if you already know they exist, which is
what completion fixes: typing `/` lists every command with what it does, and
narrowing the text narrows the list. Arguments complete too, because the
values a command accepts are exactly the ones this process already knows --
the permission modes, the models the wizard offers, the session ids on disk.

An `@` completes paths instead, against the session's working directory, so
mentioning a file is a few keystrokes rather than a remembered path.

`prompt_toolkit` owns the line editor rather than `input()`, which is what
makes any of that possible; it also brings arrow-key history for free. When
there is no terminal to drive it -- a pipe, a test, CI -- `read_line` falls
back to the plain prompt, so the REPL still runs where completion cannot.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from ..settings import SETTINGS_DIR
from . import ui
from .commands import COMMANDS
from .mentions import SKIP_DIRECTORIES

if TYPE_CHECKING:
    from .commands import ReplState

HISTORY_FILE = "history"
RULE_CHAR = "─"

_ARGUMENT_HINTS: dict[str, str] = {
    "/mode": "permission mode",
    "/model": "model identifier",
    "/resume": "session id",
    "/cwd": "directory",
}


def history_path() -> Path:
    """Where the REPL keeps its input history."""
    return Path.home() / SETTINGS_DIR / HISTORY_FILE


def command_names() -> list[tuple[str, str]]:
    """Return `(/name, description)` for every command, without its arguments."""
    return [(spec.split(" ", 1)[0], description) for spec, description in COMMANDS]


def _model_suggestions() -> list[tuple[str, str]]:
    """Every suggested model, with whatever Ollama has pulled first.

    The local server is only asked once a session and only when its address is
    set, so a user who has never touched Ollama pays nothing for this.
    """
    from .setup import OLLAMA_ENV, SUGGESTED

    suggested = [row for models in SUGGESTED.values() for row in models]
    if not os.environ.get(OLLAMA_ENV, "").strip():
        return suggested
    return [*_local_models(), *suggested]


@lru_cache(maxsize=1)
def _local_models() -> tuple[tuple[str, str], ...]:
    from .setup import local_models

    return local_models()


def _session_suggestions(state: ReplState) -> list[tuple[str, str]]:
    """Recent session ids for this directory, newest first."""
    from ..sessions.store import SessionStore

    try:
        entries = SessionStore(state.options.session_dir).list(
            state.options.resolved_cwd(), limit=10
        )
    except OSError:
        return []
    return [
        (entry.session_id, entry.title or entry.summary or f"{entry.message_count} msgs")
        for entry in entries
    ]


def argument_options(command: str, state: ReplState) -> list[tuple[str, str]]:
    """Return the completions for a command's argument, if it has any."""
    if command == "/mode":
        from .commands import MODES
        from .setup import MODES as DESCRIBED

        described = dict(DESCRIBED)
        return [(mode, described.get(mode, "")) for mode in MODES]
    if command == "/model":
        return _model_suggestions()
    if command == "/resume":
        return _session_suggestions(state)
    return []


def path_options(fragment: str, base: Path) -> list[tuple[str, str]]:
    """Return the paths an `@fragment` could mean, directories first.

    Noise directories are hidden until they are typed out, because a listing
    whose first entries are `.git/` and `__pycache__/` is a listing nobody
    reads. The same rule covers dotfiles: they appear once the dot is typed.
    """
    head, sep, tail = fragment.rpartition("/")
    directory = (base / head).expanduser() if sep else base
    try:
        entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    except OSError:
        return []

    found: list[tuple[str, str]] = []
    for entry in entries:
        if not entry.name.startswith(tail):
            continue
        if not tail and (entry.name.startswith(".") or entry.name in SKIP_DIRECTORIES):
            continue
        is_dir = entry.is_dir()
        name = f"{head}{sep}{entry.name}" + ("/" if is_dir else "")
        found.append((name, "directory" if is_dir else _size(entry)))
    return found


def _size(path: Path) -> str:
    """Render a file's size for the completion menu."""
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    for unit in ("B", "K", "M"):
        if size < 1024 or unit == "M":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return ""


def build_completer(state: ReplState) -> Any:
    """Build the completer for one REPL session.

    It is bound to the state rather than rebuilt per keystroke so that session
    ids and models reflect the session as it is now -- after a `/cwd`, the
    sessions offered are the ones in the new directory.
    """
    from prompt_toolkit.completion import Completer, Completion

    class ReplCompleter(Completer):
        """Completes slash commands and their arguments, and `@path` mentions."""

        def get_completions(
            self, document: Any, complete_event: Any
        ) -> Iterable[Completion]:
            text = document.text_before_cursor
            word = text.rpartition(" ")[2]
            if word.startswith("@"):
                fragment = word[1:]
                for name, meta in path_options(fragment, state.options.resolved_cwd()):
                    yield Completion(
                        name,
                        start_position=-len(fragment),
                        display=name,
                        display_meta=meta,
                    )
                return
            if not text.startswith("/"):
                return

            head, sep, tail = text.partition(" ")
            if not sep:
                for name, description in command_names():
                    if name.startswith(head):
                        yield Completion(
                            name,
                            start_position=-len(head),
                            display=name,
                            display_meta=description,
                        )
                return

            for value, description in argument_options(head, state):
                if value.startswith(tail):
                    yield Completion(
                        value,
                        start_position=-len(tail),
                        display=value,
                        display_meta=description,
                    )

    return ReplCompleter()


def _style() -> Any:
    """Match the completion menu and the input rules to the CLI's palette."""
    from prompt_toolkit.styles import Style

    return Style.from_dict(
        {
            "prompt": "bold #d75fd7",
            "rule": "#5f5f5f",
            "completion-menu.completion": "bg:#1c1c1c #d0d0d0",
            "completion-menu.completion.current": "bg:#d75fd7 #000000 bold",
            "completion-menu.meta.completion": "bg:#1c1c1c #808080",
            "completion-menu.meta.completion.current": "bg:#af5faf #eeeeee",
            "bottom-toolbar": "#5f5f5f noreverse",
        }
    )


def _width(default: int = 80) -> int:
    """The terminal's current width, re-read on every draw so a resize lands."""
    from prompt_toolkit.application.current import get_app

    try:
        return max(20, get_app().output.get_size().columns)
    except Exception:
        return default


def completable(text: str) -> bool:
    """True when `text` is a line the completer has anything to say about.

    This gates completion-while-typing, and through it the space prompt_toolkit
    reserves for the menu: reserving unconditionally would hold six blank lines
    open under every ordinary prompt, pushing the closing rule away from the
    input it is supposed to close.
    """
    return text.startswith("/") or text.rpartition(" ")[2].startswith("@")


def _completable() -> Any:
    """The filter form of `completable`, read on every keystroke and render."""
    from prompt_toolkit.application.current import get_app
    from prompt_toolkit.filters import Condition

    @Condition
    def check() -> bool:
        return completable(get_app().current_buffer.document.text_before_cursor)

    return check


def _rule() -> Any:
    """The line under the input, redrawn with the rest of the prompt."""
    from prompt_toolkit.formatted_text import FormattedText

    return FormattedText([("class:rule", RULE_CHAR * _width())])


def _message() -> Any:
    """The prompt itself: a rule, then the caret the user types after.

    A callable rather than a value so the rule is re-measured on every draw,
    which is what makes it still span the terminal after a resize.
    """
    from prompt_toolkit.formatted_text import FormattedText

    return FormattedText(
        [("class:rule", RULE_CHAR * _width()), ("", "\n"), ("class:prompt", "› ")]
    )


def adds_text(state: Any) -> bool:
    """True when taking the first completion would change the line.

    A menu that is still open over a line the user has finished typing offers
    only what is already there; Enter has to submit in that case, not consume
    the keypress inserting nothing.
    """
    if state is None or state.complete_index is not None or not state.completions:
        return False
    completion = state.completions[0]
    typed = state.original_document.text_before_cursor[completion.start_position :]
    return bool(completion.text != typed)


def _bindings() -> Any:
    """Make the first completion the one Enter takes.

    prompt_toolkit opens the menu with nothing selected, so accepting the
    obvious suggestion costs an arrow key first. Here Enter takes the first
    entry when none is selected and the entry would actually add something --
    that last part is what keeps Enter submitting a finished line whose menu
    happens to still be open.
    """
    from prompt_toolkit.application.current import get_app
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.key_binding import KeyBindings

    @Condition
    def first_completion_is_useful() -> bool:
        return adds_text(get_app().current_buffer.complete_state)

    bindings = KeyBindings()

    @bindings.add("enter", filter=first_completion_is_useful)
    def _take_first(event: Any) -> None:
        event.current_buffer.go_to_completion(0)

    return bindings


class LineReader:
    """Reads prompt lines for one REPL session, with completion and history.

    Holding the session on an instance is what preserves history across turns:
    a `PromptSession` rebuilt for every line would forget the line before it.
    """

    def __init__(self, state: ReplState) -> None:
        self.state = state
        self.session: Any = None
        self._build()

    def _build(self) -> None:
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import FileHistory
        except ImportError:
            self.session = None
            return

        history: Any = None
        try:
            path = history_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            history = FileHistory(str(path))
        except OSError:
            history = None

        try:
            self.session = PromptSession(
                history=history,
                completer=build_completer(self.state),
                complete_while_typing=_completable(),
                key_bindings=_bindings(),
                style=_style(),
                bottom_toolbar=_rule,
                reserve_space_for_menu=6,
            )
        except Exception:
            self.session = None

    async def read(self) -> str:
        """Read one line, raising `EOFError` when the user is done.

        Ctrl-C abandons the line being typed rather than the session, which is
        what it does everywhere else; only Ctrl-D ends the REPL.
        """
        if self.session is None:
            return await ui.ask_text("\n›")

        try:
            line = await self.session.prompt_async(_message)
        except KeyboardInterrupt:
            ui.line()
            return ""
        ui.line()
        return line.strip()


__all__ = [
    "LineReader",
    "completable",
    "adds_text",
    "path_options",
    "build_completer",
    "command_names",
    "argument_options",
    "history_path",
]
