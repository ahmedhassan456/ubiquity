"""First-run configuration.

The wizard settles three things: which provider, which model, and whether the
credential that provider needs is present. What it writes is
``~/.ubiquity/settings.json`` -- the SDK's own user settings file, not a
private format -- so a model chosen here is the same `model` key that
`Options(setting_sources=["user"])` reads, and anything else that honours those
files sees the choice too.

The API key is deliberately not written. A key in a config file outlives every
intention anybody had for it: it gets committed, synced, and backed up. The
wizard checks the provider's environment variable, and when it is missing it
prints the export line for the user's shell profile and leaves the secret in
the user's hands.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rich.syntax import Syntax
from rich.text import Text

from ..settings import SETTINGS_DIR, SETTINGS_FILE
from . import ui

PROVIDER_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "google-gla": "GOOGLE_API_KEY",
    "google-vertex": "GOOGLE_APPLICATION_CREDENTIALS",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "CO_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "grok": "GROK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "huggingface": "HF_TOKEN",
    "bedrock": "AWS_ACCESS_KEY_ID",
    "azure": "AZURE_OPENAI_API_KEY",
}
"""The environment variable each provider reads its credential from."""

FEATURED = (
    ("anthropic", "Claude — strongest tool use"),
    ("openai", "GPT — broad availability"),
    ("groq", "open models, very fast"),
    ("google-gla", "Gemini — long context"),
)
"""Providers offered by name; the rest are reachable through 'other'."""

SUGGESTED: dict[str, tuple[tuple[str, str], ...]] = {
    "anthropic": (
        ("anthropic:claude-sonnet-4-5", "balanced, the usual default"),
        ("anthropic:claude-opus-4-1", "hardest reasoning"),
        ("anthropic:claude-haiku-4-5", "cheap and quick"),
    ),
    "openai": (
        ("openai:gpt-5", "flagship"),
        ("openai:gpt-5-mini", "cheaper, still capable"),
        ("openai:o4-mini", "reasoning, low cost"),
    ),
    "groq": (
        ("groq:openai/gpt-oss-120b", "open weights, fast"),
        ("groq:llama-3.3-70b-versatile", "general purpose"),
        ("groq:moonshotai/kimi-k2-instruct", "strong at tools"),
    ),
    "google-gla": (
        ("google-gla:gemini-2.5-pro", "flagship"),
        ("google-gla:gemini-2.5-flash", "fast and cheap"),
    ),
}
"""A short list per featured provider, so the common case is one keypress."""

MODES = (
    ("default", "ask before anything that changes the world"),
    ("acceptEdits", "auto-accept file edits, ask for the rest"),
    ("plan", "read-only; the agent can look but not touch"),
    ("bypassPermissions", "never ask — only in a sandbox you can lose"),
)


def config_path() -> Path:
    """Return the user settings file the wizard writes."""
    return Path.home() / SETTINGS_DIR / SETTINGS_FILE


def load_config() -> dict[str, Any]:
    """Read the user settings file, treating an unreadable one as empty."""
    path = config_path()
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def save_config(changes: dict[str, Any]) -> Path:
    """Merge `changes` into the user settings file and return its path.

    Existing keys the wizard does not ask about -- permission rules, `env`,
    anything a user hand-wrote -- are preserved, since this file is shared with
    the SDK rather than owned by the CLI.
    """
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = load_config()
    for key, value in changes.items():
        if key == "permissions" and isinstance(value, dict):
            merged["permissions"] = {**merged.get("permissions", {}), **value}
        else:
            merged[key] = value
    path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return path


def is_configured() -> bool:
    """True when a run could start without asking the user anything.

    An explicit `UBIQUITY_MODEL` counts, and so does a `model` in the settings
    file. Nothing else is required: a missing API key is reported by the
    provider, and guessing at credentials here would mean claiming a run will
    fail before trying it.
    """
    return bool(os.environ.get("UBIQUITY_MODEL", "").strip() or load_config().get("model"))


def env_var_for(model: str) -> str | None:
    """Return the credential variable a model string implies, if it is known."""
    provider = model.split(":", 1)[0] if ":" in model else ""
    return PROVIDER_ENV.get(provider)


def credential_missing(model: str) -> str | None:
    """Return the unset environment variable this model needs, or None."""
    variable = env_var_for(model)
    if variable and not os.environ.get(variable, "").strip():
        return variable
    return None


def _export_line(variable: str, value: str = "") -> str:
    return f'export {variable}="{value or "your-key-here"}"'


def shell_profile() -> Path:
    """Guess the file the user's shell reads on login."""
    shell = Path(os.environ.get("SHELL", "")).name
    home = Path.home()
    if shell == "zsh":
        return home / ".zshrc"
    if shell == "fish":
        return home / ".config" / "fish" / "config.fish"
    return home / ".bashrc"


def credential_help(variable: str, value: str = "") -> None:
    """Show the user how to put a key in their environment."""
    profile = shell_profile()
    line = _export_line(variable, value)
    ui.console().print()
    ui.panel(
        Syntax(f"{line}\n", "bash", theme="ansi_dark", background_color="default"),
        title=f"add to {profile.name}, or run in this shell",
        style="warn",
    )
    ui.note(f"  echo '{line}' >> {profile}", "muted")
    ui.note("  then reopen the shell, or `source` that file", "muted")


async def _choose_provider() -> str:
    options = [*FEATURED, ("other", "any other pydantic-ai provider")]
    choice = await ui.ask_choice(
        "Which provider?", options, allow_other=True, default=1
    )
    if choice != "other":
        return choice

    from ..models import known_providers

    names = known_providers()
    ui.note(f"  {len(names)} providers: {', '.join(names)}", "muted")
    return await ui.ask_text("  provider")


async def _choose_model(provider: str) -> str:
    suggestions = SUGGESTED.get(provider)
    if not suggestions:
        return await ui.ask_text(
            f"  model identifier for {provider}", default=f"{provider}:"
        )
    return await ui.ask_choice(
        f"Which {provider} model?",
        list(suggestions),
        allow_other=True,
        default=1,
    )


async def run_wizard(*, first_run: bool = False) -> dict[str, Any]:
    """Walk the user through configuration and save it. Returns the settings.

    Returning the settings rather than reading them back is what lets the
    caller start a session in the same breath as configuring it.
    """
    ui.banner("a coding agent on 600+ models, 22 providers")
    if first_run:
        ui.note("  No model configured yet — let's fix that.\n", "key")

    provider = await _choose_provider()
    if not provider:
        return {}
    model = await _choose_model(provider)
    if not model:
        return {}

    mode = await ui.ask_choice(
        "How should permissions work by default?", list(MODES), default=1
    )
    changes: dict[str, Any] = {"model": model}
    if mode and mode != "default":
        changes["permissions"] = {"defaultMode": mode}

    path = save_config(changes)
    ui.console().print()
    ui.key_values(
        [("model", model), ("mode", mode or "default"), ("saved to", str(path))],
        title="configured",
    )

    variable = credential_missing(model)
    if variable:
        ui.note(f"\n  {variable} is not set in this shell.", "warn")
        key = await ui.ask_text(f"  paste your {variable} (or leave blank)", password=True)
        if key:
            os.environ[variable] = key
            ui.note("  set for this session only — make it permanent with:", "ok")
            credential_help(variable, key)
        else:
            credential_help(variable)
    elif env_var_for(model):
        ui.note(f"\n  ✓ {env_var_for(model)} is set", "ok")

    return changes


__all__ = [
    "run_wizard",
    "is_configured",
    "load_config",
    "save_config",
    "config_path",
    "credential_missing",
    "credential_help",
    "env_var_for",
    "PROVIDER_ENV",
]
