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
import re
from pathlib import Path
from typing import Any

from rich.syntax import Syntax
from rich.text import Text

from ..settings import SETTINGS_DIR, SETTINGS_FILE
from . import ui

OLLAMA_ENV = "OLLAMA_BASE_URL"
"""Where the Ollama provider reads the address of the local server."""

OLLAMA_URL = "http://localhost:11434/v1"
"""The address Ollama listens on out of the box."""

PROVIDER_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai-chat": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "google": "GOOGLE_API_KEY",
    "google-cloud": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "CO_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "zai": "ZAI_API_KEY",
    "moonshotai": "MOONSHOTAI_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "huggingface": "HF_TOKEN",
    "heroku": "HEROKU_INFERENCE_KEY",
    "bedrock": "AWS_ACCESS_KEY_ID",
    "azure": "AZURE_OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "github": "GITHUB_API_KEY",
    "vercel": "VERCEL_AI_GATEWAY_API_KEY",
    "gateway": "PYDANTIC_AI_GATEWAY_API_KEY",
    "alibaba": "ALIBABA_API_KEY",
    "nebius": "NEBIUS_API_KEY",
    "ovhcloud": "OVHCLOUD_API_KEY",
    "sambanova": "SAMBANOVA_API_KEY",
    "voyageai": "VOYAGE_API_KEY",
    "ollama": OLLAMA_ENV,
}
"""The environment variable each provider reads its credential from.

Ollama is the odd one out: what it needs is an address, not a secret, so the
wizard asks for it plainly and the value is safe to show.
"""

FEATURED = (
    ("anthropic", "Claude — strongest tool use"),
    ("openai", "GPT — broad availability"),
    ("groq", "open models, very fast"),
    ("google", "Gemini — long context"),
    ("ollama", "local models, no key and no bill"),
)
"""The providers worth trying first, shown at the top of the list."""

EMBEDDING_ONLY = frozenset({"sentence-transformers", "voyageai"})
"""Providers that serve embeddings, which no amount of prompting will answer."""

NOTES = {
    "alibaba": "Qwen",
    "azure": "OpenAI on Azure",
    "bedrock": "Claude and friends on AWS",
    "cerebras": "open models on custom silicon",
    "cohere": "Command",
    "deepseek": "DeepSeek",
    "google-cloud": "Gemini through Vertex AI",
    "heroku": "Heroku managed inference",
    "huggingface": "the Hub's inference providers",
    "mistral": "Mistral",
    "moonshotai": "Kimi",
    "fireworks": "open models, hosted",
    "github": "GitHub Models",
    "litellm": "your own LiteLLM proxy",
    "openai-chat": "OpenAI's older chat completions shape",
    "openrouter": "many providers behind one key",
    "together": "open models, hosted",
    "vercel": "the Vercel AI gateway",
    "xai": "Grok",
    "zai": "GLM",
}
"""A word on the providers that are not featured, where a word helps."""

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
    "google": (
        ("google:gemini-2.5-pro", "flagship"),
        ("google:gemini-2.5-flash", "fast and cheap"),
    ),
}
"""A short list per featured provider, so the common case is one keypress.

Ollama is absent on purpose: the only local models worth offering are the ones
already pulled, so `local_models` asks the server instead of guessing.
"""

MODES = (
    ("default", "ask before anything that changes the world"),
    ("acceptEdits", "auto-accept file edits, ask for the rest"),
    ("plan", "read-only; the agent can look but not touch"),
    ("bypassPermissions", "never ask — only in a sandbox you can lose"),
)


def ollama_url() -> str:
    """The Ollama address in force: the environment's, or the default."""
    return os.environ.get(OLLAMA_ENV, "").strip() or OLLAMA_URL


def normalise_url(url: str) -> str:
    """Return `url` as the OpenAI-compatible endpoint the provider expects.

    Ollama serves its own API at the root and an OpenAI-shaped one under
    ``/v1``. People type either, so both are accepted and one is stored.
    """
    trimmed = url.strip().rstrip("/")
    if not trimmed:
        return OLLAMA_URL
    if "://" not in trimmed:
        trimmed = f"http://{trimmed}"
    return trimmed if trimmed.endswith("/v1") else f"{trimmed}/v1"


def _size(count: Any) -> str:
    try:
        size = float(count)
    except (TypeError, ValueError):
        return ""
    if size <= 0:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _host(url: str) -> str:
    """Name the machine a model is really served from, for a cloud tag."""
    return f"via {url.split('://')[-1].strip('/')}"


def local_models(url: str = "") -> tuple[tuple[str, str], ...]:
    """Ask an Ollama server which models are pulled, newest first.

    A server that is not running is not an error worth reporting here -- the
    caller falls back to asking for a name -- so every failure is an empty
    list.
    """
    import httpx

    root = normalise_url(url or ollama_url()).removesuffix("/v1")
    try:
        response = httpx.get(f"{root}/api/tags", timeout=2.0)
        response.raise_for_status()
        listed = response.json().get("models", [])
    except Exception:
        return ()
    rows: list[tuple[str, str]] = []
    for entry in listed:
        name = str(entry.get("model") or entry.get("name") or "").strip()
        if not name:
            continue
        details = entry.get("details") or {}
        parameters = str(details.get("parameter_size") or "").strip()
        remote = str(entry.get("remote_host") or "").strip()
        facts = [
            parameters if parameters not in {"", "0"} else "",
            _host(remote) if remote else _size(entry.get("size")),
        ]
        rows.append((f"ollama:{name}", " · ".join(fact for fact in facts if fact)))
    return tuple(rows)


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
    """Return the credential variable a model string implies, if it is known.

    Every provider behind the gateway shares the gateway's own key, so a
    ``gateway/...`` prefix falls back to the entry for the gateway itself.
    """
    provider = model.split(":", 1)[0] if ":" in model else ""
    if not provider:
        return None
    return PROVIDER_ENV.get(provider) or PROVIDER_ENV.get(provider.split("/", 1)[0])


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


def _describes(name: str) -> str:
    """The one-line note a provider row carries."""
    if name.startswith("gateway/"):
        return f"{name.split('/', 1)[1]} through the pydantic-ai gateway"
    return NOTES.get(name) or PROVIDER_ENV.get(name, "")


def providers() -> list[tuple[str, str]]:
    """Every provider that can prefix a model, the featured ones first.

    The list is the one `resolve_model` actually accepts rather than a
    hand-kept copy of it, so a provider pydantic-ai gains shows up here without
    anybody remembering to add it. Each row says what it is, or failing that
    which variable it reads, since for most of them the credential is the only
    question the user has.
    """
    from ..models import known_providers

    featured = [name for name, _ in FEATURED]
    rest = sorted(
        name
        for name in known_providers()
        if name not in featured and name not in EMBEDDING_ONLY
    )
    return [*FEATURED, *((name, _describes(name)) for name in rest)]


async def _choose_provider() -> str:
    return await ui.ask_choice(
        "Which provider?", providers(), allow_other=True, default=1
    )


SNAPSHOT = re.compile(r"-(\d{8}|\d{4}-\d{2}-\d{2}|v\d[\d.]*)$")
"""The tail that marks a pinned release of a model rather than its moving name."""


def _model_note(name: str) -> str:
    """A word on a model the registry names but nobody described.

    Only the rows worth warning about get one: a pin that will never improve,
    and the safety and embedding models that share the namespace with the chat
    models but cannot answer a prompt.
    """
    identifier = name.split(":", 1)[-1].lower()
    if "guard" in identifier or "moderation" in identifier:
        return "safety filter, not a chat model"
    if "embed" in identifier or identifier.startswith("tts") or "whisper" in identifier:
        return "not a chat model"
    if SNAPSHOT.search(identifier):
        return "pinned release"
    return ""


def models_for(provider: str) -> list[tuple[str, str]]:
    """Every model `provider` offers, the ones worth trying first at the top.

    The curated rows carry the judgement -- which model is the sensible default,
    which is cheap -- and the registry supplies the rest, so choosing an
    unusual model does not mean leaving the menu for a text prompt. Ollama is
    its own case: what it serves is whatever the user has pulled.
    """
    if provider == "ollama":
        return list(local_models())

    from ..models import known_models

    curated = list(SUGGESTED.get(provider, ()))
    named = {name for name, _ in curated}
    prefix = f"{provider}:"
    rest = sorted(
        name
        for name in known_models()
        if name.startswith(prefix) and name not in named
    )
    return [*curated, *((name, _model_note(name)) for name in rest)]


KEPT = frozenset({"", "y", "yes", "ok", "keep"})
"""Answers that mean 'the address already shown is the right one'."""

CHANGED = frozenset({"n", "no"})
"""Answers that mean 'not that one', without saying what instead."""


async def _choose_ollama() -> str:
    """Settle the address of the local server before anything is asked of it.

    The prompt takes either answer to the same question: `y` or Enter keeps the
    address it shows, and a url typed in its place is that answer too, so the
    usual case is one keypress and the unusual one is still one prompt. `n`
    earns the second prompt, since it says only that the address is wrong.
    """
    shown = ollama_url()
    reply = await ui.ask_text(
        f"  use {shown}? (y to keep, or type another url)", default="y"
    )
    if reply.lower() in CHANGED:
        reply = await ui.ask_text("  ollama server url", default=OLLAMA_URL)
    url = shown if reply.lower() in KEPT else normalise_url(reply)
    os.environ[OLLAMA_ENV] = url
    ui.note(f"  using {url}", "muted")
    return url


async def _choose_model(provider: str) -> str:
    options = models_for(provider)
    if not options:
        if provider == "ollama":
            ui.note(
                f"  no models answered at {ollama_url()} — `ollama pull qwen3` first,"
                " or name one anyway",
                "warn",
            )
        return await ui.ask_text(
            f"  model identifier for {provider}", default=f"{provider}:"
        )
    question = "Which local model?" if provider == "ollama" else f"Which {provider} model?"
    return await ui.ask_choice(question, options, allow_other=True, default=1)


async def run_wizard(*, first_run: bool = False) -> dict[str, Any]:
    """Walk the user through configuration and save it. Returns the settings.

    Returning the settings rather than reading them back is what lets the
    caller start a session in the same breath as configuring it.
    """
    from ..models import known_models

    ui.banner(
        f"a coding agent on {len(known_models())} models"
        f" across {len(providers())} providers — local ones too"
    )
    if first_run:
        ui.note("  No model configured yet — let's fix that.\n", "key")

    provider = await _choose_provider()
    if not provider:
        return {}
    if provider == "ollama":
        await _choose_ollama()
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
    if env_var_for(model) == OLLAMA_ENV:
        os.environ[OLLAMA_ENV] = ollama_url()
        ui.note(f"\n  serving from {ollama_url()} — no key needed", "ok")
        credential_help(OLLAMA_ENV, ollama_url())
    elif variable:
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
    "local_models",
    "models_for",
    "providers",
    "normalise_url",
    "ollama_url",
    "PROVIDER_ENV",
    "OLLAMA_ENV",
    "OLLAMA_URL",
]
