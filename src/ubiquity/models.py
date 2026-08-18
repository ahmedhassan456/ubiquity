"""Model and provider resolution.

Model selection is delegated to pydantic-ai, which is what makes the tool
suite and permission system usable against any supported backend. Nothing in
this module names a vendor: there is no built-in default model and no built-in
shorthand for one, because baking either in would quietly privilege one
provider over the rest.

Accepted forms for `Options.model`:
    ``provider:model``   explicit, for example ``openai:gpt-5``
    ``model``            inferred by pydantic-ai from the bare name
    ``alias``            a user-registered shorthand, expanded first
    ``Model`` instance   passed through, for custom client configuration

Providers reachable out of the box include anthropic, openai, google,
google-cloud, bedrock, groq, mistral, cohere, deepseek, xai, huggingface,
cerebras, moonshotai, zai, heroku, and the gateway variants. Any
OpenAI-compatible endpoint — Ollama, OpenRouter, vLLM, LM Studio, Together —
is reachable through `openai_compatible`.

Two environment variables configure this layer:
    ``UBIQUITY_MODEL``          the model used when `Options.model` is unset
    ``UBIQUITY_MODEL_ALIASES``  comma-separated ``alias=target`` pairs
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic_ai.models import Model

logger = logging.getLogger("ubiquity")

MODEL_ENV_VAR = "UBIQUITY_MODEL"
ALIASES_ENV_VAR = "UBIQUITY_MODEL_ALIASES"

_ALIASES: dict[str, str] = {}
_ENV_ALIASES_LOADED = False


def _parse_alias_env(raw: str) -> dict[str, str]:
    """Parse ``a=x,b=y`` into an alias mapping, ignoring malformed entries."""
    parsed: dict[str, str] = {}
    for entry in raw.split(","):
        alias, sep, target = entry.partition("=")
        if sep and alias.strip() and target.strip():
            parsed[alias.strip()] = target.strip()
    return parsed


def _ensure_env_aliases() -> None:
    """Fold `UBIQUITY_MODEL_ALIASES` into the registry, once per process."""
    global _ENV_ALIASES_LOADED
    if _ENV_ALIASES_LOADED:
        return
    _ENV_ALIASES_LOADED = True
    raw = os.environ.get(ALIASES_ENV_VAR)
    if raw:
        for alias, target in _parse_alias_env(raw).items():
            _ALIASES.setdefault(alias, target)


def register_alias(alias: str, target: str) -> None:
    """Register a process-wide shorthand for a model identifier.

    Aliases let a codebase refer to a role rather than a vendor, so the
    binding can change without touching call sites:
    ``register_alias("fast", "groq:llama-3.3-70b-versatile")``.
    """
    _ensure_env_aliases()
    _ALIASES[alias] = target


def registered_aliases() -> dict[str, str]:
    """Return the current alias registry, including any from the environment."""
    _ensure_env_aliases()
    return dict(_ALIASES)


def clear_aliases() -> None:
    """Drop every registered alias, including those read from the environment."""
    global _ENV_ALIASES_LOADED
    _ALIASES.clear()
    _ENV_ALIASES_LOADED = True


def expand_alias(name: str, aliases: dict[str, str] | None = None) -> str:
    """Expand `name` through `aliases` then the global registry.

    Run-scoped aliases take precedence over registered ones. Expansion is
    resolved transitively so an alias may point at another alias, with a depth
    cap so a cycle degrades to the last name reached rather than hanging.
    """
    _ensure_env_aliases()
    seen: set[str] = set()
    current = name
    for _ in range(10):
        if current in seen:
            return current
        seen.add(current)
        target = (aliases or {}).get(current) or _ALIASES.get(current)
        if target is None:
            return current
        current = target
    return current


def default_model_spec() -> str | None:
    """Return the model named by `UBIQUITY_MODEL`, or None when it is unset."""
    value = os.environ.get(MODEL_ENV_VAR, "").strip()
    return value or None


def resolve_model(
    model: str | Model,
    aliases: dict[str, str] | None = None,
    provider_kwargs: dict[str, Any] | None = None,
) -> Model:
    """Return a pydantic-ai `Model` for `model`.

    A `Model` instance is returned unchanged. A string is expanded through the
    alias registry and then handed to pydantic-ai's inference, which accepts
    both ``provider:model`` and bare model names.

    `provider_kwargs` constructs the provider explicitly instead of letting it
    read the environment, which is the only way to give two concurrent runs
    different credentials. It is passed through pydantic-ai's `provider_factory`
    hook so the provider name still selects the model class, and the keywords
    reach the provider unfiltered: `TypeError` on a name the provider does not
    accept is the intended outcome.

    `http_client` is the one exception, dropped rather than raised on when the
    provider does not take one. It carries the retry policy, which this SDK
    adds by default and the caller did not necessarily ask for, and a provider
    that speaks to its backend through an SDK of its own rather than over HTTP
    should still be reachable. Anything the caller passed deliberately keeps
    failing loudly, because a credential silently ignored surfaces later as a
    confusing authentication error.
    """
    from pydantic_ai.models import Model as ModelBase
    from pydantic_ai.models import infer_model

    if isinstance(model, ModelBase):
        return model

    spec = expand_alias(model, aliases)
    if not provider_kwargs:
        return infer_model(spec)

    return infer_model(spec, lambda name: _build_provider(name, provider_kwargs))


def _build_provider(name: str, provider_kwargs: dict[str, Any]) -> Any:
    """Construct provider `name`, dropping an `http_client` it cannot accept."""
    from pydantic_ai.providers import infer_provider_class

    from .retry import accepts_http_client

    provider_class = infer_provider_class(name)
    kwargs = provider_kwargs
    if "http_client" in kwargs and not accepts_http_client(provider_class):
        kwargs = {k: v for k, v in kwargs.items() if k != "http_client"}
        logger.debug("provider %s takes no http_client; retries not applied", name)
    return provider_class(**kwargs)


def with_fallback(
    primary: str | Model,
    fallback: str | Model,
    aliases: dict[str, str] | None = None,
    provider_kwargs: dict[str, Any] | None = None,
) -> Model:
    """Return a model that retries `primary`'s failures against `fallback`.

    Failover is per request rather than per run, so a provider outage part way
    through a conversation costs one retry instead of the work already done.
    """
    from pydantic_ai.models.fallback import FallbackModel

    return FallbackModel(
        resolve_model(primary, aliases, provider_kwargs),
        resolve_model(fallback, aliases, provider_kwargs),
    )


def openai_compatible(
    model_name: str,
    *,
    base_url: str,
    api_key: str = "not-required",
    **kwargs: Any,
) -> Model:
    """Build a model backed by any OpenAI-compatible HTTP endpoint.

    Covers local and aggregator backends that expose the OpenAI chat API but
    are not registered providers, such as Ollama, vLLM, LM Studio, OpenRouter,
    and Together.

    Example:
        ``openai_compatible("llama3.3", base_url="http://localhost:11434/v1")``
    """
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=base_url, api_key=api_key),
        **kwargs,
    )


def known_models() -> list[str]:
    """Return every model identifier pydantic-ai recognizes by name."""
    from pydantic_ai.models import known_model_names

    return list(known_model_names())


def known_providers() -> list[str]:
    """Return the provider prefixes available for `provider:model` strings."""
    return sorted({name.split(":")[0] for name in known_models() if ":" in name})


def model_name_of(model: str | Model, aliases: dict[str, str] | None = None) -> str:
    """Return a display name for a model spec, without resolving credentials.

    Used for the init message and session records, where constructing a live
    client would be wasteful and could fail on a missing API key.
    """
    if isinstance(model, str):
        return expand_alias(model, aliases)
    return getattr(model, "model_name", str(model))


__all__ = [
    "resolve_model",
    "with_fallback",
    "openai_compatible",
    "known_models",
    "known_providers",
    "model_name_of",
    "register_alias",
    "registered_aliases",
    "clear_aliases",
    "expand_alias",
    "default_model_spec",
    "MODEL_ENV_VAR",
    "ALIASES_ENV_VAR",
]
