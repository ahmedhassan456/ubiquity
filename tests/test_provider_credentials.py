"""Credentials passed per run instead of through the process environment.

A provider reads its key from `os.environ` when nothing is passed, which is
process-global: two concurrent runs cannot use different keys, and a library
that wrote to `os.environ` on their behalf would make that worse. `api_key`
and `provider_kwargs` construct the provider explicitly instead.
"""

from __future__ import annotations

import pytest
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import FunctionModel

from ubiquity import Options, resolve_model, with_fallback


def _no_groq_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the environment key, so anything that works did so explicitly."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def credentials(options: Options) -> dict[str, object] | None:
    """The provider settings minus the retry transport, which carries no secret."""
    settings = options.provider_settings()
    if settings is None:
        return None
    return {k: v for k, v in settings.items() if k != "http_client"}


def test_no_credentials_configured_means_no_credentials_passed() -> None:
    """A run that sets neither field still lets the provider read the environment."""
    assert credentials(Options(model="test")) == {}


def test_retries_can_be_turned_off_back_to_no_provider_settings_at_all() -> None:
    """Nothing configured and no retries means the provider is built as before."""
    assert Options(model="test", max_retries=0).provider_settings() is None


def test_api_key_becomes_a_provider_keyword() -> None:
    assert credentials(Options(model="test", api_key="sk-1")) == {"api_key": "sk-1"}


def test_provider_kwargs_are_passed_through_verbatim() -> None:
    options = Options(
        model="azure:gpt-4o",
        provider_kwargs={"azure_endpoint": "https://x", "api_version": "2024-10-21"},
    )
    assert credentials(options) == {
        "azure_endpoint": "https://x",
        "api_version": "2024-10-21",
    }


def test_an_explicit_api_key_in_provider_kwargs_wins() -> None:
    """The dict is the escape hatch, so it decides when the two disagree."""
    options = Options(
        model="test", api_key="sk-field", provider_kwargs={"api_key": "sk-dict"}
    )
    assert credentials(options) == {"api_key": "sk-dict"}


def test_the_key_is_kept_out_of_the_repr() -> None:
    """`Options` lands in tracebacks and logs; a key in its repr leaks there."""
    text = repr(Options(model="test", api_key="sk-secret", provider_kwargs={"api_key": "sk-secret"}))
    assert "sk-secret" not in text


def test_a_key_resolves_a_model_with_no_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_groq_key(monkeypatch)
    with pytest.raises(Exception):
        resolve_model("groq:llama-3.3-70b-versatile")

    model = resolve_model(
        "groq:llama-3.3-70b-versatile", None, {"api_key": "sk-explicit"}
    )
    assert model.model_name == "llama-3.3-70b-versatile"
    assert model.system == "groq"


def test_the_provider_name_still_selects_the_model_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Credentials go through `provider_factory`, so inference is unchanged."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    model = resolve_model("anthropic:claude-sonnet-4-5", None, {"api_key": "sk-a"})
    assert type(model).__name__ == "AnthropicModel"


def test_an_alias_is_expanded_before_the_provider_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_groq_key(monkeypatch)
    model = resolve_model(
        "fast", {"fast": "groq:llama-3.3-70b-versatile"}, {"api_key": "sk-1"}
    )
    assert model.model_name == "llama-3.3-70b-versatile"


def test_a_constructed_model_is_returned_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Credentials configure inference; a caller who did it themselves is done."""
    built = FunctionModel(lambda messages, info: None)  # type: ignore[arg-type,return-value]
    assert resolve_model(built, None, {"api_key": "sk-1"}) is built


def test_a_keyword_the_provider_does_not_take_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loudly, and naming the keyword.

    Filtering to the arguments a provider accepts would turn a typo into a
    silent fall back to the environment, which surfaces much later as an
    authentication error that names nothing.
    """
    _no_groq_key(monkeypatch)
    with pytest.raises(TypeError, match="azure_endpoint"):
        resolve_model(
            "groq:llama-3.3-70b-versatile",
            None,
            {"api_key": "sk-1", "azure_endpoint": "https://x"},
        )


def test_credentials_reach_both_sides_of_a_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_groq_key(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    model = with_fallback(
        "groq:llama-3.3-70b-versatile",
        "anthropic:claude-sonnet-4-5",
        None,
        {"api_key": "sk-1"},
    )
    assert isinstance(model, FallbackModel)
    assert [m.system for m in model.models] == ["groq", "anthropic"]
