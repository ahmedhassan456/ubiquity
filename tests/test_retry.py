"""Tests for HTTP-level retries on rate limits and provider overload."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest

from ubiquity import Options
from ubiquity.retry import (
    BASE_DELAY,
    DEFAULT_MAX_WAIT,
    RETRY_STATUSES,
    RetryTransport,
    accepts_http_client,
    backoff_delay,
    retry_after_seconds,
    retry_client,
)


@pytest.fixture
def slept(monkeypatch) -> list[float]:
    """Record every backoff without actually waiting for it."""
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr("ubiquity.retry.asyncio.sleep", fake_sleep)
    return waits


class Backend:
    """A transport that replays a scripted sequence of outcomes."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def aclose(self) -> None:
        return None


def response(status: int, **headers: str) -> httpx.Response:
    return httpx.Response(status, headers=headers, content=b"{}")


def streaming(status: int, **headers: str) -> httpx.Response:
    """A response whose body is still open, as a real transport returns one."""

    async def body():
        yield b"{}"

    return httpx.Response(status, headers=headers, content=body())


def request() -> httpx.Request:
    return httpx.Request("POST", "https://api.example.com/v1/messages", json={"a": 1})


async def send(backend: Backend, **kwargs: Any) -> httpx.Response:
    transport = RetryTransport(backend, **kwargs)
    return await transport.handle_async_request(request())


class TestRetryAfter:
    def test_delta_seconds_are_parsed(self) -> None:
        assert retry_after_seconds(response(429, **{"retry-after": "30"})) == 30.0

    def test_a_missing_header_asks_for_nothing(self) -> None:
        assert retry_after_seconds(response(429)) is None

    def test_an_http_date_is_parsed(self) -> None:
        soon = datetime.now(timezone.utc) + timedelta(seconds=45)
        stamp = soon.strftime("%a, %d %b %Y %H:%M:%S GMT")
        assert retry_after_seconds(response(429, **{"retry-after": stamp})) == pytest.approx(
            45, abs=2
        )

    def test_a_date_in_the_past_asks_for_no_wait(self) -> None:
        """A clock skewed against the server's must not produce a negative wait."""
        past = datetime.now(timezone.utc) - timedelta(seconds=300)
        stamp = past.strftime("%a, %d %b %Y %H:%M:%S GMT")
        assert retry_after_seconds(response(429, **{"retry-after": stamp})) == 0.0

    def test_a_negative_delta_asks_for_no_wait(self) -> None:
        """`asyncio.sleep` on a negative number would skip the backoff entirely."""
        assert retry_after_seconds(response(429, **{"retry-after": "-5"})) == 0.0

    def test_garbage_is_ignored(self) -> None:
        assert retry_after_seconds(response(429, **{"retry-after": "soon"})) is None


class TestBackoff:
    def test_the_delay_never_exceeds_the_ceiling(self) -> None:
        for attempt in range(6):
            ceiling = min(BASE_DELAY * (2**attempt), DEFAULT_MAX_WAIT)
            assert 0.0 <= backoff_delay(attempt, DEFAULT_MAX_WAIT) <= ceiling

    def test_the_cap_bounds_every_attempt(self) -> None:
        assert backoff_delay(20, 5.0) <= 5.0

    def test_jitter_desynchronizes_concurrent_runs(self) -> None:
        """Identical delays would rebuild the burst that caused the rate limit."""
        draws = {backoff_delay(6, DEFAULT_MAX_WAIT) for _ in range(50)}
        assert len(draws) > 1


class TestRetrying:
    async def test_a_rate_limit_is_retried(self, slept) -> None:
        backend = Backend(response(429), response(200))
        result = await send(backend, max_retries=3)
        assert result.status_code == 200
        assert backend.calls == 2

    async def test_overload_is_retried(self, slept) -> None:
        """529 is how Anthropic reports overload, and it is always transient."""
        backend = Backend(response(529), response(529), response(200))
        assert (await send(backend, max_retries=3)).status_code == 200
        assert backend.calls == 3

    async def test_success_is_not_retried(self, slept) -> None:
        backend = Backend(response(200))
        await send(backend, max_retries=3)
        assert backend.calls == 1
        assert slept == []

    async def test_a_bad_request_is_not_retried(self, slept) -> None:
        """A malformed request fails the same way however many times it is sent."""
        backend = Backend(response(400))
        assert (await send(backend, max_retries=3)).status_code == 400
        assert backend.calls == 1

    async def test_the_budget_is_finite(self, slept) -> None:
        backend = Backend(response(429))
        assert (await send(backend, max_retries=2)).status_code == 429
        assert backend.calls == 3

    async def test_zero_retries_sends_once(self, slept) -> None:
        backend = Backend(response(429))
        assert (await send(backend, max_retries=0)).status_code == 429
        assert backend.calls == 1

    async def test_retry_after_is_obeyed_exactly(self, slept) -> None:
        """A server saying when to return beats any local guess."""
        backend = Backend(response(429, **{"retry-after": "7"}), response(200))
        await send(backend, max_retries=3, max_wait=60.0)
        assert slept == [7.0]

    async def test_a_wait_longer_than_the_cap_ends_the_retries(self, slept) -> None:
        """Sleeping for the cap would spend an attempt to be refused again."""
        backend = Backend(response(429, **{"retry-after": "900"}), response(200))
        result = await send(backend, max_retries=3, max_wait=60.0)
        assert result.status_code == 429
        assert backend.calls == 1
        assert slept == []

    async def test_backoff_is_used_when_no_wait_is_asked_for(self, slept) -> None:
        backend = Backend(response(503), response(200))
        await send(backend, max_retries=3, max_wait=60.0)
        assert len(slept) == 1
        assert 0.0 <= slept[0] <= BASE_DELAY

    async def test_a_transient_network_error_is_retried(self, slept) -> None:
        backend = Backend(httpx.ConnectError("refused"), response(200))
        assert (await send(backend, max_retries=3)).status_code == 200
        assert backend.calls == 2

    async def test_a_network_error_past_the_budget_propagates(self, slept) -> None:
        """The original error is what the caller must see, not a wrapper."""
        backend = Backend(httpx.ConnectError("refused"))
        with pytest.raises(httpx.ConnectError):
            await send(backend, max_retries=1)
        assert backend.calls == 2

    async def test_an_unexpected_error_is_not_retried(self, slept) -> None:
        backend = Backend(ValueError("bug"))
        with pytest.raises(ValueError):
            await send(backend, max_retries=3)
        assert backend.calls == 1

    async def test_the_error_body_is_released_before_backing_off(self, slept) -> None:
        """An open error body holds a pooled connection for the whole wait."""
        failure = streaming(429)
        assert not failure.is_closed
        backend = Backend(failure, response(200))
        await send(backend, max_retries=3)
        assert failure.is_closed

    async def test_a_successful_body_is_left_for_the_caller_to_stream(
        self, slept
    ) -> None:
        """Reading it here would consume the stream the provider is about to."""
        ok = streaming(200)
        await send(Backend(ok), max_retries=3)
        assert not ok.is_stream_consumed

    async def test_every_retryable_status_is_retried(self, slept) -> None:
        for status in RETRY_STATUSES:
            backend = Backend(response(status), response(200))
            assert (await send(backend, max_retries=1)).status_code == 200, status


class TestWiring:
    def test_retries_reach_every_provider_in_the_run(self) -> None:
        """One policy covers the model, the fallback, and the compaction model."""
        settings = Options(max_retries=3).provider_settings()
        assert settings is not None
        assert isinstance(settings["http_client"], httpx.AsyncClient)

    def test_zero_retries_adds_no_transport(self) -> None:
        assert Options(max_retries=0).provider_settings() is None

    def test_a_caller_supplied_client_is_left_alone(self) -> None:
        mine = httpx.AsyncClient()
        settings = Options(provider_kwargs={"http_client": mine}).provider_settings()
        assert settings is not None
        assert settings["http_client"] is mine

    def test_the_api_key_still_travels_with_the_client(self) -> None:
        settings = Options(api_key="sk-test").provider_settings()
        assert settings is not None
        assert settings["api_key"] == "sk-test"
        assert "http_client" in settings

    async def test_clients_are_shared_within_an_event_loop(self) -> None:
        assert retry_client(3, 60.0) is retry_client(3, 60.0)

    async def test_different_policies_do_not_share_a_client(self) -> None:
        assert retry_client(3, 60.0) is not retry_client(5, 60.0)

    def test_a_provider_that_speaks_http_takes_a_client(self) -> None:
        from pydantic_ai.providers.anthropic import AnthropicProvider

        assert accepts_http_client(AnthropicProvider)

    def test_a_provider_without_one_is_detected_rather_than_crashed_into(self) -> None:
        """Bedrock talks through boto3, so injecting a client would be a TypeError."""

        class Boto3Shaped:
            def __init__(self, region_name: str | None = None) -> None: ...

        assert not accepts_http_client(Boto3Shaped)
