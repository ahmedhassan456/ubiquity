"""HTTP-level retries for rate limits and provider overload.

pydantic-ai's `Agent(retries=...)` is a per-category budget for tool calls and
output validation. It never sees a 429, because a rate limit is refused by the
transport before a response exists to validate. Without something at this
layer, one refusal ends a run and everything it has already spent is lost --
the failure mode this SDK can least afford, since compaction and prompt
caching exist to make runs long, and long runs are what hit rate limits.

The retry is a transport wrapper rather than a loop around the agent. A
provider's SDK owns the request, so the only place both the status code and
the `Retry-After` header are visible is beneath it, and retrying there resends
one request instead of replaying a turn.

Only requests whose body is already in memory are retried. Every provider
sends JSON, so this is the normal case; a streaming upload cannot be replayed
and is failed rather than silently truncated.

Waits honor `Retry-After` when the provider sends one, since a server saying
when to return is better information than any local guess. Backoff is
otherwise exponential with full jitter, because synchronized retries from
concurrent runs reproduce the burst that caused the rate limit.
"""

from __future__ import annotations

import asyncio
import logging
import random
from email.utils import parsedate_to_datetime
from typing import Any
from weakref import WeakKeyDictionary

logger = logging.getLogger("ubiquity")

DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_WAIT = 60.0
BASE_DELAY = 0.5

RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})

_CLIENTS: WeakKeyDictionary[Any, dict[tuple[int, float], Any]] = WeakKeyDictionary()


def retry_after_seconds(response: Any) -> float | None:
    """Return the wait a response asks for, or None if it asks for none.

    `Retry-After` is specified in two forms, delta-seconds and an HTTP date,
    and providers use both. A date in the past yields zero rather than a
    negative wait, which is what a clock skewed against the server's produces.
    """
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    raw = raw.strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    from datetime import datetime, timezone

    now = datetime.now(parsed.tzinfo or timezone.utc)
    return max((parsed - now).total_seconds(), 0.0)


def backoff_delay(attempt: int, max_wait: float) -> float:
    """Return the wait before retry number `attempt`, counting from zero.

    Full jitter: a uniform draw from zero to the exponential ceiling, rather
    than the ceiling itself. Several runs that hit the same rate limit at the
    same moment would otherwise retry at the same moment too, rebuilding the
    burst that got them limited.
    """
    ceiling = min(BASE_DELAY * (2**attempt), max_wait)
    return random.uniform(0.0, ceiling)


def _replayable(request: Any) -> bool:
    """Whether `request` still holds its body and can therefore be resent."""
    try:
        request.content
    except Exception:
        return False
    return True


class RetryTransport:
    """An httpx transport that retries rate limits and transient failures.

    Wraps another transport rather than replacing it, so connection pooling,
    proxies, and TLS configuration are whatever the wrapped transport does.

    A retryable response is read and closed before the next attempt. Error
    bodies are small, and leaving one open would hold a pooled connection for
    the whole backoff.

    A `Retry-After` longer than `max_wait` ends the retries and returns the
    response. Sleeping for the cap and asking again would spend an attempt to
    be refused on the same grounds, and a provider that says to come back in
    fifteen minutes is better reported than waited out.
    """

    def __init__(
        self,
        wrapped: Any,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_wait: float = DEFAULT_MAX_WAIT,
    ) -> None:
        self.wrapped = wrapped
        self.max_retries = max_retries
        self.max_wait = max_wait

    async def handle_async_request(self, request: Any) -> Any:
        """Send `request`, retrying it while the failure looks transient."""
        import httpx

        transient = (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.RemoteProtocolError,
        )

        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            try:
                response = await self.wrapped.handle_async_request(request)
            except transient as exc:
                if last or not _replayable(request):
                    raise
                delay = backoff_delay(attempt, self.max_wait)
                logger.debug(
                    "retrying %s after %s in %.2fs",
                    request.url,
                    type(exc).__name__,
                    delay,
                )
                await asyncio.sleep(delay)
                continue

            if response.status_code not in RETRY_STATUSES:
                return response
            if last or not _replayable(request):
                return response

            asked = retry_after_seconds(response)
            if asked is not None and asked > self.max_wait:
                logger.debug(
                    "not retrying %s: Retry-After %.0fs exceeds the %.0fs cap",
                    request.url,
                    asked,
                    self.max_wait,
                )
                return response

            await response.aread()
            await response.aclose()
            delay = asked if asked is not None else backoff_delay(attempt, self.max_wait)
            logger.debug(
                "retrying %s after HTTP %d in %.2fs",
                request.url,
                response.status_code,
                delay,
            )
            await asyncio.sleep(delay)

        raise RuntimeError("unreachable")

    async def aclose(self) -> None:
        """Close the wrapped transport."""
        await self.wrapped.aclose()


def retry_client(
    max_retries: int = DEFAULT_MAX_RETRIES, max_wait: float = DEFAULT_MAX_WAIT
) -> Any:
    """Return a shared `httpx.AsyncClient` that retries transient failures.

    Clients are cached per event loop and per setting. Per setting because two
    runs asking for the same behavior should share a connection pool; per loop
    because a pool holds sockets registered with the loop that opened them, and
    a client reused across `asyncio.run` boundaries hands the second run
    connections belonging to a closed loop.

    The cache is keyed weakly on the loop, so the clients for a loop become
    collectable when it does.
    """
    import httpx

    transport = RetryTransport(
        httpx.AsyncHTTPTransport(), max_retries=max_retries, max_wait=max_wait
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return httpx.AsyncClient(transport=transport)

    per_loop = _CLIENTS.setdefault(loop, {})
    key = (max_retries, max_wait)
    if key not in per_loop:
        per_loop[key] = httpx.AsyncClient(transport=transport)
    return per_loop[key]


def accepts_http_client(provider_class: Any) -> bool:
    """Whether a pydantic-ai provider class can be given an `http_client`.

    Providers that do not speak HTTP directly, Bedrock's boto3 client being
    the one that matters, take no such argument. Asking first is what keeps a
    default retry policy from turning those providers into a `TypeError`.
    """
    import inspect

    try:
        params = inspect.signature(provider_class.__init__).parameters
    except (TypeError, ValueError):
        return False
    if "http_client" in params:
        return True
    return any(p.kind is p.VAR_KEYWORD for p in params.values())


__all__ = [
    "RetryTransport",
    "RETRY_STATUSES",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_WAIT",
    "retry_after_seconds",
    "backoff_delay",
    "retry_client",
    "accepts_http_client",
]
