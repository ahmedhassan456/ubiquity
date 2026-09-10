"""The shared HTTP layer behind `WebSearch` and `WebFetch`.

Everything the web tools do goes through `fetch_url`, which exists to make two
promises the tools depend on.

The first is that no request needs a key. Every search engine reached from here
is a public endpoint, so the tools work on a fresh install with nothing
configured.

The second is that a request is hard to block and never blocks. Hosts that
refuse a request do it on fingerprint reputation, so a retry after a 403 or a
429 presents a different browser user agent rather than repeating the one that
was just refused; that alone clears most transient blocks. And every call is
async with a hard timeout, so a slow host delays its own result and nothing
else -- the search tool queries its engines concurrently and keeps whatever
comes back in time.

Requests are also guarded against pointing back inward. A URL that resolves to
a private, loopback, or link-local address is refused before it is sent, on the
original URL and again on every redirect hop, so a tool call cannot be steered
into the machine's own network.
"""

from __future__ import annotations

import asyncio
import ipaddress
import random
import socket
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

import httpx

USER_AGENTS = (
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.4 Safari/605.1.15"
    ),
)

DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

BLOCKED_STATUSES = frozenset({401, 403, 429})
RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})

DEFAULT_TIMEOUT_S = 15.0
MAX_REDIRECTS = 5
MAX_ATTEMPTS = 3
MAX_BODY_BYTES = 5 * 1024 * 1024


class WebError(Exception):
    """A request that failed in a way the caller should report, not retry."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def next_user_agent(previous: str | None = None) -> str:
    """Pick a user agent, guaranteed to differ from `previous` when given.

    A cold request draws at random so concurrent engines do not all present
    the same fingerprint; a retry walks to the next agent in the pool so the
    host sees a different browser than the one it just refused.
    """
    if previous is None:
        return random.choice(USER_AGENTS)
    try:
        index = USER_AGENTS.index(previous)
    except ValueError:
        return USER_AGENTS[0]
    return USER_AGENTS[(index + 1) % len(USER_AGENTS)]


def normalize_url(url: str) -> str:
    """Return `url` with a scheme, lowercased host, and no fragment."""
    candidate = url.strip()
    if candidate.startswith("//"):
        candidate = f"https:{candidate}"
    if not candidate.lower().startswith(("http://", "https://")):
        candidate = f"https://{candidate}"
    parts = urlsplit(candidate)
    host = parts.hostname or ""
    netloc = host.lower()
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    if parts.username:
        credentials = parts.username
        if parts.password:
            credentials = f"{credentials}:{parts.password}"
        netloc = f"{credentials}@{netloc}"
    return urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", parts.query, ""))


def dedup_key(url: str) -> str:
    """A comparison key that treats trivial URL variants as one page."""
    parts = urlsplit(normalize_url(url))
    host = (parts.hostname or "").removeprefix("www.")
    path = parts.path.rstrip("/") or "/"
    return f"{host}{path}?{parts.query}" if parts.query else f"{host}{path}"


def host_of(url: str) -> str:
    """The hostname of `url`, or an empty string when it has none."""
    try:
        return urlsplit(normalize_url(url)).hostname or ""
    except ValueError:
        return ""


def _is_public_address(address: str) -> bool:
    """True when `address` is a routable public IP."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


async def _guard_host(url: str) -> None:
    """Refuse a URL that resolves into private or loopback address space."""
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        raise WebError(f"Unsupported URL scheme: {parts.scheme or 'none'}.")
    host = parts.hostname
    if not host:
        raise WebError(f"{url} has no host.")
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(
            host, parts.port or (443 if parts.scheme == "https" else 80),
            proto=socket.IPPROTO_TCP,
        )
    except OSError as exc:
        raise WebError(f"Cannot resolve {host}: {exc}") from exc
    addresses = {info[4][0] for info in infos}
    if not any(_is_public_address(a) for a in addresses):
        raise WebError(
            f"{host} resolves to a private or loopback address; refusing to fetch it."
        )


@dataclass(slots=True)
class Response:
    """A completed fetch."""

    url: str
    final_url: str
    status: int
    text: str
    content_type: str
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def is_html(self) -> bool:
        """True when the body looks like markup rather than data."""
        return "html" in self.content_type or "xml" in self.content_type

    @property
    def is_json(self) -> bool:
        """True when the server labelled the body as JSON."""
        return "json" in self.content_type


async def fetch_url(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    attempts: int = MAX_ATTEMPTS,
    abort: asyncio.Event | None = None,
) -> Response:
    """Fetch `url` and return its body, rotating fingerprint on a block.

    Redirects are followed manually so that each hop can be re-guarded against
    private address space. A retryable status or transport error backs off and
    tries again with a fresh user agent, up to `attempts` times; anything else
    is raised as a `WebError` carrying the status.
    """
    target = normalize_url(url)
    user_agent: str | None = None
    last_error: WebError | None = None

    for attempt in range(attempts):
        if abort is not None and abort.is_set():
            raise WebError("Aborted before the request completed.")
        if attempt:
            await asyncio.sleep(min(0.5 * 2 ** (attempt - 1), 4.0) + random.random() * 0.3)
        user_agent = next_user_agent(user_agent)
        request_headers = {
            **DEFAULT_HEADERS,
            "User-Agent": user_agent,
            **(headers or {}),
        }
        try:
            return await _fetch_once(target, request_headers, timeout)
        except WebError as exc:
            last_error = exc
            if exc.status is None or exc.status not in (
                RETRYABLE_STATUSES | BLOCKED_STATUSES
            ):
                raise
        except (httpx.TransportError, httpx.HTTPError) as exc:
            last_error = WebError(f"Request to {target} failed: {exc}")

    raise last_error or WebError(f"Request to {target} failed.")


async def _fetch_once(
    url: str, headers: dict[str, str], timeout: float
) -> Response:
    """Perform one request, following and re-guarding redirects by hand."""
    current = url
    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=httpx.Timeout(timeout),
        http2=False,
    ) as client:
        for _ in range(MAX_REDIRECTS + 1):
            await _guard_host(current)
            try:
                response = await client.get(current, headers=headers)
            except httpx.TimeoutException as exc:
                raise WebError(f"{current} timed out after {timeout:g}s.") from exc

            if response.is_redirect:
                location = response.headers.get("location", "")
                if not location:
                    raise WebError(f"{current} redirected without a target.")
                current = normalize_url(str(response.url.join(location)))
                continue

            if response.status_code >= 400:
                raise WebError(
                    f"{current} returned HTTP {response.status_code}.",
                    status=response.status_code,
                )

            body = response.content[:MAX_BODY_BYTES]
            return Response(
                url=url,
                final_url=str(response.url),
                status=response.status_code,
                text=body.decode(response.encoding or "utf-8", errors="replace"),
                content_type=response.headers.get("content-type", "").lower(),
                headers=dict(response.headers),
            )

    raise WebError(f"{url} exceeded {MAX_REDIRECTS} redirects.")


__all__ = [
    "USER_AGENTS",
    "Response",
    "WebError",
    "fetch_url",
    "next_user_agent",
    "normalize_url",
    "dedup_key",
    "host_of",
]
