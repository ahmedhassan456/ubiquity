"""The keyless search back ends and the fusion that merges them.

Each engine here is a public endpoint that answers without credentials: three
scrape a normal results page, three read a public JSON or OpenSearch API. None
of them is trusted to be up. `search_web` queries them concurrently, keeps
whatever answers inside the deadline, and fuses the surviving rankings, so an
engine that is blocked, rate limited, or simply slow costs recall rather than
the whole call.

Fusion is Reciprocal Rank Fusion: a result's score is the sum of ``1 / (k +
rank)`` over the engines that returned it. Rank is all it needs, which is the
point -- the engines disagree about scoring and none of them exposes a
comparable number, but they all produce an ordered list, and a page several
independent indexes rank highly is a better answer than one page's favourite.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs, quote_plus, urlsplit

from ._html import parse_html
from ._web import Response, WebError, dedup_key, fetch_url, host_of, normalize_url

RRF_K = 60

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the",
    "to", "use", "using", "vs", "was", "what", "when", "where", "which", "who",
    "why", "with", "you", "your",
})

REDIRECT_PARAMS = {
    "duckduckgo.com": "uddg",
    "lite.duckduckgo.com": "uddg",
    "html.duckduckgo.com": "uddg",
    "google.com": "q",
    "www.google.com": "q",
}


@dataclass(slots=True)
class SearchHit:
    """One result, before or after fusion."""

    title: str
    url: str
    snippet: str = ""
    engines: list[str] = field(default_factory=list)
    score: float = 0.0

    @property
    def host(self) -> str:
        """The hostname the result lives on."""
        return host_of(self.url)


def unwrap_redirect(href: str) -> str:
    """Return the real target behind an engine's click-tracking redirect."""
    try:
        parts = urlsplit(href if "//" in href else f"https://{href}")
    except ValueError:
        return href
    param = REDIRECT_PARAMS.get((parts.hostname or "").lower())
    if not param:
        return href
    target = parse_qs(parts.query).get(param, [""])[0]
    return target if target.lower().startswith(("http://", "https://")) else href


def soft_block(engine: str) -> WebError:
    """The error raised when a results page parses to nothing.

    An engine that answers 200 with no result rows has almost always served a
    consent wall, a challenge, or its own home page. Reporting that as an
    engine failure is more honest than reporting zero results, and it keeps a
    silently blocked engine from looking like one that simply found nothing.
    """
    return WebError(f"{engine} returned a page with no results, likely a soft block.")


def _clean(text: str) -> str:
    """Collapse whitespace in engine-supplied text."""
    return " ".join(text.split())


def _hit(title: str, url: str, snippet: str, engine: str) -> SearchHit | None:
    """Build a hit, dropping rows that lack a usable title or URL."""
    title, snippet = _clean(title), _clean(snippet)
    url = unwrap_redirect(url.strip())
    if not title or not url.lower().startswith(("http://", "https://")):
        return None
    return SearchHit(title=title, url=normalize_url(url), snippet=snippet, engines=[engine])


class Engine:
    """A single search back end."""

    name: str

    async def search(self, query: str, limit: int, timeout: float) -> list[SearchHit]:
        """Return this engine's ranked results for `query`."""
        raise NotImplementedError

    async def _get(
        self, url: str, timeout: float, headers: dict[str, str] | None = None
    ) -> Response:
        """Fetch a results page, refusing anything but a plain 200.

        An engine that has decided a request is a robot often answers 202 with
        a challenge page rather than an error status. That body parses to no
        results, so accepting it would report the engine as working and
        finding nothing.
        """
        response = await fetch_url(url, timeout=timeout, headers=headers)
        if response.status != 200:
            raise WebError(
                f"{self.name} answered HTTP {response.status}, which is a "
                "challenge page rather than results.",
                status=response.status,
            )
        return response


class DuckDuckGoEngine(Engine):
    """DuckDuckGo, via its no-JavaScript endpoints.

    The lite endpoint is tried first because its markup is small and stable.
    When it answers with something unparseable -- which is what a soft block
    looks like -- the fuller HTML endpoint is tried before giving up.
    """

    name = "duckduckgo"

    async def search(self, query: str, limit: int, timeout: float) -> list[SearchHit]:
        endpoints = (
            (f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}", "a.result-link", ".result-snippet"),
            (f"https://html.duckduckgo.com/html/?q={quote_plus(query)}", "a.result__a", ".result__snippet"),
        )
        last_error: Exception | None = None
        for url, link_selector, snippet_selector in endpoints:
            try:
                response = await self._get(url, timeout)
            except WebError as exc:
                last_error = exc
                continue
            hits = self._parse(response.text, link_selector, snippet_selector, limit)
            if hits:
                return hits
        raise last_error or soft_block(self.name)

    def _parse(
        self, html: str, link_selector: str, snippet_selector: str, limit: int
    ) -> list[SearchHit]:
        root = parse_html(html)
        links = root.select(link_selector)
        snippets = root.select(snippet_selector)
        hits: list[SearchHit] = []
        for index, link in enumerate(links[:limit]):
            snippet = snippets[index].text() if index < len(snippets) else ""
            hit = _hit(link.text(), link.get("href"), snippet, self.name)
            if hit is not None:
                hits.append(hit)
        return hits


class MojeekEngine(Engine):
    """Mojeek, which runs its own crawl rather than reselling another index.

    An independent index is worth including precisely because it disagrees:
    when it ranks the same page highly as the majors, fusion has real
    corroboration rather than two views of one crawl.
    """

    name = "mojeek"

    async def search(self, query: str, limit: int, timeout: float) -> list[SearchHit]:
        url = f"https://www.mojeek.com/search?q={quote_plus(query)}&safe=0"
        response = await self._get(url, timeout)
        root = parse_html(response.text)
        rows = root.select("ul.results-standard li, .results .result")
        hits: list[SearchHit] = []
        for row in rows[:limit]:
            link = row.select_one("a.title") or row.select_one("h2 a") or row.select_one("a")
            if link is None:
                continue
            snippet = row.select_one("p.s") or row.select_one(".description")
            hit = _hit(
                link.text(), link.get("href"), snippet.text() if snippet else "", self.name
            )
            if hit is not None:
                hits.append(hit)
        if not hits:
            raise soft_block(self.name)
        return hits


class BingEngine(Engine):
    """Bing's web results page.

    Result links arrive wrapped in a `bing.com/ck/a` tracker whose real target
    is base64 in the `u` parameter, so they are decoded here; a URL the model
    cannot fetch is not a result.
    """

    name = "bing"

    async def search(self, query: str, limit: int, timeout: float) -> list[SearchHit]:
        url = (
            f"https://www.bing.com/search?q={quote_plus(query)}"
            "&mkt=en-US&setlang=en&format=rss"
        )
        response = await self._get(url, timeout)
        hits = self._parse_rss(response.text, limit)
        if hits:
            return hits
        page = await self._get(
            f"https://www.bing.com/search?q={quote_plus(query)}&mkt=en-US&setlang=en",
            timeout,
        )
        hits = self._parse_html(page.text, limit)
        if not hits:
            raise soft_block(self.name)
        return hits

    def _parse_rss(self, xml: str, limit: int) -> list[SearchHit]:
        """Parse the RSS view, which Bing serves without anti-bot markup."""
        root = parse_html(xml, xml=True)
        hits: list[SearchHit] = []
        for item in root.select("item")[:limit]:
            title = item.select_one("title")
            link = item.select_one("link")
            description = item.select_one("description")
            href = (link.text() if link is not None else "") or (
                link.get("href") if link is not None else ""
            )
            hit = _hit(
                title.text() if title else "",
                href,
                description.text() if description else "",
                self.name,
            )
            if hit is not None:
                hits.append(hit)
        return hits

    def _parse_html(self, html: str, limit: int) -> list[SearchHit]:
        root = parse_html(html)
        hits: list[SearchHit] = []
        for row in root.select("li.b_algo")[:limit]:
            link = row.select_one("h2 a")
            if link is None:
                continue
            snippet = (
                row.select_one(".b_lineclamp2")
                or row.select_one(".b_lineclamp3")
                or row.select_one(".b_caption p")
            )
            hit = _hit(
                link.text(),
                decode_bing_url(link.get("href")),
                snippet.text() if snippet else "",
                self.name,
            )
            if hit is not None:
                hits.append(hit)
        return hits


def decode_bing_url(href: str) -> str:
    """Decode a `bing.com/ck/a` tracker back into its destination URL."""
    try:
        parts = urlsplit(href)
    except ValueError:
        return href
    if not (parts.hostname or "").endswith("bing.com") or parts.path != "/ck/a":
        return href
    encoded = parse_qs(parts.query).get("u", [""])[0]
    if len(encoded) < 4:
        return href
    body = encoded[2:].replace("-", "+").replace("_", "/")
    body += "=" * ((4 - len(body) % 4) % 4)
    try:
        decoded = base64.b64decode(body).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return href
    return decoded if decoded.lower().startswith(("http://", "https://")) else href


class WikipediaEngine(Engine):
    """Wikipedia's OpenSearch API, as an encyclopedic anchor.

    It answers only when the query names a real subject, which is what makes
    it useful in fusion: it breaks ties toward the topic a word denotes rather
    than the company that happens to own the domain.
    """

    name = "wikipedia"

    async def search(self, query: str, limit: int, timeout: float) -> list[SearchHit]:
        url = (
            "https://en.wikipedia.org/w/api.php?action=opensearch&format=json"
            f"&namespace=0&limit={min(max(limit, 1), 20)}&search={quote_plus(query)}"
        )
        response = await self._get(url, timeout, headers={"Accept": "application/json"})
        try:
            body = json.loads(response.text)
        except json.JSONDecodeError:
            return []
        if not isinstance(body, list) or len(body) < 4:
            return []
        titles, snippets, urls = body[1], body[2], body[3]
        hits: list[SearchHit] = []
        for index, title in enumerate(titles[:limit]):
            if index >= len(urls):
                break
            snippet = snippets[index] if index < len(snippets) else ""
            hit = _hit(str(title), str(urls[index]), str(snippet), self.name)
            if hit is not None:
                hits.append(hit)
        return hits


class MarginaliaEngine(Engine):
    """Marginalia, a non-commercial index of the small web.

    It surfaces long-tail pages the commercial crawlers deprioritise, which is
    where a technical answer often actually lives. Its public tier is reached
    with the literal key ``public`` and rate limits by returning 503.
    """

    name = "marginalia"

    async def search(self, query: str, limit: int, timeout: float) -> list[SearchHit]:
        url = (
            "https://api2.marginalia-search.com/search"
            f"?query={quote_plus(query)}&count={limit}&dc=3"
        )
        response = await self._get(
            url, timeout, headers={"Accept": "application/json", "API-Key": "public"}
        )
        try:
            body = json.loads(response.text)
        except json.JSONDecodeError:
            return []
        rows = body.get("results") if isinstance(body, dict) else None
        if not isinstance(rows, list):
            return []
        hits: list[SearchHit] = []
        for row in rows[:limit]:
            if not isinstance(row, dict):
                continue
            hit = _hit(
                str(row.get("title", "")),
                str(row.get("url", "")),
                str(row.get("description", "")),
                self.name,
            )
            if hit is not None:
                hits.append(hit)
        return hits


class HackerNewsEngine(Engine):
    """The Hacker News search API, for discussion and primary sources.

    Included because it answers a question the web engines cannot: whether
    practitioners have already argued about this, and what they linked to.
    """

    name = "hackernews"

    async def search(self, query: str, limit: int, timeout: float) -> list[SearchHit]:
        url = (
            "https://hn.algolia.com/api/v1/search"
            f"?query={quote_plus(query)}&tags=story&hitsPerPage={limit}"
        )
        response = await self._get(url, timeout, headers={"Accept": "application/json"})
        try:
            body = json.loads(response.text)
        except json.JSONDecodeError:
            return []
        rows = body.get("hits") if isinstance(body, dict) else None
        if not isinstance(rows, list):
            return []
        hits: list[SearchHit] = []
        for row in rows[:limit]:
            if not isinstance(row, dict):
                continue
            story = row.get("objectID", "")
            url_value = row.get("url") or f"https://news.ycombinator.com/item?id={story}"
            points, comments = row.get("points", 0), row.get("num_comments", 0)
            hit = _hit(
                str(row.get("title", "")),
                str(url_value),
                f"Hacker News discussion: {points} points, {comments} comments.",
                self.name,
            )
            if hit is not None:
                hits.append(hit)
        return hits


GENERAL_ENGINES: tuple[type[Engine], ...] = (
    DuckDuckGoEngine,
    MojeekEngine,
    BingEngine,
    WikipediaEngine,
)

WIDE_ENGINES: tuple[type[Engine], ...] = GENERAL_ENGINES + (
    MarginaliaEngine,
    HackerNewsEngine,
)

ENGINES_BY_NAME = {
    cls.name: cls for cls in (*WIDE_ENGINES,)
}


def fuse(ranked: list[list[SearchHit]], k: int = RRF_K) -> list[SearchHit]:
    """Merge per-engine rankings into one list by Reciprocal Rank Fusion.

    Results are keyed on a normalized URL so the same page found by several
    engines becomes one hit crediting all of them, and the engine that ranked
    it best supplies the title and snippet.
    """
    merged: dict[str, SearchHit] = {}
    best_rank: dict[str, int] = {}

    for hits in ranked:
        for rank, hit in enumerate(hits, start=1):
            key = dedup_key(hit.url)
            existing = merged.get(key)
            if existing is None:
                merged[key] = SearchHit(
                    title=hit.title,
                    url=hit.url,
                    snippet=hit.snippet,
                    engines=list(hit.engines),
                    score=1 / (k + rank),
                )
                best_rank[key] = rank
                continue
            existing.score += 1 / (k + rank)
            for engine in hit.engines:
                if engine not in existing.engines:
                    existing.engines.append(engine)
            if rank < best_rank[key]:
                best_rank[key] = rank
                existing.title = hit.title
                if hit.snippet:
                    existing.snippet = hit.snippet
            elif not existing.snippet and hit.snippet:
                existing.snippet = hit.snippet

    return sorted(merged.values(), key=lambda h: (-h.score, h.title.lower()))


def query_terms(query: str) -> list[str]:
    """The content words of a query, lowercased and de-duplicated."""
    seen: list[str] = []
    for token in re.findall(r"[a-z0-9]+", query.lower()):
        if len(token) < 3 or token in STOPWORDS:
            continue
        if token not in seen:
            seen.append(token)
    return seen


def alignment(terms: list[str], hit: SearchHit) -> float:
    """The share of query terms that appear in a hit's title, snippet, or URL."""
    if not terms:
        return 1.0
    haystack = f"{hit.title} {hit.snippet} {hit.url}".lower()
    return sum(1 for term in terms if term in haystack) / len(terms)


def rank_by_relevance(query: str, hits: list[SearchHit]) -> list[SearchHit]:
    """Reweight fused hits by how well they match the query's own words.

    Fusion on its own is topic-blind: every engine's first result gets the
    same credit, so a back end that answers a loose query loosely -- a forum
    search matching one word of five -- lands a stranger at the top of the
    list. Scaling each score by the share of query terms the result actually
    mentions costs an on-topic result nothing and pushes those strangers down,
    while leaving cross-engine agreement as the deciding signal among results
    that are all about the right thing.
    """
    terms = query_terms(query)
    for hit in hits:
        hit.score *= 1 + alignment(terms, hit)
    return sorted(hits, key=lambda h: (-h.score, h.title.lower()))


def apply_domain_filters(
    hits: list[SearchHit],
    allowed: list[str] | None,
    blocked: list[str] | None,
) -> list[SearchHit]:
    """Keep only hits whose host passes the allow and block lists.

    A list entry matches its own domain and every subdomain of it, so
    ``python.org`` covers ``docs.python.org``.
    """

    def matches(host: str, domain: str) -> bool:
        domain = domain.strip().lower().removeprefix("www.")
        host = host.removeprefix("www.")
        return bool(domain) and (host == domain or host.endswith(f".{domain}"))

    result = hits
    if allowed:
        result = [h for h in result if any(matches(h.host, d) for d in allowed)]
    if blocked:
        result = [h for h in result if not any(matches(h.host, d) for d in blocked)]
    return result


@dataclass(slots=True)
class SearchReport:
    """The outcome of one fused search, including what each engine did."""

    hits: list[SearchHit]
    succeeded: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


async def search_web(
    query: str,
    *,
    limit: int = 10,
    engines: tuple[type[Engine], ...] = GENERAL_ENGINES,
    timeout: float = 12.0,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
) -> SearchReport:
    """Query every engine concurrently and return the fused ranking.

    Each engine is given the same deadline and none can hold up the others: a
    failure is recorded against its name and the rest of the results stand.
    Only when every engine fails does the caller get an empty ranking with the
    reasons attached.
    """
    instances = [cls() for cls in engines]
    per_engine = max(limit, 10)

    async def run(engine: Engine) -> tuple[str, list[SearchHit] | str]:
        try:
            hits = await asyncio.wait_for(
                engine.search(query, per_engine, timeout), timeout=timeout + 2
            )
            return engine.name, hits
        except asyncio.TimeoutError:
            return engine.name, f"timed out after {timeout:g}s"
        except WebError as exc:
            return engine.name, str(exc)
        except Exception as exc:
            return engine.name, f"{type(exc).__name__}: {exc}"

    outcomes = await asyncio.gather(*(run(e) for e in instances))

    ranked: list[list[SearchHit]] = []
    report = SearchReport(hits=[])
    for name, outcome in outcomes:
        if isinstance(outcome, str):
            report.failed[name] = outcome
            continue
        report.succeeded.append(name)
        if outcome:
            ranked.append(outcome)

    fused = rank_by_relevance(query, fuse(ranked))
    report.hits = apply_domain_filters(fused, allowed_domains, blocked_domains)[:limit]
    return report


__all__ = [
    "Engine",
    "SearchHit",
    "SearchReport",
    "GENERAL_ENGINES",
    "WIDE_ENGINES",
    "ENGINES_BY_NAME",
    "DuckDuckGoEngine",
    "MojeekEngine",
    "BingEngine",
    "WikipediaEngine",
    "MarginaliaEngine",
    "HackerNewsEngine",
    "search_web",
    "fuse",
    "rank_by_relevance",
    "query_terms",
    "alignment",
    "apply_domain_filters",
    "decode_bing_url",
    "unwrap_redirect",
]
