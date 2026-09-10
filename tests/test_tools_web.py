"""Tests for the web tools.

Nothing here touches the network. Engines are exercised against captured
result markup and `fetch_url` is replaced where a tool would otherwise make a
request, so the suite stays offline and deterministic while still covering the
parts that break in practice: tracker-wrapped URLs, fusion across disagreeing
engines, and a failing engine that must not take the call down with it.
"""

from __future__ import annotations

import asyncio

import pytest

from ubiquity.tools import WebFetchTool, WebSearchTool, builtin_tools
from ubiquity.tools._engines import (
    BingEngine,
    DuckDuckGoEngine,
    MojeekEngine,
    SearchHit,
    WikipediaEngine,
    apply_domain_filters,
    decode_bing_url,
    alignment,
    fuse,
    query_terms,
    rank_by_relevance,
    search_web,
    unwrap_redirect,
)
from ubiquity.tools._html import html_to_markdown, page_title, parse_html
from ubiquity.tools._web import (
    Response,
    WebError,
    dedup_key,
    host_of,
    next_user_agent,
    normalize_url,
)
from ubiquity.tools.webfetch import WebFetchInput
from ubiquity.tools.websearch import WebSearchInput

DDG_HTML = """
<html><body>
  <a class="result-link" href="https://docs.python.org/3/library/asyncio.html">asyncio docs</a>
  <td class="result-snippet">Asynchronous I/O for Python.</td>
  <a class="result-link" href="https://realpython.com/async-io-python/">Async IO in Python</a>
  <td class="result-snippet">A complete walkthrough.</td>
</body></html>
"""

MOJEEK_HTML = """
<html><body><ul class="results-standard">
  <li><h2><a class="title" href="https://realpython.com/async-io-python/">Async IO</a></h2>
      <p class="s">Guide to async.</p></li>
  <li><h2><a class="title" href="https://peps.python.org/pep-3156/">PEP 3156</a></h2>
      <p class="s">Asynchronous IO support.</p></li>
</ul></body></html>
"""

ARTICLE_HTML = """
<html><head><title>Sample Page</title>
<meta name="description" content="A page about widgets.">
<script>var tracking = 1;</script></head>
<body>
  <nav><a href="/home">Home</a><a href="/about">About</a></nav>
  <article>
    <h1>Widgets</h1>
    <p>Widgets are <strong>useful</strong> and come in <em>many</em> shapes.</p>
    <ul><li>First widget</li><li>Second widget</li></ul>
    <pre><code class="language-python">print("hi")</code></pre>
    <p>See the <a href="/specs/widget">spec</a> for details.</p>
    <table><tr><th>Name</th><th>Size</th></tr><tr><td>Bolt</td><td>M4</td></tr></table>
  </article>
  <footer>Copyright</footer>
</body></html>
"""


def test_normalize_and_dedup_keys() -> None:
    assert normalize_url("example.com/a") == "https://example.com/a"
    assert normalize_url("HTTPS://Example.COM/a#frag") == "https://example.com/a"
    assert dedup_key("https://www.example.com/a/") == dedup_key("http://example.com/a")
    assert host_of("https://docs.python.org/3/") == "docs.python.org"


def test_user_agent_rotation_always_changes() -> None:
    from ubiquity.tools._web import USER_AGENTS

    for agent in USER_AGENTS:
        assert next_user_agent(agent) != agent
        assert next_user_agent(agent) in USER_AGENTS
    assert next_user_agent("not-in-the-pool") in USER_AGENTS
    assert next_user_agent() in USER_AGENTS


def test_unwrap_and_decode_tracker_urls() -> None:
    wrapped = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage"
    assert unwrap_redirect(wrapped) == "https://example.com/page"
    assert unwrap_redirect("https://example.com/plain") == "https://example.com/plain"

    import base64

    encoded = base64.b64encode(b"https://example.com/target").decode().rstrip("=")
    tracker = f"https://www.bing.com/ck/a?u=a1{encoded}"
    assert decode_bing_url(tracker) == "https://example.com/target"


def test_duckduckgo_parses_results() -> None:
    hits = DuckDuckGoEngine()._parse(DDG_HTML, "a.result-link", ".result-snippet", 10)
    assert [h.url for h in hits] == [
        "https://docs.python.org/3/library/asyncio.html",
        "https://realpython.com/async-io-python/",
    ]
    assert hits[0].snippet == "Asynchronous I/O for Python."
    assert hits[0].engines == ["duckduckgo"]


def test_mojeek_parses_results() -> None:
    async def run() -> list[SearchHit]:
        engine = MojeekEngine()

        async def fake_get(url, timeout, headers=None):
            return Response(
                url=url,
                final_url=url,
                status=200,
                text=MOJEEK_HTML,
                content_type="text/html",
            )

        engine._get = fake_get
        return await engine.search("async io", 10, 5.0)

    hits = asyncio.run(run())
    assert [h.title for h in hits] == ["Async IO", "PEP 3156"]


async def test_engine_refuses_a_challenge_status(monkeypatch) -> None:
    async def fake_fetch(url, **kwargs):
        return Response(
            url=url, final_url=url, status=202, text="<html>robot?</html>",
            content_type="text/html",
        )

    monkeypatch.setattr("ubiquity.tools._engines.fetch_url", fake_fetch)

    with pytest.raises(WebError) as excinfo:
        await MojeekEngine().search("anything", 5, 5.0)
    assert "202" in str(excinfo.value)


async def test_soft_block_is_reported_as_a_failure(monkeypatch) -> None:
    async def fake_fetch(url, **kwargs):
        return Response(
            url=url, final_url=url, status=200,
            text="<html><body><h1>Mojeek</h1></body></html>",
            content_type="text/html",
        )

    monkeypatch.setattr("ubiquity.tools._engines.fetch_url", fake_fetch)

    report = await search_web("anything", engines=(MojeekEngine,))
    assert report.succeeded == []
    assert "soft block" in report.failed["mojeek"]


def test_bing_reads_the_rss_view() -> None:
    xml = """<rss><channel>
      <item><title>Async IO</title><link>https://example.com/a</link>
        <description>About async.</description></item>
    </channel></rss>"""
    hits = BingEngine()._parse_rss(xml, 10)
    assert hits[0].url == "https://example.com/a"
    assert hits[0].engines == ["bing"]


def test_wikipedia_parses_opensearch() -> None:
    async def run() -> list[SearchHit]:
        engine = WikipediaEngine()

        async def fake_get(url, timeout, headers=None):
            body = '["asyncio",["Asyncio"],["Python library"],["https://en.wikipedia.org/wiki/Asyncio"]]'
            return Response(
                url=url,
                final_url=url,
                status=200,
                text=body,
                content_type="application/json",
            )

        engine._get = fake_get
        return await engine.search("asyncio", 5, 5.0)

    hits = asyncio.run(run())
    assert hits[0].url == "https://en.wikipedia.org/wiki/Asyncio"


def test_fusion_rewards_agreement_across_engines() -> None:
    shared = "https://example.com/shared"
    a = [
        SearchHit(title="Only A", url="https://example.com/a", engines=["alpha"]),
        SearchHit(title="Shared", url=shared, engines=["alpha"]),
    ]
    b = [
        SearchHit(title="Shared", url=shared + "/", engines=["beta"]),
        SearchHit(title="Only B", url="https://example.com/b", engines=["beta"]),
    ]

    fused = fuse([a, b])
    assert fused[0].url.startswith(shared)
    assert sorted(fused[0].engines) == ["alpha", "beta"]
    assert len(fused) == 3


def test_relevance_reweighting_demotes_off_topic_agreement() -> None:
    """An engine answering loosely must not outrank one answering the query."""
    on_topic = SearchHit(
        title="Reciprocal rank fusion explained",
        url="https://example.com/rrf",
        snippet="How RRF merges ranked lists.",
        score=0.016,
    )
    off_topic = SearchHit(
        title="Launch HN: a company brain",
        url="https://news.ycombinator.com/item?id=1",
        snippet="Discussion: 79 points.",
        score=0.017,
    )

    ranked = rank_by_relevance("reciprocal rank fusion explained", [off_topic, on_topic])
    assert ranked[0] is on_topic


def test_query_terms_drops_stopwords_and_noise() -> None:
    assert query_terms("What is the Reciprocal Rank Fusion?") == [
        "reciprocal",
        "rank",
        "fusion",
    ]


def test_alignment_is_the_share_of_query_terms_present() -> None:
    hit = SearchHit(title="Rank fusion", url="https://example.com/x", snippet="")
    assert alignment(["rank", "fusion"], hit) == 1.0
    assert alignment(["rank", "fusion", "elephant"], hit) == pytest.approx(2 / 3)
    assert alignment([], hit) == 1.0


def test_domain_filters_cover_subdomains() -> None:
    hits = [
        SearchHit(title="docs", url="https://docs.python.org/3/"),
        SearchHit(title="blog", url="https://medium.com/post"),
    ]
    assert len(apply_domain_filters(hits, ["python.org"], None)) == 1
    assert len(apply_domain_filters(hits, None, ["medium.com"])) == 1


async def test_search_survives_a_failing_engine(monkeypatch) -> None:
    class Good(DuckDuckGoEngine):
        name = "good"

        async def search(self, query, limit, timeout):
            return [SearchHit(title="Result", url="https://example.com/x", engines=["good"])]

    class Bad(DuckDuckGoEngine):
        name = "bad"

        async def search(self, query, limit, timeout):
            raise WebError("https://bad.example returned HTTP 403.", status=403)

    report = await search_web("anything", engines=(Good, Bad))
    assert report.succeeded == ["good"]
    assert "bad" in report.failed
    assert report.hits[0].url == "https://example.com/x"


async def test_search_reports_when_every_engine_fails(make_ctx, tmp_path) -> None:
    class Bad(DuckDuckGoEngine):
        name = "bad"

        async def search(self, query, limit, timeout):
            raise WebError("blocked")

    import ubiquity.tools.websearch as websearch

    async def fake_search(query, **kwargs):
        return await search_web(query, engines=(Bad,), **{
            k: v for k, v in kwargs.items() if k != "engines"
        })

    tool = WebSearchTool()
    original = websearch.search_web
    websearch.search_web = fake_search
    try:
        out = await tool.call(WebSearchInput(query="x"), make_ctx(cwd=tmp_path))
    finally:
        websearch.search_web = original

    assert out.is_error
    assert "Every search engine failed" in out.content


async def test_websearch_rejects_contradictory_domain_lists(make_ctx, tmp_path) -> None:
    error = await WebSearchTool().validate_input(
        WebSearchInput(
            query="x", allowed_domains=["example.com"], blocked_domains=["Example.com"]
        ),
        make_ctx(cwd=tmp_path),
    )
    assert error is not None
    assert "example.com" in error.message


def test_html_to_markdown_keeps_structure_and_drops_chrome() -> None:
    markdown = html_to_markdown(ARTICLE_HTML, base_url="https://site.test/docs/page")

    assert "# Widgets" in markdown
    assert "**useful**" in markdown
    assert "- First widget" in markdown
    assert "```" in markdown and 'print("hi")' in markdown
    assert "[spec](https://site.test/specs/widget)" in markdown
    assert "| Name | Size |" in markdown
    assert "tracking" not in markdown
    assert "Copyright" not in markdown
    assert "About" not in markdown


def test_code_blocks_keep_their_indentation() -> None:
    """Collapsing whitespace inside a code block would destroy the code."""
    html = (
        "<html><body><article><pre><code>def f():\n"
        "    if x:\n        return 1\n</code></pre></article></body></html>"
    )
    markdown = html_to_markdown(html)
    assert "    if x:" in markdown
    assert "        return 1" in markdown


def test_code_fences_survive_backticks_in_the_code() -> None:
    html = "<html><body><article><pre><code>a = ```b```</code></pre></article></body></html>"
    markdown = html_to_markdown(html)
    assert "````" in markdown


def test_page_title_prefers_open_graph() -> None:
    root = parse_html(
        '<html><head><meta property="og:title" content="OG Title">'
        "<title>Fallback</title></head><body></body></html>"
    )
    assert page_title(root) == "OG Title"
    assert page_title(parse_html("<html><head><title>Only</title></head></html>")) == "Only"


def test_parser_recovers_from_unclosed_tags() -> None:
    root = parse_html("<ul><li>one<li>two<li>three</ul>")
    assert [li.text() for li in root.select("li")] == ["one", "two", "three"]


async def test_webfetch_returns_markdown(make_ctx, tmp_path, monkeypatch) -> None:
    async def fake_fetch(url, **kwargs):
        return Response(
            url=url,
            final_url=url,
            status=200,
            text=ARTICLE_HTML,
            content_type="text/html; charset=utf-8",
        )

    monkeypatch.setattr("ubiquity.tools.webfetch.fetch_url", fake_fetch)

    out = await WebFetchTool().call(
        WebFetchInput(url="https://site.test/page"), make_ctx(cwd=tmp_path)
    )
    assert "# Sample Page" in out.content
    assert "Widgets are **useful**" in out.content
    assert out.metadata["status"] == 200
    assert out.metadata["truncated"] is False


async def test_webfetch_pages_long_documents(make_ctx, tmp_path, monkeypatch) -> None:
    body = "<html><body><article><p>" + ("word " * 5000) + "</p></article></body></html>"

    async def fake_fetch(url, **kwargs):
        return Response(
            url=url, final_url=url, status=200, text=body, content_type="text/html"
        )

    monkeypatch.setattr("ubiquity.tools.webfetch.fetch_url", fake_fetch)

    out = await WebFetchTool().call(
        WebFetchInput(url="https://site.test/long", max_chars=500),
        make_ctx(cwd=tmp_path),
    )
    assert out.metadata["truncated"] is True
    assert "offset=" in out.content


async def test_webfetch_formats_json(make_ctx, tmp_path, monkeypatch) -> None:
    async def fake_fetch(url, **kwargs):
        return Response(
            url=url,
            final_url=url,
            status=200,
            text='{"b":2,"a":[1,2]}',
            content_type="application/json",
        )

    monkeypatch.setattr("ubiquity.tools.webfetch.fetch_url", fake_fetch)

    out = await WebFetchTool().call(
        WebFetchInput(url="https://api.test/thing"), make_ctx(cwd=tmp_path)
    )
    assert '"b": 2' in out.content


async def test_webfetch_reports_a_failed_request(make_ctx, tmp_path, monkeypatch) -> None:
    async def fake_fetch(url, **kwargs):
        raise WebError("https://site.test returned HTTP 404.", status=404)

    monkeypatch.setattr("ubiquity.tools.webfetch.fetch_url", fake_fetch)

    out = await WebFetchTool().call(
        WebFetchInput(url="https://site.test/gone"), make_ctx(cwd=tmp_path)
    )
    assert out.is_error
    assert "404" in out.content


async def test_webfetch_refuses_private_addresses(make_ctx, tmp_path) -> None:
    out = await WebFetchTool().call(
        WebFetchInput(url="http://127.0.0.1:8080/admin"), make_ctx(cwd=tmp_path)
    )
    assert out.is_error
    assert "private or loopback" in out.content


async def test_webfetch_scopes_permission_to_the_host(make_ctx, tmp_path) -> None:
    tool = WebFetchTool()
    assert tool.permission_rule_content(
        WebFetchInput(url="https://docs.python.org/3/library/asyncio.html")
    ) == ["docs.python.org"]

    result = await tool.check_permissions(
        WebFetchInput(url="https://docs.python.org/3/"), make_ctx(cwd=tmp_path)
    )
    assert result.behavior == "ask"


async def test_a_domain_rule_authorizes_the_host_it_names(make_ctx, tmp_path) -> None:
    from ubiquity.permissions import check_permissions

    tool = WebFetchTool()
    ctx = make_ctx(cwd=tmp_path, allow={"WebFetch(*.python.org)"})

    allowed = await check_permissions(
        tool, WebFetchInput(url="https://docs.python.org/3/"), ctx
    )
    assert allowed.behavior == "allow"

    elsewhere = await check_permissions(
        tool, WebFetchInput(url="https://example.com/"), ctx
    )
    assert elsewhere.behavior == "ask"


def test_web_tools_are_part_of_the_builtin_suite() -> None:
    names = {t.name for t in builtin_tools()}
    assert {"WebSearch", "WebFetch"} <= names
