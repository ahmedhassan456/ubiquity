"""The WebSearch tool.

Search reaches the open web with no account and no key: the query fans out to
several public engines at once and the rankings they return are fused into one
list. The tool is deliberately not a wrapper around a single provider, because
a single provider is a single point of failure -- one blocked scrape and the
agent has no web at all. With several, a block costs one opinion.

The output tells the model which engines corroborated each result. That is not
decoration: a page four independent indexes agree on is worth opening before a
page only one of them found.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..tool import Tool, ToolContext, ValidationError
from ..types import PermissionResult, PermissionResultAllow, ToolOutput
from ._engines import GENERAL_ENGINES, WIDE_ENGINES, SearchHit, search_web

MAX_RESULTS = 20
DEFAULT_RESULTS = 10
SNIPPET_CHARS = 300


class WebSearchInput(BaseModel):
    query: str = Field(description="The search query.")
    max_results: int = Field(
        default=DEFAULT_RESULTS,
        description=f"How many results to return, at most {MAX_RESULTS}.",
    )
    allowed_domains: list[str] | None = Field(
        default=None,
        description=(
            "Keep only results on these domains. Subdomains count, so "
            "`python.org` also matches `docs.python.org`."
        ),
    )
    blocked_domains: list[str] | None = Field(
        default=None, description="Drop results on these domains and their subdomains."
    )
    wide: bool = Field(
        default=False,
        description=(
            "Also query the long-tail indexes. Slower, but better for niche, "
            "old, or heavily discussed topics that the major engines bury."
        ),
    )


def format_hits(hits: list[SearchHit]) -> str:
    """Render fused results as a numbered list the model can act on."""
    lines: list[str] = []
    for index, hit in enumerate(hits, start=1):
        lines.append(f"{index}. {hit.title}")
        lines.append(f"   {hit.url}")
        if hit.snippet:
            snippet = hit.snippet
            if len(snippet) > SNIPPET_CHARS:
                snippet = snippet[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "..."
            lines.append(f"   {snippet}")
        lines.append(f"   [found by: {', '.join(sorted(hit.engines))}]")
        lines.append("")
    return "\n".join(lines).rstrip()


class WebSearchTool(Tool[WebSearchInput]):
    """Search the web across several keyless engines at once."""

    name = "WebSearch"
    description = (
        "Search the web. The query is sent to several independent public "
        "search engines at once and their rankings are merged, so results "
        "carry the list of engines that found them; agreement across engines "
        "is a signal worth trusting. No API key or account is involved. "
        f"Returns up to {MAX_RESULTS} results as title, URL, and snippet -- "
        "snippets are search-engine summaries, not page content, so use "
        "WebFetch to read a result before relying on what it says. Scope the "
        "search with `allowed_domains` or `blocked_domains`, and set `wide` "
        "for niche topics the major engines bury. An engine that is blocked "
        "or slow is simply left out of the merge; the results still stand."
    )
    input_model = WebSearchInput
    search_hint = "search the web internet google lookup online current news"
    max_result_chars = 60_000

    def is_read_only(self, args: WebSearchInput) -> bool:
        return True

    def is_concurrency_safe(self, args: WebSearchInput) -> bool:
        return True

    def describe_call(self, args: WebSearchInput) -> str:
        return f'WebSearch "{args.query}"'

    async def validate_input(
        self, args: WebSearchInput, ctx: ToolContext
    ) -> ValidationError | None:
        if not args.query.strip():
            return ValidationError(message="`query` is empty.")
        if args.max_results < 1:
            return ValidationError(message="`max_results` must be at least 1.")
        if args.allowed_domains and args.blocked_domains:
            overlap = {d.lower() for d in args.allowed_domains} & {
                d.lower() for d in args.blocked_domains
            }
            if overlap:
                return ValidationError(
                    message=(
                        "These domains are in both `allowed_domains` and "
                        f"`blocked_domains`: {', '.join(sorted(overlap))}."
                    )
                )
        return None

    async def check_permissions(
        self, args: WebSearchInput, ctx: ToolContext
    ) -> PermissionResult:
        """A search reads public indexes and touches nothing local."""
        return PermissionResultAllow(reason="read-only web search")

    async def call(self, args: WebSearchInput, ctx: ToolContext) -> ToolOutput:
        limit = min(args.max_results, MAX_RESULTS)
        report = await search_web(
            args.query.strip(),
            limit=limit,
            engines=WIDE_ENGINES if args.wide else GENERAL_ENGINES,
            allowed_domains=args.allowed_domains,
            blocked_domains=args.blocked_domains,
        )

        metadata = {
            "query": args.query,
            "count": len(report.hits),
            "engines_succeeded": report.succeeded,
            "engines_failed": report.failed,
            "results": [
                {
                    "title": h.title,
                    "url": h.url,
                    "snippet": h.snippet,
                    "engines": h.engines,
                    "score": h.score,
                }
                for h in report.hits
            ],
        }

        if not report.hits:
            if not report.succeeded:
                reasons = "; ".join(f"{k}: {v}" for k, v in report.failed.items())
                return ToolOutput(
                    content=f"Every search engine failed. {reasons}",
                    metadata=metadata,
                    is_error=True,
                )
            return ToolOutput(
                content=f'No results for "{args.query}".', metadata=metadata
            )

        header = (
            f'{len(report.hits)} result(s) for "{args.query}" '
            f"via {', '.join(report.succeeded)}."
        )
        if report.failed:
            header += f" Unavailable: {', '.join(sorted(report.failed))}."

        return ToolOutput(
            content=f"{header}\n\n{format_hits(report.hits)}", metadata=metadata
        )


__all__ = ["WebSearchTool", "WebSearchInput", "format_hits"]
