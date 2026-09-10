"""The WebFetch tool.

Fetches a URL and hands the model Markdown rather than raw HTML. The
conversion is the point: a page's markup is mostly navigation, scripts, and
layout, and passing it through unchanged spends the context window on
everything except the text. Trimming to the article body and rendering
headings, lists, code blocks, and links keeps the structure the model needs to
quote and follow links, at a fraction of the size.

Long pages are paged rather than silently cut. The response says how much was
withheld and how to ask for the rest, so a truncated read is a visible state
the model can act on instead of a quiet loss of the ending.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from ..tool import Tool, ToolContext, ValidationError
from ..types import PermissionResult, PermissionResultAsk, ToolOutput
from ._html import html_to_markdown, page_description, page_title, parse_html
from ._web import WebError, fetch_url, host_of, normalize_url

MAX_CHARS = 60_000


class WebFetchInput(BaseModel):
    url: str = Field(description="The URL to fetch.")
    format: str = Field(
        default="markdown",
        description=(
            "`markdown` for the page's main content as Markdown, `text` for "
            "plain text with no markup, `raw` for the untouched response body."
        ),
    )
    full_page: bool = Field(
        default=False,
        description=(
            "Include navigation, sidebars, and footers instead of trimming to "
            "the main content. Use when the part you need is page chrome, such "
            "as a documentation index or a link list."
        ),
    )
    offset: int = Field(
        default=0,
        description="Character offset to resume from, for reading a long page in pages.",
    )
    max_chars: int = Field(
        default=MAX_CHARS, description=f"How much to return, at most {MAX_CHARS}."
    )


class WebFetchTool(Tool[WebFetchInput]):
    """Fetch a URL and return its content as Markdown."""

    name = "WebFetch"
    description = (
        "Fetch a web page and return its content, converted to Markdown. No "
        "API key is involved. By default the page is trimmed to its main "
        "content and the chrome around it is dropped; pass `full_page` when "
        "you need the navigation or link lists too, `format='text'` for plain "
        "text, or `format='raw'` for the untouched body. JSON responses are "
        "returned formatted. Long pages are truncated with a note saying what "
        "`offset` reads the next section. Redirects are followed and the final "
        "URL is reported, so a link that moved is visible rather than silent. "
        "Fetching a URL that resolves to a private or loopback address is "
        "refused."
    )
    input_model = WebFetchInput
    search_hint = "read fetch download web page url article documentation"
    max_result_chars = 100_000

    def is_read_only(self, args: WebFetchInput) -> bool:
        return True

    def is_concurrency_safe(self, args: WebFetchInput) -> bool:
        return True

    def describe_call(self, args: WebFetchInput) -> str:
        return f"WebFetch {args.url}"

    def permission_rule_content(self, args: WebFetchInput) -> list[str]:
        """Present the host, so a rule can name a domain.

        The host alone is the candidate because that is what the rule engine
        matches well: ``WebFetch(docs.python.org)`` is an exact rule,
        ``WebFetch(*.python.org)`` a wildcard that covers the subdomains, and
        ``WebFetch(*)`` the whole web. A rule bearing a path would have to
        match the URL instead, which would make every one of those forms stop
        working.
        """
        host = host_of(args.url)
        return [host] if host else []

    async def validate_input(
        self, args: WebFetchInput, ctx: ToolContext
    ) -> ValidationError | None:
        if not args.url.strip():
            return ValidationError(message="`url` is empty.")
        if not host_of(args.url):
            return ValidationError(message=f"{args.url} is not a usable URL.")
        if args.format not in {"markdown", "text", "raw"}:
            return ValidationError(
                message=f"`format` must be markdown, text, or raw; got {args.format!r}."
            )
        if args.offset < 0:
            return ValidationError(message="`offset` cannot be negative.")
        return None

    async def check_permissions(
        self, args: WebFetchInput, ctx: ToolContext
    ) -> PermissionResult:
        """Ask before reaching a host, so approval is per-domain and explicit."""
        return PermissionResultAsk(message=f"Fetch content from {host_of(args.url)}")

    async def call(self, args: WebFetchInput, ctx: ToolContext) -> ToolOutput:
        url = normalize_url(args.url)
        try:
            response = await fetch_url(url, abort=ctx.abort)
        except WebError as exc:
            return ToolOutput(content=str(exc), metadata={"url": url}, is_error=True)

        body, title = self._render(response, args)

        metadata = {
            "url": url,
            "final_url": response.final_url,
            "status": response.status,
            "content_type": response.content_type,
            "title": title,
            "total_chars": len(body),
        }

        limit = min(args.max_chars, MAX_CHARS)
        section = body[args.offset : args.offset + limit]
        remaining = max(0, len(body) - (args.offset + len(section)))
        metadata["truncated"] = remaining > 0

        header = [f"# {title}"] if title else []
        if response.final_url != url:
            header.append(f"(redirected to {response.final_url})")
        header.append("")

        footer = ""
        if remaining:
            footer = (
                f"\n\n[{remaining} more characters. Call WebFetch again with "
                f"offset={args.offset + len(section)} to continue.]"
            )

        if not section.strip():
            return ToolOutput(
                content=(
                    f"{url} returned no readable content"
                    + (" at that offset." if args.offset else ".")
                ),
                metadata=metadata,
            )

        return ToolOutput(content="\n".join(header) + section + footer, metadata=metadata)

    def _render(self, response, args: WebFetchInput) -> tuple[str, str]:
        """Turn the response body into the requested shape, plus a title."""
        if args.format == "raw":
            return response.text, ""

        if response.is_json:
            try:
                return json.dumps(json.loads(response.text), indent=2), ""
            except json.JSONDecodeError:
                return response.text, ""

        if not response.is_html:
            return response.text, ""

        root = parse_html(response.text)
        title = page_title(root)

        if args.format == "text":
            from ._html import main_content

            node = main_content(root) if not args.full_page else root
            return node.text(), title

        markdown = html_to_markdown(
            response.text, base_url=response.final_url, article=not args.full_page
        )
        description = page_description(root)
        if description and description not in markdown[:1000]:
            markdown = f"> {description}\n\n{markdown}"
        return markdown, title


__all__ = ["WebFetchTool", "WebFetchInput"]
