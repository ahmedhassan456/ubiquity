"""A dependency-free HTML layer for the web tools.

Two jobs live here. The first is a tolerant parser that turns a page into a
small tree with just enough selector support for the search engines to pick
their result rows out of a results page. The second is a renderer that turns
that tree into Markdown, which is the shape `WebFetch` hands to the model.

Both are written against the standard library on purpose. The web tools are
part of the built-in suite, so anything they need is a dependency of every
install; a parser that costs nothing is worth more here than one that handles
every malformed page perfectly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser

VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}

SKIPPED_TAGS = {"script", "style", "noscript", "template", "svg", "canvas"}

BOILERPLATE_TAGS = {"nav", "footer", "aside", "form"}

CONTAINER_HINTS = ("content", "article", "post", "main", "entry", "markdown", "prose")

BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "div", "dl", "dt", "dd",
    "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3",
    "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre",
    "section", "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
}

IMPLICIT_CLOSE = {
    "li": {"li"},
    "p": {"p"},
    "dt": {"dt", "dd"},
    "dd": {"dt", "dd"},
    "tr": {"tr"},
    "td": {"td", "th"},
    "th": {"td", "th"},
    "option": {"option"},
    "thead": {"thead", "tbody"},
    "tbody": {"tbody", "thead"},
}

TEXT = "#text"

_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_RE = re.compile(r"\n{3,}")


@dataclass
class Node:
    """One element or text run in a parsed document."""

    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[Node] = field(default_factory=list)
    parent: Node | None = None
    data: str = ""

    @property
    def classes(self) -> set[str]:
        """The element's class names, as a set."""
        return set(self.attrs.get("class", "").split())

    def get(self, name: str, default: str = "") -> str:
        """Return an attribute value, or `default` when it is absent."""
        return self.attrs.get(name, default)

    def descendants(self) -> list[Node]:
        """Every node below this one, in document order."""
        out: list[Node] = []
        stack = list(reversed(self.children))
        while stack:
            node = stack.pop()
            out.append(node)
            stack.extend(reversed(node.children))
        return out

    def raw_text(self) -> str:
        """The subtree's text with whitespace intact.

        `text` collapses runs of whitespace, which is right for prose and
        wrong for a code block, where the indentation is the meaning.
        """
        if self.tag in SKIPPED_TAGS:
            return ""
        if self.tag == TEXT:
            return self.data
        return "".join(child.raw_text() for child in self.children)

    def text(self, *, strip: bool = True) -> str:
        """The concatenated text of this subtree, with scripts left out."""
        parts: list[str] = []
        self._collect_text(parts)
        joined = "".join(parts)
        joined = _WS_RE.sub(" ", joined)
        return joined.strip() if strip else joined

    def _collect_text(self, parts: list[str]) -> None:
        if self.tag in SKIPPED_TAGS:
            return
        if self.tag == TEXT:
            parts.append(self.data)
            return
        if self.tag in BLOCK_TAGS or self.tag == "br":
            parts.append(" ")
        for child in self.children:
            child._collect_text(parts)
        if self.tag in BLOCK_TAGS:
            parts.append(" ")

    def select(self, selector: str) -> list[Node]:
        """Return every descendant matching `selector`, in document order.

        The supported grammar is the part of CSS the engines actually use:
        comma-separated alternatives, descendant combinators, and simple
        selectors built from a tag name, `.class`, and `#id`.
        """
        seen: list[Node] = []
        for group in (g.strip() for g in selector.split(",")):
            if not group:
                continue
            for node in self._select_group(group):
                if not any(node is s for s in seen):
                    seen.append(node)
        order = {id(n): i for i, n in enumerate(self.descendants())}
        seen.sort(key=lambda n: order.get(id(n), 0))
        return seen

    def select_one(self, selector: str) -> Node | None:
        """Return the first descendant matching `selector`, or None."""
        matches = self.select(selector)
        return matches[0] if matches else None

    def _select_group(self, group: str) -> list[Node]:
        steps = group.split()
        current = [self]
        for step in steps:
            nxt: list[Node] = []
            for node in current:
                nxt.extend(d for d in node.descendants() if _matches(d, step))
            current = nxt
        return current


def _matches(node: Node, simple: str) -> bool:
    """True when `node` satisfies one simple selector such as ``li.b_algo``."""
    if node.tag == TEXT:
        return False
    match = re.match(r"^([a-zA-Z0-9_-]+)?((?:[.#][^.#]+)*)$", simple)
    if match is None:
        return False
    tag, rest = match.group(1), match.group(2) or ""
    if tag and tag != node.tag:
        return False
    for part in re.findall(r"[.#][^.#]+", rest):
        value = part[1:]
        if part[0] == "." and value not in node.classes:
            return False
        if part[0] == "#" and value != node.get("id"):
            return False
    return True


class _TreeBuilder(HTMLParser):
    """Builds a `Node` tree, closing tags the page forgot to close.

    `void_tags` is a parameter rather than a constant because the same builder
    parses XML feeds, where a tag like `<link>` carries text and closes
    normally instead of standing alone as it does in HTML.
    """

    def __init__(self, void_tags: set[str] = VOID_TAGS) -> None:
        super().__init__(convert_charrefs=True)
        self.void_tags = void_tags
        self.root = Node(tag="#document")
        self._stack = [self.root]

    @property
    def _current(self) -> Node:
        return self._stack[-1]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for open_tag in IMPLICIT_CLOSE.get(tag, ()):
            if any(n.tag == open_tag for n in self._stack[1:]):
                self._close_to(open_tag)
                break
        node = Node(
            tag=tag,
            attrs={k.lower(): (v or "") for k, v in attrs},
            parent=self._current,
        )
        self._current.children.append(node)
        if tag not in self.void_tags:
            self._stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(
            tag=tag,
            attrs={k.lower(): (v or "") for k, v in attrs},
            parent=self._current,
        )
        self._current.children.append(node)

    def handle_endtag(self, tag: str) -> None:
        if tag in self.void_tags:
            return
        self._close_to(tag)

    def _close_to(self, tag: str) -> None:
        """Pop the stack through the nearest open `tag`, if there is one."""
        for depth in range(len(self._stack) - 1, 0, -1):
            if self._stack[depth].tag == tag:
                del self._stack[depth:]
                return

    def handle_data(self, data: str) -> None:
        if not data:
            return
        self._current.children.append(
            Node(tag=TEXT, data=data, parent=self._current)
        )


def parse_html(html: str, *, xml: bool = False) -> Node:
    """Parse `html` into a tree, tolerating unclosed and stray tags.

    Set `xml` for RSS and Atom, where no tag is void: an HTML parser would
    treat `<link>https://...</link>` as an empty element and drop the URL.
    """
    builder = _TreeBuilder(void_tags=set() if xml else VOID_TAGS)
    try:
        builder.feed(html)
        builder.close()
    except Exception:
        pass
    return builder.root


def page_title(root: Node) -> str:
    """The document title, preferring ``og:title`` over ``<title>``."""
    for meta in root.select("meta"):
        prop = meta.get("property").lower() or meta.get("name").lower()
        if prop in {"og:title", "twitter:title"} and meta.get("content").strip():
            return unescape(meta.get("content").strip())
    title = root.select_one("title")
    return title.text() if title is not None else ""


def page_description(root: Node) -> str:
    """The page's meta description, when it declares one."""
    for meta in root.select("meta"):
        prop = meta.get("property").lower() or meta.get("name").lower()
        if prop in {"description", "og:description"} and meta.get("content").strip():
            return unescape(meta.get("content").strip())
    return ""


def _text_score(node: Node) -> int:
    """A crude density score: text length, penalised by link-heavy markup."""
    text_len = len(node.text())
    link_len = sum(len(a.text()) for a in node.select("a"))
    return text_len - link_len


def main_content(root: Node) -> Node:
    """Pick the subtree that most likely holds the article body.

    Semantic containers win outright when they carry real text. Otherwise the
    densest `div` or `section` is chosen, which keeps navigation chrome and
    link farms out of the Markdown without needing a full readability port.
    """
    for selector in ("article", "main", "[role=main]"):
        if selector.startswith("["):
            candidates = [
                n for n in root.descendants() if n.get("role") == "main"
            ]
        else:
            candidates = root.select(selector)
        best = max(candidates, key=_text_score, default=None)
        if best is not None and _text_score(best) > 200:
            return best

    hinted = [
        n
        for n in root.select("div, section")
        if any(
            hint in (n.get("id") + " " + n.get("class")).lower()
            for hint in CONTAINER_HINTS
        )
    ]
    best = max(hinted, key=_text_score, default=None)
    if best is not None and _text_score(best) > 400:
        return best

    body = root.select_one("body")
    return body if body is not None else root


def _fence_for(code: str) -> str:
    """Return a fence long enough to survive backticks inside `code`."""
    longest = max((len(run) for run in re.findall(r"`+", code)), default=0)
    return "`" * max(3, longest + 1)


class _MarkdownWriter:
    """Renders a node tree to Markdown."""

    def __init__(self, base_url: str = "") -> None:
        self.base_url = base_url
        self.parts: list[str] = []

    def render(self, node: Node) -> str:
        self._block(node)
        text = "".join(self.parts)
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        return _BLANK_RE.sub("\n\n", text).strip()

    def _emit(self, text: str) -> None:
        self.parts.append(text)

    def _block(self, node: Node, list_depth: int = 0) -> None:
        for child in node.children:
            self._node(child, list_depth)

    def _node(self, node: Node, list_depth: int) -> None:
        tag = node.tag
        if tag in SKIPPED_TAGS or tag in BOILERPLATE_TAGS:
            return
        if tag == TEXT:
            self._emit(_WS_RE.sub(" ", node.data.replace("\n", " ")))
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            heading = node.text().rstrip("¶#").rstrip()
            if heading:
                self._emit(f"\n\n{'#' * int(tag[1])} {heading}\n\n")
            return
        if tag == "br":
            self._emit("\n")
            return
        if tag == "hr":
            self._emit("\n\n---\n\n")
            return
        if tag == "pre":
            code = node.select_one("code") or node
            body = code.raw_text().strip("\n")
            if body.strip():
                fence = _fence_for(body)
                self._emit(f"\n\n{fence}\n{body}\n{fence}\n\n")
            return
        if tag == "code":
            body = node.text()
            if body:
                self._emit(f"`{body}`")
            return
        if tag in {"strong", "b"}:
            self._inline(node, "**")
            return
        if tag in {"em", "i"}:
            self._inline(node, "*")
            return
        if tag == "a":
            self._anchor(node, list_depth)
            return
        if tag == "img":
            alt = node.get("alt").strip()
            if alt:
                self._emit(f"![{alt}]({self._absolute(node.get('src'))})")
            return
        if tag in {"ul", "ol"}:
            self._list(node, list_depth)
            return
        if tag == "blockquote":
            quoted = _MarkdownWriter(self.base_url).render(node)
            if quoted:
                body = "\n".join("> " + line for line in quoted.split("\n"))
                self._emit(f"\n\n{body}\n\n")
            return
        if tag == "table":
            self._table(node)
            return
        if tag in BLOCK_TAGS:
            self._emit("\n\n")
            self._block(node, list_depth)
            self._emit("\n\n")
            return
        self._block(node, list_depth)

    def _inline(self, node: Node, marker: str) -> None:
        body = node.text()
        if body:
            self._emit(f"{marker}{body}{marker}")

    def _anchor(self, node: Node, list_depth: int) -> None:
        label = node.text()
        href = self._absolute(node.get("href"))
        if not label:
            self._block(node, list_depth)
            return
        if not href or href.startswith("javascript:"):
            self._emit(label)
            return
        self._emit(f"[{label}]({href})")

    def _list(self, node: Node, list_depth: int) -> None:
        items = [c for c in node.children if c.tag == "li"]
        if not items:
            return
        self._emit("\n\n")
        indent = "  " * list_depth
        for index, item in enumerate(items, start=1):
            bullet = f"{index}." if node.tag == "ol" else "-"
            inner = _MarkdownWriter(self.base_url)
            inner._block(item, list_depth + 1)
            body = _BLANK_RE.sub("\n\n", "".join(inner.parts)).strip()
            if not body:
                continue
            lines = body.split("\n")
            self._emit(f"{indent}{bullet} {lines[0]}\n")
            for line in lines[1:]:
                self._emit(f"{indent}  {line}\n")
        self._emit("\n")

    def _table(self, node: Node) -> None:
        rows = node.select("tr")
        if not rows:
            return
        rendered: list[list[str]] = []
        for row in rows:
            cells = [
                c for c in row.children if c.tag in {"td", "th"}
            ] or row.select("td, th")
            rendered.append([c.text().replace("|", "\\|") for c in cells])
        rendered = [r for r in rendered if any(cell for cell in r)]
        if not rendered:
            return
        width = max(len(r) for r in rendered)
        lines = ["| " + " | ".join(r + [""] * (width - len(r))) + " |" for r in rendered]
        separator = "| " + " | ".join(["---"] * width) + " |"
        self._emit("\n\n" + "\n".join([lines[0], separator, *lines[1:]]) + "\n\n")

    def _absolute(self, href: str) -> str:
        """Resolve `href` against the page URL, when both are usable."""
        href = href.strip()
        if not href or not self.base_url:
            return href
        from urllib.parse import urljoin

        try:
            return urljoin(self.base_url, href)
        except ValueError:
            return href


def html_to_markdown(html: str, *, base_url: str = "", article: bool = True) -> str:
    """Convert `html` to Markdown, optionally trimming to the article body.

    Relative links and image sources are resolved against `base_url` so the
    model receives URLs it can pass straight back to `WebFetch`.
    """
    root = parse_html(html)
    node = main_content(root) if article else (root.select_one("body") or root)
    return _MarkdownWriter(base_url).render(node)


__all__ = [
    "Node",
    "parse_html",
    "html_to_markdown",
    "main_content",
    "page_title",
    "page_description",
]
