"""Permission rule parsing and matching.

Three rule forms are supported, tried in this order:

    prefix    ``git:*``      matches ``git`` and anything starting ``git ``
    wildcard  ``git push *`` ``*`` matches any run of characters
    exact     ``git status`` matches only that string

Escapes inside wildcard patterns: ``\\*`` is a literal asterisk and ``\\\\`` is a
literal backslash. A pattern whose only unescaped wildcard is a trailing
`` *`` also matches the bare command, so ``git *`` matches ``git`` — this
aligns wildcard rules with prefix-rule semantics.

One consequence of that ordering is worth stating plainly: a pattern whose
only asterisk is escaped contains no unescaped wildcard, so it is classified as
an exact rule and compared literally with its backslashes intact. ``echo \\*``
therefore matches the eight-character string ``echo \\*`` rather than ``echo *``,
and in practice matches nothing. It is left that way because the alternative is
a rule that quietly starts matching more than it used to, and a permission rule
that widens on its own is the worse failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TypeAlias

_ESCAPED_STAR = "\x00STAR\x00"
_ESCAPED_BACKSLASH = "\x00BSLASH\x00"

_PREFIX_RE = re.compile(r"^(.+):\*$")
_REGEX_SPECIAL_RE = re.compile(r"""[.+?^${}()|\[\]\\'"]""")


@dataclass(frozen=True, slots=True)
class ExactRule:
    """Matches only the literal command."""

    command: str

    def matches(self, candidate: str, *, is_compound: bool = False) -> bool:
        """True when `candidate` equals this rule's command exactly."""
        return self.command == candidate


@dataclass(frozen=True, slots=True)
class PrefixRule:
    """Matches the prefix itself or the prefix followed by a space."""

    prefix: str

    def matches(self, candidate: str, *, is_compound: bool = False) -> bool:
        """True when `candidate` is the prefix or begins with ``prefix + ' '``.

        Compound commands never match. A prefix rule such as ``cd:*`` must not
        authorize ``cd /tmp && rm -rf .``, and shell escaping can smuggle a
        compound command past the caller's splitting pass.
        """
        if is_compound:
            return False
        return candidate == self.prefix or candidate.startswith(self.prefix + " ")


@dataclass(frozen=True, slots=True)
class WildcardRule:
    """Matches via a regex compiled from a ``*`` glob pattern."""

    pattern: str
    regex: re.Pattern[str]

    def matches(self, candidate: str, *, is_compound: bool = False) -> bool:
        """True when the whole candidate matches the compiled pattern."""
        return self.regex.fullmatch(candidate) is not None


Rule: TypeAlias = ExactRule | PrefixRule | WildcardRule


def extract_prefix(pattern: str) -> str | None:
    """Return the prefix of a ``foo:*`` rule, or None if not that form."""
    match = _PREFIX_RE.match(pattern)
    return match.group(1) if match else None


def has_wildcards(pattern: str) -> bool:
    """True when the pattern contains an unescaped ``*`` outside ``:*`` form."""
    if pattern.endswith(":*"):
        return False
    for i, char in enumerate(pattern):
        if char != "*":
            continue
        backslashes = 0
        j = i - 1
        while j >= 0 and pattern[j] == "\\":
            backslashes += 1
            j -= 1
        if backslashes % 2 == 0:
            return True
    return False


def compile_wildcard(pattern: str, *, case_insensitive: bool = False) -> re.Pattern[str]:
    """Compile a ``*`` glob pattern into an anchored regex.

    Escape sequences are lifted out to placeholders before the regex-special
    characters are quoted, so ``\\*`` survives as a literal asterisk rather
    than becoming a wildcard.
    """
    trimmed = pattern.strip()

    processed: list[str] = []
    i = 0
    while i < len(trimmed):
        if trimmed[i] == "\\" and i + 1 < len(trimmed):
            nxt = trimmed[i + 1]
            if nxt == "*":
                processed.append(_ESCAPED_STAR)
                i += 2
                continue
            if nxt == "\\":
                processed.append(_ESCAPED_BACKSLASH)
                i += 2
                continue
        processed.append(trimmed[i])
        i += 1
    body = "".join(processed)

    escaped = _REGEX_SPECIAL_RE.sub(lambda m: "\\" + m.group(0), body)
    regex_body = escaped.replace("*", ".*")
    regex_body = regex_body.replace(_ESCAPED_STAR, r"\*").replace(
        _ESCAPED_BACKSLASH, r"\\"
    )

    if regex_body.endswith(" .*") and body.count("*") == 1:
        regex_body = regex_body[:-3] + "( .*)?"

    flags = re.DOTALL | (re.IGNORECASE if case_insensitive else 0)
    return re.compile(f"^{regex_body}$", flags)


def parse_rule(pattern: str, *, case_insensitive: bool = False) -> Rule:
    """Parse a rule-content string into the matching rule form."""
    prefix = extract_prefix(pattern)
    if prefix is not None:
        return PrefixRule(prefix=prefix)
    if has_wildcards(pattern):
        return WildcardRule(
            pattern=pattern,
            regex=compile_wildcard(pattern, case_insensitive=case_insensitive),
        )
    return ExactRule(command=pattern)


def matches_any(
    patterns: list[str],
    candidates: list[str],
    *,
    is_compound: bool = False,
    case_insensitive: bool = False,
) -> str | None:
    """Return the first pattern matching any candidate, or None.

    Use for deny and ask decisions, where one offending candidate is enough to
    trigger the behavior. The matching pattern is returned rather than a bool
    so callers can report which rule fired.
    """
    for pattern in patterns:
        rule = parse_rule(pattern, case_insensitive=case_insensitive)
        for candidate in candidates:
            if rule.matches(candidate, is_compound=is_compound):
                return pattern
    return None


def matches_all(
    patterns: list[str],
    candidates: list[str],
    *,
    is_compound: bool = False,
    case_insensitive: bool = False,
) -> list[str] | None:
    """Return the patterns authorizing every candidate, or None if any is unmatched.

    Use for allow decisions. A tool may present several candidates for one
    call — `Bash` returns each segment of a compound command — and authorizing
    the call requires a rule for every one of them. Matching on any single
    candidate would let ``Bash(git:*)`` authorize ``git status && rm -rf /``.

    An empty candidate list returns None, so a tool that exposes no matchable
    content can never be allowed by a content rule.
    """
    if not candidates:
        return None
    matched: list[str] = []
    for candidate in candidates:
        hit = matches_any(
            patterns,
            [candidate],
            is_compound=is_compound,
            case_insensitive=case_insensitive,
        )
        if hit is None:
            return None
        matched.append(hit)
    return matched


__all__ = [
    "ExactRule",
    "PrefixRule",
    "WildcardRule",
    "Rule",
    "extract_prefix",
    "has_wildcards",
    "compile_wildcard",
    "parse_rule",
    "matches_any",
]
