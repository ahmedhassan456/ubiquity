"""Tests for permission rule parsing and matching.

The expectations here are pinned deliberately, edge cases included, so that
any drift in permission semantics shows up as a test failure rather than as a
rule that quietly starts matching more than it did.
"""

from __future__ import annotations

import pytest

from ubiquity.permissions.rules import (
    ExactRule,
    PrefixRule,
    WildcardRule,
    has_wildcards,
    matches_any,
    parse_rule,
)


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("git:*", PrefixRule),
        ("npm run:*", PrefixRule),
        ("git *", WildcardRule),
        ("* run *", WildcardRule),
        ("git status", ExactRule),
        (r"echo \*", ExactRule),
    ],
)
def test_rule_classification(pattern: str, expected: type) -> None:
    assert isinstance(parse_rule(pattern), expected)


@pytest.mark.parametrize(
    ("pattern", "command", "expected"),
    [
        ("git:*", "git", True),
        ("git:*", "git status", True),
        ("git:*", "gitk", False),
        ("ls:*", "lsof", False),
        ("ls:*", "lsattr", False),
    ],
)
def test_prefix_rules_respect_word_boundary(
    pattern: str, command: str, expected: bool
) -> None:
    assert parse_rule(pattern).matches(command) is expected


@pytest.mark.parametrize(
    ("pattern", "command", "expected"),
    [
        ("git *", "git", True),
        ("git *", "git add", True),
        ("git *", "git add -A", True),
        ("* run *", "npm run", False),
        ("* run *", "npm run build", True),
        ("cat *", "cat a\nb", True),
    ],
)
def test_wildcard_rules(pattern: str, command: str, expected: bool) -> None:
    assert parse_rule(pattern).matches(command) is expected


def test_trailing_wildcard_optional_only_when_sole_wildcard() -> None:
    """``git *`` matches bare ``git``; ``* run *`` must not match ``npm run``.

    Making the trailing wildcard optional is only correct when it is the only
    unescaped wildcard in the pattern.
    """
    assert parse_rule("git *").matches("git") is True
    assert parse_rule("* run *").matches("npm run") is False


@pytest.mark.parametrize(
    ("pattern", "command", "expected"),
    [
        ("git status", "git status", True),
        ("git status", "git status -s", False),
        ("git status", "git", False),
    ],
)
def test_exact_rules(pattern: str, command: str, expected: bool) -> None:
    assert parse_rule(pattern).matches(command) is expected


def test_prefix_rule_never_matches_compound_command() -> None:
    """``cd:*`` must not authorize ``cd /tmp && rm -rf .``.

    Shell escaping can smuggle a compound command past the caller's splitting
    pass, so the guard lives in the matcher itself.
    """
    rule = parse_rule("cd:*")
    assert rule.matches("cd /tmp && rm -rf .", is_compound=True) is False
    assert rule.matches("cd /tmp", is_compound=False) is True


def test_escaped_asterisk_is_not_a_wildcard() -> None:
    assert has_wildcards(r"echo \*") is False
    assert has_wildcards("echo *") is True
    assert has_wildcards("git:*") is False


def test_escaped_only_pattern_falls_through_to_exact_rule() -> None:
    """Escapes are not unescaped on the exact path.

    ``echo \\*`` has no unescaped wildcard, so it is classified exact and
    compared literally with its backslash intact. It therefore does not match
    ``echo *``. Pinned here so the classification cannot loosen unnoticed; see
    the module docstring.
    """
    rule = parse_rule(r"echo \*")
    assert rule == ExactRule(command=r"echo \*")
    assert rule.matches("echo *") is False
    assert rule.matches(r"echo \*") is True


def test_matches_any_returns_the_matching_pattern() -> None:
    """The matching pattern is returned so callers can report the rule."""
    assert matches_any(["npm:*", "git:*"], ["git push"]) == "git:*"
    assert matches_any(["npm:*"], ["git push"]) is None


def test_matches_any_checks_every_candidate() -> None:
    assert matches_any(["git:*"], ["npm test", "git push"]) == "git:*"


def test_case_insensitive_wildcard() -> None:
    assert parse_rule("GIT *", case_insensitive=True).matches("git push") is True
    assert parse_rule("GIT *").matches("git push") is False
