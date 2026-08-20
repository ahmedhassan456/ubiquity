"""Tests for `UBIQUITY.md` memory files.

The shape under test is that nothing is read unless it was asked for, that
what is read is ordered weakest first and identically every run, and that an
``@include`` cannot be used to read a file outside the tree it was written in.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ubiquity.memory import (
    MAX_INCLUDE_DEPTH,
    MAX_MEMORY_CHARS,
    MemoryFile,
    include_paths,
    load_memory,
    memory_paths,
    render_memory,
)
from ubiquity.options import Options
from ubiquity.prompts import build_system_prompt


def write(path: Path, text: str) -> Path:
    """Create `path` and its parents with `text` in it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def names(files: list[MemoryFile]) -> list[str]:
    """The file names of a load, in load order."""
    return [f.path.name for f in files]


class TestNothingIsReadUnlessAskedFor:
    def test_no_sources_means_no_files(self, tmp_path: Path) -> None:
        write(tmp_path / "UBIQUITY.md", "project rule")
        assert load_memory((), tmp_path) == []

    def test_naming_the_source_reads_it(self, tmp_path: Path) -> None:
        write(tmp_path / "UBIQUITY.md", "project rule")
        loaded = load_memory(("project",), tmp_path)
        assert [f.content for f in loaded] == ["project rule"]

    def test_a_source_does_not_read_a_sibling_source(self, tmp_path: Path) -> None:
        write(tmp_path / "UBIQUITY.md", "project rule")
        write(tmp_path / "UBIQUITY.local.md", "local rule")
        assert names(load_memory(("project",), tmp_path)) == ["UBIQUITY.md"]
        assert names(load_memory(("local",), tmp_path)) == ["UBIQUITY.local.md"]

    def test_an_empty_file_contributes_nothing(self, tmp_path: Path) -> None:
        write(tmp_path / "UBIQUITY.md", "   \n\n  ")
        assert load_memory(("project",), tmp_path) == []

    def test_an_unreadable_file_does_not_stop_the_run(self, tmp_path: Path) -> None:
        (tmp_path / "UBIQUITY.md").write_bytes(b"\xff\xfe\x00 bad")
        write(tmp_path / ".ubiquity" / "UBIQUITY.md", "still here")
        assert [f.content for f in load_memory(("project",), tmp_path)] == ["still here"]


class TestWeakestFirst:
    def test_an_ancestor_is_read_before_the_working_directory(
        self, tmp_path: Path
    ) -> None:
        write(tmp_path / "UBIQUITY.md", "root")
        nested = tmp_path / "a" / "b"
        write(nested / "UBIQUITY.md", "nested")
        loaded = load_memory(("project",), nested)
        assert [f.content for f in loaded] == ["root", "nested"]

    def test_the_dot_directory_outranks_the_bare_file(self, tmp_path: Path) -> None:
        write(tmp_path / "UBIQUITY.md", "bare")
        write(tmp_path / ".ubiquity" / "UBIQUITY.md", "scoped")
        assert [f.content for f in load_memory(("project",), tmp_path)] == [
            "bare",
            "scoped",
        ]

    def test_sources_are_ordered_user_then_project_then_local(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        write(home / ".ubiquity" / "UBIQUITY.md", "user")
        work = tmp_path / "work"
        write(work / "UBIQUITY.md", "project")
        write(work / "UBIQUITY.local.md", "local")
        loaded = load_memory(("local", "user", "project"), work)
        assert [f.content for f in loaded] == ["user", "project", "local"]

    def test_explicit_files_are_read_last(self, tmp_path: Path) -> None:
        write(tmp_path / "UBIQUITY.md", "project")
        extra = write(tmp_path / "extra" / "rules.md", "explicit")
        loaded = load_memory(("project",), tmp_path, [extra])
        assert [f.content for f in loaded] == ["project", "explicit"]

    def test_a_file_reached_twice_keeps_its_weakest_position(
        self, tmp_path: Path
    ) -> None:
        """Adding a source may add instructions but must never reorder them."""
        both = write(tmp_path / "UBIQUITY.md", "once")
        write(tmp_path / ".ubiquity" / "UBIQUITY.md", "after")
        loaded = load_memory(("project",), tmp_path, [both])
        assert [f.content for f in loaded] == ["once", "after"]

    def test_the_order_is_the_same_every_time(self, tmp_path: Path) -> None:
        for name in ("m", "z", "a", "b"):
            write(tmp_path / f"{name}.md", name)
        write(
            tmp_path / "UBIQUITY.md",
            "root\n\n@z.md\n@a.md\n@m.md\n@b.md\n",
        )
        runs = {tuple(names(load_memory(("project",), tmp_path))) for _ in range(5)}
        assert runs == {("UBIQUITY.md", "z.md", "a.md", "m.md", "b.md")}


class TestIncludes:
    def test_an_include_is_loaded_after_the_file_that_names_it(
        self, tmp_path: Path
    ) -> None:
        write(tmp_path / "style.md", "two spaces")
        write(tmp_path / "UBIQUITY.md", "see @style.md for the rest")
        loaded = load_memory(("project",), tmp_path)
        assert [f.content for f in loaded] == ["see @style.md for the rest", "two spaces"]
        assert loaded[1].parent == (tmp_path / "UBIQUITY.md").resolve()

    def test_relative_and_dot_forms_both_resolve(self, tmp_path: Path) -> None:
        write(tmp_path / "docs" / "a.md", "a")
        write(tmp_path / "docs" / "b.md", "b")
        write(tmp_path / "UBIQUITY.md", "@docs/a.md and @./docs/b.md")
        assert names(load_memory(("project",), tmp_path))[1:] == ["a.md", "b.md"]

    def test_an_include_resolves_against_its_own_directory(
        self, tmp_path: Path
    ) -> None:
        write(tmp_path / "docs" / "deep.md", "deep")
        write(tmp_path / "docs" / "mid.md", "@deep.md")
        write(tmp_path / "UBIQUITY.md", "@docs/mid.md")
        assert names(load_memory(("project",), tmp_path)) == [
            "UBIQUITY.md",
            "mid.md",
            "deep.md",
        ]

    def test_a_cycle_terminates(self, tmp_path: Path) -> None:
        write(tmp_path / "b.md", "b @UBIQUITY.md")
        write(tmp_path / "UBIQUITY.md", "a @b.md")
        assert names(load_memory(("project",), tmp_path)) == ["UBIQUITY.md", "b.md"]

    def test_depth_is_capped(self, tmp_path: Path) -> None:
        depth = MAX_INCLUDE_DEPTH + 3
        for i in range(depth):
            write(tmp_path / f"f{i}.md", f"level {i} @f{i + 1}.md")
        write(tmp_path / "UBIQUITY.md", "top @f0.md")
        assert len(load_memory(("project",), tmp_path)) == MAX_INCLUDE_DEPTH

    def test_a_fenced_block_is_not_an_include(self, tmp_path: Path) -> None:
        write(tmp_path / "secret.md", "should not load")
        write(
            tmp_path / "UBIQUITY.md",
            "how to include:\n\n```\n@secret.md\n```\n\ndone",
        )
        assert names(load_memory(("project",), tmp_path)) == ["UBIQUITY.md"]

    def test_an_inline_code_span_is_not_an_include(self, tmp_path: Path) -> None:
        write(tmp_path / "secret.md", "should not load")
        write(tmp_path / "UBIQUITY.md", "write this: ` @secret.md ` to include it")
        assert names(load_memory(("project",), tmp_path)) == ["UBIQUITY.md"]

    def test_an_email_address_is_not_an_include(self, tmp_path: Path) -> None:
        assert include_paths("mail someone@example.com", tmp_path) == []

    def test_an_include_at_the_end_of_a_line_works(self, tmp_path: Path) -> None:
        """The form that always worked, pinned so trimming cannot break it."""
        write(tmp_path / "style.md", "two spaces")
        write(tmp_path / "UBIQUITY.md", "the rest is in @style.md")
        assert names(load_memory(("project",), tmp_path)) == [
            "UBIQUITY.md",
            "style.md",
        ]

    @pytest.mark.parametrize(
        "sentence",
        [
            "rules live in @style.md, and more follow",
            "rules live in @style.md.",
            "rules live in @style.md; more follow",
            "rules live in @style.md: like so",
            "have you read @style.md?",
            "read @style.md!",
            "see (@style.md) for the rest",
            "see [@style.md] for the rest",
            'see "@style.md" for the rest',
        ],
    )
    def test_punctuation_around_a_reference_is_not_part_of_it(
        self, tmp_path: Path, sentence: str
    ) -> None:
        """An `@path` written mid-sentence has punctuation on both sides of it.

        The capture runs to the next whitespace, so without trimming the
        filename ends up as `style.md,` -- a file nobody has, missing in
        silence. Only a reference that was the last word on its line worked.
        """
        write(tmp_path / "style.md", "two spaces")
        write(tmp_path / "UBIQUITY.md", sentence)
        assert names(load_memory(("project",), tmp_path)) == [
            "UBIQUITY.md",
            "style.md",
        ]

    def test_the_readme_example_resolves(self, tmp_path: Path) -> None:
        """The documented example is the one a reader will copy first."""
        write(tmp_path / "docs" / "style.md", "two spaces")
        write(tmp_path / "notes" / "release.md", "tag, then push")
        write(
            tmp_path / "UBIQUITY.md",
            "Style rules live in @docs/style.md, and the release steps in "
            "@notes/release.md.",
        )
        assert names(load_memory(("project",), tmp_path)) == [
            "UBIQUITY.md",
            "style.md",
            "release.md",
        ]

    def test_two_references_on_one_line_stay_separate(self, tmp_path: Path) -> None:
        write(tmp_path / "a.md", "a")
        write(tmp_path / "b.md", "b")
        write(tmp_path / "UBIQUITY.md", "first @a.md, then @b.md.")
        assert names(load_memory(("project",), tmp_path))[1:] == ["a.md", "b.md"]

    def test_an_escaped_space_survives_trimming(self, tmp_path: Path) -> None:
        """Trimming works on the tail, so it must not undo the escape handling."""
        write(tmp_path / "my notes.md", "notes")
        write(tmp_path / "UBIQUITY.md", "see @my\\ notes.md, please")
        assert names(load_memory(("project",), tmp_path))[1:] == ["my notes.md"]

    def test_a_bare_at_is_not_an_include(self, tmp_path: Path) -> None:
        """Trimming can empty a reference, and an empty one names nothing."""
        assert include_paths("ask @. or @, about it", tmp_path) == []

    def test_a_missing_include_is_ignored(self, tmp_path: Path) -> None:
        write(tmp_path / "UBIQUITY.md", "@nope.md")
        assert names(load_memory(("project",), tmp_path)) == ["UBIQUITY.md"]

    def test_a_non_text_include_is_refused(self, tmp_path: Path) -> None:
        write(tmp_path / "key.pem", "-----BEGIN PRIVATE KEY-----")
        write(tmp_path / "UBIQUITY.md", "@key.pem")
        assert names(load_memory(("project",), tmp_path)) == ["UBIQUITY.md"]


class TestAnIncludeCannotLeaveItsTree:
    def test_a_project_file_cannot_read_outside_the_working_directory(
        self, tmp_path: Path
    ) -> None:
        """A checked-in file is written by whoever wrote the repository."""
        write(tmp_path / "outside" / "secrets.md", "token")
        work = tmp_path / "work"
        write(work / "UBIQUITY.md", "@../outside/secrets.md")
        assert names(load_memory(("project",), work)) == ["UBIQUITY.md"]

    def test_a_project_file_cannot_read_the_home_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        write(home / "creds.md", "token")
        monkeypatch.setenv("HOME", str(home))
        work = tmp_path / "work"
        write(work / "UBIQUITY.md", "@~/creds.md")
        assert names(load_memory(("project",), work)) == ["UBIQUITY.md"]

    def test_a_user_file_may_read_within_the_home_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        write(home / "notes.md", "notes")
        write(home / ".ubiquity" / "UBIQUITY.md", "@~/notes.md")
        assert names(load_memory(("user",), tmp_path / "work")) == [
            "UBIQUITY.md",
            "notes.md",
        ]


class TestSize:
    def test_an_over_long_file_is_cut_and_says_so(self, tmp_path: Path) -> None:
        write(tmp_path / "UBIQUITY.md", "x" * (MAX_MEMORY_CHARS + 500))
        loaded = load_memory(("project",), tmp_path)
        assert loaded[0].truncated is True
        assert "was cut off here" in loaded[0].content
        assert len(loaded[0].content) < MAX_MEMORY_CHARS + 500

    def test_a_file_within_the_limit_is_untouched(self, tmp_path: Path) -> None:
        write(tmp_path / "UBIQUITY.md", "y" * (MAX_MEMORY_CHARS - 10))
        loaded = load_memory(("project",), tmp_path)
        assert loaded[0].truncated is False
        assert "cut off" not in loaded[0].content


class TestRendering:
    def test_nothing_loaded_renders_nothing(self) -> None:
        assert render_memory([]) == ""

    def test_each_file_is_named_with_its_origin(self, tmp_path: Path) -> None:
        write(tmp_path / "UBIQUITY.md", "the rule")
        text = render_memory(load_memory(("project",), tmp_path))
        assert "OVERRIDE" in text or "override" in text
        assert str((tmp_path / "UBIQUITY.md").resolve()) in text
        assert "checked into the codebase" in text
        assert "the rule" in text

    def test_an_included_file_says_who_included_it(self, tmp_path: Path) -> None:
        write(tmp_path / "style.md", "two spaces")
        write(tmp_path / "UBIQUITY.md", "@style.md")
        text = render_memory(load_memory(("project",), tmp_path))
        assert "included by" in text


class TestTheSystemPrompt:
    def test_memory_lands_in_the_prompt(self, tmp_path: Path) -> None:
        write(tmp_path / "UBIQUITY.md", "always use tabs")
        options = Options(cwd=tmp_path, memory_sources=("project",), model="x")
        loaded = load_memory(options.memory_sources, tmp_path)
        prompt = build_system_prompt(options, [], None, loaded)
        assert "always use tabs" in prompt

    def test_memory_comes_after_the_rest_and_before_the_appendix(
        self, tmp_path: Path
    ) -> None:
        write(tmp_path / "UBIQUITY.md", "always use tabs")
        options = Options(
            cwd=tmp_path,
            memory_sources=("project",),
            model="x",
            append_system_prompt="APPENDED",
        )
        loaded = load_memory(options.memory_sources, tmp_path)
        prompt = build_system_prompt(options, [], None, loaded)
        assert prompt.index("Working directory") < prompt.index("always use tabs")
        assert prompt.index("always use tabs") < prompt.index("APPENDED")

    def test_no_memory_adds_no_section(self, tmp_path: Path) -> None:
        options = Options(cwd=tmp_path, model="x")
        assert build_system_prompt(options, [], None, []) == build_system_prompt(
            options, [], None, None
        )


class TestOptions:
    def test_explicit_files_expand_a_leading_tilde(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        options = Options(cwd=tmp_path, model="x", memory=["~/notes.md"])
        assert options.resolved_memory_files() == [(tmp_path / "notes.md").resolve()]

    def test_the_user_path_is_under_the_dot_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        assert memory_paths("user", tmp_path) == [
            tmp_path / ".ubiquity" / "UBIQUITY.md"
        ]


def test_a_subagent_inherits_the_project_instructions(tmp_path: Path) -> None:
    """Delegating work does not suspend the rules the work is done under."""
    write(tmp_path / "UBIQUITY.md", "always use tabs")
    options = Options(cwd=tmp_path, memory_sources=("project",), model="x")
    loaded = load_memory(options.memory_sources, tmp_path)
    prompt = build_system_prompt(options, [], None, loaded, subagent=True)
    assert "always use tabs" in prompt
