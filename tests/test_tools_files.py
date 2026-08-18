"""Tests for the file tools and the read-before-write invariant.

Read-before-write is the invariant that stops the agent from clobbering
changes it has not seen. A write is permitted only when the file does not
exist, or the agent has fully read it and it has not changed since.
"""

from __future__ import annotations

import os
from pathlib import Path

from ubiquity.tools import EditTool, GlobTool, GrepTool, ReadTool, TodoWriteTool, WriteTool
from ubiquity.tools._files import MAX_LINES
from ubiquity.tools.edit import EditInput
from ubiquity.tools.glob import GlobInput
from ubiquity.tools.grep import GrepInput
from ubiquity.tools.read import ReadInput
from ubiquity.tools.todo import TodoInput, TodoItem
from ubiquity.tools.write import WriteInput, is_sensitive


async def test_read_returns_numbered_lines(make_ctx, tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("alpha\nbeta\n")
    ctx = make_ctx(cwd=tmp_path)

    out = await ReadTool().call(ReadInput(file_path=str(target)), ctx)
    assert "     1\talpha" in out.content
    assert "     2\tbeta" in out.content


async def test_read_rejects_missing_file(make_ctx, tmp_path: Path) -> None:
    ctx = make_ctx(cwd=tmp_path)
    error = await ReadTool().validate_input(
        ReadInput(file_path=str(tmp_path / "nope.txt")), ctx
    )
    assert error is not None
    assert "does not exist" in error.message


async def test_read_rejects_path_outside_cwd(make_ctx, tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")
    ctx = make_ctx(cwd=tmp_path / "inner")
    (tmp_path / "inner").mkdir()

    error = await ReadTool().validate_input(ReadInput(file_path=str(outside)), ctx)
    assert error is not None
    assert "outside the working directory" in error.message


async def test_read_slice_is_marked_partial(make_ctx, tmp_path: Path) -> None:
    target = tmp_path / "big.txt"
    target.write_text("\n".join(f"line{i}" for i in range(100)))
    ctx = make_ctx(cwd=tmp_path)

    await ReadTool().call(ReadInput(file_path=str(target), offset=10, limit=5), ctx)
    assert ctx.file_state[target.resolve()].is_partial_view is True


async def test_write_creates_new_file_without_prior_read(make_ctx, tmp_path: Path) -> None:
    """A file that does not exist yet has nothing to clobber."""
    ctx = make_ctx(cwd=tmp_path)
    target = tmp_path / "new.txt"

    error = await WriteTool().validate_input(
        WriteInput(file_path=str(target), content="hello"), ctx
    )
    assert error is None

    await WriteTool().call(WriteInput(file_path=str(target), content="hello"), ctx)
    assert target.read_text() == "hello"


async def test_write_refuses_unread_existing_file(make_ctx, tmp_path: Path) -> None:
    target = tmp_path / "exists.txt"
    target.write_text("original")
    ctx = make_ctx(cwd=tmp_path)

    error = await WriteTool().validate_input(
        WriteInput(file_path=str(target), content="clobber"), ctx
    )
    assert error is not None
    assert "has not been read" in error.message
    assert target.read_text() == "original"


async def test_write_allowed_after_read(make_ctx, tmp_path: Path) -> None:
    target = tmp_path / "exists.txt"
    target.write_text("original")
    ctx = make_ctx(cwd=tmp_path)

    await ReadTool().call(ReadInput(file_path=str(target)), ctx)
    error = await WriteTool().validate_input(
        WriteInput(file_path=str(target), content="updated"), ctx
    )
    assert error is None


async def test_write_refuses_after_external_modification(make_ctx, tmp_path: Path) -> None:
    """A file changed since the read must be re-read before writing."""
    target = tmp_path / "exists.txt"
    target.write_text("original")
    ctx = make_ctx(cwd=tmp_path)

    await ReadTool().call(ReadInput(file_path=str(target)), ctx)

    state = ctx.file_state[target.resolve()]
    target.write_text("changed by a linter")
    os.utime(target, (state.mtime + 10, state.mtime + 10))

    error = await WriteTool().validate_input(
        WriteInput(file_path=str(target), content="clobber"), ctx
    )
    assert error is not None
    assert "modified since read" in error.message


async def test_partial_read_does_not_authorize_write(make_ctx, tmp_path: Path) -> None:
    """A partial view would silently discard the unread remainder."""
    target = tmp_path / "big.txt"
    target.write_text("\n".join(f"line{i}" for i in range(100)))
    ctx = make_ctx(cwd=tmp_path)

    await ReadTool().call(ReadInput(file_path=str(target), offset=1, limit=5), ctx)
    error = await WriteTool().validate_input(
        WriteInput(file_path=str(target), content="short"), ctx
    )
    assert error is not None
    assert "Only part of this file has been read" in error.message
    assert "limit" in error.message


async def test_a_file_longer_than_the_cap_is_still_editable(
    make_ctx, tmp_path: Path
) -> None:
    """The way out of a partial view has to exist, and the description names it.

    A plain Read of a long file stops at `MAX_LINES` and so counts as partial,
    which blocks every later Edit. Reading again with `limit` past the last
    line is the only escape, so it is worth holding rather than rediscovering.
    """
    target = tmp_path / "long.txt"
    target.write_text("\n".join(f"line{i}" for i in range(MAX_LINES + 500)))
    ctx = make_ctx(cwd=tmp_path)

    await ReadTool().call(ReadInput(file_path=str(target)), ctx)
    edit = EditInput(file_path=str(target), old_string="line7\n", new_string="seven\n")
    assert await EditTool().validate_input(edit, ctx) is not None

    await ReadTool().call(
        ReadInput(file_path=str(target), limit=MAX_LINES + 1000), ctx
    )
    assert await EditTool().validate_input(edit, ctx) is None


async def test_write_records_state_so_edit_follows(make_ctx, tmp_path: Path) -> None:
    """Writing a file counts as having seen it."""
    ctx = make_ctx(cwd=tmp_path)
    target = tmp_path / "new.txt"

    await WriteTool().call(WriteInput(file_path=str(target), content="a\nb\n"), ctx)
    error = await EditTool().validate_input(
        EditInput(file_path=str(target), old_string="a", new_string="c"), ctx
    )
    assert error is None


async def test_edit_replaces_unique_occurrence(make_ctx, tmp_path: Path) -> None:
    target = tmp_path / "f.py"
    target.write_text("x = 1\ny = 2\n")
    ctx = make_ctx(cwd=tmp_path)
    await ReadTool().call(ReadInput(file_path=str(target)), ctx)

    await EditTool().call(
        EditInput(file_path=str(target), old_string="x = 1", new_string="x = 42"), ctx
    )
    assert target.read_text() == "x = 42\ny = 2\n"


async def test_edit_rejects_ambiguous_match(make_ctx, tmp_path: Path) -> None:
    """An ambiguous edit is rejected rather than guessed at."""
    target = tmp_path / "f.py"
    target.write_text("v = 1\nv = 1\n")
    ctx = make_ctx(cwd=tmp_path)
    await ReadTool().call(ReadInput(file_path=str(target)), ctx)

    error = await EditTool().validate_input(
        EditInput(file_path=str(target), old_string="v = 1", new_string="v = 2"), ctx
    )
    assert error is not None
    assert "appears 2 times" in error.message


async def test_edit_replace_all_accepts_ambiguous_match(make_ctx, tmp_path: Path) -> None:
    target = tmp_path / "f.py"
    target.write_text("v = 1\nv = 1\n")
    ctx = make_ctx(cwd=tmp_path)
    await ReadTool().call(ReadInput(file_path=str(target)), ctx)

    out = await EditTool().call(
        EditInput(
            file_path=str(target), old_string="v = 1", new_string="v = 2", replace_all=True
        ),
        ctx,
    )
    assert target.read_text() == "v = 2\nv = 2\n"
    assert out.metadata["replacements"] == 2


async def test_edit_rejects_missing_string(make_ctx, tmp_path: Path) -> None:
    target = tmp_path / "f.py"
    target.write_text("x = 1\n")
    ctx = make_ctx(cwd=tmp_path)
    await ReadTool().call(ReadInput(file_path=str(target)), ctx)

    error = await EditTool().validate_input(
        EditInput(file_path=str(target), old_string="nope", new_string="y"), ctx
    )
    assert error is not None
    assert "was not found" in error.message


async def test_edit_rejects_identical_strings(make_ctx, tmp_path: Path) -> None:
    target = tmp_path / "f.py"
    target.write_text("x = 1\n")
    ctx = make_ctx(cwd=tmp_path)
    await ReadTool().call(ReadInput(file_path=str(target)), ctx)

    error = await EditTool().validate_input(
        EditInput(file_path=str(target), old_string="x", new_string="x"), ctx
    )
    assert error is not None
    assert "identical" in error.message


def test_sensitive_paths_are_flagged() -> None:
    assert is_sensitive(Path("/home/u/project/.env")) is True
    assert is_sensitive(Path("/home/u/.ssh/id_rsa")) is True
    assert is_sensitive(Path("/home/u/project/.git/config")) is True
    assert is_sensitive(Path("/home/u/project/main.py")) is False


async def test_sensitive_write_asks_even_in_accept_edits(make_ctx, tmp_path: Path) -> None:
    """acceptEdits must not silently auto-approve a credential file."""
    ctx = make_ctx(cwd=tmp_path, mode="acceptEdits")
    result = await WriteTool().check_permissions(
        WriteInput(file_path=str(tmp_path / ".env"), content="KEY=1"), ctx
    )
    assert result.behavior == "ask"
    assert result.bypass_immune is True


async def test_ordinary_write_auto_accepts_in_accept_edits(make_ctx, tmp_path: Path) -> None:
    ctx = make_ctx(cwd=tmp_path, mode="acceptEdits")
    result = await WriteTool().check_permissions(
        WriteInput(file_path=str(tmp_path / "main.py"), content="x"), ctx
    )
    assert result.behavior == "allow"


async def test_glob_sorts_newest_first(make_ctx, tmp_path: Path) -> None:
    old, new = tmp_path / "old.py", tmp_path / "new.py"
    old.write_text("a")
    new.write_text("b")
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))

    out = await GlobTool().call(GlobInput(pattern="*.py"), make_ctx(cwd=tmp_path))
    lines = out.content.splitlines()
    assert lines[0].endswith("new.py")
    assert lines[1].endswith("old.py")


async def test_glob_skips_vendor_directories(make_ctx, tmp_path: Path) -> None:
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.py").write_text("x")
    (tmp_path / "mine.py").write_text("y")

    out = await GlobTool().call(GlobInput(pattern="**/*.py"), make_ctx(cwd=tmp_path))
    assert "mine.py" in out.content
    assert "node_modules" not in out.content


async def test_grep_finds_matches(make_ctx, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("import os\nimport sys\n")
    out = await GrepTool().call(
        GrepInput(pattern=r"^import", path=str(tmp_path)), make_ctx(cwd=tmp_path)
    )
    assert out.metadata["count"] >= 2


async def test_grep_context_lines_work_without_ripgrep(
    make_ctx, tmp_path: Path, monkeypatch
) -> None:
    """`context_lines` used to be silently dropped whenever `rg` was absent."""
    monkeypatch.setattr("ubiquity.tools.grep.shutil.which", lambda _: None)
    (tmp_path / "f.txt").write_text("one\ntwo\nneedle\nfour\nfive\n")

    result = await GrepTool().call(
        GrepInput(pattern="needle", path=str(tmp_path), context_lines=1),
        make_ctx(cwd=tmp_path),
    )
    assert "f.txt:3:needle" in result.content
    assert "f.txt-2-two" in result.content
    assert "f.txt-4-four" in result.content
    assert "one" not in result.content


async def test_grep_overlapping_context_is_not_repeated(
    make_ctx, tmp_path: Path, monkeypatch
) -> None:
    """Two nearby matches share their context rather than emitting it twice."""
    monkeypatch.setattr("ubiquity.tools.grep.shutil.which", lambda _: None)
    (tmp_path / "f.txt").write_text("a\nhit\nmiddle\nhit\nb\n")

    result = await GrepTool().call(
        GrepInput(pattern="hit", path=str(tmp_path), context_lines=1),
        make_ctx(cwd=tmp_path),
    )
    assert result.content.count("middle") == 1
    assert result.content.count(":hit") == 2


async def test_grep_rejects_invalid_regex(make_ctx, tmp_path: Path) -> None:
    error = await GrepTool().validate_input(
        GrepInput(pattern="a[b"), make_ctx(cwd=tmp_path)
    )
    assert error is not None
    assert "Invalid regular expression" in error.message


async def test_grep_rejects_unknown_output_mode(make_ctx, tmp_path: Path) -> None:
    error = await GrepTool().validate_input(
        GrepInput(pattern="x", output_mode="bogus"), make_ctx(cwd=tmp_path)
    )
    assert error is not None
    assert "Unknown output_mode" in error.message


async def test_todo_rejects_multiple_in_progress(make_ctx) -> None:
    error = await TodoWriteTool().validate_input(
        TodoInput(
            todos=[
                TodoItem(content="one", status="in_progress"),
                TodoItem(content="two", status="in_progress"),
            ]
        ),
        make_ctx(),
    )
    assert error is not None
    assert "in_progress" in error.message


async def test_todo_accepts_single_in_progress(make_ctx) -> None:
    ctx = make_ctx()
    args = TodoInput(
        todos=[
            TodoItem(content="one", status="in_progress"),
            TodoItem(content="two", status="pending"),
        ]
    )
    assert await TodoWriteTool().validate_input(args, ctx) is None

    out = await TodoWriteTool().call(args, ctx)
    assert "[~]" in out.content and "one" in out.content
    assert "[ ]" in out.content and "two" in out.content
