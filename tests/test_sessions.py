"""Tests for session persistence, forking, and listing."""

from __future__ import annotations

from pathlib import Path

from ubiquity.sessions import SessionRecord, SessionStore, project_slug
from ubiquity.types import SDKAssistantMessage, SDKUserMessage


def test_project_slug_is_filesystem_safe() -> None:
    slug = project_slug(Path("/Users/x/my project/sub"))
    assert "/" not in slug
    assert " " not in slug


def test_project_slug_never_empty() -> None:
    assert project_slug(Path("/")) == "root"


def test_append_and_read_roundtrip(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    cwd = tmp_path / "proj"

    store.append("s1", cwd, SDKUserMessage(content="hello"))
    store.append("s1", cwd, SDKAssistantMessage(content=[{"type": "text", "text": "hi"}]))

    records = store.read("s1", cwd)
    assert [r.type for r in records] == ["user", "assistant"]
    assert records[0].payload["content"] == "hello"


def test_records_chain_through_parent_uuid(tmp_path: Path) -> None:
    """The parent chain is what makes forking possible."""
    store = SessionStore(tmp_path / "sessions")
    cwd = tmp_path / "proj"

    store.append("s1", cwd, SDKUserMessage(content="one"))
    store.append("s1", cwd, SDKUserMessage(content="two"))
    store.append("s1", cwd, SDKUserMessage(content="three"))

    records = store.read("s1", cwd)
    assert records[0].parent_uuid is None
    assert records[1].parent_uuid == records[0].uuid
    assert records[2].parent_uuid == records[1].uuid


def test_torn_line_does_not_break_reading(tmp_path: Path) -> None:
    """A crashed run leaves a partial line; the rest must still parse."""
    store = SessionStore(tmp_path / "sessions")
    cwd = tmp_path / "proj"
    store.append("s1", cwd, SDKUserMessage(content="good"))

    path = store.path_for("s1", cwd)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"uuid": "broken", "type"\n')

    records = store.read("s1", cwd)
    assert len(records) == 1
    assert records[0].payload["content"] == "good"


def test_fork_creates_independent_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    cwd = tmp_path / "proj"
    for text in ("one", "two", "three"):
        store.append("s1", cwd, SDKUserMessage(content=text))

    forked = store.fork("s1", cwd)
    assert forked != "s1"

    original = store.read("s1", cwd)
    copy = store.read(forked, cwd)
    assert [r.payload["content"] for r in copy] == ["one", "two", "three"]
    assert {r.uuid for r in copy}.isdisjoint({r.uuid for r in original})


def test_fork_rewrites_parent_chain(tmp_path: Path) -> None:
    """A fork must be internally consistent, not point at the source's UUIDs."""
    store = SessionStore(tmp_path / "sessions")
    cwd = tmp_path / "proj"
    for text in ("one", "two"):
        store.append("s1", cwd, SDKUserMessage(content=text))

    copy = store.read(store.fork("s1", cwd), cwd)
    assert copy[0].parent_uuid is None
    assert copy[1].parent_uuid == copy[0].uuid


def test_fork_up_to_message_truncates(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    cwd = tmp_path / "proj"
    for text in ("one", "two", "three"):
        store.append("s1", cwd, SDKUserMessage(content=text))

    records = store.read("s1", cwd)
    copy = store.read(store.fork("s1", cwd, up_to_uuid=records[1].uuid), cwd)
    assert [r.payload["content"] for r in copy] == ["one", "two"]


def test_forking_then_appending_does_not_touch_original(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    cwd = tmp_path / "proj"
    store.append("s1", cwd, SDKUserMessage(content="shared"))

    forked = store.fork("s1", cwd)
    store.append(forked, cwd, SDKUserMessage(content="only in fork"))

    assert len(store.read("s1", cwd)) == 1
    assert len(store.read(forked, cwd)) == 2


def test_info_reports_metadata(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    cwd = tmp_path / "proj"
    store.append("s1", cwd, SDKUserMessage(content="first prompt"))
    store.append("s1", cwd, SDKAssistantMessage(content=[]))

    info = store.info("s1", cwd)
    assert info is not None
    assert info.message_count == 2
    assert info.summary == "first prompt"


def test_info_returns_none_for_unknown_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    assert store.info("nope", tmp_path / "proj") is None


def test_rename_sets_title(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    cwd = tmp_path / "proj"
    store.append("s1", cwd, SDKUserMessage(content="x"))
    store.rename("s1", "My Session", cwd)

    info = store.info("s1", cwd)
    assert info is not None
    assert info.title == "My Session"


def test_list_returns_newest_first(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    cwd = tmp_path / "proj"
    store.append("old", cwd, SDKUserMessage(content="a"))
    store.append("new", cwd, SDKUserMessage(content="b"))

    path = store.path_for("old", cwd)
    lines = path.read_text().splitlines()
    record = SessionRecord.from_json(lines[0])
    record.timestamp = 1.0
    path.write_text(record.to_json() + "\n")

    listed = store.list(cwd)
    assert [s.session_id for s in listed][0] == "new"


def test_list_respects_limit(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    cwd = tmp_path / "proj"
    for i in range(5):
        store.append(f"s{i}", cwd, SDKUserMessage(content="x"))
    assert len(store.list(cwd, limit=3)) == 3


def test_list_of_missing_root_is_empty(tmp_path: Path) -> None:
    assert SessionStore(tmp_path / "nothing").list() == []


def test_find_locates_session_across_projects(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    store.append("s1", tmp_path / "projA", SDKUserMessage(content="x"))
    assert store.find("s1") is not None
    assert store.find("missing") is None


def test_delete_removes_transcript(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    cwd = tmp_path / "proj"
    store.append("s1", cwd, SDKUserMessage(content="x"))

    assert store.delete("s1", cwd) is True
    assert store.read("s1", cwd) == []
    assert store.delete("s1", cwd) is False


def test_sessions_for_different_projects_do_not_collide(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    store.append("same-id", tmp_path / "projA", SDKUserMessage(content="a"))
    store.append("same-id", tmp_path / "projB", SDKUserMessage(content="b"))

    a = store.read("same-id", tmp_path / "projA")
    b = store.read("same-id", tmp_path / "projB")
    assert a[0].payload["content"] == "a"
    assert b[0].payload["content"] == "b"
