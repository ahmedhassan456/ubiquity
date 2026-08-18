"""Session persistence.

A session is a JSONL file, one record per line, appended as the run
proceeds. Append-only means a
crashed run still leaves a readable transcript up to the point of failure.

Records form a chain through `parent_uuid`, which is what makes forking
possible: a fork copies records up to a chosen point and remaps every UUID,
producing an independent session that shares history but diverges afterwards.

Layout:
    <session_dir>/<project-slug>/<session-id>.jsonl

The project slug is derived from the working directory so that sessions for
different projects do not collide in one flat namespace.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

DEFAULT_SESSION_ROOT = Path.home() / ".ubiquity" / "sessions"

_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")


def project_slug(cwd: Path) -> str:
    """Derive a filesystem-safe directory name from a working directory."""
    return _SLUG_RE.sub("-", str(Path(cwd).resolve())).strip("-") or "root"


@dataclass(slots=True)
class SessionRecord:
    """One line of a session transcript."""

    uuid: str
    type: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    parent_uuid: str | None = None

    def to_json(self) -> str:
        """Serialize this record as a single JSONL line."""
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, line: str) -> SessionRecord:
        """Parse a JSONL line into a record."""
        data = json.loads(line)
        return cls(
            uuid=data["uuid"],
            type=data["type"],
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", 0.0),
            parent_uuid=data.get("parent_uuid"),
        )


@dataclass(slots=True)
class SessionInfo:
    """Metadata about a stored session."""

    session_id: str
    path: Path
    cwd: str
    created_at: float
    updated_at: float
    message_count: int
    summary: str = ""
    title: str | None = None


def _message_to_payload(message: Any) -> dict[str, Any]:
    """Convert an SDK message into a JSON-serializable payload."""
    if is_dataclass(message) and not isinstance(message, type):
        return asdict(message)
    if isinstance(message, dict):
        return message
    return {"value": str(message)}


class SessionStore:
    """Reads and writes session transcripts under a root directory."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_SESSION_ROOT

    def path_for(self, session_id: str, cwd: Path) -> Path:
        """Return the transcript path for a session."""
        return self.root / project_slug(cwd) / f"{session_id}.jsonl"

    def append(self, session_id: str, cwd: Path, message: Any) -> SessionRecord:
        """Append one message to a session transcript.

        Returns the written record so the caller can chain the next one to it.
        """
        path = self.path_for(session_id, cwd)
        path.parent.mkdir(parents=True, exist_ok=True)

        last = self._last_uuid(path)
        record = SessionRecord(
            uuid=str(uuid4()),
            type=getattr(message, "type", "unknown"),
            payload=_message_to_payload(message),
            parent_uuid=last,
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(record.to_json() + "\n")
        return record

    def _last_uuid(self, path: Path) -> str | None:
        """Return the UUID of the final record, or None for a new session."""
        if not path.exists():
            return None
        last: str | None = None
        for record in self._iter_path(path):
            last = record.uuid
        return last

    def _iter_path(self, path: Path) -> Iterator[SessionRecord]:
        """Yield every parseable record in a transcript.

        Unparseable lines are skipped rather than raising, so one torn write
        at the tail of a crashed run does not make the whole session
        unreadable.
        """
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield SessionRecord.from_json(line)
                except (json.JSONDecodeError, KeyError):
                    continue

    def read(self, session_id: str, cwd: Path) -> list[SessionRecord]:
        """Return every record in a session, in write order."""
        return list(self._iter_path(self.path_for(session_id, cwd)))

    def find(self, session_id: str) -> Path | None:
        """Locate a session transcript across every project directory."""
        if not self.root.exists():
            return None
        for candidate in self.root.glob(f"*/{session_id}.jsonl"):
            return candidate
        return None

    def info(self, session_id: str, cwd: Path | None = None) -> SessionInfo | None:
        """Return metadata for one session, or None when it does not exist."""
        path = self.path_for(session_id, cwd) if cwd else self.find(session_id)
        if path is None or not path.exists():
            return None

        records = list(self._iter_path(path))
        if not records:
            return None

        summary = ""
        title = None
        for record in records:
            if record.type == "user" and not summary:
                content = record.payload.get("content")
                if isinstance(content, str):
                    summary = content[:120]
            if record.type == "title":
                title = record.payload.get("title")

        return SessionInfo(
            session_id=path.stem,
            path=path,
            cwd=str(path.parent.name),
            created_at=records[0].timestamp,
            updated_at=records[-1].timestamp,
            message_count=len(records),
            summary=summary,
            title=title,
        )

    def list(self, cwd: Path | None = None, limit: int = 50) -> list[SessionInfo]:
        """List sessions, newest first.

        Scans one project directory when `cwd` is given, every project
        otherwise.
        """
        if not self.root.exists():
            return []

        base = self.root / project_slug(cwd) if cwd else self.root
        pattern = "*.jsonl" if cwd else "*/*.jsonl"
        if not base.exists():
            return []

        out: list[SessionInfo] = []
        for path in base.glob(pattern):
            entry = self.info(path.stem, cwd)
            if entry is None and not cwd:
                entry = self._info_from_path(path)
            if entry is not None:
                out.append(entry)

        out.sort(key=lambda s: s.updated_at, reverse=True)
        return out[:limit]

    def _info_from_path(self, path: Path) -> SessionInfo | None:
        """Build metadata directly from a transcript path."""
        records = list(self._iter_path(path))
        if not records:
            return None
        return SessionInfo(
            session_id=path.stem,
            path=path,
            cwd=path.parent.name,
            created_at=records[0].timestamp,
            updated_at=records[-1].timestamp,
            message_count=len(records),
        )

    def fork(
        self,
        session_id: str,
        cwd: Path,
        *,
        up_to_uuid: str | None = None,
    ) -> str:
        """Copy a session into a new one, remapping every UUID.

        Copies records up to and including `up_to_uuid`, or the whole
        transcript when it is None. The parent chain is rewritten so the fork
        is internally consistent and can be extended independently.

        Returns the new session id.
        """
        records = self.read(session_id, cwd)
        new_id = str(uuid4())
        new_path = self.path_for(new_id, cwd)
        new_path.parent.mkdir(parents=True, exist_ok=True)

        remap: dict[str, str] = {}
        lines: list[str] = []
        for record in records:
            fresh = str(uuid4())
            remap[record.uuid] = fresh
            lines.append(
                SessionRecord(
                    uuid=fresh,
                    type=record.type,
                    payload=record.payload,
                    timestamp=record.timestamp,
                    parent_uuid=remap.get(record.parent_uuid or "", None),
                ).to_json()
            )
            if up_to_uuid is not None and record.uuid == up_to_uuid:
                break

        new_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return new_id

    def rename(self, session_id: str, title: str, cwd: Path) -> None:
        """Attach a human-readable title to a session."""
        path = self.path_for(session_id, cwd)
        if not path.exists():
            raise FileNotFoundError(f"No session {session_id} under {path.parent}")
        record = SessionRecord(
            uuid=str(uuid4()),
            type="title",
            payload={"title": title},
            parent_uuid=self._last_uuid(path),
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(record.to_json() + "\n")

    def delete(self, session_id: str, cwd: Path | None = None) -> bool:
        """Delete a session transcript. Returns True when one was removed."""
        path = self.path_for(session_id, cwd) if cwd else self.find(session_id)
        if path is None or not path.exists():
            return False
        path.unlink()
        return True


__all__ = [
    "SessionStore",
    "SessionRecord",
    "SessionInfo",
    "project_slug",
    "DEFAULT_SESSION_ROOT",
]
