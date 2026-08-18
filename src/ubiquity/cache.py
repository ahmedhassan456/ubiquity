"""Prompt cache break detection.

Providers cache the prefix of a request and charge a fraction of the normal
rate to re-read it. The cache is theirs: it cannot be inspected, warmed, or
addressed, and the only lever a client has is keeping the prefix identical from
one request to the next. That makes a break silent. Nothing fails, nothing
warns, and the only symptom is a bill that is several times larger than it
should be.

The detector here watches the one number providers do report. Before a request
the inputs that form the prefix are hashed; after it the cache-read count is
compared against the previous request, and a large drop is attributed to
whichever input changed. Only the hashing is provider-specific knowledge, and
pydantic-ai already normalizes the counts, so the whole mechanism is portable
to every backend that reports them.

Backends that report nothing hold the count at zero, which never triggers the
drop test. A provider without cache reporting is therefore silent rather than
wrong, which is the correct reading: no signal is not the same as no cache.
"""

from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from hashlib import blake2b
from typing import Any

logger = logging.getLogger("ubiquity")

MIN_BREAK_TOKENS = 2_000

BREAK_RATIO = 0.95

TTL_5M = 300.0

TTL_1H = 3600.0

MAX_TRACKED = 10


CACHE_POINT_PROVIDERS = frozenset({"bedrock"})


def wants_cache_point(model: Any) -> bool:
    """Whether this model needs a breakpoint placed in the message content.

    Bedrock is the one backend that caches on request but takes no model
    setting for it: the breakpoint has to travel as a marker inside the user
    content instead. A fallback model reports every backend it may reach, and
    any of them being Bedrock is enough, since a marker the others ignore
    costs nothing.
    """
    system = str(getattr(model, "system", "") or "")
    return any(
        part in CACHE_POINT_PROVIDERS
        for part in system.removeprefix("fallback:").split(",")
    )


def cacheable_prompt(prompt: str, model: Any, enabled: bool) -> Any:
    """Return the user prompt, with a cache breakpoint when one is needed.

    The marker goes after the text rather than before it, because a breakpoint
    has to attach to preceding content to mean anything. What it cuts is the
    system prompt, the tool definitions, and this prompt — the part of a run
    that does not change while the agent works through its turns.
    """
    if not enabled or not wants_cache_point(model):
        return prompt

    from pydantic_ai.messages import CachePoint

    return [prompt, CachePoint()]


def digest(value: Any) -> str:
    """Return a short stable hash of any JSON-serializable value.

    Sorting keys is what makes the hash a statement about content rather than
    about dictionary construction order, which is the difference between
    detecting a real change and reporting a spurious one.
    """
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return blake2b(encoded, digest_size=8).hexdigest()


@dataclass(slots=True)
class PromptSnapshot:
    """The cacheable inputs of one request, reduced to hashes.

    Tools are hashed individually as well as together. When the aggregate
    changes but the tool set does not, the per-tool hashes are what name the
    tool whose description moved. That is the common case and the hardest
    kind to spot by reading code, since nothing about the tool list looks
    different.
    """

    system: str
    tools: dict[str, str]
    model: str
    settings: str

    @property
    def tool_names(self) -> list[str]:
        """The tool names in the order they are sent."""
        return list(self.tools)

    @property
    def tools_digest(self) -> str:
        """A single hash covering every tool, including their order."""
        return digest([[name, h] for name, h in self.tools.items()])


def build_snapshot(
    system: str,
    tools: dict[str, Any],
    model: str,
    settings: dict[str, Any] | None,
) -> PromptSnapshot:
    """Reduce the cacheable parts of a request to a `PromptSnapshot`.

    `tools` maps a tool name to whatever describes it on the wire, normally its
    description and its JSON schema together.
    """
    return PromptSnapshot(
        system=digest(system),
        tools={name: digest(value) for name, value in tools.items()},
        model=model,
        settings=digest(settings or {}),
    )


@dataclass(slots=True)
class CacheBreak:
    """A detected drop in cache reads, with what is known about the cause."""

    reason: str
    previous_read: int
    current_read: int
    written: int
    call: int
    changed: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"prompt cache break: {self.reason} "
            f"[call #{self.call}, cache read {self.previous_read} -> "
            f"{self.current_read}, written {self.written}]"
        )


@dataclass(slots=True)
class _Tracked:
    """Everything remembered about one tracked conversation."""

    snapshot: PromptSnapshot
    calls: int = 1
    previous_read: int | None = None
    pending: list[str] = field(default_factory=list)
    last_request: float = field(default_factory=time.monotonic)


class CacheBreakDetector:
    """Watches cache-read counts and explains the drops.

    One instance tracks several conversations by key, so a run and the
    subagents it spawns can share a detector without their prefixes being
    compared against each other. The number of keys is capped because an agent
    id never recurs, and a long session spawning subagents would otherwise grow
    the map without bound.
    """

    def __init__(self, max_tracked: int = MAX_TRACKED) -> None:
        self._tracked: OrderedDict[str, _Tracked] = OrderedDict()
        self._max_tracked = max_tracked

    def record(self, key: str, snapshot: PromptSnapshot) -> None:
        """Note the prefix about to be sent, and what changed since last time.

        Nothing is reported here. A changed prefix is only interesting if the
        cache actually broke, and that is not known until the response comes
        back, so the findings are held until `check` can confirm them.
        """
        state = self._tracked.get(key)
        if state is None:
            while len(self._tracked) >= self._max_tracked:
                self._tracked.popitem(last=False)
            self._tracked[key] = _Tracked(snapshot=snapshot)
            return

        self._tracked.move_to_end(key)
        state.calls += 1
        state.pending = _differences(state.snapshot, snapshot)
        state.snapshot = snapshot

    def check(
        self, key: str, read_tokens: int, written_tokens: int = 0
    ) -> CacheBreak | None:
        """Compare this response's cache reads against the previous request.

        Returns None unless the reads fell by more than both thresholds. A
        proportional test alone would fire on every small conversation, and an
        absolute test alone would miss a large one, so a break has to clear
        both.
        """
        state = self._tracked.get(key)
        if state is None:
            return None

        previous = state.previous_read
        state.previous_read = read_tokens
        elapsed = time.monotonic() - state.last_request
        state.last_request = time.monotonic()
        changed = state.pending
        state.pending = []

        if previous is None:
            return None
        if read_tokens >= previous * BREAK_RATIO:
            return None
        if previous - read_tokens < MIN_BREAK_TOKENS:
            return None

        return CacheBreak(
            reason=_reason(changed, elapsed),
            previous_read=previous,
            current_read=read_tokens,
            written=written_tokens,
            call=state.calls,
            changed=changed,
        )

    def reset(self, key: str) -> None:
        """Forget the read baseline after a deliberate reduction.

        Compaction removes history on purpose, so the cache reads that follow
        it are expected to fall. Reporting that as a break would train the
        reader to ignore the warnings. Dropping the baseline is enough: the
        next response has nothing to be compared against and simply becomes
        the new baseline itself.
        """
        state = self._tracked.get(key)
        if state is not None:
            state.previous_read = None

    def forget(self, key: str) -> None:
        """Drop a conversation that has finished."""
        self._tracked.pop(key, None)


def _differences(before: PromptSnapshot, after: PromptSnapshot) -> list[str]:
    """Describe every cacheable input that changed between two snapshots."""
    changed: list[str] = []
    if before.model != after.model:
        changed.append(f"model changed ({before.model} -> {after.model})")
    if before.system != after.system:
        changed.append("system prompt changed")
    if before.settings != after.settings:
        changed.append("model settings changed")

    if before.tools_digest != after.tools_digest:
        added = [n for n in after.tools if n not in before.tools]
        removed = [n for n in before.tools if n not in after.tools]
        edited = [
            n
            for n, h in after.tools.items()
            if n in before.tools and before.tools[n] != h
        ]
        if added or removed:
            changed.append(
                f"tool set changed (+{len(added)}/-{len(removed)}: "
                f"{', '.join(added + ['-' + r for r in removed])})"
            )
        if edited:
            changed.append(f"tool schema changed ({', '.join(edited)})")
        if not added and not removed and not edited:
            changed.append("tool order changed")
    return changed


def _reason(changed: list[str], elapsed: float) -> str:
    """Explain a break, preferring an observed change to a guess.

    With nothing changed on this side, the elapsed time is the only remaining
    evidence, and it can only support a suspicion: the provider decides when to
    evict, and a short gap with an unchanged prefix usually means it evicted or
    routed the request elsewhere. Saying so is more useful than implying the
    caller has a bug to find.
    """
    if changed:
        return ", ".join(changed)
    if elapsed > TTL_1H:
        return "possible 1h TTL expiry (prefix unchanged)"
    if elapsed > TTL_5M:
        return "possible 5m TTL expiry (prefix unchanged)"
    return "likely provider-side eviction (prefix unchanged)"


__all__ = [
    "cacheable_prompt",
    "wants_cache_point",
    "CACHE_POINT_PROVIDERS",
    "CacheBreakDetector",
    "CacheBreak",
    "PromptSnapshot",
    "build_snapshot",
    "digest",
    "MIN_BREAK_TOKENS",
    "BREAK_RATIO",
    "MAX_TRACKED",
]
