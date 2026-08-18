"""Hook dispatch.

Callbacks registered for the same event run in registration order. The first
callback to return a blocking decision wins and the rest are skipped: a hook
that vetoes an action must not have that veto overturned by a later hook.

Non-blocking fields accumulate across every callback that runs:
`additional_context` is concatenated and `updated_input` is applied in
sequence, so a later hook sees the rewrite performed by an earlier one.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from .types import HookEvent, HookInput, HookMatcher, HookOutput

logger = logging.getLogger("ubiquity.hooks")


class HookRegistry:
    """Holds the hook matchers for a run and dispatches events to them."""

    def __init__(self, matchers: Sequence[HookMatcher] = ()) -> None:
        self._by_event: dict[HookEvent, list[HookMatcher]] = {}
        for matcher in matchers:
            self._by_event.setdefault(matcher.event, []).append(matcher)

    def has(self, event: HookEvent) -> bool:
        """True when at least one matcher is registered for `event`."""
        return bool(self._by_event.get(event))

    def add(self, matcher: HookMatcher) -> None:
        """Register an additional matcher."""
        self._by_event.setdefault(matcher.event, []).append(matcher)

    async def run(self, payload: HookInput) -> HookOutput:
        """Dispatch `payload` to every matching hook and merge the results.

        Returns the merged output. A callback that raises is logged and
        skipped rather than failing the run, since a broken observability hook
        should not take down the agent. A callback that blocks short-circuits.
        """
        merged = HookOutput()
        contexts: list[str] = []
        messages: list[str] = []

        for matcher in self._by_event.get(payload.hook_event_name, []):
            if not matcher.matches(payload.tool_name):
                continue
            for callback in matcher.hooks:
                try:
                    result = await callback(payload)
                except Exception:
                    logger.exception(
                        "hook callback failed for %s", payload.hook_event_name
                    )
                    continue

                if result is None:
                    continue

                if result.additional_context:
                    contexts.append(result.additional_context)
                if result.system_message:
                    messages.append(result.system_message)
                if result.suppress_output:
                    merged.suppress_output = True

                if result.updated_input is not None:
                    merged.updated_input = result.updated_input
                    payload.tool_input = result.updated_input

                if result.decision == "block" or not result.continue_:
                    merged.decision = result.decision
                    merged.reason = result.reason
                    merged.continue_ = result.continue_
                    merged.additional_context = "\n".join(contexts) or None
                    merged.system_message = "\n".join(messages) or None
                    return merged

                if result.decision == "approve":
                    merged.decision = "approve"
                    merged.reason = result.reason

        merged.additional_context = "\n".join(contexts) or None
        merged.system_message = "\n".join(messages) or None
        return merged

    async def notify(
        self,
        message: str,
        *,
        reason: str,
        session_id: str,
        cwd: str,
        permission_mode: str = "default",
        tool_name: str | None = None,
        tool_input: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
    ) -> None:
        """Surface a host-facing notification.

        `reason` is a stable identifier for the kind of notification, carried
        in `extra` so a callback can route on it without parsing `message`.

        Nothing is returned: a notification reports something that has already
        happened, so there is no decision for a hook to influence and a
        `block` here would have nowhere to take effect.
        """
        if not self.has("Notification"):
            return
        await self.run(
            HookInput(
                hook_event_name="Notification",
                session_id=session_id,
                cwd=cwd,
                permission_mode=permission_mode,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_use_id=tool_use_id,
                message=message,
                extra={"reason": reason},
            )
        )


__all__ = ["HookRegistry"]
