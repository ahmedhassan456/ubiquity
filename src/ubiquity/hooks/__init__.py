"""Hook system: event payloads, matchers, and the dispatch registry."""

from .registry import HookRegistry
from .types import (
    HOOK_EVENTS,
    HookCallback,
    HookEvent,
    HookInput,
    HookMatcher,
    HookOutput,
)

__all__ = [
    "HookRegistry",
    "HookMatcher",
    "HookInput",
    "HookOutput",
    "HookCallback",
    "HookEvent",
    "HOOK_EVENTS",
]
