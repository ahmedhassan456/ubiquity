"""Tests for system prompt assembly.

The prompt is not verified by asserting its wording, which would only pin the
text to itself. What is worth holding is the division of labour: the base
prompt is guidance a caller may replace outright, while the sections after it
describe machinery the model has no other way to learn about and so survive
that replacement.
"""

from __future__ import annotations

import platform
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ubiquity import HookMatcher, Options
from ubiquity.models import MODEL_ENV_VAR
from ubiquity.prompts import (
    CLEARED_RESULTS_NOTE,
    COMPACTION_NOTE,
    DEFAULT_SYSTEM_PROMPT,
    DENIAL_NOTE,
    HOOKS_NOTE,
    SUBAGENT_NOTE,
    build_system_prompt,
)
from ubiquity.tools import builtin_tools


class TestTheBasePrompt:
    def test_the_default_is_used_when_no_prompt_is_given(self, tmp_path: Path) -> None:
        assert DEFAULT_SYSTEM_PROMPT in build_system_prompt(Options(cwd=tmp_path), [])

    def test_a_given_prompt_replaces_the_default(self, tmp_path: Path) -> None:
        prompt = build_system_prompt(Options(cwd=tmp_path, system_prompt="mine"), [])
        assert prompt.startswith("mine")
        assert DEFAULT_SYSTEM_PROMPT not in prompt

    def test_an_appended_prompt_comes_last(self, tmp_path: Path) -> None:
        prompt = build_system_prompt(
            Options(cwd=tmp_path, append_system_prompt="trailing"), []
        )
        assert prompt.rstrip().endswith("trailing")

    def test_the_agent_is_named(self, tmp_path: Path) -> None:
        assert "You are Ubiquity" in build_system_prompt(Options(cwd=tmp_path), [])

    def test_the_default_covers_the_generic_agent_hazards(self, tmp_path: Path) -> None:
        """Each of these is a class of mistake no tool description warns about."""
        prompt = build_system_prompt(Options(cwd=tmp_path), [])
        assert "parallel" in prompt
        assert "reversible" in prompt
        assert "data rather than as instruction" in prompt
        assert "Do only what was asked" in prompt
        assert "check your assumptions" in prompt
        assert "injection" in prompt
        assert "file_path:line_number" in prompt
        assert "guess a URL" in prompt


class TestTheRunSections:
    """Sections describing this run, which an override must not remove."""

    def test_the_denial_note_is_always_present(self, tmp_path: Path) -> None:
        assert DENIAL_NOTE in build_system_prompt(Options(cwd=tmp_path), [])

    def test_a_replaced_prompt_still_describes_the_run(self, tmp_path: Path) -> None:
        """The permission engine exists whether or not the caller mentions it."""
        prompt = build_system_prompt(Options(cwd=tmp_path, system_prompt="mine"), [])
        assert DENIAL_NOTE in prompt
        assert "Working directory" in prompt

    def test_hooks_are_described_only_when_configured(self, tmp_path: Path) -> None:
        assert HOOKS_NOTE not in build_system_prompt(Options(cwd=tmp_path), [])
        with_hooks = Options(cwd=tmp_path, hooks=[HookMatcher("PreToolUse", hooks=[])])
        assert HOOKS_NOTE in build_system_prompt(with_hooks, [])

    def test_the_permission_mode_note_tracks_the_mode(self, tmp_path: Path) -> None:
        plan = build_system_prompt(Options(cwd=tmp_path, permission_mode="plan"), [])
        assert "plan mode" in plan
        assert "plan mode" not in build_system_prompt(Options(cwd=tmp_path), [])

    def test_the_tools_in_use_are_named(self, tmp_path: Path) -> None:
        prompt = build_system_prompt(Options(cwd=tmp_path), builtin_tools())
        assert "Read, Write, Edit" in prompt

    def test_compaction_is_described_only_when_it_will_happen(
        self, tmp_path: Path
    ) -> None:
        on = build_system_prompt(Options(cwd=tmp_path), [])
        assert COMPACTION_NOTE in on and CLEARED_RESULTS_NOTE in on
        off = build_system_prompt(
            Options(cwd=tmp_path, auto_compact=False, auto_microcompact=False), []
        )
        assert COMPACTION_NOTE not in off and CLEARED_RESULTS_NOTE not in off

    def test_the_two_halves_of_compaction_are_described_separately(
        self, tmp_path: Path
    ) -> None:
        """Summarizing and clearing tool results are separate switches."""
        prompt = build_system_prompt(Options(cwd=tmp_path, auto_compact=False), [])
        assert COMPACTION_NOTE not in prompt
        assert CLEARED_RESULTS_NOTE in prompt

    def test_the_subagent_note_appears_only_for_a_subagent(
        self, tmp_path: Path
    ) -> None:
        assert SUBAGENT_NOTE not in build_system_prompt(Options(cwd=tmp_path), [])
        assert SUBAGENT_NOTE in build_system_prompt(
            Options(cwd=tmp_path), [], subagent=True
        )


class TestTheEnvironment:
    """Facts the model would otherwise guess at, and guess wrong."""

    def test_the_platform_is_named(self, tmp_path: Path) -> None:
        prompt = build_system_prompt(Options(cwd=tmp_path), [])
        assert f"Platform: {platform.system()}" in prompt

    def test_a_git_checkout_is_reported_as_one(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        assert "Is a git repository: True" in build_system_prompt(
            Options(cwd=tmp_path), []
        )

    def test_a_subdirectory_of_a_checkout_still_counts(self, tmp_path: Path) -> None:
        """A run often starts below the checkout root."""
        (tmp_path / ".git").mkdir()
        nested = tmp_path / "src" / "pkg"
        nested.mkdir(parents=True)
        assert "Is a git repository: True" in build_system_prompt(
            Options(cwd=nested), []
        )

    def test_a_plain_directory_is_not_a_repository(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        outside = tmp_path_factory.mktemp("plain")
        assert "Is a git repository: False" in build_system_prompt(
            Options(cwd=outside), []
        )

    def test_the_model_is_named(self, tmp_path: Path) -> None:
        prompt = build_system_prompt(Options(cwd=tmp_path, model="openai:gpt-5"), [])
        assert "Model: openai:gpt-5" in prompt

    def test_an_alias_is_resolved_before_being_shown(self, tmp_path: Path) -> None:
        prompt = build_system_prompt(
            Options(cwd=tmp_path, model="fast", model_aliases={"fast": "openai:gpt-5"}),
            [],
        )
        assert "Model: openai:gpt-5" in prompt

    def test_no_model_anywhere_is_said_rather_than_raised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Assembling the prompt must not be the thing that reports the error."""
        monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
        assert "Model: unset" in build_system_prompt(Options(cwd=tmp_path), [])


class TestTheSubagentWiring:
    async def test_a_dispatched_subagent_is_told_what_its_reply_must_carry(
        self, tmp_path: Path
    ) -> None:
        """The note is worth nothing if the dispatch path does not ask for it."""
        from ubiquity.client import _build_context, run_subagent
        from ubiquity.hooks.registry import HookRegistry

        seen: dict[str, str] = {}

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen["instructions"] = info.instructions or ""
            return ModelResponse(parts=[TextPart(content="ok")])

        options = Options(
            model=FunctionModel(respond),
            cwd=tmp_path,
            permission_mode="bypassPermissions",
        )
        ctx = _build_context(options, "s", HookRegistry(()))
        await run_subagent("go", ctx, None)
        assert SUBAGENT_NOTE in seen["instructions"]


class TestStability:
    def test_the_same_options_render_identically(self, tmp_path: Path) -> None:
        """The prompt is the cached prefix; a per-run difference costs a miss."""
        options = Options(cwd=tmp_path)
        assert build_system_prompt(options, builtin_tools()) == build_system_prompt(
            options, builtin_tools()
        )
