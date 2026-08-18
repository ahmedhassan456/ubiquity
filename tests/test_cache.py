"""Prompt caching: the model setting, prefix stability, and break detection."""

from __future__ import annotations

from typing import Any

import pytest

from ubiquity.cache import (
    BREAK_RATIO,
    MIN_BREAK_TOKENS,
    CacheBreakDetector,
    build_snapshot,
    cacheable_prompt,
    digest,
    wants_cache_point,
)
from ubiquity.options import AgentDefinition, Options
from ubiquity.subagents.agent_tool import AgentTool


def snap(system="system", tools=None, model="m", settings=None):
    """Build a snapshot with everything defaulted except what a test varies."""
    return build_snapshot(system, tools or {"Read": "read"}, model, settings)


def detector_at(read_tokens, key="s"):
    """Return a detector primed with one recorded call at `read_tokens`."""
    d = CacheBreakDetector()
    d.record(key, snap())
    d.check(key, read_tokens)
    return d


class TestCachePromptSetting:
    def test_caching_is_on_by_default(self):
        assert Options().model_settings()["anthropic_cache"] is True

    def test_a_ttl_passes_through(self):
        settings = Options(cache_prompt="1h").model_settings()
        assert settings["anthropic_cache"] == "1h"

    def test_turning_it_off_sends_nothing(self):
        assert Options(cache_prompt=False).model_settings() is None

    def test_it_does_not_disturb_the_other_settings(self):
        settings = Options(cache_prompt=False, temperature=0.5).model_settings()
        assert settings == {"temperature": 0.5}


class TestCachePoint:
    """Bedrock takes its breakpoint in the message, not in the settings."""

    def model(self, system):
        """A stand-in reporting `system` as its provider."""

        class Stub:
            pass

        stub = Stub()
        stub.system = system
        return stub

    def test_bedrock_is_recognised(self):
        assert wants_cache_point(self.model("bedrock"))

    def test_other_providers_are_not(self):
        assert not wants_cache_point(self.model("anthropic"))
        assert not wants_cache_point(self.model("openai"))

    def test_a_fallback_reaching_bedrock_counts(self):
        assert wants_cache_point(self.model("fallback:openai,bedrock"))

    def test_a_fallback_that_never_reaches_it_does_not(self):
        assert not wants_cache_point(self.model("fallback:openai,groq"))

    def test_a_model_without_a_provider_is_not_assumed_to_cache(self):
        assert not wants_cache_point(object())

    def test_a_breakpoint_is_appended_after_the_text(self):
        from pydantic_ai.messages import CachePoint

        content = cacheable_prompt("hi", self.model("bedrock"), True)
        assert content[0] == "hi" and isinstance(content[1], CachePoint)

    def test_nothing_is_added_when_caching_is_off(self):
        assert cacheable_prompt("hi", self.model("bedrock"), False) == "hi"

    def test_nothing_is_added_for_a_provider_that_needs_no_marker(self):
        assert cacheable_prompt("hi", self.model("anthropic"), True) == "hi"

    @pytest.mark.asyncio
    async def test_a_bedrock_run_sends_the_breakpoint(self, tmp_path):
        """The marker has to survive the whole way to the model, not just the helper."""
        from pydantic_ai.messages import (
            CachePoint,
            ModelResponse,
            TextPart,
            UserPromptPart,
        )
        from pydantic_ai.models.function import FunctionModel

        from ubiquity import summon

        class BedrockLike(FunctionModel):
            @property
            def system(self) -> str:
                return "bedrock"

        seen: list[Any] = []

        def respond(messages, info):
            for message in messages:
                for part in message.parts:
                    if isinstance(part, UserPromptPart):
                        seen.append(part.content)
            return ModelResponse(parts=[TextPart(content="done")])

        options = Options(model=BedrockLike(respond), cwd=tmp_path)
        async for _ in summon("hi", options):
            pass

        assert any(
            isinstance(content, list)
            and any(isinstance(block, CachePoint) for block in content)
            for content in seen
        )


class TestSubagentCachePoint:
    """A subagent is a separate run and needs its own breakpoint."""

    @pytest.mark.asyncio
    async def test_a_subagent_prompt_carries_the_breakpoint(self, tmp_path):
        from pydantic_ai.messages import (
            CachePoint,
            ModelResponse,
            TextPart,
            ToolCallPart,
            UserPromptPart,
        )
        from pydantic_ai.models.function import FunctionModel

        from ubiquity import summon

        class BedrockLike(FunctionModel):
            @property
            def system(self) -> str:
                return "bedrock"

        seen: list[Any] = []

        def respond(messages, info):
            for message in messages:
                for part in message.parts:
                    if isinstance(part, UserPromptPart):
                        seen.append(part.content)
            delegated = any(
                isinstance(c, list) and "investigate" in str(c[0]) for c in seen
            )
            if delegated:
                return ModelResponse(parts=[TextPart(content="report")])
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="Agent",
                        args={
                            "description": "look",
                            "prompt": "investigate",
                            "subagent_type": "helper",
                        },
                    )
                ]
            )

        options = Options(
            model=BedrockLike(respond),
            cwd=tmp_path,
            permission_mode="bypassPermissions",
            agents={"helper": AgentDefinition(description="d", prompt="p")},
        )
        async for _ in summon("go", options):
            pass

        delegated = [
            content
            for content in seen
            if isinstance(content, list) and "investigate" in str(content[0])
        ]
        assert delegated
        assert any(isinstance(block, CachePoint) for block in delegated[0])


class TestPrefixStability:
    @pytest.mark.asyncio
    async def test_the_agent_list_does_not_depend_on_dict_order(self, make_ctx):
        ctx = make_ctx()
        definition = AgentDefinition(description="d", prompt="p")
        forwards = AgentTool({"alpha": definition, "beta": definition})
        backwards = AgentTool({"beta": definition, "alpha": definition})
        assert await forwards.prompt(ctx) == await backwards.prompt(ctx)

    @pytest.mark.asyncio
    async def test_a_tool_description_is_stable_across_requests(self, make_ctx):
        ctx = make_ctx()
        tool = AgentTool({"helper": AgentDefinition(description="d", prompt="p")})
        first = await tool.prompt(ctx)
        ctx.extra["todos"] = [{"content": "something", "status": "pending"}]
        assert await tool.prompt(ctx) == first


class TestDigest:
    def test_key_order_does_not_change_the_hash(self):
        assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})

    def test_content_does(self):
        assert digest({"a": 1}) != digest({"a": 2})


class TestDetection:
    def test_the_first_call_reports_nothing(self):
        d = CacheBreakDetector()
        d.record("s", snap())
        assert d.check("s", 0) is None

    def test_a_steady_cache_reports_nothing(self):
        d = detector_at(10_000)
        d.record("s", snap())
        assert d.check("s", 10_000) is None

    def test_a_growing_cache_reports_nothing(self):
        d = detector_at(10_000)
        d.record("s", snap())
        assert d.check("s", 20_000) is None

    def test_a_large_drop_is_a_break(self):
        d = detector_at(50_000)
        d.record("s", snap())
        assert d.check("s", 0) is not None

    def test_a_small_proportional_drop_is_not(self):
        d = detector_at(1_000_000)
        d.record("s", snap())
        assert d.check("s", int(1_000_000 * BREAK_RATIO) + 1) is None

    def test_a_small_absolute_drop_is_not(self):
        d = detector_at(3_000)
        d.record("s", snap())
        assert d.check("s", 3_000 - MIN_BREAK_TOKENS + 1) is None

    def test_an_untracked_key_reports_nothing(self):
        assert CacheBreakDetector().check("nobody", 0) is None


class TestAttribution:
    def blame(self, changed_snapshot):
        """Return the reason a break is given when the prefix changes this way."""
        d = detector_at(50_000)
        d.record("s", changed_snapshot)
        broke = d.check("s", 0)
        assert broke is not None
        return broke.reason

    def test_a_changed_system_prompt_is_named(self):
        assert "system prompt" in self.blame(snap(system="different"))

    def test_a_changed_model_is_named_with_both_values(self):
        reason = self.blame(snap(model="other"))
        assert "m -> other" in reason

    def test_a_changed_setting_is_named(self):
        assert "model settings" in self.blame(snap(settings={"temperature": 1}))

    def test_an_added_tool_is_named(self):
        reason = self.blame(snap(tools={"Read": "read", "Write": "write"}))
        assert "tool set changed" in reason and "Write" in reason

    def test_a_rewritten_tool_description_is_named(self):
        reason = self.blame(snap(tools={"Read": "rewritten"}))
        assert "tool schema changed" in reason and "Read" in reason

    def test_reordered_tools_are_named_as_order(self):
        d = detector_at(50_000)
        d.record("s", snap(tools={"A": "a", "B": "b"}))
        d.check("s", 50_000)
        d.record("s", snap(tools={"B": "b", "A": "a"}))
        broke = d.check("s", 0)
        assert broke is not None and "tool order" in broke.reason

    def test_an_unchanged_prefix_blames_the_provider(self):
        assert "provider-side" in self.blame(snap())

    def test_every_change_is_reported_not_just_the_first(self):
        reason = self.blame(snap(system="different", model="other"))
        assert "system prompt" in reason and "model" in reason


class TestExpectedDrops:
    def test_compaction_does_not_look_like_a_break(self):
        d = detector_at(50_000)
        d.reset("s")
        d.record("s", snap())
        assert d.check("s", 0) is None

    def test_detection_resumes_after_a_reset(self):
        d = detector_at(50_000)
        d.reset("s")
        d.record("s", snap())
        d.check("s", 40_000)
        d.record("s", snap())
        assert d.check("s", 0) is not None


class TestIsolation:
    def test_two_conversations_do_not_break_each_other(self):
        d = CacheBreakDetector()
        for key in ("a", "b"):
            d.record(key, snap())
            d.check(key, 50_000)
        d.record("a", snap())
        d.check("a", 0)
        d.record("b", snap())
        assert d.check("b", 50_000) is None

    def test_the_tracked_set_is_capped(self):
        d = CacheBreakDetector(max_tracked=2)
        for key in ("a", "b", "c"):
            d.record(key, snap())
            d.check(key, 50_000)
        d.record("a", snap())
        assert d.check("a", 0) is None

    def test_a_finished_conversation_can_be_dropped(self):
        d = detector_at(50_000, key="agent-1")
        d.forget("agent-1")
        d.record("agent-1", snap())
        assert d.check("agent-1", 0) is None


class TestInTheLoop:
    """The detector reaches the run loop and reports through the logger."""

    def scripted(self, *reads):
        """A model taking two turns, reporting `reads` cache-read tokens each.

        The first turn calls a tool so the run makes a second model request,
        which is the only way a drop between requests can exist at all.
        """
        from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
        from pydantic_ai.models.function import FunctionModel
        from pydantic_ai.usage import RequestUsage

        turns = [
            [ToolCallPart(tool_name="TodoWrite", args={"add": [{"content": "go"}]})],
            [TextPart(content="done")],
        ]
        calls = {"n": 0}

        def respond(messages, info):
            index = min(calls["n"], len(turns) - 1)
            usage = RequestUsage(cache_read_tokens=reads[min(index, len(reads) - 1)])
            calls["n"] += 1
            return ModelResponse(parts=list(turns[index]), usage=usage)

        return FunctionModel(respond)

    async def run(self, tmp_path, *reads, **kwargs):
        from ubiquity import summon

        options = Options(model=self.scripted(*reads), cwd=tmp_path, **kwargs)
        return [m async for m in summon("hi", options)]

    @pytest.mark.asyncio
    async def test_a_drop_between_requests_is_reported(self, tmp_path, caplog):
        with caplog.at_level("WARNING", logger="ubiquity"):
            await self.run(tmp_path, 50_000, 0, detect_cache_breaks=True)
        assert "prompt cache break" in caplog.text

    @pytest.mark.asyncio
    async def test_a_steady_cache_reports_nothing(self, tmp_path, caplog):
        with caplog.at_level("WARNING", logger="ubiquity"):
            await self.run(tmp_path, 50_000, 50_000, detect_cache_breaks=True)
        assert "prompt cache break" not in caplog.text

    @pytest.mark.asyncio
    async def test_it_is_off_unless_asked(self, tmp_path, caplog):
        with caplog.at_level("WARNING", logger="ubiquity"):
            await self.run(tmp_path, 50_000, 0)
        assert "prompt cache break" not in caplog.text

    @pytest.mark.asyncio
    async def test_a_run_with_detection_on_still_completes(self, tmp_path):
        messages = await self.run(tmp_path, 50_000, 0, detect_cache_breaks=True)
        assert messages[-1].type == "result" and not messages[-1].is_error

    @pytest.mark.asyncio
    async def test_compaction_in_a_real_run_is_not_called_a_break(
        self, tmp_path, caplog
    ):
        """Compacting drops the cache reads on purpose, so it must stay quiet.

        Without the reset every compaction would report a break, which is the
        fastest way to make the warnings worth ignoring.
        """
        from pydantic_ai.messages import (
            ModelMessage,
            ModelResponse,
            TextPart,
            ToolCallPart,
        )
        from pydantic_ai.models.function import AgentInfo, FunctionModel
        from pydantic_ai.usage import RequestUsage

        from ubiquity import summon

        turns = {"n": 0}

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            turns["n"] += 1
            reads = 50_000 if turns["n"] < 3 else 0
            usage = RequestUsage(input_tokens=900, cache_read_tokens=reads)
            if turns["n"] < 4:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="Glob", args={"pattern": "*"})],
                    usage=usage,
                )
            return ModelResponse(parts=[TextPart(content="done")], usage=usage)

        options = Options(
            model=FunctionModel(respond),
            cwd=tmp_path,
            allowed_tools=["Glob"],
            permission_mode="bypassPermissions",
            max_context_tokens=1_000,
            compact_threshold=0.5,
            compact_keep_recent=2,
            detect_cache_breaks=True,
            compact_model=FunctionModel(
                lambda m, i: ModelResponse(parts=[TextPart(content="COMPACTED")])
            ),
        )

        with caplog.at_level("WARNING", logger="ubiquity"):
            messages = [m async for m in summon("go", options)]

        assert any(
            m.type == "system" and m.subtype == "compact_boundary" for m in messages
        )
        assert "prompt cache break" not in caplog.text
