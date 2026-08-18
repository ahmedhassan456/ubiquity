"""Model pricing: the rates, the arithmetic, and the run total."""

from __future__ import annotations

from decimal import Decimal

import pytest

from ubiquity.options import AgentDefinition, Options
from ubiquity.pricing import (
    CostMeter,
    ModelPricing,
    clear_pricing,
    cost_of,
    find_pricing,
    register_pricing,
)


@pytest.fixture(autouse=True)
def empty_registry():
    """Keep the process-wide price registry from leaking between tests."""
    clear_pricing()
    yield
    clear_pricing()


class Usage:
    """A stand-in for pydantic-ai's RequestUsage with inclusive buckets."""

    def __init__(self, input_tokens=0, output_tokens=0, cache_read=0, cache_write=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_tokens = cache_read
        self.cache_write_tokens = cache_write


class TestRates:
    def test_cache_rates_fall_back_to_the_input_rate(self):
        pricing = ModelPricing(input=3.0, output=15.0)
        assert pricing.read_rate == 3.0
        assert pricing.write_rate == 3.0

    def test_declared_cache_rates_are_used(self):
        pricing = ModelPricing(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75)
        assert pricing.read_rate == 0.3
        assert pricing.write_rate == 3.75


class TestCostOf:
    def test_plain_input_and_output(self):
        cost = cost_of(
            Usage(input_tokens=1_000_000, output_tokens=1_000_000),
            ModelPricing(input=3.0, output=15.0),
        )
        assert cost == Decimal(18)

    def test_cached_tokens_are_not_charged_twice(self):
        """`input_tokens` already contains the cached tokens, so it must net out.

        Charging the full input count and then the cache buckets on top is the
        one arithmetic error here that inflates rather than deflates the bill,
        and it grows with exactly the caching the SDK is trying to encourage.
        """
        cost = cost_of(
            Usage(input_tokens=10_000, cache_read=8_000),
            ModelPricing(input=10.0, cache_read=1.0),
        )
        assert cost == Decimal("0.028")

    def test_a_cache_write_surcharge_is_charged_on_the_write_bucket(self):
        cost = cost_of(
            Usage(input_tokens=10_000, cache_write=8_000),
            ModelPricing(input=10.0, cache_write=12.5),
        )
        assert cost == Decimal("0.12")

    def test_reads_and_writes_together_still_net_out_of_input(self):
        cost = cost_of(
            Usage(input_tokens=10_000, cache_read=6_000, cache_write=3_000),
            ModelPricing(input=10.0, cache_read=1.0, cache_write=12.5),
        )
        assert cost == Decimal("0.0535")

    def test_a_provider_overreporting_cache_never_yields_a_credit(self):
        """Fresh input floors at zero rather than going negative."""
        cost = cost_of(
            Usage(input_tokens=1_000, cache_read=5_000),
            ModelPricing(input=10.0, cache_read=1.0),
        )
        assert cost == Decimal("0.005")

    def test_nothing_used_costs_nothing(self):
        assert cost_of(Usage(), ModelPricing(input=3.0, output=15.0)) == Decimal(0)


class TestFindPricing:
    def test_the_longest_pattern_wins(self):
        table = {
            "gpt-5": ModelPricing(input=1.0),
            "gpt-5-mini": ModelPricing(input=0.25),
        }
        assert find_pricing("openai:gpt-5-mini", table).input == 0.25
        assert find_pricing("openai:gpt-5", table).input == 1.0

    def test_matching_ignores_case(self):
        table = {"Claude-Opus": ModelPricing(input=5.0)}
        assert find_pricing("anthropic:claude-opus-4-5", table).input == 5.0

    def test_an_unlisted_model_is_unpriced(self):
        assert find_pricing("mystery", {"gpt-5": ModelPricing()}) is None

    def test_the_registry_is_consulted_when_the_run_table_misses(self):
        register_pricing("llama", ModelPricing(input=0.6))
        assert find_pricing("groq:llama-3.3-70b", {}).input == 0.6

    def test_the_run_table_beats_the_registry(self):
        register_pricing("llama", ModelPricing(input=0.6))
        table = {"llama": ModelPricing(input=0.1)}
        assert find_pricing("groq:llama-3.3-70b", table).input == 0.1

    def test_a_run_match_wins_whole_rather_than_merging(self):
        """One entry is one complete statement about a model.

        Merging would let a registry entry supply an output rate under a run
        entry that deliberately omitted one, which silently reintroduces the
        figure the caller was overriding.
        """
        register_pricing("llama", ModelPricing(input=0.6, output=99.0))
        table = {"llama": ModelPricing(input=0.1)}
        assert find_pricing("groq:llama-3.3-70b", table).output == 0.0


class TestContextWindow:
    def test_a_price_entry_can_carry_the_window(self):
        options = Options(
            model="test", model_pricing={"gpt-5": ModelPricing(context_window=400_000)}
        )
        assert options.resolved_context_window("openai:gpt-5") == 400_000

    def test_max_context_tokens_still_wins(self):
        options = Options(
            model="test",
            max_context_tokens=50_000,
            model_pricing={"gpt-5": ModelPricing(context_window=400_000)},
        )
        assert options.resolved_context_window("openai:gpt-5") == 50_000

    def test_a_priced_model_without_a_window_falls_through(self):
        from ubiquity.compaction import DEFAULT_CONTEXT_WINDOW

        options = Options(model="test", model_pricing={"gpt-5": ModelPricing(input=1)})
        assert options.resolved_context_window("openai:gpt-5") == DEFAULT_CONTEXT_WINDOW

    def test_the_window_never_comes_from_the_market_snapshot(self):
        """An overstated window makes the provider reject the request outright.

        Prices may be estimated from published figures; a context window may
        not, because being wrong about it is a hard failure rather than a
        rounding error.
        """
        from ubiquity.compaction import DEFAULT_CONTEXT_WINDOW

        options = Options(model="test")
        window = options.resolved_context_window("anthropic:claude-opus-4-5")
        assert window == DEFAULT_CONTEXT_WINDOW


class TestCostMeter:
    def meter(self, **kwargs):
        return CostMeter(
            {"known": ModelPricing(input=10.0, output=20.0)},
            use_market=False,
            **kwargs,
        )

    def test_responses_accumulate(self):
        meter = self.meter()
        meter.add("known", None, Usage(input_tokens=1_000_000))
        meter.add("known", None, Usage(output_tokens=1_000_000))
        assert meter.dollars == pytest.approx(30.0)

    def test_a_run_with_no_responses_has_no_cost(self):
        assert self.meter().dollars is None

    def test_one_unpriced_response_makes_the_whole_total_unknown(self):
        """A partial sum in a cost field reads as a complete one and understates."""
        meter = self.meter()
        meter.add("known", None, Usage(input_tokens=1_000_000))
        meter.add("mystery", None, Usage(input_tokens=1_000_000))
        assert meter.dollars is None
        assert "mystery" in meter.explain()

    def test_a_response_without_usage_is_not_counted_as_unpriced(self):
        meter = self.meter()
        meter.add("known", None, Usage(input_tokens=1_000_000))
        meter.add("mystery", None, None)
        assert meter.dollars == pytest.approx(10.0)

    def test_an_unnamed_model_is_unpriced(self):
        meter = self.meter()
        meter.add(None, None, Usage(input_tokens=1_000))
        assert meter.dollars is None

    def test_two_models_are_each_charged_at_their_own_rates(self):
        """Pricing a run's aggregate usage once would charge both at one rate."""
        meter = CostMeter(
            {
                "cheap": ModelPricing(input=1.0),
                "dear": ModelPricing(input=100.0),
            },
            use_market=False,
        )
        meter.add("cheap", None, Usage(input_tokens=1_000_000))
        meter.add("dear", None, Usage(input_tokens=1_000_000))
        assert meter.dollars == pytest.approx(101.0)

    def real(self, **kwargs):
        """Real pydantic-ai usage, which is what the market lookup consumes."""
        from pydantic_ai.usage import RequestUsage

        return RequestUsage(**kwargs)

    def test_the_market_fallback_prices_a_known_model(self):
        meter = CostMeter({})
        meter.add("claude-opus-4-5", "anthropic", self.real(input_tokens=1_000_000))
        assert meter.dollars is not None and meter.dollars > 0

    def test_turning_the_market_off_leaves_it_unpriced(self):
        meter = CostMeter({}, use_market=False)
        meter.add("claude-opus-4-5", "anthropic", self.real(input_tokens=1_000_000))
        assert meter.dollars is None

    def test_a_declared_rate_beats_the_market(self):
        meter = CostMeter({"claude-opus-4-5": ModelPricing(input=1.0)})
        meter.add("claude-opus-4-5", "anthropic", self.real(input_tokens=1_000_000))
        assert meter.dollars == pytest.approx(1.0)

    def test_a_locally_hosted_model_stays_unpriced(self):
        """Nothing publishes a per-token rate for a model on localhost."""
        meter = CostMeter({})
        meter.add(
            "deepseek-v4-pro:cloud", "ollama", self.real(input_tokens=1_000_000)
        )
        assert meter.dollars is None

    def test_the_market_is_not_consulted_without_a_provider(self):
        """A bare model name is not enough to identify whose rates apply.

        genai-prices infers a provider from the name when none is given, which
        prices a self-hosted model at the hosted vendor's rates and invents a
        bill for a run that cost nothing.
        """
        meter = CostMeter({})
        meter.add("deepseek-v4-pro:cloud", None, self.real(input_tokens=1_000_000))
        assert meter.dollars is None

    def test_the_market_discounts_cache_reads(self):
        """The fallback must respect cache tiers or caching looks free of effect."""
        plain = CostMeter({})
        plain.add("claude-opus-4-5", "anthropic", self.real(input_tokens=100_000))
        cached = CostMeter({})
        cached.add(
            "claude-opus-4-5",
            "anthropic",
            self.real(input_tokens=100_000, cache_read_tokens=90_000),
        )
        assert cached.dollars < plain.dollars


class TestInTheLoop:
    """The meter reaches the run loop and fills in the result message."""

    def scripted(self, turns=1, input_tokens=1_000_000, output_tokens=0):
        """A model answering in one turn with a fixed usage report."""
        from pydantic_ai.messages import ModelResponse, TextPart
        from pydantic_ai.models.function import FunctionModel
        from pydantic_ai.usage import RequestUsage

        def respond(messages, info):
            return ModelResponse(
                parts=[TextPart(content="done")],
                usage=RequestUsage(
                    input_tokens=input_tokens, output_tokens=output_tokens
                ),
            )

        return FunctionModel(respond)

    async def run(self, tmp_path, **kwargs):
        from ubiquity import summon

        options = Options(model=self.scripted(), cwd=tmp_path, **kwargs)
        return [m async for m in summon("hi", options)]

    @pytest.mark.asyncio
    async def test_a_priced_run_reports_a_total(self, tmp_path):
        messages = await self.run(
            tmp_path, model_pricing={"function": ModelPricing(input=10.0)}
        )
        assert messages[-1].total_cost_usd == pytest.approx(10.0)

    @pytest.mark.asyncio
    async def test_an_unpriced_run_reports_none(self, tmp_path):
        messages = await self.run(tmp_path)
        assert messages[-1].total_cost_usd is None

    @pytest.mark.asyncio
    async def test_the_reason_for_none_is_logged(self, tmp_path, caplog):
        with caplog.at_level("DEBUG", logger="ubiquity"):
            await self.run(tmp_path)
        assert "unpriced models" in caplog.text

    @pytest.mark.asyncio
    async def test_market_pricing_can_be_turned_off_for_a_run(self, tmp_path):
        register_pricing("function", ModelPricing(input=10.0))
        messages = await self.run(tmp_path, market_pricing=False)
        assert messages[-1].total_cost_usd == pytest.approx(10.0)

    @pytest.mark.asyncio
    async def test_a_subagent_spends_from_the_same_total(self, tmp_path):
        """Delegated work is still the run's money.

        A subagent is a separate nested run, so its responses never pass
        through the parent's node loop. Left unwired it would spend without
        appearing in the figure the caller is shown.
        """
        from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
        from pydantic_ai.models.function import FunctionModel
        from pydantic_ai.usage import RequestUsage

        from ubiquity import summon

        calls = {"n": 0}

        def respond(messages, info):
            calls["n"] += 1
            usage = RequestUsage(input_tokens=1_000_000)
            if calls["n"] == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="Agent",
                            args={"description": "dig", "prompt": "look"},
                        )
                    ],
                    usage=usage,
                )
            return ModelResponse(parts=[TextPart(content="done")], usage=usage)

        options = Options(
            model=FunctionModel(respond),
            cwd=tmp_path,
            permission_mode="bypassPermissions",
            agents={
                "general-purpose": AgentDefinition(description="d", prompt="p")
            },
            model_pricing={"function": ModelPricing(input=1.0)},
        )
        messages = [m async for m in summon("go", options)]

        assert calls["n"] >= 3
        assert messages[-1].total_cost_usd == pytest.approx(calls["n"] * 1.0)

    @pytest.mark.asyncio
    async def test_compaction_is_charged_to_the_run(self, tmp_path):
        """Summarizing is a real request, often against a different model."""
        from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
        from pydantic_ai.models.function import FunctionModel
        from pydantic_ai.usage import RequestUsage

        from ubiquity import summon

        turns = {"n": 0}

        def respond(messages, info):
            turns["n"] += 1
            usage = RequestUsage(input_tokens=900)
            if turns["n"] < 3:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="Glob", args={"pattern": "*"})],
                    usage=usage,
                )
            return ModelResponse(parts=[TextPart(content="done")], usage=usage)

        def summarizer(messages, info):
            return ModelResponse(
                parts=[TextPart(content="COMPACTED")],
                usage=RequestUsage(input_tokens=1_000_000),
            )

        options = Options(
            model=FunctionModel(respond),
            compact_model=FunctionModel(summarizer),
            cwd=tmp_path,
            allowed_tools=["Glob"],
            permission_mode="bypassPermissions",
            max_context_tokens=1_000,
            compact_threshold=0.5,
            compact_keep_recent=2,
            model_pricing={"function": ModelPricing(input=1.0)},
        )
        messages = [m async for m in summon("go", options)]

        assert any(
            m.type == "system" and m.subtype == "compact_boundary" for m in messages
        )
        assert messages[-1].total_cost_usd > 1.0
