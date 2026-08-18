"""Model prices, and the cost of a run.

No provider returns a price with its response. Cost is always computed on the
client from token counts, and the rates that turn one into the other are
published per model, revised without notice, and different for cached tokens
than for fresh ones. This module holds the rates and does the arithmetic.

Rates are stated in US dollars per million tokens, which is how every provider
publishes them, so a price list can be transcribed rather than converted.

The registry ships empty, like `compaction.CONTEXT_WINDOWS` and for a weaker
version of the same reason: a table of prices for hundreds of models across
every provider cannot be kept true. Unlike a context window, though, a stale
price is only a wrong estimate rather than a failed request, so an optional
fallback to the `genai-prices` snapshot is worth having. Rates given by the
caller always win over it.

A model nobody has priced yields None rather than zero. Those are different
claims -- *unknown* and *free* -- and a run that quietly reported a locally
hosted model as costing nothing would be indistinguishable from one that
actually was free.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

logger = logging.getLogger("ubiquity")

PER_MILLION = Decimal(1_000_000)


@dataclass(slots=True)
class ModelPricing:
    """What one model costs, and how much context it accepts.

    `cache_read` and `cache_write` fall back to the plain input rate when left
    unset, pricing cached tokens as ordinary input. That overstates the bill on
    every provider that discounts reads, which is the direction an estimate
    should err in; assuming a discount that the provider does not actually give
    would understate it. Providers that surcharge cache writes are the reason
    `cache_write` is separate rather than sharing the read rate.

    `context_window` rides along because a model's window and its price are the
    two facts a caller looks up together and they change on the same event, a
    new model revision. It feeds `Options.resolved_context_window`, so unlike
    the rates it is never taken from the market fallback: an overstated window
    makes the provider reject a request outright, and that is a guess worth
    refusing to make on the caller's behalf.
    """

    input: float = 0.0
    output: float = 0.0
    cache_read: float | None = None
    cache_write: float | None = None
    context_window: int | None = None

    @property
    def read_rate(self) -> float:
        """The rate charged for a cached token that was read back."""
        return self.input if self.cache_read is None else self.cache_read

    @property
    def write_rate(self) -> float:
        """The rate charged for a token written into the cache."""
        return self.input if self.cache_write is None else self.cache_write


PRICING: dict[str, ModelPricing] = {}


def register_pricing(pattern: str, pricing: ModelPricing) -> None:
    """Declare the rates for models whose name contains `pattern`.

    Registering is the process-wide equivalent of `Options.model_pricing`, for
    callers who would otherwise repeat the same table on every run.
    """
    PRICING[pattern.lower()] = pricing


def clear_pricing() -> None:
    """Drop every registered price, restoring the empty default."""
    PRICING.clear()


def find_pricing(
    model_name: str, table: dict[str, ModelPricing] | None = None
) -> ModelPricing | None:
    """Return the rates to use for `model_name`, or None if it is unpriced.

    Matching is by substring with the longest pattern winning, mirroring
    `compaction.infer_context_window`, so ``gpt-5-mini`` can be priced apart
    from ``gpt-5`` without the general entry shadowing the specific one. An
    exact name is just the longest possible pattern and needs no special case.

    A run's own table is consulted before the process-wide registry, and a
    match in it wins outright rather than merging, so one entry is one
    complete statement about a model.
    """
    lowered = model_name.lower()
    for source in (table or {}, PRICING):
        matches = [key for key in source if key.lower() in lowered]
        if matches:
            return source[max(matches, key=len)]
    return None


def cost_of(usage: Any, pricing: ModelPricing) -> Decimal:
    """Return what one response cost, in dollars.

    pydantic-ai's token buckets are inclusive rather than disjoint:
    `input_tokens` already contains `cache_read_tokens` and
    `cache_write_tokens`, normalized across providers that report them
    separately on the wire. Cached tokens are subtracted back out before the
    input rate applies, or every cached token would be billed twice.

    Decimal rather than float because these are fractions of a cent summed
    across hundreds of requests, and float error compounds in exactly the place
    a reader is least able to notice it.
    """
    total_input = int(getattr(usage, "input_tokens", 0) or 0)
    read = int(getattr(usage, "cache_read_tokens", 0) or 0)
    written = int(getattr(usage, "cache_write_tokens", 0) or 0)
    output = int(getattr(usage, "output_tokens", 0) or 0)
    fresh = max(total_input - read - written, 0)

    charged = (
        Decimal(str(pricing.input)) * fresh
        + Decimal(str(pricing.read_rate)) * read
        + Decimal(str(pricing.write_rate)) * written
        + Decimal(str(pricing.output)) * output
    )
    return charged / PER_MILLION


def market_cost(usage: Any, model_name: str, provider: str | None) -> Decimal | None:
    """Price a response from the `genai-prices` snapshot, or None if unknown.

    genai-prices ships with pydantic-ai and consumes its usage objects
    directly, so this is a lookup rather than a second price table. It knows
    nothing about locally hosted or self-hosted models, which is the correct
    answer for them: those have no published per-token rate.

    The figures track the installed snapshot, not the caller's contract, so
    negotiated rates and committed-use discounts are invisible to it. That is
    why an explicit `ModelPricing` always wins.

    A response with no provider on it is left unpriced rather than matched on
    the model name alone. genai-prices will happily infer a provider from a
    bare name, which is wrong in the one case that matters: a locally hosted
    ``deepseek-v4-pro`` prices as the hosted DeepSeek service and bills a run
    that cost nothing. A guessed price is worse than no price, because it
    cannot be told apart from a real one.
    """
    if not provider:
        return None

    try:
        from genai_prices import calc_price
    except ImportError:
        return None

    try:
        return calc_price(usage, model_name, provider_id=provider).total_price
    except Exception as exc:
        logger.debug("no market price for %s on %s: %s", model_name, provider, exc)
        return None


@dataclass(slots=True)
class CostMeter:
    """Accumulates what a run spends, one model response at a time.

    Pricing the aggregate usage of a run in a single call would be simpler and
    wrong: one run can span several models. A fallback chain answers from
    whichever backend was reachable, compaction summarizes on `compact_model`,
    and a subagent may carry a model of its own. Each response is therefore
    priced against the model that actually served it.

    `use_market` decides whether an unpriced model falls back to the
    genai-prices snapshot. Turning it off makes the caller's table the only
    source, so an unpriced model is reported as unpriced instead of being
    estimated from figures the caller never supplied.
    """

    table: dict[str, ModelPricing] = field(default_factory=dict)
    use_market: bool = True
    total: Decimal = Decimal(0)
    priced: int = 0
    unpriced: set[str] = field(default_factory=set)

    def add(self, model_name: str | None, provider: str | None, usage: Any) -> None:
        """Charge one response to the running total.

        A response with no usage attached is skipped rather than counted as
        unpriced. It carries no tokens to charge for, so it neither adds to the
        bill nor makes the bill unknowable.
        """
        if usage is None:
            return
        if not model_name:
            self.unpriced.add("unknown")
            return

        pricing = find_pricing(model_name, self.table)
        if pricing is not None:
            self.total += cost_of(usage, pricing)
            self.priced += 1
            return

        market = market_cost(usage, model_name, provider) if self.use_market else None
        if market is None:
            self.unpriced.add(model_name)
            return
        self.total += market
        self.priced += 1

    @property
    def dollars(self) -> float | None:
        """The run's cost, or None when any part of it could not be priced.

        One unpriced response makes the whole total unknown. Returning the sum
        of the rest would look like a complete figure while understating the
        bill, and a cost that is quietly too low is worse than no cost at all.
        """
        if self.unpriced or not self.priced:
            return None
        return float(self.total)

    def explain(self) -> str:
        """Describe why the total is None, for a log line."""
        if not self.unpriced:
            return "no priced responses"
        return f"unpriced models: {', '.join(sorted(self.unpriced))}"


__all__ = [
    "ModelPricing",
    "CostMeter",
    "PRICING",
    "PER_MILLION",
    "register_pricing",
    "clear_pricing",
    "find_pricing",
    "cost_of",
    "market_cost",
]
