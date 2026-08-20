"""The agent loop.

`summon()` is the entry point: it takes a prompt and `Options` and streams
`SDKMessage` objects until the run finishes. The terminal message is always
an `SDKResultMessage`, whether the run succeeded, exhausted its turns, or
raised.

The loop is built on `pydantic_ai.Agent.iter()`, which exposes the run as a
node graph. Tool execution, permissions, and hooks live in
`UbiquityToolset`; this module owns turn counting, message translation, and
the session lifecycle.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any
from uuid import uuid4

from pydantic_ai import Agent, CallToolsNode, ModelRequestNode
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)

from .cache import (
    CacheBreakDetector,
    PromptSnapshot,
    build_snapshot,
    cacheable_prompt,
)
from .compaction import MAX_CONSECUTIVE_FAILURES, compact, should_compact
from .hooks.registry import HookRegistry
from .hooks.types import HookInput
from .mcp.config import build_toolsets
from .mcp.gateway import gated
from .microcompact import microcompact
from .models import model_name_of, resolve_model, with_fallback
from .options import AgentDefinition, Options
from .pricing import CostMeter
from .agents import load_agents
from .memory import load_memory
from .prompts import build_system_prompt
from .sessions.replay import history_from
from .sessions.store import SessionStore
from .settings import apply_settings
from .skills import Skill, load_skills, select
from .subagents.agent_tool import AgentTool
from .todos import carried_over_context, key_for, store_for
from .tool import PermissionContext, Tool, ToolContext
from .toolset import UbiquityToolset, ToolDenied
from .tools import names_tool, resolve_tools, rule_names
from .tools.ask import AskUserQuestionTool
from .tools.skill import SkillTool
from .types import (
    SDKAssistantMessage,
    SDKCompactBoundaryMessage,
    SDKMessage,
    SDKMicrocompactMessage,
    SDKPartialAssistantMessage,
    SDKPermissionDenial,
    SDKResultMessage,
    SDKSystemMessage,
    SDKUserMessage,
)

logger = logging.getLogger("ubiquity")

_NOTIFICATION_REASONS = {
    "error_max_turns": "max_turns",
    "error_during_execution": "error",
}
"""The `reason` a run's terminal notification carries, keyed by result subtype.

Anything else that sets an error text falls back to ``stopped``. A `Stop` hook
is no longer one of them: blocking sends the agent back for another turn, so a
hook that never relents surfaces as `max_turns` rather than as its own reason.
"""


async def run_subagent(
    prompt: str,
    ctx: ToolContext,
    definition: AgentDefinition | None,
) -> tuple[str, list[SDKPermissionDenial]]:
    """Run a nested agent and return its final text and any denials.

    Only the final text crosses back to the parent, which is what keeps a
    subagent's intermediate work out of the parent's context. The subagent
    shares the parent's permission context so it cannot widen its own
    authority.

    Skills are inherited from the parent rather than reloaded, both to save the
    filesystem walk on every dispatch and so that a subagent cannot see a skill
    the parent run was not configured with. `AgentDefinition.skills` narrows
    that inherited set; it cannot add to it.

    `AskUserQuestion` is withheld along with `Agent`. A subagent's report goes
    to the parent rather than to the user, so a question it raised would open a
    dialog about work the user never asked for and cannot see. Anything a
    subagent cannot decide belongs in its report, for the parent to ask about.
    """
    options = ctx.options
    inherited = options.resolved_model()
    model = definition.model if definition and definition.model else inherited
    if model == "inherit":
        model = inherited

    tools = resolve_tools(
        options.tools,
        allowed=list(definition.tools) if definition and definition.tools else None,
        disallowed=list(definition.disallowed_tools) if definition else (),
    )
    tools = [t for t in tools if t.name not in ("Agent", "AskUserQuestion")]

    inherited_skills: dict[str, Skill] = ctx.extra.get("skills") or {}
    skills = select(inherited_skills, definition.skills if definition else None)
    tools = [t for t in tools if t.name != "Skill"]
    if skills:
        tools.append(SkillTool(skills))

    child_options = replace(
        options,
        model=model,
        max_turns=definition.max_turns if definition and definition.max_turns else 20,
        system_prompt=definition.prompt if definition else options.system_prompt,
    )
    child_ctx = replace(ctx, options=child_options)

    toolset = UbiquityToolset(tools, child_ctx, parent_tool_use_id=ctx.tool_use_id)
    resolved = (
        with_fallback(
            model,
            options.fallback_model,
            options.model_aliases,
            options.provider_settings(),
        )
        if options.fallback_model is not None
        else resolve_model(model, options.model_aliases, options.provider_settings())
    )
    agent = Agent(
        resolved,
        instructions=build_system_prompt(
            child_options, tools, skills, ctx.extra.get("memory"), subagent=True
        ),
        toolsets=[toolset],
        model_settings=child_options.model_settings(),
        retries=2,
    )

    meter = _meter_for(ctx)
    try:
        result = await agent.run(
            cacheable_prompt(prompt, resolved, bool(options.cache_prompt))
        )
        _charge_run(meter, result)
        return str(result.output), toolset.denials
    except Exception as exc:
        logger.exception("subagent failed")
        return f"Subagent failed: {type(exc).__name__}: {exc}", toolset.denials


def _charge_run(meter: CostMeter, result: Any) -> None:
    """Charge every model response of a completed nested run to `meter`.

    A nested run is driven by pydantic-ai rather than by the node loop in
    `summon`, so its responses are read back off the finished message history
    instead of being priced as they arrive. Each still carries the model that
    served it, which is what keeps a subagent on its own model from being
    charged at the parent's rates.
    """
    for message in result.all_messages():
        usage = getattr(message, "usage", None)
        if usage is None or getattr(message, "kind", None) != "response":
            continue
        meter.add(
            getattr(message, "model_name", None),
            getattr(message, "provider_name", None),
            usage,
        )


async def _snapshot(
    options: Options,
    tools: list[Any],
    ctx: ToolContext,
    model_label: str,
) -> PromptSnapshot:
    """Hash the cacheable parts of the request about to be sent.

    The tool descriptions are re-derived here rather than read back from the
    toolset, because a description is built per request and that is exactly the
    volatility worth catching: a tool whose prompt depends on mutable state
    changes the prefix on every turn without anything else looking different.
    """
    described = {
        tool.name: [await tool.prompt(ctx), tool.input_model.model_json_schema()]
        for tool in tools
    }
    return build_snapshot(
        build_system_prompt(
            options, tools, ctx.extra.get("skills"), ctx.extra.get("memory")
        ),
        described,
        model_label,
        options.model_settings(),
    )


def _meter_for(ctx: ToolContext) -> CostMeter:
    """Return the run's cost meter, creating and sharing it on first use.

    The meter lives in `ctx.extra` so that everything spending money on a run's
    behalf charges the same total. A subagent inherits the context and so
    inherits the meter by reference, which is what stops delegated work from
    being spent invisibly.
    """
    meter = ctx.extra.get("cost_meter")
    if not isinstance(meter, CostMeter):
        meter = CostMeter(
            dict(ctx.options.model_pricing), use_market=ctx.options.market_pricing
        )
        ctx.extra["cost_meter"] = meter
    return meter


USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "requests",
    "tool_calls",
)

DEFAULT_STOP_BLOCK = (
    "A Stop hook blocked the end of this turn but gave no reason. Review "
    "whether the work is genuinely complete before finishing again."
)


def _add_usage(totals: dict[str, Any], usage: Any) -> dict[str, Any]:
    """Fold one leg's usage into the run's running totals.

    A run vetoed by a `Stop` hook is driven by more than one `agent.iter`, and
    each of those reports only its own leg. Replacing the totals would bill the
    caller for the final continuation alone, which is a number that looks
    plausible and understates every run a hook extended.
    """
    return {
        field: totals.get(field, 0) + getattr(usage, field)
        for field in USAGE_FIELDS
    }


async def _stop_veto(
    hooks: HookRegistry,
    ctx: ToolContext,
    options: Options,
    session_id: str,
) -> str | None:
    """Fire the `Stop` hook and return the reason it blocked on, or None.

    Blocking means the agent is asked for another turn with that reason as its
    next prompt, which is the only reading that matches what a `Stop` hook is
    for: a check that the work is actually done. Firing it after the loop has
    already exited could do nothing but rewrite the result text, which turned a
    successful run into a reported failure and never gave the model the turn
    the veto was asking for.
    """
    outcome = await hooks.run(
        HookInput(
            hook_event_name="Stop",
            session_id=session_id,
            cwd=str(ctx.cwd),
            permission_mode=options.permission_mode,
        )
    )
    if outcome.decision != "block":
        return None
    return outcome.reason or DEFAULT_STOP_BLOCK


def _build_context(options: Options, session_id: str, hooks: HookRegistry) -> ToolContext:
    """Assemble the `ToolContext` shared by every tool in a run.

    `allowed_tools` and `disallowed_tools` are permission rules, so they are
    seeded into the engine's rule sets as well as being used to decide which
    tools exist. Without this a rule such as ``Bash(rm:*)`` would be silently
    inert: `resolve_tools` cannot act on it, because it is about one command
    rather than about the tool.

    Skill directories join the accessible ones. A skill's body points at the
    files bundled beside it, and those files usually live outside `cwd` -- in a
    home directory, or wherever the caller keeps them -- so without this the
    third loading step would be instructions to read a path the file tools
    refuse. The widening is exactly as broad as the roots the caller named.
    """
    return ToolContext(
        cwd=options.resolved_cwd(),
        options=options,
        permissions=PermissionContext(
            mode=options.permission_mode,
            allow_rules=set(options.allowed_tools or ()),
            deny_rules=set(options.disallowed_tools),
            ask_rules=set(options.ask_tools),
            additional_directories=options.resolved_add_dirs()
            | {r for r in options.resolved_skill_roots() if r.is_dir()},
            bypass_available=options.permission_mode == "bypassPermissions",
        ),
        session_id=session_id,
        hooks=hooks,
        abort=options.abort,
    )


def _rules_permit(tool: Tool[Any], options: Options) -> bool:
    """True when the permission rules leave room for a conditionally added tool.

    `Agent`, `Skill`, and `AskUserQuestion` are not in the built-in suite, so
    `resolve_tools` never sees them and cannot apply the allow and deny lists
    on their behalf. They are added afterwards, and this is the same filter
    applied by hand: a bare deny rule takes the tool away, and an
    `allowed_tools` list that never names it means the caller enumerated the
    run's tools without it.
    """
    denied_bare, _ = rule_names(options.disallowed_tools)
    if names_tool(tool, denied_bare):
        return False
    if options.allowed_tools is None:
        return True
    bare, scoped = rule_names(options.allowed_tools)
    return names_tool(tool, bare | scoped)


def _resume_target(options: Options, store: SessionStore) -> str | None:
    """Return the session id this run continues, or None for a fresh one.

    An explicit `resume` wins over `continue_conversation`, which picks the
    most recent session for the working directory. A named session that does
    not exist is an error rather than a silent new run: the caller asked to
    continue particular work, and starting over without it loses that work.
    """
    if options.resume:
        if store.info(options.resume, options.resolved_cwd()) is None:
            raise ValueError(f"No session {options.resume!r} to resume.")
        return options.resume
    if not options.continue_conversation:
        return None
    sessions = store.list(options.resolved_cwd(), limit=1)
    return sessions[0].session_id if sessions else None


def _resume(options: Options) -> tuple[str, list[Any]]:
    """Resolve the session id to write under and the history to replay.

    Forking is what makes it safe to branch from a session without spoiling
    it: the transcript is copied under a new id and the original is left at
    the point it was forked from.
    """
    fresh = str(uuid4())
    if not options.resume and not options.continue_conversation:
        return fresh, []

    store = SessionStore(options.session_dir)
    cwd = options.resolved_cwd()
    source = _resume_target(options, store)
    if source is None:
        return fresh, []

    history = history_from(store.read(source, cwd))
    if options.fork_session:
        return store.fork(source, cwd), history
    return source, history


def _stream_delta(event: Any) -> tuple[str, str] | None:
    """Extract the text and block kind from a streaming event, if it has any.

    Returns None for the events that carry no user-visible text, such as the
    start of a tool call.
    """
    from pydantic_ai.messages import (
        PartDeltaEvent,
        PartStartEvent,
        TextPartDelta,
        ThinkingPartDelta,
    )

    if isinstance(event, PartStartEvent):
        if isinstance(event.part, TextPart) and event.part.content:
            return event.part.content, "text"
        if isinstance(event.part, ThinkingPart) and event.part.content:
            return event.part.content, "thinking"
    elif isinstance(event, PartDeltaEvent):
        if isinstance(event.delta, TextPartDelta) and event.delta.content_delta:
            return event.delta.content_delta, "text"
        if isinstance(event.delta, ThinkingPartDelta) and event.delta.content_delta:
            return event.delta.content_delta, "thinking"
    return None


async def _stream_partials(
    node: Any, run: Any, session_id: str
) -> AsyncIterator[SDKPartialAssistantMessage]:
    """Yield the deltas of one model request as they arrive.

    Consuming the node's stream is also what executes it, so the graph
    advances exactly as it would have without streaming.
    """
    async with node.stream(run.ctx) as stream:
        async for event in stream:
            delta = _stream_delta(event)
            if delta is None:
                continue
            text, block_type = delta
            yield SDKPartialAssistantMessage(
                delta=text,
                block_type=block_type,  # type: ignore[arg-type]
                session_id=session_id,
            )


def _restore_todos(ctx: ToolContext) -> str | None:
    """Seed the run's todo list from the store and describe what came back.

    The list is put into the tool context so the first `TodoWrite` edits it
    rather than starting over, and returned as prose so the model knows it is
    there at all: a stored list the model never hears about is a list it will
    duplicate.
    """
    store = store_for(ctx.options)
    if store is None:
        return None
    restored = store.load(
        key_for(ctx.options, ctx.session_id, ctx.cwd), ctx.cwd
    )
    if not restored:
        return None
    ctx.extra["todos"] = [t.model_dump() for t in restored]
    return carried_over_context(restored)


def _response_blocks(response: ModelResponse) -> list[dict[str, Any]]:
    """Translate a pydantic-ai model response into SDK content blocks."""
    blocks: list[dict[str, Any]] = []
    for part in response.parts:
        if isinstance(part, TextPart):
            blocks.append({"type": "text", "text": part.content})
        elif isinstance(part, ThinkingPart):
            blocks.append({"type": "thinking", "thinking": part.content})
        elif isinstance(part, ToolCallPart):
            blocks.append(
                {
                    "type": "tool_use",
                    "id": part.tool_call_id,
                    "name": part.tool_name,
                    "input": part.args_as_dict(),
                }
            )
    return blocks


async def _maybe_compact(
    run: Any,
    options: Options,
    ctx: ToolContext,
    hooks: HookRegistry,
    session_id: str,
    model_name: str,
    failures: list[int],
    trigger: str = "auto",
) -> list[SDKMessage]:
    """Reclaim context in the live message history when it has grown too large.

    Two tiers, cheapest first. Microcompaction clears the content of stale
    tool results in place and costs nothing; only if that leaves the run still
    over the threshold does a full compaction summarize the history at the
    price of a model call. Returns the messages to emit for whatever happened,
    empty when nothing did.

    Failure of the second tier is deliberately non-fatal: a compaction that
    cannot run leaves the history untouched and the loop continues, because
    losing the run outright is worse than one oversized request that the
    provider may still accept.

    `failures` is a single-element counter of consecutive failed attempts,
    which trips a circuit breaker. Without it, a context that is irrecoverably
    over the limit makes every remaining turn attempt a compaction that cannot
    succeed, turning one stuck run into a stream of doomed API calls.
    """
    emitted: list[SDKMessage] = []
    if failures[0] >= MAX_CONSECUTIVE_FAILURES:
        return emitted

    history = run.ctx.state.message_history
    window = options.resolved_context_window(model_name)

    def measure(freed: int = 0) -> tuple[bool, int]:
        return should_compact(
            history,
            context_window=window,
            fraction=options.compact_threshold,
            output_reserve=options.max_tokens,
            freed=freed,
        )

    if trigger == "auto":
        needed, used = measure()
        if not needed:
            return emitted

        if options.auto_microcompact:
            reclaimed = microcompact(
                history, keep_recent=options.microcompact_keep_recent
            )
            if reclaimed is not None:
                history[:] = reclaimed.messages
                emitted.append(
                    SDKMicrocompactMessage(
                        tokens_saved=reclaimed.tokens_saved,
                        cleared=reclaimed.cleared,
                        kept=reclaimed.kept,
                        session_id=session_id,
                    )
                )
                needed, used = measure(freed=reclaimed.tokens_saved)
                if not needed:
                    return emitted
    else:
        used = 0

    pre = await hooks.run(
        HookInput(
            hook_event_name="PreCompact",
            session_id=session_id,
            cwd=str(ctx.cwd),
            permission_mode=options.permission_mode,
            extra={"trigger": trigger, "pre_tokens": used, "context_window": window},
        )
    )
    if pre.decision == "block":
        return emitted

    try:
        result = await compact(
            history,
            options.resolved_compact_model(),
            keep_recent=options.compact_keep_recent,
            instructions=options.compact_instructions,
            aliases=options.model_aliases,
            provider_kwargs=options.provider_settings(),
            trigger=trigger,
            meter=_meter_for(ctx),
        )
    except Exception:
        failures[0] += 1
        logger.exception(
            "compaction failed (%d/%d); continuing with the full history",
            failures[0],
            MAX_CONSECUTIVE_FAILURES,
        )
        return emitted

    if result is None:
        return emitted

    failures[0] = 0
    history[:] = result.messages

    await hooks.run(
        HookInput(
            hook_event_name="PostCompact",
            session_id=session_id,
            cwd=str(ctx.cwd),
            permission_mode=options.permission_mode,
            message=result.summary,
            extra={
                "trigger": trigger,
                "pre_tokens": result.pre_tokens,
                "dropped": result.dropped,
                "kept": result.kept,
            },
        )
    )

    emitted.append(
        SDKCompactBoundaryMessage(
            pre_tokens=result.pre_tokens,
            trigger=trigger,  # type: ignore[arg-type]
            session_id=session_id,
        )
    )
    return emitted


async def summon(
    prompt: str,
    options: Options | None = None,
) -> AsyncIterator[SDKMessage]:
    """Run the agent against `prompt`, streaming SDK messages until it finishes.

    The first message is always an `SDKSystemMessage` describing the resolved
    configuration, and the last is always an `SDKResultMessage`. Tool use, tool
    results, and assistant turns stream in between.

    Example:
        async for message in summon("list the python files", Options()):
            if message.type == "assistant":
                print(message.text)
    """
    options = apply_settings(options or Options())
    started = time.monotonic()
    try:
        session_id, history = _resume(options)
    except ValueError as exc:
        yield SDKResultMessage(
            subtype="error_during_execution",
            result=str(exc),
            session_id="",
            is_error=True,
            duration_ms=(time.monotonic() - started) * 1000,
        )
        return

    model_spec = options.resolved_model()
    model_label = model_name_of(model_spec, options.model_aliases)

    hooks = HookRegistry(options.hooks)
    ctx = _build_context(options, session_id, hooks)
    tools = resolve_tools(
        options.tools,
        allowed=options.allowed_tools,
        disallowed=options.disallowed_tools,
    )

    agents = load_agents(options.resolved_agent_roots())
    agents.update(options.agents)
    if agents and not any(t.name == "Agent" for t in tools):
        agent_tool = AgentTool(agents)
        if _rules_permit(agent_tool, options):
            tools.append(agent_tool)

    skills = load_skills(options.resolved_skill_roots())
    ctx.extra["skills"] = skills

    memory = load_memory(
        options.memory_sources, ctx.cwd, options.resolved_memory_files()
    )
    ctx.extra["memory"] = memory
    if skills and not any(t.name == "Skill" for t in tools):
        skill_tool = SkillTool(skills)
        if _rules_permit(skill_tool, options):
            tools.append(skill_tool)

    if options.can_use_tool is not None and not any(
        t.name == "AskUserQuestion" for t in tools
    ):
        ask_tool = AskUserQuestionTool()
        if _rules_permit(ask_tool, options):
            tools.append(ask_tool)

    store = SessionStore(options.session_dir) if options.persist_session else None
    pending: list[SDKMessage] = []

    def emit(message: SDKMessage) -> None:
        """Buffer a message from the toolset and persist it."""
        pending.append(message)
        if store is not None:
            store.append(session_id, ctx.cwd, message)

    toolset = UbiquityToolset(tools, ctx, emit=emit)
    mcp_toolsets = (
        gated(build_toolsets(options.mcp_servers), toolset)
        if options.mcp_servers
        else []
    )

    yield SDKSystemMessage(
        session_id=session_id,
        model=model_label,
        cwd=str(ctx.cwd),
        tools=[t.name for t in tools],
        permission_mode=options.permission_mode,
        agents=sorted(agents),
        mcp_servers=[
            {"name": name, "type": config.type}
            for name, config in options.mcp_servers.items()
        ],
    )

    start_hook = await hooks.run(
        HookInput(
            hook_event_name="SessionStart",
            session_id=session_id,
            cwd=str(ctx.cwd),
            permission_mode=options.permission_mode,
        )
    )

    user_hook = await hooks.run(
        HookInput(
            hook_event_name="UserPromptSubmit",
            session_id=session_id,
            cwd=str(ctx.cwd),
            permission_mode=options.permission_mode,
            prompt=prompt,
        )
    )
    if user_hook.decision == "block":
        yield SDKResultMessage(
            subtype="error_during_execution",
            result=user_hook.reason or "Prompt blocked by a UserPromptSubmit hook.",
            session_id=session_id,
            is_error=True,
            duration_ms=(time.monotonic() - started) * 1000,
        )
        return

    extra_context = "\n".join(
        c
        for c in (
            _restore_todos(ctx),
            start_hook.additional_context,
            user_hook.additional_context,
        )
        if c
    )
    full_prompt = f"{extra_context}\n\n{prompt}" if extra_context else prompt

    user_message = SDKUserMessage(content=full_prompt, session_id=session_id)
    if store is not None:
        store.append(session_id, ctx.cwd, user_message)
    yield user_message

    model = (
        with_fallback(
            model_spec,
            options.fallback_model,
            options.model_aliases,
            options.provider_settings(),
        )
        if options.fallback_model is not None
        else resolve_model(
            model_spec, options.model_aliases, options.provider_settings()
        )
    )
    agent = Agent(
        model,
        instructions=build_system_prompt(options, tools, skills, memory),
        toolsets=[toolset, *mcp_toolsets],
        model_settings=options.model_settings(),
        retries=2,
    )

    detector = CacheBreakDetector() if options.detect_cache_breaks else None
    meter = _meter_for(ctx)
    turns = 0
    failures = [0]
    subtype: str = "success"
    final_text = ""
    usage: dict[str, Any] = {}
    error: str | None = None

    next_prompt = full_prompt

    while True:
        finished = False
        try:
            cached = cacheable_prompt(next_prompt, model, bool(options.cache_prompt))
            async with agent.iter(cached, message_history=history or None) as run:
                async for node in run:
                    while pending:
                        yield pending.pop(0)

                    if options.abort is not None and options.abort.is_set():
                        subtype = "error_during_execution"
                        error = "Run aborted."
                        break

                    if isinstance(node, ModelRequestNode):
                        turns += 1
                        if turns > options.max_turns:
                            subtype = "error_max_turns"
                            error = f"Exceeded max_turns ({options.max_turns})."
                            break

                        if options.auto_compact:
                            for reclaimed in await _maybe_compact(
                                run, options, ctx, hooks, session_id, model_label, failures
                            ):
                                if detector is not None:
                                    detector.reset(session_id)
                                if store is not None:
                                    store.append(session_id, ctx.cwd, reclaimed)
                                yield reclaimed

                        if detector is not None:
                            detector.record(
                                session_id,
                                await _snapshot(options, tools, ctx, model_label),
                            )

                        if options.include_partial_messages:
                            async for partial in _stream_partials(node, run, session_id):
                                yield partial

                    elif isinstance(node, CallToolsNode):
                        meter.add(
                            node.model_response.model_name,
                            node.model_response.provider_name,
                            node.model_response.usage,
                        )
                        if detector is not None:
                            response_usage = node.model_response.usage
                            broke = detector.check(
                                session_id,
                                response_usage.cache_read_tokens,
                                response_usage.cache_write_tokens,
                            )
                            if broke is not None:
                                logger.warning("%s", broke)

                        assistant = SDKAssistantMessage(
                            content=_response_blocks(node.model_response),
                            session_id=session_id,
                            model=model_label,
                        )
                        if store is not None:
                            store.append(session_id, ctx.cwd, assistant)
                        yield assistant

                while pending:
                    yield pending.pop(0)

                if run.result is not None:
                    finished = True
                    final_text = str(run.result.output)
                    history = run.result.all_messages()
                    usage = _add_usage(usage, run.usage)

        except ToolDenied as denied:
            subtype = "error_during_execution"
            error = f"Run interrupted: {denied.message}"
        except Exception as exc:
            logger.exception("summon failed")
            subtype = "error_during_execution"
            error = f"{type(exc).__name__}: {exc}"

        while pending:
            yield pending.pop(0)

        if not finished:
            break

        blocked = await _stop_veto(hooks, ctx, options, session_id)
        if blocked is None:
            break
        if turns >= options.max_turns:
            subtype = "error_max_turns"
            error = f"Exceeded max_turns ({options.max_turns})."
            break

        next_prompt = blocked
        continuation = SDKUserMessage(content=blocked, session_id=session_id)
        if store is not None:
            store.append(session_id, ctx.cwd, continuation)
        yield continuation

    while pending:
        yield pending.pop(0)

    if error is not None:
        await hooks.notify(
            error,
            reason=_NOTIFICATION_REASONS.get(subtype, "stopped"),
            session_id=session_id,
            cwd=str(ctx.cwd),
            permission_mode=options.permission_mode,
        )

    await hooks.run(
        HookInput(
            hook_event_name="SessionEnd",
            session_id=session_id,
            cwd=str(ctx.cwd),
            permission_mode=options.permission_mode,
        )
    )

    cost = meter.dollars
    if cost is None:
        logger.debug("no total cost for this run: %s", meter.explain())

    result_message = SDKResultMessage(
        subtype=subtype,  # type: ignore[arg-type]
        result=error or final_text,
        session_id=session_id,
        duration_ms=(time.monotonic() - started) * 1000,
        num_turns=turns,
        total_cost_usd=cost,
        usage=usage,
        permission_denials=toolset.denials,
        is_error=error is not None,
    )
    if store is not None:
        store.append(session_id, ctx.cwd, result_message)
    yield result_message


__all__ = ["summon", "run_subagent"]
