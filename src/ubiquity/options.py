"""Run configuration.

`Options` is the single argument bundle accepted by `summon()`. Every knob a
run has lives here, and the model field accepts any pydantic-ai model
identifier.

Model strings take the form ``provider:model``, for example ``openai:gpt-5``,
``google:gemini-3-pro``, ``mistral:mistral-large-latest``, or
``bedrock:meta.llama3-70b-instruct-v1:0``. A bare model name is resolved by
pydantic-ai's inference, an alias is expanded through the registry in
`models`, and a `pydantic_ai.models.Model` instance may be passed directly
when a provider needs custom client configuration.

There is no default model. Leaving `model` unset falls back to the
``UBIQUITY_MODEL`` environment variable, and a run with neither fails with an
explicit error rather than silently picking a vendor.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from .pricing import ModelPricing
from .retry import DEFAULT_MAX_RETRIES, DEFAULT_MAX_WAIT
from .todos import TodoScope
from .types import PermissionMode, PermissionResult

if TYPE_CHECKING:
    from pydantic_ai.models import Model

    from .hooks.types import HookMatcher
    from .mcp.config import McpServerConfig
    from .tool import Tool, ToolContext

CanUseTool: TypeAlias = Callable[
    [str, dict[str, Any], "ToolContext"], Awaitable[PermissionResult]
]

SettingSource: TypeAlias = Literal["user", "project", "local"]


@dataclass(slots=True)
class AgentDefinition:
    """A named subagent invocable through the Agent tool.

    `tools` restricts the subagent to a subset of the parent's tools; when
    None it inherits all of them. `model` accepts the same identifiers as
    `Options.model`, or the string ``inherit`` to reuse the parent's model.

    `skills` restricts which of the run's loaded skills this subagent sees,
    following the same convention as `tools`: None inherits all of them, and an
    empty sequence grants none. A subagent defined for one job is the case this
    is for -- the skills it is not meant to use are description lines it pays
    for on every request and can only be distracted by.
    """

    description: str
    prompt: str
    tools: Sequence[str] | None = None
    skills: Sequence[str] | None = None
    disallowed_tools: Sequence[str] = ()
    model: str | None = None
    permission_mode: PermissionMode | None = None
    max_turns: int | None = None


@dataclass(slots=True)
class Options:
    """Everything `summon()` needs to configure a run.

    `allowed_tools`, `disallowed_tools`, and `ask_tools` hold permission rules
    in rule form: a bare ``Bash`` names the whole tool, and
    ``Bash(git:*)`` names particular calls. Both forms are handed to the
    permission engine, and the bare form additionally decides which tools
    exist: start from the built-in suite unless `tools` is given explicitly,
    drop anything a bare `disallowed_tools` rule names, then keep only the
    tools `allowed_tools` mentions if it is set.

    A scoped rule never changes availability. ``disallowed_tools=["Bash(rm:*)"]``
    leaves `Bash` in place and blocks `rm` when it is called, which is the only
    reading under which the rule can mean anything.

    `env` is the environment given to subprocesses the `Bash` tool spawns. It is
    not this process's environment and does not reach the model provider, which
    takes its credentials from `api_key` and `provider_kwargs`.

    `model_pricing` maps a model name fragment to the rates it is billed at,
    which is what fills in `SDKResultMessage.total_cost_usd`. It also carries
    the model's context window, since that is looked up together with the price
    and revised on the same occasion. Set `market_pricing` to False to make this
    table the only source, rather than falling back to published figures for
    models it does not name.

    `max_retries` is the HTTP-level budget for a rate-limited or overloaded
    request, which is a different thing from pydantic-ai's `Agent(retries=...)`
    for tool and output validation. Set it to zero to send requests exactly as
    before, through no transport of this SDK's making. `retry_max_wait` caps
    the wait between attempts, including one a provider asks for by
    `Retry-After`; a provider asking for longer than the cap is reported rather
    than waited out.

    `skills` and `skill_sources` decide which skills a run can load. `skills`
    names directories outright -- either one skill directory or a directory of
    them -- and `skill_sources` opts into the conventional locations under
    ``.ubiquity`` the way `setting_sources` does for settings files. Neither is
    read by default: a skill is instructions, and instructions picked up from
    the filesystem without being asked for would make everything else in this
    object a suggestion. Explicit `skills` are loaded last and so win a name
    collision with a conventional one.

    `agents` and `agent_sources` decide which subagents the `Agent` tool
    offers. `agent_sources` discovers markdown definitions under
    ``.ubiquity/agents``, and `agents` names them in code and is applied last,
    so a definition written here overrides a discovered one of the same name.
    Discovery is opt-in for the same reason skills are: a definition decides
    what a delegated run is told to do.

    `memory` and `memory_sources` decide which `UBIQUITY.md` files become
    standing instructions in the system prompt, and they are opt-in for the
    same reason skills are. `memory_sources` reads the conventional locations
    -- ``~/.ubiquity/UBIQUITY.md``, then `UBIQUITY.md` down the directories
    above the cwd, then the uncommitted `UBIQUITY.local.md` -- and `memory`
    names files outright and is read last, so an explicit file outranks a
    discovered one.

    `can_use_tool` resolves a permission prompt, and it is also what makes
    `AskUserQuestion` available: a run with no handler has nobody to ask, so
    the tool is left out of the suite rather than offered and then denied.
    `permission_prompt_timeout_s` bounds how long a prompt may go unanswered.
    It is unset by default: a prompt waits until it is answered or the run is
    aborted, since a deadline nobody asked for would answer for the user. When
    it is set, expiry denies the call;
    proceeding on a silence would report an approval nobody gave.

    `abort` stops a run in progress. Setting the event refuses the next tool
    call and ends the loop, and `Bash` kills the process it is waiting on
    rather than letting it run out its timeout. It is the caller's event rather
    than one this SDK creates, because the whole point of it is to be settable
    from outside the `summon()` coroutine -- a signal handler, a cancel button,
    or a deadline the run itself knows nothing about.
    """

    model: str | Model | None = None
    fallback_model: str | Model | None = None
    model_aliases: dict[str, str] = field(default_factory=dict)
    api_key: str | None = field(default=None, repr=False)
    provider_kwargs: dict[str, Any] = field(default_factory=dict, repr=False)
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_max_wait: float = DEFAULT_MAX_WAIT

    system_prompt: str | None = None
    append_system_prompt: str | None = None

    cwd: Path | str = field(default_factory=Path.cwd)
    add_dirs: Sequence[Path | str] = ()

    permission_mode: PermissionMode = "default"
    allowed_tools: Sequence[str] | None = None
    disallowed_tools: Sequence[str] = ()
    ask_tools: Sequence[str] = ()
    can_use_tool: CanUseTool | None = None
    permission_prompt_timeout_s: float | None = None

    tools: Sequence[Tool[Any]] | None = None
    mcp_servers: dict[str, McpServerConfig] = field(default_factory=dict)
    agents: dict[str, AgentDefinition] = field(default_factory=dict)
    agent_sources: Sequence[SettingSource] = ()
    hooks: Sequence[HookMatcher] = ()

    skills: Sequence[Path | str] = ()
    skill_sources: Sequence[SettingSource] = ()

    memory: Sequence[Path | str] = ()
    memory_sources: Sequence[SettingSource] = ()

    max_turns: int = 40
    max_tokens: int | None = None
    max_thinking_tokens: int | None = None
    temperature: float | None = None

    cache_prompt: bool | Literal["5m", "1h"] = True
    detect_cache_breaks: bool = False

    model_pricing: dict[str, ModelPricing] = field(default_factory=dict)
    market_pricing: bool = True

    auto_compact: bool = True
    auto_microcompact: bool = True
    microcompact_keep_recent: int = 5
    max_context_tokens: int | None = None
    compact_threshold: float | None = None
    compact_keep_recent: int = 6
    compact_model: str | Model | None = None
    compact_instructions: str | None = None

    include_partial_messages: bool = False
    continue_conversation: bool = False
    resume: str | None = None
    fork_session: bool = False

    session_dir: Path | None = None
    persist_session: bool = True

    persist_todos: bool = True
    todo_dir: Path | None = None
    todo_scope: TodoScope = "project"

    setting_sources: Sequence[SettingSource] = ()
    env: dict[str, str] = field(default_factory=dict)
    user: str | None = None
    abort: asyncio.Event | None = None

    def resolved_model(self) -> str | Model:
        """Return the model spec for this run, falling back to the environment.

        Raises `ValueError` when no model is configured anywhere, since the
        alternative is guessing a provider on the user's behalf.
        """
        from .models import MODEL_ENV_VAR, default_model_spec

        if self.model is not None:
            return self.model
        if (from_env := default_model_spec()) is not None:
            return from_env
        raise ValueError(
            "No model configured. Pass Options(model=...) with a pydantic-ai "
            f"model identifier such as 'openai:gpt-5', or set {MODEL_ENV_VAR}."
        )

    def provider_settings(self) -> dict[str, Any] | None:
        """Return the keyword arguments to construct this run's provider with.

        Returns None when nothing is configured, so a run that sets neither
        field resolves its model exactly as before and providers read their
        credentials from the environment.

        `api_key` is named separately because it is the one argument every
        pydantic-ai provider accepts; anything else a provider needs goes in
        `provider_kwargs` verbatim and an explicit ``api_key`` there wins.
        Passing a keyword the named provider does not accept raises `TypeError`
        from the provider itself rather than being dropped, since a credential
        silently ignored fails later as a confusing authentication error.

        A retrying `http_client` is added when `max_retries` is positive, which
        is what makes a rate limit a pause rather than the end of the run. It
        is added here because this is the one place every provider in a run is
        configured from, so the retry policy cannot apply to the main model and
        quietly miss the fallback or the compaction model. A client the caller
        supplied in `provider_kwargs` is left alone, and providers that take no
        `http_client` drop it in `models.resolve_model`.

        These apply to every provider inferred from a model *string* in this
        run, including `fallback_model` and `compact_model`. A run whose models
        span providers should pass constructed `Model` instances instead, which
        are returned unchanged.
        """
        settings = dict(self.provider_kwargs)
        if self.api_key is not None:
            settings.setdefault("api_key", self.api_key)
        if self.max_retries > 0 and "http_client" not in settings:
            from .retry import retry_client

            settings["http_client"] = retry_client(
                self.max_retries, self.retry_max_wait
            )
        return settings or None

    def model_settings(self) -> dict[str, Any] | None:
        """Translate the generation limits into pydantic-ai model settings.

        Returns None when nothing is configured, so a run that sets none of
        these sends no settings at all rather than an empty object.

        `max_thinking_tokens` and `user` are emitted as provider-specific keys
        alongside the portable ones, because pydantic-ai's portable `thinking`
        setting is an effort level rather than a token budget and it has no
        portable notion of an end user. A provider ignores the keys that are
        not its own, so naming several is how one field reaches whichever
        backend the run happens to use.

        `cache_prompt` reaches only Anthropic, and that is the whole of prompt
        caching this layer can express. Most providers cache implicitly above a
        size threshold with nothing to enable, and Google caches through a
        separate resource that has to be created and paid for by the hour, so
        there is no switch that would mean the same thing everywhere.
        """
        settings: dict[str, Any] = {}
        if self.cache_prompt:
            settings["anthropic_cache"] = self.cache_prompt
        if self.temperature is not None:
            settings["temperature"] = self.temperature
        if self.max_tokens is not None:
            settings["max_tokens"] = self.max_tokens
        if self.max_thinking_tokens is not None:
            settings["thinking"] = True
            settings["anthropic_thinking"] = {
                "type": "enabled",
                "budget_tokens": self.max_thinking_tokens,
            }
            settings["google_thinking_config"] = {
                "thinking_budget": self.max_thinking_tokens
            }
        if self.user is not None:
            settings["openai_user"] = self.user
            settings["anthropic_metadata"] = {"user_id": self.user}
        return settings or None

    def resolved_compact_model(self) -> str | Model:
        """Return the model used for compaction summaries.

        Defaults to the run's own model. Pointing `compact_model` at a cheaper
        one is usually worthwhile, since summarizing is far easier than the
        work being summarized.
        """
        if self.compact_model is not None:
            return self.compact_model
        return self.resolved_model()

    def pricing_for(self, model_name: str) -> ModelPricing | None:
        """Return the declared rates for `model_name`, or None if undeclared.

        Consults this run's `model_pricing` before the process-wide registry,
        and never the market snapshot: this answers what the caller *said* a
        model costs, which is also what may carry a context window.
        """
        from .pricing import find_pricing

        return find_pricing(model_name, self.model_pricing)

    def resolved_context_window(self, model_name: str) -> int:
        """Return the context budget in tokens for `model_name`.

        Prefers `max_context_tokens`, then the window declared alongside the
        model's price, then the registry in `compaction`, then a conservative
        default. `max_context_tokens` wins because it is set for one run and a
        price entry describes a model in general, so the narrower statement is
        the more deliberate one.
        """
        from .compaction import infer_context_window

        if self.max_context_tokens is not None:
            return self.max_context_tokens
        priced = self.pricing_for(model_name)
        if priced is not None and priced.context_window is not None:
            return priced.context_window
        return infer_context_window(model_name)

    def resolved_cwd(self) -> Path:
        """Return `cwd` as an absolute resolved Path."""
        return Path(self.cwd).resolve()

    def resolved_add_dirs(self) -> set[Path]:
        """Return `add_dirs` as absolute resolved Paths."""
        return {Path(d).resolve() for d in self.add_dirs}

    def resolved_skill_roots(self) -> list[Path]:
        """Return the directories this run loads skills from, weakest first.

        The conventional directories named by `skill_sources` come first and
        the explicit `skills` last, so a caller who passes a directory outright
        overrides a skill of the same name that a project or home directory
        happened to supply.

        A leading ``~`` is expanded. Skills are as likely to be kept in a home
        directory as in the project, and `Path.resolve` alone would turn
        ``~/skills`` into a literal directory named ``~`` under the cwd.
        """
        from .settings import ORDER
        from .skills import skills_path

        cwd = self.resolved_cwd()
        named = {s for s in self.skill_sources}
        roots = [skills_path(s, cwd) for s in ORDER if s in named]  # type: ignore[arg-type]
        roots.extend(Path(d).expanduser().resolve() for d in self.skills)
        return roots

    def resolved_agent_roots(self) -> list[Path]:
        """Return the directories this run discovers agent files in, weakest first.

        Only the conventional ones. Agents named in `agents` are already
        definitions rather than paths, and they are merged in after these so
        that code wins over a file.
        """
        from .agents import agents_path
        from .settings import ORDER

        cwd = self.resolved_cwd()
        named = set(self.agent_sources)
        return [agents_path(s, cwd) for s in ORDER if s in named]  # type: ignore[arg-type]

    def resolved_memory_files(self) -> list[Path]:
        """Return the files named by `memory`, absolute and in order.

        Only the explicit ones. The conventional locations are derived from
        `memory_sources` at load time, since they depend on the directory the
        run is walking rather than on this object alone.

        A leading ``~`` is expanded for the same reason it is for skills: a
        `Path.resolve` alone would read ``~/notes.md`` as a directory named
        ``~`` under the cwd.
        """
        return [Path(f).expanduser().resolve() for f in self.memory]


__all__ = ["Options", "AgentDefinition", "CanUseTool", "SettingSource"]
