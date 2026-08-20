<p align="center">
  <img src="https://raw.githubusercontent.com/ahmedhassan456/ubiquity/main/assets/icon.png" alt="Ubiquity" width="180">
</p>

<h1 align="center">Ubiquity</h1>

<p align="center">
  A coding-agent SDK for Python, built on
  <a href="https://ai.pydantic.dev">pydantic-ai</a>.
</p>

<p align="center">
  <a href="https://pypi.org/project/ubiquity/"><img src="https://img.shields.io/pypi/v/ubiquity.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/ubiquity/"><img src="https://img.shields.io/pypi/pyversions/ubiquity.svg" alt="Python versions"></a>
  <a href="https://pypi.org/project/ubiquity/"><img src="https://img.shields.io/pypi/dm/ubiquity.svg" alt="Downloads"></a>
  <a href="LICENSE"><img src="https://img.shields.io/pypi/l/ubiquity.svg" alt="MIT License"></a>
</p>

Everything a terminal coding agent needs — the agent loop, a built-in tool
suite, a rule-based permission system, hooks, subagents, MCP, and session
persistence — built against pydantic-ai's model layer, so the same agent runs on
**604 models across 22 providers** instead of one.

```python
import asyncio
from ubiquity import summon, Options

async def main():
    async for message in summon(
        "what Python files are in this project?",
        Options(model="openai:gpt-5"),
    ):
        if message.type == "assistant":
            print(message.text)

asyncio.run(main())
```

Swap the model string and nothing else changes:

```python
Options(model="google:gemini-3-pro")
Options(model="groq:llama-3.3-70b-versatile")
Options(model="mistral:mistral-large-latest")
Options(model="bedrock:meta.llama3-70b-instruct-v1:0")
```

There is no default model. Leaving `model` unset reads `UBIQUITY_MODEL`, and
a run with neither configured fails with an explicit error rather than
silently picking a vendor. Aliases let code name a role instead of a provider:

```python
from ubiquity import register_alias

register_alias("fast", "groq:llama-3.3-70b-versatile")
Options(model="fast")
```

`UBIQUITY_MODEL_ALIASES="fast=groq:llama-3.3-70b-versatile,big=openai:gpt-5"`
does the same from the environment.

For anything OpenAI-compatible that isn't a registered provider — Ollama, vLLM,
LM Studio, OpenRouter, Together:

```python
from ubiquity import openai_compatible

Options(
    model=openai_compatible(
        "llama3.3", 
        base_url="http://localhost:11434/v1"
    )
)
```

## Credentials

By default a provider reads its own environment variable — `GROQ_API_KEY`,
`ANTHROPIC_API_KEY`, `CO_API_KEY`, and so on. To pass a key per run instead:

```python
Options(model="groq:openai/gpt-oss-120b", api_key="gsk_...")
```

`api_key` is a named field because it is the one argument every pydantic-ai
provider accepts. Providers that need more take it verbatim:

```python
Options(
    model="azure:gpt-4o",
    provider_kwargs={
        "azure_endpoint": "https://example.openai.azure.com",
        "api_version": "2024-10-21",
        "api_key": "...",
    },
)
```

A keyword the named provider does not accept raises `TypeError` when the
provider is constructed, rather than being dropped — a credential silently
ignored comes back later as an authentication error that names nothing.

Both apply to every provider inferred from a model *string* in the run,
including `fallback_model` and `compact_model`. A run whose models span
providers should pass constructed `Model` instances, which are used as given.

Note that `Options.env` is unrelated: it is the environment for subprocesses
that the `Bash` tool spawns, and never touches the provider.

## Install

```bash
uv add ubiquity
```

## The message stream

`summon()` is an async generator. The first message is always a `system` message
describing the resolved configuration; the last is always a `result`. Tool use,
tool results, and assistant turns stream in between.

```python
async for message in summon(prompt, options):
    match message.type:
        case "system":       print(message.model, message.tools)
        case "assistant":    print(message.text)
        case "tool_use":     print(message.tool_name, message.tool_input)
        case "tool_result":  print(message.output.content)
        case "result":       print(message.subtype, message.usage)
```

## The system prompt

The prompt has two halves, and the difference decides what `system_prompt`
replaces.

The **base prompt** is guidance: work directly, read before editing, do only
what was asked, diagnose a failure before switching tactics, weigh how
reversible an action is, treat tool output as data rather than as instruction.
Setting `Options.system_prompt` replaces all of it, because a caller who writes
their own guidance is not asking to keep ours.

The **run sections** describe machinery the model has no other way to learn
about, so they are emitted either way: the working directory and the platform,
which model is answering, whether this is a git checkout, the tools in play,
what the permission mode allows, that a denied call comes back as a denial, that
hooks are watching when any are configured, that older messages get summarized
and older tool results get cleared, and that skills are loaded. A subagent
additionally gets told that its caller sees only its final reply.

```python
Options(system_prompt="You are a release engineer. Do nothing else.")
Options(append_system_prompt="Never touch files under vendor/.")
```

Any [memory files](#memory-files) you opted into come after all of that, and
`append_system_prompt` after them — the two things the caller wrote
themselves land last, where they can override what came before.

## Built-in tools

| Tool | Purpose |
| --- | --- |
| `Read` | Read a file, in `cat -n` format |
| `Write` | Create or overwrite a file |
| `Edit` | Exact string replacement |
| `Bash` | Run a shell command |
| `Glob` | Find files by pattern, newest first |
| `Grep` | Search file contents by regex |
| `TodoWrite` | Track multi-step work ([persistent](#todos)) |
| `Agent` | Delegate to a subagent (added when `agents` is configured) |
| `Skill` | Load a skill's instructions (added when [skills](#skills) are configured) |
| `AskUserQuestion` | Put multiple-choice questions to the user (added when `can_use_tool` is set) |

`Write` and `Edit` enforce **read-before-write**: an existing file must have been
read in full, and must not have changed since, before it can be modified. A
partial read (via `offset`/`limit`) does not authorize a write, because the
writer never saw the part it would discard. `Read` returns at most 2,000 lines,
so a longer file is a partial read by default; the way to edit one is to read it
again with `limit` set past its last line, which the tool's description says and
its error message repeats.

## Permissions

Five modes:

| Mode | Behavior |
| --- | --- |
| `default` | Prompt for anything not pre-approved |
| `acceptEdits` | Auto-accept file edits, prompt for the rest |
| `bypassPermissions` | Allow everything (deny rules still win) |
| `plan` | Read-only; no mutating tool may run |
| `dontAsk` | Never prompt; deny anything not pre-approved |

Rules are `Tool` or `Tool(matcher)`, in three forms:

```python
Options(
    allowed_tools=["Bash(git:*)", "Read"],
    disallowed_tools=["Bash(rm:*)"],
    ask_tools=["Bash(git push:*)"],
)
```

- `git:*` — prefix; matches `git` and anything starting `git `
- `git push *` — wildcard; `*` matches any run of characters
- `git status` — exact

A bare `Tool` rule also decides availability: `disallowed_tools=["Bash"]`
removes the tool, and setting `allowed_tools` limits the run to the tools it
names. A scoped `Tool(matcher)` rule never does — `Bash(rm:*)` leaves `Bash`
exposed and blocks `rm` at the point of the call.

Two properties are load-bearing and covered by tests:

**Deny beats everything**, including `bypassPermissions`. So do user-configured
`ask` rules and safety checks on sensitive paths (`.env`, `.ssh/`, `.git/`).

**Allow requires full coverage.** A tool may present several candidates for one
call — `Bash` returns each segment of a compound command — and every one must be
matched. This is what stops `Bash(git:*)` from authorizing
`git status && rm -rf /`. Deny and ask fire on any single segment.

To prompt a human, supply `can_use_tool`:

```python
from ubiquity import PermissionResultAllow, PermissionResultDeny

async def ask_user(tool_name, tool_input, ctx):
    if input(f"Run {tool_name}? [y/N] ").lower() == "y":
        return PermissionResultAllow()
    return PermissionResultDeny(message="User declined.")

Options(can_use_tool=ask_user)
```

Without a `can_use_tool` handler, anything that would prompt is denied rather
than hanging.

### Asking the user a question

`AskUserQuestion` lets the model put one to four multiple-choice questions in
front of the user and continue with the answers. It has no waiting machinery of
its own: it returns `ask`, and the `can_use_tool` handler that resolves the
prompt is what renders the questions and hands the answers back.

```python
async def ask_user(tool_name, tool_input, ctx):
    if tool_name == "AskUserQuestion":
        answers = {}
        for question in tool_input["questions"]:
            print(question["question"])
            for option in question["options"]:
                print(f"  {option['label']}: {option['description']}")
            answers[question["question"]] = input("> ")
        return PermissionResultAllow(updated_input={**tool_input, "answers": answers})
    ...
```

Because the prompt *is* the tool, `bypassPermissions` and a bare
`allowed_tools=["AskUserQuestion"]` do not skip it, and the model cannot send
`answers` itself. A handler that allows the call without collecting anything
gets an error result rather than a silent empty form, so the model knows nobody
was actually asked.

A question nobody answers is a permission prompt nobody answers, so it is
resolved by the things that resolve one:

- the tool is not offered at all when there is no `can_use_tool` handler, and a
  subagent never gets it — its report goes to the parent, not to the user
- `dontAsk` mode declines it without prompting
- setting `Options.abort` ends a pending prompt and interrupts the run
- `Options.permission_prompt_timeout_s` bounds the wait when you want a bound.
  It is unset by default, because a deadline nobody asked for answers for the
  user. When it is set, expiry **denies**: a call that ran because nobody
  objected in time was never approved, and an invented answer is worse than an
  unanswered question.

## Hooks

Fourteen events, dispatched in registration order. The first hook to block wins
and the rest are skipped; a hook that raises is logged and skipped rather than
failing the run.

```python
from ubiquity import HookMatcher, HookOutput

async def block_secrets(payload):
    if ".env" in str(payload.tool_input):
        return HookOutput(decision="block", reason="Refusing to touch .env")
    return None

Options(hooks=[HookMatcher("PreToolUse", [block_secrets], matcher="Write|Edit")])
```

`PreToolUse` may rewrite the tool input via `updated_input`; later hooks in the
same chain see the rewrite. `UserPromptSubmit` and `SessionStart` may inject
context via `additional_context`.

`Notification` is informational rather than a gate: it fires when a call is
waiting on approval and when a run ends by exhausting its turns or by raising.
`payload.extra["reason"]` distinguishes them (`permission_required`,
`max_turns`, `error`, `stopped`).

A `Stop` hook that returns `decision="block"` sends the agent back for another
turn, with `reason` as its next prompt and the run's history intact. That is
how a hook enforces "the tests must pass before you stop": it blocks until it
is satisfied. Blocking does not fail the run — a continued run that finishes
still reports `success`. A hook that never relents is bounded by `max_turns`,
which the run then reports.

## Subagents

A subagent is a nested run with its own history, tool subset, and turn budget.
Only its final text returns to the parent, which is the point — the parent's
context stays clean.

```python
from ubiquity import AgentDefinition

Options(
    agents={
        "reviewer": AgentDefinition(
            description="Reviews code for correctness",
            prompt="You review diffs and report defects.",
            tools=["Read", "Glob", "Grep"],
            model="anthropic:claude-haiku-4-5-20251001",
        )
    }
)
```

Isolation is deliberate and partial: a subagent gets fresh file-read
bookkeeping, but **shares the parent's permission context**, because a subagent
that could widen its own permissions would be an escalation path. Subagents
cannot spawn further subagents, and nesting is capped.

### Subagents as files

A subagent is mostly a prompt, and a prompt is the part of a program most worth
editing without editing the program. So the same definition can be written as a
markdown file under `.ubiquity/agents`, with the frontmatter carrying the
fields and the body carrying the prompt:

```markdown
---
name: reviewer
description: Reviews code for correctness
tools: Read, Glob, Grep
model: inherit
---

You review diffs and report defects.
```

```python
Options(agent_sources=["user", "project", "local"])
```

| source | directory |
| --- | --- |
| `user` | `~/.ubiquity/agents/` |
| `project` | `<cwd>/.ubiquity/agents/` |
| `local` | `<cwd>/.ubiquity/agents.local/` |

Nothing is discovered unless you ask for it, the same as with skills and
settings — a definition picked up off the filesystem decides what a delegated
run is told to do. Later sources win a name collision, and `Options.agents` is
merged in last, so a definition written in code overrides a discovered one
rather than colliding with it.

`name` defaults to the file stem. `tools`, `skills`, and `disallowed-tools`
take `a, b` or `[a, b]`, and keys can be hyphenated or camelCase
(`max-turns` and `maxTurns` both land on `max_turns`). Files may be grouped in
subdirectories, and discovery is sorted rather than left to the filesystem —
the definitions are listed in the `Agent` tool's description, which sits in the
cached prefix.

An **omitted** list inherits everything and an **empty** one grants nothing
(`tools:` with nothing after it), the same distinction `AgentDefinition` makes
in code. A file with no description or no body is skipped with a warning rather
than raising, and a field that cannot be parsed — a non-numeric `max-turns`, an
unknown `permission-mode` — is dropped so the run's own setting applies.
Guessing a value would invent one nobody chose.

## Skills

A skill is a directory holding a `SKILL.md` that says what it is for and how to
do it. Anything else in the directory — checklists, scripts, reference tables —
is a bundled file the body can point at.

```
skills/
  release/
    SKILL.md
    checklist.md
```

```markdown
---
name: release
description: Cut a release. Use when asked to publish, tag, or bump a version.
---

Follow `checklist.md` in this directory, in order. Do not skip the changelog.
```

Loading happens in three steps, each paid for only when it earns its place:

1. Every skill's **name and description** sit in the `Skill` tool's description
   — a line or two each.
2. That **tool** returns one skill's full body, when the model decides the task
   matches.
3. The body points at **bundled files**, which the model reads with `Read` or
   `Bash`.

The listing appears in exactly one place. Tool descriptions and the system
prompt are both part of the cached prefix, so a listing repeated in both would
be paid for twice on every request; the system prompt gets one sentence saying
skills exist and pointing at the tool. The listing is also budgeted: each
description is clamped to 250 characters, and a listing that still exceeds
8,000 characters degrades to names alone. A description long enough to explain
the whole procedure buys nothing, since invoking the skill supplies the rest.

That is the reason a skill is not simply appended to the system prompt. A useful
procedure runs to hundreds of lines, and a run carrying a dozen of them pays for
all twelve on every request while using at most one.

Nothing is discovered unless you ask for it:

```python
Options(
    skills=["./skills", "~/my-skills/release"],
    skill_sources=["project"],
)
```

A root is either a directory of skills or a single skill directory — both are
natural things to point at, and telling them apart costs one `exists` check.

`skill_sources` takes the same `user` / `project` / `local` vocabulary as
[settings files](#settings-files), resolving to `~/.ubiquity/skills`,
`<cwd>/.ubiquity/skills`, and `<cwd>/.ubiquity/skills.local`. Explicit `skills`
are loaded last, so they override a name that came from a conventional
directory. Skill roots also become readable to the file tools — otherwise step
three would be instructions to open a path `Read` refuses.

The `Skill` tool is added only when skills were actually loaded, and it honors
`allowed_tools` and `disallowed_tools` like any other tool. Subagents inherit the
run's skills and can be narrowed to a subset, but never widened:

```python
AgentDefinition(
    description="Cuts releases",
    prompt="You run the release process.",
    skills=["release"],
)
```

`skills=None` inherits all of them and `skills=[]` grants none, the same way
`tools` works.

Only `name` and `description` are read from the frontmatter, and `name` defaults
to the directory name. Other keys are parsed and kept on the `Skill` object for
you to inspect, but nothing acts on them — in particular there is no per-skill
tool gating, so a frontmatter `allowed-tools` restricts nothing here. A skill
missing a description is skipped with a warning rather than raising, since one
malformed file should not take down a run the other skills would have served.

## MCP

```python
from ubiquity import parse_config

Options(
    mcp_servers={
        "github": parse_config(
            {
                "command": "npx", 
                "args": [
                    "-y", 
                    "@modelcontextprotocol/server-github"
                ]
            }
        ),
        "docs": parse_config(
            {
                "url": "https://example.test/mcp"
            }
        ),
    }
)
```

Stdio, SSE, and streamable HTTP are supported. Tools arrive namespaced as
`mcp__<server>__<tool>`, so they cannot shadow a built-in and a whole server can
be targeted with `mcp__github__*`.

```python
Options(disallowed_tools=["mcp__github__*"])
```

MCP calls go through the same pipeline as a built-in tool — permission rules,
`PreToolUse` and `PostToolUse` hooks, and `tool_use` / `tool_result` messages in
the stream. A remote tool is the last thing that should run unobserved.

A server's tools are treated as able to mutate unless it sends a `readOnlyHint`
annotation, so plan mode blocks them by default rather than trusting a server
that says nothing about itself.

## Compaction

A long run eventually outgrows its context window. Two tiers reclaim it,
cheapest first.

**Microcompaction** costs nothing. The content of older tool results — the file
read forty turns ago, the command whose output has long since been acted on —
is replaced in place with a marker. No model call, no summary, and the
transcript keeps its shape. A `microcompact` message reports what was cleared.

Only tools whose results are pure observation are eligible (`Read`, `Write`,
`Edit`, `Bash`, `Glob`, `Grep`). A tool carrying state
the model is expected to still be tracking — `TodoWrite`, `Agent`, anything
from MCP — is left alone, because clearing it silently rewrites what the model
believes about the task.

**Full compaction** runs only if that leaves the run still over the threshold.
The older part of the history is replaced by a model-written summary and the
run continues, marked by a `compact_boundary` message.

```python
Options(
    auto_microcompact=True,
    microcompact_keep_recent=5,
    auto_compact=True,
    max_context_tokens=200_000,
    compact_keep_recent=6,
    compact_model="groq:llama-3.3-70b-versatile",
)
```

Three details of the second tier are load-bearing.

**The cut lands before a model response, never before a request.** A request
carries the tool results answering the calls in the response above it, so
cutting between them would leave results with no matching call — which most
providers reject outright. Cutting before a response keeps every pair whole.

**The trigger is a token reserve, not a percentage.** The threshold is the
window minus room for the summary being generated, minus headroom for the next
request. A flat 80% would waste 200k tokens on a million-token window and
leave a small local model no room to write the summary at all. `max_tokens`,
when set, caps the summary reserve — reserving 20k from a model that can only
emit 4k gives back 16k of usable context on every turn.

**Pressure is measured past the last usage record.** The provider's own
accounting is preferred, but the check runs between a response and the request
answering it, so the tool results that just landed are never in that count.
They are estimated and added on, because a single large file read is precisely
the event that pushes a run over the limit. Anything microcompaction just
reclaimed is subtracted back out for the same reason in reverse: usage reports
what was sent, not what will be sent next.

**Repeated failures trip a circuit breaker.** A context that is irrecoverably
over the limit would otherwise attempt a doomed compaction on every remaining
turn; after three consecutive failures the loop stops trying. A failed
compaction is never fatal on its own — the history is left alone and the run
continues.

There is no built-in table of per-model context windows, because a table
asserting sizes for hundreds of models across every provider cannot be kept
true, and a stale entry that overstates a window causes exactly the failure
compaction exists to prevent. The default is one conservative number; declare
the models you actually use:

```python
from ubiquity import register_context_window

register_context_window("gemini-3", 1_048_576)
```

`Options.max_context_tokens` overrides per run, and
`UBIQUITY_MAX_CONTEXT_TOKENS` overrides the default globally.

`PreCompact` can veto a compaction and `PostCompact` receives the summary.

## Prompt caching

Providers cache the prefix of a request and re-read it at a fraction of the
normal rate. The cache is theirs — it cannot be inspected, warmed, or addressed
— and the only lever a client has is keeping the prefix identical from one
request to the next.

There is no portable switch, because the providers do not agree on what
caching is. Most cache implicitly with nothing to enable; Anthropic and Bedrock
want an explicit breakpoint; Google wants a separate cached resource created
out of band and billed by the hour. `cache_prompt` is on by default, because an
agent loop resends its whole prefix every turn:

```python
Options(cache_prompt=True)     # 5-minute TTL
Options(cache_prompt="1h")     # longer TTL, higher write cost
Options(cache_prompt=False)    # off
```

| Provider | Mode | Minimum | Read | What `cache_prompt` does |
|---|---|---|---|---|
| Anthropic | explicit | 512–4096 by model | 0.1× | sets `anthropic_cache` |
| Bedrock | explicit | by model | varies | inserts a `CachePoint` |
| OpenAI | automatic | 1024, then 128-token steps | 0.1× | nothing to do |
| Google | implicit | 1024–2048 | ~0.1× | nothing to do |
| Groq | automatic | — | 0.5× | nothing to do |
| DeepSeek | automatic | 64 | ~0.1× | nothing to do |
| xAI | automatic | — | ~0.16× | nothing to do |

The two explicit providers take their breakpoint in different places, which is
why the field is not one setting: Anthropic reads a request-level flag and
advances the breakpoint itself as the conversation grows, while Bedrock takes a
marker inside the message content. Everything else caches on its own, so the
field is deliberately inert there rather than pretending to configure something.

Google's *explicit* caching is not used. It has a 32,768-token minimum and
bills storage by the hour, so an agent loop would pay rent on a cache between
turns; implicit caching already covers recent models for free.

**Conversation history is cached too, not just the system prompt.** On
Anthropic the breakpoint moves forward each request, so turn *n* reads
everything through turn *n−1* from cache and writes only what is new. On
Bedrock the breakpoint sits at the end of the user prompt, which covers the
system prompt, the tool definitions, and the request itself — the part that
does not change while the agent works through its turns.

**A prompt below the minimum is not cached, and no error says so.** Anthropic
needs 512 to 4096 tokens depending on the model, OpenAI 1024, Google 1024 to
2048, DeepSeek 64. A short run reporting zero cache tokens with caching
correctly enabled is under the threshold, not broken.

**Prefix stability is the part that is portable, and it is where the wins
are.** The prefix is the tool definitions, then the system prompt, then the
conversation, and invalidating one level invalidates every level after it — so
an unstable tool description does not cost you the tool block, it costs you the
entire conversation. Two habits keep it intact: nothing time-varying in the
system prompt, and nothing order-varying in the tool definitions. The `Agent`
tool sorts the subagent types it lists for exactly this reason — a set of
agents that renders in a different order between runs costs a full miss for a
difference no model can see. `Skill` sorts its listing, and MCP servers are
sorted by name with each server's tools sorted within it — that last one is the
only case where the order arrives from outside this process, and a server is
free to change its mind about it between two calls.

One gap worth knowing about: `detect_cache_breaks` compares the built-in tool
set only. MCP tools are not in the snapshot, so a server that revises a
description mid-run breaks the cache without being named. The sort above closes
the ordering case, which was the one a server could trip by accident.

**Breaks are silent.** Nothing fails and nothing warns; the only symptom is a
bill several times larger than it should be. `detect_cache_breaks` watches the
cache-read count reported with each response, and when it falls by more than
5% *and* 2000 tokens, names whichever input changed:

```python
Options(detect_cache_breaks=True)
```

```
prompt cache break: tool schema changed (Agent) [call #7, cache read 48210 -> 0, written 48355]
```

Both thresholds have to clear: a proportional test alone fires constantly on
small conversations, an absolute one alone misses large ones. Compaction resets
the baseline, since it drops history on purpose and reporting that as a break
is how warnings become noise. Tools are hashed individually as well as
together, so a rewritten description is named even when the tool set is
unchanged — the common case, and the hardest to spot by reading code, since
nothing about the tool list looks different.

Detection is off by default and costs a few hashes per request when on. It
covers the main run, not subagents, which do not expose per-request usage.
Providers that report no cache tokens hold the count at zero, so they stay
silent rather than reporting nonsense — no signal is not the same as no cache.

## Cost

No provider returns a price with its response, so cost is always computed on
the client from token counts. `model_pricing` supplies the rates, in US
dollars per million tokens — the unit providers publish, so a price list is
transcribed rather than converted.

```python
from ubiquity import ModelPricing, Options, summon

options = Options(
    model="anthropic:claude-opus-4-5",
    model_pricing={
        "claude-opus": ModelPricing(
            input=5.0,
            output=25.0,
            cache_read=0.5,
            cache_write=6.25,
            context_window=200_000,
        ),
    },
)

async for message in summon("audit the config", options):
    if message.type == "result":
        print(message.total_cost_usd)
```

Keys are matched as substrings of the model name, longest pattern first, so
`gpt-5-mini` can be priced apart from `gpt-5` without the general entry
shadowing the specific one. `register_pricing()` declares the same thing
process-wide for callers who would otherwise repeat a table on every run; a
run's own entry wins, and it wins whole rather than merging, so one entry is
one complete statement about a model.

`context_window` rides along because a model's window and its price are looked
up together and revised on the same occasion. It feeds compaction, and
`max_context_tokens` still overrides it — that is set for one run, while a
price entry describes a model in general.

**Cached tokens are netted out, not added on.** pydantic-ai's buckets are
inclusive: `input_tokens` already contains `cache_read_tokens` and
`cache_write_tokens`, normalized across providers that report them separately.
Cached tokens are subtracted before the input rate applies, or every cached
token would be billed twice — an error that grows with exactly the caching the
previous section exists to encourage. Leaving `cache_read` and `cache_write`
unset prices cached tokens as ordinary input, which overstates the bill on any
provider that discounts reads; an estimate should err high, not low.

**Every response is priced against the model that actually served it.** A run
can span several: a fallback chain answers from whichever backend was
reachable, compaction summarizes on `compact_model`, and a subagent may carry
its own. All three are charged to the same total, so delegated and
housekeeping work cannot be spent invisibly.

Models with no entry fall back to the published figures in `genai-prices`,
which ships with pydantic-ai. Set `market_pricing=False` to make your table the
only source. The fallback needs the provider name as well as the model name and
returns nothing without it — asked to price a bare `deepseek-v4-pro`, it
matches the hosted DeepSeek service and invents a bill for a model running on
localhost. A guessed price is worse than no price, because nothing distinguishes
it from a real one.

**`total_cost_usd` is `None`, never `0.0`, when anything went unpriced.** Those
are different claims — *unknown* and *free* — and a locally hosted model
reported as costing nothing would be indistinguishable from one that genuinely
was. One unpriced response makes the whole run unknown, since a partial sum in a
cost field reads as a complete one while understating the bill. The models
responsible are named at `DEBUG` on the `ubiquity` logger.

The figures track the installed price snapshot, not your contract: negotiated
rates, committed-use discounts, and batch pricing are invisible to it. Supply
`model_pricing` when the number has to be right.

## Retries

Rate limits are retried. Long runs are what this SDK is built for, and long
runs are what get rate limited, so a 429 arriving forty tool calls in has to be
a pause rather than the end.

```python
Options(
    model="anthropic:claude-opus-4-5",
    max_retries=3,
    retry_max_wait=60.0,
)
```

**This is not `Agent(retries=...)`.** pydantic-ai's parameter is a budget for
tool calls and output validation, and it never sees a rate limit: a 429 is
refused by the transport before there is a response to validate. `max_retries`
is the HTTP-level budget, and setting it to zero sends requests through no
transport of this SDK's making.

**The retry sits under the provider's SDK, not around the agent.** That is the
only layer where both the status code and the `Retry-After` header are visible,
and it resends one request instead of replaying a turn. It is wired in through
`provider_settings()`, which is the single place every provider in a run is
configured from — so the policy covers `fallback_model` and `compact_model`
too, rather than protecting the main model and quietly missing the others.
Providers that reach their backend through an SDK of their own instead of over
HTTP — Bedrock's boto3 client is the one that matters — are detected and left
alone rather than crashed into.

**`Retry-After` wins when the provider sends one**, since a server saying when
to come back is better information than any local guess. Otherwise the wait is
exponential with full jitter: a uniform draw below the ceiling rather than the
ceiling itself, because concurrent runs that back off by the same amount
rebuild the burst that limited them. A `Retry-After` longer than
`retry_max_wait` ends the retries and returns the response — sleeping for the
cap only to ask again spends an attempt to be refused on the same grounds.

## Cancelling a run

`abort` is your own `asyncio.Event`, not one the SDK creates — the point of it
is to be settable from outside the `summon()` coroutine.

```python
abort = asyncio.Event()
task = asyncio.create_task(drain(summon("refactor it", Options(abort=abort))))
abort.set()
```

Setting it ends the loop and refuses the next tool call. The check sits in the
one funnel every tool passes through, so it covers the built-in suite, MCP
servers, and anything you register, rather than each tool being trusted to
remember.

`Bash` goes further, because it is the one tool that can block for its whole
timeout with nothing to poll: the abort is **raced against the command**, and
an abort kills the process and reaps it before reporting the run as
interrupted. Cancelling the coroutine instead would leave the process running
with nobody holding its handle. A run that sets no `abort` takes exactly the
path it did before.

Retried are 408, 429, 500, 502, 503, 504, and 529, plus connection and timeout
errors. A 400 is not: a malformed request fails the same way however many times
it is sent. Requests whose body is no longer in memory are not retried either,
which cannot happen for the JSON every provider sends but would silently
truncate a streaming upload. Every attempt is logged at `DEBUG` on the
`ubiquity` logger.

## Todos

`TodoWrite` edits a list, and the list outlives the run.

Individual tasks can be changed without restating the rest. A task is named by
its id or by its exact content, so the model can refer to one either way:

```python
{"add": [{"content": "write the parser"}]}
{"update": [{"task": "write the parser", "status": "in_progress"}]}
{"remove": ["write the parser"]}
{"todos": [...]}
```

Whole-list writes still work and are the right call when starting a plan from
scratch, but they cannot be mixed with edits in one call — a request that both
replaces the list and patches it has no unambiguous meaning.

**A reference that matches nothing is an error, not a no-op.** Ignoring it
would leave the model believing it had completed a task it never touched, and a
plan that disagrees with reality is worse than a retry.

**The one-in-progress invariant is checked against the result**, not the
request, because with incremental edits an `add` can introduce a second
in-progress task without naming the first.

Lists persist to `~/.ubiquity/todos/<project-slug>/<key>/<task-id>.json` —
**one file per task, not one file per list.** A list stored as a single document has to be rewritten
whole on every change, so two runs editing different tasks from their own stale
copies overwrite each other outright. Disjoint tasks in separate files never
contend, which removes the problem instead of locking around it. Each write
touches only the tasks it changed, and the store is re-read on every call
rather than trusting the copy in memory.

The one piece of genuinely shared state is the ordering, carried as a
`position` on each task. Two runs appending at once can pick the same position,
which leaves their relative order undefined but loses nothing; ties break on id
so a list always reads back stably.

When a stored list has unfinished work, it is loaded into the run and described
in the first prompt — a stored list the model is never told about is a list it
duplicates. An all-completed list is not carried over, since finished work from
an unrelated run is noise.

```python
Options(
    persist_todos=True,
    todo_scope="project",
    todo_dir=None,
)
```

`todo_scope` decides what the list belongs to. `project` keys by working
directory, which is what makes a list survive a process exit — every run mints
a fresh session id, so a `session`-scoped list is written and never read again
until session resumption is wired up. The tradeoff of the default is real: two
concurrent runs in one directory share a list.

Individual task files are written through a temporary file and an atomic
rename, so a crash mid-write cannot leave a half-parsed task behind. A file
that is unreadable anyway is skipped, not fatal — one corrupt task should not
lose the rest of the list.

**A subagent gets its own list, keyed by agent id.** It shares the parent's
working directory, so without a
separate key a delegated side task would edit the plan its parent is still
working through. The list is discarded when the subagent reports: an agent id
names one delegated task and never recurs, so a session spawning hundreds of
agents would otherwise accumulate one dead list per agent. Agent ids are unique
per invocation because subagents may run in parallel.

## Sessions

Transcripts are JSONL, one record per line, appended as the run proceeds — a
crashed run still leaves a readable transcript.

```python
from ubiquity import SessionStore

store = SessionStore()
for info in store.list(limit=10):
    print(info.session_id, info.summary)

forked = store.fork(session_id, cwd, up_to_uuid=some_record_uuid)
```

Records chain through `parent_uuid`, which is what makes forking work: a fork
copies records up to a chosen point and remaps every UUID, producing an
independent session that shares history but diverges afterward.

Persistence is on by default under `~/.ubiquity/sessions`. Disable with
`Options(persist_session=False)` or redirect with `Options(session_dir=...)`.

Resuming replays a stored transcript as conversation rather than as a summary
of one:

```python
Options(resume=session_id)              # continue that session
Options(continue_conversation=True)     # continue the latest one for this cwd
Options(resume=session_id, fork_session=True)   # branch, leaving it untouched
```

A tool call whose result is missing — denied, or interrupted by a crash — is
left out of the replay, since most providers reject a dangling tool use and
would make the session unresumable.

## Streaming

`Options(include_partial_messages=True)` adds `stream_event` messages carrying
each delta as it arrives, ahead of the complete `assistant` message for that
turn.

```python
async for message in summon(prompt, Options(include_partial_messages=True)):
    if message.type == "stream_event":
        print(message.delta, end="", flush=True)
```

## Memory files

`UBIQUITY.md` holds standing instructions — how this project wants code
written, what to never touch, which command runs the tests. They go into the
system prompt after everything else and immediately before
`append_system_prompt`, because a rule the model reads after the guidance it
contradicts is the one it keeps.

Nothing is read unless asked for, for the same reason skills are not:

```python
Options(
    memory_sources=["user", "project", "local"],
    memory=["./docs/conventions.md"],
)
```

| source | files |
| --- | --- |
| `user` | `~/.ubiquity/UBIQUITY.md` |
| `project` | `UBIQUITY.md` and `.ubiquity/UBIQUITY.md`, in every directory from the filesystem root down to the cwd |
| `local` | `UBIQUITY.local.md`, in those same directories |

The order is weakest first: user, then project, then local, then the explicit
`memory` files. Within the project and local sources the walk runs downward, so
a checkout root is read before the subdirectory you started in and the nearest
file has the last word. A file reached twice keeps its first, weakest position,
so adding a source can add instructions but never reorder the ones already
there. That matters beyond tidiness — this text sits in the cached prefix, and
a listing that renders differently between two runs costs a full cache miss.

A file can pull in another with `@path`, anywhere in a sentence:

```markdown
Style rules live in @docs/style.md, and the release steps in @docs/release.md.
```

A reference ends at the whitespace after it, with any sentence punctuation
trimmed off the tail, so `@docs/style.md,` and `(@docs/style.md)` both name
`docs/style.md`. Escape a space in a filename as `@my\ notes.md`. An `@`
preceded by a letter or digit is not a reference, which is what keeps
`me@example.com` out of it.

The included file is loaded straight after the file that named it, and the
render says which file included it. Relative paths resolve against the
including file's own directory. Three limits keep this from becoming a way to
read arbitrary files into a prompt:

- **Scope.** A `project` or `local` file may only include from under the
  working directory; only a `user` file may reach into the home directory. A
  `UBIQUITY.md` is checked in, which means it is written by whoever wrote the
  repository, and `@~/.aws/credentials` in a cloned repo must not be a way to
  read a contributor's keys.
- **Type.** Only text extensions — `.md`, `.txt`, `.rst`, `.json`, `.toml`,
  `.yaml`, and friends.
- **Depth.** Five levels, with each file loaded once however many times it is
  named, so a cycle terminates.

`@` inside a fenced block or an inline code span is not an include, so
documenting the syntax does not trigger it. That is decided by scanning rather
than by a full markdown parse, which is the same answer everywhere short of an
`@path` inside an HTML block. A file over 40,000 characters is cut with a
visible marker naming the file rather than silently passed along whole.

Subagents inherit the run's memory. A project's standing instructions do not
stop applying because the work was delegated.

Not implemented: org-managed memory files, rule files with conditional
frontmatter, and deduplication across nested worktrees.

## Settings files

Nothing is read from the filesystem unless asked for, so a stray file cannot
reconfigure a caller's run:

```python
Options(setting_sources=["project", "local"])
```

| source | file |
| --- | --- |
| `user` | `~/.ubiquity/settings.json` |
| `project` | `<cwd>/.ubiquity/settings.json` |
| `local` | `<cwd>/.ubiquity/settings.local.json` |

```json
{
  "model": "openai:gpt-5",
  "env": {"NO_COLOR": "1"},
  "permissions": {
    "deny": ["Bash(rm:*)"],
    "ask": ["Bash(git push:*)"],
    "additionalDirectories": ["../shared"]
  }
}
```

Local beats project beats user, and explicit `Options` beat all three — except
for permission rules, which are unioned. A rule in a settings file is a
restriction the caller did not write, so passing a list of their own must not
drop it.

## Custom tools

Subclass `Tool` with a Pydantic input model:

```python
from pydantic import BaseModel, Field
from ubiquity import Tool, ToolContext, ToolOutput, PermissionResultAllow, builtin_tools

class SearchInput(BaseModel):
    query: str = Field(description="What to search for.")

class SearchTool(Tool[SearchInput]):
    name = "Search"
    description = "Search the knowledge base."
    input_model = SearchInput

    def is_read_only(self, args): return True
    def is_concurrency_safe(self, args): return True

    async def check_permissions(self, args, ctx):
        return PermissionResultAllow(reason="read-only lookup")

    async def call(self, args, ctx) -> ToolOutput:
        return ToolOutput(content=f"Results for {args.query}")

Options(tools=[*builtin_tools(), SearchTool()])
```

Override `permission_rule_content` to make a tool addressable by content rules
like `Search(internal:*)`. Return **every** string that must be authorized —
the engine requires all of them to match.

## Development

```bash
uv sync
uv run pytest
```

## License

MIT
