"""ubiquity: a coding-agent SDK on top of pydantic-ai.

Everything a terminal coding agent needs — the agent loop, a built-in tool
suite, a rule-based permission system, hooks, subagents, MCP, and session
persistence — built against pydantic-ai's model layer, so the same agent runs
on any of its supported providers.

Example:
    import asyncio
    from ubiquity import summon, Options

    async def main():
        async for message in summon(
            "what python files are here?",
            Options(model="openai:gpt-5", permission_mode="acceptEdits"),
        ):
            if message.type == "assistant":
                print(message.text)

    asyncio.run(main())
"""

from .cache import CacheBreak, CacheBreakDetector, PromptSnapshot, build_snapshot
from .client import summon, run_subagent
from .compaction import (
    CompactionResult,
    compact,
    compaction_threshold,
    infer_context_window,
    measure_context,
    register_context_window,
    should_compact,
    summarize,
)
from .hooks import HookInput, HookMatcher, HookOutput
from .mcp import (
    McpHttpServerConfig,
    McpServerConfig,
    McpSSEServerConfig,
    McpStdioServerConfig,
    parse_config,
)
from .microcompact import (
    COMPACTABLE_TOOLS,
    MicrocompactResult,
    compactable_call_ids,
    microcompact,
)
from .models import (
    clear_aliases,
    known_models,
    known_providers,
    openai_compatible,
    register_alias,
    registered_aliases,
    resolve_model,
    with_fallback,
)
from .options import AgentDefinition, CanUseTool, Options, SettingSource
from .pricing import (
    CostMeter,
    ModelPricing,
    clear_pricing,
    cost_of,
    find_pricing,
    register_pricing,
)
from .retry import RETRY_STATUSES, RetryTransport, retry_client
from .sessions import SessionInfo, SessionRecord, SessionStore, history_from
from .settings import apply_settings, load_settings, settings_path
from .agents import agents_path, load_agents, parse_agent
from .memory import MemoryFile, load_memory, memory_paths
from .skills import Skill, load_skills, skills_path
from .subagents.agent_tool import AgentTool
from .todos import TodoItem, TodoPatch, TodoStore
from .tool import FileState, PermissionContext, Tool, ToolContext, ValidationError
from .tools import builtin_tools, resolve_tools
from .tools.ask import AskUserQuestionInput, AskUserQuestionTool
from .tools.skill import SkillTool
from .toolset import ToolDenied
from .types import (
    PermissionMode,
    PermissionResult,
    PermissionResultAllow,
    PermissionResultAsk,
    PermissionResultDeny,
    PermissionResultPassthrough,
    PermissionRuleValue,
    PermissionUpdate,
    SDKAssistantMessage,
    SDKMessage,
    SDKPartialAssistantMessage,
    SDKResultMessage,
    SDKSystemMessage,
    SDKToolResultMessage,
    SDKCompactBoundaryMessage,
    SDKMicrocompactMessage,
    SDKToolUseMessage,
    SDKUserMessage,
    ToolOutput,
)

__version__ = "0.1.0"

__all__ = [
    "summon",
    "run_subagent",
    "Options",
    "AgentDefinition",
    "ModelPricing",
    "CostMeter",
    "register_pricing",
    "clear_pricing",
    "find_pricing",
    "cost_of",
    "RetryTransport",
    "RETRY_STATUSES",
    "retry_client",
    "AgentTool",
    "CanUseTool",
    "SessionStore",
    "SessionRecord",
    "SessionInfo",
    "history_from",
    "apply_settings",
    "Skill",
    "MemoryFile",
    "SkillTool",
    "AskUserQuestionTool",
    "AskUserQuestionInput",
    "load_skills",
    "load_memory",
    "load_agents",
    "parse_agent",
    "agents_path",
    "memory_paths",
    "skills_path",
    "load_settings",
    "settings_path",
    "SettingSource",
    "TodoStore",
    "TodoItem",
    "TodoPatch",
    "CacheBreakDetector",
    "CacheBreak",
    "PromptSnapshot",
    "build_snapshot",
    "McpServerConfig",
    "McpStdioServerConfig",
    "McpSSEServerConfig",
    "McpHttpServerConfig",
    "parse_config",
    "Tool",
    "ToolContext",
    "ToolOutput",
    "ToolDenied",
    "ValidationError",
    "FileState",
    "PermissionContext",
    "builtin_tools",
    "resolve_tools",
    "HookMatcher",
    "HookInput",
    "HookOutput",
    "resolve_model",
    "with_fallback",
    "openai_compatible",
    "known_models",
    "known_providers",
    "register_alias",
    "registered_aliases",
    "clear_aliases",
    "compact",
    "summarize",
    "should_compact",
    "measure_context",
    "infer_context_window",
    "register_context_window",
    "compaction_threshold",
    "CompactionResult",
    "microcompact",
    "compactable_call_ids",
    "MicrocompactResult",
    "COMPACTABLE_TOOLS",
    "PermissionMode",
    "PermissionResult",
    "PermissionResultAllow",
    "PermissionResultDeny",
    "PermissionResultAsk",
    "PermissionResultPassthrough",
    "PermissionRuleValue",
    "PermissionUpdate",
    "SDKMessage",
    "SDKSystemMessage",
    "SDKUserMessage",
    "SDKAssistantMessage",
    "SDKPartialAssistantMessage",
    "SDKToolUseMessage",
    "SDKToolResultMessage",
    "SDKResultMessage",
    "SDKCompactBoundaryMessage",
    "SDKMicrocompactMessage",
    "__version__",
]
