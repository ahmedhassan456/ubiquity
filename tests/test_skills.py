"""Tests for skills: discovery, the three loading steps, and the subagent cut.

The property that matters most is the one that makes skills worth having: a
skill's body must not reach the model until the model asks for it. A test suite
that only checked the body was reachable would pass on an implementation that
pasted every skill into the system prompt, which is the design being avoided.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ubiquity import AgentDefinition, Options, summon
from ubiquity.prompts import SKILLS_NOTE, build_system_prompt
from ubiquity.skills import (
    Skill,
    discover,
    load_skill,
    load_skills,
    select,
    skills_path,
    split_frontmatter,
)
from ubiquity.tools.skill import (
    MAX_DESCRIPTION_CHARS,
    SkillInput,
    SkillTool,
    format_listing,
)

BODY = "Step one: read the file.\nStep two: do the thing."


def write_skill(
    root: Path,
    name: str,
    *,
    description: str = "Use when the user mentions widgets.",
    body: str = BODY,
    frontmatter_name: str | None = None,
    extra: str = "",
) -> Path:
    """Create a skill directory under `root` and return it."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    declared = name if frontmatter_name is None else frontmatter_name
    lines = ["---", f"name: {declared}", f"description: {description}"]
    if extra:
        lines.append(extra)
    lines += ["---", "", body]
    (directory / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")
    return directory


def scripted(*turns: list[Any]) -> FunctionModel:
    """Build a model that replays `turns`, one response per model request."""
    calls = {"n": 0}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        index = min(calls["n"], len(turns) - 1)
        calls["n"] += 1
        return ModelResponse(parts=list(turns[index]))

    return FunctionModel(respond)


class TestFrontmatter:
    def test_a_file_with_no_frontmatter_is_all_body(self) -> None:
        meta, body = split_frontmatter("just text")
        assert meta == {}
        assert body == "just text"

    def test_keys_and_values_are_split_on_the_first_colon(self) -> None:
        meta, _ = split_frontmatter("---\ndescription: use: when asked\n---\nb")
        assert meta["description"] == "use: when asked"

    def test_quotes_around_a_value_are_stripped(self) -> None:
        meta, _ = split_frontmatter('---\nname: "quoted"\n---\nb')
        assert meta["name"] == "quoted"

    def test_the_body_starts_after_the_closing_fence(self) -> None:
        _, body = split_frontmatter("---\nname: a\n---\n\nbody line")
        assert body == "body line"

    def test_an_unclosed_fence_is_not_frontmatter(self) -> None:
        """Half-parsing a truncated file would invent a skill from its prose."""
        text = "---\nname: a\nno closing fence"
        meta, body = split_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_comments_and_blank_lines_are_skipped(self) -> None:
        meta, _ = split_frontmatter("---\n# note\n\nname: a\n---\nb")
        assert meta == {"name": "a"}


class TestLoading:
    def test_a_skill_loads_its_name_description_and_body(self, tmp_path: Path) -> None:
        write_skill(tmp_path, "widgets")
        skill = load_skill(tmp_path / "widgets" / "SKILL.md")
        assert skill is not None
        assert skill.name == "widgets"
        assert skill.description == "Use when the user mentions widgets."
        assert skill.body.strip() == BODY

    def test_a_skill_with_no_description_is_skipped(self, tmp_path: Path) -> None:
        """Without one the model has no basis for deciding to load it."""
        directory = tmp_path / "nameless"
        directory.mkdir()
        (directory / "SKILL.md").write_text("---\nname: nameless\n---\nbody")
        assert load_skill(directory / "SKILL.md") is None

    def test_the_directory_name_stands_in_for_a_missing_name(
        self, tmp_path: Path
    ) -> None:
        directory = tmp_path / "widgets"
        directory.mkdir()
        (directory / "SKILL.md").write_text("---\ndescription: d\n---\nbody")
        skill = load_skill(directory / "SKILL.md")
        assert skill is not None and skill.name == "widgets"

    def test_a_name_that_is_not_an_identifier_is_refused(self, tmp_path: Path) -> None:
        """The model types the name back, so it has to be typeable."""
        write_skill(tmp_path, "widgets", frontmatter_name="../../etc/passwd")
        assert load_skill(tmp_path / "widgets" / "SKILL.md") is None

    def test_unknown_frontmatter_keys_are_kept_but_not_acted_on(
        self, tmp_path: Path
    ) -> None:
        write_skill(tmp_path, "widgets", extra="allowed-tools: Read, Bash")
        skill = load_skill(tmp_path / "widgets" / "SKILL.md")
        assert skill is not None
        assert skill.metadata["allowed-tools"] == "Read, Bash"

    def test_the_path_is_absolute(self, tmp_path: Path) -> None:
        """The model reads bundled files back by path; a relative one misses."""
        write_skill(tmp_path, "widgets")
        skill = load_skill(Path("tests/..") / tmp_path / "widgets" / "SKILL.md")
        assert skill is not None and skill.path.is_absolute()


class TestDiscovery:
    def test_a_directory_of_skills_yields_each_one(self, tmp_path: Path) -> None:
        write_skill(tmp_path, "alpha")
        write_skill(tmp_path, "beta")
        assert len(discover(tmp_path)) == 2

    def test_a_single_skill_directory_is_also_a_root(self, tmp_path: Path) -> None:
        directory = write_skill(tmp_path, "alpha")
        assert discover(directory) == [directory / "SKILL.md"]

    def test_a_directory_that_does_not_exist_yields_nothing(
        self, tmp_path: Path
    ) -> None:
        assert discover(tmp_path / "absent") == []

    def test_a_subdirectory_without_a_skill_file_is_ignored(
        self, tmp_path: Path
    ) -> None:
        write_skill(tmp_path, "alpha")
        (tmp_path / "notes").mkdir()
        assert len(discover(tmp_path)) == 1

    def test_one_malformed_skill_does_not_take_down_the_others(
        self, tmp_path: Path
    ) -> None:
        write_skill(tmp_path, "good")
        broken = tmp_path / "broken"
        broken.mkdir()
        (broken / "SKILL.md").write_text("no frontmatter at all")
        assert set(load_skills([tmp_path])) == {"good"}

    def test_a_later_root_overrides_an_earlier_one_by_name(
        self, tmp_path: Path
    ) -> None:
        """This is what lets a project replace a skill it inherited."""
        first = tmp_path / "first"
        second = tmp_path / "second"
        write_skill(first, "widgets", description="inherited")
        write_skill(second, "widgets", description="overridden")
        loaded = load_skills([first, second])
        assert loaded["widgets"].description == "overridden"


class TestOptionsWiring:
    def test_nothing_is_loaded_without_being_asked_for(self, tmp_path: Path) -> None:
        """Skills are instructions; picking them up unasked overrides Options."""
        write_skill(tmp_path / ".ubiquity" / "skills", "widgets")
        assert Options(cwd=tmp_path).resolved_skill_roots() == []

    def test_a_named_source_resolves_to_the_conventional_directory(
        self, tmp_path: Path
    ) -> None:
        roots = Options(cwd=tmp_path, skill_sources=["project"]).resolved_skill_roots()
        assert roots == [tmp_path / ".ubiquity" / "skills"]

    def test_explicit_roots_come_after_conventional_ones(self, tmp_path: Path) -> None:
        """Later wins, so an explicit directory overrides an ambient skill."""
        roots = Options(
            cwd=tmp_path, skill_sources=["project"], skills=[tmp_path / "mine"]
        ).resolved_skill_roots()
        assert roots[-1] == (tmp_path / "mine").resolve()

    def test_a_home_relative_root_is_expanded(self, tmp_path: Path) -> None:
        """`resolve` alone would make `~/skills` a directory named `~` in cwd."""
        roots = Options(cwd=tmp_path, skills=["~/skills"]).resolved_skill_roots()
        assert roots == [Path.home() / "skills"]

    def test_sources_are_ordered_weakest_first_whatever_order_is_given(
        self, tmp_path: Path
    ) -> None:
        roots = Options(
            cwd=tmp_path, skill_sources=["local", "user"]
        ).resolved_skill_roots()
        assert roots == [
            skills_path("user", tmp_path),
            skills_path("local", tmp_path),
        ]


class TestTheTool:
    async def test_the_listing_names_every_skill_but_no_body(
        self, make_ctx, tmp_path: Path
    ) -> None:
        """The whole point: descriptions are cheap, bodies are not."""
        write_skill(tmp_path, "widgets")
        tool = SkillTool(load_skills([tmp_path]))
        listing = await tool.prompt(make_ctx(cwd=tmp_path))
        assert "widgets" in listing
        assert BODY not in listing

    async def test_loading_a_skill_returns_its_body(
        self, make_ctx, tmp_path: Path
    ) -> None:
        write_skill(tmp_path, "widgets")
        tool = SkillTool(load_skills([tmp_path]))
        output = await tool.call(SkillInput(name="widgets"), make_ctx(cwd=tmp_path))
        assert BODY in output.content
        assert not output.is_error

    async def test_an_unknown_name_is_rejected_with_the_available_ones(
        self, make_ctx, tmp_path: Path
    ) -> None:
        write_skill(tmp_path, "widgets")
        tool = SkillTool(load_skills([tmp_path]))
        error = await tool.validate_input(
            SkillInput(name="gadgets"), make_ctx(cwd=tmp_path)
        )
        assert error is not None
        assert "widgets" in error.message

    async def test_bundled_files_are_listed_by_absolute_path(
        self, make_ctx, tmp_path: Path
    ) -> None:
        """Step three of loading: the body points, the model reads."""
        directory = write_skill(tmp_path, "widgets")
        (directory / "checklist.md").write_text("check things")
        tool = SkillTool(load_skills([tmp_path]))
        output = await tool.call(SkillInput(name="widgets"), make_ctx(cwd=tmp_path))
        assert str((directory / "checklist.md").resolve()) in output.content

    async def test_a_skill_with_no_bundled_files_says_nothing_about_them(
        self, make_ctx, tmp_path: Path
    ) -> None:
        write_skill(tmp_path, "widgets")
        tool = SkillTool(load_skills([tmp_path]))
        output = await tool.call(SkillInput(name="widgets"), make_ctx(cwd=tmp_path))
        assert "bundled" not in output.content.lower()

    async def test_the_skill_file_is_not_listed_as_its_own_attachment(
        self, make_ctx, tmp_path: Path
    ) -> None:
        directory = write_skill(tmp_path, "widgets")
        (directory / "checklist.md").write_text("check things")
        tool = SkillTool(load_skills([tmp_path]))
        output = await tool.call(SkillInput(name="widgets"), make_ctx(cwd=tmp_path))
        assert "SKILL.md" not in output.content

    def test_the_tool_is_read_only_and_parallel_safe(self) -> None:
        tool = SkillTool({})
        args = SkillInput(name="widgets")
        assert tool.is_read_only(args)
        assert tool.is_concurrency_safe(args)


class TestTheSystemPrompt:
    """The prompt says skills exist; the tool description says which."""

    def test_the_prompt_points_at_the_tool_without_repeating_the_listing(
        self, tmp_path: Path
    ) -> None:
        write_skill(tmp_path, "widgets")
        skills = load_skills([tmp_path])
        prompt = build_system_prompt(Options(cwd=tmp_path), [], skills)
        assert SKILLS_NOTE in prompt
        assert "Use when the user mentions widgets." not in prompt
        assert "widgets" not in prompt
        assert BODY not in prompt

    def test_a_run_with_no_skills_gains_no_section(self, tmp_path: Path) -> None:
        assert SKILLS_NOTE not in build_system_prompt(Options(cwd=tmp_path), [], {})

    def test_the_prompt_does_not_grow_with_the_number_of_skills(
        self, tmp_path: Path
    ) -> None:
        """The whole point of moving the listing: this is what was paid twice."""
        write_skill(tmp_path, "alpha")
        one = build_system_prompt(Options(cwd=tmp_path), [], load_skills([tmp_path]))
        for name in ("beta", "gamma", "delta"):
            write_skill(tmp_path, name)
        many = build_system_prompt(Options(cwd=tmp_path), [], load_skills([tmp_path]))
        assert one == many

    async def test_the_description_reaches_the_model_exactly_once(
        self, make_ctx, tmp_path: Path
    ) -> None:
        """Across both halves of the cached prefix, not just within either one."""
        write_skill(tmp_path, "widgets")
        skills = load_skills([tmp_path])
        options = Options(cwd=tmp_path)
        prefix = build_system_prompt(options, [], skills) + await SkillTool(
            skills
        ).prompt(make_ctx(cwd=tmp_path))
        assert prefix.count("Use when the user mentions widgets.") == 1


class TestTheListingBudget:
    """A listing is for recognition, so it is capped rather than complete."""

    def test_a_long_description_is_clamped(self) -> None:
        skills = {"a": Skill("a", "x" * 900, "body", Path("/x/SKILL.md"))}
        listed = format_listing(skills)
        assert len(listed) < 900
        assert listed.endswith("…")

    def test_a_short_description_is_left_alone(self) -> None:
        skills = {"a": Skill("a", "Use for widgets.", "body", Path("/x/SKILL.md"))}
        assert format_listing(skills) == "  - a: Use for widgets."

    def test_a_listing_over_budget_degrades_to_names(self) -> None:
        skills = {
            f"skill-{i:03d}": Skill(
                f"skill-{i:03d}", "d" * MAX_DESCRIPTION_CHARS, "b", Path("/x/SKILL.md")
            )
            for i in range(200)
        }
        listed = format_listing(skills)
        assert "ddd" not in listed
        assert listed.count("skill-") == 200

    def test_no_skill_is_dropped_by_the_budget(self) -> None:
        """A skill the model cannot name is a skill it cannot call."""
        skills = {
            f"s{i}": Skill(f"s{i}", "d" * 400, "b", Path("/x/SKILL.md"))
            for i in range(400)
        }
        listed = format_listing(skills)
        for name in skills:
            assert f"- {name}\n" in listed + "\n"

    def test_the_listing_is_sorted_so_the_cached_prefix_is_stable(self) -> None:
        skills = {
            n: Skill(n, "d", "b", Path("/x/SKILL.md")) for n in ("zebra", "alpha")
        }
        listed = format_listing(skills)
        assert listed.index("alpha") < listed.index("zebra")

    def test_the_same_skills_in_a_different_order_render_identically(self) -> None:
        skills = {
            n: Skill(n, "d", "b", Path("/x/SKILL.md")) for n in ("alpha", "zebra")
        }
        flipped = {k: skills[k] for k in reversed(list(skills))}
        assert format_listing(skills) == format_listing(flipped)


class TestSelecting:
    def test_none_inherits_every_skill(self) -> None:
        skills = {"a": Skill("a", "d", "b", Path("/x/SKILL.md"))}
        assert select(skills, None) == skills

    def test_an_empty_sequence_grants_none(self) -> None:
        """None and `()` differ, the same way they do for a subagent's tools."""
        skills = {"a": Skill("a", "d", "b", Path("/x/SKILL.md"))}
        assert select(skills, []) == {}

    def test_a_name_that_matches_nothing_is_dropped(self) -> None:
        assert select({}, ["absent"]) == {}


class TestInTheLoop:
    async def test_a_configured_run_gets_the_tool(self, tmp_path: Path) -> None:
        write_skill(tmp_path / "skills", "widgets")
        messages = [
            m
            async for m in summon(
                "hi",
                Options(
                    model=scripted([TextPart(content="ok")]),
                    cwd=tmp_path,
                    skills=[tmp_path / "skills"],
                ),
            )
        ]
        assert "Skill" in messages[0].tools

    async def test_a_run_without_skills_does_not_get_the_tool(
        self, tmp_path: Path
    ) -> None:
        """A tool that lists nothing is one the model can only misuse."""
        messages = [
            m
            async for m in summon(
                "hi",
                Options(model=scripted([TextPart(content="ok")]), cwd=tmp_path),
            )
        ]
        assert "Skill" not in messages[0].tools

    async def test_a_bare_deny_rule_takes_the_tool_away(self, tmp_path: Path) -> None:
        write_skill(tmp_path / "skills", "widgets")
        messages = [
            m
            async for m in summon(
                "hi",
                Options(
                    model=scripted([TextPart(content="ok")]),
                    cwd=tmp_path,
                    skills=[tmp_path / "skills"],
                    disallowed_tools=["Skill"],
                ),
            )
        ]
        assert "Skill" not in messages[0].tools

    async def test_an_allow_list_that_omits_it_takes_it_away(
        self, tmp_path: Path
    ) -> None:
        write_skill(tmp_path / "skills", "widgets")
        messages = [
            m
            async for m in summon(
                "hi",
                Options(
                    model=scripted([TextPart(content="ok")]),
                    cwd=tmp_path,
                    skills=[tmp_path / "skills"],
                    allowed_tools=["Read"],
                ),
            )
        ]
        assert "Skill" not in messages[0].tools

    async def test_the_model_can_load_a_skill_mid_run(self, tmp_path: Path) -> None:
        write_skill(tmp_path / "skills", "widgets")
        messages = [
            m
            async for m in summon(
                "hi",
                Options(
                    model=scripted(
                        [
                            ToolCallPart(
                                tool_name="Skill",
                                args={"name": "widgets"},
                                tool_call_id="c0",
                            )
                        ],
                        [TextPart(content="done")],
                    ),
                    cwd=tmp_path,
                    skills=[tmp_path / "skills"],
                    permission_mode="bypassPermissions",
                ),
            )
        ]
        results = [m for m in messages if m.type == "tool_result"]
        assert results and BODY in str(results[0].output.content)

    async def test_a_skill_directory_outside_cwd_becomes_readable(
        self, tmp_path_factory
    ) -> None:
        """Step three is instructions to read a path, so the path must open."""
        cwd = tmp_path_factory.mktemp("project")
        root = tmp_path_factory.mktemp("elsewhere")
        directory = write_skill(root, "widgets")
        (directory / "checklist.md").write_text("check things")
        from ubiquity.client import _build_context
        from ubiquity.hooks.registry import HookRegistry

        ctx = _build_context(Options(cwd=cwd, skills=[root]), "s", HookRegistry(()))
        assert ctx.is_path_allowed(directory / "checklist.md")

    async def test_an_unconfigured_run_cannot_read_a_skill_directory(
        self, tmp_path_factory
    ) -> None:
        """The widening is exactly as broad as the roots the caller named."""
        cwd = tmp_path_factory.mktemp("project")
        root = tmp_path_factory.mktemp("elsewhere")
        directory = write_skill(root, "widgets")
        from ubiquity.client import _build_context
        from ubiquity.hooks.registry import HookRegistry

        ctx = _build_context(Options(cwd=cwd), "s", HookRegistry(()))
        assert not ctx.is_path_allowed(directory / "SKILL.md")


class TestSubagents:
    async def dispatch(
        self, tmp_path: Path, definition: AgentDefinition
    ) -> tuple[set[str], str]:
        """Run a subagent and return its tool names and everything it was told.

        The second value is the instructions and the tool descriptions joined,
        because which half of the cached prefix a skill listing lives in is an
        implementation choice: what a subagent is told about is the union.
        """
        from ubiquity.client import _build_context, run_subagent
        from ubiquity.hooks.registry import HookRegistry

        seen: dict[str, Any] = {}

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen["tools"] = {t.name for t in info.function_tools}
            seen["told"] = "\n".join(
                [info.instructions or ""]
                + [t.description or "" for t in info.function_tools]
            )
            return ModelResponse(parts=[TextPart(content="ok")])

        write_skill(tmp_path / "skills", "alpha")
        write_skill(tmp_path / "skills", "zebra")
        options = Options(
            model=FunctionModel(respond),
            cwd=tmp_path,
            skills=[tmp_path / "skills"],
            permission_mode="bypassPermissions",
        )
        ctx = _build_context(options, "s", HookRegistry(()))
        ctx.extra["skills"] = load_skills(options.resolved_skill_roots())
        await run_subagent("go", ctx, definition)
        return seen["tools"], seen["told"]

    async def test_a_subagent_inherits_the_parent_skills_by_default(
        self, tmp_path: Path
    ) -> None:
        tools, told = await self.dispatch(
            tmp_path, AgentDefinition(description="d", prompt="p")
        )
        assert "Skill" in tools
        assert "alpha" in told and "zebra" in told

    async def test_a_definition_narrows_what_the_subagent_is_told_about(
        self, tmp_path: Path
    ) -> None:
        """The unlisted skill is a line paid for on every request of the run."""
        tools, told = await self.dispatch(
            tmp_path, AgentDefinition(description="d", prompt="p", skills=["alpha"])
        )
        assert "Skill" in tools
        assert "alpha" in told and "zebra" not in told

    async def test_an_empty_definition_list_takes_the_tool_away(
        self, tmp_path: Path
    ) -> None:
        tools, told = await self.dispatch(
            tmp_path, AgentDefinition(description="d", prompt="p", skills=[])
        )
        assert "Skill" not in tools
        assert "alpha" not in told

    def test_a_definition_can_narrow_the_set(self, tmp_path: Path) -> None:
        write_skill(tmp_path, "alpha")
        write_skill(tmp_path, "zebra")
        skills = load_skills([tmp_path])
        definition = AgentDefinition(description="d", prompt="p", skills=["alpha"])
        assert set(select(skills, definition.skills)) == {"alpha"}

    def test_a_definition_cannot_add_a_skill_the_run_lacks(
        self, tmp_path: Path
    ) -> None:
        """Narrowing only. A subagent that could widen its own set is a hole."""
        write_skill(tmp_path, "alpha")
        skills = load_skills([tmp_path])
        definition = AgentDefinition(description="d", prompt="p", skills=["unloaded"])
        assert select(skills, definition.skills) == {}
