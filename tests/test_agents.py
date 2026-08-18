"""Tests for agent files: parsing, discovery, precedence, and the wiring.

The property worth pinning is that a file and a code definition produce the
same subagent, and that a definition written in code wins when both name the
same one. A suite that only checked a file could be parsed would pass on an
implementation that parsed it and then ignored it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from ubiquity import AgentDefinition, Options
from ubiquity.agents import (
    agents_path,
    discover,
    load_agents,
    normalize_keys,
    parse_agent,
    parse_list,
)

from test_summon_loop import collect, scripted

BODY = "You review code. Report what would break."


def write_agent(root: Path, name: str, frontmatter: str, body: str = BODY) -> Path:
    """Write one agent file and return its path."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.md"
    path.write_text(f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8")
    return path


class TestParsing:
    def test_the_body_becomes_the_prompt(self, tmp_path: Path) -> None:
        path = write_agent(tmp_path, "reviewer", "description: Reviews a diff")
        name, definition = parse_agent(path)
        assert name == "reviewer"
        assert definition.description == "Reviews a diff"
        assert definition.prompt == BODY

    def test_the_name_defaults_to_the_file_stem(self, tmp_path: Path) -> None:
        path = write_agent(tmp_path, "reviewer", "description: Reviews a diff")
        assert parse_agent(path)[0] == "reviewer"

    def test_an_explicit_name_wins_over_the_stem(self, tmp_path: Path) -> None:
        path = write_agent(tmp_path, "file-name", "name: chosen\ndescription: d")
        assert parse_agent(path)[0] == "chosen"

    def test_every_field_reaches_the_definition(self, tmp_path: Path) -> None:
        path = write_agent(
            tmp_path,
            "reviewer",
            "description: Reviews a diff\n"
            "tools: Read, Grep\n"
            "skills: release\n"
            "disallowed-tools: Bash\n"
            "model: openai:gpt-5\n"
            "permission-mode: plan\n"
            "max-turns: 7",
        )
        _, definition = parse_agent(path)
        assert definition.tools == ("Read", "Grep")
        assert definition.skills == ("release",)
        assert definition.disallowed_tools == ("Bash",)
        assert definition.model == "openai:gpt-5"
        assert definition.permission_mode == "plan"
        assert definition.max_turns == 7

    def test_a_file_and_a_code_definition_agree(self, tmp_path: Path) -> None:
        path = write_agent(
            tmp_path, "reviewer", "description: Reviews a diff\ntools: Read"
        )
        _, from_file = parse_agent(path)
        assert from_file == AgentDefinition(
            description="Reviews a diff", prompt=BODY, tools=("Read",)
        )

    def test_camel_case_and_hyphens_reach_the_same_field(self, tmp_path: Path) -> None:
        hyphen = write_agent(tmp_path / "a", "r", "description: d\nmax-turns: 3")
        camel = write_agent(tmp_path / "b", "r", "description: d\nmaxTurns: 3")
        assert parse_agent(hyphen)[1].max_turns == parse_agent(camel)[1].max_turns == 3

    def test_a_bracketed_list_parses(self) -> None:
        assert parse_list("[Read, Write]") == ("Read", "Write")

    def test_an_absent_list_inherits_and_an_empty_one_grants_nothing(
        self, tmp_path: Path
    ) -> None:
        """The two are different instructions and neither may become the other."""
        absent = write_agent(tmp_path / "a", "r", "description: d")
        empty = write_agent(tmp_path / "b", "r", "description: d\ntools:")
        assert parse_agent(absent)[1].tools is None
        assert parse_agent(empty)[1].tools == ()

    def test_unknown_keys_are_ignored(self, tmp_path: Path) -> None:
        path = write_agent(tmp_path, "r", "description: d\ncolor: blue")
        assert parse_agent(path) is not None

    def test_normalize_keys_folds_case_and_separators(self) -> None:
        assert normalize_keys({"Max-Turns": "3", "disallowedTools": "Bash"}) == {
            "max_turns": "3",
            "disallowed_tools": "Bash",
        }


class TestABadFileIsSkippedNotFatal:
    def test_no_description_is_skipped(self, tmp_path: Path) -> None:
        assert parse_agent(write_agent(tmp_path, "r", "name: r")) is None

    def test_no_body_is_skipped(self, tmp_path: Path) -> None:
        path = write_agent(tmp_path, "r", "description: d", body="   ")
        assert parse_agent(path) is None

    def test_an_unusable_name_is_skipped(self, tmp_path: Path) -> None:
        path = write_agent(tmp_path, "r", "name: not a name\ndescription: d")
        assert parse_agent(path) is None

    def test_a_bad_max_turns_falls_back_rather_than_guessing(
        self, tmp_path: Path
    ) -> None:
        path = write_agent(tmp_path, "r", "description: d\nmax-turns: soon")
        assert parse_agent(path)[1].max_turns is None

    def test_an_unknown_permission_mode_falls_back(self, tmp_path: Path) -> None:
        path = write_agent(tmp_path, "r", "description: d\npermission-mode: yolo")
        assert parse_agent(path)[1].permission_mode is None

    def test_one_bad_file_does_not_lose_the_others(self, tmp_path: Path) -> None:
        write_agent(tmp_path, "good", "description: d")
        write_agent(tmp_path, "bad", "name: r")
        assert list(load_agents([tmp_path])) == ["good"]


class TestDiscovery:
    def test_a_directory_of_files_is_loaded(self, tmp_path: Path) -> None:
        write_agent(tmp_path, "one", "description: d")
        write_agent(tmp_path, "two", "description: d")
        assert sorted(load_agents([tmp_path])) == ["one", "two"]

    def test_subdirectories_are_walked(self, tmp_path: Path) -> None:
        write_agent(tmp_path / "review", "deep", "description: d")
        assert list(load_agents([tmp_path])) == ["deep"]

    def test_non_markdown_files_are_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("---\ndescription: d\n---\n\nbody")
        assert discover(tmp_path) == []

    def test_a_missing_directory_is_not_an_error(self, tmp_path: Path) -> None:
        assert load_agents([tmp_path / "nope"]) == {}

    def test_discovery_is_sorted_not_left_to_the_filesystem(
        self, tmp_path: Path
    ) -> None:
        """The listing sits in the cached prefix, so its order has to be ours."""
        for name in ("m", "z", "a", "b"):
            write_agent(tmp_path, name, "description: d")
        assert [p.stem for p in discover(tmp_path)] == ["a", "b", "m", "z"]

    def test_a_later_root_wins_a_name_collision(self, tmp_path: Path) -> None:
        write_agent(tmp_path / "first", "r", "description: from first")
        write_agent(tmp_path / "second", "r", "description: from second")
        loaded = load_agents([tmp_path / "first", tmp_path / "second"])
        assert loaded["r"].description == "from second"


class TestOptions:
    def test_no_sources_discovers_nothing(self, tmp_path: Path) -> None:
        write_agent(tmp_path / ".ubiquity" / "agents", "r", "description: d")
        assert Options(cwd=tmp_path, model="x").resolved_agent_roots() == []

    def test_the_conventional_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        assert agents_path("user", tmp_path) == tmp_path / ".ubiquity" / "agents"
        assert agents_path("project", tmp_path) == tmp_path / ".ubiquity" / "agents"
        assert agents_path("local", tmp_path) == tmp_path / ".ubiquity" / "agents.local"

    def test_roots_are_ordered_weakest_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        work = tmp_path / "work"
        work.mkdir()
        options = Options(
            cwd=work, model="x", agent_sources=("local", "user", "project")
        )
        assert options.resolved_agent_roots() == [
            home / ".ubiquity" / "agents",
            work / ".ubiquity" / "agents",
            work / ".ubiquity" / "agents.local",
        ]


class TestTheWiring:
    async def test_a_discovered_agent_reaches_the_tool(self, tmp_path: Path) -> None:
        write_agent(tmp_path / ".ubiquity" / "agents", "reviewer", "description: d")
        messages = await collect(
            "hello",
            Options(
                model=scripted([TextPart(content="done")]),
                cwd=tmp_path,
                agent_sources=("project",),
            ),
        )
        assert "Agent" in messages[0].tools
        assert messages[0].agents == ["reviewer"]

    async def test_nothing_is_discovered_without_the_source(
        self, tmp_path: Path
    ) -> None:
        write_agent(tmp_path / ".ubiquity" / "agents", "reviewer", "description: d")
        messages = await collect(
            "hello",
            Options(model=scripted([TextPart(content="done")]), cwd=tmp_path),
        )
        assert "Agent" not in messages[0].tools
        assert messages[0].agents == []

    async def test_a_code_definition_overrides_a_discovered_one(
        self, tmp_path: Path
    ) -> None:
        """`Options` is explicit and a file is ambient, so explicit wins."""
        write_agent(
            tmp_path / ".ubiquity" / "agents", "reviewer", "description: from the file"
        )
        seen: dict[str, str] = {}

        def respond(messages, info):
            seen.update(
                {t.name: t.description or "" for t in info.function_tools}
            )
            return ModelResponse(parts=[TextPart(content="done")])

        await collect(
            "hello",
            Options(
                model=FunctionModel(respond),
                cwd=tmp_path,
                agent_sources=("project",),
                agents={
                    "reviewer": AgentDefinition(
                        description="from the code", prompt="do it"
                    )
                },
            ),
        )
        assert "from the code" in seen["Agent"]
        assert "from the file" not in seen["Agent"]

    async def test_a_discovered_agent_can_be_delegated_to(self, tmp_path: Path) -> None:
        write_agent(
            tmp_path / ".ubiquity" / "agents",
            "reviewer",
            "description: Reviews a diff",
            body="Reply with exactly: reviewed",
        )
        model = scripted(
            [
                ToolCallPart(
                    tool_name="Agent",
                    args={
                        "subagent_type": "reviewer",
                        "description": "review it",
                        "prompt": "review the diff",
                    },
                    tool_call_id="c1",
                )
            ],
            [TextPart(content="reviewed")],
        )
        messages = await collect(
            "hello",
            Options(model=model, cwd=tmp_path, agent_sources=("project",)),
        )
        assert messages[-1].subtype == "success"
        assert any(m.type == "assistant" for m in messages)

    async def test_an_unknown_type_is_rejected_against_discovered_names(
        self, tmp_path: Path
    ) -> None:
        write_agent(tmp_path / ".ubiquity" / "agents", "reviewer", "description: d")
        model = scripted(
            [
                ToolCallPart(
                    tool_name="Agent",
                    args={
                        "subagent_type": "nobody",
                        "description": "x",
                        "prompt": "x",
                    },
                    tool_call_id="c1",
                )
            ],
            [TextPart(content="gave up")],
        )
        messages = await collect(
            "hello",
            Options(model=model, cwd=tmp_path, agent_sources=("project",)),
        )
        text = "".join(str(m) for m in messages)
        assert "reviewer" in text
