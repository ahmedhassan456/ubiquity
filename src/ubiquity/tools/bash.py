"""The Bash tool.

Runs a shell command with a timeout and truncated output.

The permission story matters here: a shell command is the widest capability the
agent has, so `permission_rule_content` splits compound commands on ``&&``,
``||``, ``;``, and ``|`` and returns every segment. The permission engine then
requires a rule to match *each* segment, which is what stops
``Bash(git:*)`` from authorizing ``git status && rm -rf /``.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex

from pydantic import BaseModel, Field

from ..tool import Tool, ToolContext, ValidationError
from ..types import PermissionResult, PermissionResultAsk, ToolOutput

DEFAULT_TIMEOUT_MS = 120_000
MAX_TIMEOUT_MS = 600_000
MAX_OUTPUT_CHARS = 30_000

BANNED_COMMANDS = {
    "curl", "wget", "httpie", "http", "nc", "netcat", "telnet",
    "ssh", "scp", "sftp", "rsync", "ftp",
}

BANNED_COMMAND_LIST = ", ".join(sorted(BANNED_COMMANDS))

_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||;|\|)\s*")


class Aborted(Exception):
    """Raised inside this module when the run's abort event fires."""


async def _terminate(proc: asyncio.subprocess.Process) -> None:
    """Stop `proc` and reap it, so no orphan outlives the tool call.

    `kill` alone leaves a zombie until the event loop happens to reap it, and
    the pipes stay open with it. Waiting is what actually releases both.
    """
    if proc.returncode is None:
        proc.kill()
    await proc.wait()


async def _communicate(
    proc: asyncio.subprocess.Process,
    timeout_s: float,
    abort: asyncio.Event | None,
) -> tuple[bytes, bytes]:
    """Collect the process output, raising on timeout or abort.

    With no abort event this is `asyncio.wait_for` and nothing more, so a run
    that never sets one behaves exactly as before.
    """
    if abort is None:
        return await asyncio.wait_for(proc.communicate(), timeout=timeout_s)

    output = asyncio.ensure_future(proc.communicate())
    cancelled = asyncio.ensure_future(abort.wait())
    try:
        done, _ = await asyncio.wait(
            {output, cancelled},
            timeout=timeout_s,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if output in done:
            return output.result()
        if cancelled in done:
            raise Aborted
        raise asyncio.TimeoutError
    finally:
        for task in (output, cancelled):
            if not task.done():
                task.cancel()


class BashInput(BaseModel):
    command: str = Field(description="The shell command to execute.")
    description: str = Field(
        default="", description="Short active-voice description of what it does."
    )
    timeout: int | None = Field(
        default=None, description=f"Timeout in ms, max {MAX_TIMEOUT_MS}."
    )


def split_command(command: str) -> list[str]:
    """Split a compound shell command into its individual segments.

    Used for permission matching so that a rule authorizing one command cannot
    smuggle in a second one via ``&&`` or a pipe.
    """
    return [seg.strip() for seg in _SPLIT_RE.split(command.strip()) if seg.strip()]


def is_compound(command: str) -> bool:
    """True when the command chains more than one invocation."""
    return len(split_command(command)) > 1


def base_command(segment: str) -> str:
    """Return the executable name from a command segment."""
    try:
        parts = shlex.split(segment)
    except ValueError:
        parts = segment.split()
    return os.path.basename(parts[0]) if parts else ""


class BashTool(Tool[BashInput]):
    """Execute a shell command."""

    name = "Bash"
    description = (
        "Execute a bash command. Every call starts in the run's working "
        "directory: a `cd` does not carry over to the next call, and neither "
        "does any other shell state such as an environment variable, so prefer "
        "absolute paths. Prefer the dedicated file tools over cat, sed, and "
        "echo. Every segment of a command chained with `&&`, `||`, `;`, or a "
        "pipe must be permitted on its own, so two calls are more likely to be "
        "approved than one chain. Networking commands are refused: "
        f"{BANNED_COMMAND_LIST}. Output over {MAX_OUTPUT_CHARS} characters is "
        "truncated from the middle. Timeout is in "
        f"milliseconds, default {DEFAULT_TIMEOUT_MS}, max {MAX_TIMEOUT_MS}."
    )
    input_model = BashInput
    search_hint = "shell command terminal execute run"
    max_result_chars = MAX_OUTPUT_CHARS

    def permission_rule_content(self, args: BashInput) -> list[str]:
        return split_command(args.command)

    def describe_call(self, args: BashInput) -> str:
        return args.description or args.command.splitlines()[0][:80]

    async def validate_input(
        self, args: BashInput, ctx: ToolContext
    ) -> ValidationError | None:
        if not args.command.strip():
            return ValidationError(message="`command` is empty.")
        if args.timeout is not None and args.timeout > MAX_TIMEOUT_MS:
            return ValidationError(
                message=f"`timeout` exceeds the maximum of {MAX_TIMEOUT_MS}ms."
            )
        for segment in split_command(args.command):
            if (name := base_command(segment)) in BANNED_COMMANDS:
                return ValidationError(
                    message=f"`{name}` is not permitted: this tool has no network access."
                )
        return None

    async def check_permissions(
        self, args: BashInput, ctx: ToolContext
    ) -> PermissionResult:
        """Require every segment of a compound command to be separately allowed.

        The engine's content matching already refuses to let a prefix rule
        match a compound command. This returns an explicit ask so a partially
        matched chain surfaces a prompt rather than falling through.
        """
        return PermissionResultAsk(
            message=f"Run shell command: {args.command}",
        )

    async def call(self, args: BashInput, ctx: ToolContext) -> ToolOutput:
        """Run the command, honoring both the timeout and the run's abort.

        The abort is raced against the command rather than checked around it,
        because a shell command is the one tool that can block for its whole
        timeout with nothing to poll. Cancelling this coroutine alone would
        leave the process running with nobody holding its handle, so an abort
        kills the process first and reports the run as interrupted after.
        """
        timeout_s = (args.timeout or DEFAULT_TIMEOUT_MS) / 1000

        proc = await asyncio.create_subprocess_shell(
            args.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(ctx.cwd),
            env={**os.environ, **ctx.options.env},
        )

        try:
            stdout_b, stderr_b = await _communicate(proc, timeout_s, ctx.abort)
        except asyncio.TimeoutError:
            await _terminate(proc)
            return ToolOutput(
                content=f"Command timed out after {timeout_s:.0f}s.",
                is_error=True,
                metadata={"timed_out": True, "command": args.command},
            )
        except Aborted:
            await _terminate(proc)
            from ..toolset import ToolDenied

            raise ToolDenied(
                f"Run aborted during: {args.command}", interrupt=True
            ) from None

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        code = proc.returncode or 0

        body = stdout
        if stderr.strip():
            body = f"{body}\n{stderr}" if body.strip() else stderr

        truncated = len(body) > MAX_OUTPUT_CHARS
        if truncated:
            head = MAX_OUTPUT_CHARS // 2
            body = (
                body[:head]
                + f"\n\n[... {len(body) - MAX_OUTPUT_CHARS} characters truncated ...]\n\n"
                + body[-head:]
            )

        if code != 0:
            body = f"{body}\n[exit code {code}]" if body.strip() else f"[exit code {code}]"

        return ToolOutput(
            content=body.strip() or "(no output)",
            is_error=code != 0,
            metadata={
                "exit_code": code,
                "command": args.command,
                "truncated": truncated,
            },
        )


__all__ = ["BashTool", "BashInput", "split_command", "is_compound", "base_command"]
