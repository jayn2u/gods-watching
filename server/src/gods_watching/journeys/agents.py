"""Constrained command runner and Claude CLI adapters for Journey agents."""

import json
import re
import shlex
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, TypeGuard, cast

from pydantic import BaseModel, ValidationError

from .models import (
    AgentFailure,
    ExecutorReport,
    Journey,
    JudgeReport,
    OperatorCredentials,
)
from .prompts import (
    EXECUTOR_SCHEMA,
    JUDGE_SCHEMA,
    build_executor_prompt,
    build_judge_prompt,
)

_USAGE_LIMIT_PATTERN: Final = re.compile(r"usage limit|rate limit|limit reached", re.IGNORECASE)
_EMPTY_MCP_CONFIG: Final[dict[str, object]] = {"mcpServers": {}}


@dataclass(frozen=True, slots=True)
class CompletedCommand:
    """Capture the process result without invoking a shell."""

    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str], Path], CompletedCommand]


class JourneyExecutor(Protocol):
    """Execute a Journey in the isolated browser tool surface."""

    def execute(
        self,
        journey: Journey,
        *,
        base_url: str,
        credentials: OperatorCredentials,
        evidence_dir: Path,
    ) -> ExecutorReport | AgentFailure:
        """Return a structured Journey report or an agent failure."""
        ...


class Judge(Protocol):
    """Independently Cross-check claimed violations against run evidence."""

    def judge(
        self,
        journey: Journey,
        report: ExecutorReport,
        *,
        evidence_dir: Path,
    ) -> JudgeReport | AgentFailure:
        """Return a structured Cross-check report or an agent failure."""
        ...


@dataclass(frozen=True, slots=True)
class ClaudeCliExecutor:
    """Run a constrained Claude Code session with only Playwright MCP access."""

    runner: CommandRunner
    claude_binary: str
    node_binary: str
    playwright_cli_path: Path

    def execute(
        self,
        journey: Journey,
        *,
        base_url: str,
        credentials: OperatorCredentials,
        evidence_dir: Path,
    ) -> ExecutorReport | AgentFailure:
        """Run one browser Journey and parse Claude's structured report."""
        evidence_dir.mkdir(parents=True, exist_ok=True)
        mcp_config_path = evidence_dir / "playwright-mcp.json"
        write_playwright_mcp_config(
            mcp_config_path,
            node_binary=self.node_binary,
            cli_path=self.playwright_cli_path,
            output_dir=evidence_dir / "browser",
        )
        prompt = build_executor_prompt(
            journey,
            base_url=base_url,
            credentials=credentials,
            evidence_dir=evidence_dir,
        )
        command = _replace_executable(
            build_executor_command(prompt, mcp_config_path), self.claude_binary
        )
        return _run_report(self.runner, command, evidence_dir, ExecutorReport)


@dataclass(frozen=True, slots=True)
class ClaudeCliJudge:
    """Run an independent Claude Code session with evidence-only Read access."""

    runner: CommandRunner
    claude_binary: str

    def judge(
        self,
        journey: Journey,
        report: ExecutorReport,
        *,
        evidence_dir: Path,
    ) -> JudgeReport | AgentFailure:
        """Cross-check the executor's claims using only the evidence directory."""
        evidence_dir.mkdir(parents=True, exist_ok=True)
        prompt = build_judge_prompt(journey, report, evidence_dir=evidence_dir)
        command = _replace_executable(
            build_judge_command(prompt, evidence_dir), self.claude_binary
        )
        return _run_report(self.runner, command, evidence_dir, JudgeReport)


def build_executor_command(prompt: str, mcp_config_path: Path) -> tuple[str, ...]:
    """Build the prescribed Claude invocation for the browser executor."""
    return (
        "claude",
        "-p",
        prompt,
        "--model",
        "opus",
        "--restricted",
        "--tools",
        "",
        "--strict-mcp-config",
        "--mcp-config",
        str(mcp_config_path),
        "--allowedTools",
        "mcp__playwright",
        "--permission-mode",
        "dontAsk",
        "--no-session-persistence",
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(EXECUTOR_SCHEMA, separators=(",", ":"), sort_keys=True),
    )


def build_judge_command(prompt: str, evidence_dir: Path) -> tuple[str, ...]:
    """Build the prescribed Claude invocation for an evidence-only judge."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    empty_mcp_path = evidence_dir / "empty-mcp-config.json"
    _ = empty_mcp_path.write_text(json.dumps(_EMPTY_MCP_CONFIG), encoding="utf-8")
    return (
        "claude",
        "-p",
        prompt,
        "--model",
        "opus",
        "--restricted",
        "--tools",
        "Read",
        "--strict-mcp-config",
        "--mcp-config",
        str(empty_mcp_path),
        "--allowedTools",
        "Read",
        "--add-dir",
        str(evidence_dir),
        "--permission-mode",
        "dontAsk",
        "--no-session-persistence",
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(JUDGE_SCHEMA, separators=(",", ":"), sort_keys=True),
    )


def parse_cli_result[T: BaseModel](stdout: str, model: type[T]) -> T | AgentFailure:
    """Parse the Claude JSON envelope into a report or typed agent failure."""
    try:
        decoded = cast("object", json.loads(stdout))
    except json.JSONDecodeError as error:
        return AgentFailure(kind="invalid_output", detail=f"invalid CLI JSON: {error.msg}")
    if not _is_json_object(decoded):
        return AgentFailure(kind="invalid_output", detail="CLI result must be a JSON object")
    envelope = decoded

    if envelope.get("is_error") is True:
        result = envelope.get("result")
        detail = result if isinstance(result, str) else "Claude CLI returned an error."
        kind = "usage_limit" if _USAGE_LIMIT_PATTERN.search(detail) else "cli_error"
        return AgentFailure(kind=kind, detail=detail)

    structured_output = envelope.get("structured_output")
    if not _is_json_object(structured_output):
        return AgentFailure(kind="invalid_output", detail="CLI result has no structured_output")
    try:
        parsed = model.model_validate(structured_output)
        if isinstance(parsed, ExecutorReport):
            parsed = parsed.model_copy(
                update={
                    "observations": tuple(
                        observation.model_copy(update={"flaky": False})
                        for observation in parsed.observations
                    )
                }
            )
    except (ValidationError, TypeError, ValueError):
        return AgentFailure(kind="invalid_output", detail="structured_output is invalid")
    return parsed


def write_playwright_mcp_config(
    path: Path,
    *,
    node_binary: str,
    cli_path: Path,
    output_dir: Path,
) -> None:
    """Write an MCP configuration limited to isolated headless Playwright."""
    config = {
        "mcpServers": {
            "playwright": {
                "command": node_binary,
                "args": [
                    str(cli_path),
                    "--headless",
                    "--isolated",
                    "--browser",
                    "chromium",
                    "--output-dir",
                    str(output_dir),
                ],
            }
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def redacted_command(command: Sequence[str]) -> str:
    """Format a Claude command while masking its prompt and any embedded secrets."""
    arguments = list(command)
    try:
        prompt_index = arguments.index("-p") + 1
    except ValueError:
        pass
    else:
        if prompt_index < len(arguments):
            arguments[prompt_index] = "<prompt>"
    return shlex.join(arguments)


def run_command(command: Sequence[str], cwd: Path) -> CompletedCommand:
    """Run one argument-vector subprocess from its explicit working directory."""
    result = subprocess.run(  # noqa: S603
        tuple(command),
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
    )
    return CompletedCommand(result.returncode, result.stdout, result.stderr)


def _replace_executable(command: tuple[str, ...], binary: str) -> tuple[str, ...]:
    return (binary, *command[1:])


def _run_report[T: BaseModel](
    runner: CommandRunner,
    command: tuple[str, ...],
    cwd: Path,
    model: type[T],
) -> T | AgentFailure:
    try:
        completed = runner(command, cwd)
    except OSError as error:
        return AgentFailure(kind="cli_error", detail=f"Claude CLI could not run: {error}")
    if completed.returncode != 0:
        stdout = completed.stdout.strip()
        if stdout:
            try:
                decoded = cast("object", json.loads(stdout))
            except json.JSONDecodeError:
                decoded = None
            if _is_json_object(decoded):
                parsed = parse_cli_result(stdout, model)
                if isinstance(parsed, AgentFailure):
                    return parsed
        detail = completed.stderr.strip() or f"Claude CLI exited with code {completed.returncode}."
        return AgentFailure(kind="cli_error", detail=detail)
    parsed = parse_cli_result(completed.stdout, model)
    if isinstance(parsed, ExecutorReport) and parsed.verdict == "bug" and not parsed.violations:
        return parsed.model_copy(update={"verdict": "inconclusive"})
    return parsed


def _is_json_object(value: object) -> TypeGuard[dict[str, object]]:
    if not isinstance(value, dict):
        return False
    json_object = cast("dict[object, object]", value)
    return all(isinstance(key, str) for key in json_object)


__all__ = [
    "ClaudeCliExecutor",
    "ClaudeCliJudge",
    "CommandRunner",
    "CompletedCommand",
    "JourneyExecutor",
    "Judge",
    "build_executor_command",
    "build_judge_command",
    "parse_cli_result",
    "redacted_command",
    "run_command",
    "write_playwright_mcp_config",
]
