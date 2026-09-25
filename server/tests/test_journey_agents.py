"""Behavioral tests for the constrained Claude CLI adapters."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import SecretStr

from gods_watching.journeys.agents import (
    ClaudeCliExecutor,
    ClaudeCliJudge,
    CompletedCommand,
    build_executor_command,
    build_judge_command,
    parse_cli_result,
    redacted_command,
    write_playwright_mcp_config,
)
from gods_watching.journeys.models import (
    AgentFailure,
    ExecutorReport,
    ExpectedOutcome,
    Journey,
    JudgeReport,
    Observation,
    OperatorCredentials,
    Violation,
)
from gods_watching.journeys.prompts import EXECUTOR_SCHEMA, JUDGE_SCHEMA


def sample_journey() -> Journey:
    """Return a valid Journey with more than one outcome to check prompt coverage."""
    return Journey(
        id="login",
        title="Sign in and protect operator access",
        setup=(),
        tags=("authentication",),
        preconditions="The operator credentials are supplied by the harness.",
        goal="Use sign-in and sign-out to confirm operator access is protected.",
        expected_outcomes=(
            ExpectedOutcome(id="E1", text="An invalid password is visibly rejected."),
            ExpectedOutcome(id="C1", text="No product request returns an HTTP 5xx status."),
            ExpectedOutcome(id="C2", text="The browser shows no unhandled exception."),
        ),
        source_path=Path("qa/journeys/login.md"),
    )


def report_data(
    *,
    verdict: Literal["pass", "bug", "inconclusive"] = "pass",
    violations: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build one JSON-compatible executor report for a CLI envelope."""
    return {
        "verdict": verdict,
        "violations": violations or [],
        "observations": [{"text": "The console loaded."}],
        "steps": ["Opened the sign-in screen."],
        "summary": "The Journey completed.",
    }


def cli_envelope(structured_output: dict[str, object]) -> str:
    """Serialize the output wrapper returned by Claude Code CLI."""
    return json.dumps({"is_error": False, "structured_output": structured_output})


def test_command_builders_return_the_required_executor_and_judge_arguments(
    tmp_path: Path,
) -> None:
    """Both Claude invocations constrain tools and use their prescribed schema."""
    mcp_path = tmp_path / "playwright-mcp.json"
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()

    executor_command = build_executor_command("EXECUTOR PROMPT", mcp_path)
    judge_command = build_judge_command("JUDGE PROMPT", evidence_dir)
    empty_mcp_path = evidence_dir / "empty-mcp-config.json"

    assert executor_command == (
        "claude",
        "-p",
        "EXECUTOR PROMPT",
        "--model",
        "opus",
        "--restricted",
        "--tools",
        "",
        "--strict-mcp-config",
        "--mcp-config",
        str(mcp_path),
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
    assert judge_command == (
        "claude",
        "-p",
        "JUDGE PROMPT",
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
    assert json.loads(empty_mcp_path.read_text(encoding="utf-8")) == {"mcpServers": {}}


def test_executor_prompt_contains_journey_url_and_credentials_without_log_secret(
    tmp_path: Path,
) -> None:
    """The executor gets task context while the command log redacts its prompt."""
    credential_value = "opaque-test-value-742"
    credentials = OperatorCredentials(username="operator-id", password=SecretStr(credential_value))
    report = report_data()
    calls: list[tuple[tuple[str, ...], Path]] = []

    def runner(command: Sequence[str], cwd: Path) -> CompletedCommand:
        calls.append((tuple(command), cwd))
        return CompletedCommand(
            returncode=0,
            stdout=cli_envelope(report),
            stderr="",
        )

    evidence_dir = tmp_path / "evidence"
    executor = ClaudeCliExecutor(
        runner=runner,
        claude_binary="claude-test",
        node_binary="node-test",
        playwright_cli_path=tmp_path / "playwright-cli.js",
    )

    result = executor.execute(
        sample_journey(),
        base_url="http://127.0.0.1:18080",
        credentials=credentials,
        evidence_dir=evidence_dir,
    )

    assert isinstance(result, ExecutorReport)
    assert result.verdict == "pass"
    command, cwd = calls[0]
    prompt = command[2]
    assert command[0] == "claude-test"
    assert "Use sign-in and sign-out to confirm operator access is protected." in prompt
    assert "E1" in prompt
    assert "C1" in prompt
    assert "C2" in prompt
    assert "http://127.0.0.1:18080" in prompt
    assert "screenshots are saved under" in prompt.lower()
    assert "evidence_files" in prompt
    assert "operator-id" in prompt
    assert credential_value in prompt
    assert cwd == evidence_dir
    mcp_config_path = evidence_dir / "playwright-mcp.json"
    assert json.loads(mcp_config_path.read_text(encoding="utf-8")) == {
        "mcpServers": {
            "playwright": {
                "command": "node-test",
                "args": [
                    str(tmp_path / "playwright-cli.js"),
                    "--headless",
                    "--isolated",
                    "--output-dir",
                    str(evidence_dir / "browser"),
                    "--save-trace",
                ],
            }
        }
    }
    assert str(mcp_config_path) in command
    logged_command = redacted_command(command)
    assert "<prompt>" in logged_command
    assert credential_value not in logged_command


def test_judge_receives_claimed_violations_and_uses_the_evidence_directory(
    tmp_path: Path,
) -> None:
    """The judge's command carries its evidence boundary and its prompt includes claims."""
    report = ExecutorReport(
        verdict="bug",
        violations=(
            Violation(
                outcome_id="E1",
                observed="The invalid password opened the console.",
                evidence_files=("screen.png",),
            ),
        ),
        observations=(),
        steps=(),
        summary="The claimed behavior did not happen.",
    )
    calls: list[tuple[str, ...]] = []

    def runner(command: Sequence[str], cwd: Path) -> CompletedCommand:
        calls.append(tuple(command))
        assert cwd == evidence_dir
        return CompletedCommand(
            returncode=0,
            stdout=cli_envelope({"decisions": []}),
            stderr="",
        )

    evidence_dir = tmp_path / "attempt-1"
    evidence_dir.mkdir()
    _ = (evidence_dir / "screen.png").write_bytes(b"image")
    judge = ClaudeCliJudge(runner=runner, claude_binary="claude-test")

    result = judge.judge(sample_journey(), report, evidence_dir=evidence_dir)

    assert isinstance(result, JudgeReport)
    assert calls[0][0] == "claude-test"
    assert "The invalid password opened the console." in calls[0][2]
    assert "screen.png" in calls[0][2]
    assert "--tools" in calls[0]
    assert calls[0][calls[0].index("--tools") + 1] == "Read"
    assert calls[0][calls[0].index("--add-dir") + 1] == str(evidence_dir)


def test_parse_cli_result_returns_structured_success() -> None:
    """A valid structured output is parsed as its declared report model."""
    parsed = parse_cli_result(cli_envelope(report_data()), ExecutorReport)

    assert parsed == ExecutorReport.model_validate(report_data())


def test_parse_cli_result_classifies_usage_limit_errors() -> None:
    """Subscription limit text becomes the distinct usage-limit failure kind."""
    parsed = parse_cli_result(
        json.dumps({"is_error": True, "result": "Usage limit reached for this account."}),
        ExecutorReport,
    )

    assert parsed == AgentFailure(
        kind="usage_limit", detail="Usage limit reached for this account."
    )


def test_parse_cli_result_classifies_other_cli_errors() -> None:
    """An ordinary CLI envelope error remains separate from invalid output."""
    parsed = parse_cli_result(
        json.dumps({"is_error": True, "result": "Unable to start the browser."}),
        ExecutorReport,
    )

    assert parsed == AgentFailure(kind="cli_error", detail="Unable to start the browser.")


def test_parse_cli_result_rejects_missing_or_invalid_structured_output() -> None:
    """Malformed CLI JSON or schema-invalid output becomes invalid_output."""
    missing = parse_cli_result(json.dumps({"is_error": False}), ExecutorReport)
    invalid = parse_cli_result(
        cli_envelope({"verdict": "surprise", "violations": []}),
        ExecutorReport,
    )

    assert isinstance(missing, AgentFailure)
    assert missing.kind == "invalid_output"
    assert isinstance(invalid, AgentFailure)
    assert invalid.kind == "invalid_output"


def test_executor_classifies_nonzero_exit_with_empty_stdout_as_cli_error(
    tmp_path: Path,
) -> None:
    """A failed process with no structured output is reported as a CLI failure."""
    def runner(command: Sequence[str], cwd: Path) -> CompletedCommand:
        del command, cwd
        return CompletedCommand(returncode=2, stdout="", stderr="CLI could not start")

    executor = ClaudeCliExecutor(
        runner=runner,
        claude_binary="claude",
        node_binary="node",
        playwright_cli_path=Path("playwright-cli.js"),
    )
    result = executor.execute(
        sample_journey(),
        base_url="http://localhost",
        credentials=OperatorCredentials(username="admin", password=SecretStr("test-value")),
        evidence_dir=tmp_path / "evidence",
    )

    assert result == AgentFailure(kind="cli_error", detail="CLI could not start")


def test_executor_classifies_nonzero_exit_with_usage_limit_envelope(
    tmp_path: Path,
) -> None:
    """A structured subscription limit remains recognizable after a failed exit."""
    def runner(command: Sequence[str], cwd: Path) -> CompletedCommand:
        del command, cwd
        return CompletedCommand(
            returncode=1,
            stdout=json.dumps(
                {
                    "is_error": True,
                    "result": "You have reached your usage limit for this account.",
                }
            ),
            stderr="Claude CLI exited with code 1",
        )

    executor = ClaudeCliExecutor(
        runner=runner,
        claude_binary="claude",
        node_binary="node",
        playwright_cli_path=Path("playwright-cli.js"),
    )
    result = executor.execute(
        sample_journey(),
        base_url="http://localhost",
        credentials=OperatorCredentials(username="admin", password=SecretStr("test-value")),
        evidence_dir=tmp_path / "evidence",
    )

    assert result == AgentFailure(
        kind="usage_limit",
        detail="You have reached your usage limit for this account.",
    )


def test_executor_classifies_nonzero_exit_with_normal_report_as_cli_error(
    tmp_path: Path,
) -> None:
    """A report envelope cannot hide a non-zero CLI exit when it is not an error."""
    def runner(command: Sequence[str], cwd: Path) -> CompletedCommand:
        del command, cwd
        return CompletedCommand(
            returncode=1,
            stdout=cli_envelope(report_data()),
            stderr="Claude CLI exited with code 1",
        )

    executor = ClaudeCliExecutor(
        runner=runner,
        claude_binary="claude",
        node_binary="node",
        playwright_cli_path=Path("playwright-cli.js"),
    )
    result = executor.execute(
        sample_journey(),
        base_url="http://localhost",
        credentials=OperatorCredentials(username="admin", password=SecretStr("test-value")),
        evidence_dir=tmp_path / "evidence",
    )

    assert result == AgentFailure(kind="cli_error", detail="Claude CLI exited with code 1")


def test_executor_normalizes_bug_without_violations_to_inconclusive(tmp_path: Path) -> None:
    """A bug verdict cannot exist without a named Expected Outcome violation."""
    def runner(command: Sequence[str], cwd: Path) -> CompletedCommand:
        del command, cwd
        return CompletedCommand(
            returncode=0,
            stdout=cli_envelope(report_data(verdict="bug")),
            stderr="",
        )

    executor = ClaudeCliExecutor(
        runner=runner,
        claude_binary="claude",
        node_binary="node",
        playwright_cli_path=Path("playwright-cli.js"),
    )
    result = executor.execute(
        sample_journey(),
        base_url="http://localhost",
        credentials=OperatorCredentials(username="admin", password=SecretStr("test-value")),
        evidence_dir=tmp_path / "evidence",
    )

    assert isinstance(result, ExecutorReport)
    assert result.verdict == "inconclusive"
    assert result.observations == (Observation(text="The console loaded."),)


def test_playwright_mcp_config_has_the_isolated_headless_server_shape(tmp_path: Path) -> None:
    """The executor is restricted to the configured Playwright MCP server."""
    path = tmp_path / "playwright-mcp.json"

    write_playwright_mcp_config(
        path,
        node_binary="node",
        cli_path=Path("web/node_modules/@playwright/mcp/cli.js"),
        output_dir=tmp_path / "browser-output",
    )

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "mcpServers": {
            "playwright": {
                "command": "node",
                "args": [
                    "web/node_modules/@playwright/mcp/cli.js",
                    "--headless",
                    "--isolated",
                    "--output-dir",
                    str(tmp_path / "browser-output"),
                    "--save-trace",
                ],
            }
        }
    }
