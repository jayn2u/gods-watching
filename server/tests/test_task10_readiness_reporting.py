import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import anyio
import pytest

from gods_watching.verification import (
    ImplementedScenario,
    RunResult,
    ScenarioContextProtocol,
    ScenarioReport,
    execute_scenario,
    parse_scenario_name,
)
from gods_watching.verification.scenarios import task10_runtime
from gods_watching.verification.scenarios.task10_errors import Task10ExecutionError
from gods_watching.verification.scenarios.task10_runtime import CommandResult

REPOSITORY_ROOT = Path(__file__).parents[2]
type Runner = Callable[[ScenarioContextProtocol], Awaitable[ScenarioReport]]


async def _setup_failure_runner(context: ScenarioContextProtocol) -> ScenarioReport:
    async with context.process(
        name="task10-failure-boundary", command=(sys.executable, "-c", "pass")
    ) as process:
        _ = await process.wait()
    raise Task10ExecutionError(detail="fixture publishers did not expose all RTSP streams")


def test_task10_setup_failure_emits_structured_result_and_cleanup(tmp_path: Path) -> None:
    evidence = tmp_path / "task10-setup-failure"
    definition = ImplementedScenario(
        name=parse_scenario_name("task10-error-probe"),
        runner=_setup_failure_runner,
        required_commands=(sys.executable,),
        timeout_seconds=2.0,
    )

    result = execute_scenario(
        scenario="task10-error-probe",
        evidence_dir=evidence,
        repository_root=REPOSITORY_ROOT,
        additional=(definition,),
    )

    persisted = RunResult.model_validate_json((evidence / "result.json").read_text())
    manifest = cast(
        "dict[str, object]", json.loads((evidence / "resource-manifest.json").read_text())
    )
    events = cast("list[dict[str, str]]", manifest["events"])
    lifecycle = [event["state"] for event in events if event["name"] == "task10-failure-boundary"]
    assert result.exit_code == 1
    assert persisted.outcome == "failed"
    assert persisted.error is not None
    assert persisted.error.code == "runner_error"
    assert persisted.error.message == "fixture publishers did not expose all RTSP streams"
    assert "super(type, obj)" not in (evidence / "result.json").read_text()
    assert lifecycle == ["registered", "started", "cleaned"]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_fixture_readiness_retries_failed_streams_and_requires_h264(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = {
        1: [
            CommandResult(return_code=0, stdout="hevc\n", stderr=""),
            CommandResult(return_code=0, stdout="h264\n", stderr=""),
        ],
        2: [
            CommandResult(return_code=1, stdout="", stderr="no stream\n"),
            CommandResult(return_code=0, stdout="h264\n", stderr=""),
        ],
        3: [
            CommandResult(return_code=0, stdout="h264\n", stderr=""),
            CommandResult(return_code=0, stdout="h264\n", stderr=""),
        ],
        4: [
            CommandResult(return_code=0, stdout="", stderr=""),
            CommandResult(return_code=0, stdout="h264\n", stderr=""),
        ],
    }
    calls: list[tuple[str, ...]] = []

    async def fake_run_command(
        context: ScenarioContextProtocol, *, name: str, command: tuple[str, ...]
    ) -> CommandResult:
        del context
        calls.append((name, *command))
        camera_index = int(name.rsplit("-", maxsplit=1)[-1])
        values = responses[camera_index]
        return values.pop(0)

    monkeypatch.setattr(task10_runtime, "run_command", fake_run_command)

    ready = await task10_runtime.wait_fixture_streams(
        cast("ScenarioContextProtocol", object()), rtsp_host="172.22.0.2"
    )

    assert ready is True
    assert [call[0] for call in calls] == [
        "task10-fixture-probe-1",
        "task10-fixture-probe-2",
        "task10-fixture-probe-3",
        "task10-fixture-probe-4",
        "task10-fixture-probe-1",
        "task10-fixture-probe-2",
        "task10-fixture-probe-3",
        "task10-fixture-probe-4",
    ]
    assert all(call[1] == "ffprobe" for call in calls)
    assert all("-rtsp_transport" in call and "tcp" in call for call in calls)


@pytest.mark.anyio
async def test_fixture_readiness_restarts_the_sweep_after_a_stale_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = 0.0
    responses = {
        1: [
            CommandResult(return_code=0, stdout="h264\n", stderr=""),
            CommandResult(return_code=1, stdout="", stderr="unavailable\n"),
        ],
        2: [
            CommandResult(return_code=1, stdout="", stderr="unavailable\n"),
            CommandResult(return_code=0, stdout="h264\n", stderr=""),
        ],
        3: [CommandResult(return_code=0, stdout="h264\n", stderr="")] * 2,
        4: [CommandResult(return_code=0, stdout="h264\n", stderr="")] * 2,
    }
    calls: list[int] = []
    sleep_count = 0

    def fake_current_time() -> float:
        return clock

    async def fake_sleep(seconds: float) -> None:
        nonlocal clock, sleep_count
        del seconds
        sleep_count += 1
        clock += 12.0 if sleep_count == 1 else 20.0

    async def fake_run_command(
        context: ScenarioContextProtocol, *, name: str, command: tuple[str, ...]
    ) -> CommandResult:
        nonlocal clock
        del context, command
        camera_index = int(name.rsplit("-", maxsplit=1)[-1])
        calls.append(camera_index)
        clock += 0.5
        values = responses[camera_index]
        return values.pop(0) if values else CommandResult(1, "", "unavailable\n")

    monkeypatch.setattr(anyio, "current_time", fake_current_time)
    monkeypatch.setattr(anyio, "sleep", fake_sleep)
    monkeypatch.setattr(task10_runtime, "run_command", fake_run_command)

    ready = await task10_runtime.wait_fixture_streams(
        cast("ScenarioContextProtocol", object()), rtsp_host="172.22.0.2"
    )

    assert ready is False
    assert calls[:8] == [1, 2, 3, 4, 1, 2, 3, 4]
    assert calls.count(1) == 2


@pytest.mark.anyio
async def test_fixture_readiness_reserves_probe_cleanup_before_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = 0.0
    calls: list[int] = []

    def fake_current_time() -> float:
        return clock

    async def fake_probe(
        context: ScenarioContextProtocol, *, camera_index: int, rtsp_host: str
    ) -> CommandResult | None:
        nonlocal clock
        del context, rtsp_host
        calls.append(camera_index)
        clock += 2.0 + 2.0
        return None

    monkeypatch.setattr(anyio, "current_time", fake_current_time)
    monkeypatch.setattr(task10_runtime, "_probe_fixture_stream", fake_probe)

    ready = await task10_runtime.wait_fixture_streams(
        cast("ScenarioContextProtocol", object()), rtsp_host="172.22.0.2"
    )

    assert ready is False
    assert calls == [1, 2, 3, 4]
    assert clock <= 20.0
