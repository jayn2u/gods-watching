from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import final, override

import anyio
import pytest
from anyio import EndOfStream
from anyio.abc import ByteReceiveStream, ByteSendStream, Process

from gods_watching.verification import ImplementedScenario, build_registry, parse_scenario_name
from gods_watching.verification.models import RunId, ScenarioContextProtocol
from gods_watching.verification.scenarios.task10 import cleanup_check
from gods_watching.verification.scenarios.task10_artifacts import load_identity_observation
from gods_watching.verification.scenarios.task10_driver import build_toggle_evidence
from gods_watching.verification.scenarios.task10_models import CleanupEvidence
from gods_watching.verification.scenarios.task10_runtime import CommandResult, inspect_resources


@final
class _Stream(ByteReceiveStream):
    def __init__(self, payload: str) -> None:
        self._payload: bytes = payload.encode()
        self._sent: bool = False

    @override
    async def receive(self, max_bytes: int = 65536) -> bytes:
        del max_bytes
        if self._sent:
            raise EndOfStream
        self._sent = True
        return self._payload

    @override
    async def aclose(self) -> None:
        pass


@final
class _Process(Process):
    def __init__(self, return_code: int, stdout: str, stderr: str) -> None:
        self._return_code: int = return_code
        self._stdout: ByteReceiveStream = _Stream(stdout)
        self._stderr: ByteReceiveStream = _Stream(stderr)

    @override
    async def wait(self) -> int:
        return self._return_code

    @override
    def terminate(self) -> None:
        pass

    @override
    def kill(self) -> None:
        pass

    @override
    def send_signal(self, signal: int) -> None:
        del signal

    @override
    async def aclose(self) -> None:
        pass

    @property
    @override
    def pid(self) -> int:
        return 1

    @property
    @override
    def returncode(self) -> int | None:
        return self._return_code

    @property
    @override
    def stdin(self) -> ByteSendStream | None:
        return None

    @property
    @override
    def stdout(self) -> ByteReceiveStream | None:
        return self._stdout

    @property
    @override
    def stderr(self) -> ByteReceiveStream | None:
        return self._stderr


@final
class _InspectContext:
    run_id: RunId = RunId("test")
    run_root: Path = Path()
    runtime_root: Path = Path()
    compose_project: str = "test-project"
    allocated_port: int = 9999

    def __init__(self, responses: dict[str, _Process]) -> None:
        self.responses: dict[str, _Process] = responses
        self.commands: list[tuple[str, ...]] = []

    @asynccontextmanager
    async def process(self, *, name: str, command: Sequence[str]) -> AsyncIterator[Process]:
        del name
        self.commands.append(tuple(command))
        key = "label" if any(part.startswith("label=") for part in command) else "name"
        yield self.responses[key]

    @contextmanager
    def suppress_interruptions(self) -> Iterator[None]:
        yield

    @contextmanager
    def interrupt_cleanup(self, operation: Callable[[], Awaitable[None]]) -> Iterator[None]:
        del operation
        yield


def _inspect(context: ScenarioContextProtocol) -> CommandResult:
    async def run() -> CommandResult:
        return await inspect_resources(context, project="test-project", triton="test-triton")

    return anyio.run(run)


def test_inspect_resources_unions_label_and_name_ownership_queries() -> None:
    context = _InspectContext(
        {
            "label": _Process(0, "test-project-camera-1 Up image\n", ""),
            "name": _Process(0, "test-triton Exited image\n", ""),
        }
    )

    result = _inspect(context)

    assert result.return_code == 0
    assert result.stdout.splitlines() == [
        "test-project-camera-1 Up image",
        "test-triton Exited image",
    ]
    assert len(context.commands) == 2


def test_inspect_resources_failure_is_preserved() -> None:
    context = _InspectContext(
        {
            "label": _Process(0, "test-project-camera-1 Up image\n", ""),
            "name": _Process(17, "", "docker unavailable\n"),
        }
    )

    result = _inspect(context)

    assert result.return_code == 17
    assert "docker unavailable" in result.stderr


@pytest.mark.parametrize(
    ("triton", "compose", "inspect", "remaining", "expected"),
    [
        (0, 0, 0, "", True),
        (1, 0, 0, "", False),
        (0, 1, 0, "", False),
        (0, 0, 1, "", False),
        (0, 0, 0, "test-triton Exited image", False),
        (None, None, 0, "", True),
    ],
)
def test_cleanup_check_requires_every_cleanup_observation(
    triton: int | None,
    compose: int | None,
    inspect: int | None,
    remaining: str,
    expected: bool,
) -> None:
    evidence = CleanupEvidence(
        triton_remove_exit_code=triton,
        compose_down_exit_code=compose,
        inspect_return_code=inspect,
        remaining_owned_resources=remaining,
    )

    assert cleanup_check(evidence).passed is expected


def test_toggle_evidence_uses_only_the_disabled_interval() -> None:
    evidence = build_toggle_evidence(
        disabled_at=10.0,
        reenabled_at=12.0,
        calls_before=40,
        calls_at_disable=41,
        calls_at_reenable=41,
        calls_after=45,
        frames_at_disable=100,
        frames_at_reenable=108,
        end_events=("end",),
        handoffs_after_reenable=2,
    )

    assert evidence.disabled_for_seconds == 2.0
    assert evidence.detector_calls_during_disabled == 0
    assert evidence.detector_calls_after_reenable == 4
    assert evidence.decoded_frames_while_disabled == 8


def test_task10_scenarios_are_installed_real_scenarios() -> None:
    # Given: the default installed verification registry
    registry = build_registry()

    # When: the two original Task 10 command names are resolved
    ingest = registry.get(parse_scenario_name("ingest"))
    outage = registry.get(parse_scenario_name("ingest-outage"))

    # Then: both names are implemented and require the real runtime surface
    assert isinstance(ingest, ImplementedScenario)
    assert isinstance(outage, ImplementedScenario)
    assert {"docker", "ffmpeg", "uv"} <= set(ingest.required_commands)
    assert {"docker", "ffmpeg", "uv"} <= set(outage.required_commands)
    assert ingest.timeout_seconds >= 60.0
    assert outage.timeout_seconds >= 60.0


def test_identity_observation_is_rebound_to_current_sources_and_fixtures() -> None:
    # Given: the independently confirmed human-labeled real-footage observation
    repository_root = Path(__file__).resolve().parents[2]

    # When: Task 10 imports it into current runtime evidence
    observation = load_identity_observation(repository_root)

    # Then: its direct count and denominator are current-source bound
    assert observation.method == "human-labeled-retained-frame-review"
    assert observation.identity_switches_counted == 0
    assert observation.matched_observations == 46
    assert observation.adjacent_comparison_opportunities == 40
    assert observation.reviewed_clips == 4
    assert observation.reviewed_sample_frames == 32
    assert observation.all_source_hashes_current is True
    assert observation.all_fixture_hashes_current is True
    assert observation.unknown_outside_denominator is True
