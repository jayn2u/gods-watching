"""Typed public contracts for verification scenario authors."""

from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Literal, NewType, Protocol, override

from anyio.abc import Process
from pydantic import BaseModel, ConfigDict

RunId = NewType("RunId", str)
ScenarioName = NewType("ScenarioName", str)


class EvidenceKind(StrEnum):
    """Describe whether a check observed real or synthetic inputs."""

    REAL = "real"
    SYNTHETIC = "synthetic"


class VerificationModel(BaseModel):
    """Apply the immutable evidence schema policy."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class Check(VerificationModel):
    """Record one binary, machine-consumed scenario assertion."""

    name: str
    passed: bool
    detail: str


class ScenarioReport(VerificationModel):
    """Return the assertions and artifacts produced by one scenario."""

    checks: tuple[Check, ...]
    artifact_paths: tuple[Path, ...] = ()
    evidence_kind: EvidenceKind


class ScenarioContextProtocol(Protocol):
    """Structural surface kept stable for scenario modules owned by later tasks."""

    run_id: RunId
    run_root: Path
    runtime_root: Path
    compose_project: str
    allocated_port: int

    def process(self, *, name: str, command: Sequence[str]) -> AbstractAsyncContextManager[Process]:
        """Own one subprocess until its asynchronous context exits."""
        ...

    def suppress_interruptions(self) -> AbstractContextManager[None]:
        """Block terminal signals until owned cleanup commands finish."""
        ...

    def interrupt_cleanup(  # noqa: D102
        self, operation: Callable[[], Awaitable[None]]
    ) -> AbstractContextManager[None]: ...


ScenarioRunner = Callable[["ScenarioContextProtocol"], Awaitable[ScenarioReport]]


@dataclass(frozen=True, slots=True)
class ImplementedScenario:
    """Bind a registered name to a deadline-bounded scenario runner."""

    name: ScenarioName
    runner: ScenarioRunner
    required_commands: tuple[str, ...] = ()
    timeout_seconds: float = 30.0
    owner_task: None = None

    async def invoke(self, context: ScenarioContextProtocol) -> ScenarioReport:
        """Execute the registered runner through the common definition surface."""
        return await self.runner(context)


@dataclass(frozen=True, slots=True)
class UnavailableScenario:
    """Keep a planned scenario visible until its owner implements it."""

    name: ScenarioName
    owner_task: int
    required_commands: tuple[str, ...] = ()
    timeout_seconds: float = 0.0

    async def invoke(self, context: ScenarioContextProtocol) -> ScenarioReport:
        """Prevent direct invocation of a planned placeholder."""
        del context
        raise UnavailableInvocationError(owner_task=self.owner_task)


@dataclass(frozen=True, slots=True)
class UnavailableInvocationError(RuntimeError):
    """Protect the common scenario interface from placeholder execution."""

    owner_task: int

    @override
    def __str__(self) -> str:
        """Identify the owning task without claiming a runtime failure."""
        return f"scenario awaits implementation by task {self.owner_task}"


type ScenarioDefinition = ImplementedScenario | UnavailableScenario


class ErrorRecord(VerificationModel):
    """Describe a stable verification failure without leaking secrets."""

    code: str
    message: str


class SourceHash(VerificationModel):
    """Bind an evidence result to one source file's bytes."""

    path: str
    sha256: str


class RunResult(VerificationModel):
    """Machine-readable terminal result emitted for every accepted run."""

    schema_version: str
    harness_version: str
    run_id: RunId
    scenario: str
    outcome: Literal["passed", "failed"]
    exit_code: int
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    git_sha: str
    dirty_diff_sha256: str
    source_hashes: tuple[SourceHash, ...]
    run_root: Path
    compose_project: str
    allocated_port: int
    checks: tuple[Check, ...]
    artifact_paths: tuple[Path, ...]
    error: ErrorRecord | None
    evidence_kind: EvidenceKind
