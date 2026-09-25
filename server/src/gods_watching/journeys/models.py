"""Immutable domain types shared by Journey verification modules."""

from pathlib import Path
from typing import Annotated, ClassVar, Literal, NoReturn, override

from pydantic import BaseModel, ConfigDict, Field, SecretStr

SetupStep = Literal["login", "cameras"]
Verdict = Literal["pass", "bug", "inconclusive"]
_OutcomeId = Annotated[str, Field(pattern=r"^(E[1-9][0-9]*|C[12])$")]


class JourneyModel(BaseModel):
    """Apply the immutable schema policy for Journey domain records."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class ExpectedOutcome(JourneyModel):
    """Describe an observable condition that defines a Journey's pass verdict."""

    id: _OutcomeId
    text: str


class Journey(JourneyModel):
    """Describe one user-facing task and its observable normal outcomes."""

    id: str
    title: str
    setup: tuple[SetupStep, ...]
    tags: tuple[str, ...]
    preconditions: str
    goal: str
    expected_outcomes: tuple[ExpectedOutcome, ...]
    source_path: Path


class Violation(JourneyModel):
    """Record one observed violation of a named Expected Outcome."""

    outcome_id: str
    observed: str
    evidence_files: tuple[str, ...]


class Observation(JourneyModel):
    """Record a notable behavior that does not change the Verdict."""

    text: str
    flaky: bool = False


class ExecutorReport(JourneyModel):
    """Capture an executor's Verdict, Expected Outcome claims, and evidence notes."""

    verdict: Verdict
    violations: tuple[Violation, ...]
    observations: tuple[Observation, ...]
    steps: tuple[str, ...]
    summary: str


class JudgeDecision(JourneyModel):
    """Record an evidence-only Cross-check decision for one outcome ID."""

    outcome_id: str
    decision: Literal["bug", "not_bug"]
    reason: str


class JudgeReport(JourneyModel):
    """Collect all decisions made during an independent Cross-check."""

    decisions: tuple[JudgeDecision, ...]


class AgentFailure(JourneyModel):
    """Describe why a Claude CLI agent did not return a valid report."""

    kind: Literal["usage_limit", "cli_error", "invalid_output"]
    detail: str


class OperatorCredentials(JourneyModel):
    """Hold operator credentials while masking the password in representations."""

    username: str
    password: SecretStr


class BugReport(JourneyModel):
    """Capture a Cross-check-confirmed violation for human review."""

    journey_id: str
    outcomes: tuple[ExpectedOutcome, ...]
    observed: tuple[Violation, ...]
    judge_reasons: tuple[str, ...]
    fingerprint: str


class JourneyRun(JourneyModel):
    """Record one Journey's terminal Verdict, attempts, and Observations."""

    journey_id: str
    verdict: Verdict
    bug_report: BugReport | None
    observations: tuple[Observation, ...]
    attempts: int
    failure: str | None


class StackError(RuntimeError):
    """Report a stack preparation or evidence collection failure."""

    message: str

    def __init__(self, message: str) -> None:
        """Retain the safe stack failure message for the Journey Run."""
        super().__init__(message)
        self.message = message

    @override
    def __str__(self) -> str:
        """Return the stack failure message used in the Journey Run."""
        return self.message


def raise_stack_error(message: str, *, cause: BaseException | None = None) -> NoReturn:
    """Raise a stack failure through one message-safe domain boundary."""
    if cause is None:
        raise StackError(message)
    raise StackError(message) from cause


COMMON_OUTCOMES: tuple[ExpectedOutcome, ...] = (
    ExpectedOutcome(id="C1", text="No product request returns an HTTP 5xx status."),
    ExpectedOutcome(
        id="C2",
        text="The browser shows no unhandled exception and no error-level console message.",
    ),
)

__all__ = [
    "COMMON_OUTCOMES",
    "AgentFailure",
    "BugReport",
    "ExecutorReport",
    "ExpectedOutcome",
    "Journey",
    "JourneyModel",
    "JourneyRun",
    "JudgeDecision",
    "JudgeReport",
    "Observation",
    "OperatorCredentials",
    "SetupStep",
    "StackError",
    "Verdict",
    "Violation",
    "raise_stack_error",
]
