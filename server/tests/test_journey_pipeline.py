"""Behavioral tests for retry, Cross-check, and Verdict aggregation."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import SecretStr

from gods_watching.journeys.models import (
    AgentFailure,
    ExecutorReport,
    ExpectedOutcome,
    Journey,
    JourneyRun,
    JudgeDecision,
    JudgeReport,
    Observation,
    OperatorCredentials,
    StackError,
    Verdict,
    Violation,
)
from gods_watching.journeys.pipeline import fingerprint, run_journeys


def sample_journey(journey_id: str = "sample") -> Journey:
    """Return a Journey with two independent Expected Outcomes."""
    return Journey(
        id=journey_id,
        title=f"{journey_id} Journey",
        setup=(),
        tags=(),
        preconditions="The application is running.",
        goal="Confirm the expected application behavior.",
        expected_outcomes=(
            ExpectedOutcome(id="E1", text="The first state is visible."),
            ExpectedOutcome(id="E2", text="The second state is visible."),
        ),
        source_path=Path(f"{journey_id}.md"),
    )


def executor_report(
    verdict: Verdict,
    *outcome_ids: str,
    observations: tuple[Observation, ...] = (),
) -> ExecutorReport:
    """Make one report whose violation IDs are explicit in the test case."""
    return ExecutorReport(
        verdict=verdict,
        violations=tuple(
            Violation(
                outcome_id=outcome_id,
                observed=f"Observed failure for {outcome_id}.",
                evidence_files=(f"{outcome_id}.png",),
            )
            for outcome_id in outcome_ids
        ),
        observations=observations,
        steps=("Opened the application.",),
        summary=f"Executor verdict: {verdict}.",
    )


class RecordingStack:
    """Record public stack interactions and optionally fail preparation."""

    errors: list[StackError | None]
    evidence_error: StackError | None

    def __init__(
        self,
        errors: Sequence[StackError | None] = (),
        evidence_error: StackError | None = None,
    ) -> None:
        self.errors = list(errors)
        self.evidence_error = evidence_error
        self.prepared: list[str] = []
        self.evidence_dirs: list[Path] = []

    def prepare_journey(self, journey: Journey) -> None:
        self.prepared.append(journey.id)
        error = self.errors.pop(0) if self.errors else None
        if error is not None:
            raise error

    def collect_evidence(self, evidence_dir: Path) -> None:
        self.evidence_dirs.append(evidence_dir)
        if self.evidence_error is not None:
            raise self.evidence_error


class RecordingExecutor:
    """Return the next configured executor result for each Journey attempt."""

    results: list[ExecutorReport | AgentFailure | StackError]

    def __init__(self, results: Sequence[ExecutorReport | AgentFailure | StackError]) -> None:
        self.results = list(results)
        self.evidence_dirs: list[Path] = []

    def execute(
        self,
        journey: Journey,
        *,
        base_url: str,
        credentials: OperatorCredentials,
        evidence_dir: Path,
    ) -> ExecutorReport | AgentFailure:
        del journey, base_url, credentials
        self.evidence_dirs.append(evidence_dir)
        result = self.results.pop(0)
        if isinstance(result, StackError):
            raise result
        return result


class RecordingJudge:
    """Return the next configured Cross-check result."""

    results: list[JudgeReport | AgentFailure]

    def __init__(self, results: Sequence[JudgeReport | AgentFailure]) -> None:
        self.results = list(results)
        self.evidence_dirs: list[Path] = []
        self.reports: list[ExecutorReport] = []

    def judge(
        self,
        journey: Journey,
        report: ExecutorReport,
        *,
        evidence_dir: Path,
    ) -> JudgeReport | AgentFailure:
        del journey
        self.reports.append(report)
        self.evidence_dirs.append(evidence_dir)
        return self.results.pop(0)


def run_one(
    journey: Journey,
    stack: RecordingStack,
    executor: RecordingExecutor,
    judge: RecordingJudge,
    run_dir: Path,
) -> JourneyRun:
    """Run one Journey with inert local fakes and operator credentials."""
    return run_journeys(
        (journey,),
        stack=stack,
        executor=executor,
        judge=judge,
        base_url="http://localhost:18080",
        credentials=OperatorCredentials(username="admin", password=SecretStr("test-value")),
        run_dir=run_dir,
    )[0]


@pytest.mark.parametrize(
    ("executor_result", "expected_verdict"),
    [
        (executor_report("pass", observations=(Observation(text="Minor delay."),)), "pass"),
        (executor_report("inconclusive"), "inconclusive"),
    ],
    ids=("pass", "inconclusive"),
)
def test_executor_terminal_verdict_returns_without_retry_or_judge(
    tmp_path: Path,
    executor_result: ExecutorReport,
    expected_verdict: Verdict,
) -> None:
    """A non-bug executor Verdict needs one attempt and no Cross-check."""
    stack = RecordingStack()
    executor = RecordingExecutor([executor_result])
    judge = RecordingJudge([])

    result = run_one(sample_journey(), stack, executor, judge, tmp_path)

    assert result.verdict == expected_verdict
    assert result.attempts == 1
    assert result.observations == executor_result.observations
    assert len(executor.evidence_dirs) == 1
    assert judge.reports == []


@pytest.mark.parametrize(
    ("second_report", "expected_observation"),
    [
        (executor_report("pass"), "flaky: E1 not reproduced"),
        (executor_report("bug", "E2"), "flaky: E1 not reproduced"),
    ],
    ids=("not-reproduced", "disjoint-ids"),
)
def test_bug_that_is_not_reproduced_becomes_a_flaky_observation(
    tmp_path: Path,
    second_report: ExecutorReport,
    expected_observation: str,
) -> None:
    """A second attempt must share an outcome ID before Cross-check begins."""
    stack = RecordingStack()
    executor = RecordingExecutor([executor_report("bug", "E1"), second_report])
    judge = RecordingJudge([])

    result = run_one(sample_journey(), stack, executor, judge, tmp_path)

    assert result.verdict == "pass"
    assert result.attempts == 2
    assert result.bug_report is None
    assert Observation(text=expected_observation, flaky=True) in result.observations
    assert judge.reports == []


def test_reproduced_bug_is_cross_checked_and_reported(tmp_path: Path) -> None:
    """A repeated and confirmed violation becomes a Bug Report with a fingerprint."""
    stack = RecordingStack()
    executor_reports = [executor_report("bug", "E1"), executor_report("bug", "E1")]
    judge_report = JudgeReport(
        decisions=(
            JudgeDecision(
                outcome_id="E1",
                decision="bug",
                reason="The screenshot confirms it.",
            ),
        )
    )
    executor = RecordingExecutor(executor_reports)
    judge = RecordingJudge([judge_report])

    result = run_one(sample_journey("login"), stack, executor, judge, tmp_path)

    assert result.verdict == "bug"
    assert result.attempts == 2
    assert result.bug_report is not None
    assert result.bug_report.journey_id == "login"
    assert tuple(outcome.id for outcome in result.bug_report.outcomes) == ("E1",)
    assert tuple(violation.outcome_id for violation in result.bug_report.observed) == ("E1",)
    assert result.bug_report.judge_reasons == ("The screenshot confirms it.",)
    assert result.bug_report.fingerprint == fingerprint("login", ("E1",))
    assert judge.reports[0].violations[0].outcome_id == "E1"
    for attempt, report in enumerate(executor_reports, start=1):
        report_path = tmp_path / "login" / f"attempt-{attempt}" / "executor-report.json"
        assert json.loads(report_path.read_text(encoding="utf-8")) == report.model_dump(
            mode="json"
        )
    judge_path = tmp_path / "login" / "judge-report.json"
    assert json.loads(judge_path.read_text(encoding="utf-8")) == judge_report.model_dump(
        mode="json"
    )
    assert all(
        "test-value" not in path.read_text(encoding="utf-8")
        for path in (
            tmp_path / "login" / "attempt-1" / "executor-report.json",
            tmp_path / "login" / "attempt-2" / "executor-report.json",
            judge_path,
        )
    )


@pytest.mark.parametrize(
    "artifact_name",
    ["executor-report.json", "judge-report.json"],
)
def test_report_write_failure_is_an_observation_without_changing_verdict(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    """An unwritable report artifact is recorded without changing a bug Verdict."""
    journey_dir = tmp_path / "login"
    artifact_path = (
        journey_dir / "attempt-1" / artifact_name
        if artifact_name == "executor-report.json"
        else journey_dir / artifact_name
    )
    artifact_path.mkdir(parents=True)
    stack = RecordingStack()
    executor = RecordingExecutor(
        [executor_report("bug", "E1"), executor_report("bug", "E1")]
    )
    judge = RecordingJudge(
        [
            JudgeReport(
                decisions=(
                    JudgeDecision(
                        outcome_id="E1",
                        decision="bug",
                        reason="The screenshot confirms it.",
                    ),
                )
            )
        ]
    )

    result = run_one(sample_journey("login"), stack, executor, judge, tmp_path)

    assert result.verdict == "bug"
    assert result.bug_report is not None
    assert any(artifact_name in observation.text for observation in result.observations)


@pytest.mark.parametrize(
    ("decisions", "expected_verdict", "expected_bug_ids", "expected_observation"),
    [
        (
            (JudgeDecision(outcome_id="E1", decision="not_bug", reason="No matching evidence."),),
            "pass",
            (),
            "Cross-check rejected E1: No matching evidence.",
        ),
        (
            (
                JudgeDecision(outcome_id="E1", decision="not_bug", reason="Not visible."),
                JudgeDecision(
                    outcome_id="E2",
                    decision="bug",
                    reason="The second screenshot confirms it.",
                ),
            ),
            "bug",
            ("E2",),
            "Cross-check rejected E1: Not visible.",
        ),
    ],
    ids=("reject-all", "confirm-subset"),
)
def test_judge_decisions_only_confirm_explicit_bug_outcomes(
    tmp_path: Path,
    decisions: tuple[JudgeDecision, ...],
    expected_verdict: Verdict,
    expected_bug_ids: tuple[str, ...],
    expected_observation: str,
) -> None:
    """Rejected or unconfirmed claims become Observations and cannot enter a Bug Report."""
    stack = RecordingStack()
    executor = RecordingExecutor(
        [executor_report("bug", "E1", "E2"), executor_report("bug", "E1", "E2")]
    )
    judge = RecordingJudge([JudgeReport(decisions=decisions)])

    result = run_one(sample_journey(), stack, executor, judge, tmp_path)

    assert result.verdict == expected_verdict
    bug_ids = (
        tuple(outcome.id for outcome in result.bug_report.outcomes)
        if result.bug_report is not None
        else ()
    )
    assert bug_ids == expected_bug_ids
    assert Observation(text=expected_observation) in result.observations


def test_judge_failure_makes_the_journey_inconclusive(tmp_path: Path) -> None:
    """A failed Cross-check cannot promote a reproduced claim to a Bug Report."""
    stack = RecordingStack()
    executor = RecordingExecutor(
        [executor_report("bug", "E1"), executor_report("bug", "E1")]
    )
    judge = RecordingJudge([AgentFailure(kind="invalid_output", detail="invalid schema")])

    result = run_one(sample_journey(), stack, executor, judge, tmp_path)

    assert result.verdict == "inconclusive"
    assert result.failure == "invalid_output"
    assert result.bug_report is None
    assert result.attempts == 2
    assert json.loads((tmp_path / "sample" / "judge-report.json").read_text(encoding="utf-8")) == {
        "failure": {"kind": "invalid_output", "detail": "invalid schema"}
    }


@pytest.mark.parametrize("failure_attempt", [1, 2], ids=("first-attempt", "second-attempt"))
def test_stack_error_makes_the_journey_inconclusive(
    tmp_path: Path, failure_attempt: int
) -> None:
    """A stack failure at either preparation boundary prevents a bug report."""
    errors = [None] * (failure_attempt - 1) + [StackError("stack unavailable")]
    stack = RecordingStack(errors)
    executor_results = [executor_report("bug", "E1")]
    if failure_attempt == 2:
        executor_results.append(executor_report("bug", "E1"))
    executor = RecordingExecutor(executor_results)
    judge = RecordingJudge([])

    result = run_one(sample_journey(), stack, executor, judge, tmp_path)

    assert result.verdict == "inconclusive"
    assert result.failure == "stack: stack unavailable"
    assert result.attempts == failure_attempt
    assert len(executor.evidence_dirs) == failure_attempt - 1
    assert judge.reports == []


def test_setup_stack_error_collects_evidence_and_preserves_original_failure(
    tmp_path: Path,
) -> None:
    """Setup failure evidence is best-effort and cannot replace its original reason."""
    journey = sample_journey("camera-setup")
    stack = RecordingStack(
        [StackError("camera setup failed")],
        evidence_error=StackError("compose log unavailable"),
    )
    executor = RecordingExecutor([])

    result = run_one(journey, stack, executor, RecordingJudge([]), tmp_path)

    assert result.verdict == "inconclusive"
    assert result.failure == "stack: camera setup failed"
    assert stack.evidence_dirs == [tmp_path / "camera-setup" / "attempt-1"]
    assert executor.evidence_dirs == []


def test_executor_stack_error_collects_attempt_evidence(tmp_path: Path) -> None:
    """A StackError raised by executor execution still leaves stack evidence."""
    journey = sample_journey("executor-setup")
    stack = RecordingStack()
    executor = RecordingExecutor([StackError("executor setup failed")])

    result = run_one(journey, stack, executor, RecordingJudge([]), tmp_path)

    assert result.verdict == "inconclusive"
    assert result.failure == "stack: executor setup failed"
    assert stack.evidence_dirs == [tmp_path / "executor-setup" / "attempt-1"]


def test_usage_limit_marks_remaining_journeys_inconclusive_without_preparing_them(
    tmp_path: Path,
) -> None:
    """After a usage limit, later Journeys are recorded with zero attempts."""
    journeys = tuple(sample_journey(f"journey-{index}") for index in range(1, 5))
    stack = RecordingStack()
    executor = RecordingExecutor(
        [
            executor_report("pass"),
            AgentFailure(kind="usage_limit", detail="quota reached"),
        ]
    )

    runs = run_journeys(
        journeys,
        stack=stack,
        executor=executor,
        judge=RecordingJudge([]),
        base_url="http://localhost:18080",
        credentials=OperatorCredentials(username="admin", password=SecretStr("test-value")),
        run_dir=tmp_path,
    )

    assert tuple(run.verdict for run in runs) == (
        "pass",
        "inconclusive",
        "inconclusive",
        "inconclusive",
    )
    assert tuple(run.attempts for run in runs) == (1, 1, 0, 0)
    assert tuple(run.failure for run in runs) == (None, "usage_limit", "usage_limit", "usage_limit")
    assert stack.prepared == ["journey-1", "journey-2"]
    assert json.loads(
        (tmp_path / "journey-2" / "attempt-1" / "executor-report.json").read_text(
            encoding="utf-8"
        )
    ) == {"failure": {"kind": "usage_limit", "detail": "quota reached"}}


def test_stack_and_agents_receive_attempt_evidence_directories(tmp_path: Path) -> None:
    """Each execution and evidence collection is isolated under its attempt number."""
    journey = sample_journey("login")
    stack = RecordingStack()
    executor = RecordingExecutor(
        [executor_report("bug", "E1"), executor_report("bug", "E1")]
    )
    judge = RecordingJudge(
        [
            JudgeReport(
                decisions=(
                    JudgeDecision(outcome_id="E1", decision="bug", reason="Confirmed."),
                )
            )
        ]
    )

    _ = run_one(journey, stack, executor, judge, tmp_path)

    attempt_dirs = (tmp_path / "login" / "attempt-1", tmp_path / "login" / "attempt-2")
    assert executor.evidence_dirs == list(attempt_dirs)
    assert stack.evidence_dirs == list(attempt_dirs)
    assert judge.evidence_dirs == [tmp_path / "login"]


def test_fingerprint_sorts_and_deduplicates_outcome_ids() -> None:
    """Fingerprint identity depends on the Journey and unique sorted outcome IDs."""
    expected = hashlib.sha256(b"login|E1,E2").hexdigest()

    assert fingerprint("login", ("E2", "E1", "E2")) == expected
    assert fingerprint("login", ("E1", "E2")) == expected
