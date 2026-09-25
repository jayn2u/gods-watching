"""Retry and Cross-check policy for Journey Run Verdicts."""

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .agents import JourneyExecutor, Judge
from .models import (
    AgentFailure,
    BugReport,
    ExecutorReport,
    Journey,
    JourneyRun,
    JudgeDecision,
    JudgeReport,
    Observation,
    OperatorCredentials,
    StackError,
    Violation,
)


class JourneyStack(Protocol):
    """Expose deterministic setup and evidence collection to the pipeline."""

    def prepare_journey(self, journey: Journey) -> None:
        """Reset the stack and establish the Journey's declared preconditions."""
        ...

    def collect_evidence(self, evidence_dir: Path) -> None:
        """Save stack evidence for one completed attempt."""
        ...


@dataclass(frozen=True, slots=True)
class _Attempt:
    report: ExecutorReport | None
    failure: str | None
    usage_limit: bool
    observations: tuple[Observation, ...] = ()


@dataclass(frozen=True, slots=True)
class _RunContext:
    stack: JourneyStack
    executor: JourneyExecutor
    judge: Judge
    base_url: str
    credentials: OperatorCredentials
    run_dir: Path


def fingerprint(journey_id: str, outcome_ids: Iterable[str]) -> str:
    """Return a stable identity for a Journey and its confirmed outcome IDs."""
    normalized_ids = ",".join(sorted(set(outcome_ids)))
    return hashlib.sha256(f"{journey_id}|{normalized_ids}".encode()).hexdigest()


def run_journeys(  # noqa: PLR0913
    journeys: Sequence[Journey],
    *,
    stack: JourneyStack,
    executor: JourneyExecutor,
    judge: Judge,
    base_url: str,
    credentials: OperatorCredentials,
    run_dir: Path,
) -> tuple[JourneyRun, ...]:
    """Run every Journey, requiring reproduction and an independent Cross-check."""
    context = _RunContext(stack, executor, judge, base_url, credentials, run_dir)
    runs: list[JourneyRun] = []
    usage_limit_seen = False
    for journey in journeys:
        if usage_limit_seen:
            runs.append(_unattempted_usage_limit_run(journey))
            continue
        run, hit_usage_limit = _run_journey(journey, context)
        runs.append(run)
        usage_limit_seen = usage_limit_seen or hit_usage_limit
    return tuple(runs)


def _run_journey(journey: Journey, context: _RunContext) -> tuple[JourneyRun, bool]:
    first = _run_attempt(journey, 1, context)
    if first.failure is not None:
        return (
            _inconclusive_run(journey, first.failure, 1, _attempt_observations(first)),
            first.usage_limit,
        )
    first_report = first.report
    if first_report is None or (first_report.verdict == "bug" and not first_report.violations):
        return _inconclusive_run(journey, "invalid_output", 1, _attempt_observations(first)), False
    if first_report.verdict != "bug":
        return _report_run(journey, first_report, 1, first.observations), False

    second = _run_attempt(journey, 2, context)
    if second.failure is not None or second.report is None:
        failure = second.failure or "invalid_output"
        observations = (
            first_report.observations
            + first.observations
            + _attempt_observations(second)
        )
        return (
            _inconclusive_run(journey, failure, 2, observations),
            second.usage_limit,
        )
    second_report = second.report

    reproduced_ids = _reproduced_ids(journey, first_report, second_report)
    if second_report.verdict != "bug" or not reproduced_ids:
        return _flaky_run(
            journey,
            first_report,
            second_report,
            first.observations + second.observations,
        )
    return _cross_check(
        journey,
        first,
        second,
        reproduced_ids,
        context,
    )


def _run_attempt(journey: Journey, number: int, context: _RunContext) -> _Attempt:
    evidence_dir = context.run_dir / journey.id / f"attempt-{number}"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    try:
        context.stack.prepare_journey(journey)
        report = context.executor.execute(
            journey,
            base_url=context.base_url,
            credentials=context.credentials,
            evidence_dir=evidence_dir,
        )
    except StackError as error:
        with suppress(StackError):
            context.stack.collect_evidence(evidence_dir)
        return _Attempt(report=None, failure=f"stack: {error}", usage_limit=False)
    report_observation = _write_agent_report(evidence_dir / "executor-report.json", report)
    observations = () if report_observation is None else (report_observation,)
    try:
        context.stack.collect_evidence(evidence_dir)
    except StackError as error:
        return _Attempt(
            report=None,
            failure=f"stack: {error}",
            usage_limit=False,
            observations=observations,
        )
    if isinstance(report, AgentFailure):
        return _Attempt(
            report=None,
            failure=report.kind,
            usage_limit=report.kind == "usage_limit",
            observations=observations,
        )
    return _Attempt(
        report=report,
        failure=None,
        usage_limit=False,
        observations=observations,
    )


def _cross_check(
    journey: Journey,
    first_attempt: _Attempt,
    second_attempt: _Attempt,
    reproduced_ids: tuple[str, ...],
    context: _RunContext,
) -> tuple[JourneyRun, bool]:
    first = first_attempt.report
    second = second_attempt.report
    if first is None or second is None:
        error_message = "Cross-check requires reports from both executor attempts"
        raise RuntimeError(error_message)
    reproduced_violations = tuple(
        violation for violation in first.violations if violation.outcome_id in reproduced_ids
    )
    report_for_judge = first.model_copy(update={"violations": reproduced_violations})
    judge_result = context.judge.judge(
        journey,
        report_for_judge,
        evidence_dir=context.run_dir / journey.id,
    )
    judge_observation = _write_agent_report(
        context.run_dir / journey.id / "judge-report.json",
        judge_result,
    )
    judge_observations = () if judge_observation is None else (judge_observation,)
    if isinstance(judge_result, AgentFailure):
        observations = (
            first.observations
            + second.observations
            + first_attempt.observations
            + second_attempt.observations
            + judge_observations
        )
        return (
            _inconclusive_run(journey, judge_result.kind, 2, observations),
            judge_result.kind == "usage_limit",
        )

    decisions = {decision.outcome_id: decision for decision in judge_result.decisions}
    confirmed_ids = tuple(
        outcome_id
        for outcome_id in reproduced_ids
        if (decision := decisions.get(outcome_id)) is not None and decision.decision == "bug"
    )
    rejected = _rejected_observations(reproduced_ids, decisions)
    observations = (
        first.observations
        + second.observations
        + first_attempt.observations
        + second_attempt.observations
        + judge_observations
        + rejected
    )
    if not confirmed_ids:
        return (
            JourneyRun(
                journey_id=journey.id,
                verdict="pass",
                bug_report=None,
                observations=observations,
                attempts=2,
                failure=None,
            ),
            False,
        )

    bug_report = _make_bug_report(
        journey,
        confirmed_ids,
        reproduced_violations,
        decisions,
    )
    return (
        JourneyRun(
            journey_id=journey.id,
            verdict="bug",
            bug_report=bug_report,
            observations=observations,
            attempts=2,
            failure=None,
        ),
        False,
    )


def _make_bug_report(
    journey: Journey,
    confirmed_ids: tuple[str, ...],
    violations: tuple[Violation, ...],
    decisions: Mapping[str, JudgeDecision],
) -> BugReport:
    outcomes_by_id = {outcome.id: outcome for outcome in journey.expected_outcomes}
    confirmed_violations = tuple(
        violation for violation in violations if violation.outcome_id in confirmed_ids
    )
    reasons = tuple(decisions[outcome_id].reason for outcome_id in confirmed_ids)
    return BugReport(
        journey_id=journey.id,
        outcomes=tuple(outcomes_by_id[outcome_id] for outcome_id in confirmed_ids),
        observed=confirmed_violations,
        judge_reasons=reasons,
        fingerprint=fingerprint(journey.id, confirmed_ids),
    )


def _rejected_observations(
    reproduced_ids: tuple[str, ...],
    decisions: Mapping[str, JudgeDecision],
) -> tuple[Observation, ...]:
    observations: list[Observation] = []
    for outcome_id in reproduced_ids:
        decision = decisions.get(outcome_id)
        if decision is None:
            observations.append(
                Observation(text=f"Cross-check returned no decision for {outcome_id}.")
            )
        elif decision.decision != "bug":
            observations.append(
                Observation(text=f"Cross-check rejected {outcome_id}: {decision.reason}")
            )
    return tuple(observations)


def _reproduced_ids(
    journey: Journey,
    first: ExecutorReport,
    second: ExecutorReport,
) -> tuple[str, ...]:
    expected_ids = {outcome.id for outcome in journey.expected_outcomes}
    first_ids = {violation.outcome_id for violation in first.violations}
    second_ids = {violation.outcome_id for violation in second.violations}
    return tuple(sorted(first_ids & second_ids & expected_ids))


def _flaky_run(
    journey: Journey,
    first: ExecutorReport,
    second: ExecutorReport,
    additional_observations: tuple[Observation, ...] = (),
) -> tuple[JourneyRun, bool]:
    flaky_ids = tuple(sorted({violation.outcome_id for violation in first.violations}))
    observations = (
        first.observations
        + second.observations
        + additional_observations
        + (Observation(text=f"flaky: {', '.join(flaky_ids)} not reproduced", flaky=True),)
    )
    return (
        JourneyRun(
            journey_id=journey.id,
            verdict="pass",
            bug_report=None,
            observations=observations,
            attempts=2,
            failure=None,
        ),
        False,
    )


def _report_run(
    journey: Journey,
    report: ExecutorReport,
    attempts: int,
    additional_observations: tuple[Observation, ...] = (),
) -> JourneyRun:
    return JourneyRun(
        journey_id=journey.id,
        verdict=report.verdict,
        bug_report=None,
        observations=report.observations + additional_observations,
        attempts=attempts,
        failure=None,
    )


def _attempt_observations(attempt: _Attempt) -> tuple[Observation, ...]:
    report_observations = attempt.report.observations if attempt.report is not None else ()
    return report_observations + attempt.observations


def _write_agent_report(
    path: Path,
    report: ExecutorReport | JudgeReport | AgentFailure,
) -> Observation | None:
    if isinstance(report, AgentFailure):
        content = json.dumps({"failure": report.model_dump(mode="json")}, indent=2)
    else:
        content = report.model_dump_json(indent=2)
    try:
        _ = path.write_text(content, encoding="utf-8")
    except OSError as error:
        return Observation(text=f"Could not write {path.name}: {error}")
    return None


def _inconclusive_run(
    journey: Journey,
    failure: str,
    attempts: int,
    observations: tuple[Observation, ...] = (),
) -> JourneyRun:
    return JourneyRun(
        journey_id=journey.id,
        verdict="inconclusive",
        bug_report=None,
        observations=observations,
        attempts=attempts,
        failure=failure,
    )


def _unattempted_usage_limit_run(journey: Journey) -> JourneyRun:
    return JourneyRun(
        journey_id=journey.id,
        verdict="inconclusive",
        bug_report=None,
        observations=(),
        attempts=0,
        failure="usage_limit",
    )


__all__ = ["JourneyStack", "fingerprint", "run_journeys"]
