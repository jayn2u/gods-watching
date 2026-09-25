"""Structured prompts and output schemas for Journey agents."""

from pathlib import Path
from typing import Final

from .models import ExecutorReport, Journey, OperatorCredentials

EXECUTOR_SCHEMA: Final[dict[str, object]] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "bug", "inconclusive"]},
        "violations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "outcome_id": {"type": "string"},
                    "observed": {"type": "string"},
                    "evidence_files": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["outcome_id", "observed", "evidence_files"],
                "additionalProperties": False,
            },
        },
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
        "steps": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["verdict", "violations", "observations", "steps", "summary"],
    "additionalProperties": False,
}

JUDGE_SCHEMA: Final[dict[str, object]] = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "outcome_id": {"type": "string"},
                    "decision": {"type": "string", "enum": ["bug", "not_bug"]},
                    "reason": {"type": "string"},
                },
                "required": ["outcome_id", "decision", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["decisions"],
    "additionalProperties": False,
}


def build_executor_prompt(
    journey: Journey,
    *,
    base_url: str,
    credentials: OperatorCredentials,
    evidence_dir: Path,
) -> str:
    """Build the task and safety instructions for a browser executor."""
    outcomes = "\n".join(
        f"- [{outcome.id}] {outcome.text}" for outcome in journey.expected_outcomes
    )
    return "\n".join(
        (
            "Perform this Journey in the browser. Use only the browser tool provided to you.",
            "",
            f"Journey: {journey.id} — {journey.title}",
            f"Preconditions: {journey.preconditions}",
            f"Goal: {journey.goal}",
            "Expected Outcomes:",
            outcomes,
            f"Application base URL: {base_url}",
            f"Operator ID: {credentials.username}",
            f"Operator password: {credentials.password.get_secret_value()}",
            f"Evidence directory: {evidence_dir}",
            "Screenshots are saved under browser; report each name in evidence_files.",
            "",
            "Judge the Journey only against the Expected Outcome IDs listed above.",
            "Record anything else notable as an Observation without changing the Verdict.",
            "For every violation, include the saved screenshot file names in evidence_files.",
            "Before finishing, inspect browser console messages and product network",
            "responses for C1 and C2.",
            "Return inconclusive if the stack is unusable for a reason outside the Journey Goal.",
            "Do not use shell, Docker, direct APIs, or other tools to establish Journey",
            "preconditions.",
            "Return only the structured report requested by the output schema.",
        )
    )


def build_judge_prompt(
    journey: Journey,
    report: ExecutorReport,
    *,
    evidence_dir: Path,
) -> str:
    """Build an evidence-only prompt for the independent Cross-check judge."""
    outcomes = "\n".join(
        f"- [{outcome.id}] {outcome.text}" for outcome in journey.expected_outcomes
    )
    claims = "\n".join(
        f"- [{violation.outcome_id}] {violation.observed}"
        for violation in report.violations
    ) or "- No claimed violations."
    evidence = _list_evidence_files(evidence_dir)
    return "\n".join(
        (
            "Independently Cross-check the claimed Journey violations using evidence only.",
            "Do not use the executor's conclusions as evidence. Do not use tools other than Read.",
            f"Journey: {journey.id} — {journey.title}",
            f"Preconditions: {journey.preconditions}",
            f"Goal: {journey.goal}",
            "Expected Outcomes:",
            outcomes,
            "Claimed violations and observed text:",
            claims,
            f"Evidence directory: {evidence_dir}",
            "Evidence directory listing:",
            evidence,
            "Return one decision for each claimed outcome ID. Decide bug only when evidence",
            "shows that the corresponding Expected Outcome was violated; otherwise decide not_bug.",
            "Use only the named Expected Outcomes and evidence from the supplied directory.",
            "Return only the structured decisions requested by the output schema.",
        )
    )


def _list_evidence_files(evidence_dir: Path) -> str:
    if not evidence_dir.is_dir():
        return "(directory is empty)"
    paths = sorted(
        path.relative_to(evidence_dir).as_posix()
        for path in evidence_dir.rglob("*")
        if path.is_file()
    )
    return "\n".join(paths) if paths else "(directory is empty)"


__all__ = [
    "EXECUTOR_SCHEMA",
    "JUDGE_SCHEMA",
    "build_executor_prompt",
    "build_judge_prompt",
]
