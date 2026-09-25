"""Parse Journey definitions from their Markdown source files."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, override

from .models import COMMON_OUTCOMES, ExpectedOutcome, Journey, SetupStep

_REQUIRED_FRONTMATTER: Final = ("id", "title", "setup", "tags")
_FRONTMATTER_DELIMITER_COUNT: Final = 2
_SECTION_HEADINGS: Final = frozenset(("Preconditions", "Goal", "Expected Outcomes"))
_SETUP_STEPS: Final[dict[str, SetupStep]] = {"login": "login", "cameras": "cameras"}
_OUTCOME_PATTERN: Final = re.compile(r"^- \[([A-Za-z][A-Za-z0-9]*)\] (.+?)\s*$")
_OUTCOME_ID_PATTERN: Final = re.compile(r"^(E[1-9][0-9]*|C[12])$")


@dataclass(frozen=True, slots=True)
class JourneyDefinitionError(ValueError):
    """Reject a Journey source that does not satisfy the Markdown contract."""

    path: Path
    reason: str

    @override
    def __str__(self) -> str:
        """Identify the invalid source and the contract it violated."""
        return f"invalid Journey definition at {self.path}: {self.reason}"


def load_journey(path: Path) -> Journey:
    """Parse and validate one Journey definition."""
    try:
        markdown = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise JourneyDefinitionError(path, "source file cannot be read") from error

    lines = markdown.splitlines()
    delimiters = [index for index, line in enumerate(lines) if line == "---"]
    if len(delimiters) < _FRONTMATTER_DELIMITER_COUNT or delimiters[0] != 0:
        raise JourneyDefinitionError(path, "frontmatter must be between the first two --- lines")

    frontmatter = _parse_frontmatter(path, lines[delimiters[0] + 1 : delimiters[1]])
    _require_frontmatter(path, frontmatter)
    journey_id = frontmatter["id"]
    if journey_id != path.stem:
        raise JourneyDefinitionError(path, "frontmatter id must match the filename stem")

    setup = _parse_setup_steps(path, frontmatter["setup"])

    sections = _parse_sections(path, lines[delimiters[1] + 1 :])
    expected_outcomes = _parse_expected_outcomes(path, sections["Expected Outcomes"])
    expected_outcomes += COMMON_OUTCOMES

    return Journey(
        id=journey_id,
        title=frontmatter["title"],
        setup=setup,
        tags=tuple(_split_values(frontmatter["tags"])),
        preconditions=sections["Preconditions"],
        goal=sections["Goal"],
        expected_outcomes=expected_outcomes,
        source_path=path,
    )


def load_catalog(directory: Path) -> tuple[Journey, ...]:
    """Load all Markdown definitions recursively in stable ID order."""
    paths = sorted(directory.rglob("*.md"))
    journeys: list[Journey] = []
    seen_ids: set[str] = set()
    for path in paths:
        journey = load_journey(path)
        if journey.id in seen_ids:
            raise JourneyDefinitionError(path, f"duplicate Journey id: {journey.id}")
        seen_ids.add(journey.id)
        journeys.append(journey)
    return tuple(sorted(journeys, key=lambda journey: journey.id))


def _parse_frontmatter(path: Path, lines: list[str]) -> dict[str, str]:
    frontmatter: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=2):
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        normalized_key = key.strip()
        if not separator or not normalized_key:
            raise JourneyDefinitionError(path, f"invalid frontmatter line {line_number}")
        if normalized_key in frontmatter:
            raise JourneyDefinitionError(path, f"duplicate frontmatter key: {normalized_key}")
        frontmatter[normalized_key] = value.strip()
    return frontmatter


def _require_frontmatter(path: Path, frontmatter: dict[str, str]) -> None:
    for key in _REQUIRED_FRONTMATTER:
        if key not in frontmatter:
            raise JourneyDefinitionError(path, f"missing frontmatter key: {key}")
        if key in ("id", "title") and not frontmatter[key]:
            raise JourneyDefinitionError(path, f"frontmatter key cannot be empty: {key}")


def _split_values(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_setup_steps(path: Path, value: str) -> tuple[SetupStep, ...]:
    setup: list[SetupStep] = []
    for step in _split_values(value):
        setup_step = _SETUP_STEPS.get(step)
        if setup_step is None:
            raise JourneyDefinitionError(path, f"unknown setup step: {step}")
        setup.append(setup_step)
    return tuple(setup)


def _parse_sections(path: Path, lines: list[str]) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    for line in lines:
        if line.startswith("## "):
            heading = line[3:].strip()
            current_section = heading if heading in _SECTION_HEADINGS else None
            if current_section is not None:
                if current_section in sections:
                    raise JourneyDefinitionError(path, f"duplicate section: {current_section}")
                sections[current_section] = []
        elif current_section is not None:
            sections[current_section].append(line)

    missing_sections = _SECTION_HEADINGS.difference(sections)
    if missing_sections:
        missing = sorted(missing_sections)[0]
        raise JourneyDefinitionError(path, f"missing section: {missing}")
    return {heading: "\n".join(contents).strip() for heading, contents in sections.items()}


def _parse_expected_outcomes(path: Path, section: str) -> tuple[ExpectedOutcome, ...]:
    outcomes: list[ExpectedOutcome] = []
    seen_ids: set[str] = set()
    for line in section.splitlines():
        if not line.lstrip().startswith("- ["):
            continue
        match = _OUTCOME_PATTERN.fullmatch(line.strip())
        if match is None:
            raise JourneyDefinitionError(path, "Expected Outcome must use '- [E<n>] text' format")
        outcome_id, text = match.groups()
        if _OUTCOME_ID_PATTERN.fullmatch(outcome_id) is None:
            raise JourneyDefinitionError(path, f"invalid Expected Outcome id: {outcome_id}")
        if outcome_id in seen_ids or outcome_id in {outcome.id for outcome in COMMON_OUTCOMES}:
            raise JourneyDefinitionError(path, f"duplicate Expected Outcome id: {outcome_id}")
        seen_ids.add(outcome_id)
        outcomes.append(ExpectedOutcome(id=outcome_id, text=text))
    if not any(outcome.id.startswith("E") for outcome in outcomes):
        raise JourneyDefinitionError(path, "at least one E Expected Outcome is required")
    return tuple(outcomes)


__all__ = ["JourneyDefinitionError", "load_catalog", "load_journey"]
