"""Behavioral tests for loading Journey definitions from Markdown."""

from pathlib import Path

import pytest

from gods_watching.journeys.catalog import JourneyDefinitionError, load_catalog, load_journey
from gods_watching.journeys.models import COMMON_OUTCOMES


def journey_markdown(
    *,
    journey_id: str = "sample",
    setup: str = "login, cameras",
    tags: str = "smoke, auth",
    sections: str = (
        "## Preconditions\nA prepared account is available.\n\n"
        "## Goal\nConfirm that the operator can use the console.\n\n"
        "## Expected Outcomes\n- [E1] The console shows the expected state.\n"
    ),
    extra_frontmatter: str = "",
) -> str:
    """Build one small definition whose expected behavior is easy to read."""
    return (
        "---\n"
        f"id: {journey_id}\n"
        "title: Sample Journey\n"
        f"setup: {setup}\n"
        f"tags: {tags}\n"
        f"{extra_frontmatter}"
        "---\n"
        f"{sections}"
    )


def write_journey(directory: Path, contents: str, *, filename: str = "sample.md") -> Path:
    """Write a Markdown definition and return its source path."""
    path = directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(contents, encoding="utf-8")
    return path


def test_load_journey_parses_metadata_and_appends_common_outcomes(tmp_path: Path) -> None:
    """A valid definition becomes a typed Journey with shared outcomes last."""
    path = write_journey(tmp_path, journey_markdown())

    journey = load_journey(path)

    assert journey.id == "sample"
    assert journey.title == "Sample Journey"
    assert journey.setup == ("login", "cameras")
    assert journey.tags == ("smoke", "auth")
    assert journey.preconditions == "A prepared account is available."
    assert journey.goal == "Confirm that the operator can use the console."
    assert [(outcome.id, outcome.text) for outcome in journey.expected_outcomes] == [
        ("E1", "The console shows the expected state."),
        *((outcome.id, outcome.text) for outcome in COMMON_OUTCOMES),
    ]
    assert journey.source_path == path


def test_load_journey_accepts_empty_setup_and_tags(tmp_path: Path) -> None:
    """Empty comma-separated metadata is represented as an empty tuple."""
    path = write_journey(tmp_path, journey_markdown(setup="", tags=""))

    journey = load_journey(path)

    assert journey.setup == ()
    assert journey.tags == ()


@pytest.mark.parametrize(
    ("contents", "filename"),
    [
        ("not frontmatter\n", "sample.md"),
        (
            """---
id: sample
title: Sample Journey
setup: login
---
## Preconditions
Ready.
## Goal
Use it.
## Expected Outcomes
- [E1] It works.
""",
            "sample.md",
        ),
    ],
    ids=("missing-frontmatter", "missing-frontmatter-key"),
)
def test_load_journey_rejects_missing_frontmatter_or_key(
    tmp_path: Path, contents: str, filename: str
) -> None:
    """The parser rejects a missing metadata block or required key."""
    path = write_journey(tmp_path, contents, filename=filename)

    with pytest.raises(JourneyDefinitionError) as error:
        _ = load_journey(path)

    assert error.value.path == path


def test_load_journey_rejects_duplicate_frontmatter_key(tmp_path: Path) -> None:
    """Repeated metadata keys cannot silently overwrite one another."""
    path = write_journey(
        tmp_path,
        journey_markdown(extra_frontmatter="setup: cameras\n"),
    )

    with pytest.raises(JourneyDefinitionError) as error:
        _ = load_journey(path)

    assert error.value.path == path


def test_load_journey_rejects_unknown_setup_step(tmp_path: Path) -> None:
    """Only deterministic setup steps from the domain contract are accepted."""
    path = write_journey(tmp_path, journey_markdown(setup="logout"))

    with pytest.raises(JourneyDefinitionError) as error:
        _ = load_journey(path)

    assert error.value.path == path


@pytest.mark.parametrize("section", ["Preconditions", "Goal", "Expected Outcomes"])
def test_load_journey_rejects_missing_section(tmp_path: Path, section: str) -> None:
    """Every Journey body section in the Markdown contract is required."""
    sections = journey_markdown().split("---\n", maxsplit=2)[-1]
    contents = journey_markdown(sections=sections.replace(f"## {section}\n", ""))
    path = write_journey(tmp_path, contents)

    with pytest.raises(JourneyDefinitionError) as error:
        _ = load_journey(path)

    assert error.value.path == path


def test_load_journey_rejects_zero_expected_outcomes(tmp_path: Path) -> None:
    """Common outcomes do not satisfy a Journey's required specific outcome."""
    sections = (
        "## Preconditions\nReady.\n## Goal\nUse it.\n## Expected Outcomes\n"
    )
    path = write_journey(tmp_path, journey_markdown(sections=sections))

    with pytest.raises(JourneyDefinitionError) as error:
        _ = load_journey(path)

    assert error.value.path == path


def test_load_journey_rejects_duplicate_outcome_ids(tmp_path: Path) -> None:
    """A single Journey cannot define the same Expected Outcome twice."""
    sections = (
        "## Preconditions\nReady.\n## Goal\nUse it.\n## Expected Outcomes\n"
        "- [E1] First condition.\n- [E1] Second condition.\n"
    )
    path = write_journey(tmp_path, journey_markdown(sections=sections))

    with pytest.raises(JourneyDefinitionError) as error:
        _ = load_journey(path)

    assert error.value.path == path


def test_load_journey_rejects_id_that_differs_from_filename(tmp_path: Path) -> None:
    """The stable Journey ID comes from the source filename."""
    path = write_journey(tmp_path, journey_markdown(), filename="other.md")

    with pytest.raises(JourneyDefinitionError) as error:
        _ = load_journey(path)

    assert error.value.path == path


def test_load_journey_rejects_missing_file(tmp_path: Path) -> None:
    """A missing source produces the same typed definition error as bad content."""
    path = tmp_path / "missing.md"

    with pytest.raises(JourneyDefinitionError) as error:
        _ = load_journey(path)

    assert error.value.path == path


def test_load_catalog_sorts_journeys_by_id(tmp_path: Path) -> None:
    """Catalog order is stable regardless of filesystem enumeration order."""
    _ = write_journey(tmp_path, journey_markdown(journey_id="zebra"), filename="zebra.md")
    _ = write_journey(tmp_path, journey_markdown(journey_id="alpha"), filename="alpha.md")

    journeys = load_catalog(tmp_path)

    assert tuple(journey.id for journey in journeys) == ("alpha", "zebra")


def test_load_catalog_rejects_duplicate_ids_in_nested_files(tmp_path: Path) -> None:
    """A Journey ID cannot be declared by more than one source file."""
    _ = write_journey(
        tmp_path,
        journey_markdown(journey_id="sample"),
        filename="one/sample.md",
    )
    duplicate = write_journey(
        tmp_path, journey_markdown(journey_id="sample"), filename="two/sample.md"
    )

    with pytest.raises(JourneyDefinitionError) as error:
        _ = load_catalog(tmp_path)

    assert error.value.path == duplicate


def test_load_catalog_reads_the_five_real_journeys() -> None:
    """The source catalog contains exactly the first five planned Journeys."""
    repository_root = Path(__file__).resolve().parents[2]

    journeys = load_catalog(repository_root / "qa" / "journeys")

    assert tuple(journey.id for journey in journeys) == (
        "live-cameras",
        "login",
        "model-switch",
        "similar-search",
        "text-search",
    )
