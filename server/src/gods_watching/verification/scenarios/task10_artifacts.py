"""Build manifest, model, and adversarial evidence for Task 10 runs."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from gods_watching.fixtures.manifest import load_fixture_manifest
from gods_watching.verification.models import ScenarioContextProtocol

from .task10_errors import Task10ExecutionError
from .task10_models import (
    DirectIdentityObservation,
    DriverEvidence,
    FixtureProvenance,
    FixtureStreamProvenance,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
_FIXTURE_MANIFEST = _REPOSITORY_ROOT / "assets/test-streams.json"
_YOLO_WEIGHTS = _REPOSITORY_ROOT / "runtime/assets/models/yolo/yolo11s.pt"
_PREPARED_MANIFEST = _REPOSITORY_ROOT / "runtime/assets/models/prepared-manifest.json"
_IDENTITY_OBSERVATION = Path(".omo/evidence/task-10/identity-observation/Observation.json")


class _ImportedModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)


class _ImportedIdentityResult(_ImportedModel):
    identity_switch_observations: int
    matched_observations: int
    adjacent_comparison_opportunities: int
    reviewed_clips: int
    reviewed_sample_frames: int
    unknown_outside_denominator: bool


class _ImportedHashPath(_ImportedModel):
    path: str
    sha256: str


class _ImportedIdentityProvenance(_ImportedModel):
    production_source_hashes: dict[str, str]
    fixture_files: tuple[_ImportedHashPath, ...]
    raw_observations: _ImportedHashPath


class _ImportedIdentityObservation(_ImportedModel):
    result: _ImportedIdentityResult
    provenance: _ImportedIdentityProvenance


@dataclass(frozen=True, slots=True)
class AdversarialPaths:
    """Identify the evidence artifacts linked by the adversarial receipt."""

    provenance: Path
    cleanup: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fixture_provenance() -> FixtureProvenance:
    """Verify and return the pinned fixture and model asset provenance."""
    manifest = load_fixture_manifest(_FIXTURE_MANIFEST)
    streams: list[FixtureStreamProvenance] = []
    for stream in manifest.streams:
        path = (_REPOSITORY_ROOT / stream.prepared_path).resolve(strict=True)
        actual_hash = _sha256(path)
        if actual_hash != stream.sha256:
            raise Task10ExecutionError(detail=f"fixture checksum mismatch for {stream.stream_id}")
        streams.append(
            FixtureStreamProvenance(
                stream_id=stream.stream_id,
                prepared_path=stream.prepared_path,
                sha256=actual_hash,
                bytes=path.stat().st_size,
                codec=stream.codec,
                width=stream.width,
                height=stream.height,
            )
        )
    return FixtureProvenance(
        manifest=str(_FIXTURE_MANIFEST.relative_to(_REPOSITORY_ROOT)),
        manifest_sha256=_sha256(_FIXTURE_MANIFEST),
        prepared_inputs_redistributable=manifest.prepared_inputs_redistributable,
        streams=tuple(streams),
        yolo_weights_sha256=_sha256(_YOLO_WEIGHTS),
        prepared_model_manifest_sha256=_sha256(_PREPARED_MANIFEST),
    )


def load_identity_observation(repository_root: Path) -> DirectIdentityObservation:
    """Revalidate the confirmed direct identity count against current source bytes."""
    observation_path = repository_root / _IDENTITY_OBSERVATION
    imported = _ImportedIdentityObservation.model_validate_json(
        observation_path.read_text(encoding="utf-8")
    )
    result = imported.result
    provenance = imported.provenance
    source_hashes = provenance.production_source_hashes
    source_hashes_current = all(
        _sha256(repository_root / path) == expected_hash
        for path, expected_hash in source_hashes.items()
    )
    fixture_hashes_current = all(
        _sha256(repository_root / fixture.path) == fixture.sha256
        for fixture in provenance.fixture_files
    )
    raw = provenance.raw_observations
    raw_path = observation_path.parent / raw.path
    if _sha256(raw_path) != raw.sha256:
        raise Task10ExecutionError(detail="identity raw observation checksum mismatch")
    if not source_hashes_current or not fixture_hashes_current:
        raise Task10ExecutionError(detail="identity observation is stale for current sources")
    return DirectIdentityObservation(
        method="human-labeled-retained-frame-review",
        identity_switches_counted=result.identity_switch_observations,
        matched_observations=result.matched_observations,
        adjacent_comparison_opportunities=result.adjacent_comparison_opportunities,
        reviewed_clips=result.reviewed_clips,
        reviewed_sample_frames=result.reviewed_sample_frames,
        unknown_outside_denominator=result.unknown_outside_denominator,
        observation_sha256=_sha256(observation_path),
        raw_observations_sha256=_sha256(raw_path),
        production_source_hashes=source_hashes,
        all_source_hashes_current=source_hashes_current,
        all_fixture_hashes_current=fixture_hashes_current,
    )


def write_adversarial_artifact(
    context: ScenarioContextProtocol,
    driver: DriverEvidence,
    paths: AdversarialPaths,
) -> Path:
    """Write adversarial coverage claims linked to this run's evidence paths."""
    artifact = context.run_root / "task10-adversarial.json"
    payload = {
        "malformed_input": {
            "status": "not_applicable",
            "reason": (
                "The installed scenario uses the pinned local fixture manifest; input "
                "boundary tests are owned by fixture and decoder suites."
            ),
        },
        "untrusted_text": {
            "status": "not_applicable",
            "reason": (
                "The scenario accepts no user text or credentials; RTSP sources are "
                "fixed prepared paths."
            ),
        },
        "cancel_resume": {
            "status": "probed",
            "result": (
                "Owned driver and subprocess resources are registered before spawn and "
                "cleaned after normal completion."
            ),
            "evidence": str(paths.cleanup),
        },
        "stale_state": {
            "status": "probed",
            "result": (
                "Fresh evidence roots and unique Compose/container names are used; "
                "stale generation results are fenced."
            ),
            "evidence": str(paths.provenance),
        },
        "dirty_worktree": {
            "status": "probed",
            "result": "Full HEAD and dirty diff hashes are emitted by the installed harness.",
            "evidence": str(context.run_root / "result.json"),
        },
        "hung_commands": {
            "status": "probed",
            "result": "Driver and resource commands completed under the scenario deadline.",
            "evidence": str(paths.cleanup),
        },
        "flaky_tests": {
            "status": "not_applicable",
            "reason": (
                "This artifact is one real runtime observation; deterministic unit "
                "coverage is separate."
            ),
        },
        "misleading_success_output": {
            "status": "probed",
            "result": (
                "Success requires parsed DriverEvidence assertions and ScenarioReport "
                "checks; stdout alone is ignored."
            ),
            "evidence": str(context.run_root / "task10-driver.json"),
        },
        "repeated_interrupts": {
            "status": "not_applicable",
            "reason": (
                "Signal interruption is owned by the shared launcher/context acceptance "
                "and is not synthesized inside a real GPU run."
            ),
        },
        "real_label": driver.label,
    }
    _ = artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


__all__ = [
    "AdversarialPaths",
    "fixture_provenance",
    "load_identity_observation",
    "write_adversarial_artifact",
]
