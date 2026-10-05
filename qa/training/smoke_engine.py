"""Run a tiny GPU CLIP train, process restart, and resume continuity proof."""

# This standalone QA executable is invoked by its file path, not imported as a package.
# ruff: noqa: INP001
# ruff: noqa: TRY003, EM101, EM102

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, Protocol, cast
from uuid import UUID, uuid4

from gods_watching.contracts.training import TrainingConfig, TrainingMetric
from gods_watching.training.dataset import validate_cuhk
from gods_watching.training.engine import (
    TrainingPaths,
    TrainingRunSnapshot,
    run_training,
)
from gods_watching.training.engine_data import load_clip_components
from gods_watching.training.memory import current_source_fingerprint

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gods_watching.training.engine_api import TrainingModel

_NATIVE_SPLITS = ("train", "val", "test")
_IMAGES_PER_SPLIT = 16
_CHILD_PHASES = ("baseline", "partial", "resume")


@dataclass(frozen=True, slots=True)
class _Arguments:
    dataset_root: Path
    model_root: Path
    run_root: Path
    child_phase: Literal["baseline", "partial", "resume"] | None
    snapshot: Path | None
    subset_root: Path | None
    output: Path | None
    run_directory: Path | None
    checkpoint: Path | None


class _NumpyByteArray(Protocol):
    def tobytes(self) -> bytes: ...


class _NumpyExportableTensor(Protocol):
    def detach(self) -> _NumpyExportableTensor: ...

    def cpu(self) -> _NumpyExportableTensor: ...

    def numpy(self) -> _NumpyByteArray: ...


class _Reporter:
    cancellation: threading.Event
    _cancel_after_first_epoch: bool
    _epoch_metrics: int
    _first_checkpoint_copy: Path | None
    _captured_checkpoint: bool

    def __init__(
        self,
        cancel_after_first_epoch: bool,
        *,
        first_checkpoint_copy: Path | None = None,
    ) -> None:
        self.cancellation = threading.Event()
        self._cancel_after_first_epoch = cancel_after_first_epoch
        self._epoch_metrics = 0
        self._first_checkpoint_copy = first_checkpoint_copy
        self._captured_checkpoint = False

    def progress(self, row: dict[str, object]) -> None:
        _ = row

    def metric(self, metric: TrainingMetric) -> None:
        _ = metric
        self._epoch_metrics += 1
        if self._cancel_after_first_epoch and self._epoch_metrics == 1:
            self.cancellation.set()

    def checkpoint(self, path: Path, best_metric: float) -> None:
        _ = (path, best_metric)
        if self._first_checkpoint_copy is not None and not self._captured_checkpoint:
            _ = shutil.copyfile(path, self._first_checkpoint_copy)
            self._captured_checkpoint = True


def _config() -> TrainingConfig:
    return TrainingConfig(
        epochs=2,
        learning_rate=1e-5,
        micro_batch_size=_IMAGES_PER_SPLIT,
        gradient_accumulation=1,
        warmup_ratio=0.0,
        mixed_precision="fp32",
        gradient_checkpointing=True,
        early_stopping_patience=None,
    )


def _copy_tiny_native_split_subset(source_root: Path, subset_root: Path) -> None:
    annotations = cast(
        "object",
        json.loads((source_root / "reid_raw.json").read_text(encoding="utf-8")),
    )
    if not isinstance(annotations, list):
        raise TypeError("CUHK-PEDES annotations must be a list")
    rows = cast("list[object]", annotations)
    chosen: dict[str, list[dict[str, object]]] = {split: [] for split in _NATIVE_SPLITS}
    seen_ids: dict[str, set[int]] = {split: set() for split in _NATIVE_SPLITS}
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            continue
        row = cast("dict[str, object]", raw_row)
        split = row.get("split")
        person_id = row.get("id")
        relative = row.get("file_path")
        if (
            not isinstance(split, str)
            or split not in chosen
            or type(person_id) is not int
            or len(chosen[split]) >= _IMAGES_PER_SPLIT
            or person_id in seen_ids[split]
            or not isinstance(relative, str)
        ):
            continue
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in relative.split("/")):
            raise ValueError("source annotation contains an unsafe image path")
        source = source_root / "imgs" / Path(*pure.parts)
        if source.is_symlink() or not source.is_file():
            raise ValueError("source subset image is missing or unsafe")
        destination = subset_root / "imgs" / Path(*pure.parts)
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _ = shutil.copyfile(source, destination)
        chosen[split].append(
            {
                "split": split,
                "id": person_id,
                "file_path": relative,
                "captions": row.get("captions"),
            }
        )
        seen_ids[split].add(person_id)
    if any(len(chosen[split]) != _IMAGES_PER_SPLIT for split in _NATIVE_SPLITS):
        raise ValueError("source dataset lacks a tiny identity-disjoint sample per split")
    subset_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _ = (subset_root / "reid_raw.json").write_text(
        json.dumps([row for split in _NATIVE_SPLITS for row in chosen[split]]),
        encoding="utf-8",
    )


def _load_snapshot(snapshot_path: Path, *, checkpoint: Path | None = None) -> TrainingRunSnapshot:
    payload = cast("object", json.loads(snapshot_path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        raise TypeError("smoke snapshot must be an object")
    data = cast("dict[str, object]", payload)
    return TrainingRunSnapshot(
        job_id=UUID(str(data["job_id"])),
        owner_generation=1,
        config=TrainingConfig.model_validate(data["config"]),
        dataset_fingerprint=str(data["dataset_fingerprint"]),
        source_fingerprint=str(data["source_fingerprint"]),
        checkpoint_path=checkpoint,
    )


def _weights_sha256(model: TrainingModel) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        host_tensor = cast(
            "_NumpyExportableTensor",
            cast("object", tensor),
        ).detach().cpu()
        digest.update(host_tensor.numpy().tobytes())
    return digest.hexdigest()


def _child_run(  # noqa: PLR0913
    *,
    phase: str,
    snapshot_path: Path,
    subset_root: Path,
    model_root: Path,
    output_path: Path,
    run_directory: Path,
    checkpoint_path: Path | None,
) -> int:
    snapshot = _load_snapshot(snapshot_path, checkpoint=checkpoint_path)
    _ = validate_cuhk(subset_root)
    paths = TrainingPaths(
        run_directory=run_directory,
        dataset_root=subset_root,
        model_root=model_root,
    )
    components = load_clip_components(snapshot, paths)
    weights_before = _weights_sha256(components.model)
    checkpoint_capture = run_directory / "epoch1.pt" if phase != "resume" else None
    reporter = _Reporter(
        cancel_after_first_epoch=phase == "partial",
        first_checkpoint_copy=checkpoint_capture,
    )
    result = run_training(snapshot, paths, reporter, reporter.cancellation, components=components)
    record: dict[str, object] = {
        "phase": phase,
        "completed": result.completed,
        "cancelled": result.cancelled,
        "epochs_completed": result.epochs_completed,
        "optimizer_steps": result.optimizer_steps,
        "weights_before": weights_before,
        "weights_after": _weights_sha256(components.model),
    }
    _ = output_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    if phase == "partial" and not result.cancelled:
        raise RuntimeError("partial smoke process did not observe cancellation")
    if phase in {"baseline", "resume"} and not result.completed:
        raise RuntimeError("smoke training did not complete")
    return 0


def _child_command(  # noqa: PLR0913
    phase: str,
    snapshot: Path,
    subset: Path,
    model_root: Path,
    output: Path,
    run_dir: Path,
    checkpoint: Path | None,
) -> list[str]:
    script = Path(__file__).resolve()
    command = [
        sys.executable,
        str(script),
        "--child-phase",
        phase,
        "--snapshot",
        str(snapshot),
        "--subset-root",
        str(subset),
        "--model-root",
        str(model_root),
        "--output",
        str(output),
        "--run-directory",
        str(run_dir),
    ]
    if checkpoint is not None:
        command.extend(("--checkpoint", str(checkpoint)))
    return command


def _run_parent(args: _Arguments) -> int:
    root = args.dataset_root.resolve(strict=True)
    model_root = args.model_root.resolve(strict=True)
    if model_root != Path("/models/clip"):
        raise ValueError("smoke training must use the read-only /models/clip mount")
    run_root = args.run_root
    run_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    attempt_id = uuid4()
    subset = run_root / f"{attempt_id}-subset"
    _copy_tiny_native_split_subset(root, subset)
    manifest = validate_cuhk(subset)
    snapshot_path = run_root / f"{attempt_id}-snapshot.json"
    _ = snapshot_path.write_text(
        json.dumps(
            {
                "job_id": str(attempt_id),
                "config": _config().model_dump(mode="json"),
                "dataset_fingerprint": manifest.fingerprint,
                "source_fingerprint": current_source_fingerprint(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    baseline_dir = run_root / f"{attempt_id}-baseline"
    partial_dir = run_root / f"{attempt_id}-restart"
    output_paths = {
        "baseline": run_root / f"{attempt_id}-baseline.json",
        "partial": run_root / f"{attempt_id}-partial.json",
        "resume": run_root / f"{attempt_id}-resume.json",
    }
    phases = (
        ("baseline", baseline_dir, None),
        ("partial", partial_dir, None),
        ("resume", partial_dir, partial_dir / "last.pt"),
    )
    environment = os.environ.copy()
    for phase, run_directory, checkpoint_path in phases:
        completed = subprocess.run(  # noqa: S603 - fixed local script and enumerated phase values.
            _child_command(
                phase,
                snapshot_path,
                subset,
                model_root,
                output_paths[phase],
                run_directory,
                checkpoint_path,
            ),
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{phase} smoke child failed: {completed.stderr[-1000:]}")

    baseline = cast("dict[str, object]", json.loads(output_paths["baseline"].read_text()))
    partial = cast("dict[str, object]", json.loads(output_paths["partial"].read_text()))
    resumed = cast("dict[str, object]", json.loads(output_paths["resume"].read_text()))
    if baseline["weights_before"] == baseline["weights_after"]:
        raise RuntimeError("GPU smoke training did not change the locked CLIP weights")
    if baseline["weights_after"] != resumed["weights_after"]:
        raise RuntimeError("restart/resume weights differ from uninterrupted training")
    result = {
        "gpu": os.environ.get("GW_TRAINING_GPU_UUID"),
        "dataset_fingerprint": manifest.fingerprint,
        "tiny_samples_per_split": _IMAGES_PER_SPLIT,
        "baseline_optimizer_steps": baseline["optimizer_steps"],
        "partial_epochs_completed": partial["epochs_completed"],
        "resumed_optimizer_steps": resumed["optimizer_steps"],
        "weights_changed": True,
        "restart_resume_equivalent": True,
        "artifacts": {phase: str(path) for phase, path in output_paths.items()},
    }
    _ = sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


def _arguments(argv: Sequence[str] | None = None) -> _Arguments:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/datasets/cuhk-pedes"),
    )
    _ = parser.add_argument("--model-root", type=Path, default=Path("/models/clip"))
    _ = parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/runs/training-engine-smoke"),
    )
    _ = parser.add_argument("--child-phase", choices=_CHILD_PHASES)
    _ = parser.add_argument("--snapshot", type=Path)
    _ = parser.add_argument("--subset-root", type=Path)
    _ = parser.add_argument("--output", type=Path)
    _ = parser.add_argument("--run-directory", type=Path)
    _ = parser.add_argument("--checkpoint", type=Path)
    parsed = parser.parse_args(argv)
    dataset_root = _require_path(cast("object", parsed.dataset_root), field="dataset root")
    model_root = _require_path(cast("object", parsed.model_root), field="model root")
    run_root = _require_path(cast("object", parsed.run_root), field="run root")
    child_phase_value: object = cast("object", parsed.child_phase)
    if child_phase_value is not None and child_phase_value not in _CHILD_PHASES:
        raise ValueError("smoke child phase is invalid")
    return _Arguments(
        dataset_root=dataset_root,
        model_root=model_root,
        run_root=run_root,
        child_phase=child_phase_value,
        snapshot=_optional_path(cast("object", parsed.snapshot), field="snapshot"),
        subset_root=_optional_path(cast("object", parsed.subset_root), field="subset root"),
        output=_optional_path(cast("object", parsed.output), field="output"),
        run_directory=_optional_path(cast("object", parsed.run_directory), field="run directory"),
        checkpoint=_optional_path(cast("object", parsed.checkpoint), field="checkpoint"),
    )


def _require_path(value: object, *, field: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"smoke {field} is not a path")
    return value


def _optional_path(value: object, *, field: str) -> Path | None:
    if value is None:
        return None
    return _require_path(value, field=field)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one child phase or orchestrate the complete three-process GPU smoke."""
    args = _arguments(argv)
    if args.child_phase is not None:
        return _child_run(
            phase=args.child_phase,
            snapshot_path=_require_path(args.snapshot, field="snapshot"),
            subset_root=_require_path(args.subset_root, field="subset root"),
            model_root=args.model_root,
            output_path=_require_path(args.output, field="output"),
            run_directory=_require_path(args.run_directory, field="run directory"),
            checkpoint_path=args.checkpoint,
        )
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
