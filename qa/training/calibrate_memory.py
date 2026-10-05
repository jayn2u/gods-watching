"""Measure fixed CUHK-PEDES CLIP training profiles in isolated CUDA children."""

# ruff: noqa: E402, I001, INP001, TRY003, EM101, EM102, PLR2004

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SERVER_SOURCE = REPOSITORY_ROOT / "server" / "src"
if str(SERVER_SOURCE) not in sys.path:
    sys.path.insert(0, str(SERVER_SOURCE))

from gods_watching.contracts.training import TrainingConfig
from gods_watching.setup.models import ModelsLock, load_models_lock, validate_model_assets
from gods_watching.training.calibration import (
    CalibrationModelShape,
    CalibrationRefusedError,
    MAX_CALIBRATION_OPTIMIZER_ATTEMPTS,
    bind_cuda_device_uuid,
    builtin_base_load_options,
    build_memory_profile,
    capture_cuda_snapshot,
    check_calibration_headroom,
    optimizer_update_succeeded,
)
from gods_watching.training.determinism import (
    configure_torch_determinism,
    prepare_deterministic_cuda_environment,
)
from gods_watching.training.memory import (
    assess_admission,
    estimate_memory,
    GpuSnapshot,
    MemoryProfile,
    SUPPORTED_CUDA_VERSION,
    SUPPORTED_IMAGE_SIZE,
    SUPPORTED_MODEL_ID,
    SUPPORTED_MODEL_REVISION,
    SUPPORTED_PROCESSOR_IDENTITY,
    SUPPORTED_TEXT_MAX_LENGTH,
    SUPPORTED_TORCH_VERSION,
    model_directory_sha256,
)


class _TextTowerConfig(Protocol):
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int


class _VisionTowerConfig(_TextTowerConfig, Protocol):
    patch_size: int


class _ClipModelConfig(Protocol):
    vision_config: _VisionTowerConfig
    text_config: _TextTowerConfig


def _capture_gpu_snapshot(gpu_index: int | str) -> GpuSnapshot:
    """Read UUID/total/free from NVML through the pinned ``nvidia-smi`` command."""
    command = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=uuid,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)  # noqa: S603
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("nvidia-smi did not return exactly one selected GPU")
    fields = [part.strip() for part in rows[0].split(",")]
    if len(fields) != 3:
        raise RuntimeError("nvidia-smi returned an unexpected memory record")
    uuid, total_mib, free_mib = fields
    mib = 1024**2
    return GpuSnapshot(
        uuid=uuid,
        total_bytes=int(total_mib) * mib,
        free_bytes=int(free_mib) * mib,
        observed_at=datetime.now(UTC),
    )


def _model_shape(model_config: object) -> CalibrationModelShape:
    """Extract the two CLIP towers' activation dimensions from local config."""
    config = cast("_ClipModelConfig", model_config)
    vision = config.vision_config
    text = config.text_config
    return CalibrationModelShape(
        vision_layers=vision.num_hidden_layers,
        vision_hidden_size=vision.hidden_size,
        vision_heads=vision.num_attention_heads,
        text_layers=text.num_hidden_layers,
        text_hidden_size=text.hidden_size,
        text_heads=text.num_attention_heads,
        patch_size=vision.patch_size,
    )


def _optimizer_step_count(optimizer: object, torch_module: object) -> int:
    """Read AdamW's per-parameter step counters without guessing from scaler state."""
    torch = cast("object", torch_module)
    states = cast("object", optimizer.state)
    step_counts = [
        int(step.item() if torch.is_tensor(step) else step)
        for state in states.values()
        for step in state.values()
        if "step" in state and step is state["step"]
    ]
    return max(step_counts, default=0)


def _validate_locked_base_package(model_root: Path, lock_path: Path) -> str:
    """Verify every B/16 file against the repository lock before loading weights."""
    lock = load_models_lock(lock_path)
    matches = tuple(model for model in lock.models if model.model_id == SUPPORTED_MODEL_ID)
    if len(matches) != 1 or matches[0].revision != SUPPORTED_MODEL_REVISION:
        raise CalibrationRefusedError("assets lock does not pin the supported B/16 revision")
    base_model = matches[0]
    base_lock = ModelsLock(
        schema_version=lock.schema_version,
        container=lock.container,
        models=(base_model,),
    )
    _ = validate_model_assets(base_lock, model_root.parent)

    expected_files = {file.path.relative_to(Path("clip")).as_posix() for file in base_model.files}
    actual_files: set[str] = set()
    for path in model_root.rglob("*"):
        if path.is_symlink():
            raise CalibrationRefusedError("B/16 package contains a symlink")
        if not path.is_file():
            continue
        relative_path = path.relative_to(model_root).as_posix()
        if relative_path == "gods-watching-model.json":
            expected_marker = {
                "model_id": SUPPORTED_MODEL_ID,
                "revision": SUPPORTED_MODEL_REVISION,
                "dimension": 512,
                "processor": "CLIPProcessor",
                "runtime": "transformers",
            }
            try:
                marker = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CalibrationRefusedError("B/16 identity marker is unreadable") from exc
            if marker != expected_marker:
                raise CalibrationRefusedError("B/16 identity marker does not match the registry")
            continue
        actual_files.add(relative_path)
    if actual_files != expected_files:
        raise CalibrationRefusedError("B/16 package contents differ from the committed model lock")
    _ = builtin_base_load_options(model_root)
    return model_directory_sha256(model_root)


def _run_calibration_child(  # noqa: C901, PLR0912, PLR0915
    config: TrainingConfig,
    model_root: Path,
    lock_path: Path,
    incoming_gpu: GpuSnapshot,
) -> MemoryProfile:
    """Measure actual image/text backward and two AdamW state-initializing steps."""
    prepare_deterministic_cuda_environment()
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

    configure_torch_determinism(torch)
    if not torch.cuda.is_available():
        raise CalibrationRefusedError("CUDA is unavailable in the calibration child")
    if torch.__version__ != SUPPORTED_TORCH_VERSION:
        raise CalibrationRefusedError("calibration requires the pinned Torch runtime")
    if torch.version.cuda != SUPPORTED_CUDA_VERSION:
        raise CalibrationRefusedError("calibration requires the pinned CUDA runtime")
    if torch.cuda.device_count() != 1:
        raise CalibrationRefusedError("calibration child must see exactly one GPU")

    torch.cuda.init()
    device = torch.device("cuda:0")
    initial_nvidia = _capture_gpu_snapshot(incoming_gpu.uuid)
    _ = bind_cuda_device_uuid(
        incoming_gpu.uuid,
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        initial_nvidia.uuid,
    )
    initial_gpu = capture_cuda_snapshot(
        incoming_gpu.uuid,
        initial_nvidia,
        lambda: torch.cuda.mem_get_info(device),
    )
    used_before_context = incoming_gpu.total_bytes - incoming_gpu.free_bytes
    baseline_device_used = initial_gpu.total_bytes - initial_gpu.free_bytes
    baseline_reserved = torch.cuda.memory_reserved(device)

    model_sha256 = _validate_locked_base_package(model_root, lock_path)
    processor = CLIPProcessor.from_pretrained(str(model_root), local_files_only=True)
    model = CLIPModel.from_pretrained(
        str(model_root),
        local_files_only=True,
        trust_remote_code=False,
        **builtin_base_load_options(model_root).model_dump(),
    )
    if model.config.vision_config.image_size != SUPPORTED_IMAGE_SIZE:
        raise CalibrationRefusedError("pinned B/16 model image size does not match")
    shape = _model_shape(model.config)
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    current_nvidia = _capture_gpu_snapshot(incoming_gpu.uuid)
    _ = bind_cuda_device_uuid(
        incoming_gpu.uuid,
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        current_nvidia.uuid,
    )
    current_gpu = capture_cuda_snapshot(
        incoming_gpu.uuid,
        current_nvidia,
        lambda: torch.cuda.mem_get_info(device),
    )
    headroom = check_calibration_headroom(
        config,
        shape,
        current_gpu,
        parameter_bytes=parameter_bytes,
    )
    if not headroom.admitted:
        raise CalibrationRefusedError(
            "calibration refused before CUDA model transfer: "
            f"required={headroom.required_free_bytes} free={headroom.current_free_bytes}"
        )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config.weight_decay,
        foreach=False,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.mixed_precision == "fp16",
    )

    images = [
        Image.new("RGB", (SUPPORTED_IMAGE_SIZE, SUPPORTED_IMAGE_SIZE), (127, 127, 127))
        for _ in range(config.micro_batch_size)
    ]
    synthetic_texts = [
        "person wearing a blue jacket and dark trousers" for _ in range(config.micro_batch_size)
    ]
    encoded = processor(
        images=images,
        text=synthetic_texts,
        padding="max_length",
        truncation=True,
        max_length=SUPPORTED_TEXT_MAX_LENGTH,
        return_tensors="pt",
    )
    batch = {key: value.to(device) for key, value in encoded.items()}

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    successful_updates = 0
    attempts = 0
    while successful_updates < 2:
        if attempts >= MAX_CALIBRATION_OPTIMIZER_ATTEMPTS:
            raise CalibrationRefusedError(
                "GradScaler did not complete two optimizer updates within the bounded attempts"
            )
        attempts += 1
        optimizer.zero_grad(set_to_none=True)
        scale_before = float(scaler.get_scale())
        step_before = _optimizer_step_count(optimizer, torch)
        # Two backward passes retain one full gradient tensor while the next
        # micro-batch activations are live. More accumulation passes do not
        # retain computation graphs or allocate additional gradient buffers.
        for _ in range(2):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=config.mixed_precision == "fp16",
            ):
                output = model(**batch, return_loss=True)
                loss = output.loss
            if loss is None or not bool(torch.isfinite(loss).all()):
                raise CalibrationRefusedError("calibration loss is not finite")
            scaler.scale(loss / 2).backward()
            del output, loss
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        scale_after = float(scaler.get_scale())
        step_after = _optimizer_step_count(optimizer, torch)
        if optimizer_update_succeeded(
            scale_before,
            scale_after,
            step_before,
            step_after,
        ):
            successful_updates += 1

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    peak_nvidia = _capture_gpu_snapshot(incoming_gpu.uuid)
    _ = bind_cuda_device_uuid(
        incoming_gpu.uuid,
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        peak_nvidia.uuid,
    )
    peak_snapshot = capture_cuda_snapshot(
        incoming_gpu.uuid,
        peak_nvidia,
        lambda: torch.cuda.mem_get_info(device),
    )
    current_used = peak_snapshot.total_bytes - peak_snapshot.free_bytes
    reserved_delta = peak_reserved - baseline_reserved
    peak_device_used = max(current_used, baseline_device_used + reserved_delta)
    gradient_bytes = sum(
        parameter.grad.numel() * parameter.grad.element_size()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    optimizer_state_bytes = sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    activation_workspace_bytes = max(
        0,
        reserved_delta - parameter_bytes - gradient_bytes - optimizer_state_bytes,
    )
    context_bytes = max(0, baseline_device_used - used_before_context)

    profile = build_memory_profile(
        config,
        current_gpu,
        torch_version=torch.__version__,
        cuda_version=cast("str", torch.version.cuda),
        processor_identity=SUPPORTED_PROCESSOR_IDENTITY,
        model_sha256=model_sha256,
        baseline_reserved_bytes=baseline_reserved,
        peak_reserved_bytes=peak_reserved,
        peak_allocated_bytes=peak_allocated,
        baseline_device_used_bytes=baseline_device_used,
        peak_device_used_bytes=peak_device_used,
        parameter_bytes=parameter_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_state_bytes=optimizer_state_bytes,
        activation_workspace_bytes=activation_workspace_bytes,
        context_bytes=context_bytes,
    )
    del batch, encoded, images, model, optimizer, processor, scaler
    torch.cuda.empty_cache()
    return profile


def _child_main(args: argparse.Namespace) -> int:
    config = TrainingConfig.model_validate_json(args.config_json)
    gpu = GpuSnapshot.model_validate_json(args.gpu_snapshot_json)
    profile = _run_calibration_child(
        config,
        Path(args.model_root),
        Path(args.model_lock),
        gpu,
    )
    _ = sys.stdout.write(profile.model_dump_json() + "\n")
    return 0


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as temporary:
        _ = temporary.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    _ = temporary_path.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _profile_configs(args: argparse.Namespace) -> tuple[TrainingConfig, ...]:
    """Build the exact calibration matrix without requiring CUDA or model files."""
    checkpoint_modes = (
        (True, False) if args.checkpointing == "both" else (args.checkpointing == "on",)
    )
    return tuple(
        TrainingConfig(
            micro_batch_size=batch_size,
            mixed_precision=precision,
            gradient_checkpointing=checkpointing,
        )
        for batch_size in args.batch_sizes
        for precision in args.precisions
        for checkpointing in checkpoint_modes
    )


def _run_matrix(args: argparse.Namespace) -> int:
    configs = _profile_configs(args)
    if args.plan_only:
        plan = [config.model_dump(mode="json") for config in configs]
        _ = sys.stdout.write(json.dumps(plan, sort_keys=True) + "\n")
        return 0

    model_root = Path(args.model_root).resolve(strict=True)
    if model_root != Path("/models/clip"):
        raise RuntimeError("the calibrated model path must be the read-only /models/clip mount")
    model_lock = Path(args.model_lock).resolve(strict=True)
    profiles: list[dict[str, object]] = []
    measured_profiles: list[MemoryProfile] = []

    for config in configs:
        gpu = _capture_gpu_snapshot(args.gpu_index)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--model-root",
            str(model_root),
            "--model-lock",
            str(model_lock),
            "--config-json",
            config.model_dump_json(),
            "--gpu-snapshot-json",
            gpu.model_dump_json(),
        ]
        environment = os.environ.copy()
        existing_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            item for item in (str(SERVER_SOURCE), existing_path) if item
        )
        environment["TOKENIZERS_PARALLELISM"] = "false"
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        environment["CUDA_VISIBLE_DEVICES"] = gpu.uuid
        completed = subprocess.run(  # noqa: S603
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode != 0:
            _ = sys.stderr.write(completed.stderr)
            raise RuntimeError(
                "calibration child failed for "
                f"batch={config.micro_batch_size}, precision={config.mixed_precision}, "
                f"checkpointing={config.gradient_checkpointing}; no profile was published"
            )
        profile = MemoryProfile.model_validate_json(completed.stdout.strip())
        measured_profiles.append(profile)
        profiles.append(cast("dict[str, object]", profile.model_dump(mode="json")))

    for profile in measured_profiles:
        config = TrainingConfig(
            micro_batch_size=profile.batch_size,
            mixed_precision=profile.mixed_precision,
            gradient_checkpointing=profile.gradient_checkpointing,
        )
        current_gpu = _capture_gpu_snapshot(args.gpu_index)
        estimate = estimate_memory(config, profile, current_gpu)
        admission = assess_admission(estimate, current_gpu)
        _ = sys.stdout.write(
            json.dumps(
                {
                    "profile_identity": estimate.profile_identity,
                    "training_peak_bytes": estimate.training_peak_bytes,
                    "reserve_bytes": estimate.reserve_bytes,
                    "required_bytes": estimate.required_bytes,
                    "free_bytes": admission.free_bytes,
                    "admitted_now": admission.admitted,
                },
                sort_keys=True,
            )
            + "\n"
        )

    payload: dict[str, object] = {
        "schema_version": 1,
        "model_root": str(model_root),
        "generated_at": datetime.now(UTC).isoformat(),
        "profiles": profiles,
    }
    _atomic_write(Path(args.output), payload)
    _ = sys.stdout.write(f"published {len(profiles)} measured memory profiles to {args.output}\n")
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", default="/models/clip")
    parser.add_argument(
        "--model-lock",
        default=str(REPOSITORY_ROOT / "assets" / "models.lock.json"),
    )
    parser.add_argument("--output", default="/runs/training/memory-profiles.json")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16])
    parser.add_argument(
        "--precisions", choices=("fp16", "fp32"), nargs="+", default=["fp16", "fp32"]
    )
    parser.add_argument("--checkpointing", choices=("on", "off", "both"), default="both")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--config-json")
    parser.add_argument("--gpu-snapshot-json")
    return parser.parse_args()


def main() -> int:
    """Run measured profiles as isolated, fail-closed CUDA child processes."""
    args = _arguments()
    if args.child:
        if not args.config_json or not args.gpu_snapshot_json:
            raise RuntimeError("calibration child requires config and GPU snapshot JSON")
        return _child_main(args)
    return _run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
