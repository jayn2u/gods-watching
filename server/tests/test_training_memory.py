from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest

import gods_watching.training.memory as training_memory
from gods_watching.contracts.training import TrainingConfig
from gods_watching.training.calibration import (
    MAX_CALIBRATION_OPTIMIZER_ATTEMPTS,
    CalibrationRefusedError,
    bind_cuda_device_uuid,
    builtin_base_load_options,
    capture_cuda_snapshot,
    exported_candidate_load_options,
    optimizer_update_succeeded,
)
from gods_watching.training.memory import (
    GpuSnapshot,
    MemoryProfile,
    StaleGpuSnapshotError,
    UnsupportedMemoryProfileError,
    assess_admission,
    current_source_fingerprint,
    estimate_memory,
)

# pyright: reportUnusedFunction=false

_GIB = 1024**3
_GPU_UUID = "GPU-17913b0a-8144-5f39-7062-15265e5dca33"


def _gpu(
    *,
    free_bytes: int = 15 * _GIB,
    uuid: str = _GPU_UUID,
    total_bytes: int = 16 * _GIB,
    observed_at: datetime | None = None,
) -> GpuSnapshot:
    return GpuSnapshot(
        uuid=uuid,
        total_bytes=total_bytes,
        free_bytes=free_bytes,
        observed_at=observed_at or datetime.now(UTC),
    )


def _profile(  # noqa: PLR0913
    *,
    calibrated_at: datetime | None = None,
    gpu_uuid: str = _GPU_UUID,
    gpu_total_bytes: int = 16 * _GIB,
    source_fingerprint: str | None = None,
    mixed_precision: Literal["fp16", "fp32"] = "fp16",
    gradient_checkpointing: bool = True,
    batch_size: int = 16,
    optimizer_state_bytes: int = _GIB,
    torch_version: str = "2.7.1+cu128",
    cuda_version: str = "12.8",
    processor_identity: str = (
        "openai/clip-vit-base-patch16@57c216476eefef5ab752ec549e440a49ae4ae5f3"
    ),
    model_sha256: str = "c" * 64,
    image_size: int = 224,
    text_max_length: int = 77,
) -> MemoryProfile:
    return MemoryProfile(
        gpu_uuid=gpu_uuid,
        gpu_total_bytes=gpu_total_bytes,
        source_fingerprint=source_fingerprint or current_source_fingerprint(),
        torch_version=torch_version,
        cuda_version=cuda_version,
        processor_identity=processor_identity,
        model_sha256=model_sha256,
        mixed_precision=mixed_precision,
        gradient_checkpointing=gradient_checkpointing,
        batch_size=batch_size,
        optimizer_recipe="adamw-betas-0.9-0.999-eps-1e-8-foreach-false-v1",
        image_size=image_size,
        text_max_length=text_max_length,
        baseline_reserved_bytes=512 * 1024**2,
        peak_reserved_bytes=5 * _GIB,
        peak_allocated_bytes=4 * _GIB,
        baseline_device_used_bytes=1 * _GIB,
        peak_device_used_bytes=6 * _GIB,
        parameter_bytes=1 * _GIB,
        gradient_bytes=1 * _GIB,
        optimizer_state_bytes=optimizer_state_bytes,
        activation_workspace_bytes=2 * _GIB,
        context_bytes=512 * 1024**2,
        calibrated_at=calibrated_at or datetime.now(UTC),
    )


@pytest.fixture(autouse=True)
def _mount_matching_test_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training_memory, "current_model_sha256", lambda: "c" * 64)


def test_reserve_is_max_2gib_or_15percent() -> None:
    small_gpu = _gpu(total_bytes=8 * _GIB, free_bytes=7 * _GIB)
    large_gpu = _gpu(total_bytes=20 * _GIB, free_bytes=19 * _GIB)

    small = estimate_memory(TrainingConfig(), _profile(gpu_total_bytes=8 * _GIB), small_gpu)
    large = estimate_memory(TrainingConfig(), _profile(gpu_total_bytes=20 * _GIB), large_gpu)

    assert small.reserve_bytes == 2 * _GIB
    assert large.reserve_bytes == 3 * _GIB


def test_cuda_mem_get_info_uses_free_then_total_and_nvml_capacity() -> None:
    def mock_mem_get_info() -> tuple[int, int]:
        return 11 * _GIB, 15 * _GIB + _GIB // 2

    nvidia_snapshot = _gpu(free_bytes=12 * _GIB)

    snapshot = capture_cuda_snapshot(_GPU_UUID, nvidia_snapshot, mock_mem_get_info)

    assert snapshot.free_bytes == 11 * _GIB
    assert snapshot.total_bytes == nvidia_snapshot.total_bytes


def test_cuda_mem_get_info_rejects_free_memory_above_usable_capacity() -> None:
    with pytest.raises(CalibrationRefusedError, match="invalid free/usable"):
        _ = capture_cuda_snapshot(
            _GPU_UUID,
            _gpu(),
            lambda: (16 * _GIB, 15 * _GIB),
        )


def test_cuda_profile_is_bound_to_selected_physical_gpu_uuid() -> None:
    assert bind_cuda_device_uuid(_GPU_UUID, _GPU_UUID, _GPU_UUID) == _GPU_UUID
    with pytest.raises(CalibrationRefusedError, match="CUDA_VISIBLE_DEVICES"):
        _ = bind_cuda_device_uuid(_GPU_UUID, "GPU-other", _GPU_UUID)
    with pytest.raises(CalibrationRefusedError, match="NVIDIA-SMI"):
        _ = bind_cuda_device_uuid(_GPU_UUID, _GPU_UUID, "GPU-other")


def test_builtin_bin_and_exported_safetensors_loading_stay_separate(tmp_path: Path) -> None:
    base_root = tmp_path / "base"
    base_root.mkdir()
    _ = (base_root / "pytorch_model.bin").write_bytes(b"locked base weights")
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    _ = (candidate_root / "model.safetensors").write_bytes(b"exported candidate")

    base_options = builtin_base_load_options(base_root)
    candidate_options = exported_candidate_load_options(candidate_root)

    assert base_options.use_safetensors is False
    assert base_options.weights_only is True
    assert candidate_options.use_safetensors is True
    assert candidate_options.weights_only is True


def test_amp_overflow_is_skipped_before_a_later_successful_update() -> None:
    first_attempt_succeeded = optimizer_update_succeeded(
        scale_before=65_536,
        scale_after=32_768,
        optimizer_step_before=0,
        optimizer_step_after=0,
    )
    retry_succeeded = optimizer_update_succeeded(
        scale_before=32_768,
        scale_after=32_768,
        optimizer_step_before=0,
        optimizer_step_after=1,
    )

    assert first_attempt_succeeded is False
    assert retry_succeeded is True
    assert MAX_CALIBRATION_OPTIMIZER_ATTEMPTS == 16


def test_unknown_profile_rejected() -> None:
    with pytest.raises(UnsupportedMemoryProfileError, match="no calibrated profile"):
        _ = estimate_memory(TrainingConfig(), None, _gpu())


def test_stale_or_wrong_gpu_profile_rejected() -> None:
    with pytest.raises(UnsupportedMemoryProfileError, match="GPU UUID"):
        _ = estimate_memory(
            TrainingConfig(),
            _profile(),
            _gpu(uuid="GPU-other"),
        )
    with pytest.raises(UnsupportedMemoryProfileError, match="stale"):
        _ = estimate_memory(
            TrainingConfig(),
            _profile(calibrated_at=datetime.now(UTC) - timedelta(days=31)),
            _gpu(),
        )
    with pytest.raises(StaleGpuSnapshotError):
        _ = estimate_memory(
            TrainingConfig(),
            _profile(),
            _gpu(observed_at=datetime.now(UTC) - timedelta(minutes=2)),
        )


def test_free_bytes_change_reverses_admission() -> None:
    profile = _profile()
    enough = estimate_memory(TrainingConfig(), profile, _gpu(free_bytes=12 * _GIB))
    too_little = estimate_memory(TrainingConfig(), profile, _gpu(free_bytes=7 * _GIB))

    assert assess_admission(enough, _gpu(free_bytes=12 * _GIB)).admitted is True
    refused = assess_admission(too_little, _gpu(free_bytes=7 * _GIB))
    assert refused.admitted is False
    assert refused.required_bytes == too_little.required_bytes
    assert refused.free_bytes == 7 * _GIB


def test_optimizer_state_included_in_training_peak() -> None:
    profile = _profile(
        optimizer_state_bytes=3 * _GIB,
    )
    estimate = estimate_memory(TrainingConfig(), profile, _gpu())

    component_bound = (
        profile.parameter_bytes
        + profile.gradient_bytes
        + profile.optimizer_state_bytes
        + profile.activation_workspace_bytes
        + profile.context_bytes
    )
    assert estimate.training_peak_bytes >= component_bound
    assert estimate.training_peak_bytes >= (
        profile.peak_reserved_bytes - profile.baseline_reserved_bytes
    )


def test_profile_identity_ignores_epochs_and_learning_rate() -> None:
    config = TrainingConfig(epochs=30, learning_rate=1e-5)
    profile = _profile()
    baseline = estimate_memory(config, profile, _gpu())
    changed_schedule = estimate_memory(
        TrainingConfig(epochs=50, learning_rate=2e-5),
        profile,
        _gpu(),
    )

    assert baseline.profile_identity == changed_schedule.profile_identity


def test_profile_configuration_must_match_precision_batch_and_checkpointing() -> None:
    with pytest.raises(UnsupportedMemoryProfileError, match="no calibrated profile"):
        _ = estimate_memory(
            TrainingConfig(micro_batch_size=8),
            _profile(),
            _gpu(),
        )


def test_profile_must_match_pinned_runtime_model_and_input_shape() -> None:
    mismatched_profiles = (
        _profile(torch_version="2.8.0+cu128"),
        _profile(cuda_version="12.9"),
        _profile(processor_identity="unapproved/processor"),
        _profile(model_sha256="d" * 64),
        _profile(image_size=336),
        _profile(text_max_length=64),
    )
    for profile in mismatched_profiles:
        with pytest.raises(UnsupportedMemoryProfileError):
            _ = estimate_memory(TrainingConfig(), profile, _gpu())
    with pytest.raises(UnsupportedMemoryProfileError, match="no calibrated profile"):
        _ = estimate_memory(
            TrainingConfig(gradient_checkpointing=False),
            _profile(),
            _gpu(),
        )
    with pytest.raises(UnsupportedMemoryProfileError, match="no calibrated profile"):
        _ = estimate_memory(
            TrainingConfig(mixed_precision="fp32"),
            _profile(),
            _gpu(),
        )


def test_profile_source_fingerprint_must_match(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = _profile()
    monkeypatch.setattr(training_memory, "current_source_fingerprint", lambda: "f" * 64)

    with pytest.raises(UnsupportedMemoryProfileError, match="source fingerprint"):
        _ = estimate_memory(TrainingConfig(), profile, _gpu())
