"""Build the pinned Triton image and prepare exact model snapshots."""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, override

from pydantic import BaseModel, ConfigDict

from gods_watching.model_selection.registry import ClipModelPackage, ClipModelRegistry

from .models import (
    LockedModel,
    LockedModelFile,
    ModelsLock,
    load_models_lock,
    stream_sha256,
    validate_model_assets,
    validate_registry_agreement,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

IDENTITY_MARKER_NAME: Final = "gods-watching-model.json"
_DOWNLOAD_TIMEOUT_SECONDS: Final = 1_800
_YOLO_MODEL_ID: Final = "ultralytics/yolo11s"
_UNIT_NORM_TOLERANCE: Final = 1e-3


@dataclass(frozen=True, slots=True)
class PreparationPaths:
    """Group paths needed by model preparation."""

    repository_root: Path
    lock_path: Path
    assets_root: Path
    build_image: bool = True
    image: str | None = None
    source_image: str | None = None
    clear_markers: bool = True


class PreparationCommandError(RuntimeError):
    """Report a failed external preparation command without hiding stderr."""

    name: str
    return_code: int
    stderr: str

    def __init__(self, *, name: str, return_code: int, stderr: str) -> None:
        """Capture the command identity, exit status, and diagnostic output."""
        self.name = name
        self.return_code = return_code
        self.stderr = stderr.strip()
        super().__init__(name, return_code, self.stderr)

    @override
    def __str__(self) -> str:
        message = f"{self.name} failed with exit code {self.return_code}"
        return f"{message}: {self.stderr}" if self.stderr else message


class GpuProofSemanticError(RuntimeError):
    """Report a parseable GPU proof that does not satisfy the pinned contract."""


class GpuModelProof(BaseModel):
    """Record dimensions and unit-norm checks for one loaded CLIP package."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    revision: str
    dimension: int
    image_dimension: int
    text_dimension: int
    image_norm: float
    text_norm: float


class PreparedManifest(BaseModel):
    """Record proof emitted only after the GPU and asset gates pass."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    lock_sha256: str
    dockerfile_sha256: str
    inference_lock_sha256: str
    image_id: str
    base_digest: str
    python_abi: str
    torch_version: str
    torchvision_version: str
    cuda_version: str
    cuda_device: str
    processor: str
    clip_class: str
    yolo_class: str
    cuda_available: bool = False
    cuda_operation: float = 0.0
    detector_resident: bool = False
    files_validated: int
    model_proofs: tuple[GpuModelProof, ...] = ()


class GpuProof(BaseModel):
    """Parse the CUDA proof returned by the built image."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    python_abi: str
    torch_version: str
    torchvision_version: str
    cuda_version: str
    cuda_device: str
    cuda_available: bool
    cuda_operation: float = 0.0
    processor: str = ""
    clip_class: str = ""
    yolo_class: str = ""
    detector_resident: bool = False
    models: tuple[GpuModelProof, ...] = ()


def parse_gpu_proof(output: str) -> GpuProof:
    """Parse the final JSON line after container banner notices."""
    lines = output.splitlines()
    if not lines:
        message = "GPU proof output is empty"
        raise GpuProofSemanticError(message) from None
    return GpuProof.model_validate_json(lines[-1])


def validate_gpu_proof(  # noqa: C901
    proof: GpuProof,
    expected_packages: Iterable[ClipModelPackage] = (),
) -> None:
    """Reject proof fields that differ from the pinned GPU environment."""
    requirements = (
        ("cuda_available must be true", proof.cuda_available, True),
        ("cuda_operation must equal 1.0", proof.cuda_operation, 1.0),
        ("python_abi must equal cp312", proof.python_abi, "cp312"),
        ("torch_version must equal 2.7.1+cu128", proof.torch_version, "2.7.1+cu128"),
        ("torchvision_version must equal 0.22.1+cu128", proof.torchvision_version, "0.22.1+cu128"),
        ("cuda_version must equal 12.8", proof.cuda_version, "12.8"),
        ("cuda_device must be nonempty", bool(proof.cuda_device.strip()), True),
        ("processor must equal CLIPProcessor", proof.processor, "CLIPProcessor"),
        ("clip_class must equal CLIPModel", proof.clip_class, "CLIPModel"),
        ("yolo_class must equal YOLO", proof.yolo_class, "YOLO"),
    )
    violations = [message for message, actual, expected in requirements if actual != expected]
    packages = tuple(expected_packages)
    if packages:
        observed = {model.model_id: model for model in proof.models}
        expected_ids = {package.model_id for package in packages}
        if set(observed) != expected_ids:
            violations.append("GPU proof must cover every prepared CLIP package")
        for package in packages:
            model = observed.get(package.model_id)
            if model is None:
                continue
            if model.revision != package.revision:
                violations.append(f"GPU proof revision mismatch for {package.model_id}")
            if model.dimension != package.dimension:
                violations.append(f"GPU proof dimension mismatch for {package.model_id}")
            if model.image_dimension != package.dimension:
                violations.append(f"GPU image dimension mismatch for {package.model_id}")
            if model.text_dimension != package.dimension:
                violations.append(f"GPU text dimension mismatch for {package.model_id}")
            if not _unit_norm(model.image_norm) or not _unit_norm(model.text_norm):
                violations.append(f"GPU norm mismatch for {package.model_id}")
        if not proof.detector_resident:
            violations.append("GPU proof detector must remain resident")
    if violations:
        raise GpuProofSemanticError("GPU proof semantic gate failed: " + "; ".join(violations))


def run_preparation_command(command: tuple[str, ...], *, cwd: Path, name: str) -> str:
    """Run a bounded preparation command and preserve its stderr on failure."""
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=_DOWNLOAD_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise PreparationCommandError(
            name=name, return_code=completed.returncode, stderr=completed.stderr
        )
    return completed.stdout.strip()


def prepare_model_assets(paths: PreparationPaths) -> PreparedManifest:
    """Prepare all locked snapshots, validate them, and prove real CUDA offline."""
    manifest_path = paths.assets_root / "prepared-manifest.json"
    manifest_path.unlink(missing_ok=True)
    lock = load_models_lock(paths.lock_path)
    registry = ClipModelRegistry()
    validate_registry_agreement(lock, registry)
    image = paths.image or lock.container.built_image
    _ensure_asset_root(paths, image)
    if paths.clear_markers:
        _clear_identity_markers(paths.assets_root, registry)
    if paths.build_image:
        _reuse_yolo_before_build(paths, lock, paths.source_image or image)
        _build_triton_image(paths, image)
    _materialize_assets(paths, lock, paths.source_image or image)
    validated = validate_model_assets(lock, paths.assets_root)
    builtin_packages = tuple(
        package
        for package in registry.packages
        if any(model.model_id == package.model_id for model in lock.models)
    )
    from gods_watching.model_selection.registry import load_clip_registry  # noqa: PLC0415

    imported_packages = tuple(
        package
        for package in load_clip_registry(paths.assets_root).packages
        if package.snapshot_path.parent == Path("/models/imported")
    )
    expected_packages = builtin_packages + imported_packages
    proof_text = run_preparation_command(
        (
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--gpus",
            "device=0",
            "--mount",
            f"type=bind,src={paths.assets_root},dst=/models,readonly",
            image,
            "python",
            "-c",
            _gpu_proof_program(expected_packages),
        ),
        cwd=paths.repository_root,
        name="CUDA proof",
    )
    proof = parse_gpu_proof(proof_text)
    validate_gpu_proof(proof, expected_packages)
    image_id = run_preparation_command(
        ("docker", "image", "inspect", image, "--format", "{{.Id}}"),
        cwd=paths.repository_root,
        name="image inspection",
    )
    _publish_identity_markers(paths, builtin_packages, image)
    manifest = PreparedManifest(
        schema_version="1",
        lock_sha256=_sha256_small(paths.lock_path),
        dockerfile_sha256=_sha256_small(paths.repository_root / "deploy/Dockerfile.triton"),
        inference_lock_sha256=_sha256_small(paths.repository_root / "inference/uv.lock"),
        image_id=image_id,
        base_digest=lock.container.base_digest,
        python_abi=proof.python_abi,
        torch_version=proof.torch_version,
        torchvision_version=proof.torchvision_version,
        cuda_version=proof.cuda_version,
        cuda_device=proof.cuda_device,
        processor=proof.processor,
        clip_class=proof.clip_class,
        yolo_class=proof.yolo_class,
        cuda_available=proof.cuda_available,
        cuda_operation=proof.cuda_operation,
        detector_resident=proof.detector_resident,
        files_validated=len(validated),
        model_proofs=proof.models,
    )
    temporary_manifest = manifest_path.with_suffix(".tmp")
    _ = temporary_manifest.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _ = temporary_manifest.replace(manifest_path)
    return manifest


def invalidate_model_markers(paths: PreparationPaths) -> None:
    """Invalidate prepared identities before a lifecycle image build begins."""
    lock = load_models_lock(paths.lock_path)
    registry = ClipModelRegistry()
    validate_registry_agreement(lock, registry)
    image = paths.image or lock.container.built_image
    _ensure_asset_root(paths, image)
    _clear_identity_markers(paths.assets_root, registry)


def _materialize_assets(paths: PreparationPaths, lock: ModelsLock, image: str) -> None:
    for model in lock.models:
        _finalize_matching_partials(paths.assets_root, model)
        invalid_files = _invalid_model_files(paths.assets_root, model)
        if not invalid_files:
            continue
        if model.model_id.startswith("openai/clip-"):
            _download_clip_snapshot(paths, model, image, invalid_files)
        elif model.model_id == _YOLO_MODEL_ID and not _copy_yolo_from_existing_image(
            paths, model, image
        ):
            _download_yolo(paths, model)


def _reuse_yolo_before_build(paths: PreparationPaths, lock: ModelsLock, image: str) -> None:
    """Reuse a legacy detector image without making it a preparation prerequisite."""
    for model in lock.models:
        if model.model_id == _YOLO_MODEL_ID:
            _ = _copy_yolo_from_existing_image(paths, model, image)
            return


def reuse_existing_yolo_asset(paths: PreparationPaths) -> bool:
    """Copy the locked detector weights out of an existing image, if present."""
    lock = load_models_lock(paths.lock_path)
    image = paths.source_image or paths.image or lock.container.built_image
    for model in lock.models:
        if model.model_id == _YOLO_MODEL_ID:
            _ensure_asset_root(paths, image)
            return _copy_yolo_from_existing_image(paths, model, image)
    return False


def _ensure_asset_root(paths: PreparationPaths, image: str) -> None:
    was_present = paths.assets_root.exists()
    try:
        paths.assets_root.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        if was_present:
            raise PreparationCommandError(
                name="asset directory setup",
                return_code=1,
                stderr=f"existing model directory is not writable: {paths.assets_root}",
            ) from None
        parent = paths.assets_root.parent
        owner = f"{os.getuid()}:{os.getgid()}"
        command = (
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            "0",
            "--mount",
            f"type=bind,src={parent},dst=/host",
            image,
            "sh",
            "-c",
            "mkdir -p /host/"
            + paths.assets_root.name
            + " && chown "
            + owner
            + " /host/"
            + paths.assets_root.name,
        )
        _ = run_preparation_command(
            command, cwd=paths.repository_root, name="asset directory setup"
        )
    if not paths.assets_root.is_dir():
        raise PreparationCommandError(
            name="asset directory setup", return_code=1, stderr=f"missing: {paths.assets_root}"
        )
    if not os.access(paths.assets_root, os.W_OK):
        raise PreparationCommandError(
            name="asset directory setup",
            return_code=1,
            stderr=f"model directory is not writable: {paths.assets_root}",
        )


def _build_triton_image(paths: PreparationPaths, image: str) -> None:
    _ = run_preparation_command(
        (
            "docker",
            "build",
            "--progress=plain",
            "-f",
            "deploy/Dockerfile.triton",
            "-t",
            image,
            ".",
        ),
        cwd=paths.repository_root,
        name="triton image build",
    )


def _model_root(model: LockedModel) -> Path:
    if not model.files:
        return Path(model.model_id.rsplit("/", maxsplit=1)[-1])
    first = Path(model.files[0].path)
    if first.is_absolute() or not first.parts:
        return Path(model.model_id.rsplit("/", maxsplit=1)[-1])
    return Path(first.parts[0])


def _invalid_model_files(assets_root: Path, model: LockedModel) -> tuple[LockedModelFile, ...]:
    """Return only files that need repair, preserving valid large weights."""
    invalid: list[LockedModelFile] = []
    for locked_file in model.files:
        candidate = assets_root / locked_file.path
        if not _file_matches(candidate, locked_file):
            invalid.append(locked_file)
    return tuple(invalid)


def _finalize_matching_partials(assets_root: Path, model: LockedModel) -> None:
    for locked_file in model.files:
        candidate = assets_root / locked_file.path
        partial = candidate.with_name(candidate.name + ".partial")
        if _file_matches(candidate, locked_file) or not partial.is_file():
            continue
        try:
            metadata = partial.lstat()
            digest = stream_sha256(partial)
        except OSError:
            continue
        if metadata.st_size == locked_file.size and digest == locked_file.sha256:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            _ = partial.replace(candidate)


def _file_matches(path: Path, locked_file: LockedModelFile) -> bool:
    expected_size = locked_file.size
    expected_digest = locked_file.sha256
    try:
        metadata = path.lstat()
        if path.is_symlink() or not path.is_file() or metadata.st_size != expected_size:
            return False
        return stream_sha256(path) == expected_digest
    except OSError:
        return False


def _download_clip_snapshot(
    paths: PreparationPaths,
    model: LockedModel,
    image: str,
    invalid_files: tuple[LockedModelFile, ...],
) -> None:
    model_root = _model_root(model)
    files = tuple(Path(item.path).relative_to(model_root).as_posix() for item in invalid_files)
    force_download = any(
        (paths.assets_root / locked_file.path).is_file() for locked_file in invalid_files
    )
    download_options = ("--force-download",) if force_download else ()
    command = (
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--env",
        "HF_HUB_DISABLE_XET=1",
        "--env",
        "HF_HOME=/tmp/hf",
        "--mount",
        f"type=bind,src={paths.assets_root},dst=/models",
        image,
        "hf",
        "download",
        model.model_id,
        *files,
        "--revision",
        model.revision,
        "--local-dir",
        f"/models/{model_root}",
        *download_options,
    )
    _ = run_preparation_command(
        command, cwd=paths.repository_root, name=f"{model.model_id} download"
    )


def _copy_yolo_from_existing_image(paths: PreparationPaths, model: LockedModel, image: str) -> bool:
    if not model.files:
        return False
    target = paths.assets_root / model.files[0].path
    if _file_matches(target, model.files[0]):
        return True
    partial = target.with_name(target.name + ".partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        container_id = run_preparation_command(
            ("docker", "create", image), cwd=paths.repository_root, name="asset source container"
        )
        if not container_id:
            return False
        try:
            _ = run_preparation_command(
                ("docker", "cp", f"{container_id}:/models/yolo/yolo11s.pt", str(partial)),
                cwd=paths.repository_root,
                name="YOLO cache copy",
            )
        finally:
            _ = run_preparation_command(
                ("docker", "rm", container_id),
                cwd=paths.repository_root,
                name="asset source cleanup",
            )
    except (PreparationCommandError, OSError):
        return False
    _finalize_matching_partials(paths.assets_root, model)
    return _file_matches(target, model.files[0])


def _download_yolo(paths: PreparationPaths, model: LockedModel) -> None:
    if not model.files:
        return
    locked_file = model.files[0]
    target = paths.assets_root / locked_file.path
    partial = target.with_name(target.name + ".partial")
    if partial.is_file() and not _file_matches(partial, locked_file):
        _ = partial.unlink()
    _ = run_preparation_command(
        (
            "curl",
            "--fail",
            "--location",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
        ),
        cwd=paths.repository_root,
        name="YOLO download",
    )
    _finalize_matching_partials(paths.assets_root, model)


def _clear_identity_markers(assets_root: Path, registry: ClipModelRegistry) -> None:
    for package in registry.packages:
        snapshot = _snapshot_path(assets_root, package)
        marker = snapshot / IDENTITY_MARKER_NAME
        _ = marker.unlink(missing_ok=True)


def _publish_identity_markers(
    paths: PreparationPaths, packages: Sequence[ClipModelPackage], image: str
) -> None:
    for package in packages:
        marker = _snapshot_path(paths.assets_root, package) / IDENTITY_MARKER_NAME
        payload = {
            "model_id": package.model_id,
            "revision": package.revision,
            "dimension": package.dimension,
            "processor": package.processor,
            "runtime": package.runtime,
        }
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            temporary = marker.with_suffix(".tmp")
            _ = temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            _ = temporary.replace(marker)
        except PermissionError:
            command = (
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--user",
                "0",
                "--mount",
                f"type=bind,src={paths.assets_root},dst=/models",
                image,
                "python",
                "-c",
                "import json,os,tempfile; p="
                + repr(f"/models/{marker.relative_to(paths.assets_root)}")
                + "; d=os.path.dirname(p); os.makedirs(d,exist_ok=True); "
                + "fd,t=tempfile.mkstemp(dir=d); "
                + "os.write(fd,"
                + repr((json.dumps(payload, sort_keys=True) + "\n").encode())
                + "); os.close(fd); os.replace(t,p)",
            )
            _ = run_preparation_command(
                command, cwd=paths.repository_root, name="identity marker publish"
            )


def _snapshot_path(assets_root: Path, package: ClipModelPackage) -> Path:
    try:
        relative = package.snapshot_path.relative_to(Path("/models"))
    except ValueError:
        relative = Path(package.model_id.rsplit("/", maxsplit=1)[-1])
    return assets_root / relative


def _gpu_proof_program(packages: Sequence[ClipModelPackage]) -> str:
    specs = [
        {
            "model_id": package.model_id,
            "revision": package.revision,
            "path": str(_snapshot_path(Path("/models"), package)),
            "dimension": package.dimension,
            "imported": package.snapshot_path.parent == Path("/models/imported"),
        }
        for package in packages
    ]
    lines = [
        "import json,sys,torch,torchvision",
        "from PIL import Image",
        "from transformers import AutoProcessor,CLIPModel",
        "from ultralytics import YOLO",
        f"specs = {specs!r}",
        "available = torch.cuda.is_available()",
        "detector = YOLO('/models/yolo/yolo11s.pt').to('cuda') if available else None",
        "proofs = []",
        "image = Image.new('RGB', (224, 224), (31, 47, 61))",
        "def _normalize(features):",
        "    if not torch.isfinite(features).all():",
        "        raise RuntimeError('non-finite CLIP features')",
        "    norms = torch.linalg.vector_norm(features, dim=-1, keepdim=True)",
        "    if (norms <= 0).any():",
        "        raise RuntimeError('zero-norm CLIP features')",
        "    normalized = features / norms",
        "    if not torch.isfinite(normalized).all():",
        "        raise RuntimeError('non-finite normalized CLIP features')",
        "    return normalized",
        "for spec in specs:",
        """    processor = AutoProcessor.from_pretrained(
        spec['path'], local_files_only=True, trust_remote_code=False
    )""",
        """    loaded = CLIPModel.from_pretrained(
        spec['path'], local_files_only=True, trust_remote_code=False,
        output_loading_info=spec['imported']
    )""",
        "    clip, diagnostics = loaded if spec['imported'] else (loaded, {})",
        "    if any(diagnostics.get(key) for key in (",
        "        'missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'",
        "    )):",
        "        raise RuntimeError('clip_checkpoint_incomplete')",
        "    clip = clip.to('cuda').eval()",
        """    image_inputs = {
        key: value.to('cuda')
        for key, value in processor(images=image, return_tensors='pt').items()
    }""",
        """    text_inputs = {
        key: value.to('cuda')
        for key, value in processor(
            text=['offline proof'], return_tensors='pt', padding=True
        ).items()
    }""",
        "    with torch.inference_mode():",
        "        image_features = clip.get_image_features(**image_inputs)",
        "        text_features = clip.get_text_features(**text_inputs)",
        "    image_features = _normalize(image_features)",
        "    text_features = _normalize(text_features)",
        "    image_norm = float(torch.linalg.vector_norm(image_features, dim=-1).mean().item())",
        "    text_norm = float(torch.linalg.vector_norm(text_features, dim=-1).mean().item())",
        "    proofs.append({",
        "        'model_id': spec['model_id'],",
        "        'revision': spec['revision'],",
        "        'dimension': spec['dimension'],",
        "        'image_dimension': int(image_features.shape[-1]),",
        "        'text_dimension': int(text_features.shape[-1]),",
        "        'image_norm': image_norm,",
        "        'text_norm': text_norm,",
        "    })",
        "    del clip, processor",
        "    torch.cuda.empty_cache()",
        "payload = {",
        "    'python_abi': f'cp{sys.version_info.major}{sys.version_info.minor}',",
        "    'torch_version': torch.__version__,",
        "    'torchvision_version': torchvision.__version__,",
        "    'cuda_version': torch.version.cuda,",
        "    'cuda_device': torch.cuda.get_device_name(0) if available else '',",
        "    'cuda_available': available,",
        "    'cuda_operation': torch.ones(1, device='cuda').item() if available else 0.0,",
        "    'processor': 'CLIPProcessor',",
        "    'clip_class': 'CLIPModel',",
        "    'yolo_class': 'YOLO',",
        "    'detector_resident': detector is not None,",
        "    'models': proofs,",
        "}",
        "print(json.dumps(payload))",
        "raise SystemExit(0 if available else 3)",
    ]
    return "\n".join(lines)


def _unit_norm(value: float) -> bool:
    return math.isfinite(value) and abs(value - 1.0) <= _UNIT_NORM_TOLERANCE


def _sha256_small(path: Path) -> str:
    return stream_sha256(path)


__all__ = [
    "IDENTITY_MARKER_NAME",
    "GpuModelProof",
    "GpuProof",
    "GpuProofSemanticError",
    "PreparationCommandError",
    "PreparationPaths",
    "PreparedManifest",
    "invalidate_model_markers",
    "parse_gpu_proof",
    "prepare_model_assets",
    "reuse_existing_yolo_asset",
    "run_preparation_command",
    "validate_gpu_proof",
]
