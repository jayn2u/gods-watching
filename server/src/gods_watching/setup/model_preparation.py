"""Build the pinned Triton image and prepare exact model snapshots."""

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final, override

from pydantic import BaseModel, ConfigDict

from .models import load_models_lock, validate_model_assets

_CLIP_REVISION: Final = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
_DOWNLOAD_TIMEOUT_SECONDS: Final = 1_800
_CLIP_FILES: Final = (
    "config.json",
    "merges.txt",
    "pytorch_model.bin",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


@dataclass(frozen=True, slots=True)
class PreparationPaths:
    """Group paths needed by model preparation."""

    repository_root: Path
    lock_path: Path
    assets_root: Path


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
    files_validated: int


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


def parse_gpu_proof(output: str) -> GpuProof:
    """Parse the final JSON line after container entrypoint notices."""
    final_line = output.rsplit("\n", maxsplit=1)[-1]
    return GpuProof.model_validate_json(final_line)


def validate_gpu_proof(proof: GpuProof) -> None:
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
    violations = tuple(message for message, actual, expected in requirements if actual != expected)
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
    """Download during preparation, then prove exact assets and real CUDA."""
    manifest_path = paths.assets_root / "prepared-manifest.json"
    manifest_path.unlink(missing_ok=True)
    lock = load_models_lock(paths.lock_path)
    image = lock.container.built_image
    paths.assets_root.mkdir(parents=True, exist_ok=True)
    yolo_dir = paths.assets_root / "yolo"
    yolo_dir.mkdir(exist_ok=True)
    clip_dir = paths.assets_root / "clip"
    clip_dir.mkdir(exist_ok=True)
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
    partial = yolo_dir / "yolo11s.pt.partial"
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
    _ = partial.replace(yolo_dir / "yolo11s.pt")
    clip_command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{clip_dir.resolve()}:/models/clip",
        image,
        "hf",
        "download",
        "openai/clip-vit-base-patch16",
        *_CLIP_FILES,
        "--revision",
        _CLIP_REVISION,
        "--local-dir",
        "/models/clip",
    ]
    _ = run_preparation_command(
        tuple(clip_command), cwd=paths.repository_root, name="CLIP download"
    )
    validated = validate_model_assets(lock, paths.assets_root)
    proof_program = (
        "import json,sys,torch,torchvision; "
        "from transformers import AutoProcessor,CLIPModel; "
        "from ultralytics import YOLO; "
        "available=torch.cuda.is_available(); "
        "processor=AutoProcessor.from_pretrained('/models/clip',local_files_only=True); "
        "clip=CLIPModel.from_pretrained('/models/clip',local_files_only=True).to('cuda'); "
        "yolo=YOLO('/models/yolo/yolo11s.pt').to('cuda'); "
        "cuda_operation=torch.ones(1,device='cuda').item(); "
        "print(json.dumps({"
        "'python_abi':f'cp{sys.version_info.major}{sys.version_info.minor}',"
        "'torch_version':torch.__version__,"
        "'torchvision_version':torchvision.__version__,"
        "'cuda_version':torch.version.cuda,"
        "'cuda_device':torch.cuda.get_device_name(0) if available else '',"
        "'cuda_available':available,"
        "'cuda_operation':cuda_operation,"
        "'processor':processor.__class__.__name__,"
        "'clip_class':clip.__class__.__name__,"
        "'yolo_class':yolo.__class__.__name__})); "
        "raise SystemExit(0 if available else 3)"
    )
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
            proof_program,
        ),
        cwd=paths.repository_root,
        name="CUDA proof",
    )
    proof = parse_gpu_proof(proof_text)
    validate_gpu_proof(proof)
    image_id = run_preparation_command(
        ("docker", "image", "inspect", image, "--format", "{{.Id}}"),
        cwd=paths.repository_root,
        name="image inspection",
    )
    manifest = PreparedManifest(
        schema_version="1",
        lock_sha256=hashlib.sha256(paths.lock_path.read_bytes()).hexdigest(),
        dockerfile_sha256=hashlib.sha256(
            (paths.repository_root / "deploy/Dockerfile.triton").read_bytes()
        ).hexdigest(),
        inference_lock_sha256=hashlib.sha256(
            (paths.repository_root / "inference/uv.lock").read_bytes()
        ).hexdigest(),
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
        files_validated=len(validated),
    )
    temporary_manifest = manifest_path.with_suffix(".tmp")
    _ = temporary_manifest.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _ = temporary_manifest.replace(manifest_path)
    return manifest
