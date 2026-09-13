"""Verify pinned model assets and the real CUDA container baseline."""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Final

from gods_watching.setup.model_preparation import GpuProof, PreparedManifest
from gods_watching.setup.models import AssetValidationError, load_models_lock, validate_model_assets
from gods_watching.verification.models import (
    Check,
    EvidenceKind,
    ImplementedScenario,
    ScenarioContextProtocol,
    ScenarioReport,
)
from gods_watching.verification.registry import parse_scenario_name, register_scenario

_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[5]
_LOCK_PATH: Final = _REPOSITORY_ROOT / "assets/models.lock.json"
_ASSETS_ROOT: Final = _REPOSITORY_ROOT / "runtime/assets/models"
_GPU_TIMEOUT_SECONDS: Final = 600.0


def _gpu_program() -> str:
    return (
        "import json,sys,torch,torchvision; "
        "from pathlib import Path; "
        "from transformers import AutoProcessor,CLIPModel; "
        "from ultralytics import YOLO; "
        "available=torch.cuda.is_available(); "
        "processor=AutoProcessor.from_pretrained('/models/clip',local_files_only=True); "
        "clip=CLIPModel.from_pretrained('/models/clip',local_files_only=True).to('cuda'); "
        "yolo=YOLO('/models/yolo/yolo11s.pt').to('cuda'); "
        "value=torch.ones(1,device='cuda').item(); "
        "proof={'python_abi':f'cp{sys.version_info.major}{sys.version_info.minor}',"
        "'cuda_available':available,'cuda_device':torch.cuda.get_device_name(0),"
        "'cuda_operation':value,'torch_version':torch.__version__,"
        "'torchvision_version':torchvision.__version__,"
        "'cuda_version':torch.version.cuda,'processor':processor.__class__.__name__,"
        "'clip_class':clip.__class__.__name__,'yolo_class':yolo.__class__.__name__}; "
        "Path('/evidence/gpu-proof.json').write_text(json.dumps(proof),encoding='utf-8'); "
        "raise SystemExit(0 if available else 3)"
    )


async def _inspect_image_id(context: ScenarioContextProtocol, image: str) -> str:
    command = ("docker", "image", "inspect", image, "--format", "{{.Id}}")
    async with context.process(name="model-assets-image-inspect", command=command) as process:
        stdout = process.stdout
        image_id = "" if stdout is None else (await stdout.receive()).decode().strip()
        return_code = await process.wait()
    if return_code != 0:
        return ""
    return image_id


async def _run_model_assets(context: ScenarioContextProtocol) -> ScenarioReport:
    lock = load_models_lock(_LOCK_PATH)
    validated = validate_model_assets(lock, _ASSETS_ROOT)
    prepared_path = _ASSETS_ROOT / "prepared-manifest.json"
    prepared = PreparedManifest.model_validate_json(prepared_path.read_text(encoding="utf-8"))
    lock_digest = hashlib.sha256(_LOCK_PATH.read_bytes()).hexdigest()
    dockerfile_digest = hashlib.sha256(
        (_REPOSITORY_ROOT / "deploy/Dockerfile.triton").read_bytes()
    ).hexdigest()
    inference_lock_digest = hashlib.sha256(
        (_REPOSITORY_ROOT / "inference/uv.lock").read_bytes()
    ).hexdigest()
    image_id = await _inspect_image_id(context, lock.container.built_image)
    proof_path = context.run_root / "gpu-proof.json"
    command = (
        "docker",
        "run",
        "--rm",
        "--name",
        f"{context.compose_project}-model-assets",
        "--network",
        "none",
        "--gpus",
        "device=0",
        "--mount",
        f"type=bind,src={_ASSETS_ROOT},dst=/models,readonly",
        "--mount",
        f"type=bind,src={context.run_root},dst=/evidence",
        lock.container.built_image,
        "python",
        "-c",
        _gpu_program(),
    )
    async with context.process(name="model-assets-gpu-container", command=command) as process:
        return_code = await process.wait()
    proof = (
        GpuProof.model_validate_json(proof_path.read_text(encoding="utf-8"))
        if proof_path.is_file()
        else None
    )
    summary_path = context.run_root / "asset-validation.json"
    _ = summary_path.write_text(
        json.dumps(
            {
                "lock_sha256": lock_digest,
                "files_validated": len(validated),
                "image_id": prepared.image_id,
                "base_digest": prepared.base_digest,
                "dockerfile_sha256": dockerfile_digest,
                "inference_lock_sha256": inference_lock_digest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checks = (
        Check(
            name="all-model-files-match-lock",
            passed=len(validated) == sum(len(model.files) for model in lock.models),
            detail=f"validated {len(validated)} locked files",
        ),
        Check(
            name="prepared-manifest-matches-lock",
            passed=(
                prepared.lock_sha256 == lock_digest
                and prepared.base_digest == lock.container.base_digest
                and prepared.python_abi == lock.container.python_abi
                and prepared.image_id == image_id
                and prepared.dockerfile_sha256 == dockerfile_digest
                and prepared.inference_lock_sha256 == inference_lock_digest
            ),
            detail="prepared manifest identifies the current lock, ABI, base and built image",
        ),
        Check(
            name="cuda-model-cache-proof",
            passed=(
                return_code == 0
                and proof is not None
                and proof.cuda_available
                and proof.cuda_operation == 1.0
                and proof.processor == "CLIPProcessor"
                and proof.clip_class == "CLIPModel"
                and proof.yolo_class == "YOLO"
            ),
            detail="CUDA tensor operation and offline YOLO/CLIP/processor loads completed",
        ),
        Check(
            name="approved-pytorch-cuda-stack",
            passed=(
                proof is not None
                and proof.python_abi == "cp312"
                and proof.torch_version == "2.7.1+cu128"
                and proof.torchvision_version == "0.22.1+cu128"
                and proof.cuda_version == "12.8"
            ),
            detail="Python 3.12 and torch/torchvision CUDA 12.8 versions match the lock",
        ),
    )
    return ScenarioReport(
        checks=checks,
        artifact_paths=(summary_path, proof_path),
        evidence_kind=EvidenceKind.REAL,
    )


async def _run_corrupt_model_assets(context: ScenarioContextProtocol) -> ScenarioReport:
    lock = load_models_lock(_LOCK_PATH)
    disposable = context.runtime_root / "models"
    _ = shutil.copytree(_ASSETS_ROOT, disposable, copy_function=os.link)
    target = min(
        (locked_file for model in lock.models for locked_file in model.files),
        key=lambda locked_file: locked_file.size,
    )
    corrupt = disposable / target.path
    corrupt.unlink()
    _ = corrupt.write_bytes(b"corrupt")
    detected = False
    try:
        _ = validate_model_assets(lock, disposable)
    except AssetValidationError as error:
        detected = error.code == "asset_corrupt" and error.path == target.path
    artifact = context.run_root / "corrupt-validation.json"
    _ = artifact.write_text(
        json.dumps({"corrupt_path": str(target.path), "detected": detected}) + "\n",
        encoding="utf-8",
    )
    return ScenarioReport(
        checks=(
            Check(
                name="corrupt-model-file-rejected",
                passed=detected,
                detail="disposable corrupt copy failed locked checksum validation",
            ),
        ),
        artifact_paths=(artifact,),
        evidence_kind=EvidenceKind.SYNTHETIC,
    )


register_scenario(
    ImplementedScenario(
        name=parse_scenario_name("model-assets"),
        runner=_run_model_assets,
        required_commands=("docker",),
        timeout_seconds=_GPU_TIMEOUT_SECONDS,
    )
)
register_scenario(
    ImplementedScenario(
        name=parse_scenario_name("corrupt-model-assets"),
        runner=_run_corrupt_model_assets,
        timeout_seconds=30.0,
    )
)
