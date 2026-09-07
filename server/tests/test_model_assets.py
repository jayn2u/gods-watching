import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from gods_watching.setup.model_preparation import (
    PreparationCommandError,
    PreparationPaths,
    parse_gpu_proof,
    prepare_model_assets,
    run_preparation_command,
)

REPOSITORY_ROOT = Path(__file__).parents[2]


def _write_lock(path: Path, *, expected: bytes) -> None:
    lock = {
        "schema_version": "1",
        "container": {
            "base_image": "nvcr.io/nvidia/tritonserver:25.02-py3",
            "base_digest": "sha256:" + "1" * 64,
            "built_image": "gods-watching-triton:25.02",
            "python_abi": "cp312",
        },
        "models": [
            {
                "model_id": "test/model",
                "revision": "revision-1",
                "license": "test-only",
                "source": "https://example.invalid/model",
                "files": [
                    {
                        "path": "model/file.bin",
                        "sha256": hashlib.sha256(expected).hexdigest(),
                        "size": len(expected),
                    }
                ],
            }
        ],
    }
    _ = path.write_text(json.dumps(lock), encoding="utf-8")


def _validate(lock: Path, assets: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "gods_watching.setup.model_cli",
            "validate",
            "--lock",
            str(lock),
            "--assets",
            str(assets),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_validation_fails_offline_when_model_asset_is_missing(tmp_path: Path) -> None:
    # Given: a valid lock whose required model file is absent
    lock = tmp_path / "models.lock.json"
    assets = tmp_path / "assets"
    assets.mkdir()
    _write_lock(lock, expected=b"expected model bytes")
    # When: offline model validation reads the prepared asset directory
    completed = _validate(lock, assets)
    # Then: it fails with the stable missing-asset code without creating files
    assert completed.returncode == 2
    assert '"code":"asset_missing"' in completed.stdout
    assert tuple(assets.rglob("*")) == ()


def test_validation_fails_offline_when_model_asset_is_corrupt(tmp_path: Path) -> None:
    # Given: a required model file whose bytes differ from the locked checksum
    lock = tmp_path / "models.lock.json"
    assets = tmp_path / "assets"
    model_file = assets / "model/file.bin"
    model_file.parent.mkdir(parents=True)
    _ = model_file.write_bytes(b"corrupt model bytes")
    _write_lock(lock, expected=b"expected model bytes")
    # When: offline model validation hashes the prepared asset directory
    completed = _validate(lock, assets)
    # Then: it fails with the stable corrupt-asset code and preserves the bytes
    assert completed.returncode == 2
    assert '"code":"asset_corrupt"' in completed.stdout
    assert model_file.read_bytes() == b"corrupt model bytes"



def test_preparation_command_failure_preserves_stderr_and_traceback(tmp_path: Path) -> None:
    command = (
        sys.executable,
        "-c",
        "import sys; print('gpu stderr', file=sys.stderr); raise SystemExit(7)",
    )
    with pytest.raises(
        PreparationCommandError, match="GPU proof failed with exit code 7"
    ) as caught:
        _ = run_preparation_command(command, cwd=tmp_path, name="GPU proof")
    assert caught.value.return_code == 7
    assert "gpu stderr" in str(caught.value)


def test_gpu_proof_parser_ignores_container_banner() -> None:
    payload = json.dumps(
        {
            "python_abi": "cp312",
            "torch_version": "2.7.1+cu128",
            "torchvision_version": "0.22.1+cu128",
            "cuda_version": "12.8",
            "cuda_device": "RTX A6000",
            "cuda_available": True,
            "cuda_operation": 1.0,
            "processor": "CLIPProcessor",
            "clip_class": "CLIPModel",
            "yolo_class": "YOLO",
        }
    )
    proof = parse_gpu_proof(f"Triton Inference Server\n{payload}")
    assert proof.cuda_available
    assert proof.cuda_device == "RTX A6000"


def test_preparation_rejects_zero_exit_invalid_gpu_proof_and_withholds_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: valid locked files, a stale completion marker, and the verifier's zero-exit
    # CUDA payload whose parseable fields all report the wrong approved environment.
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    (repository_root / "deploy").mkdir()
    _ = (repository_root / "deploy/Dockerfile.triton").write_text("FROM scratch\n")
    (repository_root / "inference").mkdir()
    _ = (repository_root / "inference/uv.lock").write_text("version = 1\n")
    lock_path = tmp_path / "models.lock.json"
    assets_root = tmp_path / "assets"
    partial_yolo = assets_root / "yolo/yolo11s.pt.partial"
    partial_yolo.parent.mkdir(parents=True)
    _ = partial_yolo.write_bytes(b"locked yolo bytes")
    clip_config = assets_root / "clip/config.json"
    clip_config.parent.mkdir()
    _ = clip_config.write_bytes(b"locked clip bytes")
    lock = {
        "schema_version": "1",
        "container": {
            "base_image": "nvcr.io/nvidia/tritonserver:25.02-py3",
            "base_digest": "sha256:" + "1" * 64,
            "built_image": "gods-watching-triton:25.02",
            "python_abi": "cp312",
        },
        "models": [
            {
                "model_id": "test/models",
                "revision": "revision-1",
                "license": "test-only",
                "source": "https://example.invalid/model",
                "files": [
                    {
                        "path": "yolo/yolo11s.pt",
                        "sha256": hashlib.sha256(b"locked yolo bytes").hexdigest(),
                        "size": len(b"locked yolo bytes"),
                    },
                    {
                        "path": "clip/config.json",
                        "sha256": hashlib.sha256(b"locked clip bytes").hexdigest(),
                        "size": len(b"locked clip bytes"),
                    },
                ],
            }
        ],
    }
    _ = lock_path.write_text(json.dumps(lock), encoding="utf-8")
    manifest_path = assets_root / "prepared-manifest.json"
    _ = manifest_path.write_text("stale completion", encoding="utf-8")
    invalid_proof = json.dumps(
        {
            "python_abi": "cp999",
            "torch_version": "0.bad",
            "torchvision_version": "0.bad",
            "cuda_version": "0",
            "cuda_device": "",
            "cuda_available": False,
            "cuda_operation": 0.0,
            "processor": "WrongProcessor",
            "clip_class": "WrongClip",
            "yolo_class": "WrongYolo",
        }
    )

    def command_output(
        command: tuple[str, ...], *, cwd: Path, name: str
    ) -> str:
        _ = (command, cwd)
        if name == "CUDA proof":
            return invalid_proof
        if name == "image inspection":
            return "sha256:test-image"
        return ""

    monkeypatch.setattr(
        "gods_watching.setup.model_preparation.run_preparation_command", command_output
    )

    # When: preparation receives a successful command status with invalid GPU semantics.
    with pytest.raises(RuntimeError, match="GPU proof semantic gate"):
        _ = prepare_model_assets(
            PreparationPaths(
                repository_root=repository_root,
                lock_path=lock_path,
                assets_root=assets_root,
            )
        )

    # Then: the stale marker was removed and no new completion manifest was emitted.
    assert not manifest_path.exists()
