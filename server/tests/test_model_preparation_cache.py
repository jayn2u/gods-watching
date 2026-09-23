import hashlib
import json
import os
from pathlib import Path

import pytest

from gods_watching.setup.model_preparation import (
    PreparationPaths,
    invalidate_model_markers,
    prepare_model_assets,
)


def test_preparation_reuses_a_valid_cached_package_without_download_or_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = tmp_path / "repository"
    (repository_root / "deploy").mkdir(parents=True)
    _ = (repository_root / "deploy/Dockerfile.triton").write_text(
        "FROM scratch\n", encoding="utf-8"
    )
    (repository_root / "inference").mkdir()
    _ = (repository_root / "inference/uv.lock").write_text("version = 1\n", encoding="utf-8")
    assets_root = tmp_path / "models"
    model_file = assets_root / "model/weights.bin"
    model_file.parent.mkdir(parents=True)
    payload = b"already prepared"
    _ = model_file.write_bytes(payload)
    lock_path = tmp_path / "models.lock.json"
    _ = lock_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "container": {
                    "base_image": "test",
                    "base_digest": "sha256:" + "0" * 64,
                    "built_image": "test:image",
                    "python_abi": "cp312",
                },
                "models": [
                    {
                        "model_id": "example.invalid/model",
                        "revision": "revision-1",
                        "license": "test",
                        "source": "https://example.invalid",
                        "files": [
                            {
                                "path": "model/weights.bin",
                                "sha256": hashlib.sha256(payload).hexdigest(),
                                "size": len(payload),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    proof = json.dumps(
        {
            "python_abi": "cp312",
            "torch_version": "2.7.1+cu128",
            "torchvision_version": "0.22.1+cu128",
            "cuda_version": "12.8",
            "cuda_device": "test-gpu",
            "cuda_available": True,
            "cuda_operation": 1.0,
            "processor": "CLIPProcessor",
            "clip_class": "CLIPModel",
            "yolo_class": "YOLO",
        }
    )
    calls: list[str] = []

    def fake_command(command: tuple[str, ...], *, cwd: Path, name: str) -> str:
        _ = (command, cwd)
        calls.append(name)
        if name == "CUDA proof":
            return proof
        if name == "image inspection":
            return "sha256:test-image"
        message = f"unexpected preparation command: {name}"
        raise AssertionError(message)

    monkeypatch.setattr(
        "gods_watching.setup.model_preparation.run_preparation_command", fake_command
    )

    manifest = prepare_model_assets(
        PreparationPaths(
            repository_root=repository_root,
            lock_path=lock_path,
            assets_root=assets_root,
            build_image=False,
        )
    )

    assert manifest.files_validated == 1
    assert calls == ["CUDA proof", "image inspection"]
    assert (assets_root / "prepared-manifest.json").is_file()


def test_preparation_forces_only_corrupt_hf_files_and_keeps_valid_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = tmp_path / "repository"
    (repository_root / "deploy").mkdir(parents=True)
    _ = (repository_root / "deploy/Dockerfile.triton").write_text(
        "FROM scratch\n", encoding="utf-8"
    )
    (repository_root / "inference").mkdir()
    _ = (repository_root / "inference/uv.lock").write_text("version = 1\n", encoding="utf-8")
    assets_root = tmp_path / "models"
    snapshot = assets_root / "clip-test"
    snapshot.mkdir(parents=True)
    valid_config = b"valid config"
    valid_weights = b"valid weights"
    _ = (snapshot / "config.json").write_bytes(b"corrupt config")
    _ = (snapshot / "pytorch_model.bin").write_bytes(valid_weights)
    metadata = snapshot / ".cache/huggingface/download/config.json.metadata"
    metadata.parent.mkdir(parents=True)
    _ = metadata.write_text("pinned-revision\n", encoding="utf-8")
    lock_path = tmp_path / "models.lock.json"
    _ = lock_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "container": {
                    "base_image": "test",
                    "base_digest": "sha256:" + "0" * 64,
                    "built_image": "test:image",
                    "python_abi": "cp312",
                },
                "models": [
                    {
                        "model_id": "openai/clip-test",
                        "revision": "revision-1",
                        "license": "test",
                        "source": "https://example.invalid",
                        "files": [
                            {
                                "path": "clip-test/config.json",
                                "sha256": hashlib.sha256(valid_config).hexdigest(),
                                "size": len(valid_config),
                            },
                            {
                                "path": "clip-test/pytorch_model.bin",
                                "sha256": hashlib.sha256(valid_weights).hexdigest(),
                                "size": len(valid_weights),
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    proof = json.dumps(
        {
            "python_abi": "cp312",
            "torch_version": "2.7.1+cu128",
            "torchvision_version": "0.22.1+cu128",
            "cuda_version": "12.8",
            "cuda_device": "test-gpu",
            "cuda_available": True,
            "cuda_operation": 1.0,
            "processor": "CLIPProcessor",
            "clip_class": "CLIPModel",
            "yolo_class": "YOLO",
        }
    )
    commands: list[tuple[tuple[str, ...], str]] = []

    def fake_command(command: tuple[str, ...], *, cwd: Path, name: str) -> str:
        _ = cwd
        commands.append((command, name))
        if name.endswith(" download"):
            assert "config.json" in command
            assert "pytorch_model.bin" not in command
            assert "--force-download" in command
            _ = (snapshot / "config.json").write_bytes(valid_config)
            return ""
        if name == "CUDA proof":
            return proof
        if name == "image inspection":
            return "sha256:test-image"
        message = f"unexpected preparation command: {name}"
        raise AssertionError(message)

    monkeypatch.setattr(
        "gods_watching.setup.model_preparation.run_preparation_command", fake_command
    )
    manifest = prepare_model_assets(
        PreparationPaths(
            repository_root=repository_root,
            lock_path=lock_path,
            assets_root=assets_root,
            build_image=False,
        )
    )

    assert manifest.files_validated == 2
    assert [name for _, name in commands] == [
        "openai/clip-test download",
        "CUDA proof",
        "image inspection",
    ]


def test_valid_partial_replaces_a_corrupt_target_before_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = tmp_path / "repository"
    (repository_root / "deploy").mkdir(parents=True)
    _ = (repository_root / "deploy/Dockerfile.triton").write_text(
        "FROM scratch\n", encoding="utf-8"
    )
    (repository_root / "inference").mkdir()
    _ = (repository_root / "inference/uv.lock").write_text("version = 1\n", encoding="utf-8")
    assets_root = tmp_path / "models"
    target = assets_root / "model/file.bin"
    target.parent.mkdir(parents=True)
    payload = b"complete partial payload"
    _ = target.write_bytes(b"corrupt target")
    partial = target.with_name(target.name + ".partial")
    _ = partial.write_bytes(payload)
    lock_path = tmp_path / "models.lock.json"
    _ = lock_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "container": {
                    "base_image": "test",
                    "base_digest": "sha256:" + "0" * 64,
                    "built_image": "test:image",
                    "python_abi": "cp312",
                },
                "models": [
                    {
                        "model_id": "example.invalid/model",
                        "revision": "revision-1",
                        "license": "test",
                        "source": "https://example.invalid",
                        "files": [
                            {
                                "path": "model/file.bin",
                                "sha256": hashlib.sha256(payload).hexdigest(),
                                "size": len(payload),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    proof = json.dumps(
        {
            "python_abi": "cp312",
            "torch_version": "2.7.1+cu128",
            "torchvision_version": "0.22.1+cu128",
            "cuda_version": "12.8",
            "cuda_device": "test-gpu",
            "cuda_available": True,
            "cuda_operation": 1.0,
            "processor": "CLIPProcessor",
            "clip_class": "CLIPModel",
            "yolo_class": "YOLO",
        }
    )
    calls: list[str] = []

    def fake_command(command: tuple[str, ...], *, cwd: Path, name: str) -> str:
        _ = (command, cwd)
        calls.append(name)
        if name == "CUDA proof":
            return proof
        if name == "image inspection":
            return "sha256:test-image"
        message = f"unexpected preparation command: {name}"
        raise AssertionError(message)

    monkeypatch.setattr(
        "gods_watching.setup.model_preparation.run_preparation_command", fake_command
    )
    _ = prepare_model_assets(
        PreparationPaths(
            repository_root=repository_root,
            lock_path=lock_path,
            assets_root=assets_root,
            build_image=False,
        )
    )

    assert target.read_bytes() == payload
    assert not partial.exists()
    assert calls == ["CUDA proof", "image inspection"]


def test_new_cache_directory_fallback_chowns_only_the_created_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    parent = tmp_path / "runtime/assets"
    parent.mkdir(parents=True)
    assets_root = parent / "models"
    lock_path = tmp_path / "models.lock.json"
    _ = lock_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "container": {
                    "base_image": "test",
                    "base_digest": "sha256:" + "0" * 64,
                    "built_image": "test:image",
                    "python_abi": "cp312",
                },
                "models": [],
            }
        ),
        encoding="utf-8",
    )
    original_mkdir = Path.mkdir

    def deny_cache_mkdir(
        path: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
    ) -> None:
        if path == assets_root:
            message = "simulate root-owned parent"
            raise PermissionError(message)
        original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

    commands: list[tuple[str, ...]] = []

    def fake_command(command: tuple[str, ...], *, cwd: Path, name: str) -> str:
        _ = (cwd, name)
        commands.append(command)
        original_mkdir(assets_root, parents=True, exist_ok=True)
        return ""

    parent_before = parent.stat()
    monkeypatch.setattr(Path, "mkdir", deny_cache_mkdir)
    monkeypatch.setattr(
        "gods_watching.setup.model_preparation.run_preparation_command", fake_command
    )

    invalidate_model_markers(
        PreparationPaths(
            repository_root=repository_root,
            lock_path=lock_path,
            assets_root=assets_root,
        )
    )

    assert assets_root.is_dir()
    assert commands
    command = commands[0]
    assert any("chown" in part for part in command)
    assert any(f"{os.getuid()}:{os.getgid()}" in part for part in command)
    assert parent.stat().st_ino == parent_before.st_ino
