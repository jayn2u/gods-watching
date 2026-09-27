import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from gods_watching import lifecycle
from gods_watching.cli import app
from gods_watching.model_selection.assets import PreparedModelCatalog
from gods_watching.model_selection.registry import ClipModelPackage, ClipModelRegistry
from gods_watching.setup.model_preparation import (
    PreparationPaths,
    _gpu_proof_program,
    _publish_identity_markers,
)
from test_prepared_model_catalog import _write_fixture_package


def test_gpu_proof_specs_identify_builtin_and_imported_packages() -> None:
    builtin = ClipModelRegistry().default
    imported = ClipModelPackage(
        model_id="local/cuhk",
        revision="a" * 64,
        snapshot_path=Path("/models/imported") / ("a" * 64),
        dimension=512,
        processor="CLIPProcessor",
        runtime="transformers",
    )
    program = _gpu_proof_program((builtin, imported))
    assert "cuda_device_uuid" in program
    assert "--query-gpu=uuid" not in program
    spec_line = next(line for line in program.splitlines() if line.startswith("specs = "))
    namespace: dict[str, object] = {}
    exec(spec_line, namespace)  # noqa: S102
    specs = namespace["specs"]
    assert [spec["imported"] for spec in specs] == [False, True]


def test_emitted_gpu_proof_helper_normalizes_only_canonical_cuda_uuids() -> None:
    device_uuid = "17913b0a-8144-5f39-7062-15265e5dca33"
    cases = (
        (device_uuid, f"GPU-{device_uuid}"),
        (f"GPU-{device_uuid}", f"GPU-{device_uuid}"),
        (None, ""),
        ("", ""),
        ("not-a-uuid", ""),
        ("GPU-not-a-uuid", ""),
    )
    program = _gpu_proof_program((ClipModelRegistry().default,))
    function = next(
        node
        for node in ast.parse(program).body
        if isinstance(node, ast.FunctionDef) and node.name == "_gpu_uuid"
    )
    helper_source = ast.get_source_segment(program, function)
    assert helper_source is not None

    class FakeCuda:
        def __init__(self, reported_uuid: str | None, device_indices: list[int]) -> None:
            self.reported_uuid = reported_uuid
            self.device_indices = device_indices

        def get_device_properties(self, index: int) -> SimpleNamespace:
            self.device_indices.append(index)
            return SimpleNamespace(uuid=self.reported_uuid)

    for reported_uuid, expected_uuid in cases:
        device_indices: list[int] = []
        namespace: dict[str, object] = {
            "available": True,
            "torch": SimpleNamespace(cuda=FakeCuda(reported_uuid, device_indices)),
        }
        exec(helper_source, namespace)  # noqa: S102

        assert namespace["_gpu_uuid"]() == expected_uuid
        assert device_indices == [0]


def test_published_builtin_marker_remains_prepared(tmp_path: Path) -> None:
    registry, lock, assets, package = _write_fixture_package(tmp_path)
    paths = PreparationPaths(
        repository_root=tmp_path,
        lock_path=lock,
        assets_root=assets,
        build_image=False,
    )
    _publish_identity_markers(paths, (package,), "fixture:image")
    assert PreparedModelCatalog(registry, lock, assets).status(package).prepared


def test_prepare_lifecycle_runs_model_preparation_after_the_compose_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "commands.log"
    docker = tmp_path / "docker"
    _ = docker.write_text(
        '#!/bin/sh\nprintf "docker %s\\n" "$*" >> "$GW_TEST_COMMAND_LOG"\n',
        encoding="utf-8",
    )
    _ = docker.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("GW_TEST_COMMAND_LOG", str(log_path))
    monkeypatch.setattr(lifecycle, "_ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(lifecycle, "_prepare_model_asset_directory", lambda: None)
    monkeypatch.setattr(lifecycle, "_invalidate_model_markers", lambda: None)
    monkeypatch.setattr(lifecycle, "_reuse_existing_yolo_asset", lambda: None)
    prepared: list[str] = []
    monkeypatch.setattr(lifecycle, "_prepare_model_assets", lambda: prepared.append("prepared"))

    result = CliRunner().invoke(app, ["prepare"])

    assert result.exit_code == 0
    assert prepared == ["prepared"]
    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "docker compose config --quiet",
        "docker compose build",
    ]


def test_marker_invalidation_failure_prevents_the_compose_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "commands.log"
    docker = tmp_path / "docker"
    _ = docker.write_text(
        '#!/bin/sh\nprintf "docker %s\\n" "$*" >> "$GW_TEST_COMMAND_LOG"\n',
        encoding="utf-8",
    )
    _ = docker.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("GW_TEST_COMMAND_LOG", str(log_path))
    monkeypatch.setattr(lifecycle, "_ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(lifecycle, "_prepare_model_asset_directory", lambda: None)
    monkeypatch.setattr(lifecycle, "_reuse_existing_yolo_asset", lambda: None)

    def fail_invalidation() -> None:
        raise lifecycle.LifecycleCommandError(command=("model marker invalidation",), exit_code=2)

    monkeypatch.setattr(lifecycle, "_invalidate_model_markers", fail_invalidation)

    result = CliRunner().invoke(app, ["prepare"])

    assert result.exit_code == 2
    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "docker compose config --quiet",
    ]
