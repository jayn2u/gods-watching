import hashlib
import json
import os
import struct
from pathlib import Path

import pytest

from gods_watching.model_selection import importer
from gods_watching.model_selection.importer import ClipPackageImportError, import_clip_package


def package(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    payloads = {
        "config.json": {
            "model_type": "clip",
            "architectures": ["CLIPModel"],
            "projection_dim": 512,
            "text_config": {
                "hidden_size": 512,
                "intermediate_size": 2048,
                "num_hidden_layers": 12,
                "num_attention_heads": 8,
                "vocab_size": 49408,
                "max_position_embeddings": 77,
            },
            "vision_config": {
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "image_size": 224,
                "patch_size": 16,
            },
        },
        "preprocessor_config.json": {"do_resize": True, "size": 224},
        "tokenizer_config.json": {"tokenizer_class": "CLIPTokenizer"},
        "special_tokens_map.json": {"unk_token": "<|endoftext|>"},
        "vocab.json": {"a": 0},
        "cuhk-report.json": {
            "dataset_split": "test",
            "protocol": "identity_disjoint",
            "source_checkpoint": "openai/clip-vit-base-patch16",
            "source_checkpoint_revision": "pinned-base-revision",
            "candidate_weights_sha256": "placeholder",
            "evaluation_code_revision": "abc123",
            "metric_definition": "Recall@1",
            "baseline_score": 0.4,
            "candidate_score": 0.5,
        },
    }
    for name, value in payloads.items():
        (source / name).write_text(json.dumps(value))
    (source / "merges.txt").write_text("#version: 0.2\na b\n")
    header = json.dumps(
        {
            "text_model.embeddings.token_embedding.weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            },
            "vision_model.embeddings.class_embedding": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [4, 8],
            },
        }
    ).encode()
    (source / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)
    report = json.loads((source / "cuhk-report.json").read_text())
    report["candidate_weights_sha256"] = hashlib.sha256(
        (source / "model.safetensors").read_bytes()
    ).hexdigest()
    (source / "cuhk-report.json").write_text(json.dumps(report))
    files = []
    for path in sorted(source.iterdir()):
        data = path.read_bytes()
        files.append(
            {"path": path.name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
    (source / "package.json").write_text(
        json.dumps(
            {
                "model_id": "local/cuhk-clip",
                "display_name": "CUHK CLIP",
                "base_model_id": "openai/clip-vit-base-patch16",
                "dimension": 512,
                "files": files,
                "cuhk_report": "cuhk-report.json",
            }
        )
    )
    return source


def assert_empty(assets: Path) -> None:
    assert (
        not [p for p in (assets / "imported").iterdir() if p.is_dir()]
        if (assets / "imported").exists()
        else True
    )


def test_import_and_reimport(tmp_path: Path) -> None:
    source = package(tmp_path)
    assets = tmp_path / "assets"
    first = import_clip_package(source, assets)
    assert first == import_clip_package(source, assets)
    assert first.revision == first.package_sha256
    assert (assets / "imported" / first.package_sha256 / "manifest.json").is_file()
    assert first.cuhk_report == "cuhk-report.json"


def test_builtin_model_id_cannot_be_published(tmp_path: Path) -> None:
    source = package(tmp_path)
    metadata = json.loads((source / "package.json").read_text())
    metadata["model_id"] = "openai/clip-vit-base-patch16"
    (source / "package.json").write_text(json.dumps(metadata))
    assets = tmp_path / "assets"
    with pytest.raises(ClipPackageImportError, match="reserved_model_id"):
        import_clip_package(source, assets)
    assert_empty(assets)


def test_report_for_different_weights_cannot_be_imported(tmp_path: Path) -> None:
    source = package(tmp_path)
    report = json.loads((source / "cuhk-report.json").read_text())
    report["candidate_weights_sha256"] = "a" * 64
    (source / "cuhk-report.json").write_text(json.dumps(report))
    _update_hash(source, "cuhk-report.json")
    assets = tmp_path / "assets"
    with pytest.raises(ClipPackageImportError, match="cuhk_candidate_weights_mismatch"):
        import_clip_package(source, assets)
    assert_empty(assets)


def test_post_rename_fsync_failure_removes_published_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = package(tmp_path)
    assets = tmp_path / "assets"
    original = importer._fsync_directory  # noqa: SLF001

    def fail_imported(path: Path) -> None:
        if path == assets / "imported":
            raise OSError
        original(path)

    monkeypatch.setattr(importer, "_fsync_directory", fail_imported)
    with pytest.raises(ClipPackageImportError, match="package_import_failed"):
        import_clip_package(source, assets)
    assert_empty(assets)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("vision_config", "hidden_size", 1024),
        ("vision_config", "num_hidden_layers", 11),
        ("vision_config", "num_attention_heads", 11),
        ("vision_config", "image_size", 256),
        ("text_config", "hidden_size", 768),
        ("text_config", "num_hidden_layers", 11),
        ("text_config", "num_attention_heads", 7),
        ("text_config", "vocab_size", 1),
        ("text_config", "max_position_embeddings", 76),
    ],
)
def test_nonbaseline_architecture_rejected(
    tmp_path: Path, section: str, field: str, value: int
) -> None:
    source = package(tmp_path)
    config = json.loads((source / "config.json").read_text())
    config[section][field] = value
    (source / "config.json").write_text(json.dumps(config))
    _update_hash(source, "config.json")
    with pytest.raises(ClipPackageImportError, match="unsupported_clip_architecture"):
        import_clip_package(source, tmp_path / "assets")


def test_changed_bytes_same_id_rejected(tmp_path: Path) -> None:
    source = package(tmp_path)
    assets = tmp_path / "assets"
    import_clip_package(source, assets)
    changed = (source / "model.safetensors").read_bytes()[:-1] + b"1"
    (source / "model.safetensors").write_bytes(changed)
    _update_hash(source, "model.safetensors")
    with pytest.raises(ClipPackageImportError) as exc:
        import_clip_package(source, assets)
    assert exc.value.code == "model_id_conflict"
    assert len([p for p in (assets / "imported").iterdir() if p.is_dir()]) == 1


@pytest.mark.parametrize(
    "mutation",
    ["symlink", "traversal", "missing_tokenizer", "missing_config", "bin", "remote_code"],
)
def test_reject_invalid_package(tmp_path: Path, mutation: str) -> None:
    source = package(tmp_path)
    data = json.loads((source / "package.json").read_text())
    if mutation == "symlink":
        (source / "model.safetensors").unlink()
        (source / "model.safetensors").symlink_to(source / "config.json")
    elif mutation == "traversal":
        data["files"][0]["path"] = "../outside"
    elif mutation == "missing_tokenizer":
        (source / "vocab.json").unlink()
    elif mutation == "missing_config":
        (source / "config.json").unlink()
    elif mutation == "bin":
        (source / "model.safetensors").rename(source / "pytorch_model.bin")
        data["files"][-1]["path"] = "pytorch_model.bin"
    else:
        config = json.loads((source / "config.json").read_text())
        config["auto_map"] = {"AutoModel": "custom.Model"}
        (source / "config.json").write_text(json.dumps(config))
    (source / "package.json").write_text(json.dumps(data))
    assets = tmp_path / "assets"
    with pytest.raises(ClipPackageImportError):
        import_clip_package(source, assets)
    assert_empty(assets)


def test_failed_publish_leaves_no_partial_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = package(tmp_path)
    assets = tmp_path / "assets"

    def fail(*_args: object, **_kwargs: object) -> None:
        error = "failed"
        raise OSError(error)

    monkeypatch.setattr("gods_watching.model_selection.importer.os.rename", fail)
    with pytest.raises(ClipPackageImportError):
        import_clip_package(source, assets)
    assert_empty(assets)


def _update_hash(source: Path, filename: str) -> None:
    if filename == "model.safetensors":
        report = json.loads((source / "cuhk-report.json").read_text())
        report["candidate_weights_sha256"] = hashlib.sha256(
            (source / filename).read_bytes()
        ).hexdigest()
        (source / "cuhk-report.json").write_text(json.dumps(report))
        _update_hash(source, "cuhk-report.json")
    data = json.loads((source / "package.json").read_text())
    payload = (source / filename).read_bytes()
    entry = next(item for item in data["files"] if item["path"] == filename)
    entry["size"] = len(payload)
    entry["sha256"] = hashlib.sha256(payload).hexdigest()
    (source / "package.json").write_text(json.dumps(data))


def test_manifest_order_does_not_change_identity(tmp_path: Path) -> None:
    source = package(tmp_path)
    assets = tmp_path / "assets"
    first = import_clip_package(source, assets)
    data = json.loads((source / "package.json").read_text())
    data["files"].reverse()
    (source / "package.json").write_text(json.dumps(data))
    assert import_clip_package(source, assets) == first


@pytest.mark.parametrize("filename", ["package.json", "config.json", "model.safetensors"])
def test_fifo_rejected_without_open(tmp_path: Path, filename: str) -> None:
    source = package(tmp_path)
    (source / filename).unlink()
    os.mkfifo(source / filename)
    assets = tmp_path / "assets"
    with pytest.raises(ClipPackageImportError) as exc:
        import_clip_package(source, assets)
    assert exc.value.code == "invalid_package_file"
    assert_empty(assets)


def test_null_vision_config_is_typed_error(tmp_path: Path) -> None:
    source = package(tmp_path)
    config = json.loads((source / "config.json").read_text())
    config["vision_config"] = None
    (source / "config.json").write_text(json.dumps(config))
    _update_hash(source, "config.json")
    assets = tmp_path / "assets"
    with pytest.raises(ClipPackageImportError) as exc:
        import_clip_package(source, assets)
    assert exc.value.code == "unsupported_clip_architecture"
    assert_empty(assets)


def test_installed_manifest_list_is_typed_error(tmp_path: Path) -> None:
    source = package(tmp_path)
    assets = tmp_path / "assets"
    installed = assets / "imported" / "fake"
    installed.mkdir(parents=True)
    (installed / "manifest.json").write_text("[]")
    with pytest.raises(ClipPackageImportError) as exc:
        import_clip_package(source, assets)
    assert exc.value.code == "invalid_installed_package"
    assert list((assets / "imported").glob("*/manifest.json")) == [installed / "manifest.json"]


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("model.safetensors", b"\x02\x00\x00\x00\x00\x00\x00\x00{}"),
        ("preprocessor_config.json", b"{}"),
        ("vocab.json", b"[]"),
        ("merges.txt", b"garbage"),
        ("cuhk-report.json", b"{}"),
    ],
)
def test_malformed_payload_structure_rejected(
    tmp_path: Path, filename: str, content: bytes
) -> None:
    source = package(tmp_path)
    (source / filename).write_bytes(content)
    _update_hash(source, filename)
    assets = tmp_path / "assets"
    with pytest.raises(ClipPackageImportError):
        import_clip_package(source, assets)
    assert_empty(assets)


def _set_tensors(source: Path, tensors: dict[str, dict[str, object]], payload: bytes) -> None:
    header = json.dumps(tensors).encode()
    (source / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
    _update_hash(source, "model.safetensors")


def test_scalar_logit_scale_is_accepted(tmp_path: Path) -> None:
    source = package(tmp_path)
    header = {
        "text_model.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "vision_model.weight": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
        "logit_scale": {"dtype": "F32", "shape": [], "data_offsets": [8, 12]},
    }
    _set_tensors(source, header, b"\0" * 12)
    assert import_clip_package(source, tmp_path / "assets").dimension == 512


@pytest.mark.parametrize("case", ["wrong_length", "overlap"])
def test_safetensors_invalid_tensor_ranges(tmp_path: Path, case: str) -> None:
    source = package(tmp_path)
    first_end = 8 if case == "wrong_length" else 4
    second_start = 2 if case == "overlap" else 8
    header = {
        "text_model.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, first_end]},
        "vision_model.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [second_start, second_start + 4],
        },
    }
    _set_tensors(source, header, b"\0" * (6 if case == "overlap" else 12))
    assets = tmp_path / "assets"
    with pytest.raises(ClipPackageImportError) as exc:
        import_clip_package(source, assets)
    assert exc.value.code == "invalid_safetensors"
    assert_empty(assets)


def test_added_token_object_is_accepted(tmp_path: Path) -> None:
    source = package(tmp_path)
    (source / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "unk_token": {
                    "content": "<|endoftext|>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": True,
                    "special": True,
                }
            }
        )
    )
    _update_hash(source, "special_tokens_map.json")
    assert import_clip_package(source, tmp_path / "assets").dimension == 512


@pytest.mark.parametrize(
    "token", [{"content": ""}, {"content": "x", "lstrip": "false"}, {"unknown": "x"}]
)
def test_bad_added_token_object_rejected(tmp_path: Path, token: dict[str, object]) -> None:
    source = package(tmp_path)
    (source / "special_tokens_map.json").write_text(json.dumps({"unk_token": token}))
    _update_hash(source, "special_tokens_map.json")
    assets = tmp_path / "assets"
    with pytest.raises(ClipPackageImportError) as exc:
        import_clip_package(source, assets)
    assert exc.value.code == "unsupported_clip_processor"
    assert_empty(assets)
