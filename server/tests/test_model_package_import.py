import hashlib
import json
import struct
from pathlib import Path

import pytest

from gods_watching.model_selection.importer import ClipPackageImportError, import_clip_package


def package(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    payloads = {
        "config.json": {
            "model_type": "clip",
            "architectures": ["CLIPModel"],
            "projection_dim": 512,
            "text_config": {"hidden_size": 512},
            "vision_config": {"hidden_size": 768, "patch_size": 16},
        },
        "preprocessor_config.json": {"do_resize": True, "size": 224},
        "tokenizer_config.json": {"tokenizer_class": "CLIPTokenizer"},
        "special_tokens_map.json": {"unk_token": "<|endoftext|>"},
        "vocab.json": {"a": 0},
        "cuhk-report.json": {"split": "test", "protocol": "identity_disjoint", "score": 0.5},
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


def test_changed_bytes_same_id_rejected(tmp_path: Path) -> None:
    source = package(tmp_path)
    assets = tmp_path / "assets"
    import_clip_package(source, assets)
    changed = (source / "model.safetensors").read_bytes()[:-1] + b"1"
    (source / "model.safetensors").write_bytes(changed)
    data = json.loads((source / "package.json").read_text())
    weight = next(f for f in data["files"] if f["path"] == "model.safetensors")
    weight["size"] = len(changed)
    weight["sha256"] = hashlib.sha256(changed).hexdigest()
    (source / "package.json").write_text(json.dumps(data))
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
