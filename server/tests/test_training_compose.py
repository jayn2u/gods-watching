from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml

_ROOT = Path(__file__).resolve().parents[2]


def _compose() -> dict[str, object]:
    document = cast(
        "object",
        yaml.safe_load((_ROOT / "compose.yaml").read_text(encoding="utf-8")),
    )
    assert isinstance(document, dict)
    return cast("dict[str, object]", cast("object", document))


def _services() -> dict[str, dict[str, object]]:
    services = _compose().get("services")
    assert isinstance(services, dict)
    return cast("dict[str, dict[str, object]]", cast("object", services))


def _mounts(service: dict[str, object]) -> dict[str, dict[str, object]]:
    values = service.get("volumes")
    if values is None:
        return {}
    assert isinstance(values, list)
    result: dict[str, dict[str, object]] = {}
    volume_values = cast("list[object]", values)
    for value in volume_values:
        if isinstance(value, str):
            source_target = value.split(":", maxsplit=2)
            if len(source_target) < 2:
                continue
            source, target = source_target[:2]
            result[target] = {"source": source, "read_only": source_target[-1] == "ro"}
            continue
        assert isinstance(value, dict)
        mount = cast("dict[str, object]", cast("object", value))
        target = mount.get("target")
        assert isinstance(target, str)
        result[target] = mount
    return result


def test_training_mounts_keep_data_and_history_inside_their_approved_boundaries() -> None:
    services = _services()
    api = _mounts(services["api"])
    inference_worker = _mounts(services["worker"])
    trainer = _mounts(services["training"])

    assert api["/dataset"]["read_only"] is True
    assert api["/runs"]["read_only"] is True
    assert api["/models"]["read_only"] is True
    assert api["/models/imported"]["read_only"] is True

    assert "/dataset" not in inference_worker
    assert "/runs" not in inference_worker
    assert inference_worker["/models"]["read_only"] is True
    assert inference_worker["/models/imported"]["read_only"] is True

    assert trainer["/dataset"]["read_only"] is True
    assert trainer["/runs"].get("read_only", False) is False
    assert trainer["/models"]["read_only"] is True
    assert trainer["/models/imported"].get("read_only", False) is False


def test_training_service_isolated_cpu_gpu_network_and_build_context_contract() -> None:
    services = _services()
    api = services["api"]
    trainer = services["training"]
    assert isinstance(api, dict)
    assert isinstance(trainer, dict)
    assert "gpus" not in api
    assert trainer.get("gpus") == "all"
    assert trainer.get("network_mode") == "host"
    assert "ports" not in trainer
    assert trainer.get("read_only") is True
    assert trainer.get("user") is None
    assert trainer.get("cap_drop") == ["ALL"]
    assert trainer.get("cap_add") == ["DAC_OVERRIDE"]

    all_mounts = [mount for service in services.values() for mount in _mounts(service).values()]
    assert all(
        not str(mount.get("source", "")).startswith("/var/run/docker.sock") for mount in all_mounts
    )

    dockerfile = (_ROOT / "deploy" / "Dockerfile.training").read_text(encoding="utf-8")
    assert "COPY ." not in dockerfile
    assert "server/src/gods_watching" in dockerfile
    assert "training/uv.lock" in dockerfile
    ignored = (_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert "output" in ignored
    assert ".superpowers" in ignored
    assert ".env" in ignored
    assert ".env.*" in ignored
    assert "!.env.example" in ignored
