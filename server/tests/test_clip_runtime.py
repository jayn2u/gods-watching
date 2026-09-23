import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest

from gods_watching.inference.clip import (
    ClipRuntimeError,
    ClipRuntimeIdentityError,
    ClipRuntimeManager,
)
from gods_watching.model_selection.registry import ClipModelPackage, ClipModelRegistry


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@dataclass
class FakeTritonClient:
    configs: dict[str, dict[str, object]] = field(default_factory=dict)
    loaded: list[tuple[str, str]] = field(default_factory=list)
    unloaded: list[str] = field(default_factory=list)
    mismatch: bool = False
    fail_after_load: str | None = None
    cancel_after_load: str | None = None
    cancel_during_unload: str | None = None
    fail_during_unload: str | None = None
    slow_during_unload: str | None = None
    block_after_cancel: str | None = None
    unload_started: asyncio.Event = field(default_factory=asyncio.Event)
    release_after_cancel: asyncio.Event = field(default_factory=asyncio.Event)

    async def load_model(self, model_name: str, *, config: str) -> None:
        self.loaded.append((model_name, config))
        payload = cast("dict[str, object]", json.loads(config))
        if self.mismatch:
            parameters = cast("dict[str, object]", payload["parameters"])
            revision = cast("dict[str, object]", parameters["model_revision"])
            revision["string_value"] = "wrong-revision"
        self.configs[model_name] = {"config": payload}
        if model_name == self.fail_after_load:
            error_code = "load_response_lost_after_side_effect"
            raise RuntimeError(error_code)
        if model_name == self.cancel_after_load:
            raise asyncio.CancelledError

    async def unload_model(self, model_name: str) -> None:
        if model_name == self.block_after_cancel:
            self.unload_started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                _ = await self.release_after_cancel.wait()
        if model_name == self.slow_during_unload:
            await asyncio.sleep(60)
        self.unloaded.append(model_name)
        _ = self.configs.pop(model_name, None)
        if model_name == self.cancel_during_unload:
            raise asyncio.CancelledError
        if model_name == self.fail_during_unload:
            error_code = "unload_failed"
            raise RuntimeError(error_code)

    async def get_model_config(self, model_name: str, *, as_json: bool) -> Mapping[str, object]:
        assert as_json
        return self.configs[model_name]


@pytest.mark.anyio
async def test_runtime_loads_both_modalities_with_explicit_identity_config() -> None:
    client = FakeTritonClient()
    package = ClipModelRegistry().require("openai/clip-vit-large-patch14")
    runtime = ClipRuntimeManager(client)

    identity = await runtime.load_model(package)

    assert identity.model_id == package.model_id
    assert identity.revision == package.revision
    assert identity.snapshot_path == package.snapshot_path
    assert identity.dimension == 768
    assert [name for name, _config in client.loaded] == ["clip_image", "clip_text"]
    for _name, config_text in client.loaded:
        config = cast("dict[str, object]", json.loads(config_text))
        parameters = cast("dict[str, object]", config["parameters"])
        assert parameters == {
            "model_id": {"string_value": package.model_id},
            "snapshot_path": {"string_value": str(package.snapshot_path)},
            "model_revision": {"string_value": package.revision},
            "embedding_dimension": {"string_value": "768"},
            "processor": {"string_value": package.processor},
            "runtime": {"string_value": package.runtime},
        }
        output = cast("list[dict[str, object]]", config["output"])
        assert output[0]["dims"] == [768]


@pytest.mark.anyio
async def test_runtime_unloads_every_modality_and_clears_active_identity() -> None:
    client = FakeTritonClient()
    runtime = ClipRuntimeManager(client)
    package = ClipModelRegistry().default

    _ = await runtime.load_model(package)
    await runtime.unload_model()

    assert client.unloaded[-2:] == ["clip_image", "clip_text"]
    assert runtime.active_identity is None


@pytest.mark.anyio
async def test_runtime_fails_closed_when_triton_reports_the_wrong_identity() -> None:
    client = FakeTritonClient(mismatch=True)
    runtime = ClipRuntimeManager(client)

    with pytest.raises(ClipRuntimeIdentityError, match="clip_runtime_identity_mismatch"):
        _ = await runtime.load_model(ClipModelRegistry().default)

    assert client.unloaded[-2:] == ["clip_image", "clip_text"]
    assert runtime.active_identity is None


@pytest.mark.anyio
async def test_runtime_rejects_non_explicit_model_control() -> None:
    client = FakeTritonClient()

    with pytest.raises(ValueError, match="clip_runtime_explicit_mode_required"):
        _ = ClipRuntimeManager(client, model_control_mode="poll")


@pytest.mark.anyio
async def test_new_runtime_manager_clears_resident_models_before_loading() -> None:
    client = FakeTritonClient()
    first = ClipRuntimeManager(client)
    package = ClipModelRegistry().default
    _ = await first.load_model(package)

    second = ClipRuntimeManager(client)
    _ = await second.load_model(package)

    assert client.unloaded[:2] == ["clip_image", "clip_text"]


@pytest.mark.anyio
async def test_runtime_cleans_both_models_when_load_side_effect_then_rpc_fails() -> None:
    client = FakeTritonClient(fail_after_load="clip_image")
    runtime = ClipRuntimeManager(client)

    with pytest.raises(Exception, match="clip_runtime_load_failed"):
        _ = await runtime.load_model(ClipModelRegistry().default)

    assert client.unloaded[-2:] == ["clip_image", "clip_text"]
    assert client.configs == {}
    assert runtime.active_identity is None


@pytest.mark.anyio
async def test_runtime_cleans_both_models_when_second_load_is_cancelled() -> None:
    client = FakeTritonClient(cancel_after_load="clip_text")
    runtime = ClipRuntimeManager(client)

    with pytest.raises(asyncio.CancelledError):
        _ = await runtime.load_model(ClipModelRegistry().default)

    assert client.unloaded[-2:] == ["clip_image", "clip_text"]
    assert client.configs == {}
    assert runtime.active_identity is None


@pytest.mark.anyio
async def test_runtime_cleans_both_models_when_unload_is_cancelled() -> None:
    client = FakeTritonClient()
    runtime = ClipRuntimeManager(client)
    _ = await runtime.load_model(ClipModelRegistry().default)
    client.cancel_during_unload = "clip_image"

    with pytest.raises(asyncio.CancelledError):
        await runtime.unload_model()

    assert client.unloaded[-4:] == ["clip_image", "clip_text", "clip_image", "clip_text"]
    assert client.configs == {}
    assert runtime.active_identity is None


@pytest.mark.anyio
@pytest.mark.parametrize("failure_field", ["fail_during_unload", "slow_during_unload"])
async def test_runtime_does_not_load_after_preload_cleanup_failure(failure_field: str) -> None:
    client = FakeTritonClient()
    setattr(client, failure_field, "clip_image")
    runtime = ClipRuntimeManager(client, cleanup_timeout_seconds=0.01)

    with pytest.raises(ClipRuntimeError, match="clip_runtime_preload_cleanup_failed"):
        _ = await runtime.load_model(ClipModelRegistry().default)

    assert client.loaded == []
    assert runtime.active_identity is None


@pytest.mark.anyio
async def test_runtime_drains_timed_out_cleanup_before_retrying_a_load() -> None:
    client = FakeTritonClient(block_after_cancel="clip_image")
    runtime = ClipRuntimeManager(client, cleanup_timeout_seconds=0.01)
    package = ClipModelRegistry().default

    with pytest.raises(ClipRuntimeError, match="clip_runtime_preload_cleanup_failed"):
        _ = await runtime.load_model(package)
    _ = await client.unload_started.wait()
    client.block_after_cancel = None

    with pytest.raises(ClipRuntimeError, match="clip_runtime_cleanup_pending"):
        _ = await runtime.load_model(package)
    assert client.loaded == []

    client.release_after_cancel.set()
    _ = await runtime.load_model(package)
    await asyncio.sleep(0)
    assert [name for name, _config in client.loaded] == ["clip_image", "clip_text"]
    assert set(client.configs) == {"clip_image", "clip_text"}


def test_clip_package_type_is_available_for_runtime_consumers() -> None:
    package = ClipModelPackage(
        model_id="example.invalid/clip",
        revision="revision",
        snapshot_path=Path("/models/example"),
        dimension=512,
        processor="CLIPProcessor",
        runtime="transformers",
    )

    assert package.snapshot_path.as_posix() == "/models/example"
