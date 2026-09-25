"""Refusal and publication boundaries for offline full-transition rehearsals."""

# ruff: noqa: TC003, EM101, PLR0911

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import anyio
import pytest

from gods_watching.model_selection.preflight import (
    measured_rehearsal,
    rehearsal_path,
    runtime_code_sha256,
)
from gods_watching.model_selection.registry import DEFAULT_CLIP_MODEL, get_clip_model
from gods_watching.model_selection.rehearsal import (
    InnerInputs,
    RehearsalError,
    _measurement_payload,
    _publish,
    _queue_verified_job,
    _verified_inner_payload,
    rehearse_switch,
    run_inner,
)
from gods_watching.model_selection.rehearsal_stack import RehearsalInputs, RehearsalStack
from gods_watching.model_selection.transition_observer import (
    PHASES,
    TransitionMeasurement,
    TransitionObserver,
)
from gods_watching.pipeline_worker import app as worker_app


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


SOURCE = DEFAULT_CLIP_MODEL
TARGET = get_clip_model("openai/clip-vit-base-patch32")
GPU = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def measurement(*, complete: bool = True, count: int = 16) -> TransitionMeasurement:
    return TransitionMeasurement(
        (SOURCE.model_id, SOURCE.revision, SOURCE.dimension),
        (TARGET.model_id, TARGET.revision, TARGET.dimension),
        dict.fromkeys(PHASES, 1.0),
        dict.fromkeys(PHASES, (1.0, 2.0)),
        (0.1,) * count,
        count,
        complete,
    )


@pytest.mark.parametrize(("complete", "count"), [(False, 16), (True, 15)])
def test_observation_rejects_incomplete_or_insufficient(complete: bool, count: int) -> None:
    with pytest.raises(RehearsalError):
        _measurement_payload(
            measurement(complete=complete, count=count), source=SOURCE, target=TARGET, job_id="j"
        )


def test_observation_rejects_phase_failure_and_wrong_identity() -> None:
    good = measurement()
    bad = TransitionMeasurement(
        good.source_identity,
        good.target_identity,
        {"pipeline_stop": 1.0},
        good.phase_spans,
        good.crop_seconds,
        completed_crops=16,
        complete=True,
    )
    with pytest.raises(RehearsalError, match="phase_incomplete"):
        _measurement_payload(bad, source=SOURCE, target=TARGET, job_id="j")
    with pytest.raises(RehearsalError, match="target_observation_mismatch"):
        _measurement_payload(good, source=SOURCE, target=SOURCE, job_id="j")


def test_inner_payload_cannot_supply_success_flags() -> None:
    observed = _measurement_payload(measurement(), source=SOURCE, target=TARGET, job_id="j")
    observed["retained_corpus_sha256"] = "a" * 64
    observed["runtime_code_sha256"] = "b" * 64
    observed["complete"] = False
    observed["detector_resident"] = False
    checked = _verified_inner_payload(observed, SOURCE, TARGET)
    assert "complete" not in checked
    assert "detector_resident" not in checked


class FakeDocker:
    def __init__(
        self,
        *,
        inner_payload: dict[str, Any] | None,
        fail_inner: bool = False,
        gpu: str = GPU,
        after_inner: Callable[[], None] | None = None,
    ) -> None:
        self.inner_payload = inner_payload
        self.fail_inner = fail_inner
        self.gpu = gpu
        self.after_inner = after_inner
        self.commands: list[list[str]] = []

    async def run(self, argv: list[str], *, check: bool = True) -> str:
        _ = check
        self.commands.append(argv)
        if argv[1:3] == ["image", "inspect"]:
            return "sha256:app"
        if argv[1:3] == ["network", "inspect"]:
            return json.dumps(
                {
                    "Internal": True,
                    "EnableIPv6": False,
                    "Options": {"com.docker.network.bridge.gateway_mode_ipv4": "isolated"},
                    "IPAM": {"Config": [{"Subnet": "172.30.0.0/24"}]},
                }
            )
        if argv[1] == "inspect" and argv[3] == "{{json .HostConfig}}":
            return json.dumps({"NetworkMode": "isolated", "PortBindings": None})
        if argv[1] == "inspect" and argv[3] == "{{.Id}}":
            return "pgid" if argv[-1].endswith("-pg") else "tritonid"
        if argv[1] == "run":
            if self.fail_inner:
                raise RehearsalError("inner_container_nonzero")
            scratch = Path(argv[argv.index("--cidfile") + 1]).parent
            (scratch / "container-id").write_text("a" * 64)
            if self.inner_payload is not None:
                (scratch / "observation.json").write_text(json.dumps(self.inner_payload))
            if self.after_inner is not None:
                self.after_inner()
            return ""
        if "nvidia-smi" in argv:
            return self.gpu
        return ""


@pytest.mark.anyio
async def test_outer_only_publishes_after_successful_observation(tmp_path: Path) -> None:
    dump = tmp_path / "backup.dump"
    dump.write_bytes(b"offline backup")
    crops = tmp_path / "crops"
    crops.mkdir()
    (crops / "one").write_bytes(b"crop")
    assets = tmp_path / "assets"
    assets.mkdir()
    lock = tmp_path / "models.lock.json"
    lock.write_text("{}")
    (assets / "prepared-manifest.json").write_text(
        json.dumps({"cuda_device": "GPU", "cuda_device_uuid": GPU, "image_id": "sha256:triton"})
    )
    key = tmp_path / "camera.key"
    key.write_text("test key")
    inputs = RehearsalInputs(dump, crops, assets, GPU, TARGET.model_id, lock)
    stack = RehearsalStack(
        "postgresql+asyncpg://clone",
        crops,
        "triton:8001",
        GPU,
        "pgid",
        "tritonid",
        "isolated",
        "123",
        "456",
    )

    @asynccontextmanager
    async def fake_stack(
        _inputs: RehearsalInputs, *, runner: FakeDocker
    ) -> AsyncIterator[RehearsalStack]:
        _ = runner
        yield stack

    payload = _measurement_payload(measurement(), source=SOURCE, target=TARGET, job_id="job")
    payload["retained_corpus_sha256"] = "a" * 64
    payload["runtime_code_sha256"] = runtime_code_sha256()
    proof = rehearsal_path(assets, TARGET)
    proof.parent.mkdir()
    proof.write_text("stale")
    good = FakeDocker(inner_payload=payload)
    with (
        patch(
            "gods_watching.model_selection.rehearsal.validate_rehearsal_inputs",
            return_value=type(
                "Manifest", (),
                {"cuda_device": "GPU", "cuda_device_uuid": GPU, "image_id": "sha256:triton"},
            )(),
        ),
        patch("gods_watching.model_selection.rehearsal.isolated_rehearsal_stack", fake_stack),
    ):
        result = await rehearse_switch(
            inputs,
            source_model_id=SOURCE.model_id,
            source_revision=SOURCE.revision,
            source_dimension=SOURCE.dimension,
            app_image="app:local",
            camera_cipher_key_file=key,
            runner=good,
        )
        record = json.loads(result.read_text())
        assert record["sample_count"] == 16
        assert record["measured_seconds"] == 1.0
        assert record["container_id"] == "a" * 64
        assert record["database_dump_sha256"]
        assert measured_rehearsal(assets, TARGET, corpus_sha256="a" * 64) == (16.0, 5.0)
        assert "sha256:app" in good.commands[1]
        assert "app:local" not in good.commands[1]
        assert "--network" in good.commands[1]
        assert "--mount" in good.commands[1]
        assert "/var/run/docker.sock" not in str(good.commands)
        for bad in (
            FakeDocker(inner_payload=None),
            FakeDocker(inner_payload=payload, fail_inner=True),
            FakeDocker(inner_payload=payload, gpu="GPU-wrong"),
        ):
            with pytest.raises(RehearsalError):
                await rehearse_switch(
                    inputs,
                    source_model_id=SOURCE.model_id,
                    source_revision=SOURCE.revision,
                    source_dimension=SOURCE.dimension,
                    app_image="app:local",
                    camera_cipher_key_file=key,
                    runner=bad,
                )
            assert not proof.exists()
        for mutation, expected in (
            (lambda: dump.write_bytes(b"mutated offline backup"), "database_dump_changed"),
            (lambda: (crops / "one").write_bytes(b"mutated crop"), "crop_snapshot_changed"),
        ):
            with pytest.raises(RehearsalError, match=expected):
                await rehearse_switch(
                    inputs,
                    source_model_id=SOURCE.model_id,
                    source_revision=SOURCE.revision,
                    source_dimension=SOURCE.dimension,
                    app_image="app:local",
                    camera_cipher_key_file=key,
                    runner=FakeDocker(inner_payload=payload, after_inner=mutation),
                )
            assert not proof.exists()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("identity_ok", "has_job", "retained", "missing", "expected"),
    [
        (False, False, 16, 0, "source_database_mismatch"),
        (True, True, 16, 0, "existing_transition_job"),
        (True, False, 15, 0, "insufficient_decodable_crops"),
        (True, False, 16, 1, "insufficient_decodable_crops"),
    ],
)
async def test_clone_queue_refuses_invalid_state(
    identity_ok: bool, has_job: bool, retained: int, missing: int, expected: str
) -> None:
    class Repository:
        created = False

        async def active_identity(self, _session: object) -> object:
            return type(
                "Active",
                (),
                {
                    "model_id": SOURCE.model_id if identity_ok else TARGET.model_id,
                    "model_revision": SOURCE.revision,
                    "embedding_dimension": SOURCE.dimension,
                },
            )()

        async def active_job(self, _session: object) -> object | None:
            return object() if has_job else None

        async def create_job(self, _session: object, *, target: object, default: object) -> object:
            _ = target, default
            self.created = True
            return type("Job", (), {"id": "job"})()

    repository = Repository()
    with (
        patch(
            "gods_watching.model_selection.rehearsal.scan_retained",
            return_value=(retained, missing),
        ),
        pytest.raises(RehearsalError, match=expected),
    ):
        await _queue_verified_job(repository, object(), object(), source=SOURCE, target=TARGET)
    assert not repository.created


@pytest.mark.anyio
async def test_inner_refuses_non_stack_urls_before_database_access(tmp_path: Path) -> None:
    inputs = InnerInputs(
        "postgresql+asyncpg://localhost/db",
        "localhost:8001",
        tmp_path,
        tmp_path,
        tmp_path / "lock",
        TARGET.model_id,
        SOURCE.model_id,
        SOURCE.revision,
        SOURCE.dimension,
        tmp_path / "observation.json",
        tmp_path / "key",
    )
    with pytest.raises(RehearsalError, match="inner_stack_handles_required"):
        await run_inner(inputs)


@pytest.mark.anyio
async def test_no_stack_does_not_publish(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    dump = tmp_path / "dump"
    dump.write_bytes(b"dump")
    crops = tmp_path / "crops"
    crops.mkdir()
    lock = tmp_path / "lock"
    lock.write_text("lock")
    key = tmp_path / "key"
    key.write_text("key")
    inputs = RehearsalInputs(dump, crops, assets, GPU, TARGET.model_id, lock)
    proof = rehearsal_path(assets, TARGET)
    proof.parent.mkdir()
    proof.write_text("stale")

    @asynccontextmanager
    async def no_stack(
        _inputs: RehearsalInputs, *, runner: FakeDocker
    ) -> AsyncIterator[RehearsalStack]:
        _ = runner
        raise RehearsalError("stack_creation_failed")
        yield  # pragma: no cover

    with (
        patch(
            "gods_watching.model_selection.rehearsal.validate_rehearsal_inputs",
            return_value=type(
                "Manifest", (),
                {"cuda_device": "GPU", "cuda_device_uuid": GPU, "image_id": "sha256:triton"},
            )(),
        ),
        patch("gods_watching.model_selection.rehearsal.isolated_rehearsal_stack", no_stack),
        pytest.raises(RehearsalError, match="stack_creation_failed"),
    ):
        await rehearse_switch(
            inputs,
            source_model_id=SOURCE.model_id,
            source_revision=SOURCE.revision,
            source_dimension=SOURCE.dimension,
            app_image="app:local",
            camera_cipher_key_file=key,
            runner=FakeDocker(inner_payload=None),
        )
    assert not proof.exists()


def test_restart_probe_failure_prevents_complete_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Transport:
        async def __aexit__(self, *_args: object) -> None:
            events.append("transport_closed")

    class Selection:
        async def run_pending(self, **_kwargs: object) -> object:
            events.append("transition_done")
            return type("Result", (), {"activated": True})()

    class Lifecycle:
        async def stop_and_join(self) -> None:
            events.append("generation_joined")

    monkeypatch.setattr(worker_app, "TritonClipTransport", lambda *_args: Transport())
    runner = worker_app.OneShotTransition(
        Selection(), Lifecycle(), object(), object(), "isolated-triton:8001"
    )
    observer = TransitionObserver()
    observer.complete = True

    async def fail_probe() -> None:
        events.append("restart_failed")
        raise RehearsalError("pipeline_restart_failed")

    async def run() -> None:
        with pytest.raises(RehearsalError, match="pipeline_restart_failed"):
            await runner.run_pending(observer=observer, restart_probe=fail_probe)

    anyio.run(run)
    assert not observer.complete
    assert events == ["transition_done", "restart_failed", "generation_joined", "transport_closed"]


@pytest.mark.parametrize("elapsed", [0.0, float("nan"), float("inf")])
def test_observation_rejects_zero_or_nonfinite_phase(elapsed: float) -> None:
    good = measurement()
    phases = dict(good.phase_seconds)
    phases["crop_embedding"] = elapsed
    bad = TransitionMeasurement(
        good.source_identity,
        good.target_identity,
        phases,
        good.phase_spans,
        good.crop_seconds,
        good.completed_crops,
        good.complete,
    )
    with pytest.raises(RehearsalError, match="phase_invalid"):
        _measurement_payload(bad, source=SOURCE, target=TARGET, job_id="job")


def test_publish_removes_record_if_directory_fsync_fails(tmp_path: Path) -> None:
    proof = tmp_path / "switch-rehearsal" / "proof.json"
    calls = 0
    actual_fsync = os.fsync

    def failing_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            message = "directory fsync failed"
            raise OSError(message)
        actual_fsync(fd)

    with (
        patch("gods_watching.model_selection.rehearsal.os.fsync", side_effect=failing_fsync),
        pytest.raises(OSError, match="directory fsync failed"),
    ):
        _publish(proof, {"kind": "full_transition_rehearsal_v1"})
    assert not proof.exists()
    assert list(proof.parent.iterdir()) == []
