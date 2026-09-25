from __future__ import annotations

# ruff: noqa: PLC0415, TRY003, EM101, PLR0915, SLF001
# pyright: reportPrivateUsage=false, reportUnusedCallResult=false
import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast, final, override
from uuid import UUID, uuid4

import anyio
import pytest
from cryptography.fernet import Fernet
from sqlalchemy import delete, select

from gods_watching.contracts.appearances import AppearancePublication, BoundingBox
from gods_watching.contracts.cameras import CameraCreateRequest
from gods_watching.contracts.identifiers import AppearanceId, CameraId, CameraSessionId
from gods_watching.contracts.search import SimilarSearchRequest
from gods_watching.inference.clip import (
    ClipImageDecodeError,
    ClipInferenceError,
    ClipRuntimeIdentity,
)
from gods_watching.model_selection.assets import PreparedModelStatus
from gods_watching.model_selection.coordinator import TransitionCoordinator
from gods_watching.model_selection.models import (
    ModelNotPreparedError,
    TransitionPhase,
    TransitionRecoveryError,
)
from gods_watching.model_selection.quality import QualityStatus
from gods_watching.model_selection.registry import (
    DEFAULT_CLIP_MODEL,
    ClipModelPackage,
    ClipModelRegistry,
)
from gods_watching.model_selection.repository import StageResult, TransitionRepository
from gods_watching.model_selection.service import ModelSelectionService
from gods_watching.model_selection.transition_observer import PHASES, TransitionObserver
from gods_watching.search import SearchRepository
from gods_watching.storage import (
    ActiveModelIdentity,
    Appearance,
    CredentialCipher,
    CropObjectStore,
    ModelTransitionJob,
    ModelTransitionStage,
    StorageRepository,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic import AnyUrl
    from sqlalchemy.ext.asyncio import AsyncSession

    from gods_watching.appearances import AppearanceHandoffConsumer
    from gods_watching.inference.clip import TritonClipTransport
    from gods_watching.ingest import IngestCoordinator
    from gods_watching.live_detections import LiveDetectionPublisher
    from gods_watching.pipeline_worker.service import PipelineWorker
    from gods_watching.retention import RetentionService
    from gods_watching.storage import Database

_AT = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)
_SOURCE = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
_REAL_PREFLIGHT = ModelSelectionService.preflight


@pytest.fixture(autouse=True)
def _isolate_existing_transition_tests_from_offline_gpu_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transition tests exercise durable jobs; GPU proof has separate tests."""
    from gods_watching.model_selection.preflight import estimate_switch

    async def approved_preflight(
        self: ModelSelectionService, session: object, model_id: str
    ) -> object:
        del self, session
        return estimate_switch(0, 1.0, 0, target_model_id=model_id, measured_fixed_seconds=0.0)

    monkeypatch.setattr(ModelSelectionService, "preflight", approved_preflight)


@pytest.mark.anyio
async def test_missing_measurement_rejects_before_job_or_maintenance(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gods_watching.model_selection.registry import ClipModelRegistry
    from gods_watching.storage import Database

    monkeypatch.setattr(ModelSelectionService, "preflight", _REAL_PREFLIGHT)
    source = ClipModelPackage(
        model_id="fixture/source", revision="source", snapshot_path=Path("/models/source"),
        dimension=512, processor="fixture", runtime="fixture",
    )
    target = _target_package()
    database = Database.connect(database_url)
    service = ModelSelectionService(
        database, ClipModelRegistry((source, target), default_model_id=source.model_id), _Prepared()
    )
    try:
        async with database.transaction() as session:
            with pytest.raises(ModelNotPreparedError) as error:
                await service.apply(session, target.model_id)
            assert error.value.code == "model_preflight_ineligible"
        async with database.transaction() as session:
            state = await service.get(session)
            assert state.maintenance is False
            assert state.transition is None
    finally:
        await database.close()


@pytest.mark.anyio
async def test_apply_recounts_after_preview_and_rejects_growth(
    database_url: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from gods_watching.model_selection.registry import ClipModelRegistry
    from gods_watching.storage import Database

    monkeypatch.setattr(ModelSelectionService, "preflight", _REAL_PREFLIGHT)
    source = ClipModelPackage(
        model_id="fixture/source", revision="source", snapshot_path=Path("/models/source"),
        dimension=512, processor="fixture", runtime="fixture",
    )
    target = _target_package()
    counts = iter(((900, 0), (901, 0)))

    async def count_crops(session: object, store: object) -> tuple[int, int]:
        del session, store
        return next(counts)

    monkeypatch.setattr("gods_watching.model_selection.service.scan_retained", count_crops)
    monkeypatch.setattr(
        "gods_watching.model_selection.service.measured_rehearsal",
        lambda *_: (1.0, 0.0),
    )
    database = Database.connect(database_url)
    service = ModelSelectionService(
        database, ClipModelRegistry((source, target), default_model_id=source.model_id),
        _Prepared(), preflight_assets_root=tmp_path,
        preflight_crop_store=CropObjectStore(tmp_path / "crops"),
    )
    try:
        async with database.transaction() as session:
            preview = await service.preflight(session, target.model_id)
            assert preview.eligible
            with pytest.raises(ModelNotPreparedError) as error:
                await service.apply(session, target.model_id)
            assert error.value.code == "model_preflight_ineligible"
        async with database.transaction() as session:
            state = await service.get(session)
            assert state.transition is None
            assert state.maintenance is False
    finally:
        await database.close()


@pytest.mark.anyio
async def test_imported_model_apply_fails_without_real_product_cases() -> None:
    imported = ClipModelPackage(
        model_id="fixture/imported",
        revision="a" * 64,
        snapshot_path=Path("/models/imported") / ("a" * 64),
        dimension=512,
        processor="CLIPProcessor",
        runtime="transformers",
    )
    service = ModelSelectionService(
        database=cast("Database", object()),
        registry=ClipModelRegistry((DEFAULT_CLIP_MODEL, imported)),
        prepared=_Prepared(),
    )
    with pytest.raises(ModelNotPreparedError, match="quality evidence store unavailable") as error:
        await service.apply(cast("AsyncSession", object()), imported.model_id)
    assert error.value.code == "model_quality_ineligible"


@final
class _Prepared:
    def status(self, package: ClipModelPackage) -> PreparedModelStatus:
        del package
        return PreparedModelStatus(prepared=True)


@final
class _Runtime:
    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.loaded_packages: list[ClipModelPackage] = []

    async def unload_model(self) -> None:
        self.loaded.clear()
        self.loaded_packages.clear()

    async def load_model(self, package: ClipModelPackage) -> ClipRuntimeIdentity:
        self.loaded.append(package.model_id)
        self.loaded_packages.append(package)
        return ClipRuntimeIdentity.from_package(package)

    async def inspect_identity(self) -> ClipRuntimeIdentity:
        raise AssertionError("test runtime does not inspect without a loaded package")


@final
class _Pipeline:
    def __init__(self, *, fail_target: bool = False) -> None:
        self.starts: list[str] = []
        self.fail_target = fail_target

    async def stop_and_join(self) -> None:
        return

    async def start(self, package: ClipModelPackage) -> None:
        self.starts.append(package.model_id)
        if self.fail_target and package.model_id == "fixture/clip-large":
            raise RuntimeError("target pipeline failed")


@final
class _ImageClip:
    def __init__(self, *, dimension: int, failure: BaseException | None = None) -> None:
        self.dimension = dimension
        self.failure = failure

    async def embed_image(self, payload: bytes) -> tuple[float, ...]:
        del payload
        if self.failure is not None:
            raise self.failure
        return _unit(self.dimension)


def _failing_clip_factory(package: ClipModelPackage) -> _ImageClip:
    del package
    return _ImageClip(dimension=768, failure=ClipInferenceError(code="clip_gpu_oom"))


def _working_clip_factory(package: ClipModelPackage) -> _ImageClip:
    del package
    return _ImageClip(dimension=768)


def _unit(dimension: int) -> tuple[float, ...]:
    return tuple(1.0 if index == 0 else 0.0 for index in range(dimension))


def _target_package() -> ClipModelPackage:
    return ClipModelPackage(
        model_id="fixture/clip-large",
        revision="fixture-large-revision",
        snapshot_path=Path("/models/fixture-large"),
        dimension=768,
        processor="fixture",
        runtime="fixture",
    )


def _source(name: str) -> AnyUrl:
    return CameraCreateRequest.model_validate(
        {"name": name, "source_url": f"rtsp://fixture:8554/{name}"}
    ).source_url


# ruff: noqa: PLR0913
def _publication(
    *,
    appearance_id: UUID,
    camera_id: UUID,
    session_id: UUID,
    embedding: tuple[float, ...],
    model_id: str = "openai/clip-vit-base-patch16",
    revision: str = _SOURCE,
    track_id: int = 1,
    crop_object_key: str | None = None,
    byte_size: int = 4,
) -> AppearancePublication:
    return AppearancePublication(
        appearance_id=AppearanceId(appearance_id),
        camera_id=CameraId(camera_id),
        session_id=CameraSessionId(session_id),
        track_id=track_id,
        first_seen=_AT,
        last_seen=_AT,
        ended_at=_AT,
        representative_version=1,
        crop_object_key=crop_object_key or f"aa/bb/{appearance_id}.jpg",
        bounding_box=BoundingBox(x_min=1, y_min=2, x_max=40, y_max=80),
        source_width=1920,
        source_height=1080,
        detector_confidence=0.9,
        crop_quality=10.0,
        byte_size=byte_size,
        embedded_at=_AT,
        model_id=model_id,
        model_revision=revision,
        embedding=embedding,
    )


@pytest.mark.anyio
async def test_transition_stages_and_activates_768_without_materializing_rows(
    session: AsyncSession,
) -> None:
    storage = StorageRepository(CredentialCipher(Fernet.generate_key()))
    camera = await storage.add_camera(
        session,
        name=f"selector-{uuid4().hex[:8]}",
        source_url=_source("selector"),
    )
    camera_session = await storage.start_camera_session(session, camera.id, cause="fixture")
    appearance = await storage.publish_appearance(
        session,
        _publication(
            appearance_id=uuid4(),
            camera_id=camera.id,
            session_id=camera_session.id,
            embedding=_unit(512),
        ),
    )
    target = ClipModelPackage(
        model_id="fixture/clip-large",
        revision="fixture-large-revision",
        snapshot_path=Path("/models/fixture-large"),
        dimension=768,
        processor="fixture",
        runtime="fixture",
    )
    repository = TransitionRepository()
    job = await repository.create_job(
        session,
        target=target,
        default=ClipModelPackage(
            model_id="openai/clip-vit-base-patch16",
            revision=_SOURCE,
            snapshot_path=Path("/models/clip"),
            dimension=512,
            processor="CLIPProcessor",
            runtime="transformers",
        ),
    )
    assert await repository.populate_stages(session, job) == 1
    await repository.record_stage_results(
        session,
        job,
        (StageResult(appearance.id, _unit(768), None),),
    )
    assert job.phase == TransitionPhase.ACTIVATING.value
    _ = await repository.activate(
        session,
        job.id,
        expected_source=("openai/clip-vit-base-patch16", _SOURCE, 512),
    )
    assert (
        await session.scalar(
            select(ActiveModelIdentity.model_id).where(ActiveModelIdentity.singleton.is_(True))
        )
        == target.model_id
    )
    refreshed = await session.scalar(select(Appearance).where(Appearance.id == appearance.id))
    assert refreshed is not None
    assert refreshed.embedding_dimension == 768
    assert refreshed.model_id == target.model_id
    assert refreshed.embedding is not None
    assert (
        await session.scalar(
            select(ModelTransitionStage.job_id).where(ModelTransitionStage.job_id == job.id)
        )
        is None
    )


@pytest.mark.anyio
async def test_768_search_query_keeps_dimension_specific_identity_filter() -> None:
    repository = SearchRepository()
    statement = repository._similar_statement(
        SimilarSearchRequest(mode="similar", appearance_id=AppearanceId(uuid4())),
        embedding=_unit(768),
        model_revision="fixture-large-revision",
        model_id="fixture/clip-large",
        dimension=768,
        exclude_appearance_id=None,
    )
    sql = str(statement.compile(compile_kwargs={"literal_binds": False}))
    assert "vector(768)" in sql
    assert "embedding_dimension" in sql
    assert "model_id" in sql


def test_equal_512_dimensional_revisions_keep_separate_search_predicates() -> None:
    repository = SearchRepository()
    request = SimilarSearchRequest(mode="similar", appearance_id=AppearanceId(uuid4()))
    first = repository._similar_statement(
        request,
        embedding=_unit(512),
        model_revision="revision-a",
        model_id="local/clip",
        dimension=512,
        exclude_appearance_id=None,
    )
    second = repository._similar_statement(
        request,
        embedding=_unit(512),
        model_revision="revision-b",
        model_id="local/clip",
        dimension=512,
        exclude_appearance_id=None,
    )
    first_sql = first.compile()
    second_sql = second.compile()
    assert "revision-a" in first_sql.params.values()
    assert "revision-b" in second_sql.params.values()
    assert "revision-b" not in first_sql.params.values()


@pytest.mark.anyio
async def test_inference_failure_restores_source_identity_and_pipeline(
    database_url: str,
    tmp_path: Path,
) -> None:
    from gods_watching.storage import Camera, CameraSession, Database

    database = Database.connect(database_url)
    crop_store = CropObjectStore(tmp_path / "failure-crops")
    target = _target_package()
    default = ClipModelPackage(
        model_id="openai/clip-vit-base-patch16",
        revision=_SOURCE,
        snapshot_path=Path("/models/clip"),
        dimension=512,
        processor="CLIPProcessor",
        runtime="transformers",
    )
    from gods_watching.model_selection.registry import ClipModelRegistry

    registry = ClipModelRegistry((default, target))
    appearance_id = uuid4()
    camera_id: UUID | None = None
    session_id: UUID | None = None
    try:
        async with database.transaction() as setup:
            storage = StorageRepository(CredentialCipher(Fernet.generate_key()))
            camera = await storage.add_camera(
                setup,
                name=f"selector-failure-{uuid4().hex[:8]}",
                source_url=_source("selector-failure"),
            )
            camera_session = await storage.start_camera_session(setup, camera.id, cause="test")
            crop = crop_store.write(b"jpeg")
            _ = await storage.publish_appearance(
                setup,
                _publication(
                    appearance_id=appearance_id,
                    camera_id=camera.id,
                    session_id=camera_session.id,
                    embedding=_unit(512),
                    crop_object_key=crop.object_key,
                ),
            )
            camera_id, session_id = camera.id, camera_session.id
        service = ModelSelectionService(
            database=database,
            registry=registry,
            prepared=_Prepared(),
            storage=storage,
        )
        async with database.transaction() as apply_session:
            _ = await service.apply(apply_session, target.model_id)
        runtime = _Runtime()
        pipeline = _Pipeline()
        observer = TransitionObserver()
        result = await service.run_pending(
            crop_store=crop_store,
            runtime=runtime,
            clip_factory=_failing_clip_factory,
            pipeline=pipeline,
            observer=observer,
        )
        assert result is not None
        assert result.state.phase == TransitionPhase.FAILED
        assert runtime.loaded == [default.model_id]
        assert pipeline.starts == [default.model_id]
        assert observer.measurement is not None
        assert not observer.measurement.complete
        assert observer.measurement.completed_crops == 0
        async with database.transaction() as check:
            active = await check.scalar(
                select(ActiveModelIdentity).where(ActiveModelIdentity.singleton.is_(True))
            )
            assert active is not None
            assert (active.model_id, active.model_revision, active.embedding_dimension) == (
                default.model_id, default.revision, default.dimension,
            )
        restarted_service = ModelSelectionService(
            database=database, registry=ClipModelRegistry((default, target)),
            prepared=_Prepared(), storage=storage,
        )
        recovered_runtime = _Runtime()
        recovered_pipeline = _Pipeline()
        recovered = await restarted_service.recover_startup(
            runtime=recovered_runtime, pipeline=recovered_pipeline,
        )
        assert recovered_runtime.loaded == [default.model_id]
        assert recovered_pipeline.starts == [default.model_id]
        assert recovered is None or not recovered.activated
    finally:
        async with database.transaction() as cleanup:
            await cleanup.execute(delete(ModelTransitionJob))
            await cleanup.execute(delete(Appearance).where(Appearance.id == appearance_id))
            if session_id is not None:
                await cleanup.execute(delete(CameraSession).where(CameraSession.id == session_id))
            if camera_id is not None:
                await cleanup.execute(delete(Camera).where(Camera.id == camera_id))
        await database.close()


@pytest.mark.anyio
async def test_pipeline_failure_after_activation_keeps_target_identity_in_maintenance(
    database_url: str,
    tmp_path: Path,
) -> None:
    from gods_watching.storage import Camera, CameraSession, CropObjectStore, Database

    database = Database.connect(database_url)
    crop_store = CropObjectStore(tmp_path / "after-commit-crops")
    target = _target_package()
    default = ClipModelPackage(
        model_id="openai/clip-vit-base-patch16",
        revision=_SOURCE,
        snapshot_path=Path("/models/clip"),
        dimension=512,
        processor="CLIPProcessor",
        runtime="transformers",
    )
    from gods_watching.model_selection.registry import ClipModelRegistry

    registry = ClipModelRegistry((default, target))
    appearance_id = uuid4()
    camera_id: UUID | None = None
    session_id: UUID | None = None
    try:
        async with database.transaction() as setup:
            storage = StorageRepository(CredentialCipher(Fernet.generate_key()))
            camera = await storage.add_camera(
                setup,
                name=f"selector-after-{uuid4().hex[:8]}",
                source_url=_source("selector-after"),
            )
            camera_session = await storage.start_camera_session(setup, camera.id, cause="test")
            crop = crop_store.write(b"jpeg")
            _ = await storage.publish_appearance(
                setup,
                _publication(
                    appearance_id=appearance_id,
                    camera_id=camera.id,
                    session_id=camera_session.id,
                    embedding=_unit(512),
                    crop_object_key=crop.object_key,
                ),
            )
            camera_id, session_id = camera.id, camera_session.id
        service = ModelSelectionService(
            database=database,
            registry=registry,
            prepared=_Prepared(),
            storage=storage,
        )
        async with database.transaction() as apply_session:
            _ = await service.apply(apply_session, target.model_id)
        runtime = _Runtime()
        pipeline = _Pipeline(fail_target=True)
        observer = TransitionObserver()
        result = await service.run_pending(
            crop_store=crop_store,
            runtime=runtime,
            clip_factory=_working_clip_factory,
            pipeline=pipeline,
            observer=observer,
        )
        assert result is not None
        assert result.activated
        assert result.state.phase == TransitionPhase.ROLLING_BACK
        assert observer.measurement is not None
        assert not observer.measurement.complete
        assert "pipeline_restart" not in observer.measurement.phase_spans
        restarted_service = ModelSelectionService(
            database=database, registry=ClipModelRegistry((default, target)),
            prepared=_Prepared(), storage=storage,
        )
        recovered_runtime = _Runtime()
        recovered_pipeline = _Pipeline()
        recovered = await restarted_service.recover_startup(
            runtime=recovered_runtime, pipeline=recovered_pipeline,
        )
        assert recovered is not None
        assert recovered.activated
        assert recovered.state.phase == TransitionPhase.SUCCEEDED
        assert recovered_runtime.loaded == [target.model_id]
        assert recovered_pipeline.starts == [target.model_id]
        async with database.transaction() as readback:
            status = await restarted_service.get(readback)
            assert status.active_model_id == target.model_id
            assert not status.maintenance
            assert (
                next(m for m in status.models if m.model_id == target.model_id).revision
                == target.revision
            )
        async with database.transaction() as check:
            active = await check.scalar(
                select(ActiveModelIdentity).where(ActiveModelIdentity.singleton.is_(True))
            )
            assert active is not None
            assert (active.model_id, active.model_revision, active.embedding_dimension) == (
                target.model_id,
                target.revision,
                target.dimension,
            )
    finally:
        async with database.transaction() as cleanup:
            await cleanup.execute(delete(ModelTransitionJob))
            await cleanup.execute(delete(Appearance).where(Appearance.id == appearance_id))
            if session_id is not None:
                await cleanup.execute(delete(CameraSession).where(CameraSession.id == session_id))
            if camera_id is not None:
                await cleanup.execute(delete(Camera).where(Camera.id == camera_id))
            active = await cleanup.scalar(
                select(ActiveModelIdentity).where(ActiveModelIdentity.singleton.is_(True))
            )
            if active is not None:
                active.model_id = default.model_id
                active.model_revision = default.revision
                active.embedding_dimension = default.dimension
        await database.close()


@pytest.mark.anyio
async def test_recovery_keeps_staged_source_pipeline_stopped_until_resume(
    database_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from gods_watching.model_selection.registry import ClipModelRegistry
    from gods_watching.storage import Camera, CameraSession, Database

    database = Database.connect(database_url)
    crop_store = CropObjectStore(tmp_path / "restart-crops")

    def _no_headroom(_path: Path) -> SimpleNamespace:
        return SimpleNamespace(free=0)

    monkeypatch.setattr("gods_watching.model_selection.service.shutil.disk_usage", _no_headroom)
    target = ClipModelPackage(
        model_id="fixture/imported-restart",
        revision="a" * 64,
        snapshot_path=tmp_path / "imported" / ("a" * 64),
        dimension=768,
        processor="CLIPProcessor",
        runtime="transformers",
    )
    # This recovery test supplies a synthetic approved package descriptor;
    # quality evidence validation has separate fail-closed tests.
    monkeypatch.setattr(
        ModelSelectionService, "_quality_status",
        lambda _self, _package: QualityStatus(passed=True, reason=None),
    )
    default = ClipModelPackage(
        model_id="openai/clip-vit-base-patch16",
        revision=_SOURCE,
        snapshot_path=Path("/models/clip"),
        dimension=512,
        processor="CLIPProcessor",
        runtime="transformers",
    )
    registry = ClipModelRegistry((default, target))
    appearance_id = uuid4()
    camera_id: UUID | None = None
    session_id: UUID | None = None
    try:
        async with database.transaction() as setup:
            storage = StorageRepository(CredentialCipher(Fernet.generate_key()))
            camera = await storage.add_camera(
                setup,
                name=f"selector-restart-{uuid4().hex[:8]}",
                source_url=_source("selector-restart"),
            )
            camera_session = await storage.start_camera_session(setup, camera.id, cause="test")
            crop = crop_store.write(b"jpeg")
            _ = await storage.publish_appearance(
                setup,
                _publication(
                    appearance_id=appearance_id,
                    camera_id=camera.id,
                    session_id=camera_session.id,
                    embedding=_unit(512),
                    crop_object_key=crop.object_key,
                ),
            )
            camera_id, session_id = camera.id, camera_session.id
        service = ModelSelectionService(
            database=database,
            registry=registry,
            prepared=_Prepared(),
            storage=storage,
        )
        async with database.transaction() as apply_session:
            _ = await service.apply(apply_session, target.model_id)
        repository = TransitionRepository()
        async with database.transaction() as interrupted:
            job = await repository.active_job(interrupted, lock=True)
            assert job is not None
            assert await repository.populate_stages(interrupted, job) == 1
            job.phase = TransitionPhase.REINDEXING.value
            await interrupted.flush()
        # A new worker process reconstructs the service from the same durable registry.
        restarted_service = ModelSelectionService(
            database=database, registry=ClipModelRegistry((default, target)),
            prepared=_Prepared(), storage=storage,
        )
        async with database.transaction() as readback:
            pending = await restarted_service.get(readback)
            assert pending.maintenance
            assert pending.transition is not None
            assert pending.transition.phase == TransitionPhase.REINDEXING
            assert pending.transition.target_model_id == target.model_id
            assert (
                next(m for m in pending.models if m.model_id == target.model_id).revision
                == target.revision
            )
        wrong_revision = ClipModelPackage(
            model_id=target.model_id,
            revision="b" * 64,
            snapshot_path=tmp_path / "imported" / ("b" * 64),
            dimension=target.dimension,
            processor=target.processor,
            runtime=target.runtime,
        )
        mismatched_service = ModelSelectionService(
            database=database, registry=ClipModelRegistry((default, wrong_revision)),
            prepared=_Prepared(), storage=storage,
        )
        with pytest.raises(TransitionRecoveryError):
            await mismatched_service.run_pending(
                crop_store=crop_store, runtime=_Runtime(),
                clip_factory=_working_clip_factory, pipeline=_Pipeline(),
            )
        runtime = _Runtime()
        pipeline = _Pipeline()
        recovery = await restarted_service.recover_startup(runtime=runtime, pipeline=pipeline)
        assert recovery is not None
        assert recovery.activated is False
        assert pipeline.starts == []
        observer = TransitionObserver()
        result = await restarted_service.run_pending(
            crop_store=crop_store,
            runtime=runtime,
            clip_factory=_working_clip_factory,
            pipeline=pipeline,
            observer=observer,
        )
        assert result is not None
        assert result.state.phase == TransitionPhase.SUCCEEDED
        assert pipeline.starts == [target.model_id]
        assert runtime.loaded_packages == [target]
        async with database.transaction() as readback:
            active = await readback.scalar(
                select(ActiveModelIdentity).where(ActiveModelIdentity.singleton.is_(True))
            )
            assert active is not None
            assert (active.model_id, active.model_revision, active.embedding_dimension) == (
                target.model_id, target.revision, target.dimension,
            )
        assert observer.measurement is not None
        assert not observer.measurement.complete
        assert set(observer.measurement.phase_seconds) == set(PHASES) - {"stage_population"}
        assert observer.measurement.completed_crops == 1
        assert observer.measurement.source_identity == (
            default.model_id, default.revision, default.dimension
        )
        assert observer.measurement.target_identity == (
            target.model_id, target.revision, target.dimension
        )
    finally:
        async with database.transaction() as cleanup:
            await cleanup.execute(delete(ModelTransitionJob))
            await cleanup.execute(delete(Appearance).where(Appearance.id == appearance_id))
            if session_id is not None:
                await cleanup.execute(delete(CameraSession).where(CameraSession.id == session_id))
            if camera_id is not None:
                await cleanup.execute(delete(Camera).where(Camera.id == camera_id))
            active = await cleanup.scalar(
                select(ActiveModelIdentity).where(ActiveModelIdentity.singleton.is_(True))
            )
            if active is not None:
                active.model_id = default.model_id
                active.model_revision = default.revision
                active.embedding_dimension = default.dimension
        await database.close()


@pytest.mark.anyio
async def test_pipeline_lifecycle_stop_join_waits_for_terminal_cleanup() -> None:
    from gods_watching.pipeline_worker.app import _Generation, _PipelineLifecycle

    class _DoneTransport:
        async def __aexit__(self, *_args: object) -> None:
            return

    generation = _Generation(
        package=_target_package(),
        stop_event=anyio.Event(),
        done_event=anyio.Event(),
        coordinator=cast("IngestCoordinator", object()),
        consumer=cast("AppearanceHandoffConsumer", object()),
        retention=cast("RetentionService", object()),
        pipeline=cast("PipelineWorker", object()),
        live_detection=cast("LiveDetectionPublisher", object()),
        clip_transport=cast("TritonClipTransport", cast("object", _DoneTransport())),
    )
    lifecycle = object.__new__(_PipelineLifecycle)
    lifecycle._generation = generation  # type: ignore[attr-defined]

    async def finish_terminal_handoffs() -> None:
        await generation.stop_event.wait()
        await anyio.sleep(0.05)
        generation.done_event.set()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(finish_terminal_handoffs)
        await lifecycle.stop_and_join()
    assert generation.stop_event.is_set()
    assert lifecycle.generation is None

    cancelled_generation = _Generation(
        package=_target_package(),
        stop_event=anyio.Event(),
        done_event=anyio.Event(),
        coordinator=cast("IngestCoordinator", object()),
        consumer=cast("AppearanceHandoffConsumer", object()),
        retention=cast("RetentionService", object()),
        pipeline=cast("PipelineWorker", object()),
        live_detection=cast("LiveDetectionPublisher", object()),
        clip_transport=cast("TritonClipTransport", cast("object", _DoneTransport())),
    )
    lifecycle._generation = cancelled_generation  # type: ignore[attr-defined]

    async def finish_cancelled_handoffs() -> None:
        await cancelled_generation.stop_event.wait()
        await anyio.sleep(0.05)
        cancelled_generation.done_event.set()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(finish_cancelled_handoffs)
        caller = asyncio.create_task(lifecycle.stop_and_join())
        await anyio.sleep(0.01)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
    assert cancelled_generation.done_event.is_set()


@pytest.mark.anyio
async def test_activation_commit_acknowledgement_loss_recovers_durable_target(
    database_url: str,
    tmp_path: Path,
) -> None:
    from contextlib import asynccontextmanager

    from gods_watching.model_selection.registry import ClipModelRegistry
    from gods_watching.storage import Camera, CameraSession, Database

    real_database = Database.connect(database_url)

    @final
    class _CommitAcknowledgementDatabase:
        raise_after_commit: bool

        def __init__(self) -> None:
            self.raise_after_commit = False

        @asynccontextmanager
        async def transaction(self) -> AsyncIterator[AsyncSession]:
            async with real_database.transaction() as transaction:
                yield transaction
            if self.raise_after_commit:
                self.raise_after_commit = False
                raise RuntimeError("activation commit acknowledgement lost")

    wrapped_database = _CommitAcknowledgementDatabase()
    target = _target_package()
    default = ClipModelPackage(
        model_id="openai/clip-vit-base-patch16",
        revision=_SOURCE,
        snapshot_path=Path("/models/clip"),
        dimension=512,
        processor="CLIPProcessor",
        runtime="transformers",
    )
    registry = ClipModelRegistry((default, target))

    class _AcknowledgementRepository(TransitionRepository):
        @override
        async def activate(
            self,
            session: AsyncSession,
            job_id: UUID,
            *,
            expected_source: tuple[str, str, int],
        ) -> ModelTransitionJob:
            result = await super().activate(
                session,
                job_id,
                expected_source=expected_source,
            )
            wrapped_database.raise_after_commit = True
            return result

    crop_store = CropObjectStore(tmp_path / "ack-loss-crops")
    storage = StorageRepository(CredentialCipher(Fernet.generate_key()))
    appearance_id = uuid4()
    camera_id: UUID | None = None
    session_id: UUID | None = None
    try:
        async with real_database.transaction() as setup:
            camera = await storage.add_camera(
                setup,
                name=f"selector-ack-{uuid4().hex[:8]}",
                source_url=_source("selector-ack"),
            )
            camera_session = await storage.start_camera_session(setup, camera.id, cause="test")
            crop = crop_store.write(b"jpeg")
            _ = await storage.publish_appearance(
                setup,
                _publication(
                    appearance_id=appearance_id,
                    camera_id=camera.id,
                    session_id=camera_session.id,
                    embedding=_unit(512),
                    crop_object_key=crop.object_key,
                ),
            )
            camera_id, session_id = camera.id, camera_session.id
        service = ModelSelectionService(
            database=cast("Database", cast("object", wrapped_database)),
            registry=registry,
            prepared=_Prepared(),
            repository=_AcknowledgementRepository(),
            coordinator=TransitionCoordinator(real_database),
            storage=storage,
        )
        async with real_database.transaction() as apply_session:
            _ = await service.apply(apply_session, target.model_id)
        runtime = _Runtime()
        pipeline = _Pipeline()
        result = await service.run_pending(
            crop_store=crop_store,
            runtime=runtime,
            clip_factory=_working_clip_factory,
            pipeline=pipeline,
        )
        assert result is not None
        assert result.activated
        assert result.state.phase == TransitionPhase.SUCCEEDED
        assert runtime.loaded == [target.model_id]
        assert pipeline.starts == [target.model_id]
    finally:
        async with real_database.transaction() as cleanup:
            await cleanup.execute(delete(ModelTransitionJob))
            await cleanup.execute(delete(Appearance).where(Appearance.id == appearance_id))
            if session_id is not None:
                await cleanup.execute(delete(CameraSession).where(CameraSession.id == session_id))
            if camera_id is not None:
                await cleanup.execute(delete(Camera).where(Camera.id == camera_id))
            active = await cleanup.scalar(
                select(ActiveModelIdentity).where(ActiveModelIdentity.singleton.is_(True))
            )
            if active is not None:
                active.model_id = default.model_id
                active.model_revision = default.revision
                active.embedding_dimension = default.dimension
        await real_database.close()


@pytest.mark.anyio
async def test_only_missing_empty_and_decode_failures_are_skips(
    database_url: str,
    tmp_path: Path,
) -> None:
    from gods_watching.model_selection.registry import ClipModelRegistry
    from gods_watching.storage import Database

    database = Database.connect(database_url)
    crop_store = CropObjectStore(tmp_path / "skip-crops")
    service = ModelSelectionService(
        database=database,
        registry=ClipModelRegistry(),
        prepared=_Prepared(),
    )
    missing = await service._embed_stage(
        uuid4(),
        "aa/bb/00000000-0000-4000-8000-000000000000.jpg",
        crop_store,
        _ImageClip(dimension=512),
    )
    empty = crop_store.write(b"")
    empty_result = await service._embed_stage(
        uuid4(), empty.object_key, crop_store, _ImageClip(dimension=512)
    )
    corrupt = crop_store.write(b"not-an-image")
    corrupt_result = await service._embed_stage(
        uuid4(),
        corrupt.object_key,
        crop_store,
        _ImageClip(dimension=512, failure=ClipImageDecodeError(code="clip_image_decode_failed")),
    )
    assert missing.skip_reason == "missing_crop"
    assert empty_result.skip_reason == "undecodable_crop"
    assert corrupt_result.skip_reason == "undecodable_crop"
    await database.close()


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["success", "stage", "activation", "cancel"])
async def test_observer_incomplete_on_production_phase_failure(
    database_url: str, tmp_path: Path, failure: str
) -> None:
    from gods_watching.storage import Database

    class FailingRepository(TransitionRepository):
        async def populate_stages(self, session: AsyncSession, job: ModelTransitionJob) -> int:
            if failure in {"stage", "cancel"}:
                if failure == "cancel":
                    raise asyncio.CancelledError
                raise RuntimeError("stage population failed")
            return await super().populate_stages(session, job)

        async def activate(
            self, session: AsyncSession, job_id: UUID, *, expected_source: tuple[str, str, int]
        ) -> ModelTransitionJob:
            if failure == "activation":
                raise RuntimeError("activation failed")
            return await super().activate(session, job_id, expected_source=expected_source)

    source = ClipModelPackage(
        model_id="openai/clip-vit-base-patch16", revision=_SOURCE,
        snapshot_path=Path("/models/clip"), dimension=512,
        processor="CLIPProcessor", runtime="transformers",
    )
    target = _target_package()
    database = Database.connect(database_url)
    service = ModelSelectionService(
        database=database,
        registry=ClipModelRegistry((source, target)),
        prepared=_Prepared(), repository=FailingRepository(),
        storage=StorageRepository(CredentialCipher(Fernet.generate_key())),
    )
    observer = TransitionObserver()
    try:
        async with database.transaction() as session:
            _ = await service.apply(session, target.model_id)
        if failure == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await service.run_pending(
                    crop_store=CropObjectStore(tmp_path / failure), runtime=_Runtime(),
                    clip_factory=_working_clip_factory, pipeline=_Pipeline(), observer=observer,
                )
        else:
            result = await service.run_pending(
                crop_store=CropObjectStore(tmp_path / failure), runtime=_Runtime(),
                clip_factory=_working_clip_factory, pipeline=_Pipeline(), observer=observer,
            )
            assert result is not None
            assert result.state.phase == (
                TransitionPhase.SUCCEEDED if failure == "success" else TransitionPhase.FAILED
            )
        assert observer.measurement is not None
        assert observer.measurement.complete is (failure == "success")
        assert observer.measurement.completed_crops == 0
        assert ("pipeline_restart" in observer.measurement.phase_spans) is (
            failure == "success"
        )
        if failure == "success":
            idle = await service.run_pending(
                crop_store=CropObjectStore(tmp_path / "idle"), runtime=_Runtime(),
                clip_factory=_working_clip_factory, pipeline=_Pipeline(), observer=observer,
            )
            assert idle is None
            assert observer.measurement is None
    finally:
        async with database.transaction() as session:
            await session.execute(delete(ModelTransitionJob))
            active = await session.scalar(
                select(ActiveModelIdentity).where(ActiveModelIdentity.singleton.is_(True))
            )
            if active is not None:
                active.model_id = source.model_id
                active.model_revision = source.revision
                active.embedding_dimension = source.dimension
        await database.close()


@pytest.mark.anyio
async def test_observer_counts_only_committed_crop_before_later_failure(
    database_url: str, tmp_path: Path
) -> None:
    from gods_watching.storage import Camera, CameraSession, Database

    database = Database.connect(database_url)
    store = CropObjectStore(tmp_path / "partial-crops")
    storage = StorageRepository(CredentialCipher(Fernet.generate_key()))
    source = ClipModelPackage(
        model_id="openai/clip-vit-base-patch16", revision=_SOURCE,
        snapshot_path=Path("/models/clip"), dimension=512,
        processor="CLIPProcessor", runtime="transformers",
    )
    target = _target_package()
    service = ModelSelectionService(
        database=database, registry=ClipModelRegistry((source, target)),
        prepared=_Prepared(), storage=storage,
    )
    appearances = [uuid4(), uuid4()]
    camera_id: UUID | None = None
    session_id: UUID | None = None
    try:
        async with database.transaction() as session:
            camera = await storage.add_camera(
                session, name=f"observer-partial-{uuid4().hex[:8]}",
                source_url=_source("observer-partial"),
            )
            camera_session = await storage.start_camera_session(session, camera.id, cause="test")
            camera_id, session_id = camera.id, camera_session.id
            for track_id, appearance_id in enumerate(appearances, start=1):
                crop = store.write(b"jpeg")
                _ = await storage.publish_appearance(
                    session,
                    _publication(
                        appearance_id=appearance_id, camera_id=camera.id,
                        session_id=camera_session.id, embedding=_unit(512),
                        crop_object_key=crop.object_key, track_id=track_id,
                    ),
                )
        async with database.transaction() as session:
            _ = await service.apply(session, target.model_id)

        class SecondCropFails:
            calls = 0

            async def embed_image(self, payload: bytes) -> tuple[float, ...]:
                del payload
                self.calls += 1
                if self.calls == 2:
                    raise ClipInferenceError(code="clip_gpu_oom")
                return _unit(768)

        clip = SecondCropFails()
        observer = TransitionObserver()
        result = await service.run_pending(
            crop_store=store, runtime=_Runtime(), clip_factory=lambda _package: clip,
            pipeline=_Pipeline(), batch_size=1, observer=observer,
        )
        assert result is not None
        assert result.state.phase == TransitionPhase.FAILED
        assert observer.measurement is not None
        assert not observer.measurement.complete
        assert observer.measurement.completed_crops == 1
        assert len(observer.measurement.crop_seconds) == 1
        assert observer.measurement.crop_seconds[0] <= observer.measurement.phase_seconds.get(
            "crop_embedding", float("inf")
        )
    finally:
        async with database.transaction() as session:
            await session.execute(delete(ModelTransitionJob))
            await session.execute(delete(Appearance).where(Appearance.id.in_(appearances)))
            if session_id is not None:
                await session.execute(delete(CameraSession).where(CameraSession.id == session_id))
            if camera_id is not None:
                await session.execute(delete(Camera).where(Camera.id == camera_id))
        await database.close()


@pytest.mark.anyio
async def test_low_filesystem_headroom_is_rejected_before_runtime_shutdown(
    database_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from gods_watching.model_selection.registry import ClipModelRegistry
    from gods_watching.storage import Database

    database = Database.connect(database_url)
    storage = StorageRepository(CredentialCipher(Fernet.generate_key()))
    service = ModelSelectionService(
        database=database,
        registry=ClipModelRegistry(),
        prepared=_Prepared(),
        storage=storage,
    )
    crop_store = CropObjectStore(tmp_path / "low-space-crops")

    def _low_space(_path: Path) -> SimpleNamespace:
        return SimpleNamespace(free=0)

    monkeypatch.setattr("gods_watching.model_selection.service.shutil.disk_usage", _low_space)
    try:
        async with database.transaction() as session:
            reason = await service._headroom_error(
                session,
                crop_store=crop_store,
                target_dimension=768,
                retained_count=1,
            )
        assert reason == "insufficient filesystem headroom for model transition staging"
    finally:
        await database.close()


@pytest.mark.anyio
async def test_transition_advisory_lock_waits_for_shared_search_and_releases_on_cancel(
    database_url: str,
) -> None:
    from gods_watching.storage import Database

    database = Database.connect(database_url)
    coordinator = TransitionCoordinator(database)
    entered = anyio.Event()

    async def acquire_transition() -> None:
        async with coordinator.transition_lock():
            entered.set()

    try:
        async with database.transaction() as search_session:
            transition = asyncio.create_task(acquire_transition())
            async with coordinator.search_lock(search_session):
                with anyio.fail_after(0.2):
                    await anyio.sleep(0.05)
                assert not entered.is_set()
        with anyio.fail_after(2.0):
            await entered.wait()
        await transition

        async def cancelled_owner() -> None:
            async with coordinator.transition_lock():
                await anyio.sleep_forever()

        owner = asyncio.create_task(cancelled_owner())
        await anyio.sleep(0.05)
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        async with coordinator.transition_lock():
            pass
    finally:
        await database.close()


@pytest.mark.anyio
async def test_concurrent_apply_requests_create_at_most_one_durable_job(
    database_url: str,
) -> None:
    from gods_watching.model_selection.registry import ClipModelRegistry
    from gods_watching.storage import Database

    target = _target_package()
    default = ClipModelPackage(
        model_id="openai/clip-vit-base-patch16",
        revision=_SOURCE,
        snapshot_path=Path("/models/clip"),
        dimension=512,
        processor="CLIPProcessor",
        runtime="transformers",
    )
    registry = ClipModelRegistry((default, target))
    first_db = Database.connect(database_url)
    second_db = Database.connect(database_url)
    first = ModelSelectionService(first_db, registry, _Prepared())
    second = ModelSelectionService(second_db, registry, _Prepared())

    async def apply(service: ModelSelectionService, database: Database) -> object:
        async with database.transaction() as session:
            return await service.apply(session, target.model_id)

    try:
        results = await asyncio.gather(
            apply(first, first_db),
            apply(second, second_db),
            return_exceptions=True,
        )
        accepted = [result for result in results if not isinstance(result, Exception)]
        conflicts = [result for result in results if isinstance(result, Exception)]
        assert len(accepted) == 1
        assert len(conflicts) == 1
        assert isinstance(conflicts[0], Exception)
    finally:
        async with first_db.transaction() as cleanup:
            await cleanup.execute(delete(ModelTransitionJob))
            active = await cleanup.scalar(
                select(ActiveModelIdentity).where(ActiveModelIdentity.singleton.is_(True))
            )
            if active is not None:
                active.model_id = default.model_id
                active.model_revision = default.revision
                active.embedding_dimension = default.dimension
        await first_db.close()
        await second_db.close()
