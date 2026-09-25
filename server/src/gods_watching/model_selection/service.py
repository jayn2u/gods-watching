"""Application and worker services for durable CLIP model transitions."""

# ruff: noqa: TRY003, EM101, TRY301, BLE001, TC001, TC002, TC003, C901, PLR0913, PLR0912, PLR0915

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gods_watching.contracts.model_selection import (
    ModelCatalogEntry,
    ModelSettingsResponse,
    ModelTransitionPhase,
    ModelTransitionResponse,
)
from gods_watching.inference.clip import (
    ClipImageDecodeError,
    ClipInferenceError,
    ClipInputError,
    ClipRuntimeIdentity,
)
from gods_watching.model_selection.assets import PreparedModelStatus
from gods_watching.model_selection.coordinator import TransitionCoordinator
from gods_watching.model_selection.imported_manifest import ImportedClipManifest, ImportedFile
from gods_watching.model_selection.models import (
    CropSkipReason,
    ModelNotPreparedError,
    ModelSelectionConflictError,
    TransitionPhase,
    TransitionRecoveryError,
    TransitionResult,
)
from gods_watching.model_selection.preflight import (
    SwitchPreflight,
    estimate_switch,
    measured_rehearsal,
    scan_retained,
)
from gods_watching.model_selection.quality import (
    QualityStatus,
    assess_quality,
    load_quality_evidence,
    load_quality_policy,
)
from gods_watching.model_selection.registry import ClipModelPackage, ClipModelRegistry
from gods_watching.model_selection.repository import (
    StageResult,
    TransitionRepository,
    transition_state,
)
from gods_watching.model_selection.transition_observer import TransitionObserver
from gods_watching.retention.models import DEFAULT_CLEANUP_THRESHOLD, DEFAULT_MINIMUM_FREE_BYTES
from gods_watching.storage import (
    ApplicationSettings,
    CropObjectStore,
    Database,
    StorageRepository,
)
from gods_watching.storage.managed_files import safe_scan

if TYPE_CHECKING:
    from gods_watching.storage.models import ModelTransitionJob


class PreparedModelCatalogPort(Protocol):
    """Read-only preparation status supplied by deployment preparation."""

    def status(self, package: ClipModelPackage) -> PreparedModelStatus:
        """Return a cached local package check without downloading weights."""
        ...


class ClipRuntimePort(Protocol):
    """Own one exact resident CLIP package."""

    async def load_model(self, package: ClipModelPackage) -> ClipRuntimeIdentity:
        """Load and verify one package."""
        ...

    async def unload_model(self) -> None:
        """Unload all CLIP modalities."""
        ...

    async def inspect_identity(self) -> ClipRuntimeIdentity:
        """Read Triton's verified identity."""
        ...


class PipelineLifecyclePort(Protocol):
    """Stop and restart the publication/retention owner around a switch."""

    async def stop_and_join(self) -> None:
        """Deliver terminal handoffs and wait for all pipeline tasks to exit."""
        ...

    async def start(self, package: ClipModelPackage) -> None:
        """Construct a fresh pipeline bound to the package."""
        ...


class ClipImageEmbeddingPort(Protocol):
    """Image-only inference boundary used while staging one crop."""

    async def embed_image(self, image: bytes, /) -> Sequence[float]:
        """Return a normalized target image vector."""
        ...


class ClipFactoryPort(Protocol):
    """Create an inference adapter bound to one loaded package."""

    def __call__(self, package: ClipModelPackage) -> ClipImageEmbeddingPort:
        """Return the target image embedding adapter."""
        ...


@dataclass(frozen=True, slots=True)
class ModelSelectionService:
    """Serve catalog/apply requests and execute one durable job at a time."""

    database: Database
    registry: ClipModelRegistry
    prepared: PreparedModelCatalogPort
    repository: TransitionRepository = field(default_factory=TransitionRepository)
    coordinator: TransitionCoordinator | None = None
    storage: StorageRepository | None = None
    imported_assets_root: Path | None = None
    quality_policy_path: Path | None = None
    quality_evidence_root: Path | None = None
    preflight_assets_root: Path | None = None
    preflight_crop_store: CropObjectStore | None = None

    def __post_init__(self) -> None:
        """Bind the database advisory coordinator when one is not injected."""
        if self.coordinator is None:
            object.__setattr__(self, "coordinator", TransitionCoordinator(self.database))

    async def get(self, session: AsyncSession) -> ModelSettingsResponse:
        """Return the durable catalog and current progress/outcome."""
        active, job = await self.repository.state(session, default=self.registry.default)
        return await self._response(active.model_id, job)

    async def apply(self, session: AsyncSession, model_id: str) -> ModelSettingsResponse:
        """Validate and queue one prepared model without downloading weights."""
        package = self.registry.get(model_id)
        if package is None:
            # Keep unknown identifiers separate from unavailable local assets;
            # the API maps both to the existing structured 422 shape.
            raise ModelNotPreparedError(model_id, code="unknown_model")
        status = await self._prepared_status(package)
        if not status.prepared:
            raise ModelNotPreparedError(
                status.reason or "selected model is not prepared",
                code="model_not_prepared",
            )
        quality = await asyncio.to_thread(self._quality_status, package)
        if not quality.passed:
            raise ModelNotPreparedError(
                quality.reason or "selected model has no qualifying evidence",
                code="model_quality_ineligible",
            )
        active, job = await self.repository.state(session, default=self.registry.default)
        if (
            active.model_id == package.model_id
            and active.model_revision == package.revision
            and active.embedding_dimension == package.dimension
        ):
            return await self._response(active.model_id, job)
        if job is not None and TransitionPhase(job.phase) in {
            TransitionPhase.QUEUED,
            TransitionPhase.PREPARING,
            TransitionPhase.REINDEXING,
            TransitionPhase.ACTIVATING,
            TransitionPhase.ROLLING_BACK,
        }:
            raise ModelSelectionConflictError("another model transition is already active")
        preflight = await self.preflight(session, model_id)
        if not preflight.eligible:
            raise ModelNotPreparedError(
                preflight.reason or "model switch preflight failed",
                code="model_preflight_ineligible",
            )
        try:
            created = await self.repository.create_job(
                session,
                target=package,
                default=self.registry.default,
            )
        except RuntimeError as error:
            raise ModelSelectionConflictError(str(error)) from error
        return await self._response(active.model_id, created)

    async def preflight(self, session: AsyncSession, model_id: str) -> SwitchPreflight:
        """Recompute the current corpus estimate without changing runtime identity."""
        package = self.registry.get(model_id)
        if package is None:
            raise ModelNotPreparedError(model_id, code="unknown_model")
        if self.preflight_crop_store is None or self.preflight_assets_root is None:
            return estimate_switch(
                0, None, 0, target_model_id=package.model_id
            )
        retained, missing = await scan_retained(session, self.preflight_crop_store)
        rehearsal = measured_rehearsal(self.preflight_assets_root, package)
        return estimate_switch(
            retained, rehearsal[0] if rehearsal else None, missing,
            target_model_id=package.model_id,
            measured_fixed_seconds=rehearsal[1] if rehearsal else None,
        )

    async def run_pending(
        self,
        *,
        crop_store: CropObjectStore,
        runtime: ClipRuntimePort,
        clip_factory: ClipFactoryPort,
        pipeline: PipelineLifecyclePort,
        batch_size: int = 32,
        observer: TransitionObserver | None = None,
    ) -> TransitionResult | None:
        """Run the oldest active job through staging and atomic activation."""
        if observer is not None:
            observer.clear()
        async with self.database.transaction() as session:
            job = await self.repository.active_job(session, lock=True)
            if job is None:
                return None
            source_package = self.registry.get(job.source_model_id)
            target_package = self.registry.get(job.target_model_id)
            if source_package is None or target_package is None:
                raise TransitionRecoveryError(
                    "transition references an unavailable registry package"
                )
            _require_identity(source_package, job.source_model_revision, job.source_dimension)
            _require_identity(target_package, job.target_model_revision, job.target_dimension)
            job_id = job.id
            recovery_only = job.phase == TransitionPhase.ROLLING_BACK.value
            if observer is not None and not recovery_only:
                observer.begin(
                    source_package,
                    target_package,
                    fresh=job.phase == TransitionPhase.QUEUED.value,
                )
            if not recovery_only and job.phase == TransitionPhase.QUEUED.value:
                retained = await self.repository.retained_count(session)
                headroom_error = await self._headroom_error(
                    session,
                    crop_store=crop_store,
                    target_dimension=target_package.dimension,
                    retained_count=retained,
                )
                if headroom_error is not None:
                    failed = await self.repository.set_phase(
                        session,
                        job.id,
                        TransitionPhase.FAILED,
                        error=headroom_error,
                    )
                    await self.repository.delete_stages(session, job.id)
                    return TransitionResult(state=transition_state(failed), activated=False)
        if recovery_only:
            return await self.recover_startup(runtime=runtime, pipeline=pipeline)
        # Terminal handoffs may still publish while close is shielded.  Do not
        # acquire the model lock or unload inference until every pipeline task
        # has joined.
        if observer is not None:
            observer.start("pipeline_stop")
        await pipeline.stop_and_join()
        if observer is not None:
            observer.end("pipeline_stop")
        coordinator = self.coordinator
        if coordinator is None:
            raise RuntimeError("model transition coordinator is not configured")
        try:
            async with coordinator.transition_lock():
                return await self._run_locked(
                    job_id=job_id,
                    source_package=source_package,
                    target_package=target_package,
                    crop_store=crop_store,
                    runtime=runtime,
                    clip_factory=clip_factory,
                    pipeline=pipeline,
                    batch_size=batch_size,
                    observer=observer,
                )
        except ModelSelectionConflictError:
            # A second owner cannot occur under worker ownership, but a caller
            # may have raced an external process.  Restore publication first.
            await pipeline.start(source_package)
            raise

    async def _run_locked(
        self,
        *,
        job_id: UUID,
        source_package: ClipModelPackage,
        target_package: ClipModelPackage,
        crop_store: CropObjectStore,
        runtime: ClipRuntimePort,
        clip_factory: ClipFactoryPort,
        pipeline: PipelineLifecyclePort,
        batch_size: int,
        observer: TransitionObserver | None,
    ) -> TransitionResult:
        try:
            if observer is not None and observer.fresh:
                observer.start("stage_population")
            async with self.database.transaction() as session:
                job = await self.repository.get_job(session, job_id, lock=True)
                if job is None:
                    raise TransitionRecoveryError("transition job disappeared")
                if job.phase == TransitionPhase.QUEUED.value:
                    _ = await self.repository.populate_stages(session, job)
                elif observer is not None:
                    observer.fresh = False
                if job.phase == TransitionPhase.PREPARING.value:
                    job.phase = TransitionPhase.REINDEXING.value
                    await session.flush()
            if observer is not None:
                if observer.fresh:
                    observer.end("stage_population")
                observer.start("runtime_switch")
            # Ensure only the selected runtime remains resident.  The runtime
            # manager itself verifies Triton identity after each load.
            await runtime.unload_model()
            identity = await runtime.load_model(target_package)
            if observer is not None:
                if (identity.model_id, identity.revision, identity.dimension) != (
                    target_package.model_id, target_package.revision, target_package.dimension
                ):
                    raise TransitionRecoveryError("loaded runtime identity does not match target")
                observer.end("runtime_switch")
                observer.start("crop_embedding")
            target_clip = clip_factory(target_package)
            while True:
                async with self.database.transaction() as session:
                    rows = await self.repository.pending_stages(
                        session,
                        job_id,
                        limit=batch_size,
                    )
                if not rows:
                    break
                results: list[StageResult] = []
                crop_seconds: list[float | None] = []
                for stage, _appearance in rows:
                    started_at = monotonic() if observer is not None else None
                    results.append(
                        await self._embed_stage(
                            stage.appearance_id,
                            stage.source_crop_object_key,
                            crop_store,
                            target_clip,
                        )
                    )
                    crop_seconds.append(
                        monotonic() - started_at if started_at is not None else None
                    )
                async with self.database.transaction() as session:
                    job = await self.repository.get_job(
                        session,
                        job_id,
                        lock=True,
                    )
                    if job is None:
                        raise TransitionRecoveryError("transition job disappeared")
                    await self.repository.record_stage_results(session, job, results)
                if observer is not None:
                    for result, elapsed in zip(results, crop_seconds, strict=True):
                        if result.embedding is not None and elapsed is not None:
                            observer.committed_crop(elapsed)
            if observer is not None:
                observer.end("crop_embedding")
                observer.start("activation")
            async with self.database.transaction() as session:
                job = await self.repository.get_job(session, job_id, lock=True)
                if job is None:
                    raise TransitionRecoveryError("transition job disappeared")
                job.phase = TransitionPhase.ACTIVATING.value
                await session.flush()
                _ = await self.repository.activate(
                    session,
                    job_id,
                    expected_source=(
                        source_package.model_id,
                        source_package.revision,
                        source_package.dimension,
                    ),
                )
            if observer is not None:
                observer.end("activation")
                observer.start("pipeline_restart")
            await pipeline.start(target_package)
            if observer is not None:
                observer.end("pipeline_restart")
            async with self.database.transaction() as session:
                job = await self.repository.get_job(session, job_id)
                if job is None:
                    raise TransitionRecoveryError("transition job disappeared after activation")
                result = TransitionResult(state=transition_state(job), activated=True)
                if observer is not None and job.phase == TransitionPhase.SUCCEEDED.value:
                    observer.finish()
                return result
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:
            if isinstance(error, (asyncio.CancelledError,)):
                # Leave the durable active phase for startup recovery.  A
                # process crash cannot execute the rollback half-way through.
                raise
            return await self._recover_after_error(
                job_id=job_id,
                source_package=source_package,
                target_package=target_package,
                runtime=runtime,
                pipeline=pipeline,
                error=error,
            )

    async def _recover_after_error(
        self,
        *,
        job_id: UUID,
        source_package: ClipModelPackage,
        target_package: ClipModelPackage,
        runtime: ClipRuntimePort,
        pipeline: PipelineLifecyclePort,
        error: BaseException,
    ) -> TransitionResult:
        """Choose recovery from the committed identity, never a local flag."""
        try:
            async with self.database.transaction() as session:
                active, _job = await self.repository.state(session, default=self.registry.default)
                identity = (
                    active.model_id,
                    active.model_revision,
                    active.embedding_dimension,
                )
        except BaseException as read_error:
            if isinstance(read_error, asyncio.CancelledError):
                raise
            # Commit acknowledgement loss or database failure is ambiguous.
            # Leave the pipeline stopped and force startup reconciliation.
            message = _safe_error(read_error)
            error_message = f"durable identity unavailable during recovery: {message}"
            raise TransitionRecoveryError(error_message) from read_error
        source_identity = (
            source_package.model_id,
            source_package.revision,
            source_package.dimension,
        )
        target_identity = (
            target_package.model_id,
            target_package.revision,
            target_package.dimension,
        )
        if identity == target_identity:
            return await self._restore_after_activation(
                job_id=job_id,
                target_package=target_package,
                runtime=runtime,
                pipeline=pipeline,
                error=error,
            )
        if identity == source_identity:
            return await self._rollback(
                job_id=job_id,
                source_package=source_package,
                runtime=runtime,
                pipeline=pipeline,
                error=error,
            )
        raise TransitionRecoveryError("durable identity does not match transition endpoints")

    async def _headroom_error(
        self,
        session: AsyncSession,
        *,
        crop_store: CropObjectStore,
        target_dimension: int,
        retained_count: int,
    ) -> str | None:
        """Apply the existing retention quota/free-space policy before downtime."""
        storage = self.storage
        if storage is None:
            return "storage accounting unavailable"
        settings = await session.scalar(
            select(ApplicationSettings).where(ApplicationSettings.singleton.is_(True))
        )
        quota = 100_000_000_000 if settings is None else settings.quota_bytes
        physical = sum(item.byte_size for item in safe_scan(crop_store.root))
        relations = await storage.application_relation_sizes(session)
        # A stage row carries a target vector plus metadata/TOAST overhead.
        estimate = retained_count * (target_dimension * 4 + 2048)
        projected = physical + sum(relations.values()) + estimate
        if projected > int(quota * DEFAULT_CLEANUP_THRESHOLD):
            return "insufficient storage quota for model transition staging"
        free = shutil.disk_usage(crop_store.root).free
        if free - estimate < DEFAULT_MINIMUM_FREE_BYTES:
            return "insufficient filesystem headroom for model transition staging"
        return None

    async def _restore_after_activation(
        self,
        *,
        job_id: UUID,
        target_package: ClipModelPackage,
        runtime: ClipRuntimePort,
        pipeline: PipelineLifecyclePort,
        error: BaseException,
    ) -> TransitionResult:
        """Keep a committed target identity paired with a verified target runtime."""
        safe_message = _safe_error(error)
        try:
            async with self.database.transaction() as session:
                _ = await self.repository.set_phase(
                    session,
                    job_id,
                    TransitionPhase.ROLLING_BACK,
                    error=safe_message,
                )
            with suppress(Exception):
                await runtime.unload_model()
            _ = await runtime.load_model(target_package)
            await pipeline.start(target_package)
        except BaseException as restore_error:
            async with self.database.transaction() as session:
                job = await self.repository.set_phase(
                    session,
                    job_id,
                    TransitionPhase.ROLLING_BACK,
                    error=_safe_error(restore_error),
                )
                await self.repository.delete_stages(session, job_id)
            return TransitionResult(state=transition_state(job), activated=True)
        async with self.database.transaction() as session:
            job = await self.repository.set_phase(
                session,
                job_id,
                TransitionPhase.SUCCEEDED,
                error=None,
            )
            return TransitionResult(state=transition_state(job), activated=True)

    async def recover_startup(
        self,
        *,
        runtime: ClipRuntimePort,
        pipeline: PipelineLifecyclePort,
    ) -> TransitionResult | None:
        """Reconcile runtime identity with the durable active identity after restart."""
        async with self.database.transaction() as session:
            active, job = await self.repository.state(session, default=self.registry.default)
            active_package = self.registry.get(active.model_id)
            if active_package is None:
                raise TransitionRecoveryError("active model is absent from the registry")
            pending = (
                job
                if job is not None
                and TransitionPhase(job.phase)
                in {
                    TransitionPhase.QUEUED,
                    TransitionPhase.PREPARING,
                    TransitionPhase.REINDEXING,
                    TransitionPhase.ACTIVATING,
                    TransitionPhase.ROLLING_BACK,
                }
                else None
            )
        _require_identity(
            active_package,
            active.model_revision,
            active.embedding_dimension,
        )
        if pending is not None and (
            active.model_id,
            active.model_revision,
            active.embedding_dimension,
        ) == (
            pending.target_model_id,
            pending.target_model_revision,
            pending.target_dimension,
        ):
            # The activation transaction committed before the process died.
            _ = await runtime.load_model(active_package)
            await pipeline.start(active_package)
            async with self.database.transaction() as session:
                row = await self.repository.get_job(session, pending.id, lock=True)
                if row is not None and row.phase != TransitionPhase.SUCCEEDED.value:
                    row.phase = TransitionPhase.SUCCEEDED.value
                    row.error = None
                    row.updated_at = datetime.now(UTC)
                    row.finished_at = datetime.now(UTC)
                    await session.flush()
                if row is None:
                    return None
                return TransitionResult(state=transition_state(row), activated=True)
        if (
            pending is not None
            and (
                active.model_id,
                active.model_revision,
                active.embedding_dimension,
            )
            == (
                pending.source_model_id,
                pending.source_model_revision,
                pending.source_dimension,
            )
            and pending.phase != TransitionPhase.ROLLING_BACK.value
        ):
            # The source identity is still authoritative.  Reload it and let
            # the worker resume the queued/staged job from its last durable
            # batch rather than discarding progress. A staged job keeps
            # publication and retention stopped until its snapshot resumes.
            _ = await runtime.load_model(active_package)
            if pending.phase == TransitionPhase.QUEUED.value:
                await pipeline.start(active_package)
            return TransitionResult(state=transition_state(pending), activated=False)
        if pending is not None:
            source = self.registry.get(pending.source_model_id)
            if source is None:
                raise TransitionRecoveryError("transition source package is absent")
            _require_identity(source, pending.source_model_revision, pending.source_dimension)
            return await self._rollback(
                job_id=pending.id,
                source_package=source,
                runtime=runtime,
                pipeline=pipeline,
                error=RuntimeError("transition interrupted before activation"),
            )
        _ = await runtime.load_model(active_package)
        await pipeline.start(active_package)
        return None

    async def _rollback(
        self,
        *,
        job_id: UUID,
        source_package: ClipModelPackage,
        runtime: ClipRuntimePort,
        pipeline: PipelineLifecyclePort,
        error: BaseException,
    ) -> TransitionResult:
        safe_message = _safe_error(error)
        try:
            async with self.database.transaction() as session:
                _ = await self.repository.set_phase(
                    session,
                    job_id,
                    TransitionPhase.ROLLING_BACK,
                    error=safe_message,
                )
            with suppress(Exception):
                await runtime.unload_model()
            _ = await runtime.load_model(source_package)
            await pipeline.start(source_package)
        except BaseException as restore_error:
            recovery_message = _safe_error(restore_error)
            async with self.database.transaction() as session:
                job = await self.repository.set_phase(
                    session,
                    job_id,
                    TransitionPhase.ROLLING_BACK,
                    error=recovery_message,
                )
                await self.repository.delete_stages(session, job_id)
            return TransitionResult(state=transition_state(job), activated=False)
        async with self.database.transaction() as session:
            job = await self.repository.set_phase(
                session,
                job_id,
                TransitionPhase.FAILED,
                error=safe_message,
            )
            await self.repository.delete_stages(session, job_id)
            return TransitionResult(state=transition_state(job), activated=False)

    async def _embed_stage(
        self,
        appearance_id: UUID,
        crop_key: str,
        crop_store: CropObjectStore,
        clip: ClipImageEmbeddingPort,
    ) -> StageResult:
        try:
            payload = crop_store.read(crop_key)
        except FileNotFoundError:
            return StageResult(
                appearance_id=appearance_id,
                embedding=None,
                skip_reason=CropSkipReason.MISSING.value,
            )
        except OSError:
            # Permission, I/O, and capacity errors are infrastructure failures.
            raise
        except ValueError:
            # A key that cannot be parsed is a missing/invalid crop reference.
            return StageResult(
                appearance_id=appearance_id,
                embedding=None,
                skip_reason=CropSkipReason.MISSING.value,
            )
        if not payload:
            return StageResult(
                appearance_id=appearance_id,
                embedding=None,
                skip_reason=CropSkipReason.UNDECODABLE.value,
            )
        try:
            embedding = await clip.embed_image(payload)
        except ClipImageDecodeError:
            return StageResult(
                appearance_id=appearance_id,
                embedding=None,
                skip_reason=CropSkipReason.UNDECODABLE.value,
            )
        except ClipInputError as error:
            if error.code == "clip_image_empty":
                return StageResult(
                    appearance_id=appearance_id,
                    embedding=None,
                    skip_reason=CropSkipReason.UNDECODABLE.value,
                )
            raise
        except ClipInferenceError:
            raise
        return StageResult(
            appearance_id=appearance_id,
            embedding=tuple(float(value) for value in embedding),
            skip_reason=None,
        )

    async def _response(
        self,
        active_model_id: str,
        job: ModelTransitionJob | None,
    ) -> ModelSettingsResponse:
        entries: list[ModelCatalogEntry] = []
        for package in self.registry.packages:
            status = await self._prepared_status(package)
            quality = await asyncio.to_thread(self._quality_status, package)
            entries.append(
                ModelCatalogEntry(
                    model_id=package.model_id,
                    display_name=package.display_name or package.model_id,
                    revision=package.revision,
                    dimension=package.dimension,
                    prepared=status.prepared,
                    reason=status.reason,
                    quality_passed=quality.passed,
                    quality_reason=quality.reason,
                )
            )
        transition = None
        if job is not None:
            state = transition_state(job)
            transition = ModelTransitionResponse(
                id=state.id,
                source_model_id=state.source_model_id,
                target_model_id=state.target_model_id,
                phase=ModelTransitionPhase(state.phase.value),
                processed=state.processed,
                total=state.total,
                skipped=state.skipped,
                skip_reasons=state.skip_reasons,
                error=state.error,
            )
        maintenance = transition is not None and transition.phase in {
            ModelTransitionPhase.QUEUED,
            ModelTransitionPhase.PREPARING,
            ModelTransitionPhase.REINDEXING,
            ModelTransitionPhase.ACTIVATING,
            ModelTransitionPhase.ROLLING_BACK,
        }
        return ModelSettingsResponse(
            active_model_id=active_model_id,
            maintenance=maintenance,
            models=tuple(entries),
            transition=transition,
        )

    def _quality_status(self, package: ClipModelPackage) -> QualityStatus:
        if not package.snapshot_path.parts or "imported" not in package.snapshot_path.parts:
            return QualityStatus(passed=True, reason=None)
        if self.imported_assets_root is None or self.quality_evidence_root is None:
            return QualityStatus(passed=False, reason="quality evidence store unavailable")
        if self.quality_policy_path is None:
            return QualityStatus(passed=False, reason="trusted quality policy missing")
        package_dir = self.imported_assets_root / package.revision
        try:
            raw = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest = ImportedClipManifest(
                model_id=raw["model_id"],
                revision=raw["revision"],
                display_name=raw["display_name"],
                base_model_id=raw["base_model_id"],
                dimension=raw["dimension"],
                files=tuple(ImportedFile(**item) for item in raw["files"]),
                package_sha256=raw["package_sha256"],
                cuhk_report=raw["cuhk_report"],
            )
        except (OSError, ValueError, TypeError, KeyError):
            return QualityStatus(passed=False, reason="installed quality manifest invalid")
        if manifest.package_sha256 != package.revision or manifest.model_id != package.model_id:
            return QualityStatus(passed=False, reason="installed package identity mismatch")
        evidence = load_quality_evidence(self.quality_evidence_root / f"{package.revision}.json")
        return assess_quality(manifest, evidence, load_quality_policy(self.quality_policy_path))

    async def _prepared_status(self, package: ClipModelPackage) -> PreparedModelStatus:
        try:
            return await asyncio.to_thread(self.prepared.status, package)
        except Exception as error:
            return PreparedModelStatus(prepared=False, reason=_safe_error(error))


def _safe_error(error: BaseException) -> str:
    """Keep durable errors bounded and free of paths/credentials."""
    message = str(error).strip().replace("\n", " ")
    if not message:
        message = type(error).__name__
    return message[:240]


def _require_identity(
    package: ClipModelPackage,
    revision: str,
    dimension: int,
) -> None:
    """Reject a durable identity whose full package identity drifted."""
    if package.revision != revision or package.dimension != dimension:
        message = f"registry identity mismatch for {package.model_id}"
        raise TransitionRecoveryError(message)


__all__ = [
    "ClipFactoryPort",
    "ClipImageEmbeddingPort",
    "ClipRuntimePort",
    "ModelSelectionService",
    "PipelineLifecyclePort",
    "PreparedModelCatalogPort",
]
