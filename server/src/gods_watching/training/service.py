"""Service errors and durable request APIs for the training routes."""

# ruff: noqa: TRY003, EM101

from __future__ import annotations

import asyncio
import base64
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal, cast
from uuid import UUID, uuid4

from gods_watching.contracts.training import (
    TrainingConfig,
    TrainingDatasetSnapshot,
    TrainingDatasetStatus,
    TrainingJobPage,
    TrainingJobResponse,
    TrainingJobSubmitRequest,
    TrainingLogPage,
    TrainingMetricPage,
    TrainingPreflightResponse,
)
from gods_watching.training.dataset import (
    DatasetManifest,
    DatasetValidationError,
    validate_cuhk,
)
from gods_watching.training.metrics import (
    MetricCursorError,
    MetricHistoryError,
    read_log_page,
    read_metric_page,
)
from gods_watching.training.models import (
    TrainingJob,
    TrainingRequest,
    TrainingRequestKind,
    TrainingRequestPhase,
)
from gods_watching.training.repository import (
    TrainingJobConflictError as RepositoryJobConflictError,
)
from gods_watching.training.repository import (
    TrainingJobNotFoundError as RepositoryJobNotFoundError,
)
from gods_watching.training.repository import (
    TrainingPhaseTransitionError,
    TrainingRepository,
)
from gods_watching.training.repository import (
    TrainingRequestConflictError as RepositoryRequestConflictError,
)
from gods_watching.training.settings import (
    TrainingDatasetNotConfiguredError,
    TrainingSettings,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from gods_watching.storage import Database

TrainingJobPhaseValue = Literal[
    "starting",
    "training",
    "evaluating",
    "publishing",
    "succeeded",
    "cancelling",
    "cancelled",
    "failed",
    "interrupted",
]


class TrainingServiceError(RuntimeError):
    """Base class for bounded, code-bearing training failures."""

    code: str = "training_service_failed"
    message: str

    def __init__(self, message: str, *, code: str | None = None) -> None:
        """Initialize an error with its public message and optional code."""
        self.message = message
        if code is not None:
            self.code = code
        super().__init__(message)


class TrainingMemoryRefusedError(TrainingServiceError):
    """Memory admission declined a requested profile without creating a job."""

    code: str = "training_memory_refused"
    required_bytes: int
    free_bytes: int
    reserve_bytes: int
    training_peak_bytes: int | None
    reason: str

    def __init__(
        self,
        *,
        required_bytes: int,
        free_bytes: int,
        reserve_bytes: int,
        training_peak_bytes: int | None = None,
        reason: str = "insufficient_free_memory",
    ) -> None:
        """Store the exact byte counts that explain a memory refusal."""
        self.required_bytes = required_bytes
        self.free_bytes = free_bytes
        self.reserve_bytes = reserve_bytes
        self.training_peak_bytes = training_peak_bytes
        self.reason = reason
        super().__init__("training memory admission refused")


class TrainingSupervisorUnavailableError(TrainingServiceError):
    """No supervisor accepted the durable request before its bounded deadline."""

    code: str = "training_supervisor_unavailable"

    def __init__(self) -> None:
        """Describe a request that expired before supervisor resolution."""
        super().__init__("training supervisor did not respond before request expiry")


class TrainingRequestConflictError(TrainingServiceError):
    """A request UUID was reused with different immutable inputs."""

    code: str = "training_request_conflict"

    def __init__(self) -> None:
        """Describe an idempotency key reused with different inputs."""
        super().__init__("request identity was already used with different inputs")


class TrainingJobConflictError(TrainingServiceError):
    """Another training job or orphan process currently owns the GPU slot."""

    code: str = "training_job_conflict"

    def __init__(self, message: str = "another training job is active") -> None:
        """Describe a job or orphan process that owns the GPU slot."""
        super().__init__(message)


class TrainingJobNotFoundError(TrainingServiceError):
    """The requested durable training job does not exist."""

    code: str = "training_job_not_found"

    def __init__(self) -> None:
        """Describe a requested job that is absent from durable history."""
        super().__init__("training job was not found")


class TrainingJobStateError(TrainingServiceError):
    """A requested operation is not valid for the durable job phase."""

    code: str = "training_job_state_invalid"

    def __init__(self, message: str = "training job operation is invalid in this phase") -> None:
        """Describe an operation that is not valid for the current phase."""
        super().__init__(message)


class TrainingDatasetUnavailableError(TrainingServiceError):
    """The operator-configured CUHK-PEDES dataset cannot be used."""

    code: str = "training_dataset_unavailable"

    def __init__(self, message: str = "registered training dataset is unavailable") -> None:
        """Describe an unavailable or changed registered dataset."""
        super().__init__(message)


class TrainingCursorError(TrainingServiceError):
    """A history cursor is malformed or no longer identifies a job boundary."""

    code: str = "training_cursor_invalid"

    def __init__(self) -> None:
        """Describe a malformed or stale history cursor."""
        super().__init__("training history cursor is invalid")


class TrainingHistoryUnavailableError(TrainingServiceError):
    """A durable metric/log history file is malformed or unreadable."""

    code: str = "training_history_unavailable"

    def __init__(self) -> None:
        """Describe a safe history read failure without returning filesystem details."""
        super().__init__("training metric/log history is unavailable")


class TrainingService:
    """CPU-only API coordinator that commits requests before waiting for workers."""

    _database: Database
    _settings: TrainingSettings
    _dataset_validator: Callable[[Path], DatasetManifest]
    _repository: TrainingRepository
    _manifest: DatasetManifest | None
    _manifest_lock: asyncio.Lock

    def __init__(
        self,
        database: Database,
        settings: TrainingSettings,
        *,
        dataset_validator: Callable[[Path], DatasetManifest] = validate_cuhk,
        repository: TrainingRepository | None = None,
    ) -> None:
        """Build an API service without importing CUDA or reading run directories."""
        self._database = database
        self._settings = settings
        self._dataset_validator = dataset_validator
        self._repository = repository or TrainingRepository()
        self._manifest = None
        self._manifest_lock = asyncio.Lock()

    async def datasets(self) -> TrainingDatasetStatus:
        """Return only public dataset identity/counts, never sample paths or captions."""
        try:
            manifest = await self._registered_manifest()
        except TrainingDatasetUnavailableError:
            return TrainingDatasetStatus(
                registered=self._settings.dataset_root is not None,
                valid=False,
                reason="dataset_invalid_or_unavailable",
                snapshot=None,
            )
        snapshot = TrainingDatasetSnapshot.model_validate(manifest.public_snapshot())
        return TrainingDatasetStatus(registered=True, valid=True, snapshot=snapshot)

    async def config(self) -> TrainingConfig:
        """Return the pinned server config defaults/ranges without GPU imports."""
        return TrainingConfig()

    async def preflight(self, config: TrainingConfig) -> TrainingPreflightResponse:
        """Commit an idempotent, short-lived supervisor request before waiting."""
        manifest = await self._registered_manifest()
        self._require_train_batch(config, manifest)
        request_id = uuid4()
        request = await self._create_request(
            request_id,
            TrainingRequestKind.PREFLIGHT,
            config,
            manifest,
        )
        response = await self._wait_for_request(request.request_id, request.expires_at)
        return TrainingPreflightResponse.model_validate(response)

    async def submit(self, request: TrainingJobSubmitRequest) -> TrainingJobResponse:
        """Commit caller identity/config/dataset, then await a bounded worker result."""
        existing = await self._existing_request(request.request_id)
        if existing is not None:
            self._require_request_match(
                existing,
                kind=TrainingRequestKind.SUBMIT,
                config=request.config,
            )
            if (
                not existing.dataset_snapshot
                or existing.dataset_snapshot.get("dataset_id") != request.dataset_id
            ):
                raise TrainingRequestConflictError
            response = await self._wait_for_request(existing.request_id, existing.expires_at)
            return await self.get_job(UUID(str(response["job_id"])))

        manifest = await self._registered_manifest()
        self._require_train_batch(request.config, manifest)
        try:
            durable = await self._create_request(
                request.request_id,
                TrainingRequestKind.SUBMIT,
                request.config,
                manifest,
            )
        except RepositoryRequestConflictError as error:
            raise TrainingRequestConflictError from error
        response = await self._wait_for_request(durable.request_id, durable.expires_at)
        job_id = UUID(str(response["job_id"]))
        return await self.get_job(job_id)

    async def list_jobs(self, *, cursor: str | None, limit: int) -> TrainingJobPage:
        """Return a stable descending keyset page of durable jobs."""
        before = self._decode_cursor(cursor) if cursor is not None else None
        async with self._database.transaction() as session:
            rows = await self._repository.list_jobs(session, limit=limit + 1, before=before)
        selected = rows[:limit]
        next_cursor = self._encode_cursor(selected[-1]) if len(rows) > limit and selected else None
        return TrainingJobPage(
            items=tuple(_job_response(row) for row in selected),
            next_cursor=next_cursor,
        )

    async def get_job(self, job_id: UUID) -> TrainingJobResponse:
        """Return one safe durable job view."""
        async with self._database.transaction() as session:
            job = await self._repository.get_job(session, job_id)
        if job is None:
            raise TrainingJobNotFoundError
        return _job_response(job)

    async def cancel(self, job_id: UUID, request_id: UUID) -> TrainingJobResponse:
        """Record a cooperative cancellation request in the job's durable row."""
        del request_id
        try:
            async with self._database.transaction() as session:
                job = await self._repository.request_cancel(session, job_id)
        except RepositoryJobConflictError as error:
            raise TrainingJobConflictError(str(error)) from error
        except RepositoryJobNotFoundError as error:
            raise TrainingJobNotFoundError from error
        except TrainingPhaseTransitionError as error:
            raise TrainingJobStateError from error
        return _job_response(job)

    async def resume(self, job_id: UUID, request_id: UUID) -> TrainingJobResponse:
        """Queue a same-snapshot resume request for the supervisor to re-admit."""
        existing = await self._existing_request(request_id)
        if existing is not None:
            self._require_request_match(
                existing,
                kind=TrainingRequestKind.RESUME,
                parent_job_id=job_id,
            )
            response = await self._wait_for_request(existing.request_id, existing.expires_at)
            return await self.get_job(UUID(str(response["job_id"])))

        async with self._database.transaction() as session:
            job = await self._repository.get_job(session, job_id)
        if job is None:
            raise TrainingJobNotFoundError
        if job.phase != "interrupted" or not job.checkpoint_path:
            raise TrainingJobStateError
        manifest = await self._registered_manifest()
        if manifest.fingerprint != job.dataset_fingerprint:
            raise TrainingDatasetUnavailableError("registered dataset fingerprint changed")
        config = TrainingConfig.model_validate(job.config_snapshot)
        try:
            durable = await self._create_request(
                request_id,
                TrainingRequestKind.RESUME,
                config,
                manifest,
                parent_job_id=job_id,
            )
        except RepositoryRequestConflictError as error:
            raise TrainingRequestConflictError from error
        response = await self._wait_for_request(durable.request_id, durable.expires_at)
        resumed_id = UUID(str(response["job_id"]))
        return await self.get_job(resumed_id)

    async def metrics(self, job_id: UUID, *, cursor: str | None, limit: int) -> TrainingMetricPage:
        """Return a bounded page of durable epoch metrics for a job."""
        job = await self._require_job(job_id)
        path = self._settings.training_root / "jobs" / str(job.id) / "metrics.jsonl"
        try:
            return read_metric_page(path, cursor=cursor, limit=limit)
        except MetricHistoryError as error:
            raise TrainingHistoryUnavailableError from error
        except MetricCursorError as error:
            raise TrainingCursorError from error

    async def logs(self, job_id: UUID, *, cursor: str | None, limit: int) -> TrainingLogPage:
        """Return a bounded page of safe worker log entries for a job."""
        job = await self._require_job(job_id)
        path = self._settings.training_root / "jobs" / str(job.id) / "logs.jsonl"
        try:
            return read_log_page(path, cursor=cursor, limit=limit)
        except MetricHistoryError as error:
            raise TrainingHistoryUnavailableError from error
        except MetricCursorError as error:
            raise TrainingCursorError from error

    async def _require_job(self, job_id: UUID) -> TrainingJob:
        async with self._database.transaction() as session:
            job = await self._repository.get_job(session, job_id)
        if job is None:
            raise TrainingJobNotFoundError
        return job

    async def _registered_manifest(self) -> DatasetManifest:
        # Revalidate the stat/content cache on every decision boundary. This
        # returns quickly for unchanged read-only data but invalidates changes.
        async with self._manifest_lock:
            try:
                root = self._settings.require_dataset_root()
                manifest = await asyncio.to_thread(self._dataset_validator, root)
            except (TrainingDatasetNotConfiguredError, DatasetValidationError, OSError) as error:
                raise TrainingDatasetUnavailableError from error
            self._manifest = manifest
            return manifest

    @staticmethod
    def _require_train_batch(config: TrainingConfig, manifest: DatasetManifest) -> None:
        if manifest.split_counts["train"].identities < config.micro_batch_size:
            raise TrainingDatasetUnavailableError(
                "training split does not have enough distinct identities for the micro-batch"
            )

    async def _create_request(
        self,
        request_id: UUID,
        kind: TrainingRequestKind,
        config: TrainingConfig,
        manifest: DatasetManifest,
        *,
        parent_job_id: UUID | None = None,
    ) -> TrainingRequest:
        expires_at = datetime.now(UTC) + timedelta(seconds=self._settings.request_timeout_seconds)
        async with self._database.transaction() as session:
            return await self._repository.create_request(
                session,
                request_id,
                kind,
                config,
                dataset=manifest.public_snapshot(),
                parent_job_id=parent_job_id,
                expires_at=expires_at,
            )

    async def _existing_request(self, request_id: UUID) -> TrainingRequest | None:
        async with self._database.transaction() as session:
            return await self._repository.get_request(session, request_id)

    @staticmethod
    def _require_request_match(
        request: TrainingRequest,
        *,
        kind: TrainingRequestKind,
        config: TrainingConfig | None = None,
        parent_job_id: UUID | None = None,
    ) -> None:
        if (
            request.kind != kind.value
            or request.parent_job_id != parent_job_id
            or (config is not None and request.config_snapshot != config.model_dump(mode="json"))
        ):
            raise TrainingRequestConflictError

    async def _wait_for_request(
        self,
        request_id: UUID,
        expires_at: datetime,
    ) -> dict[str, object]:
        deadline = asyncio.get_running_loop().time() + self._settings.request_timeout_seconds
        while True:
            async with self._database.transaction() as session:
                request = await self._repository.get_request(session, request_id)
            if request is None:
                raise TrainingSupervisorUnavailableError
            phase = TrainingRequestPhase(request.phase)
            if phase == TrainingRequestPhase.ACCEPTED:
                if request.job_id is not None:
                    return {"job_id": str(request.job_id)}
                if request.response_snapshot is None:
                    raise TrainingSupervisorUnavailableError
                return request.response_snapshot
            if phase == TrainingRequestPhase.REFUSED:
                self._raise_refusal(request.error, request.response_snapshot)
            if phase in {TrainingRequestPhase.EXPIRED, TrainingRequestPhase.FAILED}:
                raise TrainingSupervisorUnavailableError

            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0 or datetime.now(UTC) >= expires_at:
                async with self._database.transaction() as session:
                    expired = await self._repository.expire_request(session, request_id)
                if expired:
                    raise TrainingSupervisorUnavailableError
                # A supervisor may have committed just before the expiry CAS.
                continue
            await asyncio.sleep(min(self._settings.request_poll_interval_seconds, remaining))

    @staticmethod
    def _raise_refusal(error_code: str | None, response: dict[str, object] | None) -> None:
        result = response or {}
        if error_code == "training_job_conflict":
            raise TrainingJobConflictError
        raise TrainingMemoryRefusedError(
            required_bytes=_non_negative_response_int(result.get("required_bytes")),
            free_bytes=_non_negative_response_int(result.get("free_bytes")),
            reserve_bytes=_non_negative_response_int(result.get("reserve_bytes")),
            training_peak_bytes=_non_negative_response_int(result.get("training_peak_bytes")),
            reason=error_code or "memory_profile_unsupported",
        )

    @staticmethod
    def _encode_cursor(job: TrainingJob) -> str:
        raw = f"{job.created_at.isoformat()}|{job.id}".encode()
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @staticmethod
    def _decode_cursor(cursor: str) -> tuple[datetime, UUID]:
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            created_text, job_text = base64.urlsafe_b64decode(padded).decode("utf-8").split("|", 1)
            return datetime.fromisoformat(created_text), UUID(job_text)
        except (ValueError, UnicodeDecodeError) as error:
            raise TrainingCursorError from error


def _job_response(job: TrainingJob) -> TrainingJobResponse:
    """Convert ORM state to an API view that excludes paths and raw inputs."""
    return TrainingJobResponse(
        id=job.id,
        request_id=job.request_id,
        phase=cast("TrainingJobPhaseValue", job.phase),
        config=TrainingConfig.model_validate(job.config_snapshot),
        dataset=TrainingDatasetSnapshot.model_validate(job.dataset_snapshot),
        current_epoch=job.current_epoch,
        current_step=job.current_step,
        owner_generation=job.owner_generation,
        cancel_requested=job.cancel_requested,
        attempts=job.attempts,
        best_metric=job.best_metric,
        candidate_model_id=job.candidate_model_id,
        candidate_revision=job.candidate_revision,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
        finished_at=job.finished_at,
    )


def _non_negative_response_int(value: object) -> int:
    return value if type(value) is int and value >= 0 else 0


__all__ = [
    "TrainingCursorError",
    "TrainingDatasetUnavailableError",
    "TrainingHistoryUnavailableError",
    "TrainingJobConflictError",
    "TrainingJobNotFoundError",
    "TrainingJobStateError",
    "TrainingMemoryRefusedError",
    "TrainingRequestConflictError",
    "TrainingService",
    "TrainingServiceError",
    "TrainingSupervisorUnavailableError",
]
