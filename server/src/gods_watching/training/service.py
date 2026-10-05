"""Service errors and durable request APIs for the training routes."""

# ruff: noqa: TRY003, EM101

from __future__ import annotations

import asyncio
import base64
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal, cast
from uuid import UUID, uuid4

from gods_watching.contracts.training import (
    TrainingConfig,
    TrainingDatasetSnapshot,
    TrainingDatasetStatus,
    TrainingEvaluationSummary,
    TrainingJobPage,
    TrainingJobResponse,
    TrainingJobSubmitRequest,
    TrainingLogPage,
    TrainingMetricPage,
    TrainingPreflightResponse,
    TrainingSupervisorStatus,
)
from gods_watching.training.dataset import (
    DatasetManifest,
    DatasetValidationError,
    validate_cached_cuhk,
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
from gods_watching.training.readiness import read_supervisor_status
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


@dataclass(frozen=True, slots=True)
class TrainingMemoryRefusalMetadata:
    """Supervisor observation details attached to a memory refusal when available."""

    reason: str = "insufficient_free_memory"
    observed_at: datetime | None = None
    profile_identity: str | None = None


class TrainingMemoryRefusedError(TrainingServiceError):
    """Memory admission declined a requested profile without creating a job."""

    code: str = "training_memory_refused"
    required_bytes: int
    free_bytes: int
    reserve_bytes: int
    training_peak_bytes: int | None
    reason: str
    observed_at: datetime | None
    profile_identity: str | None

    def __init__(
        self,
        *,
        required_bytes: int,
        free_bytes: int,
        reserve_bytes: int,
        training_peak_bytes: int | None = None,
        metadata: TrainingMemoryRefusalMetadata | None = None,
    ) -> None:
        """Store the exact byte counts that explain a memory refusal."""
        self.required_bytes = required_bytes
        self.free_bytes = free_bytes
        self.reserve_bytes = reserve_bytes
        self.training_peak_bytes = training_peak_bytes
        details = metadata or TrainingMemoryRefusalMetadata()
        self.reason = details.reason
        self.observed_at = details.observed_at
        self.profile_identity = details.profile_identity
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
    _dataset_signature_probe: Callable[[Path], DatasetManifest | None]
    _repository: TrainingRepository
    _manifest: DatasetManifest | None
    _manifest_lock: asyncio.Lock
    _dataset_state: Literal["not_started", "validating", "ready", "unavailable"]
    _dataset_reason: str | None
    _dataset_retry_after: float
    _dataset_warm_task: asyncio.Task[None] | None

    def __init__(
        self,
        database: Database,
        settings: TrainingSettings,
        *,
        dataset_validator: Callable[[Path], DatasetManifest] = validate_cuhk,
        dataset_signature_probe: Callable[[Path], DatasetManifest | None] = validate_cached_cuhk,
        repository: TrainingRepository | None = None,
    ) -> None:
        """Build an API service without importing CUDA or reading run directories."""
        self._database = database
        self._settings = settings
        self._dataset_validator = dataset_validator
        self._dataset_signature_probe = dataset_signature_probe
        self._repository = repository or TrainingRepository()
        self._manifest = None
        self._manifest_lock = asyncio.Lock()
        self._dataset_state = "not_started"
        self._dataset_reason = None
        self._dataset_retry_after = 0.0
        self._dataset_warm_task = None

    async def warm_dataset(self) -> None:
        """Warm one full dataset validation outside API request and request-lease paths."""
        current = asyncio.current_task()
        existing = self._dataset_warm_task
        if existing is not None and existing is not current:
            await existing
            return
        if self._dataset_state == "ready" and self._manifest is not None:
            return
        if self._settings.dataset_root is None:
            self._dataset_state = "unavailable"
            self._dataset_reason = "dataset_not_configured"
            return
        if existing is None and current is not None:
            self._dataset_warm_task = current
        self._dataset_state = "validating"
        self._dataset_reason = "dataset_validating"
        try:
            async with self._manifest_lock:
                root = self._settings.require_dataset_root()
                manifest = await asyncio.to_thread(self._dataset_validator, root)
            self._manifest = manifest
            self._dataset_state = "ready"
            self._dataset_reason = None
            self._dataset_retry_after = 0.0
        except (TrainingDatasetNotConfiguredError, DatasetValidationError, OSError):
            self._manifest = None
            self._dataset_state = "unavailable"
            self._dataset_reason = "dataset_invalid_or_unavailable"
            self._dataset_retry_after = asyncio.get_running_loop().time() + 30.0
        finally:
            if self._dataset_warm_task is current:
                self._dataset_warm_task = None

    async def datasets(self) -> TrainingDatasetStatus:
        """Return only public dataset identity/counts, never sample paths or captions."""
        if self._settings.dataset_root is None:
            self._dataset_state = "unavailable"
            self._dataset_reason = "dataset_not_configured"
            manifest = None
        elif self._dataset_state == "not_started":
            self._schedule_dataset_warmup()
            manifest = None
        elif self._dataset_state == "validating":
            manifest = None
        elif self._dataset_state == "unavailable":
            if asyncio.get_running_loop().time() >= self._dataset_retry_after:
                self._schedule_dataset_warmup()
            manifest = None
        else:
            try:
                manifest = await self._registered_manifest()
            except TrainingDatasetUnavailableError:
                manifest = None

        valid = manifest is not None
        snapshot = (
            TrainingDatasetSnapshot.model_validate(manifest.public_snapshot())
            if manifest is not None
            else None
        )
        return TrainingDatasetStatus(
            registered=self._settings.dataset_root is not None,
            valid=valid,
            reason=None if valid else self._dataset_reason or "dataset_validating",
            snapshot=snapshot,
            supervisor=self._supervisor_status(
                manifest.fingerprint if manifest is not None else None,
            ),
        )

    async def config(self) -> TrainingConfig:
        """Return the pinned server config defaults/ranges without GPU imports."""
        return TrainingConfig()

    async def preflight(self, config: TrainingConfig) -> TrainingPreflightResponse:
        """Commit an idempotent, short-lived supervisor request before waiting."""
        manifest = await self._registered_manifest()
        self._require_supervisor_ready(manifest)
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
        self._require_supervisor_ready(manifest)
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
        self._require_supervisor_ready(manifest)
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

    def _dataset_validation_in_progress(self) -> bool:
        """Read validation state behind a method boundary for each lock check."""
        return self._dataset_state == "validating"

    async def _registered_manifest(self) -> DatasetManifest:
        # Every decision rechecks all source signatures. A cache miss starts one
        # background content/hash validation so request lifetime never owns it.
        if self._dataset_validation_in_progress():
            raise TrainingDatasetUnavailableError("registered dataset validation is in progress")
        async with self._manifest_lock:
            if self._dataset_validation_in_progress():
                raise TrainingDatasetUnavailableError(
                    "registered dataset validation is in progress"
                )
            if self._settings.dataset_root is None:
                self._dataset_state = "unavailable"
                self._dataset_reason = "dataset_not_configured"
                raise TrainingDatasetUnavailableError
            if self._dataset_state != "ready" or self._manifest is None:
                self._manifest = None
                self._schedule_dataset_warmup()
                raise TrainingDatasetUnavailableError(
                    "registered dataset validation is in progress"
                )
            try:
                root = self._settings.require_dataset_root()
                manifest = await asyncio.to_thread(self._dataset_signature_probe, root)
            except (TrainingDatasetNotConfiguredError, DatasetValidationError, OSError) as error:
                self._manifest = None
                self._dataset_state = "validating"
                self._dataset_reason = "dataset_validating"
                self._schedule_dataset_warmup()
                raise TrainingDatasetUnavailableError from error
            if manifest is None:
                self._manifest = None
                self._dataset_state = "validating"
                self._dataset_reason = "dataset_validating"
                self._schedule_dataset_warmup()
                raise TrainingDatasetUnavailableError(
                    "registered dataset validation is in progress"
                )
            self._manifest = manifest
            self._dataset_state = "ready"
            self._dataset_reason = None
            self._dataset_retry_after = 0.0
            return manifest

    def _schedule_dataset_warmup(self) -> None:
        if self._settings.dataset_root is None:
            self._dataset_state = "unavailable"
            self._dataset_reason = "dataset_not_configured"
            return
        if self._dataset_state == "validating":
            if self._dataset_warm_task is None:
                self._dataset_state = "not_started"
            else:
                return
        self._dataset_state = "validating"
        self._dataset_reason = "dataset_validating"
        self._dataset_warm_task = asyncio.create_task(self.warm_dataset())

    def _supervisor_status(self, dataset_fingerprint: str | None) -> TrainingSupervisorStatus:
        return read_supervisor_status(
            self._settings.training_root,
            source_fingerprint=self._repository.source_fingerprint,
            dataset_fingerprint=dataset_fingerprint,
        )

    def _require_supervisor_ready(self, manifest: DatasetManifest) -> None:
        status = self._supervisor_status(manifest.fingerprint)
        if status.state != "ready":
            raise TrainingSupervisorUnavailableError

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
        response_reason = result.get("reason")
        reason = response_reason if isinstance(response_reason, str) else error_code
        raise TrainingMemoryRefusedError(
            required_bytes=_non_negative_response_int(result.get("required_bytes")),
            free_bytes=_non_negative_response_int(result.get("free_bytes")),
            reserve_bytes=_non_negative_response_int(result.get("reserve_bytes")),
            training_peak_bytes=_non_negative_response_int(result.get("training_peak_bytes")),
            metadata=TrainingMemoryRefusalMetadata(
                reason=reason or "memory_profile_unsupported",
                observed_at=_parse_response_datetime(result.get("observed_at")),
                profile_identity=_parse_profile_identity(result.get("profile_identity")),
            ),
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
        evaluation=(
            TrainingEvaluationSummary.model_validate(job.evaluation_report)
            if job.evaluation_report is not None
            else None
        ),
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
        finished_at=job.finished_at,
    )


def _non_negative_response_int(value: object) -> int:
    return value if type(value) is int and value >= 0 else 0


def _parse_response_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None and parsed.utcoffset() is not None else None


def _parse_profile_identity(value: object) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) else None


__all__ = [
    "TrainingCursorError",
    "TrainingDatasetUnavailableError",
    "TrainingHistoryUnavailableError",
    "TrainingJobConflictError",
    "TrainingJobNotFoundError",
    "TrainingJobStateError",
    "TrainingMemoryRefusalMetadata",
    "TrainingMemoryRefusedError",
    "TrainingRequestConflictError",
    "TrainingService",
    "TrainingServiceError",
    "TrainingSupervisorUnavailableError",
]
