"""Authenticated CUHK-PEDES training settings, jobs, and bounded telemetry routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, NoReturn, Protocol
from uuid import UUID  # noqa: TC003

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from gods_watching.contracts.training import (
    TrainingActionRequest,
    TrainingConfig,
    TrainingDatasetStatus,
    TrainingJobPage,
    TrainingJobResponse,
    TrainingJobSubmitRequest,
    TrainingLogPage,
    TrainingMetricPage,
    TrainingPreflightRequest,
    TrainingPreflightResponse,
)
from gods_watching.training.service import (
    TrainingCursorError,
    TrainingDatasetUnavailableError,
    TrainingJobConflictError,
    TrainingJobNotFoundError,
    TrainingJobStateError,
    TrainingMemoryRefusedError,
    TrainingRequestConflictError,
    TrainingSupervisorUnavailableError,
)

if TYPE_CHECKING:
    from .camera_routes import SessionDependency, SessionDependencyFactory

MAX_PAGE_SIZE = 200
DEFAULT_PAGE_SIZE = 50
MAX_CURSOR_LENGTH = 160


class TrainingServiceProvider(Protocol):
    """CPU-only service boundary consumed by authenticated route handlers."""

    async def datasets(self) -> TrainingDatasetStatus:
        """Return the registered dataset's validated public snapshot."""
        ...

    async def config(self) -> TrainingConfig:
        """Return immutable server-side config defaults and bounds."""
        ...

    async def preflight(self, config: TrainingConfig) -> TrainingPreflightResponse:
        """Ask the supervisor for a current memory estimate without creating a job."""
        ...

    async def submit(self, request: TrainingJobSubmitRequest) -> TrainingJobResponse:
        """Persist and await one idempotent supervisor admission request."""
        ...

    async def list_jobs(
        self,
        *,
        cursor: str | None,
        limit: int,
    ) -> TrainingJobPage:
        """Return a bounded durable history page."""
        ...

    async def get_job(self, job_id: UUID) -> TrainingJobResponse:
        """Return one durable job without paths or process identity."""
        ...

    async def cancel(self, job_id: UUID, request_id: UUID) -> TrainingJobResponse:
        """Request cooperative cancellation for one active job."""
        ...

    async def resume(self, job_id: UUID, request_id: UUID) -> TrainingJobResponse:
        """Request a fenced resume from an existing complete checkpoint."""
        ...

    async def metrics(
        self,
        job_id: UUID,
        *,
        cursor: str | None,
        limit: int,
    ) -> TrainingMetricPage:
        """Return a bounded metric history page."""
        ...

    async def logs(
        self,
        job_id: UUID,
        *,
        cursor: str | None,
        limit: int,
    ) -> TrainingLogPage:
        """Return bounded, redacted log entries."""
        ...


@dataclass(frozen=True, slots=True)
class TrainingRouterDependencies:
    """Compose the CPU service with the shared database-independent auth guard."""

    service: TrainingServiceProvider
    require_session: SessionDependencyFactory


class _TrainingHandlers:
    _service: TrainingServiceProvider

    def __init__(self, dependencies: TrainingRouterDependencies) -> None:
        self._service = dependencies.service
        self._passive: SessionDependency = dependencies.require_session(user_action=False)
        self._mutation: SessionDependency = dependencies.require_session(user_action=True)

    @property
    def passive_dependency(self) -> SessionDependency:
        return self._passive

    @property
    def mutation_dependency(self) -> SessionDependency:
        return self._mutation

    async def datasets(self) -> TrainingDatasetStatus:
        try:
            return await self._service.datasets()
        except TrainingDatasetUnavailableError as error:
            _raise_error(503, error.code, "registered CUHK-PEDES dataset is unavailable")

    async def config(self) -> TrainingConfig:
        return await self._service.config()

    async def preflight(self, request: TrainingPreflightRequest) -> TrainingPreflightResponse:
        try:
            return await self._service.preflight(request.config)
        except TrainingMemoryRefusedError as error:
            _raise_memory_refusal(error)
        except TrainingSupervisorUnavailableError as error:
            _raise_error(503, error.code, "training supervisor is unavailable")
        except TrainingDatasetUnavailableError as error:
            _raise_error(503, error.code, "registered CUHK-PEDES dataset is unavailable")

    async def submit(
        self,
        request: TrainingJobSubmitRequest,
        response: Response,
    ) -> TrainingJobResponse:
        try:
            result = await self._service.submit(request)
        except TrainingMemoryRefusedError as error:
            _raise_memory_refusal(error)
        except TrainingSupervisorUnavailableError as error:
            _raise_error(503, error.code, "training supervisor did not accept the request")
        except TrainingRequestConflictError as error:
            _raise_error(409, error.code, error.message)
        except TrainingJobConflictError as error:
            _raise_error(409, error.code, "another training job owns the GPU slot")
        except TrainingDatasetUnavailableError as error:
            _raise_error(503, error.code, "registered CUHK-PEDES dataset is unavailable")
        response.status_code = status.HTTP_202_ACCEPTED
        return result

    async def list_jobs(
        self,
        cursor: Annotated[str | None, Query(max_length=MAX_CURSOR_LENGTH)] = None,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    ) -> TrainingJobPage:
        try:
            return await self._service.list_jobs(cursor=cursor, limit=limit)
        except TrainingCursorError as error:
            _raise_error(422, error.code, "training history cursor is invalid")

    async def get_job(self, job_id: UUID) -> TrainingJobResponse:
        try:
            return await self._service.get_job(job_id)
        except TrainingJobNotFoundError as error:
            _raise_error(404, error.code, "training job was not found")

    async def cancel(
        self,
        job_id: UUID,
        request: TrainingActionRequest,
        response: Response,
    ) -> TrainingJobResponse:
        try:
            result = await self._service.cancel(job_id, request.request_id)
        except TrainingJobNotFoundError as error:
            _raise_error(404, error.code, "training job was not found")
        except TrainingJobStateError as error:
            _raise_error(409, error.code, "training job cannot be cancelled in its current phase")
        except TrainingSupervisorUnavailableError as error:
            _raise_error(503, error.code, "training supervisor is unavailable")
        response.status_code = status.HTTP_202_ACCEPTED
        return result

    async def resume(
        self,
        job_id: UUID,
        request: TrainingActionRequest,
        response: Response,
    ) -> TrainingJobResponse:
        try:
            result = await self._service.resume(job_id, request.request_id)
        except TrainingJobNotFoundError as error:
            _raise_error(404, error.code, "training job was not found")
        except TrainingJobConflictError as error:
            _raise_error(409, error.code, "another training child still owns the GPU slot")
        except TrainingJobStateError as error:
            _raise_error(409, error.code, "training job cannot be resumed in its current phase")
        except TrainingMemoryRefusedError as error:
            _raise_memory_refusal(error)
        except TrainingRequestConflictError as error:
            _raise_error(409, error.code, error.message)
        except TrainingDatasetUnavailableError as error:
            _raise_error(409, error.code, "registered dataset no longer matches the job")
        except TrainingSupervisorUnavailableError as error:
            _raise_error(503, error.code, "training supervisor is unavailable")
        response.status_code = status.HTTP_202_ACCEPTED
        return result

    async def metrics(
        self,
        job_id: UUID,
        cursor: Annotated[str | None, Query(max_length=MAX_CURSOR_LENGTH)] = None,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    ) -> TrainingMetricPage:
        try:
            return await self._service.metrics(job_id, cursor=cursor, limit=limit)
        except TrainingJobNotFoundError as error:
            _raise_error(404, error.code, "training job was not found")

    async def logs(
        self,
        job_id: UUID,
        cursor: Annotated[str | None, Query(max_length=MAX_CURSOR_LENGTH)] = None,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    ) -> TrainingLogPage:
        try:
            return await self._service.logs(job_id, cursor=cursor, limit=limit)
        except TrainingJobNotFoundError as error:
            _raise_error(404, error.code, "training job was not found")


def build_training_router(
    *,
    service: TrainingServiceProvider,
    require_session: SessionDependencyFactory,
) -> APIRouter:
    """Build all authenticated training routes with the existing cookie/origin guard."""
    handlers = _TrainingHandlers(TrainingRouterDependencies(service, require_session))
    router = APIRouter(prefix="/api/training", tags=["training"])
    router.add_api_route(
        "/datasets",
        handlers.datasets,
        methods=["GET"],
        response_model=TrainingDatasetStatus,
        dependencies=[Depends(handlers.passive_dependency)],
    )
    router.add_api_route(
        "/config",
        handlers.config,
        methods=["GET"],
        response_model=TrainingConfig,
        dependencies=[Depends(handlers.passive_dependency)],
    )
    router.add_api_route(
        "/preflight",
        handlers.preflight,
        methods=["POST"],
        response_model=TrainingPreflightResponse,
        dependencies=[Depends(handlers.mutation_dependency)],
    )
    router.add_api_route(
        "/jobs",
        handlers.list_jobs,
        methods=["GET"],
        response_model=TrainingJobPage,
        dependencies=[Depends(handlers.passive_dependency)],
    )
    router.add_api_route(
        "/jobs",
        handlers.submit,
        methods=["POST"],
        status_code=status.HTTP_202_ACCEPTED,
        response_model=TrainingJobResponse,
        dependencies=[Depends(handlers.mutation_dependency)],
    )
    router.add_api_route(
        "/jobs/{job_id}",
        handlers.get_job,
        methods=["GET"],
        response_model=TrainingJobResponse,
        dependencies=[Depends(handlers.passive_dependency)],
    )
    router.add_api_route(
        "/jobs/{job_id}/cancel",
        handlers.cancel,
        methods=["POST"],
        status_code=status.HTTP_202_ACCEPTED,
        response_model=TrainingJobResponse,
        dependencies=[Depends(handlers.mutation_dependency)],
    )
    router.add_api_route(
        "/jobs/{job_id}/resume",
        handlers.resume,
        methods=["POST"],
        status_code=status.HTTP_202_ACCEPTED,
        response_model=TrainingJobResponse,
        dependencies=[Depends(handlers.mutation_dependency)],
    )
    router.add_api_route(
        "/jobs/{job_id}/metrics",
        handlers.metrics,
        methods=["GET"],
        response_model=TrainingMetricPage,
        dependencies=[Depends(handlers.passive_dependency)],
    )
    router.add_api_route(
        "/jobs/{job_id}/logs",
        handlers.logs,
        methods=["GET"],
        response_model=TrainingLogPage,
        dependencies=[Depends(handlers.passive_dependency)],
    )
    return router


def _raise_memory_refusal(error: TrainingMemoryRefusedError) -> NoReturn:
    raise HTTPException(
        status_code=409,
        detail={
            "code": error.code,
            "message": "training would exceed currently available GPU memory",
            "reason": error.reason,
            "training_peak_bytes": error.training_peak_bytes,
            "reserve_bytes": error.reserve_bytes,
            "required_bytes": error.required_bytes,
            "free_bytes": error.free_bytes,
        },
    )


def _raise_error(status_code: int, code: str, message: str) -> NoReturn:
    raise HTTPException(status_code=status_code, detail={"code": code, "message": message})


__all__ = ["TrainingServiceProvider", "build_training_router"]
