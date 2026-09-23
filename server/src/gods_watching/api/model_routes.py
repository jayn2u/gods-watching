"""Authenticated catalog and durable CLIP model switch routes."""

from dataclasses import dataclass
from typing import NoReturn, Protocol, final

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gods_watching.contracts.model_selection import (
    ModelApplyRequest,
    ModelSettingsResponse,
)
from gods_watching.model_selection.models import (
    ModelNotPreparedError,
    ModelSelectionConflictError,
    TransitionError,
)

from .camera_routes import SessionDependency, SessionDependencyFactory, TransactionProvider


class ModelSelectionProvider(Protocol):
    """Provide model catalog/status and durable queue operations."""

    async def get(self, session: AsyncSession) -> ModelSettingsResponse:
        """Return the catalog and latest transition."""
        ...

    async def apply(self, session: AsyncSession, model_id: str) -> ModelSettingsResponse:
        """Validate and queue a selected model."""
        ...


@dataclass(frozen=True, slots=True)
class ModelRouterDependencies:
    """Compose the model service with database and auth boundaries."""

    database: TransactionProvider
    model_selection: ModelSelectionProvider | None
    require_session: SessionDependencyFactory


@final
class _ModelHandlers:
    def __init__(self, dependencies: ModelRouterDependencies) -> None:
        self._database = dependencies.database
        self._model_selection = dependencies.model_selection
        self._passive = dependencies.require_session(user_action=False)
        self._mutation = dependencies.require_session(user_action=True)

    @property
    def passive_dependency(self) -> SessionDependency:
        return self._passive

    @property
    def mutation_dependency(self) -> SessionDependency:
        return self._mutation

    async def get_models(self) -> ModelSettingsResponse:
        if self._model_selection is None:
            _raise_model_error(503, "model_selection_unavailable", "model selection is unavailable")
        async with self._database.transaction() as session:
            return await self._model_selection.get(session)

    async def apply_model(
        self,
        payload: ModelApplyRequest,
        response: Response,
    ) -> ModelSettingsResponse:
        if self._model_selection is None:
            _raise_model_error(503, "model_selection_unavailable", "model selection is unavailable")
        try:
            async with self._database.transaction() as session:
                result = await self._model_selection.apply(session, payload.model_id)
        except ModelSelectionConflictError as error:
            _raise_model_error(409, error.code, "another model transition is already active")
        except IntegrityError:
            _raise_model_error(
                409,
                "model_transition_conflict",
                "another model transition is already active",
            )
        except ModelNotPreparedError as error:
            _raise_model_error(422, error.code, "selected model is unavailable locally")
        except TransitionError as error:
            _raise_model_error(422, error.code, "model selection request is invalid")
        response.status_code = status.HTTP_202_ACCEPTED
        return result


def build_model_router(
    *,
    database: TransactionProvider,
    model_selection: ModelSelectionProvider | None,
    require_session: SessionDependencyFactory,
) -> APIRouter:
    """Build the authenticated settings/model routes."""
    handlers = _ModelHandlers(
        ModelRouterDependencies(
            database=database,
            model_selection=model_selection,
            require_session=require_session,
        )
    )
    router = APIRouter(prefix="/api/settings/models", tags=["settings"])
    router.add_api_route(
        "",
        handlers.get_models,
        methods=["GET"],
        response_model=ModelSettingsResponse,
        dependencies=[Depends(handlers.passive_dependency)],
    )
    router.add_api_route(
        "/apply",
        handlers.apply_model,
        methods=["POST"],
        status_code=status.HTTP_202_ACCEPTED,
        response_model=ModelSettingsResponse,
        dependencies=[Depends(handlers.mutation_dependency)],
    )
    return router


def _raise_model_error(status_code: int, code: str, message: str) -> NoReturn:
    raise HTTPException(status_code=status_code, detail={"code": code, "message": message})


__all__ = ["ModelSelectionProvider", "build_model_router"]
