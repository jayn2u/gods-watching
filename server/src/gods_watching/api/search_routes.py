"""Authenticated search HTTP routes."""

from typing import NoReturn, Protocol, final

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from gods_watching.contracts.search import SearchRequest, SearchResponse
from gods_watching.model_selection.models import MaintenanceModeError
from gods_watching.search import (
    SearchInferenceUnavailableError,
    SearchSeedNotFoundError,
    SearchTextInvalidError,
    UnknownCameraError,
)

from .camera_routes import SessionDependency, SessionDependencyFactory, TransactionProvider


class SearchServiceProvider(Protocol):
    """Provide the verified retrieval service consumed by the HTTP boundary."""

    async def search(self, session: AsyncSession, request: SearchRequest) -> SearchResponse:
        """Execute one typed browse, similar, or text search."""
        ...


@final
class _SearchHandlers:
    """Keep database and activity boundaries explicit for search requests."""

    def __init__(
        self,
        database: TransactionProvider,
        search: SearchServiceProvider,
        require_session: SessionDependencyFactory,
    ) -> None:
        self._database = database
        self._search = search
        self._action = require_session(user_action=True)

    @property
    def action_dependency(self) -> SessionDependency:
        """Return the dependency that authenticates and refreshes user activity."""
        return self._action

    async def search(self, payload: SearchRequest) -> SearchResponse:
        """Run one search inside a caller-owned committed transaction."""
        try:
            async with self._database.transaction() as session:
                return await self._search.search(session, payload)
        except UnknownCameraError:
            _raise_api_error(422, "unknown_camera", "one or more cameras were not found")
        except SearchSeedNotFoundError:
            _raise_api_error(404, "appearance_not_found", "appearance was not found")
        except SearchTextInvalidError:
            _raise_api_error(422, "invalid_text", "search text is invalid")
        except SearchInferenceUnavailableError:
            _raise_api_error(503, "inference_unavailable", "search inference is unavailable")
        except MaintenanceModeError:
            _raise_api_error(503, "maintenance", "search is temporarily unavailable")


def build_search_router(
    *,
    database: TransactionProvider,
    search: SearchServiceProvider,
    require_session: SessionDependencyFactory,
) -> APIRouter:
    """Build the authenticated search route around prepared services."""
    handlers = _SearchHandlers(database, search, require_session)
    router = APIRouter(prefix="/api/search", tags=["search"])
    router.add_api_route(
        "",
        handlers.search,
        methods=["POST"],
        response_model=SearchResponse,
        dependencies=[Depends(handlers.action_dependency)],
    )
    return router


def _raise_api_error(status_code: int, code: str, message: str) -> NoReturn:
    raise HTTPException(status_code=status_code, detail={"code": code, "message": message})


__all__ = ["SearchServiceProvider", "build_search_router"]
