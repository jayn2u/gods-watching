"""Authenticated durable retention, quota, and wall-slot settings routes."""

from dataclasses import dataclass
from typing import NoReturn, Protocol, final

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gods_watching.contracts.settings import SettingsPatchRequest, SettingsResponse
from gods_watching.settings.service import (
    SettingsNoChangesError,
    SettingsStorageError,
)

from .camera_routes import SessionDependency, SessionDependencyFactory, TransactionProvider


class SettingsServiceProvider(Protocol):
    """Provide the durable settings operations consumed by route composition."""

    async def get(self, session: AsyncSession) -> SettingsResponse:
        """Read the singleton settings row."""
        ...

    async def update(
        self,
        session: AsyncSession,
        patch: SettingsPatchRequest,
    ) -> SettingsResponse:
        """Persist one validated settings patch."""
        ...


@dataclass(frozen=True, slots=True)
class SettingsRouterDependencies:
    """Compose settings persistence with the shared auth guard."""

    database: TransactionProvider
    settings: SettingsServiceProvider
    require_session: SessionDependencyFactory


@final
class _SettingsHandlers:
    def __init__(self, dependencies: SettingsRouterDependencies) -> None:
        self._database: TransactionProvider = dependencies.database
        self._settings: SettingsServiceProvider = dependencies.settings
        self._passive = dependencies.require_session(user_action=False)
        self._mutation = dependencies.require_session(user_action=True)

    @property
    def passive_dependency(self) -> SessionDependency:
        return self._passive

    @property
    def mutation_dependency(self) -> SessionDependency:
        return self._mutation

    async def get_settings(self) -> SettingsResponse:
        async with self._database.transaction() as session:
            return await self._settings.get(session)

    async def patch_settings(self, patch: SettingsPatchRequest) -> SettingsResponse:
        try:
            async with self._database.transaction() as session:
                return await self._settings.update(session, patch)
        except SettingsNoChangesError as error:
            _raise_settings_error(422, "empty_settings_patch", str(error), error)
        except SettingsStorageError as error:
            _raise_settings_error(422, "invalid_settings", str(error), error)
        except IntegrityError as error:
            _raise_settings_error(
                409,
                "settings_conflict",
                "settings conflict with stored state",
                error,
            )


def build_settings_router(
    *,
    database: TransactionProvider,
    settings: SettingsServiceProvider,
    require_session: SessionDependencyFactory,
) -> APIRouter:
    """Build settings routes without a public or unauthenticated fallback."""
    handlers = _SettingsHandlers(
        SettingsRouterDependencies(
            database=database,
            settings=settings,
            require_session=require_session,
        )
    )
    router = APIRouter(prefix="/api/settings", tags=["settings"])

    router.add_api_route(
        "",
        handlers.get_settings,
        methods=["GET"],
        dependencies=[Depends(handlers.passive_dependency)],
    )
    router.add_api_route(
        "",
        handlers.patch_settings,
        methods=["PATCH"],
        dependencies=[Depends(handlers.mutation_dependency)],
    )

    return router


def _raise_settings_error(
    status_code: int,
    code: str,
    message: str,
    cause: BaseException | None = None,
) -> NoReturn:
    """Raise a stable settings error without exposing database details."""
    del cause
    raise HTTPException(status_code=status_code, detail={"code": code, "message": message})


__all__ = ["SettingsRouterDependencies", "SettingsServiceProvider", "build_settings_router"]
