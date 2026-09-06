from __future__ import annotations

import os
import secrets
import socket
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING, Final
from uuid import uuid4

import anyio
import pytest
from sqlalchemy import literal, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_POSTGRES_IMAGE: Final = "pgvector/pgvector:0.8.1-pg17"


def _free_port() -> int:
    for _ in range(100):
        candidate = 20_000 + secrets.randbelow(30_000)
        try:
            with socket.create_server(("127.0.0.1", candidate)):
                return candidate
        except OSError:
            continue
    pytest.fail("could not reserve a disposable PostgreSQL port")


def _executable(name: str) -> str:
    executable = which(name)
    if executable is None:
        error_message = f"required executable is unavailable: {name}"
        pytest.fail(error_message)
    return executable


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
async def database_url() -> AsyncIterator[str]:
    container_name = f"gw-task4-{uuid4().hex[:12]}"
    postgres_password = uuid4().hex
    host_port = _free_port()
    docker = _executable("docker")
    _ = await anyio.run_process(
        [
            docker,
            "run",
            "--detach",
            "--name",
            container_name,
            "--publish",
            f"127.0.0.1:{host_port}:5432",
            "--env",
            f"POSTGRES_PASSWORD={postgres_password}",
            "--env",
            "POSTGRES_DB=gods_watching_test",
            _POSTGRES_IMAGE,
        ],
        check=True,
    )
    try:
        url = (
            f"postgresql+asyncpg://postgres:{postgres_password}"
            f"@127.0.0.1:{host_port}/gods_watching_test"
        )
        ready = False
        for _ in range(300):
            probe = await anyio.run_process(
                [
                    docker,
                    "exec",
                    container_name,
                    "pg_isready",
                    "-h",
                    "127.0.0.1",
                    "-U",
                    "postgres",
                    "-d",
                    "gods_watching_test",
                ],
                check=False,
            )
            if probe.returncode == 0:
                ready = True
                break
            await anyio.sleep(0.1)
        if not ready:
            pytest.fail("disposable PostgreSQL did not become ready")
        host_ready = False
        readiness_engine = create_async_engine(url, poolclass=NullPool)
        try:
            for _ in range(300):
                try:
                    async with readiness_engine.connect() as connection:
                        host_ready = await connection.scalar(select(literal(1))) == 1
                except (OSError, SQLAlchemyError):
                    await anyio.sleep(0.1)
                    continue
                if host_ready:
                    break
        finally:
            await readiness_engine.dispose()
        if not host_ready:
            pytest.fail("disposable PostgreSQL host connection did not become ready")
        migration_env = os.environ.copy()
        migration_env["GW_DATABASE_URL"] = url
        migration = await anyio.run_process(
            [_executable("uv"), "run", "alembic", "upgrade", "head"],
            check=False,
            cwd=Path(__file__).parents[3],
            env=migration_env,
        )
        if migration.returncode != 0:
            pytest.fail(migration.stderr.decode())
        yield url
    finally:
        _ = await anyio.run_process([docker, "rm", "--force", container_name], check=True)


@pytest.fixture(scope="session")
async def engine(database_url: str) -> AsyncIterator[AsyncEngine]:
    database_engine = create_async_engine(database_url, pool_pre_ping=True)
    try:
        yield database_engine
    finally:
        await database_engine.dispose()


@pytest.fixture
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    connection = await engine.connect()
    transaction = await connection.begin()
    factory = async_sessionmaker(bind=connection, expire_on_commit=False)
    database_session = factory()
    try:
        yield database_session
    finally:
        await database_session.close()
        if transaction.is_active:
            await transaction.rollback()
        await connection.close()
