"""Async SQLAlchemy engine and transaction factories."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


@dataclass(frozen=True, slots=True)
class Database:
    """Own the asynchronous PostgreSQL engine and transaction lifecycle."""

    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]

    @classmethod
    def connect(cls, database_url: str) -> "Database":
        """Create a pooled async PostgreSQL database boundary."""
        engine = create_async_engine(database_url, pool_pre_ping=True)
        return cls(
            engine=engine,
            session_factory=async_sessionmaker(engine, expire_on_commit=False),
        )

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[AsyncSession]:
        """Commit on success and roll back every exception."""
        async with self.session_factory.begin() as session:
            yield session

    async def close(self) -> None:
        """Dispose all pooled database connections."""
        await self.engine.dispose()
