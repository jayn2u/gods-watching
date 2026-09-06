"""Owned resources available to verification scenario runners."""

import json
import secrets
import socket
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from subprocess import PIPE
from typing import Final, final

import anyio
from anyio.abc import Process

from .models import RunId
from .redaction import redact

_CLEANUP_GRACE_SECONDS: Final = 2.0


class PortAllocationError(RuntimeError):
    """Report exhaustion of bounded loopback port allocation attempts."""


@final
class ResourceLedger:
    """Mutable append-only ledger that persists before resource acquisition."""

    def __init__(self, path: Path) -> None:
        """Create the manifest before any owned resource is acquired."""
        self._path = path
        self._events: list[dict[str, str | int | None]] = []
        self._write()

    def append(self, *, kind: str, name: str, state: str, detail: str = "") -> None:
        """Persist a lifecycle transition atomically."""
        self._events.append(
            {"kind": kind, "name": name, "state": state, "detail": redact(detail)}
        )
        self._write()

    def _write(self) -> None:
        temporary = self._path.with_suffix(".tmp")
        _ = temporary.write_text(
            json.dumps({"events": self._events}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _ = temporary.replace(self._path)


@dataclass(frozen=True, slots=True)
class ScenarioContextConfig:
    """Group the immutable namespace allocated to one run."""

    run_id: RunId
    run_root: Path
    runtime_root: Path
    compose_project: str
    allocated_port: int


@final
class ScenarioContext:
    """Expose a unique namespace and ownership-aware process lifecycle."""

    def __init__(
        self,
        config: ScenarioContextConfig,
        ledger: ResourceLedger,
    ) -> None:
        """Attach resource ownership to a previously allocated namespace."""
        self.run_id: RunId = config.run_id
        self.run_root: Path = config.run_root
        self.runtime_root: Path = config.runtime_root
        self.compose_project: str = config.compose_project
        self.allocated_port: int = config.allocated_port
        self._ledger = ledger

    @asynccontextmanager
    async def process(
        self, *, name: str, command: Sequence[str]
    ) -> AsyncIterator[Process]:
        """Register, start, and terminate one task-owned subprocess."""
        safe_command = " ".join(redact(part) for part in command)
        self._ledger.append(kind="process", name=name, state="registered", detail=safe_command)
        try:
            process = await anyio.open_process(command, stdout=PIPE, stderr=PIPE)
        except OSError:
            self._ledger.append(kind="process", name=name, state="cleaned", detail="spawn failed")
            raise
        self._ledger.append(
            kind="process", name=name, state="started", detail=f"pid={process.pid}"
        )
        try:
            yield process
        finally:
            with anyio.CancelScope(shield=True):
                if process.returncode is None:
                    process.terminate()
                    with anyio.move_on_after(_CLEANUP_GRACE_SECONDS):
                        _ = await process.wait()
                    if process.returncode is None:
                        process.kill()
                        _ = await process.wait()
                self._ledger.append(
                    kind="process",
                    name=name,
                    state="cleaned",
                    detail=f"returncode={process.returncode}",
                )


def allocate_loopback_port() -> int:
    """Ask the kernel for a currently free loopback TCP port."""
    for _attempt in range(100):
        port = 20_000 + secrets.randbelow(40_000)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.bind(("127.0.0.1", port))
        except OSError:
            continue
        return port
    raise PortAllocationError
