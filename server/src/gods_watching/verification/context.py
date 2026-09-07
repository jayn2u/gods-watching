"""Owned resources available to verification scenario runners."""

import json
import secrets
import signal
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
_INTERRUPTION_SIGNALS: Final = (signal.SIGINT, signal.SIGTERM)


class VerificationInterruptedError(Exception):
    """Carry a cooperative verification interruption to the CLI boundary."""

    signum: int

    def __init__(self, signum: int) -> None:
        """Record the originating operating-system signal number."""
        super().__init__(signum)
        self.signum = signum


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
        self._events.append({"kind": kind, "name": name, "state": state, "detail": redact(detail)})
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
        self._signal_scope_active = False
        self._signal_signum: int | None = None

    @asynccontextmanager
    async def process(self, *, name: str, command: Sequence[str]) -> AsyncIterator[Process]:
        """Register, start, and terminate one task-owned subprocess."""
        if self._signal_scope_active:
            async with self._owned_process(name=name, command=command) as process:
                yield process
            return

        self._signal_scope_active = True
        self._signal_signum = None
        previous_handlers = {signum: signal.getsignal(signum) for signum in _INTERRUPTION_SIGNALS}
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        try:
            try:
                with anyio.CancelScope() as signal_scope:
                    with anyio.open_signal_receiver(*_INTERRUPTION_SIGNALS) as signals:
                        async with anyio.create_task_group() as signal_tasks:
                            process_started = anyio.Event()
                            signal_tasks.start_soon(
                                self._observe_signals, signals, signal_scope, process_started
                            )
                            try:
                                async with self._owned_process(
                                    name=name, command=command
                                ) as process:
                                    process_started.set()
                                    yield process
                            finally:
                                signal_tasks.cancel_scope.cancel()
            except BaseExceptionGroup:
                if self._signal_signum is None:
                    raise
            interrupted_signum = self._signal_signum
            if interrupted_signum is not None:
                raise VerificationInterruptedError(interrupted_signum)
        finally:
            if self._signal_signum is None:
                for signum, handler in previous_handlers.items():
                    _ = signal.signal(signum, handler)
                _ = signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            else:
                for signum in _INTERRUPTION_SIGNALS:
                    _ = signal.signal(signum, signal.SIG_IGN)
            self._signal_scope_active = False
            self._signal_signum = None

    async def _observe_signals(
        self,
        signals: AsyncIterator[signal.Signals],
        signal_scope: anyio.CancelScope,
        process_started: anyio.Event,
    ) -> None:
        async for signum in signals:
            if self._signal_signum is None:
                self._signal_signum = int(signum)
                _ = signal.pthread_sigmask(signal.SIG_BLOCK, set(_INTERRUPTION_SIGNALS))
                for interruption_signum in _INTERRUPTION_SIGNALS:
                    _ = signal.signal(interruption_signum, signal.SIG_IGN)
            await process_started.wait()
            signal_scope.cancel()
            return

    @asynccontextmanager
    async def _owned_process(self, *, name: str, command: Sequence[str]) -> AsyncIterator[Process]:
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set(_INTERRUPTION_SIGNALS))
        safe_command = " ".join(redact(part) for part in command)
        process: Process | None = None
        mask_restored = False
        try:
            self._ledger.append(kind="process", name=name, state="registered", detail=safe_command)
            try:
                process = await anyio.open_process(command, stdout=PIPE, stderr=PIPE)
                self._ledger.append(
                    kind="process", name=name, state="started", detail=f"pid={process.pid}"
                )
            except OSError:
                self._ledger.append(
                    kind="process", name=name, state="cleaned", detail="spawn failed"
                )
                raise
            finally:
                _ = signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                mask_restored = True
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
        finally:
            if not mask_restored:
                _ = signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


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
