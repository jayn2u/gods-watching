"""Owned resources available to verification scenario runners."""

import json
import secrets
import signal
import socket
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from subprocess import PIPE
from typing import Final, final

import anyio
from anyio.abc import Process
from anyio.to_thread import run_sync

from .models import RunId
from .redaction import redact

_CLEANUP_GRACE_SECONDS: Final = 2.0
_INTERRUPTED_CLEANUP_GRACE_SECONDS: Final = 0.25
_INTERRUPTION_SIGNALS: Final = (signal.SIGINT, signal.SIGTERM)
_INTERRUPT_CLEANUP_SECONDS: Final = 3.0


def _no_interrupt() -> int | None:
    return None


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
    interrupt_reader: Callable[[], int | None] | None = None


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
        self._interrupt_reader: Callable[[], int | None] = config.interrupt_reader or _no_interrupt
        self._ledger = ledger
        self._signal_scope_active = False
        self._signal_signum: int | None = None
        self._interruptions_blocked = False
        self._interrupt_cleanups: list[Callable[[], Awaitable[None]]] = []

    @asynccontextmanager
    async def process(self, *, name: str, command: Sequence[str]) -> AsyncIterator[Process]:
        """Register, start, and terminate one task-owned subprocess."""
        signum = self._interrupt_reader()
        if signum is not None and not self._interruptions_blocked:
            self._signal_signum = signum
            self._ledger.append(
                kind="signal",
                name="verification",
                state="received",
                detail=f"signum={signum}",
            )
            raise VerificationInterruptedError(signum)
        if threading.current_thread() is not threading.main_thread():
            async with self._owned_process(name=name, command=command) as process:
                yield process
            return
        if self._signal_scope_active or self._interruptions_blocked:
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
                    signal_event = threading.Event()
                    observer_stopping = threading.Event()
                    self._install_signal_handler(signal_event)
                    async with anyio.create_task_group() as signal_tasks:
                        process_started = anyio.Event()
                        signal_tasks.start_soon(
                            self._observe_signals,
                            signal_event,
                            observer_stopping,
                            signal_scope,
                            process_started,
                        )
                        try:
                            async with self._owned_process(name=name, command=command) as process:
                                process_started.set()
                                yield process
                        finally:
                            observer_stopping.set()
                            signal_event.set()
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

    def _install_signal_handler(self, signal_event: threading.Event) -> None:
        def receive_signal(signum: int, _frame: object) -> None:
            if self._signal_signum is None:
                self._signal_signum = signum
                _ = signal.pthread_sigmask(signal.SIG_BLOCK, set(_INTERRUPTION_SIGNALS))
                for interruption_signum in _INTERRUPTION_SIGNALS:
                    _ = signal.signal(interruption_signum, signal.SIG_IGN)
            signal_event.set()

        for interruption_signum in _INTERRUPTION_SIGNALS:
            _ = signal.signal(interruption_signum, receive_signal)

    @contextmanager
    def suppress_interruptions(self) -> Iterator[None]:
        """Block terminal signals until owned cleanup commands finish."""
        previous_handlers = {signum: signal.getsignal(signum) for signum in _INTERRUPTION_SIGNALS}
        self._interruptions_blocked = True
        for signum in _INTERRUPTION_SIGNALS:
            _ = signal.signal(signum, signal.SIG_IGN)
        try:
            yield
        finally:
            self._interruptions_blocked = False
            for signum, handler in previous_handlers.items():
                _ = signal.signal(signum, handler)

    @contextmanager
    def interrupt_cleanup(  # noqa: D102
        self, operation: Callable[[], Awaitable[None]]
    ) -> Iterator[None]:
        self._interrupt_cleanups.append(operation)
        try:
            yield
        finally:
            self._interrupt_cleanups.remove(operation)

    async def _run_interrupt_cleanups(self) -> None:
        operations = tuple(self._interrupt_cleanups)
        with anyio.move_on_after(_INTERRUPT_CLEANUP_SECONDS, shield=True):
            for operation in operations:
                try:
                    await operation()
                except (OSError, RuntimeError, TimeoutError) as error:
                    _ = error
                    continue

    async def _observe_signals(
        self,
        signal_event: threading.Event,
        observer_stopping: threading.Event,
        signal_scope: anyio.CancelScope,
        process_started: anyio.Event,
    ) -> None:
        _ = await run_sync(signal_event.wait)
        if observer_stopping.is_set() and self._signal_signum is None:
            return
        self._ledger.append(
            kind="signal",
            name="verification",
            state="received",
            detail=f"signum={self._signal_signum}",
        )
        await process_started.wait()
        await self._run_interrupt_cleanups()
        signal_scope.cancel()

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
                        grace_seconds = (
                            _INTERRUPTED_CLEANUP_GRACE_SECONDS
                            if self._signal_signum is not None
                            else _CLEANUP_GRACE_SECONDS
                        )
                        with anyio.move_on_after(grace_seconds):
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
