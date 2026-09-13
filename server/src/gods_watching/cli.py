"""Initial command dispatcher for the research server lifecycle."""

import os
import signal

_INTERRUPTION_SIGNALS = (signal.SIGINT, signal.SIGTERM)
_STARTUP_PREVIOUS_HANDLERS = tuple(
    (signum, signal.getsignal(signum)) for signum in _INTERRUPTION_SIGNALS
)
_startup_signum = [0]


def _capture_startup_signal(signum: int, _frame: object) -> None:
    if _startup_signum[0] == 0:
        _startup_signum[0] = signum


for _signum in _INTERRUPTION_SIGNALS:
    _ = signal.signal(_signum, _capture_startup_signal)

_startup_signal_name = os.environ.pop("_GW_STARTUP_SIGNAL", "")
if _startup_signal_name == "INT":
    _startup_signum[0] = signal.SIGINT
elif _startup_signal_name == "TERM":
    _startup_signum[0] = signal.SIGTERM

from pathlib import Path  # noqa: E402
from typing import Annotated, NoReturn  # noqa: E402

import anyio  # noqa: E402
import typer  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from typer.models import OptionInfo  # noqa: E402

from gods_watching.auth.credentials_command import credentials_app  # noqa: E402
from gods_watching.pipeline_worker.app import run_pipeline_worker  # noqa: E402
from gods_watching.pipeline_worker.settings import PipelineWorkerSettings  # noqa: E402
from gods_watching.verification import (  # noqa: E402
    EvidencePathError,
    RunResult,
    execute_scenario,
)
from gods_watching.verification.context import VerificationInterruptedError  # noqa: E402

app = typer.Typer(
    name="gods-watching",
    help="Operate and verify the Gods Watching research server.",
    no_args_is_help=True,
)
app.add_typer(credentials_app, name="credentials")


def _unimplemented(command: str) -> NoReturn:
    typer.echo(f"{command} is not implemented yet", err=True)
    raise typer.Exit(code=1)


@app.command()
def prepare() -> None:
    """Prepare pinned assets and runtime configuration."""
    _unimplemented("prepare")


@app.command()
def up() -> None:
    """Start the prepared research server."""
    _unimplemented("up")


@app.command()
def down() -> None:
    """Stop this research server instance."""
    _unimplemented("down")


@app.command()
def status() -> None:
    """Inspect the running research server."""
    _unimplemented("status")


@app.command()
def doctor() -> None:
    """Check local runtime prerequisites."""
    _unimplemented("doctor")


@app.command()
def worker() -> None:
    """Run the pipeline worker: RTSP ingest, appearance publication, and retention."""
    try:
        settings = PipelineWorkerSettings()  # pyright: ignore[reportCallIssue]
    except ValidationError as error:
        invalid = sorted({f"GW_{str(item['loc'][0]).upper()}" for item in error.errors()})
        typer.echo(f"worker configuration is invalid: {', '.join(invalid)}", err=True)
        raise typer.Exit(code=2) from error
    startup_signum = _take_startup_signal()
    if startup_signum:
        raise typer.Exit(code=128 + startup_signum)
    _restore_startup_signal_handlers()
    lock_path = Path(__file__).resolve().parents[3] / "assets/models.lock.json"
    anyio.run(_serve_worker, settings, lock_path)


async def _serve_worker(settings: PipelineWorkerSettings, lock_path: Path) -> None:
    stop_event = anyio.Event()
    async with anyio.create_task_group() as task_group:

        async def stop_on_signal() -> None:
            with anyio.open_signal_receiver(*_INTERRUPTION_SIGNALS) as signals:
                async for _signum in signals:
                    stop_event.set()
                    return

        task_group.start_soon(stop_on_signal)
        await run_pipeline_worker(settings, lock_path=lock_path, stop_event=stop_event)
        task_group.cancel_scope.cancel()


def _execute_with_signal_handlers(
    *, scenario: str, evidence: Path, repository_root: Path
) -> tuple[RunResult | None, int | None, int]:
    previous_handlers = dict(_STARTUP_PREVIOUS_HANDLERS)
    observed_signum = _take_startup_signal()

    def observe(signum: int, _frame: object) -> None:
        nonlocal observed_signum
        if observed_signum == 0:
            observed_signum = signum

    for signum in _INTERRUPTION_SIGNALS:
        _ = signal.signal(signum, observe)
    startup_signum = _take_startup_signal()
    if observed_signum == 0:
        observed_signum = startup_signum
    if observed_signum:
        _restore_startup_signal_handlers()
        return None, observed_signum, observed_signum
    interrupted_signum: int | None = None
    result: RunResult | None = None
    try:
        try:
            result = execute_scenario(
                scenario=scenario,
                evidence_dir=evidence,
                repository_root=repository_root,
                interrupt_reader=lambda: observed_signum or None,
            )
        except VerificationInterruptedError as error:
            interrupted_signum = error.signum
    finally:
        if interrupted_signum is None and observed_signum == 0:
            for signum, handler in previous_handlers.items():
                _ = signal.signal(signum, handler)
    return result, interrupted_signum, observed_signum


def _take_startup_signal() -> int:
    startup_signum = _startup_signum[0]
    _startup_signum[0] = 0
    return startup_signum


def _restore_startup_signal_handlers() -> None:
    for signum, handler in _STARTUP_PREVIOUS_HANDLERS:
        _ = signal.signal(signum, handler)


@app.command()
def verify(
    scenario: Annotated[
        str,
        OptionInfo(default=..., param_decls=("--scenario",), help="Scenario name to execute"),
    ],
    evidence: Annotated[
        Path,
        OptionInfo(default=..., param_decls=("--evidence",), help="Evidence directory"),
    ],
) -> None:
    """Execute an isolated verification scenario."""
    repository_root = Path(__file__).resolve().parents[3]
    try:
        result, interrupted_signum, observed_signum = _execute_with_signal_handlers(
            scenario=scenario,
            evidence=evidence,
            repository_root=repository_root,
        )
    except EvidencePathError as error:
        typer.echo(
            f'{{"outcome":"failed","error":{{"code":"invalid_evidence","message":"{error}"}}}}'
        )
        raise typer.Exit(code=2) from error
    if interrupted_signum is not None:
        raise typer.Exit(code=128 + interrupted_signum)
    if observed_signum:
        raise typer.Exit(code=128 + observed_signum)
    if result is None:
        raise typer.Exit(code=1)
    typer.echo(result.model_dump_json())
    if result.exit_code != 0:
        raise typer.Exit(code=result.exit_code)


def main() -> None:
    """Run the command dispatcher."""
    try:
        app()
    finally:
        _restore_startup_signal_handlers()


if __name__ == "__main__":
    main()
