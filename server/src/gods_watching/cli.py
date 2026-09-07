"""Initial command dispatcher for the research server lifecycle."""

import signal
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from typer.models import OptionInfo

from gods_watching.verification import EvidencePathError, RunResult, execute_scenario
from gods_watching.verification.context import VerificationInterruptedError

app = typer.Typer(
    name="gods-watching",
    help="Operate and verify the Gods Watching research server.",
    no_args_is_help=True,
)
credentials_app = typer.Typer(help="Manage the local operator credential.")
app.add_typer(credentials_app, name="credentials")

_INTERRUPTION_SIGNALS = (signal.SIGINT, signal.SIGTERM)


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


def _execute_with_signal_handlers(
    *, scenario: str, evidence: Path, repository_root: Path
) -> tuple[RunResult | None, int | None, int]:
    previous_handlers = {signum: signal.getsignal(signum) for signum in _INTERRUPTION_SIGNALS}
    observed_signum = 0

    def observe(signum: int, _frame: object) -> None:
        nonlocal observed_signum
        if observed_signum == 0:
            observed_signum = signum

    for signum in _INTERRUPTION_SIGNALS:
        _ = signal.signal(signum, observe)
    interrupted_signum: int | None = None
    result: RunResult | None = None
    try:
        try:
            result = execute_scenario(
                scenario=scenario,
                evidence_dir=evidence,
                repository_root=repository_root,
            )
        except VerificationInterruptedError as error:
            interrupted_signum = error.signum
    finally:
        if interrupted_signum is None and observed_signum == 0:
            for signum, handler in previous_handlers.items():
                _ = signal.signal(signum, handler)
    return result, interrupted_signum, observed_signum


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


@credentials_app.command("set")
def set_credentials(
    password_file: Annotated[
        Path,
        OptionInfo(default=..., param_decls=("--password-file",), help="Mode-0600 password file"),
    ],
) -> None:
    """Atomically replace the local operator password."""
    _unimplemented(f"credentials set --password-file {password_file}")


def main() -> None:
    """Run the command dispatcher."""
    app()


if __name__ == "__main__":
    main()
