"""Initial command dispatcher for the research server lifecycle."""

from pathlib import Path
from typing import Annotated, NoReturn

import typer
from typer.models import OptionInfo

app = typer.Typer(
    name="gods-watching",
    help="Operate and verify the Gods Watching research server.",
    no_args_is_help=True,
)
credentials_app = typer.Typer(help="Manage the local operator credential.")
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
    _unimplemented(f"verify --scenario {scenario} --evidence {evidence}")


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
