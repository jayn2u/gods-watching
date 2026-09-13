"""Standalone operator-credential command for future launcher registration."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, NoReturn, override

import anyio
import typer
from pydantic import Field, ValidationError
from sqlalchemy.exc import SQLAlchemyError
from typer.models import OptionInfo

from gods_watching.settings import AppSettings
from gods_watching.storage import Database

from .passwords import SecretFileError, SecretFileReason
from .repository import CredentialStorageError
from .service import AuthService
from .types import PasswordReplacement

_SECRET_FILE_MAX_BYTES: Final = 514
_SECRET_FILE_MODE: Final = 0o600
_SECRET_FILE_PERMISSION_MASK: Final = 0o777


def _database_url_from_environment() -> str:
    return os.environ.get("GW_DATABASE_URL", "")


class CredentialCommandSettings(AppSettings):
    """Load the command's required database URL through process settings."""

    database_url: str = Field(default_factory=_database_url_from_environment, min_length=1)


@dataclass(frozen=True, slots=True)
class CredentialConfigurationError(Exception):
    """Report missing process configuration without exposing configuration values."""

    @override
    def __str__(self) -> str:
        """Return a stable configuration error."""
        return "database configuration is unavailable"


credentials_app = typer.Typer(
    name="credentials",
    add_completion=False,
    no_args_is_help=True,
    help="Manage the local operator credential.",
)


def _credentials_group() -> None:
    """Manage the local operator credential."""


_ = credentials_app.callback()(_credentials_group)


def _load_settings() -> CredentialCommandSettings:
    try:
        return CredentialCommandSettings()
    except ValidationError as error:
        raise CredentialConfigurationError from error


def _preflight_password_file(path: Path) -> None:
    try:
        metadata = path.stat()
    except FileNotFoundError as error:
        raise SecretFileError(path, SecretFileReason.MISSING) from error
    except OSError as error:
        raise SecretFileError(path, SecretFileReason.METADATA) from error
    if metadata.st_mode & _SECRET_FILE_PERMISSION_MASK != _SECRET_FILE_MODE:
        raise SecretFileError(path, SecretFileReason.MODE)
    if metadata.st_size > _SECRET_FILE_MAX_BYTES:
        raise SecretFileError(path, SecretFileReason.POLICY)


async def _replace_password_file(
    path: Path,
    settings: CredentialCommandSettings,
) -> PasswordReplacement:
    database = Database.connect(settings.database_url)
    try:
        service = AuthService(transactions=database)
        return await service.replace_password_file(path)
    finally:
        await database.close()


def _write_failure(message: str, exit_code: int) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code=exit_code)


@credentials_app.command("set")
def set_credentials(
    password_file: Annotated[
        Path,
        OptionInfo(default=..., param_decls=("--password-file",), help="Mode-0600 password file"),
    ],
) -> None:
    """Atomically replace the local operator password."""
    try:
        _preflight_password_file(password_file)
        settings = _load_settings()
    except SecretFileError:
        _write_failure("password file rejected", 2)
    except CredentialConfigurationError as error:
        _write_failure(str(error), 2)

    try:
        replacement = anyio.run(_replace_password_file, password_file, settings)
    except SecretFileError:
        _write_failure("password file rejected", 2)
    except (CredentialStorageError, SQLAlchemyError):
        _write_failure("credential update failed", 1)
    except Exception:  # noqa: BLE001
        _write_failure("credential update failed", 1)
    typer.echo(
        json.dumps(
            {"changed": replacement.changed, "revoked_sessions": len(replacement.revoked)},
            separators=(",", ":"),
        )
    )


def main() -> None:
    """Run the standalone credentials command group."""
    credentials_app()


if __name__ == "__main__":
    main()


__all__ = ["CredentialCommandSettings", "credentials_app", "main", "set_credentials"]
