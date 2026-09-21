"""Docker Compose lifecycle boundary for the packaged product."""

import json
import os
import secrets
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, override

from cryptography.fernet import Fernet

_REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_ENV_PATH: Path = _REPOSITORY_ROOT / ".env"
_ENV_MODE: Final = 0o600
_MODE_MASK: Final = 0o777
_MIN_OPERATOR_PASSWORD_LENGTH: Final = 12
_MIN_POSTGRES_PASSWORD_LENGTH: Final = 20


class LifecycleAction(StrEnum):
    """Name one supported product lifecycle operation."""

    PREPARE = "prepare"
    UP = "up"
    DOWN = "down"
    STATUS = "status"


_COMPOSE_OPERATIONS: Final[Mapping[LifecycleAction, tuple[tuple[str, ...], ...]]] = {
    LifecycleAction.PREPARE: (("config", "--quiet"), ("build",)),
    LifecycleAction.UP: (("up", "-d"),),
    LifecycleAction.DOWN: (("down",),),
    LifecycleAction.STATUS: (("ps",),),
}


@dataclass(frozen=True, slots=True)
class LifecycleCommandError(RuntimeError):
    """Report a failed runtime command without exposing environment values."""

    command: tuple[str, ...]
    exit_code: int

    @override
    def __str__(self) -> str:
        return f"command failed ({self.exit_code}): {' '.join(self.command)}"


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """Capture machine-readable local runtime readiness."""

    docker: bool
    compose: bool
    gpu: bool
    configuration: bool
    credentials: bool

    @property
    def ready(self) -> bool:
        """Return whether every required local capability passed."""
        return self.docker and self.compose and self.gpu and self.configuration and self.credentials

    def to_json(self) -> str:
        """Serialize the stable operator-facing report."""
        return json.dumps(
            {**asdict(self), "ready": self.ready},
            sort_keys=True,
            separators=(",", ":"),
        )


def execute_lifecycle(action: LifecycleAction) -> None:
    """Execute one Compose operation from the repository root."""
    match action:
        case LifecycleAction.PREPARE:
            _prepare_environment()
        case LifecycleAction.UP | LifecycleAction.DOWN | LifecycleAction.STATUS:
            pass
    for arguments in _COMPOSE_OPERATIONS[action]:
        _run_compose(*arguments)


def _prepare_environment() -> None:
    if _ENV_PATH.exists():
        return
    lines = (
        "GW_BIND_HOST=0.0.0.0",
        f"GW_CAMERA_CIPHER_KEY={Fernet.generate_key().decode('ascii')}",
        f"GW_OPERATOR_PASSWORD={secrets.token_urlsafe(24)}",
        f"GW_POSTGRES_PASSWORD={secrets.token_urlsafe(24)}",
        "GW_PUBLIC_HOST=127.0.0.1",
        "GW_PUBLIC_ORIGIN=http://localhost:8080",
        "GW_SECURE_COOKIE=false",
    )
    try:
        descriptor = os.open(_ENV_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _ENV_MODE)
    except FileExistsError:
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as environment_file:
        _ = environment_file.write("\n".join(lines) + "\n")


def _environment_ready() -> bool:
    try:
        metadata = _ENV_PATH.stat()
        if metadata.st_mode & _MODE_MASK != _ENV_MODE:
            return False
        values = dict(
            line.split("=", maxsplit=1)
            for line in _ENV_PATH.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#") and "=" in line
        )
        camera_key = values["GW_CAMERA_CIPHER_KEY"]
        operator_password = values["GW_OPERATOR_PASSWORD"]
        postgres_password = values["GW_POSTGRES_PASSWORD"]
        _ = Fernet(camera_key.encode("ascii"))
    except (FileNotFoundError, KeyError, OSError, UnicodeError, ValueError):
        return False
    return (
        len(operator_password) >= _MIN_OPERATOR_PASSWORD_LENGTH
        and len(postgres_password) >= _MIN_POSTGRES_PASSWORD_LENGTH
        and "replace-with" not in operator_password
        and "replace-with" not in postgres_password
    )


def inspect_runtime() -> DoctorReport:
    """Probe Docker, Compose, NVIDIA GPU access, and the packaged configuration."""
    return DoctorReport(
        docker=_probe("docker", "version"),
        compose=_probe("docker", "compose", "version"),
        gpu=_probe("nvidia-smi", "--query-gpu=name", "--format=csv,noheader"),
        configuration=_probe("docker", "compose", "config", "--quiet"),
        credentials=_environment_ready(),
    )


def _run_compose(*arguments: str) -> None:
    command = ("docker", "compose", *arguments)
    try:
        completed = subprocess.run(command, cwd=_REPOSITORY_ROOT, check=False)  # noqa: S603
    except FileNotFoundError as error:
        raise LifecycleCommandError(command=command, exit_code=127) from error
    if completed.returncode != 0:
        raise LifecycleCommandError(command=command, exit_code=completed.returncode)


def _probe(*command: str) -> bool:
    try:
        completed = subprocess.run(  # noqa: S603
            command,
            cwd=_REPOSITORY_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, PermissionError):
        return False
    return completed.returncode == 0


__all__ = [
    "DoctorReport",
    "LifecycleAction",
    "LifecycleCommandError",
    "execute_lifecycle",
    "inspect_runtime",
]
