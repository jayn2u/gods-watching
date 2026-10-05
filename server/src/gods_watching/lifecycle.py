"""Docker Compose lifecycle boundary for the packaged product."""

import json
import os
import secrets
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from fcntl import LOCK_EX, LOCK_UN, flock
from pathlib import Path
from typing import Final, override
from urllib.parse import urlsplit

from cryptography.fernet import Fernet

from gods_watching.model_selection import ClipModelRegistry, PreparedModelCatalog
from gods_watching.setup.model_preparation import (
    PreparationCommandError,
    PreparationPaths,
    invalidate_model_markers,
    prepare_model_assets,
    reuse_existing_yolo_asset,
)

_REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_ENV_PATH: Path = _REPOSITORY_ROOT / ".env"
_MODEL_LOCK_PATH: Final[Path] = _REPOSITORY_ROOT / "assets/models.lock.json"
_MODEL_ASSETS_ROOT: Final[Path] = _REPOSITORY_ROOT / "runtime/assets/models"
_MODEL_IMPORTED_ROOT: Final[Path] = _MODEL_ASSETS_ROOT / "imported"
_TRAINING_RUNTIME_ROOT: Final[Path] = _REPOSITORY_ROOT / "runtime/training"
_TRAINING_RUNS_ROOT: Final[Path] = _TRAINING_RUNTIME_ROOT / "runs"
_UNCONFIGURED_DATASET_ROOT: Final[Path] = _TRAINING_RUNTIME_ROOT / "unconfigured-dataset"
_ENV_MODE: Final = 0o600
_MODE_MASK: Final = 0o777
_MIN_OPERATOR_PASSWORD_LENGTH: Final = 4
_DEFAULT_OPERATOR_USERNAME: Final = "admin"
_DEFAULT_FIXTURE_RTSP_PORT: Final = 28554
_DEFAULT_PUBLIC_PORT: Final = 8080
_DEFAULT_PUBLIC_TLS_PORT: Final = 8443
# Every service joins the host network namespace, so each of these binds the host
# directly and no two may name the same port.
_HOST_PORT_DEFAULTS: Final[Mapping[str, int]] = {
    "GW_PUBLIC_PORT": _DEFAULT_PUBLIC_PORT,
    "GW_PUBLIC_TLS_PORT": _DEFAULT_PUBLIC_TLS_PORT,
    "GW_API_PORT": 18000,
    "GW_POSTGRES_PORT": 15432,
    "GW_TRITON_HTTP_PORT": 18010,
    "GW_TRITON_GRPC_PORT": 18011,
    "GW_TRITON_METRICS_PORT": 18012,
    "GW_MEDIA_RTSP_PORT": 18554,
    "GW_MEDIA_WHEP_PORT": 18889,
    "GW_MEDIA_CONTROL_PORT": 19997,
    "GW_MEDIA_WEBRTC_UDP_PORT": 8189,
    "GW_FIXTURE_RTSP_PORT": _DEFAULT_FIXTURE_RTSP_PORT,
}
_MIN_TCP_PORT: Final = 1
_MAX_TCP_PORT: Final = 65535
_MIN_POSTGRES_PASSWORD_LENGTH: Final = 20
_MIN_MEDIA_PASSWORD_LENGTH: Final = 20


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
    detail: str | None = None

    @override
    def __str__(self) -> str:
        message = f"command failed ({self.exit_code}): {' '.join(self.command)}"
        return f"{message}: {self.detail}" if self.detail else message


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """Capture machine-readable local runtime readiness."""

    docker: bool
    compose: bool
    gpu: bool
    configuration: bool
    credentials: bool
    models: bool | None = None

    @property
    def ready(self) -> bool:
        """Return whether every required local capability passed."""
        return (
            self.docker
            and self.compose
            and self.gpu
            and self.configuration
            and self.credentials
            and self.models is not False
        )

    def to_json(self) -> str:
        """Serialize the stable operator-facing report."""
        values = {**asdict(self), "ready": self.ready}
        if values["models"] is None:
            del values["models"]
        return json.dumps(
            values,
            sort_keys=True,
            separators=(",", ":"),
        )


def execute_lifecycle(action: LifecycleAction) -> None:
    """Execute one Compose operation from the repository root."""
    match action:
        case LifecycleAction.PREPARE:
            _prepare_environment()
            _prepare_model_asset_directory()
            _prepare_training_directories()
            _run_compose("config", "--quiet")
            _invalidate_model_markers()
            _reuse_existing_yolo_asset()
            _run_compose("build")
            _prepare_model_assets()
            return
        case LifecycleAction.UP | LifecycleAction.DOWN | LifecycleAction.STATUS:
            pass
    for arguments in _COMPOSE_OPERATIONS[action]:
        _run_compose(*arguments)


def _prepare_model_asset_directory() -> None:
    """Create only the bounded host model directory before Compose validates it."""
    try:
        _MODEL_ASSETS_ROOT.mkdir(parents=True, exist_ok=True)
    except PermissionError as error:
        try:
            _ = reuse_existing_yolo_asset(
                PreparationPaths(
                    repository_root=_REPOSITORY_ROOT,
                    lock_path=_MODEL_LOCK_PATH,
                    assets_root=_MODEL_ASSETS_ROOT,
                )
            )
        except (PreparationCommandError, OSError, ValueError) as helper_error:
            raise LifecycleCommandError(
                command=("model directory setup",),
                exit_code=2,
                detail=f"cannot create {_MODEL_ASSETS_ROOT}: {helper_error}",
            ) from helper_error
        if not _MODEL_ASSETS_ROOT.is_dir():
            raise LifecycleCommandError(
                command=("model directory setup",),
                exit_code=2,
                detail=f"cannot create {_MODEL_ASSETS_ROOT}: {error}",
            ) from error


def _prepare_training_directories() -> None:
    """Create only the bind sources used by the isolated training services."""
    try:
        for path in (
            _MODEL_IMPORTED_ROOT,
            _TRAINING_RUNS_ROOT,
            _UNCONFIGURED_DATASET_ROOT,
        ):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as error:
        raise LifecycleCommandError(
            command=("training storage setup",),
            exit_code=2,
            detail="cannot create the training data, history, or publication directories",
        ) from error


def _reuse_existing_yolo_asset() -> None:
    """Reuse detector weights from the previous image before rebuilding it."""
    try:
        _ = reuse_existing_yolo_asset(
            PreparationPaths(
                repository_root=_REPOSITORY_ROOT,
                lock_path=_MODEL_LOCK_PATH,
                assets_root=_MODEL_ASSETS_ROOT,
            )
        )
    except (PreparationCommandError, OSError, ValueError) as error:
        raise LifecycleCommandError(
            command=("model cache copy",), exit_code=2, detail=str(error)
        ) from error


def _invalidate_model_markers() -> None:
    """Clear stale identities before Compose can build a replacement image."""
    try:
        invalidate_model_markers(
            PreparationPaths(
                repository_root=_REPOSITORY_ROOT,
                lock_path=_MODEL_LOCK_PATH,
                assets_root=_MODEL_ASSETS_ROOT,
            )
        )
    except (PreparationCommandError, OSError, RuntimeError, ValueError) as error:
        raise LifecycleCommandError(
            command=("model marker invalidation",), exit_code=2, detail=str(error)
        ) from error


def _prepare_model_assets() -> None:
    """Run the immutable model preparation gate after the Compose image build."""
    try:
        _ = prepare_model_assets(
            PreparationPaths(
                repository_root=_REPOSITORY_ROOT,
                lock_path=_MODEL_LOCK_PATH,
                assets_root=_MODEL_ASSETS_ROOT,
                build_image=False,
                source_image="gods-watching-triton:25.02",
                clear_markers=False,
            )
        )
    except (PreparationCommandError, OSError, RuntimeError, ValueError) as error:
        raise LifecycleCommandError(
            command=("model preparation",), exit_code=2, detail=str(error)
        ) from error


def _prepare_environment() -> None:
    if _ENV_PATH.exists():
        _append_missing_media_credentials()
        return
    lines = (
        "GW_BIND_HOST=0.0.0.0",
        f"GW_CAMERA_CIPHER_KEY={Fernet.generate_key().decode('ascii')}",
        f"GW_MEDIA_CONTROL_PASSWORD={secrets.token_urlsafe(24)}",
        f"GW_MEDIA_READER_PASSWORD={secrets.token_urlsafe(24)}",
        f"GW_OPERATOR_PASSWORD={secrets.token_urlsafe(24)}",
        f"GW_OPERATOR_USERNAME={_DEFAULT_OPERATOR_USERNAME}",
        f"GW_POSTGRES_PASSWORD={secrets.token_urlsafe(24)}",
        "GW_PUBLIC_HOST=127.0.0.1",
        f"GW_PUBLIC_ORIGIN=http://localhost:{_DEFAULT_PUBLIC_PORT}",
        f"GW_PUBLIC_PORT={_DEFAULT_PUBLIC_PORT}",
        f"GW_PUBLIC_TLS_PORT={_DEFAULT_PUBLIC_TLS_PORT}",
        "GW_SECURE_COOKIE=false",
        "GW_TRAINING_DATASET_ROOT=",
        *(
            f"{key}={port}"
            for key, port in _HOST_PORT_DEFAULTS.items()
            if key not in {"GW_PUBLIC_PORT", "GW_PUBLIC_TLS_PORT"}
        ),
    )
    try:
        descriptor = os.open(_ENV_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _ENV_MODE)
    except FileExistsError:
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as environment_file:
        _ = environment_file.write("\n".join(lines) + "\n")


def _append_missing_media_credentials() -> None:
    try:
        metadata = _ENV_PATH.stat()
    except FileNotFoundError:
        return
    if metadata.st_mode & _MODE_MASK != _ENV_MODE:
        return
    try:
        environment_file = _ENV_PATH.open("r+", encoding="utf-8")
    except FileNotFoundError:
        return
    with environment_file:
        flock(environment_file.fileno(), LOCK_EX)
        try:
            contents = environment_file.read()
            keys = {
                line.split("=", maxsplit=1)[0]
                for line in contents.splitlines()
                if line and not line.startswith("#") and "=" in line
            }
            additions = tuple(
                f"{key}={secrets.token_urlsafe(24)}"
                for key in ("GW_MEDIA_CONTROL_PASSWORD", "GW_MEDIA_READER_PASSWORD")
                if key not in keys
            )
            if not additions:
                return
            _ = environment_file.seek(0, os.SEEK_END)
            if contents and not contents.endswith("\n"):
                _ = environment_file.write("\n")
            _ = environment_file.write("\n".join(additions) + "\n")
            environment_file.flush()
            os.fsync(environment_file.fileno())
        finally:
            flock(environment_file.fileno(), LOCK_UN)


def _ports_valid(values: Mapping[str, str]) -> bool:
    """Return whether every host-bound port override is usable and unshared."""
    ports: list[int] = []
    for key, fallback in _HOST_PORT_DEFAULTS.items():
        raw = values.get(key, str(fallback)).strip()
        if not raw.isdigit():
            return False
        ports.append(int(raw))
    return all(_MIN_TCP_PORT <= port <= _MAX_TCP_PORT for port in ports) and len(set(ports)) == len(
        ports
    )


def _origin_matches_published_port(values: Mapping[str, str]) -> bool:
    """Return whether the browser-facing origin names the port the gateway publishes."""
    origin = values.get("GW_PUBLIC_ORIGIN", "").strip()
    if not origin:
        return True
    parsed = urlsplit(origin)
    try:
        port = parsed.port
    except ValueError:
        return False
    if parsed.scheme == "http":
        expected = int(values.get("GW_PUBLIC_PORT", str(_DEFAULT_PUBLIC_PORT)))
        return (port or 80) == expected
    if parsed.scheme == "https":
        expected = int(values.get("GW_PUBLIC_TLS_PORT", str(_DEFAULT_PUBLIC_TLS_PORT)))
        return (port or 443) == expected
    return False


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
        media_control_password = values["GW_MEDIA_CONTROL_PASSWORD"]
        media_reader_password = values["GW_MEDIA_READER_PASSWORD"]
        operator_password = values["GW_OPERATOR_PASSWORD"]
        operator_username = values.get("GW_OPERATOR_USERNAME", _DEFAULT_OPERATOR_USERNAME)
        postgres_password = values["GW_POSTGRES_PASSWORD"]
        ports_valid = _ports_valid(values) and _origin_matches_published_port(values)
        _ = Fernet(camera_key.encode("ascii"))
    except (FileNotFoundError, KeyError, OSError, UnicodeError, ValueError):
        return False
    return (
        ports_valid
        and len(operator_username.strip()) > 0
        and len(media_control_password) >= _MIN_MEDIA_PASSWORD_LENGTH
        and len(media_reader_password) >= _MIN_MEDIA_PASSWORD_LENGTH
        and len(operator_password) >= _MIN_OPERATOR_PASSWORD_LENGTH
        and len(postgres_password) >= _MIN_POSTGRES_PASSWORD_LENGTH
        and "replace-with" not in media_control_password
        and "replace-with" not in media_reader_password
        and "replace-with" not in operator_password
        and "replace-with" not in postgres_password
    )


def inspect_runtime() -> DoctorReport:
    """Probe Docker, Compose, NVIDIA GPU access, and the packaged configuration."""
    models: bool | None = None
    if _ENV_PATH == _REPOSITORY_ROOT / ".env":
        catalog = PreparedModelCatalog(
            ClipModelRegistry(), _MODEL_LOCK_PATH, _MODEL_ASSETS_ROOT
        )
        models = all(catalog.status(package).prepared for package in catalog.registry.packages)
    return DoctorReport(
        docker=_probe("docker", "version"),
        compose=_probe("docker", "compose", "version"),
        gpu=_probe("nvidia-smi", "--query-gpu=name", "--format=csv,noheader"),
        configuration=_probe("docker", "compose", "config", "--quiet"),
        credentials=_environment_ready(),
        models=models,
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
