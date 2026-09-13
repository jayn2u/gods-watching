"""Drive the disposable PostgreSQL, MediaMTX, FastAPI, and browser stack."""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import secrets
import shutil
import signal
import ssl
import subprocess
from dataclasses import dataclass
from functools import partial
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Final, NoReturn, Protocol, cast

import anyio
from anyio import to_thread
from cryptography import x509
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.serialization import Encoding
from pydantic import TypeAdapter

from gods_watching.verification.context import allocate_loopback_port

from .task12_models import (
    BrowserArtifact,
    HttpResult,
    JsonObject,
    JsonValue,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from anyio.abc import Process

_REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[5]
_MEDIA_IMAGE: Final[str] = (
    "bluenviron/mediamtx@sha256:206139c58377b7544d6ef63f8af86bcf46b9e89262aaa6f7513e1347dbab7d36"
)
_DATABASE_IMAGE: Final[str] = "pgvector/pgvector:0.8.1-pg17"
_READER_USER: Final[str] = "task12-media-reader"
_PUBLISHER_USER: Final[str] = "task12-test-publisher"
_CONTROL_USER: Final[str] = "task12-media-control"
_SOURCE: Final[Path] = _REPOSITORY_ROOT / "runtime/assets/fixtures/crosswalk.mp4"
_HTTP_OK: Final[int] = 200
_HTTP_CREATED: Final[int] = 201
_HTTP_NO_CONTENT: Final[int] = 204
_HTTP_UNAUTHORIZED: Final[int] = 401
_HTTP_FORBIDDEN: Final[int] = 403
_HTTP_UNPROCESSABLE: Final[int] = 422
_CLI_FAILURE: Final[int] = 1
_CLI_USAGE: Final[int] = 2
_RETENTION_DAYS: Final[int] = 14


def _fail(message: str) -> NoReturn:
    raise RuntimeError(message)


class _HttpClient:
    _port: int
    _origin: str
    _ca_file: Path | None

    def __init__(self, port: int, origin: str, ca_file: Path | None = None) -> None:
        self._port = port
        self._origin = origin
        self._ca_file = ca_file
        self.cookie: str | None = None

    @property
    def ca_file(self) -> Path | None:
        return self._ca_file

    def request(
        self,
        method: str,
        path: str,
        payload: JsonObject | None = None,
        *,
        origin: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> HttpResult:
        headers = {"Origin": origin or self._origin}
        if extra_headers is not None:
            headers.update(extra_headers)
        body: bytes | None = None
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        if self.cookie is not None:
            headers["Cookie"] = self.cookie
        if self._origin.startswith("https://"):
            ca_file = self._ca_file
            if ca_file is None:
                _fail("Task 12 HTTPS client is missing its CA bundle")
            connection = HTTPSConnection(
                "127.0.0.1",
                self._port,
                timeout=20,
                context=ssl.create_default_context(cafile=ca_file),
            )
        else:
            connection = HTTPConnection("127.0.0.1", self._port, timeout=20)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read(1_100_000)
            response_headers = {key.lower(): value for key, value in response.getheaders()}
        finally:
            connection.close()
        set_cookie = response_headers.get("set-cookie")
        if set_cookie is not None:
            self.cookie = set_cookie.split(";", 1)[0]
        try:
            decoded: JsonValue = _JSON_VALUE_ADAPTER.validate_json(raw) if raw else None
        except (UnicodeDecodeError, ValueError):
            decoded = None
        return HttpResult(status=response.status, headers=response_headers, body=decoded)


_JSON_VALUE_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


def _required(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value:
        _fail(f"Task 12 driver configuration is missing: {name}")
    return value


def _sha256_secret(value: str) -> str:
    return base64.b64encode(hashlib.sha256(value.encode()).digest()).decode("ascii")


def _write_json(path: Path, payload: JsonValue) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _ = temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _ = temporary.replace(path)


class _CompletedProcess(Protocol):
    returncode: int


def _safe_returncode(result: _CompletedProcess) -> int:
    return result.returncode


async def _command(
    command: tuple[str, ...],
    *,
    env: dict[str, str] | None = None,
    cwd: Path = _REPOSITORY_ROOT,
) -> subprocess.CompletedProcess[bytes]:
    return await anyio.run_process(command, env=env, cwd=cwd, check=False)


async def _wait_tcp(port: int, *, deadline_seconds: float = 25.0) -> None:
    with anyio.fail_after(deadline_seconds):
        while True:
            try:
                stream = await anyio.connect_tcp("127.0.0.1", port)
            except OSError:
                await anyio.sleep(0.1)
            else:
                await stream.aclose()
                return


async def _wait_postgres(container: str) -> None:
    with anyio.fail_after(40):
        while True:
            result = await _command(
                (
                    "/usr/bin/docker",
                    "exec",
                    container,
                    "pg_isready",
                    "-h",
                    "127.0.0.1",
                    "-U",
                    "postgres",
                    "-d",
                    "gods_watching_test",
                )
            )
            if _safe_returncode(result) == 0:
                return
            await anyio.sleep(0.2)


async def _wait_http(client: _HttpClient, process: Process) -> None:
    with anyio.fail_after(35):
        while True:
            if process.returncode is not None:
                _fail("Task 12 app exited during startup")
            try:
                result = await to_thread.run_sync(partial(client.request, "GET", "/api/session"))
            except OSError:
                await anyio.sleep(0.2)
            else:
                if result.status == _HTTP_OK:
                    return
                await anyio.sleep(0.2)


async def _stop(process: Process | None) -> None:
    if process is None or process.returncode is not None:
        return
    with anyio.CancelScope(shield=True):
        _ = process.terminate()
        with anyio.move_on_after(5):
            _ = await process.wait()
        if process.returncode is None:
            process.kill()
            _ = await process.wait()


async def _remove(name: str) -> None:
    _ = await _command(("/usr/bin/docker", "rm", "--force", name))


def _render_media_config(
    path: Path,
    *,
    reader_password: str,
    publisher_password: str,
    control_password: str,
    ports: tuple[int, int, int, int],
) -> None:
    rtsp_port, whep_port, control_port, udp_port = ports
    template = (_REPOSITORY_ROOT / "deploy/mediamtx.yml").read_text(encoding="utf-8")
    rendered = (
        template.replace("__GW_MEDIA_READER_USER__", _READER_USER)
        .replace("__GW_MEDIA_READER_PASS_SHA256__", _sha256_secret(reader_password))
        .replace("__GW_TEST_PUBLISHER_USER__", _PUBLISHER_USER)
        .replace("__GW_TEST_PUBLISHER_PASS_SHA256__", _sha256_secret(publisher_password))
        .replace("__GW_MEDIA_CONTROL_USER__", _CONTROL_USER)
        .replace("__GW_MEDIA_CONTROL_PASS_SHA256__", _sha256_secret(control_password))
        .replace("rtspAddress: :8554", f"rtspAddress: :{rtsp_port}")
        .replace("webrtcAddress: :8889", f"webrtcAddress: :{whep_port}")
        .replace("apiAddress: :9997", f"apiAddress: :{control_port}")
        .replace("webrtcLocalUDPAddress: :8189", f"webrtcLocalUDPAddress: :{udp_port}")
        .replace("webrtcIPsFromInterfaces: no", "webrtcIPsFromInterfaces: yes")
    )
    _ = path.write_text(rendered, encoding="utf-8")
    path.chmod(0o644)


async def _start_publisher(
    *,
    source: str,
    password: str,
    rtsp_port: int,
) -> Process:
    destination = f"rtsp://{_PUBLISHER_USER}:{password}@127.0.0.1:{rtsp_port}/{source}"
    return await anyio.open_process(
        (
            "/usr/bin/ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-re",
            "-stream_loop",
            "-1",
            "-i",
            str(_SOURCE),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-profile:v",
            "baseline",
            "-pix_fmt",
            "yuv420p",
            "-bf",
            "0",
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            destination,
        ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


async def _probe(source: str, *, expected: bool, deadline_seconds: float = 20.0) -> bool:
    with anyio.fail_after(deadline_seconds):
        while True:
            result = await _command(
                (
                    "/usr/bin/ffprobe",
                    "-v",
                    "error",
                    "-rtsp_transport",
                    "tcp",
                    "-rw_timeout",
                    "1500000",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,width,height",
                    "-of",
                    "json",
                    source,
                )
            )
            found = _safe_returncode(result) == 0
            if found is expected:
                return found
            await anyio.sleep(0.2)


async def _run_browser(
    configuration: _BrowserConfiguration,
) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "PLAYWRIGHT_BROWSERS_PATH": str(_REPOSITORY_ROOT / "runtime/playwright-browsers"),
            "GW_TASK12_CONTROL_PORT": str(configuration.control_port),
            "GW_TASK12_CONTROL_USER": _CONTROL_USER,
            "GW_TASK12_CONTROL_PASSWORD": configuration.control_password,
            "GW_TASK12_MEDIA_PATH": configuration.media_path,
            "HOME": str(configuration.profile),
            "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
            "SSL_CERT_DIR": "/etc/ssl/certs",
        }
    )
    browser_script = str(Path(__file__).with_name("task12_browser.mjs"))
    command = (
        "/usr/bin/bwrap",
        "--die-with-parent",
        "--new-session",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/bin",
        "/bin",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--ro-bind",
        "/sbin",
        "/sbin",
        "--ro-bind",
        "/etc",
        "/etc",
        "--ro-bind",
        str(configuration.ca_bundle),
        "/etc/ssl/certs/ca-certificates.crt",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",  # noqa: S108
        "--ro-bind",
        str(_REPOSITORY_ROOT),
        str(_REPOSITORY_ROOT),
        "--bind",
        str(configuration.profile),
        str(configuration.profile),
        "--bind",
        str(configuration.output.parent),
        str(configuration.output.parent),
        "--chdir",
        str(_REPOSITORY_ROOT),
        "--setenv",
        "HOME",
        str(configuration.profile),
        "/usr/bin/node",
        browser_script,
        configuration.origin,
        configuration.password,
        configuration.camera_id,
        str(configuration.output),
        configuration.mode,
        str(configuration.profile),
    )
    browser_log_path = configuration.output.with_name("task12-browser.log")
    with browser_log_path.open("wb") as browser_log:
        process = await anyio.open_process(
            command, env=environment, stdout=browser_log, stderr=browser_log
        )
        returncode = await process.wait()
    if returncode != 0:
        _fail(f"Task 12 browser driver failed in {configuration.mode} mode")


@dataclass(frozen=True, slots=True)
class _BrowserConfiguration:
    origin: str
    password: str
    camera_id: str
    output: Path
    mode: str
    control_port: int
    control_password: str
    media_path: str
    ca_bundle: Path
    profile: Path


class _NssSecItem(ctypes.Structure):
    pass


_NssSecItem._fields_ = [
    ("type", ctypes.c_int),
    ("data", ctypes.POINTER(ctypes.c_ubyte)),
    ("length", ctypes.c_uint),
]


class _NssCertTrust(ctypes.Structure):
    pass


_NssCertTrust._fields_ = [
    ("ssl_flags", ctypes.c_uint),
    ("email_flags", ctypes.c_uint),
    ("object_signing_flags", ctypes.c_uint),
]


def _prepare_browser_nss(profile: Path, ca_certificate: Path) -> None:  # noqa: PLR0915
    """Add the run's CA to the disposable Chromium NSS profile."""
    database = profile / ".pki" / "nssdb"
    database.mkdir(mode=0o700, parents=True, exist_ok=True)
    certificate = x509.load_pem_x509_certificate(ca_certificate.read_bytes())
    der_certificate = certificate.public_bytes(Encoding.DER)
    data = (ctypes.c_ubyte * len(der_certificate)).from_buffer_copy(der_certificate)
    item = _NssSecItem(
        type=0,
        data=ctypes.cast(data, ctypes.POINTER(ctypes.c_ubyte)),
        length=len(der_certificate),
    )
    try:
        nss = ctypes.CDLL("libnss3.so")
    except OSError as error:
        _fail(f"Task 12 Chromium NSS library is unavailable: {error}")
    nss.NSS_InitReadWrite.argtypes = [ctypes.c_char_p]
    nss.NSS_InitReadWrite.restype = ctypes.c_int
    nss.NSS_Shutdown.argtypes = []
    nss.NSS_Shutdown.restype = ctypes.c_int
    nss.PK11_GetInternalKeySlot.argtypes = []
    nss.PK11_GetInternalKeySlot.restype = ctypes.c_void_p
    nss.PK11_InitPin.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    nss.PK11_InitPin.restype = ctypes.c_int
    nss.PK11_FreeSlot.argtypes = [ctypes.c_void_p]
    nss.PK11_FreeSlot.restype = None
    nss.CERT_GetDefaultCertDB.argtypes = []
    nss.CERT_GetDefaultCertDB.restype = ctypes.c_void_p
    nss.CERT_NewTempCertificate.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_NssSecItem),
        ctypes.c_char_p,
        ctypes.c_ubyte,
        ctypes.c_ubyte,
    ]
    nss.CERT_NewTempCertificate.restype = ctypes.c_void_p
    raw_add_temp_cert = getattr(nss, "__CERT_AddTempCertToPerm")  # pyright: ignore[reportAny]
    raw_add_temp_cert.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(_NssCertTrust),
    ]
    raw_add_temp_cert.restype = ctypes.c_int
    add_temp_cert = cast("Callable[[ctypes.c_void_p, bytes, object], int]", raw_add_temp_cert)
    nss.CERT_DestroyCertificate.argtypes = [ctypes.c_void_p]
    nss.CERT_DestroyCertificate.restype = None
    if nss.NSS_InitReadWrite(f"sql:{database}".encode()) != 0:
        _fail("Task 12 Chromium NSS database initialization failed")
    cert: ctypes.c_void_p | None = None
    try:
        slot = cast("ctypes.c_void_p", nss.PK11_GetInternalKeySlot())
        if not slot:
            _fail("Task 12 Chromium NSS internal slot initialization failed")
        if nss.PK11_InitPin(slot, None, b"") != 0:
            _fail("Task 12 Chromium NSS token initialization failed")
        nss.PK11_FreeSlot(slot)
        database_handle = cast("ctypes.c_void_p", nss.CERT_GetDefaultCertDB())
        if not database_handle:
            _fail("Task 12 Chromium NSS certificate database is unavailable")
        cert = cast(
            "ctypes.c_void_p | None",
            nss.CERT_NewTempCertificate(
                database_handle,
                ctypes.byref(item),
                b"Task 12 HTTPS CA",
                0,
                1,
            ),
        )
        if not cert:
            _fail("Task 12 Chromium NSS certificate decoding failed")
        trust = _NssCertTrust(ssl_flags=24, email_flags=0, object_signing_flags=0)
        if add_temp_cert(cert, b"Task 12 HTTPS CA", ctypes.byref(trust)) != 0:
            _fail("Task 12 Chromium NSS CA trust import failed")
    finally:
        if cert is not None:
            nss.CERT_DestroyCertificate(cert)
        if nss.NSS_Shutdown() != 0:
            _fail("Task 12 Chromium NSS database shutdown failed")


@dataclass(frozen=True, slots=True)
class _CredentialConfiguration:
    run_root: Path
    database_url: str
    replacement_password: str
    rollback_password: str
    app_port: int


def _body_dict(result: HttpResult) -> JsonObject:
    if isinstance(result.body, dict):
        return result.body
    return {}


def _body_list(result: HttpResult) -> list[JsonValue]:
    if isinstance(result.body, list):
        return result.body
    return []


async def _camera_checks(
    client: _HttpClient,
    *,
    source_url: str,
    reader_password: str,
    rtsp_port: int,
) -> tuple[dict[str, bool], JsonObject]:
    valid_test = await to_thread.run_sync(
        partial(client.request, "POST", "/api/cameras/test", {"source_url": source_url})
    )
    malformed_test = await to_thread.run_sync(
        partial(
            client.request,
            "POST",
            "/api/cameras/test",
            {"source_url": "http://not-rtsp.example/source"},
        )
    )
    create = await to_thread.run_sync(
        partial(
            client.request,
            "POST",
            "/api/cameras",
            {
                "name": "Task 12 edit fixture",
                "source_url": source_url,
                "detection_enabled": False,
            },
        )
    )
    created = _body_dict(create)
    extra_id = created.get("camera_id")
    extra_version = created.get("version")
    invalid_edit = HttpResult(status=500, headers={}, body=None)
    preserved = False
    deleted = HttpResult(status=500, headers={}, body=None)
    if isinstance(extra_id, str) and isinstance(extra_version, int):
        invalid_edit = await to_thread.run_sync(
            partial(
                client.request,
                "PATCH",
                f"/api/cameras/{extra_id}",
                {"source_url": f"rtsp://{_READER_USER}:{reader_password}@127.0.0.1:9/nope"},
                extra_headers={"X-Camera-Version": str(extra_version)},
            )
        )
        current = await to_thread.run_sync(partial(client.request, "GET", "/api/cameras"))
        preserved = any(
            isinstance(row, dict)
            and row.get("camera_id") == extra_id
            and row.get("source_port") == rtsp_port
            for row in _body_list(current)
        )
        deleted = await to_thread.run_sync(
            partial(
                client.request,
                "DELETE",
                f"/api/cameras/{extra_id}",
                extra_headers={"X-Camera-Version": str(extra_version)},
            )
        )
    settings_patch = await to_thread.run_sync(
        partial(
            client.request,
            "PATCH",
            "/api/settings",
            {"retention_days": 14, "quota_bytes": 123456789},
        )
    )
    settings_read = await to_thread.run_sync(partial(client.request, "GET", "/api/settings"))
    checks = {
        "source_probe_success": valid_test.status == _HTTP_OK,
        "invalid_source_rejected": malformed_test.status == _HTTP_UNPROCESSABLE,
        "invalid_edit_preserves_source": (invalid_edit.status == _HTTP_UNPROCESSABLE and preserved),
        "camera_delete": deleted.status == _HTTP_NO_CONTENT,
        "settings_persist": (
            settings_patch.status == _HTTP_OK
            and settings_read.status == _HTTP_OK
            and _body_dict(settings_read).get("retention_days") == _RETENTION_DAYS
        ),
    }
    observations: JsonObject = {
        "camera_test_status": valid_test.status,
        "malformed_test_status": malformed_test.status,
        "invalid_edit_status": invalid_edit.status,
        "settings_status": settings_read.status,
    }
    return checks, observations


async def _credential_checks(
    client: _HttpClient,
    configuration: _CredentialConfiguration,
) -> tuple[dict[str, bool], JsonObject]:
    secret_root = configuration.run_root / "secrets"
    secret_root.mkdir(mode=0o700, exist_ok=True)
    new_file = secret_root / "replacement.secret"
    _ = new_file.write_text(configuration.replacement_password + "\n", encoding="utf-8")
    new_file.chmod(0o600)
    cli_environment = os.environ.copy()
    cli_environment["GW_DATABASE_URL"] = configuration.database_url
    command_prefix = (
        str(_REPOSITORY_ROOT / "gods-watching"),
        "credentials",
        "set",
        "--password-file",
    )
    replacement = await _command((*command_prefix, str(new_file)), env=cli_environment)
    old_session = await to_thread.run_sync(partial(client.request, "GET", "/api/session"))
    replacement_client = _HttpClient(
        configuration.app_port,
        f"https://127.0.0.1:{configuration.app_port}",
        client.ca_file,
    )
    replacement_login = await to_thread.run_sync(
        partial(
            replacement_client.request,
            "POST",
            "/api/session",
            {"password": configuration.replacement_password},
        )
    )
    same_password = await _command((*command_prefix, str(new_file)), env=cli_environment)
    missing = await _command(
        (*command_prefix, str(secret_root / "missing.secret")),
        env=cli_environment,
    )
    rollback_file = secret_root / "rollback.secret"
    _ = rollback_file.write_text(configuration.rollback_password + "\n", encoding="utf-8")
    rollback_file.chmod(0o600)
    rollback_environment = cli_environment.copy()
    rollback_environment["GW_DATABASE_URL"] = (
        "postgresql+asyncpg://postgres:bad@127.0.0.1:9/gods_watching_test"
    )
    rollback = await _command((*command_prefix, str(rollback_file)), env=rollback_environment)
    replacement_still_valid = await to_thread.run_sync(
        partial(replacement_client.request, "GET", "/api/session")
    )
    checks = {
        "cli_replacement_revokes_old_session": (
            _safe_returncode(replacement) == 0
            and _body_dict(old_session).get("authenticated") is False
            and replacement_login.status == _HTTP_OK
        ),
        "cli_same_password_noop": _safe_returncode(same_password) == 0,
        "cli_missing_file_preserves": _safe_returncode(missing) == _CLI_USAGE,
        "cli_rollback_preserves": (
            _safe_returncode(rollback) == _CLI_FAILURE
            and _body_dict(replacement_still_valid).get("authenticated") is True
        ),
    }
    observations: JsonObject = {
        "replacement_exit": _safe_returncode(replacement),
        "same_password_exit": _safe_returncode(same_password),
        "missing_exit": _safe_returncode(missing),
        "rollback_exit": _safe_returncode(rollback),
    }
    return checks, observations


def _denied_checks(browser: BrowserArtifact, api_observations: JsonObject) -> dict[str, bool]:
    return {
        "anonymous_api_denied": browser.anonymous_list == _HTTP_UNAUTHORIZED,
        "login_throttle": browser.wrong_login_statuses == [_HTTP_UNAUTHORIZED] * 5 + [429],
        "cross_origin_login_denied": api_observations.get("cross_origin_login") == _HTTP_FORBIDDEN,
        "cross_origin_mutation_denied": api_observations.get("cross_origin_logout")
        == _HTTP_FORBIDDEN,
        "malformed_source_rejected": api_observations.get("malformed_source")
        == _HTTP_UNPROCESSABLE,
        "passive_poll_does_not_refresh": browser.passive_idle_unchanged,
    }


async def _denied_api_checks(client: _HttpClient) -> JsonObject:
    cross_origin_login = await to_thread.run_sync(
        partial(
            client.request,
            "POST",
            "/api/session",
            {"password": "unused-origin-check"},
            origin="https://attacker.invalid",
        )
    )
    cross_origin_logout = await to_thread.run_sync(
        partial(client.request, "DELETE", "/api/session", origin="https://attacker.invalid")
    )
    malformed_source = await to_thread.run_sync(
        partial(
            client.request,
            "POST",
            "/api/cameras/test",
            {"source_url": "http://not-rtsp.example/source"},
        )
    )
    return {
        "cross_origin_login": cross_origin_login.status,
        "cross_origin_logout": cross_origin_logout.status,
        "malformed_source": malformed_source.status,
    }


@dataclass(slots=True)
class _Stack:
    run_root: Path
    output: Path
    compose_project: str
    app_port: int
    database_port: int
    rtsp_port: int
    whep_port: int
    control_port: int
    udp_port: int
    database_name: str
    media_name: str
    database_password: str
    operator_password: str
    replacement_password: str
    rollback_password: str
    reader_password: str
    publisher_password: str
    control_password: str
    cipher_key: str
    media_path: str
    source_url: str
    database_url: str
    config_path: Path
    runtime_root: Path
    tls_key_path: Path
    tls_cert_path: Path
    ca_bundle_path: Path
    browser_profile: Path
    app_log: BinaryIO
    manifest_path: Path
    events: list[dict[str, str]]
    database_started: bool = False
    media_process: Process | None = None
    publisher: Process | None = None
    app_process: Process | None = None

    def event(self, name: str, state: str) -> None:
        self.events.append({"name": name, "state": state})
        events: list[JsonValue] = [dict(event) for event in self.events]
        _write_json(self.manifest_path, {"events": events})


@dataclass(frozen=True, slots=True)
class _AppState:
    client: _HttpClient
    camera_id: str
    browser: BrowserArtifact


def _browser_cookie_flags(browser: BrowserArtifact) -> bool:
    cookie = browser.cookie
    return (
        cookie is not None
        and cookie.http_only
        and cookie.same_site == "Strict"
        and cookie.value_length > 0
        and cookie.secure
    )


async def _prepare_tls(stack: _Stack) -> None:
    stack.runtime_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    ca_key = stack.runtime_root / "ca.key"
    ca_cert = stack.runtime_root / "ca.crt"
    server_key = stack.tls_key_path
    server_csr = stack.runtime_root / "server.csr"
    serial = stack.runtime_root / "ca.srl"
    config = stack.runtime_root / "openssl.cnf"
    config_text = "\n".join(  # noqa: FLY002
        (
            "[req]",
            "distinguished_name = req_distinguished_name",
            "req_extensions = req_ext",
            "[req_distinguished_name]",
            "[req_ext]",
            "subjectAltName = IP:127.0.0.1,DNS:localhost",
            "[v3_req]",
            "subjectAltName = IP:127.0.0.1,DNS:localhost",
            "",
        )
    )
    _ = config.write_text(config_text, encoding="utf-8")
    commands = (
        (
            "/usr/bin/openssl",
            "genrsa",
            "-out",
            str(ca_key),
            "2048",
        ),
        (
            "/usr/bin/openssl",
            "req",
            "-x509",
            "-new",
            "-nodes",
            "-key",
            str(ca_key),
            "-sha256",
            "-days",
            "1",
            "-out",
            str(ca_cert),
            "-subj",
            "/CN=Task 12 test CA",
        ),
        (
            "/usr/bin/openssl",
            "genrsa",
            "-out",
            str(server_key),
            "2048",
        ),
        (
            "/usr/bin/openssl",
            "req",
            "-new",
            "-key",
            str(server_key),
            "-out",
            str(server_csr),
            "-subj",
            "/CN=127.0.0.1",
            "-config",
            str(config),
        ),
        (
            "/usr/bin/openssl",
            "x509",
            "-req",
            "-in",
            str(server_csr),
            "-CA",
            str(ca_cert),
            "-CAkey",
            str(ca_key),
            "-CAcreateserial",
            "-CAserial",
            str(serial),
            "-out",
            str(stack.tls_cert_path),
            "-days",
            "1",
            "-sha256",
            "-extfile",
            str(config),
            "-extensions",
            "v3_req",
        ),
    )
    for command in commands:
        result = await _command(command)
        if _safe_returncode(result) != 0:
            _fail("Task 12 TLS material generation failed")
    bundle = Path("/etc/ssl/certs/ca-certificates.crt").read_bytes()
    _ = stack.ca_bundle_path.write_bytes(bundle + b"\n" + ca_cert.read_bytes())
    _ = ca_key.chmod(0o600)
    _ = server_key.chmod(0o600)
    _ = stack.ca_bundle_path.chmod(0o644)
    _ = stack.tls_cert_path.chmod(0o644)
    stack.browser_profile.mkdir(mode=0o700, parents=True, exist_ok=True)
    _prepare_browser_nss(stack.browser_profile, stack.runtime_root / "ca.crt")


def _prepare_stack() -> _Stack:
    run_root = Path(_required("GW_TASK12_RUN_ROOT"))
    output = Path(_required("GW_TASK12_OUTPUT"))
    compose_project = _required("GW_TASK12_PROJECT")
    app_port = int(_required("GW_TASK12_APP_PORT"))
    run_root.mkdir(parents=True, exist_ok=True)
    database_port = allocate_loopback_port()
    rtsp_port = allocate_loopback_port()
    whep_port = allocate_loopback_port()
    control_port = allocate_loopback_port()
    udp_port = allocate_loopback_port()
    database_name = f"{compose_project}-postgres"
    media_name = f"{compose_project}-mediamtx"
    database_password = secrets.token_urlsafe(24)
    operator_password = secrets.token_urlsafe(24)
    replacement_password = secrets.token_urlsafe(24)
    rollback_password = secrets.token_urlsafe(24)
    reader_password = secrets.token_urlsafe(24)
    publisher_password = secrets.token_urlsafe(24)
    control_password = secrets.token_urlsafe(24)
    cipher_key = Fernet.generate_key().decode("ascii")
    media_path = f"test-publisher/{secrets.token_hex(5)}"
    source_url = f"rtsp://{_READER_USER}:{reader_password}@127.0.0.1:{rtsp_port}/{media_path}"
    database_url = (
        f"postgresql+asyncpg://postgres:{database_password}@127.0.0.1:"
        f"{database_port}/gods_watching_test"
    )
    manifest_path = run_root / "task12-resource-manifest.json"
    runtime_root = run_root / "runtime"
    _write_json(manifest_path, {"events": []})
    return _Stack(
        run_root=run_root,
        output=output,
        compose_project=compose_project,
        app_port=app_port,
        database_port=database_port,
        rtsp_port=rtsp_port,
        whep_port=whep_port,
        control_port=control_port,
        udp_port=udp_port,
        database_name=database_name,
        media_name=media_name,
        database_password=database_password,
        operator_password=operator_password,
        replacement_password=replacement_password,
        rollback_password=rollback_password,
        reader_password=reader_password,
        publisher_password=publisher_password,
        control_password=control_password,
        cipher_key=cipher_key,
        media_path=media_path,
        source_url=source_url,
        database_url=database_url,
        config_path=run_root / "mediamtx.yml",
        runtime_root=runtime_root,
        tls_key_path=runtime_root / "server.key",
        tls_cert_path=runtime_root / "server.crt",
        ca_bundle_path=runtime_root / "ca-bundle.crt",
        browser_profile=runtime_root / "browser-profile",
        app_log=(run_root / "task12-app.log").open("wb"),
        manifest_path=manifest_path,
        events=[],
    )


async def _start_database(stack: _Stack) -> None:
    stack.event(stack.database_name, "registered")
    result = await _command(
        (
            "/usr/bin/docker",
            "run",
            "--detach",
            "--name",
            stack.database_name,
            "--publish",
            f"127.0.0.1:{stack.database_port}:5432",
            "--env",
            f"POSTGRES_PASSWORD={stack.database_password}",
            "--env",
            "POSTGRES_DB=gods_watching_test",
            _DATABASE_IMAGE,
        )
    )
    if _safe_returncode(result) != 0:
        _fail("Task 12 PostgreSQL container failed to start")
    stack.database_started = True
    stack.event(stack.database_name, "started")
    await _wait_postgres(stack.database_name)
    environment = os.environ.copy()
    environment["GW_DATABASE_URL"] = stack.database_url
    migration = await _command(
        ("uv", "run", "--project", str(_REPOSITORY_ROOT), "alembic", "upgrade", "head"),
        env=environment,
    )
    if _safe_returncode(migration) != 0:
        _fail("Task 12 database migration failed")


async def _start_media(stack: _Stack) -> None:
    _render_media_config(
        stack.config_path,
        reader_password=stack.reader_password,
        publisher_password=stack.publisher_password,
        control_password=stack.control_password,
        ports=(stack.rtsp_port, stack.whep_port, stack.control_port, stack.udp_port),
    )
    stack.event(stack.media_name, "registered")
    environment = os.environ.copy()
    environment.update(
        {
            "MTX_RTSPADDRESS": f":{stack.rtsp_port}",
            "MTX_WEBRTCADDRESS": f":{stack.whep_port}",
            "MTX_APIADDRESS": f":{stack.control_port}",
            "MTX_WEBRTCLOCALUDPADDRESS": f":{stack.udp_port}",
            "MTX_WEBRTCADDITIONALHOSTS": "127.0.0.1",
        }
    )
    stack.media_process = await anyio.open_process(
        (
            "/usr/bin/docker",
            "run",
            "--rm",
            "--name",
            stack.media_name,
            "--network",
            "host",
            "--read-only",
            "--security-opt",
            "no-new-privileges:true",
            "--cap-drop",
            "ALL",
            "--volume",
            f"{stack.config_path}:/mediamtx.yml:ro",
            _MEDIA_IMAGE,
        ),
        env=environment,
        stdout=stack.app_log,
        stderr=stack.app_log,
    )
    stack.event(stack.media_name, "started")
    await _wait_tcp(stack.rtsp_port)
    await _wait_tcp(stack.whep_port)
    await _wait_tcp(stack.control_port)


def application_uvicorn_command(
    *,
    repository_root: Path,
    port: int,
    keyfile: Path,
    certfile: Path,
) -> tuple[str, ...]:
    """Build the FastAPI launch command without Uvicorn client-IP rewriting."""
    return (
        "uv",
        "run",
        "--project",
        str(repository_root),
        "uvicorn",
        "gods_watching.verification.scenarios.task12_app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
        "--no-proxy-headers",
        "--ssl-keyfile",
        str(keyfile),
        "--ssl-certfile",
        str(certfile),
    )


async def _start_application(stack: _Stack, mode: str) -> _AppState:
    await _prepare_tls(stack)
    stack.publisher = await _start_publisher(
        source=stack.media_path,
        password=stack.publisher_password,
        rtsp_port=stack.rtsp_port,
    )
    stack.event("ffmpeg-publisher", "started")
    if not await _probe(stack.source_url, expected=True):
        _fail("Task 12 publisher did not become readable")
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(_REPOSITORY_ROOT / "server/src"),
            "GW_DATABASE_URL": stack.database_url,
            "GW_TRITON_GRPC_URL": os.environ.get("GW_TRITON_GRPC_URL", "127.0.0.1:8001"),
            "GW_CROPS_ROOT": str(stack.run_root / "crops"),
            "GW_OPERATOR_PASSWORD": stack.operator_password,
            "GW_CAMERA_CIPHER_KEY": stack.cipher_key,
            "GW_TASK12_SOURCE_URL": stack.source_url,
            "GW_TASK12_PUBLIC_ORIGIN": f"https://127.0.0.1:{stack.app_port}",
            "GW_TASK12_SECURE_COOKIE": "true",
            "GW_MEDIA_CONTROL_PORT": str(stack.control_port),
            "GW_MEDIA_CONTROL_USER": _CONTROL_USER,
            "GW_MEDIA_CONTROL_PASSWORD": stack.control_password,
            "GW_MEDIA_WHEP_PORT": str(stack.whep_port),
            "GW_MEDIA_READER_USER": _READER_USER,
            "GW_MEDIA_READER_PASSWORD": stack.reader_password,
        }
    )
    stack.app_process = await anyio.open_process(
        application_uvicorn_command(
            repository_root=_REPOSITORY_ROOT,
            port=stack.app_port,
            keyfile=stack.tls_key_path,
            certfile=stack.tls_cert_path,
        ),
        env=environment,
        stdout=stack.app_log,
        stderr=stack.app_log,
    )
    stack.event("fastapi-app", "started")
    client = _HttpClient(
        stack.app_port,
        f"https://127.0.0.1:{stack.app_port}",
        stack.ca_bundle_path,
    )
    await _wait_http(client, stack.app_process)
    login = await to_thread.run_sync(
        partial(client.request, "POST", "/api/session", {"password": stack.operator_password})
    )
    if login.status != _HTTP_OK:
        _fail("Task 12 app login failed")
    camera_rows = await to_thread.run_sync(partial(client.request, "GET", "/api/cameras"))
    rows = _body_list(camera_rows)
    if not rows or not isinstance(rows[0], dict):
        _fail("Task 12 fixture camera was not returned")
    camera_id_value = rows[0].get("camera_id")
    if not isinstance(camera_id_value, str):
        _fail("Task 12 fixture camera id was not returned")
    camera_id = camera_id_value
    browser_output = stack.run_root / "task12-browser.json"
    await _run_browser(
        _BrowserConfiguration(
            origin=f"https://127.0.0.1:{stack.app_port}",
            password=stack.operator_password,
            camera_id=camera_id,
            output=browser_output,
            mode=mode,
            control_port=stack.control_port,
            control_password=stack.control_password,
            media_path=f"camera/{camera_id}",
            ca_bundle=stack.ca_bundle_path,
            profile=stack.browser_profile,
        )
    )
    return _AppState(
        client=client,
        camera_id=camera_id,
        browser=BrowserArtifact.model_validate_json(browser_output.read_text(encoding="utf-8")),
    )


async def _exercise(stack: _Stack, state: _AppState, mode: str) -> JsonObject:
    checks: dict[str, bool] = {
        "real_postgres_migration": True,
        "real_h264_source": True,
        "browser_driver": True,
    }
    if mode == "happy":
        checks["session_cookie_flags"] = _browser_cookie_flags(state.browser)
    observations: JsonObject = {"browser": state.browser.model_dump(mode="json")}
    if mode == "happy":
        stack.event("camera-checks", "started")
        camera_checks, camera_observations = await _camera_checks(
            state.client,
            source_url=stack.source_url,
            reader_password=stack.reader_password,
            rtsp_port=stack.rtsp_port,
        )
        stack.event("camera-checks", "completed")
        stack.event("credential-checks", "started")
        credential_checks, credential_observations = await _credential_checks(
            state.client,
            _CredentialConfiguration(
                run_root=stack.run_root,
                database_url=stack.database_url,
                replacement_password=stack.replacement_password,
                rollback_password=stack.rollback_password,
                app_port=stack.app_port,
            ),
        )
        stack.event("credential-checks", "completed")
        checks.update(camera_checks)
        checks.update(credential_checks)
        observations.update(camera_observations)
        observations.update(credential_observations)
    else:
        api_observations = await _denied_api_checks(state.client)
        checks.update(_denied_checks(state.browser, api_observations))
        observations["api"] = api_observations
    check_values: JsonObject = {}
    for key, value in checks.items():
        check_values[key] = value
    return {"mode": mode, "checks": check_values, "observations": observations}


async def _cleanup(stack: _Stack) -> None:
    with anyio.move_on_after(30, shield=True):
        await _stop(stack.app_process)
        await _stop(stack.publisher)
        await _stop(stack.media_process)
        if stack.media_process is not None:
            await _remove(stack.media_name)
        if stack.database_started:
            await _remove(stack.database_name)
        stack.app_log.close()
        shutil.rmtree(stack.run_root / "secrets", ignore_errors=True)
        shutil.rmtree(stack.runtime_root, ignore_errors=True)
        stack.runtime_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        stack.event("fastapi-app", "cleaned")
        stack.event("ffmpeg-publisher", "cleaned")
        stack.event(stack.media_name, "cleaned")
        stack.event(stack.database_name, "cleaned")


CONTROL_USER = _CONTROL_USER
HttpClient = _HttpClient
PUBLISHER_USER = _PUBLISHER_USER
READER_USER = _READER_USER
SOURCE = _SOURCE
Stack = _Stack
body_dict = _body_dict
body_list = _body_list
cleanup = _cleanup
command = _command
prepare_browser_nss = _prepare_browser_nss
prepare_stack = _prepare_stack
probe = _probe
safe_returncode = _safe_returncode
start_application = _start_application
start_database = _start_database
start_media = _start_media
stop = _stop
write_json = _write_json


async def _run(mode: str) -> JsonObject:
    stack = _prepare_stack()
    try:
        await _start_database(stack)
        await _start_media(stack)
        state = await _start_application(stack, mode)
        result = await _exercise(stack, state, mode)
        _write_json(stack.output, result)
        return result
    finally:
        await _cleanup(stack)


async def _cancel_on_signal(
    signals: AsyncIterator[signal.Signals],
    run_scope: anyio.CancelScope,
) -> None:
    """Cancel the active stack run after the first external interruption."""
    async for _signum in signals:
        run_scope.cancel()
        return


async def _run_with_signal_cleanup(mode: str) -> JsonObject | None:
    """Run a mode while routing SIGINT/SIGTERM through stack cleanup."""
    result: JsonObject | None = None
    with anyio.CancelScope() as run_scope:
        with anyio.open_signal_receiver(signal.SIGINT, signal.SIGTERM) as signals:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(_cancel_on_signal, signals, run_scope)
                try:
                    result = await _run(mode)
                finally:
                    tasks.cancel_scope.cancel()
    return result


async def main() -> None:
    """Run one isolated Task 12 driver mode from environment configuration."""
    mode = _required("GW_TASK12_MODE")
    if mode not in {"happy", "denied"}:
        _fail("Task 12 mode is invalid")
    _ = await _run_with_signal_cleanup(mode)


if __name__ == "__main__":
    anyio.run(main)
