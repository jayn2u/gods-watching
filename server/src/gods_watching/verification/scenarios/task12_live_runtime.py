"""Drive the real Task 12 authorization and lifecycle boundaries."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import signal
import subprocess
from dataclasses import dataclass
from functools import partial
from http.client import HTTPConnection
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, NoReturn, Protocol

import anyio
from anyio import to_thread
from pydantic import TypeAdapter, ValidationError

from . import task12_driver as base
from .task12_models import JsonObject, JsonValue

if TYPE_CHECKING:
    from io import BufferedWriter

    from anyio.abc import Process

_REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[5]
_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_STATUS: Final[re.Pattern[str]] = re.compile(r"HTTP/\d(?:\.\d)?\s+(\d{3})")
_BWRAP_TMPFS: Final[str] = str(Path(os.sep, "tmp"))
_COOKIE_FIELDS: Final[int] = 7
_HTTP_FORBIDDEN: Final[int] = 403
_HTTP_NOT_FOUND: Final[int] = 404
_HTTP_NO_CONTENT: Final[int] = 204
_HTTP_OK: Final[int] = 200
_HTTP_UNAUTHORIZED: Final[int] = 401
_HTTP_UNPROCESSABLE: Final[int] = 422
_MAX_EXPIRY_ELAPSED_MS: Final[int] = 5000
type ExpiryColumn = Literal["idle_expires_at", "absolute_expires_at"]
_EXPIRY_SQL: Final[dict[ExpiryColumn, str]] = {
    "idle_expires_at": (
        "UPDATE sessions SET idle_expires_at = NOW() - INTERVAL '1 second' "
        "WHERE token_hash = decode(:'token_digest', 'hex');"
    ),
    "absolute_expires_at": (
        "UPDATE sessions SET absolute_expires_at = NOW() - INTERVAL '1 second' "
        "WHERE token_hash = decode(:'token_digest', 'hex');"
    ),
}


class _LiveBoundaryError(RuntimeError):
    pass


def _fail(message: str, *, cause: BaseException | None = None) -> NoReturn:
    if cause is None:
        raise _LiveBoundaryError(message)
    raise _LiveBoundaryError(message) from cause


class _DriverCommandContext(Protocol):
    @property
    def run_root(self) -> Path: ...

    @property
    def compose_project(self) -> str: ...

    @property
    def allocated_port(self) -> int: ...


def driver_command(
    context: _DriverCommandContext,
    *,
    output: Path,
) -> tuple[str, ...]:
    """Build the secret-free command used by the installed scenario runner."""
    return (
        "/usr/bin/env",
        "GW_TASK12_MODE=all",
        f"GW_TASK12_RUN_ROOT={context.run_root / 'live-boundaries'}",
        f"GW_TASK12_OUTPUT={output}",
        f"GW_TASK12_PROJECT={context.compose_project}",
        f"GW_TASK12_APP_PORT={context.allocated_port}",
        "uv",
        "run",
        "--project",
        str(_REPOSITORY_ROOT),
        "python",
        "-m",
        "gods_watching.verification.scenarios.task12_live_runtime",
    )


def _read_object(path: Path) -> JsonObject:
    try:
        value = _JSON.validate_json(path.read_bytes())
    except (ValidationError, ValueError) as error:
        _fail(f"invalid live-boundary artifact: {path.name}", cause=error)
    if not isinstance(value, dict):
        _fail(f"live-boundary artifact is not an object: {path.name}")
    return value


async def _wait_object(path: Path, deadline_seconds: float = 120.0) -> JsonObject:
    with anyio.fail_after(deadline_seconds):
        while True:
            try:
                return _read_object(path)
            except (OSError, RuntimeError):
                await anyio.sleep(0.05)


def _write_secret(path: Path, value: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _ = path.write_text(value + "\n", encoding="utf-8")
    _ = path.chmod(0o600)


def _cookie_header(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= _COOKIE_FIELDS and fields[-2] == "gw_session":
            return f"gw_session={fields[-1]}"
        if line.startswith("gw_session="):
            return line
    _fail(f"browser cookie artifact is missing gw_session: {path.name}")


@dataclass(frozen=True, slots=True)
class _BrowserCommandConfiguration:
    stack: base.Stack
    password_file: Path
    camera_id: str
    output: Path
    mode: str
    profile_a: Path
    profile_b: Path


def _client_for_cookie(stack: base.Stack, cookie: Path) -> base.HttpClient:
    client = base.HttpClient(
        stack.app_port,
        f"https://127.0.0.1:{stack.app_port}",
        stack.ca_bundle_path,
    )
    client.cookie = _cookie_header(cookie)
    return client


def _browser_command(configuration: _BrowserCommandConfiguration) -> tuple[str, ...]:
    stack = configuration.stack
    script = str(Path(__file__).with_name("task12_live_browser.mjs"))
    return (
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
        str(stack.ca_bundle_path),
        "/etc/ssl/certs/ca-certificates.crt",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        _BWRAP_TMPFS,
        "--ro-bind",
        str(_REPOSITORY_ROOT),
        str(_REPOSITORY_ROOT),
        "--bind",
        str(configuration.profile_a),
        str(configuration.profile_a),
        "--bind",
        str(configuration.profile_b),
        str(configuration.profile_b),
        "--bind",
        str(configuration.output.parent),
        str(configuration.output.parent),
        "--chdir",
        str(_REPOSITORY_ROOT),
        "--setenv",
        "HOME",
        str(configuration.profile_a),
        "/usr/bin/node",
        script,
        f"https://127.0.0.1:{stack.app_port}",
        str(configuration.password_file),
        configuration.camera_id,
        str(configuration.output),
        configuration.mode,
        str(configuration.profile_a),
        str(configuration.profile_b),
    )


async def _open_browser(
    stack: base.Stack,
    *,
    password_file: Path,
    camera_id: str,
    output: Path,
    mode: str,
) -> tuple[Process, BufferedWriter]:
    profile_a = stack.runtime_root / f"{mode}-profile-a"
    profile_b = stack.runtime_root / f"{mode}-profile-b"
    profile_a.mkdir(mode=0o700, parents=True, exist_ok=True)
    profile_b.mkdir(mode=0o700, parents=True, exist_ok=True)
    base.prepare_browser_nss(profile_a, stack.runtime_root / "ca.crt")
    base.prepare_browser_nss(profile_b, stack.runtime_root / "ca.crt")
    command = _browser_command(
        _BrowserCommandConfiguration(
            stack=stack,
            password_file=password_file,
            camera_id=camera_id,
            output=output,
            mode=mode,
            profile_a=profile_a,
            profile_b=profile_b,
        )
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PLAYWRIGHT_BROWSERS_PATH": str(_REPOSITORY_ROOT / "runtime/playwright-browsers"),
            "GW_TASK12_CONTROL_PORT": str(stack.control_port),
            "GW_TASK12_CONTROL_USER": base.CONTROL_USER,
            "GW_TASK12_CONTROL_PASSWORD": stack.control_password,
            "GW_TASK12_MEDIA_PATH": f"camera/{camera_id}",
            "HOME": str(profile_a),
            "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
            "SSL_CERT_DIR": "/etc/ssl/certs",
        }
    )
    log_path = output.with_name(f"task12-live-{mode}.log")
    log = log_path.open("wb")
    stack.event(f"live-browser-{mode}", "started")
    try:
        process = await anyio.open_process(command, env=environment, stdout=log, stderr=log)
    except OSError:
        log.close()
        stack.event(f"live-browser-{mode}", "cleaned")
        raise
    return process, log


async def _finish_browser(
    stack: base.Stack,
    process: Process,
    log: BufferedWriter,
    mode: str,
) -> None:
    try:
        return_code = await process.wait()
        if return_code != 0:
            _fail(f"Task 12 live browser failed in {mode} mode")
    finally:
        await base.stop(process)
        log.close()
        stack.event(f"live-browser-{mode}", "cleaned")


def _reader_count_sync(stack: base.Stack, camera_id: str) -> int:
    authorization = base64.b64encode(
        f"{base.CONTROL_USER}:{stack.control_password}".encode()
    ).decode("ascii")
    connection = HTTPConnection("127.0.0.1", stack.control_port, timeout=5)
    status = 0
    try:
        connection.request(
            "GET",
            f"/v3/paths/get/camera/{camera_id}",
            headers={"Authorization": f"Basic {authorization}"},
        )
        response = connection.getresponse()
        status = response.status
        body = response.read(1_000_000)
    finally:
        connection.close()
    if status != _HTTP_OK:
        return 0
    try:
        value = _JSON.validate_json(body)
    except (ValidationError, ValueError):
        return 0
    if not isinstance(value, dict):
        return 0
    readers = value.get("readers")
    if not isinstance(readers, list):
        return 0
    return len(readers)


async def _reader_count(stack: base.Stack, camera_id: str) -> int:
    return await to_thread.run_sync(partial(_reader_count_sync, stack, camera_id))


async def _wait_readers(
    stack: base.Stack,
    camera_id: str,
    expected: int,
    deadline_seconds: float = 15.0,
) -> int:
    with anyio.fail_after(deadline_seconds):
        count = await _reader_count(stack, camera_id)
        while count != expected:
            await anyio.sleep(0.1)
            count = await _reader_count(stack, camera_id)
    return count


def _redact_http(value: str) -> str:
    return re.sub(r"gw_session=[^;\r\n]*", "gw_session=<redacted>", value)


@dataclass(frozen=True, slots=True)
class _CurlConfiguration:
    stack: base.Stack
    method: str
    path: str
    cookie: Path
    output: Path
    headers: tuple[str, ...] = ()


async def _curl(configuration: _CurlConfiguration) -> int:
    stack = configuration.stack
    command = [
        "/usr/bin/curl",
        "--silent",
        "--show-error",
        "--include",
        "--cacert",
        str(stack.ca_bundle_path),
        "--cookie",
        str(configuration.cookie),
        "-X",
        configuration.method,
        "-H",
        f"Origin: https://127.0.0.1:{stack.app_port}",
    ]
    for header in configuration.headers:
        command.extend(("-H", header))
    command.append(f"https://127.0.0.1:{stack.app_port}{configuration.path}")
    result = await base.command(tuple(command))
    text = _redact_http(result.stdout.decode("utf-8", errors="replace"))
    _ = configuration.output.write_text(text, encoding="utf-8")
    statuses = _STATUS.findall(text)
    status = statuses[-1] if statuses else None
    return int(status) if isinstance(status, str) else 0


async def _start_unsupported_publisher(stack: base.Stack) -> tuple[Process, str]:
    path = f"test-publisher/unsupported-{os.getpid()}"
    publisher_source = (
        f"rtsp://{base.PUBLISHER_USER}:{stack.publisher_password}"
        f"@127.0.0.1:{stack.rtsp_port}/{path}"
    )
    reader_source = (
        f"rtsp://{base.READER_USER}:{stack.reader_password}@127.0.0.1:{stack.rtsp_port}/{path}"
    )
    process = await anyio.open_process(
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
            str(base.SOURCE),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            publisher_source,
        ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    stack.event("ffmpeg-unsupported-publisher", "started")
    try:
        readable = await base.probe(reader_source, expected=True)
    except BaseException:
        await base.stop(process)
        stack.event("ffmpeg-unsupported-publisher", "cleaned")
        raise
    if not readable:
        await base.stop(process)
        stack.event("ffmpeg-unsupported-publisher", "cleaned")
        _fail("unsupported codec fixture did not become readable")
    return process, reader_source


async def _unsupported_phase(stack: base.Stack, client: base.HttpClient) -> JsonObject:
    process, source = await _start_unsupported_publisher(stack)
    try:
        response = await to_thread.run_sync(
            partial(client.request, "POST", "/api/cameras/test", {"source_url": source})
        )
        cameras = await to_thread.run_sync(partial(client.request, "GET", "/api/cameras"))
    finally:
        await base.stop(process)
        stack.event("ffmpeg-unsupported-publisher", "cleaned")
    detail = base.body_dict(response).get("detail")
    code = detail.get("code") if isinstance(detail, dict) else None
    rows = base.body_list(cameras)
    preserved = any(
        isinstance(row, dict)
        and row.get("source_port") == stack.rtsp_port
        and "source_url" not in row
        for row in rows
    )
    response_text = json.dumps(response.body, sort_keys=True)
    credentials_redacted = (
        stack.reader_password not in response_text and "rtsp://" not in response_text
    )
    return {
        "status": response.status,
        "code": code if isinstance(code, str) else None,
        "config_preserved": preserved,
        "credentials_redacted": credentials_redacted,
    }


async def _credential_command(stack: base.Stack, password_file: Path) -> tuple[int, JsonObject]:
    environment = os.environ.copy()
    environment["GW_DATABASE_URL"] = stack.database_url
    result = await base.command(
        (
            str(_REPOSITORY_ROOT / "gods-watching"),
            "credentials",
            "set",
            "--password-file",
            str(password_file),
        ),
        env=environment,
    )
    try:
        value = _JSON.validate_json(result.stdout)
    except (ValidationError, ValueError):
        value = None
    return_code = base.safe_returncode(result)
    return return_code, value if isinstance(value, dict) else {}


async def _credential_phase(
    stack: base.Stack,
    camera_id: str,
    old_password_file: Path,
    replacement_file: Path,
) -> JsonObject:
    output = stack.run_root / "credential.json"
    process, log = await _open_browser(
        stack,
        password_file=old_password_file,
        camera_id=camera_id,
        output=output,
        mode="credential",
    )
    try:
        ready_path = stack.run_root / "boundary-credential-replacement-ready.json"
        ready = await _wait_object(ready_path)
        replacement_exit, replacement = await _credential_command(stack, replacement_file)
        if replacement_exit != 0:
            _fail("credential replacement command failed")
        _write_secret(old_password_file, stack.replacement_password)
        old_values = [ready.get("cookieA"), ready.get("cookieB")]
        if not all(isinstance(value, str) for value in old_values):
            _fail("credential boundary did not emit old cookie paths")
        old_cookies = [Path(value) for value in old_values if isinstance(value, str)]
        old_statuses: list[bool] = []
        for cookie in old_cookies:
            old_client = _client_for_cookie(stack, cookie)
            session = await to_thread.run_sync(partial(old_client.request, "GET", "/api/session"))
            old_statuses.append(base.body_dict(session).get("authenticated") is False)
        readers_after_replacement = await _wait_readers(stack, camera_id, 0)
        _ = (stack.run_root / "boundary-credential-replacement-go.json").write_text("{}\n")
        _ = (stack.run_root / "boundary-credential-noop-go.json").write_text("{}\n")
        _ = await _wait_object(stack.run_root / "boundary-credential-noop-ready.json")
        noop_exit, noop = await _credential_command(stack, replacement_file)
        readers_after_noop = await _reader_count(stack, camera_id)
        _ = (stack.run_root / "boundary-credential-finish-go.json").write_text("{}\n")
        await _finish_browser(stack, process, log, "credential")
        _ = await _wait_readers(stack, camera_id, 0)
    except BaseException:
        await base.stop(process)
        log.close()
        raise
    browser = _read_object(output)
    return {
        "browser": browser,
        "replacement": replacement,
        "replacement_exit": replacement_exit,
        "old_sessions_denied": all(old_statuses),
        "readers_after_replacement": readers_after_replacement,
        "noop": noop,
        "noop_exit": noop_exit,
        "readers_after_noop": readers_after_noop,
    }


async def _expire_token(stack: base.Stack, cookie: Path, column: ExpiryColumn) -> int:
    token = _cookie_header(cookie).split("=", 1)[1]
    digest = hashlib.sha256(token.encode()).hexdigest()
    sql = _EXPIRY_SQL[column]
    result = await anyio.run_process(
        (
            "/usr/bin/docker",
            "exec",
            "-i",
            stack.database_name,
            "psql",
            "-U",
            "postgres",
            "-d",
            "gods_watching_test",
            "-v",
            "ON_ERROR_STOP=1",
            "-v",
            f"token_digest={digest}",
            "-f",
            "-",
        ),
        input=(sql + "\n").encode(),
        check=False,
    )
    return base.safe_returncode(result)


async def _expiry_phase(stack: base.Stack, camera_id: str, password_file: Path) -> JsonObject:
    output = stack.run_root / "expiry.json"
    process, log = await _open_browser(
        stack,
        password_file=password_file,
        camera_id=camera_id,
        output=output,
        mode="expiry",
    )
    try:
        idle_ready = await _wait_object(stack.run_root / "boundary-expiry-idle-ready.json")
        idle_value = idle_ready.get("cookie")
        if not isinstance(idle_value, str):
            _fail("idle expiry boundary did not emit a cookie path")
        idle_cookie = Path(idle_value)
        if await _expire_token(stack, idle_cookie, "idle_expires_at") != 0:
            _fail("idle expiry database fixture failed")
        _ = (stack.run_root / "boundary-expiry-idle-go.json").write_text("{}\n")
        idle_zero = await _wait_object(stack.run_root / "boundary-expiry-idle-zero.json")
        idle_status = await _curl(
            _CurlConfiguration(
                stack=stack,
                method="GET",
                path="/api/cameras",
                cookie=idle_cookie,
                output=stack.run_root / "boundary-expiry-idle-protected-curl.txt",
            )
        )
        if idle_status != _HTTP_UNAUTHORIZED:
            _fail("idle expiry protected request was not unauthorized")
        _ = (stack.run_root / "boundary-expiry-idle-verified.json").write_text(
            json.dumps({"protected_curl_status": idle_status}) + "\n",
            encoding="utf-8",
        )

        absolute_ready = await _wait_object(stack.run_root / "boundary-expiry-absolute-ready.json")
        absolute_value = absolute_ready.get("cookie")
        if not isinstance(absolute_value, str):
            _fail("absolute expiry boundary did not emit a cookie path")
        absolute_cookie = Path(absolute_value)
        if await _expire_token(stack, absolute_cookie, "absolute_expires_at") != 0:
            _fail("absolute expiry database fixture failed")
        _ = (stack.run_root / "boundary-expiry-absolute-go.json").write_text("{}\n")
        absolute_zero = await _wait_object(stack.run_root / "boundary-expiry-absolute-zero.json")
        absolute_status = await _curl(
            _CurlConfiguration(
                stack=stack,
                method="GET",
                path="/api/cameras",
                cookie=absolute_cookie,
                output=stack.run_root / "boundary-expiry-absolute-protected-curl.txt",
            )
        )
        if absolute_status != _HTTP_UNAUTHORIZED:
            _fail("absolute expiry protected request was not unauthorized")
        _ = (stack.run_root / "boundary-expiry-absolute-verified.json").write_text(
            json.dumps({"protected_curl_status": absolute_status}) + "\n",
            encoding="utf-8",
        )
        await _finish_browser(stack, process, log, "expiry")
        _ = await _wait_readers(stack, camera_id, 0)
    except BaseException:
        await base.stop(process)
        log.close()
        raise
    browser = _read_object(output)
    browser["idle_zero_marker"] = idle_zero
    browser["absolute_zero_marker"] = absolute_zero
    browser["idle_protected_curl_status"] = idle_status
    browser["absolute_protected_curl_status"] = absolute_status
    return browser


async def _delete_phase(stack: base.Stack, camera_id: str, password_file: Path) -> JsonObject:
    output = stack.run_root / "delete.json"
    process, log = await _open_browser(
        stack,
        password_file=password_file,
        camera_id=camera_id,
        output=output,
        mode="delete",
    )
    try:
        ready = await _wait_object(stack.run_root / "boundary-delete-ready.json")
        cookie_value = ready.get("cookie")
        if not isinstance(cookie_value, str):
            _fail("delete boundary did not emit a cookie path")
        cookie = Path(cookie_value)
        client = _client_for_cookie(stack, cookie)
        cameras = await to_thread.run_sync(partial(client.request, "GET", "/api/cameras"))
        rows = base.body_list(cameras)
        version = next(
            row.get("version")
            for row in rows
            if isinstance(row, dict) and row.get("camera_id") == camera_id
        )
        if not isinstance(version, int):
            _fail("camera version missing before live delete")
        delete_status = await _curl(
            _CurlConfiguration(
                stack=stack,
                method="DELETE",
                path=f"/api/cameras/{camera_id}",
                cookie=cookie,
                output=stack.run_root / "boundary-delete-curl.txt",
                headers=(f"X-Camera-Version: {version}",),
            )
        )
        _ = (stack.run_root / "boundary-delete-go.json").write_text("{}\n")
        await _finish_browser(stack, process, log, "delete")
    except BaseException:
        await base.stop(process)
        log.close()
        raise
    browser = _read_object(output)
    browser["delete_status"] = delete_status
    return browser


async def _run_browser_phase(
    stack: base.Stack,
    camera_id: str,
    password_file: Path,
    mode: str,
) -> JsonObject:
    output = stack.run_root / f"{mode}.json"
    process, log = await _open_browser(
        stack,
        password_file=password_file,
        camera_id=camera_id,
        output=output,
        mode=mode,
    )
    try:
        await _finish_browser(stack, process, log, mode)
    except BaseException:
        await base.stop(process)
        log.close()
        raise
    return _read_object(output)


def _expiry_check(value: JsonValue | None, key: str) -> bool:
    return isinstance(value, dict) and value.get(key) is True


def _expiry_zero_check(value: JsonValue | None, key: str) -> bool:
    return isinstance(value, dict) and value.get(key) == 0


def _expiry_elapsed_check(value: JsonValue | None, key: str) -> bool:
    elapsed = value.get(key) if isinstance(value, dict) else None
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        return False
    return 0 <= elapsed <= _MAX_EXPIRY_ELAPSED_MS


def _checks(
    patch: JsonObject,
    unsupported: JsonObject,
    credential: JsonObject,
    expiry: JsonObject,
    delete: JsonObject,
) -> dict[str, bool]:
    replacement = credential.get("replacement")
    noop = credential.get("noop")
    idle_expiry = expiry.get("idle_zero_marker")
    absolute_expiry = expiry.get("absolute_zero_marker")
    frames_after = patch.get("frames_after_patch")
    return {
        "whep_patch_authorized_204": patch.get("own_patch") == _HTTP_NO_CONTENT,
        "whep_patch_unauthorized_denied": patch.get("anonymous_patch")
        in {_HTTP_UNAUTHORIZED, _HTTP_FORBIDDEN},
        "whep_patch_cross_session_denied": patch.get("cross_session_patch")
        in {_HTTP_UNAUTHORIZED, _HTTP_FORBIDDEN, _HTTP_NOT_FOUND},
        "whep_media_remains_after_patch": (
            patch.get("own_patch") == _HTTP_NO_CONTENT
            and patch.get("readers_during") == 1
            and patch.get("frames_advanced_after_patch") is True
            and isinstance(frames_after, int)
            and frames_after > 0
        ),
        "unsupported_codec_rejected": unsupported.get("status") == _HTTP_UNPROCESSABLE
        and unsupported.get("code") == "unsupported_codec",
        "unsupported_codec_preserves_config": unsupported.get("config_preserved") is True,
        "unsupported_codec_redacts_credentials": unsupported.get("credentials_redacted") is True,
        "credential_replacement_changed": (
            credential.get("replacement_exit") == 0
            and isinstance(replacement, dict)
            and replacement.get("changed") is True
        ),
        "credential_replacement_denies_old_sessions": credential.get("old_sessions_denied") is True,
        "credential_replacement_closes_readers": credential.get("readers_after_replacement") == 0,
        "same_password_noop": credential.get("noop_exit") == 0
        and isinstance(noop, dict)
        and noop.get("changed") is False,
        "same_password_preserves_reader": credential.get("readers_after_noop") == 1,
        "expiry_readers_zero": (
            _expiry_zero_check(idle_expiry, "readers_zero")
            and _expiry_zero_check(absolute_expiry, "readers_zero")
        ),
        "autonomous_idle_expiry": _expiry_elapsed_check(idle_expiry, "idle_elapsed_ms"),
        "autonomous_absolute_expiry": _expiry_elapsed_check(absolute_expiry, "absolute_elapsed_ms"),
        "expiry_frames_stop": (
            _expiry_check(idle_expiry, "frames_stopped")
            and _expiry_check(absolute_expiry, "frames_stopped")
        ),
        "expiry_protected_curl_401": (
            expiry.get("idle_protected_curl_status") == _HTTP_UNAUTHORIZED
            and expiry.get("absolute_protected_curl_status") == _HTTP_UNAUTHORIZED
        ),
        "expiry_no_app_requests_until_zero": (
            _expiry_check(idle_expiry, "no_app_requests")
            and _expiry_check(absolute_expiry, "no_app_requests")
        ),
        "camera_delete_closes_reader": delete.get("delete_status") == _HTTP_NO_CONTENT
        and delete.get("readers_after_delete") == 0,
        "camera_delete_hides_camera": delete.get("cameras_after_delete") == 0,
        "camera_delete_no_resurrection": delete.get("resurrection_status")
        in {_HTTP_UNAUTHORIZED, _HTTP_NOT_FOUND},
    }


def _write_cleanup_receipt(stack: base.Stack) -> None:
    _ = (stack.run_root / "cleanup-receipt.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "compose_project": stack.compose_project,
                "containers": [stack.database_name, stack.media_name],
                "resource_manifest": str(stack.manifest_path),
                "runtime_root_entries": sorted(path.name for path in stack.runtime_root.iterdir()),
                "secret_directory_absent": not (stack.run_root / "secrets").exists(),
                "note": (
                    "Task 12 live-boundary resources were cleaned by the owned "
                    "driver finally block."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


async def _run_live() -> None:
    stack = base.prepare_stack()
    try:
        await base.start_database(stack)
        await base.start_media(stack)
        state = await base.start_application(stack, "happy")
        secret_file = stack.run_root / "secrets/operator.secret"
        replacement_file = stack.run_root / "secrets/replacement.secret"
        _write_secret(secret_file, stack.operator_password)
        _write_secret(replacement_file, stack.replacement_password)
        patch = await _run_browser_phase(stack, state.camera_id, secret_file, "patch")
        unsupported = await _unsupported_phase(stack, state.client)
        credential = await _credential_phase(stack, state.camera_id, secret_file, replacement_file)
        _write_secret(secret_file, stack.replacement_password)
        expiry = await _expiry_phase(stack, state.camera_id, secret_file)
        delete = await _delete_phase(stack, state.camera_id, secret_file)
        result: JsonObject = {
            "mode": "all",
            "observations": {
                "patch": patch,
                "unsupported_codec": unsupported,
                "credential": credential,
                "expiry": expiry,
                "delete": delete,
            },
        }
        check_values: JsonObject = {}
        for name, passed in _checks(patch, unsupported, credential, expiry, delete).items():
            check_values[name] = passed
        result["checks"] = check_values
        base.write_json(stack.output, result)
    finally:
        try:
            await base.cleanup(stack)
        finally:
            _write_cleanup_receipt(stack)


async def _main() -> None:
    with anyio.CancelScope() as scope:
        with anyio.open_signal_receiver(signal.SIGINT, signal.SIGTERM) as signals:
            async with anyio.create_task_group() as tasks:

                async def stop_on_signal() -> None:
                    async for _signal in signals:
                        scope.cancel()
                        return

                tasks.start_soon(stop_on_signal)
                try:
                    await _run_live()
                finally:
                    tasks.cancel_scope.cancel()


if __name__ == "__main__":
    anyio.run(_main)
