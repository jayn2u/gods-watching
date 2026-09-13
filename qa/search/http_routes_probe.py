"""Executable seed and HTTP probe for the task 14b live QA run."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast
from urllib.parse import urlsplit
from uuid import UUID

import anyio
from cryptography.fernet import Fernet

from gods_watching.auth import AuthService
from gods_watching.contracts.appearances import AppearancePublication, BoundingBox
from gods_watching.contracts.cameras import CameraCreateRequest
from gods_watching.contracts.identifiers import AppearanceId, CameraId, CameraSessionId
from gods_watching.search.service import DEFAULT_CLIP_MODEL_REVISION
from gods_watching.storage import CredentialCipher, CropObjectStore, Database, StorageRepository

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import AnyUrl


_BASE_URL: Final[str] = "http://127.0.0.1:18181"
_JPEG_1X1: Final[bytes] = base64.b64decode(
    b"/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAH/AP/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAT8Af//EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQIBAT8Af//EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQMBAT8Af//Z"
)
_SEED_ID = UUID("11111111-1111-4111-8111-111111111111")
_NEAREST_ID = UUID("22222222-2222-4222-8222-222222222222")
_OBSERVED_AT = datetime(2026, 9, 8, 8, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class _PublicationSpec:
    appearance_id: UUID
    camera_id: UUID
    session_id: UUID
    track_id: int
    first_seen: datetime
    embedding: tuple[float, ...]
    crop_object_key: str


@dataclass(frozen=True, slots=True)
class _RequestSpec:
    base_url: str
    path: str
    method: str = "GET"
    payload: Mapping[str, object] | None = None
    cookie: str | None = None
    origin: str | None = None


def _vector(*coordinates: tuple[int, float]) -> tuple[float, ...]:
    values = [0.0] * 512
    norm = math.sqrt(sum(value * value for _, value in coordinates))
    for index, value in coordinates:
        values[index] = value / norm
    return tuple(values)


def _source(name: str) -> AnyUrl:
    return CameraCreateRequest.model_validate(
        {"name": name, "source_url": f"rtsp://fixture:8554/{name.lower()}"}
    ).source_url


def _password() -> str:
    path = os.environ.get("GW_QA_PASSWORD_FILE")
    if not path:
        message = "GW_QA_PASSWORD_FILE is required for the HTTP QA probe"
        raise RuntimeError(message)
    return Path(path).read_text(encoding="utf-8").strip()


def _publication(spec: _PublicationSpec) -> AppearancePublication:
    return AppearancePublication(
        appearance_id=AppearanceId(spec.appearance_id),
        camera_id=CameraId(spec.camera_id),
        session_id=CameraSessionId(spec.session_id),
        track_id=spec.track_id,
        first_seen=spec.first_seen,
        last_seen=spec.first_seen + timedelta(minutes=5),
        ended_at=spec.first_seen + timedelta(minutes=5),
        representative_version=1,
        crop_object_key=spec.crop_object_key,
        bounding_box=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        source_width=1920,
        source_height=1080,
        detector_confidence=0.91,
        crop_quality=42.5,
        byte_size=len(_JPEG_1X1),
        embedded_at=spec.first_seen,
        model_id="openai/clip-vit-base-patch32",
        model_revision=DEFAULT_CLIP_MODEL_REVISION,
        embedding=spec.embedding,
    )


async def _seed(metadata_path: Path) -> None:
    database_url = os.environ["GW_DATABASE_URL"]
    crops_root = Path(os.environ["GW_CROPS_ROOT"])
    database = Database.connect(database_url)
    store = CropObjectStore(crops_root)
    storage = StorageRepository(CredentialCipher(Fernet.generate_key()))
    try:
        async with database.transaction() as session:
            camera = await storage.add_camera(
                session,
                name="Task 14b QA Camera",
                source_url=_source("task14b-qa-camera"),
            )
            camera_session = await storage.start_camera_session(session, camera.id, cause="qa")
            seed_crop = store.write(_JPEG_1X1)
            nearest_crop = store.write(_JPEG_1X1)
            _ = await storage.publish_appearance(
                session,
                _publication(
                    _PublicationSpec(
                        appearance_id=_SEED_ID,
                        camera_id=camera.id,
                        session_id=camera_session.id,
                        track_id=1,
                        first_seen=_OBSERVED_AT,
                        embedding=_vector((0, 1.0)),
                        crop_object_key=seed_crop.object_key,
                    )
                ),
            )
            _ = await storage.publish_appearance(
                session,
                _publication(
                    _PublicationSpec(
                        appearance_id=_NEAREST_ID,
                        camera_id=camera.id,
                        session_id=camera_session.id,
                        track_id=2,
                        first_seen=_OBSERVED_AT + timedelta(hours=1),
                        embedding=_vector((0, 0.99), (1, 0.14)),
                        crop_object_key=nearest_crop.object_key,
                    )
                ),
            )
        auth = AuthService(database)
        _ = await auth.initialize_password(_password())
        document = {
            "appearance_id": str(_SEED_ID),
            "nearest_appearance_id": str(_NEAREST_ID),
            "camera_id": str(camera.id),
            "model_revision": DEFAULT_CLIP_MODEL_REVISION,
            "password_file_required": True,
        }
        _ = metadata_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    finally:
        await database.close()


def _request(spec: _RequestSpec) -> tuple[int, dict[str, str], bytes]:
    headers = {"Accept": "application/json"}
    if spec.payload is not None:
        headers["Content-Type"] = "application/json"
    if spec.cookie is not None:
        headers["Cookie"] = spec.cookie
    if spec.origin is not None:
        headers["Origin"] = spec.origin
    parsed_base = urlsplit(spec.base_url)
    if parsed_base.scheme not in {"http", "https"} or not parsed_base.netloc:
        message = "HTTP QA base URL must use http or https"
        raise ValueError(message)
    connection = (
        HTTPConnection(parsed_base.netloc, timeout=30)
        if parsed_base.scheme == "http"
        else HTTPSConnection(parsed_base.netloc, timeout=30)
    )
    path = f"{parsed_base.path.rstrip('/')}{spec.path}"
    if parsed_base.query:
        path = f"{path}?{parsed_base.query}"
    try:
        connection.request(
            spec.method,
            path,
            body=json.dumps(spec.payload).encode() if spec.payload is not None else None,
            headers=headers,
        )
        response = connection.getresponse()
        return int(response.status), dict(response.getheaders()), response.read()
    finally:
        connection.close()


def _probe(metadata_path: Path) -> int:
    metadata = cast(
        "dict[str, str]",
        json.loads(metadata_path.read_text(encoding="utf-8")),
    )
    base_url = os.environ.get("GW_BASE_URL", _BASE_URL)
    origin = os.environ.get("GW_PUBLIC_ORIGIN", base_url)
    records: list[dict[str, object]] = []

    def record(name: str, status: int, headers: dict[str, str], body: bytes) -> None:
        try:
            parsed: object = cast("object", json.loads(body))
        except (UnicodeDecodeError, json.JSONDecodeError):
            parsed = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        selected_headers = {
            key.lower(): value
            for key, value in headers.items()
            if key.lower()
            in {"cache-control", "etag", "vary", "x-representative-version", "content-type"}
        }
        records.append(
            {
                "name": name,
                "status": status,
                "headers": selected_headers,
                "body": parsed,
            }
        )

    status, headers, body = _request(
        _RequestSpec(
            base_url,
            "/api/search",
            method="POST",
            payload={"mode": "browse"},
            origin=origin,
        ),
    )
    record("unauthorized_search", status, headers, body)
    status, headers, body = _request(
        _RequestSpec(
            base_url,
            "/api/session",
            method="POST",
            payload={"password": _password()},
            origin=origin,
        ),
    )
    record("login", status, headers, body)
    cookie = next(
        (value.split(";", 1)[0] for key, value in headers.items() if key.lower() == "set-cookie"),
        None,
    )
    if cookie is None:
        message = "login did not return a session cookie"
        raise RuntimeError(message)
    for name, payload in (
        ("browse", {"mode": "browse", "limit": 10}),
        ("similar", {"mode": "similar", "appearance_id": metadata["appearance_id"], "limit": 10}),
        ("text", {"mode": "text", "query": "person near entrance", "limit": 10}),
        ("invalid_text", {"mode": "text", "query": "\uac00", "limit": 10}),
    ):
        status, headers, body = _request(
            _RequestSpec(
                base_url,
                "/api/search",
                method="POST",
                payload=payload,
                cookie=cookie,
                origin=origin,
            )
        )
        record(name, status, headers, body)
    status, headers, body = _request(
        _RequestSpec(
            base_url,
            "/api/search",
            method="POST",
            payload={"mode": "browse", "limit": 10},
            cookie=cookie,
            origin="http://evil.invalid",
        )
    )
    record("cross_origin_search", status, headers, body)
    appearance_id = metadata["appearance_id"]
    status, headers, body = _request(
        _RequestSpec(
            base_url,
            f"/api/appearances/{appearance_id}",
            cookie=cookie,
        )
    )
    record("detail", status, headers, body)
    status, headers, body = _request(
        _RequestSpec(
            base_url,
            f"/api/appearances/{appearance_id}/crop",
            cookie=cookie,
        )
    )
    record("crop", status, headers, body)
    _ = sys.stdout.write(json.dumps(records, indent=2, sort_keys=True) + "\n")
    expected = {
        "unauthorized_search": 401,
        "cross_origin_search": 403,
        "login": 200,
        "browse": 200,
        "similar": 200,
        "text": 200,
        "invalid_text": 422,
        "detail": 200,
        "crop": 200,
    }
    return (
        0
        if all(
            next(item["status"] for item in records if item["name"] == key) == value
            for key, value in expected.items()
        )
        else 1
    )


def main() -> int:
    """Run the requested seed or probe command."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("command", choices=("seed", "probe"))
    _ = parser.add_argument("metadata", type=Path)
    args = parser.parse_args()
    command = cast("str", args.command)
    metadata = cast("Path", args.metadata)
    if command == "seed":
        _ = anyio.run(_seed, metadata)
        return 0
    return _probe(metadata)


if __name__ == "__main__":
    raise SystemExit(main())
