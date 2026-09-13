"""Crash-safe local object storage for representative JPEG crops."""

import errno
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final, final, override
from uuid import uuid4

_OBJECT_KEY_PATTERN: Final = re.compile(
    r"[0-9a-f]{2}/[0-9a-f]{2}/[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\.jpg"
)


@final
class CropPathError(ValueError):
    """Reject an object key outside the generated crop-key grammar."""

    def __init__(self, object_key: str) -> None:
        """Retain the rejected untrusted object key."""
        self.object_key = object_key
        super().__init__(object_key)

    @override
    def __str__(self) -> str:
        """Return the rejected key without resolving it."""
        return f"invalid crop object key: {self.object_key!r}"


@dataclass(frozen=True, slots=True)
class StoredCrop:
    """Describe one durable crop object."""

    object_key: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class CropObjectStore:
    """Generate object paths and durably install bytes under one configured root."""

    root: Path

    def __post_init__(self) -> None:
        """Create the configured root before accepting object operations."""
        _ = self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _components(object_key: str) -> tuple[str, str, str]:
        if _OBJECT_KEY_PATTERN.fullmatch(object_key) is None:
            raise CropPathError(object_key=object_key)
        first, second, filename = object_key.split("/")
        return first, second, filename

    @staticmethod
    def _open_at(
        name: str | Path,
        flags: int,
        *,
        object_key: str,
        mode: int = 0o777,
        dir_fd: int | None = None,
    ) -> int:
        try:
            return os.open(name, flags, mode, dir_fd=dir_fd)
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise CropPathError(object_key=object_key) from error
            raise

    @contextmanager
    def _parent_directory(
        self,
        object_key: str,
        *,
        create: bool,
    ) -> Iterator[tuple[int, str]]:
        first, second, filename = self._components(object_key)
        descriptors = [
            self._open_at(
                self.root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                object_key=object_key,
            )
        ]
        try:
            for component in (first, second):
                if create:
                    with suppress(FileExistsError):
                        os.mkdir(component, 0o700, dir_fd=descriptors[-1])
                descriptors.append(
                    self._open_at(
                        component,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        object_key=object_key,
                        dir_fd=descriptors[-1],
                    )
                )
            yield descriptors[-1], filename
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    def write(self, payload: bytes) -> StoredCrop:
        """Durably install bytes at a generated UUID object key."""
        object_id = uuid4()
        object_key = f"{object_id.hex[:2]}/{object_id.hex[2:4]}/{object_id}.jpg"
        temporary = f".{object_id}.tmp"
        with self._parent_directory(object_key, create=True) as (parent_fd, filename):
            descriptor = self._open_at(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                mode=0o600,
                object_key=object_key,
                dir_fd=parent_fd,
            )
            try:
                with os.fdopen(descriptor, "wb") as crop_file:
                    _ = crop_file.write(payload)
                    crop_file.flush()
                    os.fsync(crop_file.fileno())
                os.replace(
                    temporary,
                    filename,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                os.fsync(parent_fd)
            finally:
                with suppress(FileNotFoundError):
                    os.unlink(temporary, dir_fd=parent_fd)
        return StoredCrop(object_key=object_key, byte_size=len(payload))

    def read(self, object_key: str) -> bytes:
        """Read an object after enforcing the generated-key grammar."""
        with self._parent_directory(object_key, create=False) as (parent_fd, filename):
            descriptor = self._open_at(
                filename,
                os.O_RDONLY | os.O_NOFOLLOW,
                object_key=object_key,
                dir_fd=parent_fd,
            )
            with os.fdopen(descriptor, "rb") as crop_file:
                return crop_file.read()

    def delete(self, object_key: str) -> None:
        """Idempotently unlink a validated crop object."""
        with (
            suppress(FileNotFoundError),
            self._parent_directory(object_key, create=False) as (parent_fd, filename),
            suppress(FileNotFoundError),
        ):
            os.unlink(filename, dir_fd=parent_fd)
