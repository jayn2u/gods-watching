"""Symlink-safe enumeration and deletion for the managed crop root."""

import os
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, final, override

_HEX_COMPONENT_PATTERN: Final = r"[0-9a-f]{2}"
_UUID_V4_COMPONENT: Final = r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
_OBJECT_KEY: Final = re.compile(
    _HEX_COMPONENT_PATTERN
    + "/"
    + _HEX_COMPONENT_PATTERN
    + "/"
    + _UUID_V4_COMPONENT
    + r"\.jpg"
)
_HEX_COMPONENT: Final = re.compile(_HEX_COMPONENT_PATTERN)
_OBJECT_FILENAME: Final = re.compile(_UUID_V4_COMPONENT + r"\.jpg")
_TEMP_NAME: Final = re.compile(r"\." + _UUID_V4_COMPONENT + r"\.tmp")
_PATH_COMPONENT_COUNT: Final = 3
_DIRECTORY_COMPONENT_COUNT: Final = _PATH_COMPONENT_COUNT - 1


@final
class ManagedPathError(ValueError):
    """Reject a crop path outside the generated object grammar."""

    def __init__(self, object_key: str) -> None:
        """Retain the rejected path for diagnostics."""
        self.object_key = object_key
        super().__init__(object_key)

    @override
    def __str__(self) -> str:
        """Render the rejected path without resolving it."""
        return f"invalid managed crop path: {self.object_key!r}"


@final
class CropRootError(ValueError):
    """Reject a configured crop root that is not a real directory."""

    def __init__(self, root: Path) -> None:
        """Retain the configured root for diagnostics."""
        self.root = root
        super().__init__(root)

    @override
    def __str__(self) -> str:
        """Render the rejected root path."""
        return f"crop root is not a real directory: {self.root}"


class ManagedFileKind(StrEnum):
    """Classify generated objects that retention may remove."""

    JPEG = "jpeg"
    TEMPORARY = "temporary"


@dataclass(frozen=True, slots=True)
class ManagedFile:
    """Describe a regular generated file below the crop root."""

    object_key: str
    kind: ManagedFileKind
    byte_size: int


def safe_scan(root: Path) -> tuple[ManagedFile, ...]:
    """Enumerate only regular files matching the generated object grammar."""
    root_path = _checked_root(root)
    pending: list[tuple[str, str, int]] = [(os.fspath(root_path), "", 0)]
    files: list[ManagedFile] = []
    while pending:
        directory, key_prefix, depth = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if depth < _DIRECTORY_COMPONENT_COUNT:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    if _HEX_COMPONENT.fullmatch(entry.name) is None:
                        continue
                    next_prefix = (
                        entry.name if depth == 0 else f"{key_prefix}/{entry.name}"
                    )
                    pending.append((entry.path, next_prefix, depth + 1))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                classified = _classify_filename(entry.name)
                if classified is None:
                    continue
                try:
                    byte_size = entry.stat(follow_symlinks=False).st_size
                except FileNotFoundError:
                    continue
                files.append(
                    ManagedFile(f"{key_prefix}/{entry.name}", classified, byte_size)
                )
    return tuple(files)


def safe_unlink(root: Path, object_key: str) -> bool:
    """Unlink one generated JPEG or temporary file without following links."""
    components = object_key.split("/")
    if len(components) != _PATH_COMPONENT_COUNT:
        raise ManagedPathError(object_key)
    kind = classify_managed_path(object_key)
    if kind is None:
        raise ManagedPathError(object_key)
    root_path = _checked_root(root)
    descriptors = [
        os.open(root_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW),
    ]
    try:
        for component in components[:2]:
            try:
                descriptor = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptors[-1],
                )
            except FileNotFoundError:
                return False
            descriptors.append(descriptor)
        try:
            os.unlink(components[2], dir_fd=descriptors[-1])
        except FileNotFoundError:
            return False
        return True
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _checked_root(root: Path) -> Path:
    if root.is_symlink():
        raise CropRootError(root)
    try:
        absolute = root.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise CropRootError(root) from error
    if not absolute.is_dir():
        raise CropRootError(root)
    return absolute


def classify_managed_path(relative: str) -> ManagedFileKind | None:
    """Return the managed kind for an exact generated relative path."""
    parts = relative.split("/")
    if len(parts) != _PATH_COMPONENT_COUNT or not all(
        _HEX_COMPONENT.fullmatch(part) for part in parts[:2]
    ):
        return None
    if _OBJECT_KEY.fullmatch(relative) is not None:
        return ManagedFileKind.JPEG
    return _classify_filename(parts[2])


def _classify_filename(filename: str) -> ManagedFileKind | None:
    if filename.startswith("."):
        if _TEMP_NAME.fullmatch(filename) is not None:
            return ManagedFileKind.TEMPORARY
    elif _OBJECT_FILENAME.fullmatch(filename) is not None:
        return ManagedFileKind.JPEG
    return None


__all__ = [
    "CropRootError",
    "ManagedFile",
    "ManagedFileKind",
    "ManagedPathError",
    "classify_managed_path",
    "safe_scan",
    "safe_unlink",
]
