"""Symlink-safe enumeration and deletion for the managed crop root."""

import os
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, final, override

_OBJECT_KEY: Final = re.compile(
    r"[0-9a-f]{2}/[0-9a-f]{2}/[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\.jpg"
)
_TEMP_NAME: Final = re.compile(
    r"\.[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\.tmp"
)
_PATH_COMPONENT_COUNT: Final = 3


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
    pending = [root_path]
    files: list[ManagedFile] = []
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                relative = Path(entry.path).relative_to(root_path).as_posix()
                classified = _classify(relative)
                if classified is None:
                    continue
                try:
                    byte_size = entry.stat(follow_symlinks=False).st_size
                except FileNotFoundError:
                    continue
                files.append(ManagedFile(relative, classified, byte_size))
    return tuple(files)


def safe_unlink(root: Path, object_key: str) -> bool:
    """Unlink one generated JPEG or temporary file without following links."""
    components = object_key.split("/")
    if len(components) != _PATH_COMPONENT_COUNT:
        raise ManagedPathError(object_key)
    kind = _classify(object_key)
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


def _classify(relative: str) -> ManagedFileKind | None:
    parts = relative.split("/")
    if len(parts) != _PATH_COMPONENT_COUNT or not all(
        re.fullmatch(r"[0-9a-f]{2}", part) for part in parts[:2]
    ):
        return None
    if _OBJECT_KEY.fullmatch(relative) is not None:
        return ManagedFileKind.JPEG
    if _TEMP_NAME.fullmatch(parts[2]) is not None:
        return ManagedFileKind.TEMPORARY
    return None


__all__ = [
    "CropRootError",
    "ManagedFile",
    "ManagedFileKind",
    "ManagedPathError",
    "safe_scan",
    "safe_unlink",
]
