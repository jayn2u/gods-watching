"""Count physical crop bytes with a native, symlink-safe filesystem walk."""

from __future__ import annotations

import os
import subprocess
from shutil import which
from typing import TYPE_CHECKING, Final, final

from .managed_files import CropRootError, classify_managed_path

_SCAN_TIMEOUT_SECONDS: Final = 10.0
_FIND_UNAVAILABLE: Final = "GNU find is unavailable for physical crop accounting"
_FIND_TIMEOUT: Final = "physical crop size scan timed out"
_FIND_START_FAILED: Final = "physical crop size scan could not start"
_FIND_OUTPUT_TRUNCATED: Final = "physical crop size scan returned truncated output"
_FIND_OUTPUT_INVALID: Final = "physical crop size scan returned an invalid size"

if TYPE_CHECKING:
    from pathlib import Path


@final
class PhysicalUsageError(RuntimeError):
    """Report a failed or malformed native crop-size scan."""


def physical_crop_bytes(root: Path) -> int:
    """Sum apparent bytes for regular JPEG and temporary files without following links."""
    output = _run_find(
        root,
        "-type",
        "f",
        "(",
        "-name",
        "*.jpg",
        "-o",
        "-name",
        "*.tmp",
        ")",
        printf="%s\\0",
    )
    return _sum_sizes(output)


def managed_crop_bytes(root: Path) -> int:
    """Sum only generated crop and temp files recognized by ``safe_scan``."""
    output = _run_find(
        root,
        "-mindepth",
        "3",
        "-maxdepth",
        "3",
        "-type",
        "f",
        printf="%P\\0%s\\0",
    )
    return _sum_managed_sizes(output)


def _run_find(root: Path, *expression: str, printf: str) -> bytes:
    root_path = _checked_root(root)
    find_binary = which("find")
    if find_binary is None:
        raise PhysicalUsageError(_FIND_UNAVAILABLE)
    try:
        result = subprocess.run(  # noqa: S603
            [
                find_binary,
                "-P",
                os.fspath(root_path),
                *expression,
                "-printf",
                printf,
            ],
            check=False,
            shell=False,
            capture_output=True,
            timeout=_SCAN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise PhysicalUsageError(_FIND_TIMEOUT) from error
    except OSError as error:
        raise PhysicalUsageError(_FIND_START_FAILED) from error

    if result.returncode != 0:
        message = f"physical crop size scan exited with status {result.returncode}"
        raise PhysicalUsageError(message)
    return result.stdout


def _sum_sizes(output: bytes) -> int:
    records = output.split(b"\0")
    if records[-1] != b"":
        raise PhysicalUsageError(_FIND_OUTPUT_TRUNCATED)
    total = 0
    for record in records[:-1]:
        if not record or not record.isdigit():
            raise PhysicalUsageError(_FIND_OUTPUT_INVALID)
        total += int(record)
    return total


def _sum_managed_sizes(output: bytes) -> int:
    records = output.split(b"\0")
    if records[-1] != b"":
        raise PhysicalUsageError(_FIND_OUTPUT_TRUNCATED)
    fields = records[:-1]
    if len(fields) % 2 != 0:
        raise PhysicalUsageError(_FIND_OUTPUT_TRUNCATED)
    total = 0
    for index in range(0, len(fields), 2):
        relative_path = os.fsdecode(fields[index])
        size = _parse_size(fields[index + 1])
        if classify_managed_path(relative_path) is not None:
            total += size
    return total


def _parse_size(record: bytes) -> int:
    if not record or not record.isdigit():
        raise PhysicalUsageError(_FIND_OUTPUT_INVALID)
    return int(record)


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


__all__ = ["PhysicalUsageError", "managed_crop_bytes", "physical_crop_bytes"]
