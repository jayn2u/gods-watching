#!/usr/bin/env python3
"""Run the Task 14a search checks and emit hash-bound QA evidence.

The integration test provisions a disposable PostgreSQL/pgvector container and
labels every ranking vector explicitly. It verifies query behavior, rather than
claiming semantic retrieval quality from synthetic vectors.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from shutil import which
from typing import Final

import anyio

_ARGUMENT_COUNT: Final = 2
_COMMANDS: Final[tuple[tuple[str, ...], ...]] = (
    (
        "focused-tests",
        "uv",
        "run",
        "pytest",
        "-q",
        "server/tests/test_search.py",
        "server/tests/integration/test_search.py",
    ),
    (
        "ruff",
        "uv",
        "run",
        "ruff",
        "check",
        "server/src/gods_watching/search",
        "server/tests/test_search.py",
        "server/tests/integration/test_search.py",
    ),
    (
        "basedpyright",
        "uv",
        "run",
        "basedpyright",
        "server/src/gods_watching/search",
        "server/tests/test_search.py",
        "server/tests/integration/test_search.py",
    ),
    (
        "contract-errors",
        "uv",
        "run",
        "pytest",
        "-q",
        "server/tests/test_contracts.py",
        "-k",
        "search",
    ),
)
_OWNED_FILES: Final[tuple[str, ...]] = (
    "server/src/gods_watching/search/__init__.py",
    "server/src/gods_watching/search/cache.py",
    "server/src/gods_watching/search/crop.py",
    "server/src/gods_watching/search/errors.py",
    "server/src/gods_watching/search/repository.py",
    "server/src/gods_watching/search/service.py",
    "server/tests/test_search.py",
    "server/tests/integration/test_search.py",
    "qa/search/query_service_probe.py",
)


def _resolve_executable(command: str) -> str:
    executable = which(command)
    if executable is None:
        raise FileNotFoundError(command)
    return executable


def _decode_output(output: bytes | None) -> str:
    return output.decode("utf-8") if output else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def _run(repo_root: Path, evidence_root: Path) -> int:
    evidence_root.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []
    for command in _COMMANDS:
        name, *argv = command
        completed = await anyio.run_process(
            [_resolve_executable(argv[0]), *argv[1:]],
            cwd=repo_root,
            check=False,
        )
        stdout = _decode_output(completed.stdout)
        stderr = _decode_output(completed.stderr)
        log_path = evidence_root / f"{name}.log"
        _ = log_path.write_text(
            f"$ {' '.join(argv)}\n\n{stdout}{stderr}",
            encoding="utf-8",
        )
        checks.append(
            {
                "name": name,
                "command": list(argv),
                "exit_code": completed.returncode,
                "log": os.path.relpath(log_path, repo_root),
            }
        )

    source_hashes = {path: _sha256(repo_root / path) for path in _OWNED_FILES}
    git = await anyio.run_process(
        [_resolve_executable("git"), "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
    )
    semantic_quality_prefix = "This probe does not claim CLIP semantic retrieval quality"
    semantic_quality_limitation = "or end-to-end HTTP/auth behavior."
    document = {
        "schema_version": 1,
        "task": "14a",
        "scenario": "query-service",
        "status": "passed" if all(item["exit_code"] == 0 for item in checks) else "failed",
        "evidence_kind": "synthetic vectors against real PostgreSQL/pgvector",
        "semantic_quality_claim": False,
        "observed_at": datetime.now(UTC).isoformat(),
        "git_head": _decode_output(git.stdout).strip(),
        "source_sha256": source_hashes,
        "checks": checks,
        "scope": [
            "typed text/similar/browse dispatch",
            "camera and inclusive time-overlap filters",
            "revision-aware bounded text cache",
            "strict HNSW setting with exact filtered fallback",
            "stable cosine ranking and default seed exclusion",
            "archived-camera visibility and tombstone/vectorless exclusion",
            "crop pointer/version revalidation service",
        ],
        "limitations": [
            "The ranking vectors are separately labeled synthetic fixtures.",
            f"{semantic_quality_prefix} {semantic_quality_limitation}",
        ],
    }
    _ = (evidence_root / "task-14-search.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    error_check = next(item for item in checks if item["name"] == "contract-errors")
    errors_document = {
        "schema_version": 1,
        "task": "14a",
        "scenario": "search-errors",
        "status": "passed" if error_check["exit_code"] == 0 else "failed",
        "checks": [error_check],
        "binary_observable": (
            "search contract tests reject blank/unsupported text and reversed UTC ranges "
            "before retrieval; the service keeps CLIP token rejection distinct from outage"
        ),
        "limitations": [
            "HTTP/auth error mapping belongs to Task 14b and is not exercised here.",
        ],
    }
    _ = (evidence_root / "task-14-errors.json").write_text(
        json.dumps(errors_document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if document["status"] == "passed" else 1


def main() -> int:
    """Run the Task 14a search checks and emit hash-bound QA evidence."""
    if len(sys.argv) != _ARGUMENT_COUNT:
        _ = sys.stderr.write(f"usage: {Path(sys.argv[0]).name} EVIDENCE_DIRECTORY\n")
        return 2
    repo_root = Path(__file__).resolve().parents[2]
    return anyio.run(_run, repo_root, Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
