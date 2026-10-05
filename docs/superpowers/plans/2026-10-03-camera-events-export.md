# Camera Events and CSV Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement transactional camera-management history and authenticated CSV export, and measure the actual export implementation honestly.

**Architecture:** CameraService records immutable events in its existing mutation transaction. A dedicated exporter prepares one read-only Read Committed SELECT and owns its cursor/session until the HTTP download ends. Benchmark-only buffered and multi-SELECT variants exercise the same table and projection without adding public API modes.

**Tech Stack:** Existing Python 3.12, FastAPI, SQLAlchemy 2.0.43, asyncpg 0.30.0, Alembic, PostgreSQL 17/pgvector, pytest; stdlib HTTP, CSV, subprocess and resource measurement.

**Spec:** `docs/superpowers/specs/2026-10-03-camera-events-export-design.md` (approved by the user).

## Global Constraints

- Branch `codex/event-log-export`; worktree `/home/jwchoi/Documents/Codex/2026-10-03/task/gods-watching`; preserve original checkout edits.
- No new dependencies, image pulls, GPUs, global DB changes, real application data, UI, merge or deployment.
- Event types: `camera.created`, `camera.updated`, `camera.deleted`; name maximum 80 characters; no credentials or free-form payload.
- GET `/api/events/export.csv`; columns `id,occurred_at,event_type,camera_id,camera_name`; ID ordering; inclusive `since`, exclusive `until`; timezone-aware inputs and UTC output.
- Single SELECT, Read Committed, cursor batches 1,000 rows; 60-second statement timeout; five-minute total export deadline; cancellation closes resources.
- Benchmark datasets 10,000/50,000 rows, three repeats per size/mode; one export and at most one insertion connection; aggregate DB connections at most four.
- DB limit: 1 CPU, 384MiB memory, 256MiB disposable data tmpfs. App/client limit: at most 1 CPU and 256MiB per container; no persistent real volumes; loopback-only published ports if used.
- Three samples support median/range reporting, not meaningful p95 or statistical-significance claims. Wrong output is not a performance improvement.
- Existing full-unit baseline stalled after two fixture failures and was interrupted. Preserve its logs; do not repeat that stalled full-suite command. Use focused checks with explicit deadlines and disclose unchecked coverage.

## Review Focus

- No-op updates, stale versions and failed recording must not produce orphan/duplicate events (Task 1).
- CSV names containing whitespace, formula prefixes, quotes, commas and newlines must round-trip safely without secret leakage (Task 2).
- Invalid filters and unauthorized requests must fail before exporting; empty exports retain a valid header (Task 2).
- Disconnect, timeout and failure before/after response start must release the cursor/session without pretending a truncated file succeeded (Task 2).
- Buffered/streaming samples must use identical data/query/encoding and exclude client/DB RSS; overlapping insertion and multi-SELECT anomalies must be proven rather than assumed (Task 3).

---

## Execution and tool setup

Use `/mnt/data/gods-watching/.venv/bin/python` and set `PYTHONPATH` to this worktree's `server/src` for host checks. Never accidentally import the original editable installation. Use the existing image `gods-watching-app:local` with this worktree mounted read-only and the same PYTHONPATH for isolated application runs. Use `PYTHONDONTWRITEBYTECODE=1` and pytest `-p no:cacheprovider`.

Commands below use `PY` to mean that existing interpreter and must be recorded with their resolved path in evidence. Use `timeout` around shell test commands; orchestrator subprocesses also receive explicit timeout arguments. Inspect each nonzero exit and retain stdout/stderr; do not hide errors in a summary.

For real-DB tests, adapt the existing disposable PostgreSQL fixture to use `--pull=never`, CPU/memory/PID/tmpfs bounds, loopback publication, and `sys.executable -m alembic` instead of confined `uv`. It must delete only the container it created. Do not run media/RTSP integration tests as part of this feature.

## Task 1: Transactional event storage and camera hooks

**Files**
- Create `server/alembic/versions/0008_camera_events.py` and `server/src/gods_watching/events/__init__.py`, `repository.py`.
- Modify `server/src/gods_watching/storage/models.py`, `storage/__init__.py`, `cameras/service.py`.
- Modify `server/tests/integration/conftest.py` for bounded existing-image DB execution only.
- Create `server/tests/integration/test_camera_events.py`; extend `server/tests/test_camera_services.py` only where needed for existing test doubles.

**Interfaces**
- `CameraEvent`: identity bigint `id`, DB-generated timezone-aware `occurred_at`, constrained `event_type`, UUID `camera_id`, `camera_name` length 1..80; no FK cascade that deletes history.
- `EventRepository.record(session: AsyncSession, *, event_type: str, camera_id: UUID, camera_name: str) -> CameraEvent`: add/flush but never commit or open another transaction.

- [ ] First write `test_event_schema_contract` against the existing `Base.metadata`: assert `camera_events` exists with exactly the five specified columns, defaults and constraints. Watch its expected missing-table assertion fail, then implement only the mapping/migration needed to pass.
- [ ] Write `test_create_update_delete_record_ordered_events`: create, change and delete a synthetic camera; assert exactly three event types, correct ID/name snapshots, no secret fields, history retained after soft deletion.
- [ ] Write `test_noop_and_stale_update_do_not_record`, `test_failed_probe_does_not_record`, `test_mutation_and_event_rollback_together`, and `test_event_insert_failure_rolls_back_camera`: exercise the real service/DB, asserting committed state from a fresh session.
- [ ] Run the new focused tests before hooks exist and retain the expected missing-history failures; separate missing symbols/fixture errors from the behavioral red step.
- [ ] Implement the repository/hook behavior after the corresponding red tests; call recording exactly once after each successful durable create/changed update/delete, before the existing transaction commits. Do not record reconnects or post-commit runtime success.
- [ ] Run `timeout 120 PY -m pytest server/tests/integration/test_camera_events.py server/tests/test_camera_services.py -p no:cacheprovider -q` in the safe DB fixture environment. Expected: new event tests and existing service behavior pass.
- [ ] In a separate disposable migration database, run upgrade head, downgrade 0007, upgrade head with `PY -m alembic`; verify event-table schema/defaults/indexes and no dangling migration state. Downgrade intentionally drops only synthetic event data.
- [ ] Commit the event-storage deliverable; keep the human Git author and established agent attribution convention.

## Task 2: Authenticated bounded CSV download

**Files**
- Create `server/src/gods_watching/contracts/events.py`, `events/export.py`, `api/event_routes.py`.
- Modify `events/repository.py`/`__init__.py` and `api/application.py` to compose the route using the existing session guard.
- Create `server/tests/test_event_export.py`, `server/tests/test_api_events.py`, `server/tests/integration/test_event_export.py`.

**Interfaces**
- `EventExportFilters`: optional UUID `camera_id`, timezone-aware `since`/`until`; reject inverted ranges. No raw SQL inputs.
- `EventExportRow`: immutable value with `id: int`, `occurred_at: datetime`, `event_type: str`, `camera_id: UUID`, `camera_name: str`; returned in bounded partitions, not full-table ORM materialization.
- `EventRepository.statement(filters: EventExportFilters) -> Select`: shared five-column filtered projection ordered by event ID.
- `encode_csv_row(row: EventExportRow) -> bytes`: shared UTF-8 standard CSV encoding, UTC timestamp and documented formula-prefix protection.
- `PreparedCsvExport`: `chunks: AsyncIterator[bytes]`, `aclose() -> None`; idempotent cleanup of owned DB transaction/result/session.
- `EventExportService(database: Database).prepare(filters: EventExportFilters) -> PreparedCsvExport` (async): initialize the read-only transaction/SELECT and obtain the first bounded partition before returning, so the snapshot exists before HTTP bytes are sent.
- `build_event_router(*, database: Database, require_session: SessionDependencyFactory, exporter: EventExporter | None = None) -> APIRouter`: default to production service; injection exists for tests/benchmark only, never as an HTTP query option.
- `EventExporter` protocol: async `prepare(filters) -> PreparedCsvExport`; buffered comparison in Task 3 implements the same boundary.

- [ ] Write encoder tests: header/order, empty result, UTF-8, UTC, comma/quote/newline round-trip and all dangerous prefixes including whitespace before `=`. Assert no RTSP URL/token is present.
- [ ] Write API tests: unauthorized request rejected, authenticated attachment response, camera/time filters, invalid UUID/naive timestamp/inverted range rejected before exporter use, no mutation/event for download.
- [ ] Write real-DB `test_single_select_snapshot_excludes_later_commits`: prepare a cursor over >1,000 known event rows, consume first rows, insert/commit events on another connection, finish and assert exact original IDs with no duplicates/new rows.
- [ ] Write resource-lifetime tests for normal completion, cancellation, statement failure before bytes, failure after bytes and total deadline. Real-DB cancellation must return the connection to the pool; test-only short deadlines may be injected without exposing public options.
- [ ] Run each test first and retain expected failure evidence before its production implementation.
- [ ] Implement the common filter/projection, encoder, bounded exporter and route. Keep 60-second statement timeout and five-minute overall response deadline. Set `Content-Type` CSV, safe fixed attachment filename and `Cache-Control: no-store`.
- [ ] Ensure the response owner closes prepared resources even if iteration never starts or the client disconnects. Preserve pre-response errors as HTTP errors; do not claim a truncated post-response stream was successful.
- [ ] Run `timeout 120 PY -m pytest server/tests/test_event_export.py server/tests/test_api_events.py server/tests/integration/test_event_export.py server/tests/test_api_cameras.py server/tests/test_api_auth.py server/tests/test_api_session_activity.py -p no:cacheprovider -q`. Expected: all new export tests and selected existing API/auth checks pass.
- [ ] Run existing ruff and basedpyright on changed production/test files; address new errors and report baseline tool/configuration problems separately. Commit the export deliverable.

## Task 3: Real implementation benchmark and separate isolation probe

**Files**
- Create `qa/events/benchmark_export.py`, `qa/events/benchmark_app.py`, `qa/events/buffered_export.py`, `qa/events/README.md`.
- Create `server/tests/test_event_benchmark.py` for result validation/controlled fixture orchestration, not mirrored timing code.
- Create `docs/experiments/2026-10-03-event-export/results.json`, `RESULTS.md`, environment/command logs after actual runs.

**Interfaces**
- Buffered exporter implements Task 2's `EventExporter` using the same repository projection and CSV encoder; only buffering differs.
- Benchmark app composes actual export and session routers with real AuthService/Database, synthetic operator/session, and no media services. Use real login, not an authentication bypass. It binds loopback only.
- Orchestrator launches a fresh app subprocess per sample and a bounded writer using `EventRepository.record`. Client uses stdlib HTTP streaming to a temporary file, never `response.read()` for the full body.
- Sample JSON includes dataset size/mode/repeat, latency/first-byte latency, bytes/rows/IDs verification, application peak RSS/startup RSS, writer count/commit times/overlap, outcome and exact environment. DB and client memory are excluded from application RSS.

- [ ] Write a validator test: intentionally missing/duplicate/unexpected IDs or invalid CSV fail correctness and are excluded from improvement calculations. Write a test rejecting production/non-disposable DB targets.
- [ ] Run the validator red tests; implement safe orchestration and validation. Collect application RSS from the app process itself (`resource.getrusage(RUSAGE_SELF).ru_maxrss`, Linux KiB) in benchmark-only shutdown/control reporting; use a fresh process to isolate high-water marks.
- [ ] Run a small smoke sample through the real migrated table, real login and HTTP route. Validate identical buffered/streaming CSV output before large measurements.
- [ ] Create only task-owned containers from cached images with the spec limits. App/client share the isolated DB container network when possible; no public port or existing volume. Apply migrations explicitly to the synthetic database and retain image IDs/version output.
- [ ] Seed 10,000 or 50,000 synthetic event rows; run both modes three times each (12 export samples total), alternating mode order. Reset data outside timed regions; use one export/one writer at a time.
- [ ] Start a fixed 100-event writer after the first CSV data is observed. Insert all 100 through the real repository in a bounded transaction. Record start/commit/end timestamps and test whether commit overlaps the observed export interval. Verify the output contains precisely the pre-insertion snapshot IDs and correct field values.
- [ ] Distinguish overlap with HTTP transfer from overlap with DB fetch: a buffered exporter has already fetched/encoded before its first bytes. Report absent overlap as absent; do not slow the exporter artificially or silently rerun until a desired result appears.
- [ ] Separately reset the synthetic data to the chosen dataset size before each trial, outside timing. Perform initial COUNT, writer commit of 100 events, then materialize the same repository projection within one transaction under RC and RR. Three trials for each size/level (12 isolation trials), forced scheduling disclosed. Expect RC initial `N` and exported `N+100`, RR initial/exported `N`; verify exact IDs, the committed writer rows, errors and elapsed time.
- [ ] Validate all raw samples before calculating medians/ranges. Report absolute peak RSS plus startup baseline and any delta; do not equate tracemalloc with process RSS. Explain fresh-process startup, small samples, tmpfs/CPU quota, forced scheduling and reduced app composition.
- [ ] Write reproduction commands/raw evidence and neutral/worse results faithfully. Attribute any buffering gains to streaming, and multi-SELECT consistency to isolation; never label this historical Innodep production evidence. Commit benchmark/evidence/docs and clean up task-owned containers.

## Task 4: Bounded verification, independent review and draft PR

**Files**
- Modify `README.md` for event semantics/export usage and link the experiment report.
- Modify evidence/docs only as needed to reflect the final implementation and actual results.

- [ ] Run focused event, exporter, API, auth and camera tests under a 120-second command deadline. Run related storage integration tests in the bounded DB fixture under a 180-second deadline. Do not start RTSP/media/GPU integration services.
- [ ] Run available ruff/basedpyright checks under 120-second deadlines and migration smoke under 60 seconds. Record passed test counts and all named failures; keep the prior baseline failure logs. Do not retry the stalled full-suite command or describe partial checks as a full-suite pass.
- [ ] Request independent review of the complete immutable branch diff, particularly authorization, atomic logging, cancellation cleanup and benchmark comparability. Fix actionable findings with regression red/green tests, and rerun affected checks only.
- [ ] Verify final raw measurements, reproduction commands and working-tree diff; compare original checkout status to the preserved baseline. Verify no task containers remain. Commit all reviewed changes; no unrelated artifacts.
- [ ] Push only `codex/event-log-export`; create a draft PR against `develop` using `gh` outside the sandbox, body-file with real newlines. Lead with actual product behavior and include verification/results/limitations.
- [ ] Query PR head SHA and CI/check runs for that exact SHA. If no workflow exists, report no checks configured; if pending, wait with bounded updates; if failure/infrastructure blocker remains, report it accurately. Never imply green CI without a successful check result.
- [ ] Return PR URL, head commit, precise result table, evidence paths, test coverage, limitations and remaining blockers in Korean. Do not merge or deploy.

## Self-review and execution recommendation

All spec sections map to Tasks 1-4, and each Review Focus item has a test/validation step. Production/export/benchmark interfaces share the same filter, projection and encoder. The only additional scope is bounding the existing test DB fixture, required to test safely with cached tools. Existing API/auth test filenames and static-tool availability were checked in this worktree.

Recommend **Subagent-driven** execution: storage/hooks, streaming lifetime and HTTP behavior, and benchmark measurement each deserve an independent gate because an apparently fast export with wrong data or unreleased sessions would invalidate the outcome. Use **GPT-6 Luna at max reasoning for code writing**, as required by the local global AGENTS.md, with separate reviewers after each task and a whole-branch review before PR publication. A Native approach is cheaper but must still satisfy that code-writer instruction and provide independent final review.

The design and plan are approved (`aa5df2f`, `2274047`). Tasks 1–3 and their review gates are complete; Task 4 is recording bounded verification evidence before the controller's final branch review and publication steps.
