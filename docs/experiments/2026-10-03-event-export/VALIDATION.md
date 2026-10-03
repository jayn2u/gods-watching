# Pre-publication validation

Validation was limited to event/export, API/auth, camera, storage and changed-file static checks. The accepted benchmark measurements in [RESULTS.md](RESULTS.md) were not rerun.

## Functional checks

The focused unit/API/auth/camera/export/benchmark set passed **59 tests in 2.27s**:

```bash
timeout 120s env PYTHONPATH=server/src:. /mnt/data/gods-watching/.venv/bin/pytest -q \
  server/tests/test_event_export.py server/tests/test_api_events.py \
  server/tests/test_api_auth.py server/tests/test_api_cameras.py \
  server/tests/test_camera_services.py server/tests/test_event_benchmark.py
```

The first invocation omitted the repository root from `PYTHONPATH` and stopped during collection with `ModuleNotFoundError: No module named 'qa'`; the command above passed after adding `.`.

The bounded database set passed **21 tests in 5.77s**:

```bash
timeout 180s env PYTHONPATH=server/src:. /mnt/data/gods-watching/.venv/bin/pytest -q \
  server/tests/integration/test_camera_events.py \
  server/tests/integration/test_event_export.py server/tests/integration/test_storage.py
```

The fixture used the cached `pgvector/pgvector:0.8.1-pg17` image, a disposable database with 1 CPU, 384 MiB memory and swap, and a 256 MiB data tmpfs. Its Alembic `upgrade head` migration smoke has an internal 60-second deadline and completed successfully. The fixture removed its container; a task-prefix-filtered `docker ps -a` query found no remaining event-test container.

One earlier integration selection accidentally included the RTSP/media test file, contrary to the no-media boundary. It completed **26 passed, 1 failed in 33.18s**:

```bash
timeout 180s env PYTHONPATH=server/src:. /mnt/data/gods-watching/.venv/bin/pytest -q \
  server/tests/integration/test_camera_events.py \
  server/tests/integration/test_event_export.py \
  server/tests/integration/test_camera_services.py server/tests/integration/test_storage.py
```

The named failure was `server/tests/integration/test_camera_services.py::test_real_rtsp_probe_decodes_one_1080p_h264_frame`; its 10-second RTSP-path readiness wait expired. The test launched the task-owned MediaMTX container `gw-camera-probe-3ae80efe0b-media` and a finite FFmpeg publisher. The retained, redacted resource ledger is [`accidental-rtsp-resource-manifest.json`](validation-artifacts/accidental-rtsp-resource-manifest.json). Docker events show container create/attach at 19:16:08, start at 19:16:10, kill at 19:16:21, exit 0 at 19:16:23, and destroy at 19:16:24 (+09:00). The process ledger records the Docker client cleaned with return code -9 and publisher return code 254. A name-filtered container query returned no container afterward. No image or pull event was recorded for the exact MediaMTX digest during the run; this does not establish when the image first entered the local cache. The test was not rerun.

The repository-wide unit baseline is still the earlier interrupted **477-test** run with two fixture failures; it was not repeated. The preserved, redacted logs are [`baseline-unit-tests.log`](validation-artifacts/baseline-unit-tests.log) and [`baseline-fixture-failures.log`](validation-artifacts/baseline-fixture-failures.log). The failures were `server/tests/test_fixtures.py::test_real_manifest_probes_four_prepared_inputs` (`missing_fixture: camera-1`) and `server/tests/test_fixtures.py::test_corrupt_input_fails_preparation` (`probe_timeout`, where the assertion accepts `checksum_mismatch` or `invalid_video`).

## Static checks

Configured Ruff covered all 19 changed Python files with this command:

```bash
timeout 120s /mnt/data/gods-watching/.venv/bin/ruff check --output-format concise \
  qa/events/benchmark_app.py qa/events/benchmark_export.py qa/events/buffered_export.py \
  server/alembic/versions/0008_camera_events.py \
  server/src/gods_watching/api/application.py server/src/gods_watching/api/event_routes.py \
  server/src/gods_watching/cameras/service.py server/src/gods_watching/contracts/events.py \
  server/src/gods_watching/events/__init__.py server/src/gods_watching/events/export.py \
  server/src/gods_watching/events/repository.py server/src/gods_watching/storage/__init__.py \
  server/src/gods_watching/storage/models.py server/tests/integration/conftest.py \
  server/tests/integration/test_camera_events.py server/tests/integration/test_event_export.py \
  server/tests/test_api_events.py server/tests/test_event_benchmark.py \
  server/tests/test_event_export.py
```

It returned **165 diagnostics**: `qa/events/benchmark_app.py` 24, `qa/events/benchmark_export.py` 123, `qa/events/buffered_export.py` 3, and `server/tests/test_event_benchmark.py` 15. All changed production source files had zero Ruff diagnostics. Ruff also reports the explicitly approved private-network `0.0.0.0` benchmark bind as S104. No style-only cleanup was made.

The full redacted Ruff diagnostic JSON is [`ruff-diagnostics-94f5c9d.json`](validation-artifacts/ruff-diagnostics-94f5c9d.json). It is the captured snapshot from head `94f5c9d16efdb4bcb9fd24f942ef284bf4c09cca`, timestamped 2026-10-03 19:26:51 +09:00; it predates the final URL-guard fix and does not report checks on that fix commit.

Ruff counts by rule: TRY003 29, E501 25, EM101 19, E402 13, BLE001 11, EM102 10, PLR2004 10, SLF001 4; INP001, D107, ANN401, D102, C901, PLR0913 and TRY004 3 each; I001, PLC0415, D103, S603, S607, PLR0912, PLR0915 and S105 2 each; TRY300, S104, TRY301, PLW2901, TC003, PLR0402 and PT018 1 each. This is the configured result for the new branch changes, not a baseline comparison.

Configured basedpyright analyzed the same 19 changed Python files in 3.04s using this original invocation (the private task-root path is redacted):

```bash
timeout 120s /mnt/data/gods-watching/.venv/bin/basedpyright \
  --project <TASK_ROOT>/typecheck-events-config.json \
  --pythonpath /mnt/data/gods-watching/.venv/bin/python
```

It reported **288 errors and 42 warnings**. All changed `server/src` production files had zero diagnostics. The remaining diagnostics were in `qa/events/benchmark_app.py` (52 errors), `qa/events/benchmark_export.py` (204 errors, 36 warnings), `qa/events/buffered_export.py` (9 errors), `server/tests/integration/test_camera_events.py` (8 errors, 2 warnings), and `server/tests/test_event_benchmark.py` (15 errors, 3 warnings); the migration file had one unused-call-result warning. These QA/test diagnostics remain disclosed and are not a clean check or a baseline comparison. The redacted full JSON snapshot is [`basedpyright-diagnostics-94f5c9d.json`](validation-artifacts/basedpyright-diagnostics-94f5c9d.json), from the same 94f5c9d head and 2026-10-03 19:26:51 +09:00 timestamp as the Ruff snapshot.

The exact diagnostic configuration is preserved as a portable equivalent at [`basedpyright-events-config.json`](validation-artifacts/basedpyright-events-config.json). It keeps the recorded strict settings and the same 19-file include set, with paths relative to the config directory. Reproduce the configured analysis from the repository root with the chosen interpreter:

```bash
timeout 120s /mnt/data/gods-watching/.venv/bin/basedpyright \
  --project docs/experiments/2026-10-03-event-export/validation-artifacts/basedpyright-events-config.json \
  --pythonpath /mnt/data/gods-watching/.venv/bin/python
```

## Prior independent reviews and measurement limits

Task 1 (`2274047..b5c2a26`) passed its independent specification/quality review with no important or critical findings. Its two minor suggestions—stronger schema metadata assertions and checking stored values directly for secrets—remain deferred. Task 2 (`b5c2a26..2dbd97a`) fixed the actual PostgreSQL statement-failure lifetime issue; the regression demonstrated SQLSTATE `42703`, HTTP 500 with no CSV body, and a returned pool count of zero. Its scoped rereview found no important or critical issue. The historical Task 2 integration RED evidence was not retained and was not reconstructed; minor package-export and secret-assertion suggestions remain deferred. Task 3 (`2dbd97a..3445d41`) addressed all five review findings, and its scoped rereview found no new important or critical finding.

The existing benchmark evidence comprises 12 correct HTTP samples and 12 separate RC/RR probe trials. At 50,000 rows, streaming/buffered total-transfer medians were 0.494080/0.465526 seconds, while first CSV body-byte medians were 0.037267/0.459619 seconds. The metric is the first body byte (the header), not first-row latency. Writer activity intersected transfer in 11/12 samples, but its commit occurred inside the HTTP transfer in 0/12; all commits followed the response by 3.427–35.963 ms. There is no measured memory or general throughput improvement, no per-app cgroup peak, and three repeats do not support p95 or significance claims. The forced RC/RR probe is a separate multi-SELECT snapshot demonstration, not evidence of a production isolation-level speedup. The initial database exit cause remains unresolved, and historical server stderr was not retained; five missing trials were recovered once in a fresh bounded fixture. Raw records, commands and details remain linked from [RESULTS.md](RESULTS.md).

## Final narrow fix and retained review items

The final branch review found that nonempty SQLAlchemy URL query options could override the validated asyncpg host, database, or user. The benchmark-only `ensure_disposable_database` guard now rejects any nonempty query before `_make_engine` creates an engine. The focused regression checks those three effective dialect overrides and verifies that engine creation is never reached. Its RED run was `3 failed, 16 deselected`; after the guard, the same command was `3 passed, 16 deselected`. The complete `server/tests/test_event_benchmark.py` run passed **19 tests in 0.27s**. Python compilation passed for the changed benchmark source and test module.

The final scoped Ruff command covered `qa/events/benchmark_export.py` and `server/tests/test_event_benchmark.py` and reported **140 diagnostics**; these are retained QA/test static debt, not a clean check. The portable strict basedpyright reproduction analyzed 19 files in 3.104s and reported **294 errors and 42 warnings** (exit code 1). The full JSON files linked above remain the older, commit-qualified 94f5c9d snapshots; these fresh totals describe the final fix working tree. No type/style cleanup was made in this wave.

The final code review and Task 4 evidence review's important findings are addressed by the URL-query guard and these tracked evidence copies/links. The public validation-artifact copies are redacted versions of existing local evidence; originals were left unchanged. Redactions cover URL user information, the RTSP publisher path token, credential values, and private home/pytest paths. Diagnostic snapshots remain the full 94f5c9d records described above and must not be read as diagnostics for the final fix commit. Existing measurements were not rerun or changed. The final review's minor findings remain deferred: numeric epoch query strings bypass the explicit-offset syntax contract; benchmark/test type/style debt remains; schema/default/index and stored-secret assertions are weaker than their names suggest; and the disposable integration fixture has no PID bound. These items were not changed in this final narrow fix.
