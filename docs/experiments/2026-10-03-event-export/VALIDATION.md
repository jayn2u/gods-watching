# Event export validation

Validation was limited to event/export, API/auth, camera, storage and changed-file static checks. The accepted benchmark measurements in [RESULTS.md](RESULTS.md) were not rerun.

## Scoped static cleanup after draft publication

This follow-up resolves the branch-added QA/test diagnostics recorded at `7f7c4a9e550f427338e53b83d5ae0007773a1db9`. A fresh **19-file** prechange snapshot recorded **167 Ruff diagnostics** and **294 basedpyright errors / 42 warnings**; the earlier 140 Ruff total covered only two files. The full fresh snapshots are [Ruff](validation-artifacts/event-export-static-fix-prechange-ruff-167.json) and [basedpyright](validation-artifacts/event-export-static-fix-prechange-basedpyright-294-42.json). Their diagnostic-containing files were added by this branch; the included previously existing production/conftest files had no diagnostics. This is not a repository-wide clean-baseline claim.

The cleanup validates JSON/SQLAlchemy boundaries, types measurement and request state, separates cohesive control, download and recovery helpers, and checks schema types before assertions. The migration only explicitly consumes the existing execution result. No production behavior or global database configuration changes. Four QA package/bootstrap files preserve package and direct-script imports; strict configuration now includes **23 files** without reducing rules or dropping earlier includes.

Intentional exact-line Ruff exceptions remain for background/owned-resource failure capture (`BLE001`), bounded shell-free Docker and read-only Git calls (`S603`), and the app bind on its internal Docker network with no published ports (`S104`). Two command-string expressions retain explicit concatenation with `ISC003` exceptions because strict basedpyright rejects implicit concatenation while Ruff requests it. These preserve exact command bytes. Docker/Git paths resolve lazily to absolute existing executables and fail explicitly if missing. There are no blanket/file-wide rule disables or type ignores.

The independent review required preserving recovery cleanup with an owner `try/finally` even when execution or evidence writing raises, and preserving transfer-end timing after the output file closes. These boundaries are covered by focused pure regressions. Fresh final verification and its source-qualified evidence are recorded below; all historical outcomes below remain qualified to their original runs.

Final configured Ruff is **0 diagnostics**, and basedpyright is **23 files / 0 errors / 0 warnings** (3.189s), both exit 0. The final snapshots are [Ruff](validation-artifacts/ruff-static-cleanup-final.json) and [basedpyright](validation-artifacts/basedpyright-static-cleanup-final.json); [verification metadata](validation-artifacts/static-cleanup-verification.json) qualifies them to exact source SHA-256 values and tool versions.

After both runtime review fixes, the six-module focused command below passed **70 tests in 11.11s** ([log](validation-artifacts/static-cleanup-focused-tests.log)). The schema-only command passed **1 test, 5 deselected in 0.73s** without a database fixture. The final subsequent change touched only test helpers to satisfy lint/types; the writer reran `server/tests/test_event_benchmark.py`, with **27 passed in 2.12s**. Independent rereview found no remaining Critical/Important issue and accepted the exact-line lint exceptions. The earlier full suite and database integration runs below were not repeated.

```bash
timeout 120s env PYTHONPATH=server/src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/gods-static-mpl \
  /mnt/data/gods-watching/.venv/bin/pytest -q \
  server/tests/test_event_export.py server/tests/test_api_events.py \
  server/tests/test_api_auth.py server/tests/test_api_cameras.py \
  server/tests/test_camera_services.py server/tests/test_event_benchmark.py

timeout 60s env PYTHONPATH=server/src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/gods-static-mpl \
  /mnt/data/gods-watching/.venv/bin/pytest -q \
  server/tests/integration/test_camera_events.py -k event_schema_contract
```

Direct-script `benchmark_export.py --help` and a `runpy.run_path` import of `benchmark_app.py` (with its sibling script directory on `sys.path`) passed without starting a server. The first import-harness attempt omitted that script directory and failed import resolution; the corrected harness models Python direct-script path setup. `git diff --check` passed. No fresh CI success is inferred from these local results.

The host is `vis-lab` (Linux), using existing Python 3.12.11, Ruff 0.12.12, basedpyright 1.31.4 (pyright 1.1.405) and pytest 8.4.2. Final commands run from the isolated worktree; `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` are set to 1 for tests/import checks, and `MPLCONFIGDIR=/tmp/gods-static-mpl` keeps incidental cache writes in temporary storage.

Reproduce the exact configured Ruff file set with the existing tools (no installation):

```bash
python3 - <<'PY_RUFF'
import json, subprocess
from pathlib import Path
config = Path('docs/experiments/2026-10-03-event-export/validation-artifacts/basedpyright-events-config.json')
files = [str((config.parent / item).resolve()) for item in json.loads(config.read_text())['include']]
result = subprocess.run(['/mnt/data/gods-watching/.venv/bin/ruff', 'check', '--output-format', 'json', *files], timeout=120)
raise SystemExit(result.returncode)
PY_RUFF

timeout 120s /mnt/data/gods-watching/.venv/bin/basedpyright --outputjson \
  --project docs/experiments/2026-10-03-event-export/validation-artifacts/basedpyright-events-config.json \
  --pythonpath /mnt/data/gods-watching/.venv/bin/python
```

The four saved measurement artifacts match [their pre-fix SHA-256 values](validation-artifacts/measurement-hashes-before-static-fix.json). Recalculating both HTTP and isolation summaries from the saved raw records gives exactly the pre-fix summaries (12 HTTP samples and 12 isolation trials). No benchmark, container, database, GPU, media service or full-suite execution was performed in this follow-up, and no dependency/image was installed or pulled.

## Historical pre-publication functional checks

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

## Historical pre-publication static checks

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

The exact diagnostic configuration is preserved as a portable equivalent at [`basedpyright-events-config.json`](validation-artifacts/basedpyright-events-config.json). It keeps the recorded strict settings and paths relative to the config directory. The later static cleanup expands its include set from 19 to 23 files to cover the new QA package/bootstrap files and adds the QA script directory to import search paths; no diagnostic rule is relaxed. Reproduce the configured analysis from the repository root with the chosen interpreter:

```bash
timeout 120s /mnt/data/gods-watching/.venv/bin/basedpyright \
  --project docs/experiments/2026-10-03-event-export/validation-artifacts/basedpyright-events-config.json \
  --pythonpath /mnt/data/gods-watching/.venv/bin/python
```

## Prior independent reviews and measurement limits

Task 1 (`2274047..b5c2a26`) passed its independent specification/quality review with no important or critical findings. Its two minor suggestions—stronger schema metadata assertions and checking stored values directly for secrets—remain deferred. Task 2 (`b5c2a26..2dbd97a`) fixed the actual PostgreSQL statement-failure lifetime issue; the regression demonstrated SQLSTATE `42703`, HTTP 500 with no CSV body, and a returned pool count of zero. Its scoped rereview found no important or critical issue. The historical Task 2 integration RED evidence was not retained and was not reconstructed; minor package-export and secret-assertion suggestions remain deferred. Task 3 (`2dbd97a..3445d41`) addressed all five review findings, and its scoped rereview found no new important or critical finding.

The existing benchmark evidence comprises 12 correct HTTP samples and 12 separate RC/RR probe trials. At 50,000 rows, streaming/buffered total-transfer medians were 0.494080/0.465526 seconds, while first CSV body-byte medians were 0.037267/0.459619 seconds. The metric is the first body byte (the header), not first-row latency. Writer activity intersected transfer in 11/12 samples, but its commit occurred inside the HTTP transfer in 0/12; all commits followed the response by 3.427–35.963 ms. There is no measured memory or general throughput improvement, no per-app cgroup peak, and three repeats do not support p95 or significance claims. The forced RC/RR probe is a separate multi-SELECT snapshot demonstration, not evidence of a production isolation-level speedup. The initial database exit cause remains unresolved, and historical server stderr was not retained; five missing trials were recovered once in a fresh bounded fixture. Raw records, commands and details remain linked from [RESULTS.md](RESULTS.md).

## Historical URL guard fix and retained review items

The final branch review found that nonempty SQLAlchemy URL query options could override the validated asyncpg host, database, or user. The benchmark-only `ensure_disposable_database` guard now rejects any nonempty query before `_make_engine` creates an engine. The focused regression checks those three effective dialect overrides and verifies that engine creation is never reached. Its RED run was `3 failed, 16 deselected`; after the guard, the same command was `3 passed, 16 deselected`. The complete `server/tests/test_event_benchmark.py` run passed **19 tests in 0.27s**. Python compilation passed for the changed benchmark source and test module.

The final scoped Ruff command covered `qa/events/benchmark_export.py` and `server/tests/test_event_benchmark.py` and reported **140 diagnostics**; these are retained QA/test static debt, not a clean check. The portable strict basedpyright reproduction analyzed 19 files in 3.104s and reported **294 errors and 42 warnings** (exit code 1). The full JSON files linked above remain the older, commit-qualified 94f5c9d snapshots; these fresh totals describe the final fix working tree. No type/style cleanup was made in this wave.

The final code review and Task 4 evidence review's important findings are addressed by the URL-query guard and these tracked evidence copies/links. The public validation-artifact copies are redacted versions of existing local evidence; originals were left unchanged. Redactions cover URL user information, the RTSP publisher path token, credential values, and private home/pytest paths. Diagnostic snapshots remain the full 94f5c9d records described above and must not be read as diagnostics for the final fix commit. Existing measurements were not rerun or changed. The final review's minor findings remain deferred: numeric epoch query strings bypass the explicit-offset syntax contract; schema/default/index and stored-secret assertions are weaker than their names suggest; and the disposable integration fixture has no PID bound. These items were not changed in this final narrow fix.
