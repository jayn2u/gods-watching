# Task 3 report: actual export benchmark and review fixes

Implemented the bounded HTTP export benchmark and separate RC/RR multi-SELECT probe. The benchmark-only buffered exporter uses the production `EventRepository` projection and `encode_csv_row`; `benchmark_app.py` mounts the real session and event routers with the real auth and database services. The client uses actual login/cookie authentication, streams the HTTP body to disk, validates each field and ordered ID, and triggers the bounded 100-row writer after reading the first CSV data row. The production route has no benchmark mode.

The smoke passed in both modes with identical 9,834-byte output and SHA-256 `99a073d10efc1924c7e70a84905a7d902f8bf55ce378556ebbb359f687dfc79d`. The measured matrix has 12 correctness-passing HTTP samples (10k/50k rows, two modes, three repetitions per cell). Median total transfer was 0.107754s streaming versus 0.107506s buffered at 10k, and 0.494080s streaming versus 0.465526s buffered at 50k. First-body-byte median was earlier for streaming at both sizes. No overall throughput or memory improvement is claimed. All raw values and per-size/mode summaries are in `docs/experiments/2026-10-03-event-export/results.json` and `RESULTS.md`.

The independent isolation probe has 12 valid trials. Under RC the projection returned N+100 after the writer committed; under RR it returned N, with exact IDs at both sizes. The first bounded run completed all HTTP samples and seven isolation trials, then PostgreSQL exited with code 1 during setup for 50k RC repeat 2. The utility saw an asyncpg connection reset. Exactly the five missing trials were resumed once in a fresh disposable database with the same resource limits; no HTTP sample was rerun. The original 12-sample/7-trial artifact remains preserved in `initial-run-results.json`; recovery details and resource diagnostics are in `isolation-recovery-103e24dc.json`.

Round-one review fixes:

- I1: kept raw timestamps unchanged; separated transaction-activity overlap from commit-inside-transfer. Recalculation yields activity overlap in 11/12 samples and commit inside transfer in 0/12. Every commit followed transfer end by 3.427–35.963 ms. The derived JSON and `RESULTS.md` now state this precisely; no benchmark rerun was performed.
- I2: CSV validation rejects surplus columns (`None` from `DictReader`) as well as missing columns; added a malformed-row regression.
- I3: `isolation_trial_is_valid` is shared by primary-run gates, recovery, merge, and summaries. It checks error absence, exact IDs, initial N, expected export count (N+100 for RC, N for RR), and writer count 100. Invalid trials remain in evidence but do not enter timing summaries; invalid primary results cause a nonzero run.
- I4: added separate labeled stdout/stderr capture for Docker logs while structured Docker commands still return stdout only. The historical run and recovery logs were stdout-only, so the original crash message's absence cannot be inferred; its cause remains unresolved.
- I5: planned app, DB, utility, and client names are registered before launch. Utility/client containers are force-removed in `finally`; cleanup errors and retained ownership are recorded and make the run fail. Mock timeout/cleanup-failure coverage exercises both utility and client launch paths.
- Minor binding exception: the app binds `0.0.0.0` only inside the task-owned `--internal` Docker network with no published ports, as authorized by the controller. This is documented in `RESULTS.md`.

The initial pre-implementation test attempt failed collection because the module did not yet exist; that is not behavioral red-test evidence. Review-fix behavioral regressions then failed as expected (the expanded run showed 7 failed and 8 passed); the focused extra-column case specifically demonstrated that the old validator incorrectly returned `correct=True` and `performance_eligible=True`. Historical Task 2 integration-evidence gaps remain disclosed; this work did not reconstruct them or rerun the stalled full suite.

Validation in the task worktree `/home/jwchoi/Documents/Codex/2026-10-03/task/gods-watching`, using `/mnt/data/gods-watching/.venv/bin/python`:

```text
PYTHONPATH=server/src /mnt/data/gods-watching/.venv/bin/python -m pytest -q server/tests/test_event_benchmark.py
................                                                         [100%]
16 passed in 0.24s

/mnt/data/gods-watching/.venv/bin/python -m py_compile qa/events/benchmark_export.py qa/events/benchmark_app.py qa/events/buffered_export.py server/tests/test_event_benchmark.py
exit 0

/mnt/data/gods-watching/.venv/bin/ruff check --select F,E9 qa/events/benchmark_export.py qa/events/benchmark_app.py qa/events/buffered_export.py server/tests/test_event_benchmark.py
All checks passed!

git diff --check
exit 0
```

An evidence assertion compared every monotonic/UTC timestamp path in the 12 final HTTP samples with the pre-fix committed artifact and found no changes. It also independently recomputed 11/12 activity overlaps, 0/12 commit-inside overlaps, the 0.003426526–0.035963442s post-transfer gap range, and valid isolation counts of 12/12 overall and 5/5 in recovery. No new benchmark measurements were taken for the review fixes.

The full run command was `PYTHONPATH=server/src /mnt/data/gods-watching/.venv/bin/python qa/events/benchmark_export.py run`; the one-time recovery command was `PYTHONPATH=server/src /mnt/data/gods-watching/.venv/bin/python qa/events/benchmark_export.py resume-isolation`. Docker used cached images only, with one CPU per container, DB 384 MiB memory+swap and 256 MiB tmpfs, app/client 256 MiB memory+swap, a task-owned internal network, and no published ports. Alembic head was `0008_camera_events`; mounted source imports were verified under `/work/server/src`.

Limitations: three samples per cell support median and range only; app composition excludes media services and workers; initial database exit cause is unresolved. App `ru_maxrss` exceeded the nominal 256 MiB limit and per-app cgroup peaks were not retained, so charged-memory use cannot be established. Recovery DB measurements are not a diagnosis of the original exit. No memory-gain claim is made. All task-owned containers and networks were removed; unrelated containers were left untouched.

Implementation commit: `96afd34` (`Fix camera event benchmark review findings`).
