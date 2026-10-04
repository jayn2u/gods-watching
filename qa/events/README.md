# Event export benchmark

`benchmark_export.py` runs a reduced, real HTTP benchmark for the camera event CSV route. It applies the repository migrations to a task-owned PostgreSQL database, creates an operator through the normal session router, logs in over HTTP, and downloads the export to disk using `http.client` streaming. `benchmark_app.py` composes the production session and event routers with the real auth/database services. `buffered_export.py` is the benchmark-only full-materialization comparator; it uses the same repository query and CSV encoder as the production streaming exporter.

From the repository root, run the focused correctness checks with:

```sh
PYTHONPATH=server/src /mnt/data/gods-watching/.venv/bin/python -m pytest -q server/tests/test_event_benchmark.py
```

Run the 100-row HTTP smoke check with `PYTHONPATH=server/src /mnt/data/gods-watching/.venv/bin/python qa/events/benchmark_export.py run --smoke-only`. The full 12-sample HTTP matrix and separate 12-trial RC/RR probe use `PYTHONPATH=server/src /mnt/data/gods-watching/.venv/bin/python qa/events/benchmark_export.py run`. `resume-isolation` is a guarded recovery command: it refuses to run unless the recorded results contain exactly 12 HTTP samples and the expected 7/12 isolation state, then runs only the five missing 50k trials. Planned container names enter the ownership registry before launch; utility and client containers are force-removed in `finally`, and cleanup failures remain recorded and make the runner exit nonzero. All Docker containers are task-owned, network-isolated, use cached image IDs, and have explicit CPU/memory limits.

Raw samples, environment, exact Docker commands, run-recovery evidence, and interpretation are in [`docs/experiments/2026-10-03-event-export`](../../docs/experiments/2026-10-03-event-export/RESULTS.md).
