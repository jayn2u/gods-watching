# Camera event CSV export benchmark

The production streaming exporter and a benchmark-only buffered exporter returned correct, identical CSV for all sampled rows. In this reduced local environment, streaming delivered the first CSV body byte much earlier, while total transfer time was similar at 10,000 rows and about 6.1% slower at 50,000 rows. The three-repetition samples do not support a general throughput or memory-improvement claim.

## Method and environment

The run used an actual migrated `camera_events` table, the real login/session router and cookie, the authenticated `/api/events/export.csv` HTTP route, the production `EventRepository` projection, and the production CSV encoder. The client streamed the body to a temporary file and validated ordered IDs and every CSV field against a manifest written before the writer transaction. At the start of each HTTP export, a real repository writer inserted 100 events after the client had read the first CSV data row. The production exporter had no benchmark mode; only the reduced benchmark app injected the buffered comparator.

The original run ID was `4bd6ab61`; its follow-up isolation run ID was `103e24dc`. The starting repository revision was `2dbd97a3362061c75397399db17a936dd37c8b13`. Docker server 29.8.1, PostgreSQL 17.8, Python 3.12.11 on the host, and Python 3.12.12 in the cached app image were used. The pinned local images were `pgvector/pgvector:0.8.1-pg17` (`sha256:3e8b3adfd27b5707128f60956f62a793c3c9326ea8cfaf0eab7adccb5d700b21`) and `gods-watching-app:local` (`sha256:d37830285e079b037ad9383eafa3b33ce1a72f36df22083c0c6d652e4512843f`). No image pulls or installs were used. Alembic head was `0008_camera_events`, and the imported auth, model, exporter, and repository modules were all under `/work/server/src`.

The DB was on an internal Docker network with no published ports, 1 CPU, 384 MiB memory, equal 384 MiB memory+swap limit, and a 256 MiB tmpfs data directory. App and client containers each had 1 CPU and equal 256 MiB memory+swap limits. Only task-owned containers and a synthetic database were used; media services and workers were excluded. Exact commands and redacted synthetic credentials are in [`COMMANDS.md`](COMMANDS.md), with the five-trial recovery commands in [`isolation-recovery-103e24dc-commands.md`](isolation-recovery-103e24dc-commands.md).

Both modes passed the 100-row smoke. Each produced 9,834 bytes and SHA-256 `99a073d10efc1924c7e70a84905a7d902f8bf55ce378556ebbb359f687dfc79d`. All 12 measured HTTP samples passed exact ordered-ID and field validation; each writer committed 100 rows. CSV output sizes were 998,936 bytes for 10,000 rows and 5,038,936 bytes for 50,000 rows.

## HTTP results

“First body byte” is measured from request start through receipt of the first response-body byte, which is the start of the CSV header. It is not first-row latency. Total transfer covers the full streamed response. Each cell has three correctness-passing samples; ranges are min–max. Process RSS is Linux `ru_maxrss` in KiB. Baseline was captured after real login in the fresh app process.

| Rows | Mode | Total transfer median (range), s | First body byte median (range), s | Baseline RSS median (range), KiB | Peak RSS median (range), KiB |
|---:|---|---:|---:|---:|---:|
| 10,000 | Streaming | 0.107754 (0.106691–0.158920) | 0.017742 (0.014780–0.018633) | 272,364 (264,764–272,532) | 272,364 (264,764–272,532) |
| 10,000 | Buffered | 0.107506 (0.091024–0.110582) | 0.105886 (0.089433–0.108913) | 270,996 (270,960–272,048) | 270,996 (270,960–272,048) |
| 50,000 | Streaming | 0.494080 (0.487125–0.543026) | 0.037267 (0.015679–0.079051) | 271,696 (271,664–271,980) | 271,696 (271,664–271,980) |
| 50,000 | Buffered | 0.465526 (0.463863–0.493671) | 0.459619 (0.457569–0.487950) | 271,328 (270,852–271,608) | 271,328 (270,852–271,608) |

Streaming reached its first CSV body byte sooner in both dataset sizes. Median total transfer differed by about 0.2% at 10,000 rows; at 50,000 rows, streaming was about 6.1% slower than buffered in this run. With three samples and a reduced app, these are descriptive observations, not evidence of a general gain. Peak RSS equaled the marked post-login high-water baseline for every sample, so this run observed no export-stage high-water increase; it does not establish lower memory use for streaming.

The writer commit overlapped the HTTP transfer in 11/12 samples (streaming 6/6, buffered 5/6). It overlapped app-side stream-pull/fetch-plus-encoding spans in all 6 streaming samples and none of the buffered samples. Database fetch time is not instrumented separately from CSV encoding. Buffered preparation completes before the first response body, so absent stream-pull overlap is expected; no artificial delay was added. Writer trigger was after the client read the first CSV data row, not merely after response headers.

The app Docker run command set `--memory=256m --memory-swap=256m`; commands are preserved. Per-app cgroup `memory.peak` was not captured before those containers were removed. The observed process `ru_maxrss` values (264,764–272,532 KiB) exceed the nominal 256 MiB container setting; without per-app cgroup counters, the RSS-versus-charged-memory difference remains unresolved. No memory improvement is claimed.

## Separate RC/RR multi-SELECT probe

This experiment is separate from the single-SELECT export. Each trial seeded the initial `N` events outside timing, started a transaction, ran `COUNT`, committed a 100-row writer batch, then materialized the repository projection in that transaction. This forced the interleaving and is not a speed comparison between isolation levels.

| Initial rows | Isolation | Trials | Export count each | Exact IDs | Count differs from initial N? | Probe elapsed median (range), s |
|---:|---|---:|---:|---|---|---:|
| 10,000 | READ COMMITTED | 3 | 10,100 | Yes, exact expected IDs | Yes | 0.111054 (0.108182–0.112956) |
| 10,000 | REPEATABLE READ | 3 | 10,000 | Yes, exact expected IDs | No | 0.108456 (0.107629–0.109694) |
| 50,000 | READ COMMITTED | 3 | 50,100 | Yes, exact expected IDs | Yes | 0.186313 (0.186021–0.186670) |
| 50,000 | REPEATABLE READ | 3 | 50,000 | Yes, exact expected IDs | No | 0.186537 (0.186031–0.195411) |

All 12 trials committed the 100-row writer and matched the expected ordered ID list. Under RC the second statement observed the committed inserts; under RR it retained the transaction snapshot. This illustrates the multi-statement snapshot difference only.

The first run completed 7 isolation trials (all six 10k trials and 50k RC repeat 1) after completing all 12 HTTP samples. While setting up 50k RC repeat 2, PostgreSQL exited with code 1 and the setup utility saw an asyncpg connection reset during its 1,000-row-batched seed. Retained Docker events had no OOM event/attribute; the database had been auto-removed, and its log had no crash or tmpfs exhaustion message. The cause is unresolved. The original raw file is retained as [`initial-run-results.json`](initial-run-results.json); the remaining five coordinates were run once in a fresh bounded DB/network, without rerunning any HTTP sample. The final merged results retain the initial failure in `failures` and identify the recovery run.

Recovery diagnostics confirmed effective PostgreSQL Docker limits of 402,653,184 bytes memory and memory+swap, 1,000,000,000 NanoCPUs (1 CPU), and cgroup `memory.max=402653184`. Its observed `memory.peak` was 227,594,240 bytes; swap current was zero. The 256 MiB data tmpfs reached 160,165,888 bytes used (60%) after recovery trial 5. These recovery-container measurements do not prove the initial failure's cause and do not substitute for app cgroup measurements.

## Reproduction and raw evidence

The full run command was `PYTHONPATH=server/src /mnt/data/gods-watching/.venv/bin/python qa/events/benchmark_export.py run`. Focused validator tests passed 8/8 with `PYTHONPATH=server/src /mnt/data/gods-watching/.venv/bin/python -m pytest -q server/tests/test_event_benchmark.py`; `py_compile` passed for the three benchmark modules. The initial pre-implementation pytest failure was collection-only (`ModuleNotFoundError` for the not-yet-created module), not behavioral red-test evidence.

`results.json` contains all 12 HTTP samples, all 12 RC/RR trials, environment, stratified summary, and the original/recovery metadata. `initial-run-results.json` preserves the exact 12-sample/7-trial failed run. `isolation-recovery-103e24dc.json` preserves the five recovered trials and per-trial resource snapshots. `environment.json`, `COMMANDS.md`, and `logs/` retain environment and raw run evidence. Setup issues and their corrections are in [`SETUP_ATTEMPTS.json`](SETUP_ATTEMPTS.json).

All containers and the internal recovery network were removed. The task-owned original benchmark network/containers were also removed; unrelated existing containers were left untouched. The early Task 2 historical integration-evidence gap remains as recorded in the Task 2 report; no historical integration evidence was reconstructed here.
