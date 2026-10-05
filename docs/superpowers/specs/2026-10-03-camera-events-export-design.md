# Camera-management events and CSV export

## Approved intent

The user approved camera creation/update/deletion history, authenticated backend
CSV downloads, controlled buffered-versus-streaming measurements, a separate
multi-SELECT Read Committed/Repeatable Read consistency experiment, tests,
review, and a draft PR. No UI, merge, deployment, historical production claim,
or guaranteed performance gain is included. The goal is reproducible numerical
evidence from the actual project implementation for an honestly described
project experiment.

## Existing system and isolation

The repository uses FastAPI, SQLAlchemy 2.0.43, asyncpg 0.30.0, Alembic and
PostgreSQL 17/pgvector. Camera mutations accept a caller-owned AsyncSession;
the enclosing API transaction commits durable state before runtime effects.
Worker diagnostic logging and model transition audit rows already exist.
A general camera-event history and CSV export do not.

Implement on `codex/event-log-export` in the isolated worktree
`/home/jwchoi/Documents/Codex/2026-10-03/task/gods-watching`, based on
`origin/develop` at `09a1517`. Preserve the original checkout's existing edits.
Do not install dependencies, pull images, use GPUs or change global DB settings.
Use existing local tools and disposable synthetic databases only.

## Event storage and recording

Add a migration and mapped `CameraEvent` table. Each immutable event contains
an identity bigint event ID, timezone-aware DB-generated occurrence time,
event type (`camera.created`, `camera.updated`, `camera.deleted`), camera UUID,
and the camera name at that operation (maximum 80 characters). Store no source
URL, username/password, encryption key, authentication session or free-form
configuration payload. Historical camera identity and name remain available
after deletion; no cascade should remove events.

Record events in the camera service, using the same caller-owned transaction
as the mutation. Successful create/change/delete produces exactly one row.
An update that changes nothing produces no event. Version conflicts, failed
probes and rolled-back mutations produce no committed event. A failed event
insert fails the mutation transaction rather than silently losing history.
These events describe durable configuration changes, not successful media
activation or video-analysis detections.

Add only the indexes justified by time/camera-filtered exports and stable
event-ID ordering. Do not introduce an event-broker, background queue,
per-frame logging, arbitrary external event ingestion, retention job or UI.

## Authenticated CSV contract

Add `GET /api/events/export.csv` using the existing operator session guard.
Accept optional `camera_id`, `since`, and `until` filters. Require explicit
timezone offsets for supplied timestamps and reject inverted ranges; use
inclusive `since` and exclusive `until`. A missing filter exports all retained
events. Output columns are `id,occurred_at,event_type,camera_id,camera_name`,
ordered by event ID. Timestamp output uses an unambiguous UTC representation.

Send UTF-8 CSV with a fixed safe download filename and `Content-Disposition`
attachment. Use standard CSV quoting for commas, quotes and newlines. Protect
spreadsheet consumers by prefixing a quote on string cells beginning with
formula/control prefixes (`=`, `+`, `-`, `@`, tab, CR, LF), including leading
whitespace before a formula prefix. The exported name may therefore differ
from the raw name only by this documented protective prefix. Do not log or
return credentials in errors. The endpoint is read-only and never creates
an event for its own export.

The production implementation executes one SQL SELECT in a read-only
Read Committed transaction, with a server-side cursor and fetch batches of
1,000 rows. Encode and emit bounded chunks; do not materialize all ORM
entities, result rows or the whole CSV. The session/cursor must remain open
until streaming ends and close on success, exception or cancellation.
Prepare the SELECT/cursor before emitting CSV bytes so the snapshot already
exists when the consumer observes progress. Ensure DB/query failures before
the first emitted bytes are normal HTTP
errors; a later failure may terminate the download, which is documented and
must release resources. Set a 60-second statement timeout and a five-minute
total export deadline to avoid unlimited resource retention by a slow
consumer; close on client cancellation without waiting for that deadline.

A single Read Committed SELECT already has a consistent statement snapshot.
The streaming optimization addresses buffering and serialization memory;
it is not attributed to a change of transaction isolation.

## Controlled implementation benchmark

Use the actual migration, camera-event recording/repository code, CSV encoder
and authenticated HTTP export path. Run a real loopback HTTP application with
only the services needed by the export route, without media/GPU workers. Seed
synthetic rows in the real event table. Concurrent insertion uses the real
event recording code; separate correctness tests prove the camera CRUD hooks.

Compare the production streaming exporter against a benchmark-only buffered
exporter using the same SELECT, ordering, filtering, CSV encoder and columns.
The buffered implementation reads all rows and builds the complete CSV before
sending it. It is not a public production API option. Document which application
composition is used; do not claim full production runtime benchmarking.

Measure 10,000 and 50,000 preexisting events, three repeats per mode and size.
Run one export at a time with at most one concurrent insertion connection;
limit aggregate DB connections to four. Use fresh app processes for each
sample to avoid inherited RSS high-water marks. Download to a temporary file
without retaining the response in the client. Capture complete download
latency with a monotonic client clock and peak application RSS independently
of client/DB memory. Report bytes, row count, raw sample values, medians and
ranges; three samples do not support a meaningful latency p95 or strong
statistical significance. Record first-byte latency if available separately.

Start a bounded writer after export progress is observed and insert a fixed
batch through the event repository. Verify exact output IDs against the
snapshot that actually began the SELECT, uniqueness and row-to-field
consistency. Concurrent commits after that snapshot must not appear. Record
writer count and timing; verify the inserts really overlap the export. If
export finishes too quickly to establish overlap, report that run separately
rather than falsely claiming concurrent correctness evidence.

Use existing images only. Bound DB CPU to 1 and memory to at most 384MiB;
bound the app/client workload to at most 1 CPU and 256MiB per process/container.
Store DB data in disposable tmpfs capped at 256MiB, with no existing volume
mounts or real credentials. Publish only loopback ports if needed. Clean up
only this task's containers and temporary data. Record image IDs, OS, Python,
PostgreSQL, dependency versions, commands, resource limits, output correctness
and any failures. Preserve raw JSON measurements in the PR evidence folder.

## Separate isolation-consistency experiment

This benchmark-only comparison uses the real event table and repository
projection, not the recommended streaming HTTP export. Each trial performs
an initial event-count query, allows the writer to commit a fixed synthetic
batch, then queries the exported rows in the same transaction. Execute this
identical sequence under Read Committed and Repeatable Read; compare the
initial count with exported IDs/count. Force the interleaving and disclose it.
Use three repeated trials at each dataset size and record writer commits,
count mismatches, elapsed time and any DB errors.

This demonstrates statement-versus-transaction snapshot consistency for
multiple SELECTs. It does not establish the frequency of real-world errors,
improve the single-SELECT export, or justify attributing streaming gains to
isolation. It deliberately measures the separately approved isolation goal.

## Verification and delivery

Follow test-first development: demonstrate failing event atomicity/no-op,
CSV filtering/escaping/auth, cursor cleanup and concurrent snapshot tests,
then implement and verify them. Test migration upgrade/downgrade in a disposable
database. Run the project's Python test command and available static checks;
bound any test-created containers or classify unsafe/unavailable integration
checks as blocked rather than invoking unrelated media services. Report all
failures by name and distinguish baseline failures from new regressions.

Review transaction ownership, authorization, CSV injection, streaming resource
lifetime and honest benchmark comparability. Obtain independent code review
before publishing. Use the human's configured Git author identity and the
repository's established AI attribution convention. Commit/push only this
branch, create a draft PR targeting develop, and check CI for the exact
published commit. No merge or deployment. Explain neutral or worse results
without inventing a gain, and describe evidence as gods-watching project
experimentation rather than historical Innodep production work.
