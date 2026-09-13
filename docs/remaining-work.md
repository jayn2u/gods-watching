# Remaining work

This branch is an implementation checkpoint. Tasks 1-9 and 12 have completed
their task-level acceptance. The items below must be completed before the
system can be presented as production-ready or the original implementation
plan can be closed.

## Immediate verification and integration

- [x] **Task 10: ingest final GPU verification**
  - Fresh `ingest` and `ingest-outage` scenarios passed all 15 checks with real
    evidence on `98894d2` (2026-09-13), including bounded in-flight work,
    generation fencing, reconnect, identity observation, and exact cleanup.
  - Cadence note: 20 s `ingest` dispatch turns were `[100, 95, 100, 99]`
    against 100 expected at 5 Hz. The encoded check enforces the 3.5 fps
    functional floor; the lagging camera (4.75 fps) stays tracked by the
    Task 21 sustained 4.8 fps/camera gate.
- [x] **Task 11: appearance publication integration**
  - Focused appearance suite: `18 passed`. Real `appearance` (4 checks) and
    `appearance-stale` (6 checks) scenarios passed with RTSP, PostgreSQL, and
    Triton CLIP on `98894d2` (2026-09-13): active-before-exit visibility,
    version-consistent JPEG/vector, stale generation/version rejection, border
    upgrade, old-crop GC, orphan and temp recovery, and exact cleanup.
- [ ] **Task 13: retention integration**
  - Complete age/quota eviction, reservation and suppression integration on
    the final appearance pipeline.
  - Exercise crash boundaries, storage-full recovery, physical accounting,
    unrelated-path protection, and restart reconciliation.
- [ ] **Task 14: search API integration**
  - Bind the completed search service and HTTP routes to final appearance and
    retention behavior.
  - Verify text, filtered browse, image similarity, expired seed/crop behavior,
    inference outage handling, stable ranking, and authorization.

## Browser application

- [ ] **Task 15: live wall integration acceptance**
  - Fence session-check callbacks by request ownership so a late initial
    response cannot overwrite a successful login or explicit expiry state.
  - Repeat real Chromium QA against four 1080p H.264 RTSP sources. The stopped
    checkpoint reached the browser harness but failed a harness strict-mode
    selector before any WHEP POST, so it is not product acceptance evidence.
  - Verify rising `framesDecoded`, slot replacement and persistence, source
    loss, logout/expiry teardown, activity cadence, keyboard use, and
    fullscreen behavior.
- [ ] **Task 16: person search UI acceptance**
  - Verify the implemented text search, filters, detail, Find similar,
    back-state preservation, stale-response fencing, and recoverable error
    states against the real API; fix gaps found by that run.
- [ ] **Task 17: camera and retention settings UI acceptance**
  - Fence camera-load callbacks by request ownership so an older response
    cannot replace a newer refresh or post-expiry reset.
  - Run the prepared real browser/API scenario; no runtime acceptance scenario
    was started at this checkpoint.
  - Verify camera create/test/edit/delete, masked credentials, invalid-source
    recovery, detection toggle, retention validation, restart persistence, and
    no worker resurrection.

## Operations and system acceptance

- [ ] **Task 18:** expose truthful per-camera, inference, indexing, storage,
  readiness, and recovery status with bounded reconnect behavior.
- [ ] **Task 19:** implement the currently stubbed `prepare`, `up`, `down`,
  `status`, and `doctor` commands; add the missing root Compose/Caddy lifecycle,
  TLS, production proxy-header policy, and credential setup.
- [ ] **Task 20:** label and run real text/image retrieval-quality cases; meet
  the declared Recall@5 target without synthetic expectations.
- [ ] **Task 21:** run the 15-minute four-stream load test and meet the declared
  detector-rate and search-latency targets with bounded queues and memory.
- [ ] **Task 22:** execute the crash, retention, credential, path, storage-full,
  restart, no-egress, and crop-only storage fault matrix.
- [ ] **Task 23:** complete production-browser fidelity, responsive,
  accessibility, console, resource-leak, and performance QA.
- [ ] **Task 24:** replace the title-only README and stale design-baseline claims
  in `docs/architecture.md`, finish operator and validation documentation, run
  the installed full scenario, and complete CLI negative cases.

## Final gates

- [ ] **F1:** plan and evidence compliance audit on the final commit.
- [ ] **F2:** code-quality, cancellation, transaction, auth, and dependency
  review on the final commit; split oversized mixed-responsibility modules where
  needed and resolve the repository-wide Ruff-format baseline.
- [ ] **F3:** independent end-to-end manual QA from RTSP through detection,
  crop, CLIP, search, similarity, and live viewing.
- [ ] **F4:** scope audit for recording/playback exclusions, private endpoint
  boundaries, UI fidelity, and actual crop-only storage writes.

All final review reports must bind to the exact final commit and evidence
identity. Any source change after a passing report requires the affected check
to be rerun.
