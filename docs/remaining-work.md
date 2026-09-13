# Remaining work

This branch is an implementation checkpoint. Tasks 1-9 and 12 have completed
their task-level acceptance. The items below must be completed before the
system can be presented as production-ready or the original implementation
plan can be closed.

## Immediate verification and integration

- [ ] **Task 10: ingest final GPU verification**
  - Run fresh `ingest` and `ingest-outage` scenarios after the deadline-aware
    scheduler repair.
  - Confirm the configured 5 Hz per-camera cadence, bounded in-flight work,
    generation fencing, reconnect behavior, identity observations, and exact
    cleanup on the current source hashes.
- [ ] **Task 11: appearance publication integration**
  - Rerun the focused appearance suite after the final fixture-coordinate
    corrections. The preceding run was `2 failed, 16 passed`; the corrections
    have not been rerun.
  - Run both real `appearance` and `appearance-stale` scenarios with RTSP,
    PostgreSQL, and Triton CLIP.
  - Verify active-before-exit search visibility, version-consistent JPEG/vector
    publication, stale-generation rejection, crash recovery, crop GC, and
    orphan reconciliation.
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

- [ ] **Task 15: authenticated live wall**
  - Repeat real Chromium QA against four 1080p H.264 RTSP sources. The stopped
    checkpoint reached the browser harness but failed a harness strict-mode
    selector before any WHEP POST, so it is not product acceptance evidence.
  - Verify rising `framesDecoded`, slot replacement and persistence, source
    loss, logout/expiry teardown, activity cadence, keyboard use, and
    fullscreen behavior.
- [ ] **Task 16: person search UI**
  - Implement and verify text search, filters, detail, Find similar, back-state
    preservation, stale-response fencing, and recoverable error states against
    the real API.
- [ ] **Task 17: camera and retention settings UI**
  - Run the prepared real browser/API scenario; no runtime acceptance scenario
    was started at this checkpoint.
  - Verify camera create/test/edit/delete, masked credentials, invalid-source
    recovery, detection toggle, retention validation, restart persistence, and
    no worker resurrection.

## Operations and system acceptance

- [ ] **Task 18:** expose truthful per-camera, inference, indexing, storage,
  readiness, and recovery status with bounded reconnect behavior.
- [ ] **Task 19:** finish the offline-ready Compose lifecycle, Caddy/TLS,
  production proxy-header policy, credential setup, and CLI commands.
- [ ] **Task 20:** label and run real text/image retrieval-quality cases; meet
  the declared Recall@5 target without synthetic expectations.
- [ ] **Task 21:** run the 15-minute four-stream load test and meet the declared
  detector-rate and search-latency targets with bounded queues and memory.
- [ ] **Task 22:** execute the crash, retention, credential, path, storage-full,
  restart, no-egress, and crop-only storage fault matrix.
- [ ] **Task 23:** complete production-browser fidelity, responsive,
  accessibility, console, resource-leak, and performance QA.
- [ ] **Task 24:** finish operator, architecture, and validation documentation;
  run the installed full scenario and complete CLI negative cases.

## Final gates

- [ ] **F1:** plan and evidence compliance audit on the final commit.
- [ ] **F2:** code-quality, cancellation, transaction, auth, and dependency
  review on the final commit.
- [ ] **F3:** independent end-to-end manual QA from RTSP through detection,
  crop, CLIP, search, similarity, and live viewing.
- [ ] **F4:** scope audit for recording/playback exclusions, private endpoint
  boundaries, UI fidelity, and actual crop-only storage writes.

All final review reports must bind to the exact final commit and evidence
identity. Any source change after a passing report requires the affected check
to be rerun.
