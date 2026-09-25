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
- [x] **Task 13: retention integration**
  - Ingest, appearance publication, and retention run in a separate
    `gods-watching worker` process that follows the API's committed camera
    sessions (`server/src/gods_watching/pipeline_worker/`); the API hands
    detector effects to it through `CommittedStateDetector`.
  - Real `retention` (8 checks) and `retention-crash` (6 checks) pass on
    `b2d9f22` against the real worker: age and oldest-first quota eviction,
    suppressed active victims, managed bytes within budget, no broken search
    references, untouched unrelated paths and symlinks, graceful SIGTERM, and
    SIGKILL restart with orphan/temp reconciliation, GC replay, and resumed
    publishing.
  - The run found and fixed publisher startup cleanup deleting a crop-named
    symlink. Storage-full recovery beyond the quota budget stays with Task 22.
- [x] **Task 14: search API integration**
  - Real `search` (7 checks) and `search-errors` (5 checks) pass on `b2d9f22`
    over appearances published by the real worker with Triton CLIP text
    search: 401/403 authorization, camera and time filtered browse, ranked and
    repeatable text and similar search excluding the seed, private JPEG crops,
    422 invalid input, 404 for a seed and crop expired by real retention, and
    text 503 while browse and similar keep answering during an inference
    outage.

## Browser application

- [ ] **Task 15: live wall integration acceptance**
  - [x] Fence session-check callbacks by request ownership so a late initial
    response cannot overwrite a successful login or explicit expiry state.
  - [x] On 2026-09-20 the current-source `media` scenario passed all 5 real
    RTSP/WHEP/Chromium checks (run `4680d493a97cd4b565d4c240`): decoded frames
    rose from 1 to 4, H.264 was negotiated, receive-only teardown succeeded,
    the gateway filesystem stayed recording-free, and the finite publisher
    reopen boundary held. The live-wall source suite also passed both Chromium
    cases for four durable slots, replacement persistence, truthful offline
    state, and fullscreen controls. These prove the media path and UI behavior
    separately, not the remaining real-app integration gate.
  - [x] The current-source `camera-auth-live-boundaries` scenario passed all 21
    real-runtime checks (run `519a8cc7ad7798c016cb43bc`), including decoded
    frame shutdown on idle/absolute expiry, reader cleanup, delete without
    resurrection, credential replacement, unsupported-codec rejection, and
    authenticated WHEP PATCH ownership. This exercises the real API/media
    boundary but does not render the production React wall.
  - [x] The current-source `media-denied` scenario passed all 4 real-runtime
    checks on 2026-09-20 (run `6669e46b1789cd762a4dab0e`): an absent session
    was denied with HTTP 401, uncredentialed private RTSP read and publish both
    failed, no ephemeral credential persisted in the evidence, and a finite
    publisher closed and reopened cleanly across two generations.
  - [x] On 2026-09-20 the deployed production React application was exercised
    through the TLS gateway against its real session, camera, settings, and
    activity endpoints at 375/768/1280 px. Nine fresh captures in
    `/tmp/gw-task23-production-readonly` had no browser or failed-API errors and
    two independent visual reviews passed. The runtime had zero configured
    cameras, so this proves the real empty-state shell and API reads, not live
    four-source playback or wall interactions.
  - Repeat real Chromium QA against four 1080p H.264 RTSP sources. The stopped
    checkpoint reached the browser harness but failed a harness strict-mode
    selector before any WHEP POST, so it is not product acceptance evidence.
  - Verify rising `framesDecoded`, slot replacement and persistence, source
    loss, logout/expiry teardown, activity cadence, keyboard use, and
    fullscreen behavior.
- [x] **Task 16: person search UI acceptance**
  - `web/e2e/search.spec.ts` runs against the built web app and real API on
    one origin through the `search-ui` scenario, which passes (4 checks) on
    `b2d9f22`: text search with a camera filter, detail, Find similar, back
    with query and filters preserved, latest-of-two rapid searches, and an
    inference outage shown with Retry that recovers once Triton returns.
  - Search failures now render actionable operator wording without exposing API
    codes, including inference outages, expired crops, unknown cameras, invalid
    text, and mismatched response modes. Retry behavior remains explicit.
- [ ] **Task 17: camera and retention settings UI acceptance**
  - [x] Fence camera-load callbacks by request ownership so an older response
    cannot replace a newer refresh or post-expiry reset.
  - [x] The isolated source-QA browser suite passes all 7 Chromium scenarios:
    test/create boundaries and credential redaction, 422 and stale-version draft
    recovery, reversible deletion, retention validation, client-side rejection,
    session expiry, and responsive captures at 375/768/1280/1440 px. This suite
    intercepts API responses and is not the real backend integration gate.
  - [x] The current-source `camera-auth` scenario passed all 13 real integration
    checks (run `d47c2a0ed843874c919d7173`): real H.264 source probe, PostgreSQL
    migration, camera test/delete, invalid source/edit preservation, durable
    settings, secure cookie flags, credential replacement/rollback, authenticated
    browser decode, logout, and reader cleanup. Its browser is a verification
    client rather than the production camera/settings React screens.
  - [x] The current-source `camera-auth-denied` scenario passed all 9 real
    authorization and validation checks (run `d1a88d9b06b60bd26500d14b`):
    anonymous API denial, cross-origin login and mutation denial, login
    throttling, malformed-source rejection, passive polling without session
    refresh, the browser driver, real H.264 input, and real PostgreSQL migration.
  - [x] The same deployed production-browser pass rendered the Cameras screen
    and loaded real camera/settings responses successfully at 375/768/1280 px,
    with no browser or failed-API errors. It was deliberately read-only and the
    runtime had no configured cameras, so CRUD, source testing, credential
    masking, retention save, and restart persistence remain acceptance work.
  - Verify camera create/test/edit/delete, masked credentials, invalid-source
    recovery, detection toggle, retention validation, restart persistence, and
    no worker resurrection.

## Operations and system acceptance

- [x] **Task 18:** expose truthful per-camera, inference, indexing, storage,
  readiness, and recovery status with bounded reconnect behavior.
  - [x] `GET /api/status` is authenticated and now reads bounded worker
    heartbeats plus per-camera ingest snapshots from PostgreSQL. It reports
    source session, frame freshness, decode and accepted-detector rates, drops,
    detector requests/results, sanitized source errors, embedding queue depth,
    last searchable latency, Triton readiness, persistence pressure, and live
    storage accounting. A stopped worker degrades within five seconds and
    recovers after restart; the expanded contract and migration were exercised
    through the Compose/Caddy surface on 2026-09-20. Paused publication work is
    retried by a bounded worker-owned drain after storage pressure clears.
- [x] **Task 19:** implement the currently stubbed `prepare`, `up`, `down`,
  `status`, and `doctor` commands; add the missing root Compose/Caddy lifecycle,
  TLS, production proxy-header policy, and credential setup.
  - [x] Root Compose/Caddy packaging and the `prepare`, `up`, `down`, `status`,
    and `doctor` CLI commands are implemented. Real launcher QA completed a
    cached pinned-image build, full down/up cycle, GPU/runtime doctor, and
    healthy service status on 2026-09-20.
  - [x] LAN TLS is available through `compose.tls.yaml` with a persistent Caddy
    internal CA, HTTP redirect, and secure cookies. The pinned gateway's default
    proxy policy overwrites untrusted forwarding identity, while the API trusts
    only the fixed gateway address. `prepare` atomically creates a mode-0600
    credential `.env` with random operator/database secrets and a Fernet key,
    never overwrites it, and `doctor` validates it without rendering secrets.
- [ ] **Task 20:** label and run real text/image retrieval-quality cases; meet
  the declared Recall@5 target without synthetic expectations.
  - [ ] `retrieval-quality` and `retrieval-negative` remain intentional Task 20
    `scenario_unavailable` placeholders; no scenario module currently replaces
    them in the verification registry.
  - [ ] Create `qa/retrieval-cases.json` before evaluating results. It must bind
    at least 40 real appearance/distractor crops from at least two prepared
    scenes to source/crop hashes and complete relevance sets for at least 20
    English text queries and 20 held-out image queries. Labels must be frozen
    before retrieval output is inspected; failed queries cannot be discarded.
  - [ ] Implement both scenarios against that immutable label corpus and report
    text and image macro Recall@5 independently, requiring at least 0.8 for
    each. The four prepared fixture videos are present, but their existence is
    not a substitute for human-grounded relevance labels.
- [ ] **Task 21:** run the 15-minute four-stream load test and meet the declared
  detector-rate and search-latency targets with bounded queues and memory.
  - [ ] `load` and `overload` are still planned `scenario_unavailable`
    placeholders in the current verification registry; there is no executable
    sustained-load driver from which the 15-minute acceptance can be claimed.
- [ ] **Task 22:** execute the crash, retention, credential, path, storage-full,
  restart, no-egress, and crop-only storage fault matrix.
  - [x] Task 13 real scenarios cover graceful/crash worker cleanup, retention
    recovery, orphan/temp reconciliation, and resumed publication. On
    2026-09-20 the packaged worker was rebuilt from the current source; a
    controlled stop changed the authenticated public status to `degraded`, and
    restart returned it to `ready` with inference available and persistence
    unpaused. Current-source `retention` replays also found and fixed a
    cancellation window between tracker mutation and terminal handoff delivery,
    plus a late-reconcile race that could add a worker after the coordinator's
    shutdown snapshot. The final replay passed all 8 checks, including graceful
    SIGTERM with no open appearances and exact owned-resource cleanup (run
    `642fa37301ef9a709b4b0289`).
  - [x] Credential preparation and mode/placeholder rejection plus crop object
    key traversal rejection have boundary tests. The live crop volume was also
    inspected read-only from a `--network none` container without exposing any
    non-JPEG media, but it was empty and therefore is not non-vacuous write
    evidence.
  - [ ] Execute storage-full recovery, whole-stack no-egress startup, and a
    non-empty crop-only volume audit from a real publication run; bind the
    consolidated `faults` and `restart` scenario evidence to the final source.
- [ ] **Task 23:** complete production-browser fidelity, responsive,
  accessibility, console, resource-leak, and performance QA.
  - [x] Search recovery states passed current-source Chromium QA on 2026-09-20:
    six fresh screenshots covered inference-unavailable and crop-expired states
    at 375/768/1280 px; 17 browser cases passed with 3 environment-gated real
    API cases skipped, no console errors, no clipping or horizontal overflow,
    and exact URL cleanup. Known and unexpected failures expose operator wording
    rather than internal codes, response-mode mismatches are covered, and each
    failure owns exactly one `role=alert` announcement. Two independent visual
    review passes returned PASS with no blockers.
  - [x] A deployed production build was then exercised through the real TLS
    gateway at 375/768/1280 px across Live wall, Person search, and Cameras.
    Nine fresh captures plus `observation.json` recorded five successful real
    API requests, no failed requests, no browser errors, and exact viewport
    containment. Two fresh independent visual reviews returned PASS with no
    product or evidence blockers. This is truthful zero-camera/read-only
    evidence; configured-camera interactions, screen-reader validation on a
    populated runtime, populated-result/live-media resource lifetimes, and
    production performance/Lighthouse still remain.
  - [x] The deployed build passed a separate keyboard and reflow audit across
    all three screens. At the 640 CSS-px layout equivalent of a 1280px viewport
    at 200% zoom, every screen stayed horizontally contained; all 38 visible
    controls had accessible names, IDs were unique, and every enabled sequential
    Tab stop was visited with visible focus and wraparound. Native segmented
    datetime and radio-group traversal were covered, including Search and Browse
    latest after both datetime fields. At 768 px the Search and Cameras internal
    panes reached their exact scroll ends. Five real API requests returned 200,
    browser/API error lists were empty, and two fresh independent visual reviews
    returned PASS. This does not substitute for screen-reader semantics on a
    populated runtime, configured-camera interactions, populated-result/live-media
    resource lifetimes, or production performance/Lighthouse.
  - [x] A separate Chrome Accessibility Tree audit exercised the authenticated
    deployed Live, Search, and Cameras screens. Each exposed a named `Primary`
    navigation landmark, a screen-specific named main landmark, and named
    `Camera tree` and `Events` complementary landmarks. The expected headings
    and `Gods Watching` root name were present, and none of the 50 exposed
    control nodes (15/25/10 by screen) were focusable without an accessible
    name. The run had no browser errors or failed responses. This validates the
    browser accessibility semantics of the zero-camera states; it is not a
    substitute for an assistive-technology session over populated and dynamic
    states.
  - [x] A production-browser resource audit ran three independent 100-screen
    cycles and one 500-cycle slope run against the deployed TLS build. Across
    the comparable Live checkpoints at 10, 100, and 250 cycles, DOM nodes and
    event listeners remained exactly `274` and `173`; the document count made
    one transition from `1` to `2` after the first Search/Cameras mount and then
    stayed flat through 500 cycles. A 500-click active-screen control lane kept
    its document/node/listener counts exactly constant. The three-run median
    screen transition was 45.28 ms and median p95 was 54.53 ms, with no browser
    errors or failed responses. Heap growth also occurred in the no-op control,
    so it is not attributed to screen-transition leakage. This closes only the
    zero-camera screen-cycle DOM/listener slice; populated results, Blob URL and
    live-media resource lifetimes still require runtime coverage.
  - [ ] The current React Doctor 0.9.13 audit completes with 0 errors but 18
    warnings, including performance findings, so the static React performance
    gate is not clean. An authenticated audit using an official, temporary
    Chrome Stable 153.0.8010.52 binary ran three times per preset against the
    deployed TLS build. Median scores (performance/accessibility/best
    practices/SEO) were mobile `99/100/100/90` and desktop `100/100/100/90`.
    Every run reported the missing document meta description; mobile also had
    sub-perfect FCP and occasional LCP/CLS, while unused JavaScript, cache
    lifetime, dependency-tree, and render-blocking diagnostics remain. A
    follow-up response audit confirmed the two deterministic causes: both the
    source and deployed HTML omit a meta description, and production serves the
    hashed JavaScript and CSS through Starlette with validators but no
    `Cache-Control` header. The deployed build is also a single 260,925-byte
    JavaScript entry plus one 46,014-byte stylesheet, which matches the unused
    JavaScript and render-blocking diagnostics but does not by itself prescribe
    the eventual split. This is a real-Chrome baseline and root-cause record,
    not a pass: every category must reach 100 on both presets and the React
    static/runtime gates must also clear.
- [ ] **Task 24:** finish operator handoff and the installed full scenario.
  - [x] The README now documents preparation, HTTP/LAN TLS operation, status,
    verification evidence, recovery, destructive volume removal, and the
    minimum operator acceptance flow. `docs/architecture.md` now distinguishes
    implemented runtime from the remaining evidence gates instead of describing
    the repository as a scaffold.
  - [x] Installed-launcher negative acceptance passed on 2026-09-20: `full`
    fails closed as `scenario_unavailable`, an unknown valid name reports
    `unknown_scenario`, and a malformed name reports `invalid_scenario`; all
    returned exit 1 and secret-free structured evidence.
  - [ ] Implement and run `full` only after Tasks 15, 17, and 20–23 have supplied
    their required real evidence. Enabling it earlier would turn missing quality,
    load, fault, and browser acceptance into a false success.

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

## 학습된 CLIP 가져오기 release gate (2026-09-25)

- 실제 CUHK-PEDES held-out 보고서와 dataset SHA-256에 맞는 deployment policy, 같은 package에 결속된 실사용 crop 검색 증거를 준비하고 두 quality gate 결과를 확인한다. 실제 checkpoint와 검색 사례는 이 작업 공간에 제공되지 않았다.
- 전체 전환 rehearsal 생산자 `models rehearse-switch`와 `full_transition_rehearsal_v2` 소비자는 구현됐다. 실제 fine-tuned checkpoint, 오프라인 DB dump·crop snapshot·카메라 암호화 키 파일이 제공되지 않아 배포 GPU에서 detector를 함께 올린 target package·현재 보존 corpus의 실측 증거와 900초 이내 gate는 아직 확인하지 못했다. embedding-only benchmark는 이 증거를 생성하지 않는다.
- API/worker 재시작 중 reindex 및 activation 후 복구, 이전 package로의 rollback, exact package identity 재발견을 통합 환경에서 검증한다. 4대 카메라 15분 부하 gate도 실제 target GPU에서 별도로 실행해야 한다.
