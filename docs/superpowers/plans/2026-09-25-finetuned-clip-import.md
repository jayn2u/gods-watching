# Fine-tuned CLIP Package Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register and apply an externally fine-tuned CUHK-PEDES CLIP ViT-B/16 package for English person search while preserving the existing safe model switch.

**Architecture:** A local import command atomically publishes a verified, immutable model package into the shared asset cache. API and worker construct one catalog from built-in and imported manifests. GPU and two independent quality gates make a model eligible; a measured preflight bounds manual global activation before the existing durable reindex/rollback path runs.

**Tech Stack:** Python 3.12, Typer, Pydantic, FastAPI, SQLAlchemy/PostgreSQL/pgvector, Triton Python backend, Transformers `CLIPModel`/`CLIPProcessor`, React/TypeScript.

**Spec:** `docs/superpowers/specs/2026-09-25-finetuned-clip-import.md`

## Global Constraints

- No CUHK-PEDES image, caption, identity, or gallery content enters the product runtime.
- Custom package v1 is ViT-B/16, 512-D, complete paired encoders/processor, `safetensors` only, without custom executable model code.
- No browser checkpoint upload or runtime network download; import and validation use local files.
- Apply is manual and global; model spaces remain separated; search/analysis pause during reindexing.
- Candidate must improve on fixed CUHK-PEDES held-out results and fixed product cases; product text and image macro Recall@5 each reach 0.8.
- A measured target-GPU estimate over 15 minutes, or missing measurement, blocks apply.
- The target-GPU four-camera gate remains 4.8 accepted detector fps per camera and p95 first-searchable latency at most 5 seconds.
- Use the repository's installed `./.venv/bin/pytest` and `web/node_modules/.bin` tools when available; do not describe an environment-blocked full suite as passing.

## Review Focus

- Malformed or malicious local package paths, symlinks, missing files, and pickle weights must fail without publishing partial assets (Task 1 tests).
- Two different model revisions with the same 512-D size must never share search vectors or cache entries (Task 2 tests).
- A stale, missing, or package-mismatched quality report must block apply even if GPU preparation passes (Task 3 tests).
- Missing throughput evidence or a corpus estimate above 15 minutes must reject apply before maintenance starts (Task 4 tests).
- Worker/API restart during or after switching must restore the exact imported package and durable transition state (Task 5 integration tests).

---

### Task 1: Immutable offline package import

**Files:**
- Create: `server/src/gods_watching/model_selection/imported_manifest.py` — strict manifest schema and file hash verification.
- Create: `server/src/gods_watching/model_selection/importer.py` — safe local copy, validation, and atomic publication.
- Modify: `server/src/gods_watching/setup/model_cli.py` — `import` command with source and asset-root options.
- Modify: `server/src/gods_watching/cli.py` — expose the import subcommand through `./gods-watching`.
- Test: `server/tests/test_model_package_import.py`.

**Interfaces:**
- `ImportedClipManifest` records `model_id`, `revision`, `display_name`, `base_model_id`, `dimension`, `files`, `package_sha256`, and `cuhk_report`.
- `import_clip_package(source: Path, assets_root: Path) -> ImportedClipManifest` returns the published manifest or raises a typed import error.
- Published files live at `assets_root / "imported" / package_sha256`; a manifest in that directory is the only registration record.

- [ ] Write tests with a minimal local package fixture proving accepted 512-D `safetensors` metadata, idempotent reimport, and rejection of changed bytes under the same ID, symlinks, path traversal, missing tokenizer/config, `.bin`/pickle weights, and failed atomic copy. Assert the asset catalog contains no partial directory after every rejected case.
- [ ] Run `./.venv/bin/pytest server/tests/test_model_package_import.py -q`; confirm the new cases fail before implementation.
- [ ] Add strict parsing and stream hashing; reject every path not contained within the source directory and every file that is not a regular file. Validate exact required Transformers file types and refuse custom model code. Copy into a private temporary directory on the same filesystem, fsync files and directory, then rename to the content-addressed destination. Return stable error codes without leaking server paths through API responses.
- [ ] Add `./gods-watching models import --source <local-directory> --assets <asset-root>` and JSON output containing model ID, immutable revision, and package hash. Re-running the command with identical bytes returns the same record.
- [ ] Run the focused pytest file and `git diff --check`; commit the task on the dedicated branch with the human Git identity and required AI co-author trailer.

### Task 2: One catalog and safe Triton loading across processes

**Files:**
- Modify: `server/src/gods_watching/model_selection/registry.py` — combine built-ins with imported manifests at composition time.
- Modify: `server/src/gods_watching/model_selection/assets.py` — verify imported package preparation separately from the built-in lock.
- Modify: `server/src/gods_watching/api/production.py` — load the shared catalog during API composition.
- Modify: `server/src/gods_watching/pipeline_worker/app.py` — load the same catalog during worker composition.
- Modify: `server/src/gods_watching/setup/model_preparation.py` — GPU proof for imported paired encoders with detector resident.
- Modify: `inference/models/clip_image/1/model.py` and `inference/models/clip_text/1/model.py` — explicitly refuse executable remote model code and verify paired identity.
- Test: `server/tests/test_model_registry.py`, `server/tests/test_prepared_model_catalog.py`, `server/tests/test_model_preparation_lifecycle.py`, `server/tests/integration/test_model_selection.py`.

**Interfaces:**
- `load_clip_registry(assets_root: Path) -> ClipModelRegistry` includes all built-ins plus validated imported manifests in deterministic order.
- Each imported `ClipModelPackage` uses the package's content-addressed `/models/imported/<package_sha256>` path and exact revision. `ClipRuntimeManager.load_model(package)` remains the serving entrypoint.

- [ ] Add tests for API/worker registry equality, restart discovery, duplicate ID with different hash, tampered installed file, image/text dimension mismatch, exact revision mismatch, and two distinct 512-D revisions not sharing vectors. Confirm GPU proof tests require both modalities and detector residency.
- [ ] Run the named focused tests and confirm new expectations fail.
- [ ] Build the registry at process composition from the immutable asset directory; keep built-in lock validation intact. Make prepared status verify the imported manifest and bytes. Load both Triton modalities from the same package path with `trust_remote_code=False`, verify the returned runtime identity, and keep Triton's mount read-only.
- [ ] Run the focused tests, `./.venv/bin/ruff check server inference`, and `git diff --check`; commit the task.

### Task 3: Independent quality evidence and eligibility

**Files:**
- Create: `server/src/gods_watching/model_selection/quality.py` — report schema, package binding, and eligibility decision.
- Create: `qa/clip/evaluate_product_cases.py` — run built-in and candidate embeddings against fixed `qa/retrieval-cases.json` and emit text/image macro Recall@5 with input hashes.
- Modify: `server/src/gods_watching/model_selection/service.py` — reject apply when either report or GPU proof is missing, stale, or failing.
- Modify: `server/src/gods_watching/contracts/model_selection.py` — expose quality status and reason in `ModelCatalogEntry`.
- Test: `server/tests/test_model_quality_gate.py`, `server/tests/test_model_selection_routes.py`, `server/tests/integration/test_model_selection.py`.

**Interfaces:**
- `QualityEvidence` binds candidate package hash, baseline identity, CUHK split/protocol and score, product cases hash, text/image baseline and candidate Recall@5, evaluation code revision, and creation time.
- `assess_quality(manifest: ImportedClipManifest, evidence: QualityEvidence) -> QualityStatus` is pure and fails closed on any missing or mismatched binding.
- `ModelCatalogEntry` gains `quality_passed: bool` and `quality_reason: str | None`; `prepared` continues to mean installed/GPU validated, not quality approved.

- [ ] Add tests asserting CUHK train results do not count as held-out test results; changed model bytes, dataset hashes, baseline identity, or evaluator revision invalidate a report; product text and image each require Recall@5 >= 0.8 and improvement over the same baseline; missing real-crop cases block apply. Include API 422 for an ineligible model.
- [ ] Run focused tests to observe the expected failures.
- [ ] Implement strict evidence parsing and local product-case evaluation. Use only predeclared relevance sets; reject missing, duplicate, or altered case IDs and any case-set hash mismatch. Store records outside the model weight directory so the immutable package hash stays stable. Report CUHK and product results separately in CLI output and catalog status.
- [ ] Run focused tests and the evaluator once against available fixed fixtures. If real `qa/retrieval-cases.json` is absent, record the gate as blocked; do not generate synthetic success evidence or allow apply.
- [ ] Run `git diff --check`; commit the task.

### Task 4: Measured 15-minute preflight before queueing

**Files:**
- Create: `server/src/gods_watching/model_selection/preflight.py` — retained-crop count, skip scan, measured throughput, and duration estimate.
- Modify: `server/src/gods_watching/model_selection/service.py` — call preflight before `create_job`.
- Modify: `server/src/gods_watching/contracts/model_selection.py` and `server/src/gods_watching/api/model_routes.py` — return estimate and typed rejection reason.
- Test: `server/tests/test_model_preflight.py`, `server/tests/test_model_selection_routes.py`, `server/tests/integration/test_model_selection.py`.

**Interfaces:**
- `SwitchPreflight` contains `target_model_id`, `retained_count`, `estimated_missing_count`, `measured_crops_per_second`, `estimated_seconds`, `max_seconds=900`, and `eligible`.
- `estimate_switch(retained_count: int, measured_crops_per_second: float | None, estimated_missing_count: int) -> SwitchPreflight` rejects nonfinite/nonpositive rates and estimates over 900 seconds.
- An authenticated `GET /api/settings/models/preflight?model_id=...` returns current preflight; apply recalculates it inside the server boundary before queueing.

- [ ] Test zero, unknown, NaN, and stale GPU throughput; count changes between preview and apply; 900-second boundary; skipped/corrupt crop counts; and a rejected switch leaving no queued job or maintenance state.
- [ ] Run focused tests to confirm failures.
- [ ] Measure target embedding throughput on the deployment GPU with the detector resident; persist device, package identity, sample count, and measured time. Count retained, nondeleted appearances from the repository, inspect available crop objects, calculate estimate, and make apply repeat the check. Refuse a switch without trustworthy measurement.
- [ ] Run focused tests, `git diff --check`, and commit the task.

### Task 5: Selector, recovery, and release proof

**Files:**
- Modify: `web/src/features/settings/ModelSelector.tsx` and `web/src/features/settings/settings.css` — quality state, estimate, skip preview, and confirmation.
- Modify: `web/src/app/clientTypes.ts`, `web/src/app/client.ts`, and `web/src/features/cameras/cameraTypes.ts` — consume new catalog and preflight contracts.
- Modify: `web/e2e/model-selector.spec.ts` — unavailable candidate, eligible apply, estimate refresh, rollback.
- Modify: `README.md`, `docs/architecture.md`, `docs/remaining-work.md` — import/export contract and verified status.
- Test: `server/tests/integration/test_model_selection.py` and existing model selector E2E.

**Interfaces:**
- Existing `POST /api/settings/models/apply` remains the only activation mutation.
- The selector consumes `ModelCatalogEntry.quality_passed`, `quality_reason`, and `SwitchPreflight`; it does not accept model files.

- [ ] Add UI tests for prepared but quality-blocked models, missing measurement, estimate over 15 minutes, a changed estimate before confirmation, durable progress after refresh, and visible skip reasons. Add integration coverage for API/worker restart during reindex and after activation, rollback to the previous package, and exact package identity after recovery.
- [ ] Run focused frontend and integration tests to see the missing behavior fail.
- [ ] Implement status copy and preflight polling; keep apply unavailable until both prepared and quality approved. Display retained count, estimate, expected skips, and the analysis/search pause before confirmation. Reuse current durable transition display rather than creating a second state machine.
- [ ] Run `web/node_modules/.bin/tsc --noEmit` with the repository's configured project, focused frontend tests, model selector E2E, focused backend integration, and `git diff --check`. Run the existing four-camera 15-minute load gate on the target GPU; report actual outcome and environment blocks separately.
- [ ] Document actual commands, package example, rollback, proof artifacts, and any unfinished gate. Commit the task. Link the branch or PR for review only after the verified outcome is accurately stated.

## Self-review before execution

- Check each requirement in the linked spec against Tasks 1–5, including offline import, two quality gates, 15-minute preflight, durability, and no product CUHK data.
- Check the plan for unresolved placeholders and interface-name drift.
- Confirm each Review Focus case has a test in its owning task.
- Confirm the target GPU and real retrieval cases exist before claiming the release gates pass.
