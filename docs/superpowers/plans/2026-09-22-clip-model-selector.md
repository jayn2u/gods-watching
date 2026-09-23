# CLIP Model Selector Implementation Plan

> **For agentic workers:** Code must be written by GPT Luna at max reasoning effort, as requested by the user. Execute disjoint runtime, preparation, transition, and UI ownership sets with superpowers:dispatching-parallel-agents once their interfaces are fixed; review and verify the integrated result before completion.

**Goal:** Deliver a working global CLIP selector with durable reindexing and rollback.

**Architecture:** A model registry and offline prepared assets feed explicit Triton model loading. A database-backed transition coordinates worker maintenance, staged vectors, atomic activation and API search gating. The settings screen consumes authenticated model catalog and transition endpoints.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy/asyncpg, pgvector, Triton, React/TypeScript.

**Spec:** docs/superpowers/specs/2026-09-22-clip-model-selector.md

**Execution refinement:** Task 1 is split into runtime/registry, offline preparation,
and durable transition domains. The frozen registry and HTTP contracts allow those
domains and Task 2 to proceed without editing each other's files. Task 3 remains
the final integration and review gate.

## Global Constraints

- Work on codex/clip-model-selector in the existing clean checkout; do not switch other branches or edit unrelated work.
- Use GPT Luna Max for all source code, test code and implementation fixes.
- Preserve configured human Git identity; do not commit with invented agent email addresses. Leave changes uncommitted unless valid attribution is known.
- Default remains openai/clip-vit-base-patch16. Initial additional entries are openai/clip-vit-base-patch32 and openai/clip-vit-large-patch14.
- Prepare immutable snapshots ahead of use. No remote-code loading or runtime downloads.
- Preserve existing data and old vectors until atomic activation. Recovery must not resume analysis with a mismatched runtime identity.
- No Docker socket exposure to the API. Use Triton model management and worker lifecycle control.

## Review Focus

- In-flight search and publication crossing a switch must not mix model spaces.
- Crash before and after atomic activation must restore a consistent runtime/database pair.
- Missing crops are skippable; infrastructure or inference failure is not mislabeled crop corruption.
- Retention must not delete/revive staged or original data incorrectly during maintenance.
- Authentication, repeated apply requests, unsupported models and unprepared assets are checked server-side.

### Task 1: Server, storage, model preparation and inference lifecycle

**Files:** server/src/gods_watching/model_selection/ (new focused package), server/src/gods_watching/contracts/, server/src/gods_watching/api/, server/src/gods_watching/pipeline_worker/, server/src/gods_watching/inference/clip.py, server/src/gods_watching/search/, server/src/gods_watching/storage/, server/alembic/ or existing migration location, server/src/gods_watching/setup/, inference/models/, deploy/Dockerfile.triton, compose.yaml, assets/models.lock.json, corresponding server/tests/.

**Interfaces:** Publish an authenticated catalog/status endpoint and an apply endpoint under /api/settings/models; provide a concrete response contract to Task 2. Catalog includes display name, model identity, dimension and prepared availability. Status includes active model, target, phase, processed/total/skipped and actionable outcome. Worker owns the transition; API submits and polls it.

- [ ] Inspect existing migration, auth, pipeline shutdown, retention, preparation and search contracts.
- [ ] Add failing behavioral tests for catalog validation, dynamic dimensions, staged activation, skip handling, rollback, interruption recovery, concurrent requests and maintenance search/publication gating.
- [ ] Implement the registry, offline asset preparation and explicit Triton runtime selection without hardcoded checkpoint identity.
- [ ] Implement migration and model-aware vector handling, durable switch state, worker execution, API integration and safe startup reconciliation.
- [ ] Run focused tests then the server suite, lint/type checks and migration checks. Record actual commands and results.
- [ ] Provide response examples and UI integration instructions in the task report. Do not leave dead code or a mock-only transition engine.

### Task 2: Settings UI and operator documentation

**Files:** web/src/features/settings/, web/src API client/types and existing settings mount, relevant web tests, README.md and deployment documentation.

**Interfaces:** Consume Task 1's final model catalog/apply/status endpoints. Reuse session/error conventions and polling cleanup.

- [ ] Add behavioral coverage for selection, unprepared models, downtime notice, duplicate submission, transition progress, skip report and failure state.
- [ ] Add the selector to the existing settings screen and make completed/failed state survive page reload via the backend status.
- [ ] Document preparation, switching, rollback/recovery, analysis gaps and future package metadata.
- [ ] Run frontend tests, typecheck, lint and build.

### Task 3: Integrated review and GPU verification

**Files:** Only fixes required by the completed feature; executable GPU checks must be authored by Luna.

- [ ] Review the full diff against the approved spec and concurrency/recovery failure modes.
- [ ] Run isolated PostgreSQL migration and transition integration checks.
- [ ] Verify all three model image/text dimensions and normalized outputs alongside detector on GPU without modifying existing services.
- [ ] Fix findings through Luna, rerun affected checks, and run git diff --check.
- [ ] Report delivered behavior, tests, branch, and any real execution limitations.
