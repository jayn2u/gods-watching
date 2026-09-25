# Task 5 report — 2026-09-25

## Delivered

- Catalog exposes exact immutable revision; browser parser and selector display it.
- Quality-blocked prepared packages show the API reason and cannot be selected or applied.
- Selector loads and polls typed preflight, shows retained count, expected missing crop count, estimate and limit, and disables apply when ineligible. It refreshes preflight before opening confirmation and again before mutation; a changed estimate closes confirmation for another review. The server repeats preflight inside apply.
- Existing durable transition display retains progress and skip reasons after refresh. Documentation covers local import, package evidence, rollback, and incomplete gates.

## Verification

- web/node_modules/.bin/tsc --noEmit -p web/tsconfig.json: pass.
- web/node_modules/.bin/vitest run web/src/app/clientDomain.test.ts: 4 passed.
- ./.venv/bin/pytest server/tests/test_model_selection_routes.py -q: 8 passed.
- ./.venv/bin/pytest server/tests/integration/test_model_selection.py -q: 20 passed.
- Scoped Biome and Ruff: pass. git diff --check: pass.
- web/node_modules/.bin/playwright test web/e2e/model-selector.spec.ts: blocked before test execution because Playwright Chromium headless shell is not installed. UI interactions remain unverified in a real browser.

## Remaining release proof

- No isolated model clone, supplied checkpoint, real CUHK report, product retrieval cases, or target GPU 15-minute run was available. Neither retrieval quality gate nor four-camera 15-minute load gate was passed here.
- Task 4 currently consumes full-transition rehearsal proof, but its producer is still incomplete. benchmark-embedding writes a diagnostic and cannot authorize a model switch. Without a valid rehearsal, preflight is ineligible.
- API/worker restart during reindex and after activation, exact imported package identity after recovery, and rollback to the previous package require integration environment proof. Existing focused integration tests passed but do not prove those restart paths.

## Review fix round 1 — 2026-09-25

- Browser E2E now covers missing full-transition measurement, over-limit estimate, a changed estimate between confirmation and mutation, positive quality-approved status, and the expected skipped crop label. The label includes missing and corrupt/unreadable crops. Confirmation compares against its own frozen preflight snapshot even if polling refreshes the preview.
- Integration tests reconstruct a new service during durable reindex and after target activation, then verify resumed progress, target revision, and runtime package descriptor. A synthetic imported package descriptor with a different revision is rejected during recovery. The pre-activation failure test verifies the previous ID, revision, and dimension after rollback and startup reconciliation.
- Initial Chromium E2E attempt with installed browser failed because `pnpm` was absent from PATH. Running via a temporary Corepack shim exposed one test locator strict-mode error (6 passed, 1 failed); the locator was corrected. `PATH=/tmp/task5-bin:$PATH node_modules/.bin/playwright test e2e/model-selector.spec.ts` from `web/`: 7 passed. This verifies the mocked browser UI, not deployment GPU or real inference.
- `./.venv/bin/pytest server/tests/integration/test_model_selection.py -q`: 20 passed after recovery assertions. The synthetic descriptor test does not replace a real imported checkpoint, product retrieval evidence, a separate API/worker process restart, or a target-GPU 15-minute load run.
- Final focused verification: Chromium E2E 7 passed; Vitest 4 passed; TypeScript, Biome, Ruff, and diff check passed. Combined integration plus route run first had 27 passed and one advisory-lock timing assertion failure while Chromium E2E ran concurrently. That existing advisory-lock case passed alone, then the full focused Python run passed sequentially (28 passed). No change was made to that lock test.
