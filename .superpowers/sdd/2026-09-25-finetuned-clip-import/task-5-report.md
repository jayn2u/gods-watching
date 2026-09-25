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
