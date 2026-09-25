# Agent Journeys runbook

Add the `agent-verify` label to a pull request to have a Claude Opus agent perform every Journey in `qa/journeys/` against a real GPU stack built from the PR head. Terms follow [`CONTEXT.md`](../CONTEXT.md). [ADR 0001](adr/0001-agent-journey-verification-in-ci.md) records why the runner uses the development machine and the Claude subscription.

## What a run does

1. Removes the `agent-verify` label. The label is one-shot, so add it again to start another run.
2. Runs the deterministic checks: ruff, basedpyright, and pytest on the Journey harness, plus the web typecheck and tests. Journeys run only if these pass. The Python checks are scoped to the harness because the repository-wide Python suite is not green on `develop`. Widen the step once it is.
3. Builds the PR stack with `./gods-watching prepare` in the runner workspace, under Compose project `gw-ci` with app image tag `ci`.
4. Stops any running development Compose project (`GW_JOURNEY_DEV_PROJECTS`, default `gods-watching`) with `docker compose stop`, and starts exactly those projects again when the run ends.
5. For each Journey: resets `gw-ci` (volumes included), starts the fixture cameras, performs the Journey's setup steps through the API, and lets the agent use the browser. A `bug` Verdict is re-run once and then Cross-checked by an independent Claude session.
6. Publishes the `agent-journeys` check (`neutral` when a Bug Report exists), one PR comment, and a GitHub Issue for each Bug Report. It also uploads all evidence as a workflow artifact.

Issues close themselves when the same Journey passes on the same PR (`completed`), or when the PR closes without merge (`not planned`). Issues from merged PRs stay open.

## One-time runner setup

Do this on the GPU development machine as `jwchoi`.

1. Register a repository runner. In GitHub, go to Settings → Actions → Runners → New self-hosted runner, and add the extra label `gods-watching-gpu`. Install it as a user service (`./svc.sh install jwchoi && ./svc.sh start`).
2. Put Node 24 (with Corepack), `uv`, `docker`, `gh`, and `claude` on the runner service's `PATH` by listing their directories in the runner's `.path` file. `node` must resolve to Node 24, which `web/package.json` `engines` requires.
3. Confirm that `claude` is logged in for `jwchoi` (`claude` then `/login`). Runs use the same subscription quota as interactive use. When the quota is exhausted, the remaining Journeys end `inconclusive`.
4. Create the labels once:

   ```bash
   gh label create agent-verify --color 5319e7 --description "Run agent Journeys on this PR once"
   gh label create agent-reported --color d93f0b --description "Bug Report filed by agent Journeys"
   gh label create needs-triage --color fbca04 --description "Needs human triage"
   ```

5. Trigger one run so the runner checks out the repository, then create `.env` in the runner workspace (`<runner>/_work/gods-watching/gods-watching/.env`) **before** the first `prepare`. `prepare` fills in the generated secrets and keeps these values:

   ```dotenv
   COMPOSE_PROJECT_NAME=gw-ci
   GW_APP_IMAGE_TAG=ci
   GW_PUBLIC_PORT=18080
   GW_PUBLIC_ORIGIN=http://localhost:18080
   GW_PUBLIC_TLS_PORT=18443
   GW_API_PORT=28000
   GW_POSTGRES_PORT=25432
   GW_TRITON_HTTP_PORT=28010
   GW_TRITON_GRPC_PORT=28011
   GW_TRITON_METRICS_PORT=28012
   GW_MEDIA_RTSP_PORT=28555
   GW_MEDIA_WHEP_PORT=28889
   GW_MEDIA_CONTROL_PORT=29997
   GW_MEDIA_WEBRTC_UDP_PORT=18189
   GW_FIXTURE_RTSP_PORT=38554
   ```

   Then run `./gods-watching prepare` and `./gods-watching doctor` in that workspace. The workflow checks out with `clean: false`, so this `.env`, the model cache under `runtime/assets/models`, and the fixture videos under `runtime/assets/fixtures` persist between runs.

## Troubleshooting

- **The development stack is still stopped after a run**: the runner process was killed before cleanup. Start it with `docker compose -p gods-watching start`.
- **Every Journey is `inconclusive` with `usage_limit`**: the Claude subscription window is exhausted. Re-add the label after it resets.
- **Stack never becomes ready**: download the run's `compose.log` from the workflow artifact, then run `./gods-watching doctor` in the runner workspace.
