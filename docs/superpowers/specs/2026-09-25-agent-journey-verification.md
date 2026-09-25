# Agent Journey verification in CI

## Goal and scope

When an operator adds the `agent-verify` label to a pull request, a Claude Opus agent performs every Journey against a real GPU stack built from the PR head, and every confirmed Bug Report becomes a GitHub Issue. Vocabulary follows `CONTEXT.md`; the decision record is `docs/adr/0001-agent-journey-verification-in-ci.md`.

Codex, nightly or push triggers, a mock UI tier, per-PR Journey selection, and blocking merges are outside this feature.

## Trigger and runner

- Only the `pull_request` `labeled` event with label `agent-verify` starts a run. The workflow removes the label as its first step, so every run is one-shot and re-labelling starts a new run.
- Jobs run on the self-hosted runner labelled `gods-watching-gpu`: the development machine, as the `jwchoi` user, with the existing Claude subscription login. No Anthropic or OpenAI API key is stored in GitHub.
- Deterministic checks run first (`ruff`, `basedpyright`, `pytest`, web `typecheck` and `test`). Journeys run only if they pass.
- A second workflow on `pull_request` `closed` without merge closes that PR's open agent Issues as `not planned`.

## CI stack

- The runner workspace keeps untracked files between runs (`actions/checkout` with `clean: false`). Its `.env` sets `COMPOSE_PROJECT_NAME=gw-ci`, `GW_APP_IMAGE_TAG=ci`, and host ports that do not overlap the development defaults. Its `runtime/assets/models` cache and `runtime/assets/fixtures` videos persist and are separate from the development checkout.
- `compose.yaml` names the app image `gods-watching-app:${GW_APP_IMAGE_TAG:-local}`, so a CI build never replaces the development image.
- Every run executes `./gods-watching prepare` to build images from the PR head. Fixture videos listed in `assets/test-streams.json` are downloaded only when missing or when their SHA-256 does not match.
- Before the stack starts, every running Compose project named in `GW_JOURNEY_DEV_PROJECTS` (default `gods-watching`) is stopped with `docker compose stop`. After the run, including failure or cancellation, exactly the projects that were stopped are started again with `docker compose start`. Projects that were not running stay down.
- Each Journey starts from a fresh stack: `down --volumes` for `gw-ci`, `up -d`, then wait until the gateway serves `/api/session`. The fixture cameras run in a separate `gw-ci-fixtures` project. Model assets and fixture videos are bind mounts and survive the reset.
- Journey preconditions are established deterministically through the HTTP API, never by the agent. The supported setup steps are `login` and `cameras`. `cameras` registers the four fixture streams, then waits until every camera reports streaming.

## Journeys

- One Markdown file per Journey lives in `qa/journeys/<id>.md`. Frontmatter holds `id`, `title`, `setup` (a comma-separated list of setup steps), and `tags`. The body holds `## Preconditions`, `## Goal`, and `## Expected Outcomes`. Every Expected Outcome is a line `- [E<n>] <observable condition>`.
- Two common Expected Outcomes are appended to every Journey: `[C1]` no HTTP 5xx response from the product, and `[C2]` no unhandled browser exception or error-level console message.
- The first catalog has five Journeys: `login`, `live-cameras`, `text-search`, `similar-search`, and `model-switch`. `model-switch` accepts either a completed switch or a clearly explained blocked apply with no state change, because the rehearsal evidence gate can legitimately block apply in CI.

## Agent execution

- The executor runs `claude -p` as a subprocess with `--model opus --restricted --tools "" --strict-mcp-config --mcp-config <file> --allowedTools mcp__playwright --permission-mode dontAsk --no-session-persistence --output-format json --json-schema <schema>`. The agent's only tool is a headless, isolated Playwright MCP browser pinned in `web/package.json`. It has no shell, Docker, file, or network tool.
- The executor returns a Verdict (`pass`, `bug`, or `inconclusive`), the IDs of violated Expected Outcomes, what it observed for each violation, Observations, and a step log. A `bug` Verdict without at least one violated ID is treated as `inconclusive`.
- After each attempt the harness saves `docker compose logs` and the browser output directory as evidence.
- A `bug` Verdict triggers one fresh re-run: reset, setup, and execute again. If the re-run is not `bug` with at least one of the same violated IDs, the result is an Observation marked `flaky` and the Verdict is `pass`.
- A reproduced `bug` goes to Cross-check. The judge is a separate `claude -p --model opus` session without the executor transcript. Its only tool is `Read`, limited to the run's evidence directory. It returns `bug` or `not_bug` per violated ID. Only IDs that both the executor and the judge confirm become a Bug Report. The judge sits behind a `Judge` interface so that Codex can replace it later.
- There is no turn, time, or spend limit. If the CLI reports an error, including a subscription usage limit, the Journey is `inconclusive`. After a usage limit, every remaining Journey is `inconclusive` without being attempted.

## Reporting

- A GitHub check run named `agent-journeys` concludes `neutral` if any Bug Report exists, `failure` if the harness itself failed, and `success` otherwise. It never blocks merging.
- One PR comment, found by a hidden marker, is created or updated per run. It holds the Verdict table, Bug Reports, Observations, and a link to the workflow run. All evidence is uploaded as a workflow artifact.
- Each Bug Report has a fingerprint: the SHA-256 of the Journey ID and the sorted confirmed Expected Outcome IDs. If an open Issue labelled `agent-reported` already carries the same fingerprint marker, the harness adds a comment to it. Otherwise it creates a new Issue labelled `agent-reported` and `needs-triage`. The Issue body contains reproduction context: PR, head SHA, Journey, expected versus observed, and the run link.
- When a Journey is `pass` on a PR, open agent Issues for that Journey found on the same PR are closed as `completed` with a comment. When a PR closes without merge, its open agent Issues are closed as `not planned`. Issues from merged PRs stay open.
