# Agent Journey Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Label a PR `agent-verify` and have a Claude Opus agent run every Journey against a real GPU stack. Confirmed bugs become GitHub Issues that close automatically.

**Architecture:** A new `gods_watching.journeys` package holds five parts: a Journey catalog parsed from `qa/journeys/*.md`, a Claude CLI adapter (an executor and a judge behind protocols), a pure verdict pipeline (attempt → re-run → Cross-check), a CI stack controller (dev-stack stop/start, per-Journey reset, deterministic API setup), and a `gh`-backed reporter. The package is exposed as `gods-watching journeys ...`. Two GitHub workflows run it on the self-hosted GPU runner.

**Tech Stack:** Python 3.12 stdlib, Typer, Pydantic; Claude Code CLI 2.1.x (`claude -p`); `@playwright/mcp` pinned in `web/package.json`; Docker Compose; GitHub Actions; `gh` CLI.

**Spec:** `docs/superpowers/specs/2026-09-25-agent-journey-verification.md`

## Global Constraints

- Vocabulary follows `CONTEXT.md` exactly: Journey, Journey Run, Verdict, Expected Outcome, Observation, Cross-check, Bug Report. Code names mirror these terms (`Journey`, `JourneyRun`, `Verdict`, `ExpectedOutcome`, `Observation`, `BugReport`).
- Add no new Python runtime dependency. Use the stdlib (`subprocess`, `urllib.request`, `hashlib`, `json`) plus the existing Pydantic and Typer.
- The executor agent's only tool is Playwright MCP. The judge's only tool is `Read` within the run evidence directory. Neither agent receives a shell, Docker, or network tool.
- No turn, time, or spend limit on agents (`--max-turns` and `--max-budget-usd` are never passed).
- Model flag `--model opus`. Subscription auth only: never read or require `ANTHROPIC_API_KEY`, and never pass `--bare`.
- Every subprocess call takes an argument tuple without a shell. Secrets from `.env` never appear in logs, comments, or Issues.
- Code passes `ruff check` (`select = ["ALL"]`) and `basedpyright` (`typeCheckingMode = "all"`) as configured in `pyproject.toml`. Use `./.venv/bin/pytest`, `./.venv/bin/ruff`, `./.venv/bin/basedpyright`.
- Commit each task on branch `claude/agent-journey-verification` with the human Git identity and the AI co-author trailers.

## Review Focus

- A `bug` that does not reproduce, or that the judge rejects, never creates an Issue (Task 3 tests).
- Development Compose projects that were running are started again after success, failure, or exception. Projects that were stopped before the run stay stopped (Task 4 tests).
- `down --volumes` only ever targets `gw-ci` and `gw-ci-fixtures`, never a development project (Task 4 tests).
- A usage-limit error marks the current and all remaining Journeys `inconclusive` without attempting them (Task 3 tests).
- A duplicate fingerprint comments on the existing Issue instead of opening a new one. A pass closes only Issues for the same Journey and PR (Task 5 tests).

---

### Task 1: Journey catalog and first five Journeys

**Files:**
- Create: `server/src/gods_watching/journeys/__init__.py`
- Create: `server/src/gods_watching/journeys/models.py` — domain types.
- Create: `server/src/gods_watching/journeys/catalog.py` — Markdown parsing.
- Create: `qa/journeys/login.md`, `qa/journeys/live-cameras.md`, `qa/journeys/text-search.md`, `qa/journeys/similar-search.md`, `qa/journeys/model-switch.md`
- Test: `server/tests/test_journey_catalog.py`

**Interfaces:**
- `ExpectedOutcome(id: str, text: str)`: frozen Pydantic model. IDs match `^(E[1-9][0-9]*|C[12])$`.
- `SetupStep = Literal["login", "cameras"]`
- `Journey(id: str, title: str, setup: tuple[SetupStep, ...], tags: tuple[str, ...], preconditions: str, goal: str, expected_outcomes: tuple[ExpectedOutcome, ...], source_path: Path)`: frozen. `expected_outcomes` includes the common `C1` and `C2` appended last.
- `COMMON_OUTCOMES: tuple[ExpectedOutcome, ...]`: C1 "No product request returns an HTTP 5xx status." and C2 "The browser shows no unhandled exception and no error-level console message."
- `load_journey(path: Path) -> Journey` and `load_catalog(directory: Path) -> tuple[Journey, ...]` (sorted by `id`). They raise `JourneyDefinitionError(path: Path, reason: str)` for a missing or duplicate frontmatter key, unknown setup step, missing section, zero `E` outcomes, a duplicate outcome ID, an `id` that differs from the filename stem, or duplicate Journey IDs across files.
- Frontmatter is the block between the first two `---` lines, one `key: value` per line. `setup` and `tags` are comma-separated and may be empty.

- [ ] Write tests: parse a valid Journey written to `tmp_path`, including C1/C2 appended; each rejection listed above; `load_catalog` sorts and rejects duplicate IDs. Also add one test that loads the real `qa/journeys` directory and asserts exactly the IDs `live-cameras`, `login`, `model-switch`, `similar-search`, `text-search`.
- [ ] Run `./.venv/bin/pytest server/tests/test_journey_catalog.py -q` and confirm the tests fail.
- [ ] Implement `models.py` and `catalog.py`.
- [ ] Write the five Journey files. Base UI wording on the running web app (`web/src`). Keep each Goal to intent, not click paths. Required content:
  - `login` (setup: none): wrong password is rejected with a visible message and no session; correct `GW_OPERATOR_USERNAME`/`GW_OPERATOR_PASSWORD` credentials are supplied to the agent in the prompt by the harness, not written in the file; logout returns to the login screen and protected pages require login again.
  - `live-cameras` (setup: login, cameras): all four fixture cameras are listed; opening live view shows moving video for at least one camera within 30 seconds; detection overlays or person indicators appear for at least one camera within 2 minutes.
  - `text-search` (setup: login, cameras): after person crops appear, the English query "a person walking" returns at least one result with a crop image, camera name, and time; a camera filter narrows the results to that camera.
  - `similar-search` (setup: login, cameras): starting "Find similar" from a result returns a ranked list whose first results include crops from the same camera; the source result is identifiable.
  - `model-switch` (setup: login, cameras): the Cameras page lists the prepared CLIP models with the active one marked; choosing a different prepared model shows a preflight with a crop count and an estimate, or an actionable blocked reason; if apply is allowed, confirming shows durable progress that survives a page reload and ends with the new model active and search usable; if apply is blocked, the active model and search results are unchanged.
- [ ] Run the focused tests, `./.venv/bin/ruff check server/src/gods_watching/journeys server/tests/test_journey_catalog.py`, and `./.venv/bin/basedpyright server/src/gods_watching/journeys`. Commit.

### Task 2: Claude CLI executor and judge

**Files:**
- Create: `server/src/gods_watching/journeys/agents.py` — protocols and the Claude CLI adapter.
- Create: `server/src/gods_watching/journeys/prompts.py` — prompt text and JSON schemas.
- Test: `server/tests/test_journey_agents.py`

**Interfaces:**
- `Verdict = Literal["pass", "bug", "inconclusive"]`, defined in `models.py`.
- `Violation(outcome_id: str, observed: str, evidence_files: tuple[str, ...])`, `Observation(text: str, flaky: bool = False)`, `ExecutorReport(verdict: Verdict, violations: tuple[Violation, ...], observations: tuple[Observation, ...], steps: tuple[str, ...], summary: str)`, defined in `models.py`.
- `JudgeDecision(outcome_id: str, decision: Literal["bug", "not_bug"], reason: str)`, `JudgeReport(decisions: tuple[JudgeDecision, ...])`, defined in `models.py`.
- `AgentFailure(kind: Literal["usage_limit", "cli_error", "invalid_output"], detail: str)`, defined in `models.py`.
- `class JourneyExecutor(Protocol): def execute(self, journey: Journey, *, base_url: str, credentials: OperatorCredentials, evidence_dir: Path) -> ExecutorReport | AgentFailure`
- `class Judge(Protocol): def judge(self, journey: Journey, report: ExecutorReport, *, evidence_dir: Path) -> JudgeReport | AgentFailure`
- `OperatorCredentials(username: str, password: SecretStr)`, defined in `models.py`.
- `CommandRunner = Callable[[Sequence[str], Path], CompletedCommand]` with `CompletedCommand(returncode: int, stdout: str, stderr: str)`. The default `run_command` uses `subprocess.run(..., capture_output=True, text=True, check=False, cwd=...)`.
- `ClaudeCliExecutor(runner: CommandRunner, claude_binary: str, mcp_config_path: Path)` and `ClaudeCliJudge(runner: CommandRunner, claude_binary: str)`.
- `build_executor_command(prompt: str, mcp_config_path: Path) -> tuple[str, ...]` returns exactly: `claude -p <prompt> --model opus --restricted --tools "" --strict-mcp-config --mcp-config <path> --allowedTools mcp__playwright --permission-mode dontAsk --no-session-persistence --output-format json --json-schema <EXECUTOR_SCHEMA json>`.
- `build_judge_command(prompt: str, evidence_dir: Path) -> tuple[str, ...]` is the same as the executor command but uses `--tools Read --allowedTools Read --add-dir <evidence_dir>` and an empty `--mcp-config` file, with `JUDGE_SCHEMA`.
- `write_playwright_mcp_config(path: Path, *, node_binary: str, cli_path: Path, output_dir: Path) -> None` writes `{"mcpServers": {"playwright": {"command": node, "args": [cli, "--headless", "--isolated", "--output-dir", out, "--save-trace"]}}}`.
- `parse_cli_result(stdout: str, model: type[T]) -> T | AgentFailure` reads the CLI JSON envelope. `is_error: true` becomes `usage_limit` when the `result` text matches `/usage limit|rate limit|limit reached/i`, and `cli_error` otherwise. A missing or schema-invalid `structured_output` becomes `invalid_output`.

- [ ] Write tests with a fake `CommandRunner`: exact executor and judge argument tuples; prompt contains the goal, every Expected Outcome ID, the base URL, and the credentials, and the password is absent from the command log produced by `redacted_command()`; envelope parsing for success, `is_error` usage limit, other CLI error, non-zero exit with empty stdout, and invalid structured output; a `bug` report with no violations is normalized to `inconclusive`; MCP config JSON shape.
- [ ] Run the focused tests and confirm they fail.
- [ ] Implement `prompts.py`. The executor prompt states the Journey sections, lists every Expected Outcome with its ID, instructs the agent to judge only against those IDs, to record anything else as an Observation, to save a screenshot for each violation and name the file, to check console and network messages before finishing (for C1/C2), and to return `inconclusive` when the stack is unusable for reasons outside the Goal. The judge prompt gives the Journey, the claimed violations and their observed text, and the evidence directory listing; it asks for a decision per outcome ID from evidence only.
- [ ] Implement `agents.py`, including `redacted_command(command) -> str`, which replaces the prompt argument with `<prompt>` so logs never contain credentials.
- [ ] Run the focused tests, ruff, and basedpyright on the package. Commit.

### Task 3: Verdict pipeline

**Files:**
- Create: `server/src/gods_watching/journeys/pipeline.py`
- Test: `server/tests/test_journey_pipeline.py`

**Interfaces:**
- `StackError(message: str)`: exception defined in `models.py`.
- `class JourneyStack(Protocol)` with `def prepare_journey(self, journey: Journey) -> None` (reset and deterministic setup; raises `StackError`) and `def collect_evidence(self, evidence_dir: Path) -> None`.
- `BugReport(journey_id: str, outcomes: tuple[ExpectedOutcome, ...], observed: tuple[Violation, ...], judge_reasons: tuple[str, ...], fingerprint: str)`, defined in `models.py`.
- `JourneyRun(journey_id: str, verdict: Verdict, bug_report: BugReport | None, observations: tuple[Observation, ...], attempts: int, failure: str | None)`, defined in `models.py`.
- `fingerprint(journey_id: str, outcome_ids: Iterable[str]) -> str` returns `sha256(f"{journey_id}|{','.join(sorted(set(ids)))}")` as hex.
- `run_journeys(journeys: Sequence[Journey], *, stack: JourneyStack, executor: JourneyExecutor, judge: Judge, base_url: str, credentials: OperatorCredentials, run_dir: Path) -> tuple[JourneyRun, ...]`. Evidence goes under `run_dir/<journey_id>/attempt-<n>/`.

Behavior, in order:
1. If a usage limit was already hit in this run, return `inconclusive` with failure `usage_limit` and 0 attempts.
2. `prepare_journey` → execute → `collect_evidence`. A `StackError` gives `inconclusive` (failure `stack: <message>`). An `AgentFailure` gives `inconclusive` with its kind; `usage_limit` also sets the flag from step 1.
3. `pass` or `inconclusive` from the executor → return it with the executor's Observations.
4. `bug` → a second attempt with steps 2–3. The re-run confirms only if it is `bug` and shares at least one violated ID with attempt 1. Otherwise return `pass` with an extra `Observation(text="flaky: <ids> not reproduced", flaky=True)`.
5. Judge with the reproduced violations (the intersection of IDs). If the judge fails, return `inconclusive` with the failure kind. IDs the judge rejects become Observations. If no confirmed ID remains, return `pass`; otherwise return `bug` with a `BugReport` over the confirmed IDs.

- [ ] Write table-driven tests with fake stack/executor/judge covering: pass; inconclusive; bug reproduced and confirmed; bug not reproduced (flaky); reproduced with a disjoint ID set (flaky); judge rejects all; judge confirms a subset; judge failure; stack error on the first and on the second attempt; a usage limit on Journey 2 of 4 makes Journeys 2–4 `inconclusive` and Journeys 3–4 are never prepared; fingerprint stable across ID order and duplicates; evidence directory paths.
- [ ] Run the tests, confirm they fail, implement, and run again with ruff and basedpyright. Commit.

### Task 4: CI stack controller

**Files:**
- Modify: `compose.yaml:27` — `image: gods-watching-app:${GW_APP_IMAGE_TAG:-local}`
- Create: `server/src/gods_watching/journeys/stack.py` — `ComposeJourneyStack` and `DevStackGuard`.
- Create: `server/src/gods_watching/journeys/fixtures.py` — fixture video download and verification.
- Create: `server/src/gods_watching/journeys/http_setup.py` — login and camera registration over HTTP.
- Test: `server/tests/test_journey_stack.py`, `server/tests/test_journey_fixtures.py`, `server/tests/test_journey_http_setup.py`

**Interfaces:**
- `DevStackGuard(runner: CommandRunner, project_names: tuple[str, ...], repository_root: Path)` is a context manager. On enter, it lists running projects via `docker compose ls --format json`, stops each running name from `project_names` with `docker compose -p <name> stop`, and records them. On exit, even on exception, it runs `docker compose -p <name> start` for exactly the recorded names and re-raises the original exception. It refuses (raises `StackError`) if a name equals `gw-ci` or `gw-ci-fixtures`.
- `ComposeJourneyStack(runner: CommandRunner, repository_root: Path, env: CiEnvironment, http: HttpSetup)` implements `JourneyStack`. `prepare_journey`: `docker compose -p gw-ci down --volumes --remove-orphans`; fixtures project `docker compose -p gw-ci-fixtures -f deploy/compose.fixtures.yaml --profile fixtures up -d`; `docker compose -p gw-ci up -d`; poll `GET <base_url>/api/session` until it returns HTTP 200 or 401 (every 2 s, up to 600 s, then `StackError`); then run each setup step. `collect_evidence` writes `compose.log` from `docker compose -p gw-ci logs --no-color --timestamps`. `shutdown()` tears down both CI projects with `down --volumes`.
- `CiEnvironment(base_url: str, credentials: OperatorCredentials, fixture_rtsp_port: int)` comes from `load_ci_environment(env_path: Path)`, which parses `.env` (`GW_PUBLIC_ORIGIN`, `GW_OPERATOR_USERNAME`, `GW_OPERATOR_PASSWORD`, `GW_FIXTURE_RTSP_PORT`) and requires `COMPOSE_PROJECT_NAME=gw-ci`.
- `HttpSetup(opener: urllib.request.OpenerDirector, base_url: str)`: `login(credentials) -> None` posts JSON to `/api/session` with an `Origin: <base_url>` header and keeps the cookie jar; `register_fixture_cameras(rtsp_port: int) -> tuple[str, ...]` posts `CameraCreateRequest` bodies for `camera-1`..`camera-4` at `rtsp://127.0.0.1:<port>/camera-N` to `/api/cameras`; `wait_cameras_streaming(camera_ids, timeout_s=180)` polls `GET /api/cameras` until each camera's runtime state shows streaming. Read `server/src/gods_watching/contracts/cameras.py` for the exact response field names.
- `ensure_fixture_videos(manifest_path: Path, repository_root: Path, *, download: Callable[[str, Path], None]) -> None`. For each stream in `assets/test-streams.json`: if `prepared_path` is a directory (a stale bind-mount artifact), remove it only when it is empty; if the file is missing or its SHA-256 differs, download to a temp file in the same directory, verify SHA-256, and rename. A mismatch after download raises `FixtureDownloadError`.

- [ ] Write tests with a fake runner that records commands: the guard stops only running listed projects, restarts them on normal exit and on exception, restarts nothing that was not running, and refuses CI project names. `prepare_journey` emits exactly the listed commands in order and never includes a non-CI `-p`. The health timeout raises `StackError`. Evidence is written. Fixtures: an existing valid file is untouched, a missing file is downloaded and verified, a bad hash raises and leaves no partial file, an empty stale directory is replaced. HTTP setup: run against a stdlib `http.server` test double, asserting the Origin header, cookie reuse, camera request bodies, and the wait timeout.
- [ ] Run the tests, confirm they fail, implement, and run again with ruff and basedpyright. Run `docker compose config --quiet` to confirm that `compose.yaml` still validates. Commit.

### Task 5: GitHub reporting and CLI

**Files:**
- Create: `server/src/gods_watching/journeys/github.py` — the `gh` wrapper.
- Create: `server/src/gods_watching/journeys/report.py` — Markdown rendering.
- Create: `server/src/gods_watching/journeys/cli.py` — Typer app.
- Modify: `server/src/gods_watching/cli.py` — `app.add_typer(journeys_app, name="journeys")`.
- Test: `server/tests/test_journey_report.py`, `server/tests/test_journey_github.py`, `server/tests/test_journey_cli.py`

**Interfaces:**
- `RunContext(repository: str, pr_number: int, head_sha: str, run_url: str)`
- `render_pr_comment(runs: Sequence[JourneyRun], context: RunContext) -> str` starts with `<!-- agent-journeys:pr-comment -->`, then a Verdict table, a Bug Reports section, an Observations section, and a run link.
- `render_issue(report: BugReport, journey: Journey, context: RunContext) -> tuple[str, str]` (title, body). The body contains `<!-- agent-journeys:fingerprint=<fp> journey=<id> pr=<n> -->`.
- `check_conclusion(runs: Sequence[JourneyRun], harness_failed: bool) -> Literal["success", "neutral", "failure"]`
- `GitHubClient(runner: CommandRunner, repository: str)` with `upsert_pr_comment(pr, body)`, `find_open_agent_issues() -> tuple[AgentIssue, ...]` (parses markers from `gh issue list --label agent-reported --state open --json number,body --limit 500`), `create_issue(title, body)`, `comment_issue(number, body)`, `close_issue(number, reason: Literal["completed", "not planned"], comment)`, and `create_check_run(head_sha, conclusion, summary)` via `gh api repos/<repo>/check-runs`. It never passes a body on the command line; use `--body-file`/`--input` with a temp file.
- `publish_results(runs, journeys, context, client) -> None`: upsert the comment; for each Bug Report, comment on the Issue with the same fingerprint or create a new one; for each `pass` Journey, close open agent Issues whose marker has the same `journey` and `pr` as `completed`.
- CLI:
  - `gods-watching journeys run --pr <n> --head-sha <sha> --run-url <url> --run-dir <dir> [--repository owner/name] [--journeys-dir qa/journeys] [--env-file .env] [--publish/--no-publish]`. It wraps everything in `DevStackGuard`, ensures fixture videos, runs `run_journeys`, always calls `stack.shutdown()`, writes `run_dir/results.json`, publishes when `--publish` is set, and exits 0 unless the harness failed.
  - `gods-watching journeys close-pr-issues --pr <n> --repository owner/name` closes that PR's open agent Issues as `not planned`.

- [ ] Write tests: comment and Issue rendering snapshots (inline expected strings); conclusion table; publish with a fake gh runner, covering a new Issue, a duplicate fingerprint that comments, a pass that closes only the matching journey+pr, and an unrelated-PR Issue left open; `close-pr-issues`; the CLI `run` path with fakes injected through a factory seam, asserting that shutdown and guard exit happen when `run_journeys` raises and that `results.json` is written.
- [ ] Run the tests, confirm they fail, implement, and run again with ruff and basedpyright over `server/src/gods_watching/journeys` and `server/src/gods_watching/cli.py`. Commit.

### Task 6: Workflows, runner bootstrap, and a real run

**Files:**
- Modify: `web/package.json` and `pnpm-lock.yaml` — add devDependency `@playwright/mcp` `0.0.82`.
- Create: `.github/workflows/agent-journeys.yml`
- Create: `.github/workflows/agent-journeys-pr-closed.yml`
- Create: `docs/agent-journeys.md` — runner bootstrap and operation runbook.
- Modify: `README.md` — a short section linking the runbook.

**Workflow `agent-journeys.yml`:**
- `on: pull_request: types: [labeled]`; job `if: github.event.label.name == 'agent-verify'`
- `concurrency: { group: agent-journeys-gpu, cancel-in-progress: false }`
- Permissions: `contents: read`, `pull-requests: write`, `issues: write`, `checks: write`
- `runs-on: [self-hosted, gods-watching-gpu]`
- Steps:
  1. `gh pr edit <n> --remove-label agent-verify` with `GH_TOKEN: ${{ github.token }}`.
  2. `actions/checkout` with `ref: ${{ github.event.pull_request.head.sha }}` and `clean: false`.
  3. `uv sync --frozen`, `pnpm install --frozen-lockfile` in `web`, then `node web/scripts/prepare-playwright.mjs`.
  4. Deterministic job steps: `./.venv/bin/ruff check`, `./.venv/bin/basedpyright`, `./.venv/bin/pytest -q`, `pnpm --dir web typecheck`, `pnpm --dir web test --run`.
  5. `./gods-watching prepare`
  6. `./gods-watching journeys run --pr ... --head-sha ... --run-url ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }} --run-dir ${{ runner.temp }}/agent-journeys --publish`
  7. `actions/upload-artifact` of the run dir with `if: always()`.
- If a deterministic step fails, a final `if: failure()` step creates an `agent-journeys` check run with conclusion `failure` and the summary "deterministic checks failed; Journeys not run".

**Workflow `agent-journeys-pr-closed.yml`:** `on: pull_request: types: [closed]`, `if: github.event.pull_request.merged == false`, on the same runner. It runs `./gods-watching journeys close-pr-issues --pr ... --repository ${{ github.repository }}`.

**Runbook `docs/agent-journeys.md`:**
- Register this machine as a repository runner under the `jwchoi` account with the extra label `gods-watching-gpu`, and run it as a user service.
- Create the CI `.env` in the runner workspace before the first `prepare`: `COMPOSE_PROJECT_NAME=gw-ci`, `GW_APP_IMAGE_TAG=ci`, and non-overlapping ports (`GW_PUBLIC_PORT=18080`, `GW_PUBLIC_ORIGIN=http://localhost:18080`, `GW_PUBLIC_TLS_PORT=18443`, `GW_API_PORT=28000`, `GW_POSTGRES_PORT=25432`, `GW_TRITON_HTTP_PORT=28010`, `GW_TRITON_GRPC_PORT=28011`, `GW_TRITON_METRICS_PORT=28012`, `GW_MEDIA_RTSP_PORT=28555`, `GW_MEDIA_WHEP_PORT=28889`, `GW_MEDIA_CONTROL_PORT=29997`, `GW_MEDIA_WEBRTC_UDP_PORT=18189`, `GW_FIXTURE_RTSP_PORT=38554`). Then run `./gods-watching doctor`.
- Create the repository labels `agent-verify`, `agent-reported`, and `needs-triage`.
- Explain that `claude` must be logged in for `jwchoi` (`claude /login`), that runs consume the same subscription quota as interactive use, and that the development stack is stopped during a run and restored afterwards.

- [ ] Add the dependency with `pnpm --dir web add -D @playwright/mcp@0.0.82`. Confirm that `node web/node_modules/@playwright/mcp/cli.js --help` works and that `ClaudeCliExecutor` points `cli_path` at that file.
- [ ] Write both workflows and the runbook. Validate both workflows with `docker run --rm -v "$PWD:/repo" -w /repo rhysd/actionlint:1.7.7 -color`.
- [ ] Real local run on this machine with `--no-publish`: create a scratch CI env file per the runbook, run `./gods-watching journeys run --pr 0 --head-sha $(git rev-parse HEAD) --run-url local --run-dir <scratch>/run --journeys-dir qa/journeys --env-file <ci .env> --no-publish` for at least the `login` Journey (via a temporary `--journeys-dir` containing only `login.md`). Record the Verdict, attempts, and evidence paths. Confirm that the development stack state before and after matches `docker compose ls`.
- [ ] Run the full deterministic suite named in the workflow and `git diff --check`. Commit.

## Self-review

- Spec coverage: trigger and one-shot label (T6), runner and subscription auth (T2, T6), deterministic gate (T6), CI stack isolation and image tag (T4, T6), dev stop/start (T4), per-Journey reset and API setup (T4), catalog and five Journeys (T1), executor tools and schema (T2), re-run, Cross-check, and usage limit (T3), check run, comment, Issues, fingerprint, and auto-close (T5), closed-PR closing (T5, T6).
- Types: `Journey`, `ExpectedOutcome`, `ExecutorReport`, `Violation`, `Observation`, `JudgeReport`, `AgentFailure`, `JourneyRun`, `BugReport`, `OperatorCredentials`, `CommandRunner`, `CompletedCommand`, `JourneyStack`, `StackError` are defined once (models.py, agents.py, or pipeline.py) and are consumed under the same names.
