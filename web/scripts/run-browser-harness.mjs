import { spawn } from "node:child_process"
import { access, mkdir, readdir, readFile, stat, writeFile } from "node:fs/promises"
import { isAbsolute, join, resolve } from "node:path"
import { browserCachePath, browserEnvironment, webRoot } from "./playwright-runtime.mjs"

function fail(message) {
  throw new Error(message)
}

function evidenceRootPath() {
  const configuredPath = process.env.GW_E2E_EVIDENCE_ROOT
  if (configuredPath === undefined || configuredPath === "") {
    fail("GW_E2E_EVIDENCE_ROOT is required and must name a new absolute directory.")
  }
  if (!isAbsolute(configuredPath)) {
    fail("GW_E2E_EVIDENCE_ROOT must be an absolute path.")
  }
  return resolve(configuredPath)
}

async function createEmptyEvidenceRoot(evidenceRoot) {
  try {
    const entries = await readdir(evidenceRoot)
    if (entries.length > 0) {
      fail(
        `EVIDENCE_ROOT_NOT_EMPTY: ${evidenceRoot} already contains artifacts; choose a new directory.`,
      )
    }
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT") {
      await mkdir(evidenceRoot, { recursive: true })
      return
    }
    throw error
  }
}

async function ensurePreparedBrowser(cachePath) {
  process.env.PLAYWRIGHT_BROWSERS_PATH = cachePath
  const { chromium } = await import("@playwright/test")
  const executablePath = chromium.executablePath()
  try {
    await access(executablePath)
  } catch {
    fail(
      `BROWSER_NOT_PREPARED: Chromium is missing from ${cachePath}. Run pnpm -C web run browser:prepare while online.`,
    )
  }
}

function runPlaywright(arguments_, environment) {
  return new Promise((resolve, reject) => {
    const cliPath = join(webRoot, "node_modules", "@playwright", "test", "cli.js")
    const child = spawn(process.execPath, [cliPath, ...arguments_], {
      cwd: webRoot,
      env: environment,
      stdio: "inherit",
    })
    child.once("error", reject)
    child.once("exit", (code, signal) => resolve({ code: code ?? 1, signal }))
  })
}

async function readJson(path) {
  return JSON.parse(await readFile(path, "utf8"))
}

function hasObservationShape(value) {
  if (typeof value !== "object" || value === null) {
    return false
  }
  return (
    typeof value.browser_version === "string" &&
    value.browser_version.length > 0 &&
    value.title === "GW browser harness neutral page" &&
    value.dom_observation === "Browser harness ready" &&
    typeof value.origin === "string" &&
    value.origin.startsWith("http://127.0.0.1:") &&
    typeof value.screenshot === "string"
  )
}

function hasIsolationShape(value) {
  if (typeof value !== "object" || value === null) {
    return false
  }
  return (
    value.unique_origins === true &&
    typeof value.first_origin === "string" &&
    typeof value.second_origin === "string" &&
    value.first_origin !== value.second_origin
  )
}

async function validateArtifacts(evidenceRoot) {
  const observationPath = join(evidenceRoot, "browser-observation.json")
  const isolationPath = join(evidenceRoot, "isolation-observation.json")
  const resultsPath = join(evidenceRoot, "playwright-results.json")
  let observation
  let isolation
  try {
    observation = await readJson(observationPath)
    isolation = await readJson(isolationPath)
    await access(resultsPath)
  } catch {
    fail("MISLEADING_SUCCESS: Playwright returned zero without all required evidence artifacts.")
  }

  if (!hasObservationShape(observation)) {
    fail(
      "MISLEADING_SUCCESS: Playwright returned zero but browser-observation.json is missing required browser observations.",
    )
  }
  if (!hasIsolationShape(isolation)) {
    fail(
      "MISLEADING_SUCCESS: Playwright returned zero but isolation-observation.json does not prove distinct live origins.",
    )
  }
  if (
    !isAbsolute(observation.screenshot) ||
    !resolve(observation.screenshot).startsWith(`${evidenceRoot}/`)
  ) {
    fail("MISLEADING_SUCCESS: screenshot path is outside this evidence root.")
  }

  let screenshot
  try {
    screenshot = await readFile(observation.screenshot)
  } catch {
    fail("MISLEADING_SUCCESS: browser-observation.json references a missing screenshot.")
  }
  const pngSignature = "89504e470d0a1a0a"
  if (screenshot.length < 8 || screenshot.subarray(0, 8).toString("hex") !== pngSignature) {
    fail("MISLEADING_SUCCESS: browser screenshot is absent or not a PNG artifact.")
  }
  if ((await stat(resultsPath)).size === 0) {
    fail("MISLEADING_SUCCESS: Playwright JSON result artifact is empty.")
  }
}

const startedAt = new Date().toISOString()
let evidenceRoot
let evidenceRootReady = false
let browserCache
let playwrightExitCode = null
let failure

try {
  evidenceRoot = evidenceRootPath()
  await createEmptyEvidenceRoot(evidenceRoot)
  evidenceRootReady = true
  browserCache = browserCachePath()
  await ensurePreparedBrowser(browserCache)
  const result = await runPlaywright(
    ["test", "--config", "playwright.config.ts", "--project", "chromium"],
    {
      ...browserEnvironment(browserCache),
      GW_E2E_EVIDENCE_ROOT: evidenceRoot,
    },
  )
  playwrightExitCode = result.code
  if (result.code !== 0) {
    fail(`Playwright harness failed (code=${result.code}, signal=${result.signal ?? "none"}).`)
  }
  await validateArtifacts(evidenceRoot)
} catch (error) {
  failure = error instanceof Error ? error.message : String(error)
} finally {
  if (evidenceRoot !== undefined && evidenceRootReady) {
    await writeFile(
      join(evidenceRoot, "run-manifest.json"),
      `${JSON.stringify(
        {
          started_at: startedAt,
          finished_at: new Date().toISOString(),
          browser_cache: browserCache ?? null,
          playwright_exit_code: playwrightExitCode,
          status: failure === undefined ? "passed" : "failed",
          failure: failure ?? null,
        },
        null,
        2,
      )}\n`,
      "utf8",
    )
  }
}

if (failure !== undefined) {
  console.error(failure)
  process.exitCode = 1
}
