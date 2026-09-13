import { spawn } from "node:child_process"
import { access, mkdir, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { browserCachePath, browserEnvironment, webRoot } from "./playwright-runtime.mjs"

function run(command, arguments_, environment) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, arguments_, {
      cwd: webRoot,
      env: environment,
      stdio: "inherit",
    })
    child.once("error", reject)
    child.once("exit", (code, signal) => {
      if (code === 0) {
        resolve()
        return
      }
      reject(
        new Error(
          `Playwright Chromium preparation failed (code=${code ?? "null"}, signal=${signal ?? "none"}).`,
        ),
      )
    })
  })
}

const cachePath = browserCachePath()
const environment = browserEnvironment(cachePath)
const cliPath = join(webRoot, "node_modules", "@playwright", "test", "cli.js")

try {
  await access(cliPath)
} catch {
  throw new Error(
    "@playwright/test is not installed. Run pnpm install --frozen-lockfile before browser preparation.",
  )
}

await mkdir(cachePath, { recursive: true })
console.log("Preparing the pinned Playwright Chromium cache without --with-deps.")
await run(process.execPath, [cliPath, "install", "chromium"], environment)

process.env.PLAYWRIGHT_BROWSERS_PATH = cachePath
const { chromium } = await import("@playwright/test")
const executablePath = chromium.executablePath()
await access(executablePath)

const browser = await chromium.launch({ headless: true })
let browserVersion
try {
  browserVersion = browser.version()
} finally {
  await browser.close()
}

await writeFile(
  join(cachePath, "preparation.json"),
  `${JSON.stringify(
    {
      prepared_at: new Date().toISOString(),
      browser: "chromium",
      browser_version: browserVersion,
      executable_path: executablePath,
      host_dependency_install: false,
    },
    null,
    2,
  )}\n`,
  "utf8",
)

console.log(`Prepared Chromium ${browserVersion} in ${cachePath}.`)
