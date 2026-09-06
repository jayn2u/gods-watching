import { isAbsolute, join, resolve } from "node:path"
import { defineConfig } from "@playwright/test"

function configuredBaseURL(): string | undefined {
  const rawValue = process.env["GW_BASE_URL"]
  if (rawValue === undefined || rawValue === "") {
    return undefined
  }

  let parsed: URL
  try {
    parsed = new URL(rawValue)
  } catch {
    throw new Error("GW_BASE_URL must be an absolute http(s) origin.")
  }
  if (
    (parsed.protocol !== "http:" && parsed.protocol !== "https:") ||
    parsed.username !== "" ||
    parsed.password !== "" ||
    parsed.pathname !== "/" ||
    parsed.search !== "" ||
    parsed.hash !== ""
  ) {
    throw new Error(
      "GW_BASE_URL must be a credential-free http(s) origin without a path, query, or fragment.",
    )
  }
  return parsed.origin
}

function configuredEvidenceRoot(): string {
  const rawValue = process.env["GW_E2E_EVIDENCE_ROOT"]
  if (rawValue === undefined || rawValue === "") {
    return resolve("test-results")
  }
  if (!isAbsolute(rawValue)) {
    throw new Error("GW_E2E_EVIDENCE_ROOT must be an absolute path when supplied.")
  }
  return resolve(rawValue)
}

const baseURL = configuredBaseURL()
const evidenceRoot = configuredEvidenceRoot()
const showcaseOrigin = "http://127.0.0.1:4175"

export default defineConfig({
  testDir: "./e2e",
  testMatch: "*.spec.ts",
  fullyParallel: true,
  workers: 2,
  projects: [{ name: "chromium" }],
  forbidOnly: Boolean(process.env["CI"]),
  outputDir: join(evidenceRoot, "playwright-output"),
  reporter: [["line"], ["json", { outputFile: join(evidenceRoot, "playwright-results.json") }]],
  webServer: {
    command: "VITE_DISABLE_REACT_DEVTOOLS=1 pnpm exec vite --host 127.0.0.1 --port 4175",
    url: showcaseOrigin,
    reuseExistingServer: false,
  },
  use: {
    browserName: "chromium",
    headless: true,
    screenshot: "off",
    trace: "off",
    baseURL: baseURL ?? showcaseOrigin,
  },
})
