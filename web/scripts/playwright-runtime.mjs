import { dirname, isAbsolute, resolve } from "node:path"
import { fileURLToPath } from "node:url"

export const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..")
export const workspaceRoot = resolve(webRoot, "..")

export function browserCachePath() {
  const configuredPath = process.env.GW_PLAYWRIGHT_BROWSERS_PATH
  if (configuredPath === undefined || configuredPath === "") {
    return resolve(workspaceRoot, "runtime", "playwright-browsers")
  }

  if (!isAbsolute(configuredPath)) {
    throw new Error("GW_PLAYWRIGHT_BROWSERS_PATH must be an absolute path.")
  }

  return resolve(configuredPath)
}

export function browserEnvironment(cachePath) {
  const environment = { ...process.env, PLAYWRIGHT_BROWSERS_PATH: cachePath }
  delete environment.PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD
  return environment
}
