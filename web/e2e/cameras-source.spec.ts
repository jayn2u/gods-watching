import { mkdir, writeFile } from "node:fs/promises"
import { join, resolve } from "node:path"
import { expect, type Page, test } from "@playwright/test"

const SOURCE_QA_FIXTURE_ID = "task-17-labelled-client-fixture-v1"
const cameraId = "00000000-0000-0000-0000-000000000017"

type FixtureCamera = {
  camera_id: string
  name: string
  source_host: string
  source_port: number | null
  detection_enabled: boolean
  detection_threshold: number
  version: number
  deleted_at: string | null
}

type FixtureSettings = {
  retention_days: number
  quota_bytes: number
  wall_slot_ids: Array<string | null>
}

type FixtureApi = {
  readonly createBodies: string[]
  readonly deleteBodies: string[]
  readonly deleteVersions: number[]
  readonly settingsBodies: string[]
  readonly testBodies: string[]
  readonly updateBodies: string[]
  readonly updateVersions: number[]
  holdDelete: boolean
  releaseDelete: () => void
  settingsConflict: boolean
  staleNextUpdate: boolean
  unauthorizedNextMutation: boolean
}

const initialCamera = (): FixtureCamera => ({
  camera_id: cameraId,
  name: "North entrance",
  source_host: "fixture-publisher",
  source_port: 8554,
  detection_enabled: true,
  detection_threshold: 0.5,
  version: 7,
  deleted_at: null,
})

const initialSettings = (): FixtureSettings => ({
  retention_days: 7,
  quota_bytes: 100_000_000_000,
  wall_slot_ids: [null, null, null, null],
})

function fixtureJson(body: unknown, status = 200) {
  return {
    status,
    contentType: "application/json",
    headers: { "X-Source-QA-Fixture": SOURCE_QA_FIXTURE_ID },
    body: JSON.stringify(body),
  }
}

function recordPayload(payload: string | null): Record<string, unknown> {
  if (payload === null) {
    throw new Error("source QA fixture expected a JSON request body")
  }
  const parsed: unknown = JSON.parse(payload)
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("source QA fixture expected a JSON object")
  }
  return Object.fromEntries(Object.entries(parsed))
}

async function installFixture(page: Page): Promise<FixtureApi> {
  let cameras: FixtureCamera[] = [initialCamera()]
  let settings = initialSettings()
  let releaseDeletePromise: (() => void) | undefined
  const session = {
    authenticated: true,
    idle_expires_at: "2026-09-08T12:30:00Z",
    absolute_expires_at: "2026-09-08T20:00:00Z",
  }
  const api: FixtureApi = {
    createBodies: [],
    deleteBodies: [],
    deleteVersions: [],
    settingsBodies: [],
    testBodies: [],
    updateBodies: [],
    updateVersions: [],
    holdDelete: false,
    releaseDelete: () => releaseDeletePromise?.(),
    settingsConflict: false,
    staleNextUpdate: false,
    unauthorizedNextMutation: false,
  }

  await page.route("**/api/session", async (route) => {
    if (route.request().method() === "DELETE") {
      await route.fulfill({
        status: 204,
        headers: { "X-Source-QA-Fixture": SOURCE_QA_FIXTURE_ID },
      })
      return
    }
    await route.fulfill(fixtureJson(session))
  })
  await page.route("**/api/session/activity", async (route) => {
    await route.fulfill(fixtureJson(session))
  })
  await page.route("**/api/cameras**", async (route) => {
    const request = route.request()
    const pathname = new URL(request.url()).pathname
    const method = request.method()

    if (pathname === "/api/cameras" && method === "GET") {
      await route.fulfill(fixtureJson(cameras))
      return
    }
    if (pathname === "/api/cameras/test" && method === "POST") {
      api.testBodies.push(request.postData() ?? "")
      await route.fulfill(
        fixtureJson({
          source_host: "fixture-publisher",
          source_port: 8554,
          codec: "h264",
          width: 1920,
          height: 1080,
        }),
      )
      return
    }
    if (pathname === "/api/cameras" && method === "POST") {
      api.createBodies.push(request.postData() ?? "")
      if (api.unauthorizedNextMutation) {
        api.unauthorizedNextMutation = false
        await route.fulfill(
          fixtureJson({ detail: { code: "unauthorized", message: "Session expired." } }, 401),
        )
        return
      }
      const body = recordPayload(request.postData())
      const name = typeof body["name"] === "string" ? body["name"] : "Unnamed camera"
      const newCamera: FixtureCamera = {
        camera_id: "00000000-0000-0000-0000-000000000018",
        name,
        source_host: "fixture-publisher",
        source_port: 8554,
        detection_enabled: body["detection_enabled"] !== false,
        detection_threshold:
          typeof body["detection_threshold"] === "number" ? body["detection_threshold"] : 0.5,
        version: 1,
        deleted_at: null,
      }
      cameras = [...cameras, newCamera]
      await route.fulfill(fixtureJson(newCamera))
      return
    }
    if (!pathname.startsWith("/api/cameras/")) {
      await route.fallback()
      return
    }

    const targetId = pathname.slice("/api/cameras/".length)
    const current = cameras.find((camera) => camera.camera_id === targetId)
    if (current === undefined) {
      await route.fulfill(
        fixtureJson({ detail: { code: "camera_missing", message: "Camera not found." } }, 404),
      )
      return
    }
    if (method === "PATCH") {
      api.updateBodies.push(request.postData() ?? "")
      const versionHeader = request.headers()["x-camera-version"]
      api.updateVersions.push(Number(versionHeader))
      if (api.unauthorizedNextMutation) {
        api.unauthorizedNextMutation = false
        await route.fulfill(
          fixtureJson({ detail: { code: "unauthorized", message: "Session expired." } }, 401),
        )
        return
      }
      if (api.staleNextUpdate) {
        api.staleNextUpdate = false
        const bumped = { ...current, version: current.version + 1 }
        cameras = cameras.map((camera) => (camera.camera_id === targetId ? bumped : camera))
        await route.fulfill(
          fixtureJson(
            { detail: { code: "stale_version", message: "Camera version is stale." } },
            409,
          ),
        )
        return
      }
      const body = recordPayload(request.postData())
      if (typeof body["source_url"] === "string" && body["source_url"].includes("offline")) {
        await route.fulfill(
          fixtureJson(
            {
              detail: {
                code: "source_unavailable",
                message: "The RTSP source could not be reached.",
              },
            },
            422,
          ),
        )
        return
      }
      const updated: FixtureCamera = {
        ...current,
        name: typeof body["name"] === "string" ? body["name"] : current.name,
        detection_enabled:
          typeof body["detection_enabled"] === "boolean"
            ? body["detection_enabled"]
            : current.detection_enabled,
        detection_threshold:
          typeof body["detection_threshold"] === "number"
            ? body["detection_threshold"]
            : current.detection_threshold,
        version: current.version + 1,
      }
      cameras = cameras.map((camera) => (camera.camera_id === targetId ? updated : camera))
      await route.fulfill(fixtureJson(updated))
      return
    }
    if (method === "DELETE") {
      api.deleteBodies.push(request.postData() ?? "")
      const versionHeader = request.headers()["x-camera-version"]
      api.deleteVersions.push(Number(versionHeader))
      if (api.holdDelete) {
        await new Promise<void>((release) => {
          releaseDeletePromise = release
        })
      }
      cameras = cameras.filter((camera) => camera.camera_id !== targetId)
      await route.fulfill({
        status: 204,
        headers: { "X-Source-QA-Fixture": SOURCE_QA_FIXTURE_ID },
      })
      return
    }
    await route.fallback()
  })
  await page.route("**/api/settings", async (route) => {
    const request = route.request()
    if (request.method() === "GET") {
      await route.fulfill(fixtureJson(settings))
      return
    }
    if (request.method() !== "PATCH") {
      await route.fallback()
      return
    }
    api.settingsBodies.push(request.postData() ?? "")
    if (api.unauthorizedNextMutation) {
      api.unauthorizedNextMutation = false
      await route.fulfill(
        fixtureJson({ detail: { code: "unauthorized", message: "Session expired." } }, 401),
      )
      return
    }
    if (api.settingsConflict) {
      api.settingsConflict = false
      await route.fulfill(
        fixtureJson(
          { detail: { code: "stale_settings", message: "Settings changed elsewhere." } },
          409,
        ),
      )
      return
    }
    const body = recordPayload(request.postData())
    settings = {
      ...settings,
      retention_days:
        typeof body["retention_days"] === "number"
          ? body["retention_days"]
          : settings.retention_days,
      quota_bytes:
        typeof body["quota_bytes"] === "number" ? body["quota_bytes"] : settings.quota_bytes,
    }
    await route.fulfill(fixtureJson(settings))
  })
  return api
}

async function openCameras(page: Page): Promise<void> {
  await page.goto("/?sourceQA=task-17-labelled-client-fixture")
  await page.getByRole("button", { name: "Cameras" }).click()
  await expect(page.getByRole("heading", { name: "Cameras", exact: true })).toBeVisible()
  await expect(page.getByRole("heading", { name: "North entrance", exact: true })).toBeVisible()
}

function evidenceRoot(): string {
  const configured = process.env["GW_E2E_EVIDENCE_ROOT"]
  return configured === undefined || configured === ""
    ? resolve(process.cwd(), "../.omo/evidence/task-17/source/browser")
    : resolve(configured)
}

test("source QA fixture proves test metadata and create request boundaries", async ({ page }) => {
  const api = await installFixture(page)
  await openCameras(page)
  await page.getByRole("button", { name: "Add camera" }).click()
  await page.getByLabel("Camera name").fill("Field entrance")
  const source = "rtsp://operator:secret@fixture-publisher:8554/live"
  await page.getByLabel("RTSP source").fill(source)
  await page.getByRole("button", { name: "Test connection" }).click()
  await expect(page.getByText("Connection verified")).toBeVisible()
  await expect(page.getByText("fixture-publisher:8554 · h264 · 1920×1080")).toBeVisible()
  expect(await page.locator("body").innerText()).not.toContain("secret")
  await page.getByRole("button", { name: "Create camera" }).click()
  await expect(page.getByRole("heading", { name: "Field entrance", exact: true })).toBeVisible()

  expect(recordPayload(api.testBodies.at(0) ?? null)).toEqual({ source_url: source })
  expect(recordPayload(api.createBodies.at(0) ?? null)).toEqual({
    name: "Field entrance",
    source_url: source,
    detection_enabled: true,
    detection_threshold: 0.5,
  })
})

test("source QA fixture preserves edit drafts across 422 and stale-version responses", async ({
  page,
}) => {
  const api = await installFixture(page)
  await openCameras(page)
  const row = page.locator('[data-camera-id="' + cameraId + '"]')
  await row.getByRole("button", { name: "Edit" }).click()
  const source = page.getByLabel("Replacement RTSP source (optional)")
  const threshold = page.getByLabel("Detection threshold")
  await source.fill("rtsp://offline:8554/live")
  await threshold.fill("0.75")
  await page.getByRole("button", { name: "Save camera" }).click()
  await expect(page.getByText(/source_unavailable:/)).toBeVisible()
  await expect(source).toHaveValue("rtsp://offline:8554/live")
  await expect(row).toContainText("fixture-publisher:8554")

  api.staleNextUpdate = true
  await page.getByRole("button", { name: "Save camera" }).click()
  await expect(
    page.getByText(
      "This camera changed elsewhere. The latest row was requested; review your draft before trying again.",
    ),
  ).toBeVisible()
  await expect(source).toHaveValue("rtsp://offline:8554/live")
  await expect(row).toContainText("fixture-publisher:8554")
  await expect(row.locator("dd").nth(1)).toHaveText("8")

  await source.fill("")
  await page.getByLabel("Detection enabled").uncheck()
  await page.getByRole("button", { name: "Save camera" }).click()
  await expect(page.getByText("Detection paused")).toBeVisible()

  const successfulPatch = recordPayload(api.updateBodies.at(-1) ?? null)
  expect(successfulPatch).not.toHaveProperty("source_url")
  expect(successfulPatch).toMatchObject({
    detection_enabled: false,
    detection_threshold: 0.75,
  })
  expect(api.updateVersions).toEqual([7, 7, 8])
})

test("source QA fixture keeps deletion reversible until 204 and states retained history", async ({
  page,
}) => {
  const api = await installFixture(page)
  await openCameras(page)
  const row = page.locator('[data-camera-id="' + cameraId + '"]')
  const deleteButton = row.getByRole("button", { name: "Delete" })
  await deleteButton.click()
  const dialog = page.getByRole("dialog")
  await expect(dialog).toContainText("appearance history")
  await page.keyboard.press("Escape")
  await expect(dialog).not.toBeVisible()
  await expect(deleteButton).toBeFocused()

  await deleteButton.click()
  api.holdDelete = true
  await dialog.getByRole("button", { name: "Remove camera" }).click()
  await expect(page.getByText("Removing camera…")).toBeVisible()
  await expect(row).toBeVisible()
  api.releaseDelete()
  await expect(row).toHaveCount(0)
  expect(api.deleteBodies).toHaveLength(1)
  expect(api.deleteVersions).toEqual([7])
})

test("source QA fixture preserves invalid retention drafts and sends only settings fields", async ({
  page,
}) => {
  const api = await installFixture(page)
  await openCameras(page)
  const days = page.getByLabel("Retention days")
  const quota = page.getByLabel("Managed quota (decimal GB)")
  await days.fill("0")
  await quota.fill("5")
  await page.getByRole("button", { name: "Save retention" }).click()
  await expect(page.getByText("Retention days must be a positive whole number.")).toBeVisible()
  await expect(days).toHaveValue("0")
  expect(api.settingsBodies).toHaveLength(0)

  api.settingsConflict = true
  await days.fill("14")
  await quota.fill("12.5")
  await page.getByRole("button", { name: "Save retention" }).click()
  await expect(page.getByText(/stale_settings:/)).toBeVisible()
  await expect(days).toHaveValue("14")
  await expect(quota).toHaveValue("12.5")

  await page.getByRole("button", { name: "Save retention" }).click()
  await expect(days).toHaveValue("14")
  expect(recordPayload(api.settingsBodies.at(-1) ?? null)).toEqual({
    retention_days: 14,
    quota_bytes: 12_500_000_000,
  })
})

test("source QA fixture rejects malformed client input before network calls", async ({ page }) => {
  const api = await installFixture(page)
  await openCameras(page)
  await page.getByRole("button", { name: "Add camera" }).click()
  await page.getByLabel("Camera name").fill("Malformed probe")
  await page.getByLabel("RTSP source").fill("https://not-an-rtsp-source/live")
  await page.getByRole("button", { name: "Test connection" }).click()
  await expect(page.getByText("The source URL must use the rtsp:// scheme.")).toBeVisible()
  expect(api.testBodies).toHaveLength(0)

  await page.getByLabel("RTSP source").fill("rtsp://fixture-publisher:8554/live")
  await page.getByLabel("Detection threshold").fill("0")
  await page.getByRole("button", { name: "Create camera" }).click()
  await expect(page.getByText("Detection threshold must be between 0.10 and 0.95.")).toBeVisible()
  expect(api.createBodies).toHaveLength(0)
})

test("source QA fixture reports session expiry on a failed camera mutation", async ({ page }) => {
  const api = await installFixture(page)
  await openCameras(page)
  await page.getByRole("button", { name: "Edit" }).first().click()
  await page.getByLabel("Detection threshold").fill("0.60")
  api.unauthorizedNextMutation = true
  await page.getByRole("button", { name: "Save camera" }).click()
  await expect(page.getByText("Your session expired. Sign in again.")).toBeVisible()
})

test("source QA fixture captures camera settings at required responsive widths", async ({
  page,
}, testInfo) => {
  const consoleErrors: string[] = []
  const pageErrors: string[] = []
  page.on("console", (message) => {
    if (message.type() === "error") {
      consoleErrors.push(message.text())
    }
  })
  page.on("pageerror", (error) => {
    pageErrors.push(error.message)
  })

  await installFixture(page)
  await openCameras(page)
  const root = evidenceRoot()
  await mkdir(root, { recursive: true })
  for (const width of [375, 768, 1280, 1440]) {
    await page.setViewportSize({ width, height: 900 })
    await page.screenshot({
      path: join(root, "cameras-settings-" + width + ".png"),
      fullPage: true,
    })
  }
  await writeFile(
    join(root, "cameras-browser-observation.json"),
    JSON.stringify(
      {
        fixture_id: SOURCE_QA_FIXTURE_ID,
        viewport_widths: [375, 768, 1280, 1440],
        console_errors: consoleErrors,
        page_errors: pageErrors,
      },
      null,
      2,
    ) + "\n",
    "utf8",
  )
  expect(consoleErrors).toEqual([])
  expect(pageErrors).toEqual([])
  await expect(page.getByRole("heading", { name: "Cameras", exact: true })).toBeVisible()
  void testInfo
})
