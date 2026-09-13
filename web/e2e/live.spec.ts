import { expect, type Page, test } from "@playwright/test"

const cameraIds = [
  "00000000-0000-0000-0000-000000000001",
  "00000000-0000-0000-0000-000000000002",
  "00000000-0000-0000-0000-000000000003",
  "00000000-0000-0000-0000-000000000004",
] as const

const cameras = cameraIds.map((cameraId, index) => ({
  camera_id: cameraId,
  name: `Camera ${index + 1}`,
  source_host: `camera-${index + 1}.lan`,
  source_port: 554,
  detection_enabled: true,
  detection_threshold: 0.5,
  version: 1,
  deleted_at: null,
}))

const session = {
  authenticated: true,
  idle_expires_at: "2026-09-08T12:30:00Z",
  absolute_expires_at: "2026-09-08T20:00:00Z",
}

async function installAuthenticatedApi(page: Page) {
  let slots: Array<string | null> = [cameraIds[0], cameraIds[1], null, null]
  const patchBodies: unknown[] = []

  await page.route("**/api/session", async (route) => {
    if (route.request().method() === "DELETE") {
      await route.fulfill({ status: 204, body: "" })
      return
    }
    await route.fulfill({ contentType: "application/json", body: JSON.stringify(session) })
  })
  await page.route("**/api/session/activity", async (route) => {
    await route.fulfill({ contentType: "application/json", body: JSON.stringify(session) })
  })
  await page.route("**/api/cameras", async (route) => {
    await route.fulfill({ contentType: "application/json", body: JSON.stringify(cameras) })
  })
  await page.route("**/api/settings", async (route) => {
    if (route.request().method() === "PATCH") {
      const body = route.request().postDataJSON() as { wall_slot_ids?: Array<string | null> }
      patchBodies.push(body)
      if (body.wall_slot_ids !== undefined) {
        slots = body.wall_slot_ids
      }
    }
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        retention_days: 7,
        quota_bytes: 10_000_000_000,
        wall_slot_ids: slots,
      }),
    })
  })
  await page.route("**/api/live/*/whep", async (route) => {
    await route.fulfill({ status: 503, body: "" })
  })

  return {
    patchBodies,
    get slots() {
      return slots
    },
  }
}

test.describe("authenticated live wall", () => {
  test("renders four durable slots and persists a camera replacement", async ({ page }) => {
    const api = await installAuthenticatedApi(page)
    await page.goto("/")

    await expect(page.locator("[data-wall-slot]")).toHaveCount(4)
    await expect(page.getByRole("heading", { name: "Authenticated live wall" })).toBeVisible()
    await expect(page.getByRole("combobox", { name: "Slot 1 camera" })).toHaveValue(cameraIds[0])
    await expect(page.getByRole("combobox", { name: "Slot 2 camera" })).toHaveValue(cameraIds[1])

    await page.getByRole("combobox", { name: "Slot 2 camera" }).selectOption(cameraIds[3])
    await expect.poll(() => api.patchBodies.length).toBeGreaterThan(0)
    expect(api.slots).toEqual([cameraIds[0], cameraIds[3], null, null])
    await expect(page.locator("[data-wall-slot]").nth(1)).toContainText("Camera 4")
  })

  test("keeps unavailable media truthful and exposes fullscreen controls", async ({ page }) => {
    await installAuthenticatedApi(page)
    await page.addInitScript(() => {
      Object.defineProperty(HTMLElement.prototype, "requestFullscreen", {
        configurable: true,
        value() {
          window.dispatchEvent(new Event("test-fullscreen"))
          return Promise.resolve()
        },
      })
    })
    await page.goto("/")

    await expect(page.getByText("Offline").first()).toBeVisible({ timeout: 12_000 })
    await expect(page.getByRole("button", { name: "Fullscreen slot 1" })).toBeVisible()
    await page.getByRole("button", { name: "Fullscreen slot 1" }).click()
    await expect(page.locator("[data-wall-slot]").first()).toHaveAttribute(
      "data-fullscreen-requested",
      "true",
    )
  })
})
