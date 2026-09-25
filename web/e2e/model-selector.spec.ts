import { expect, type Page, test } from "@playwright/test"

const B16 = "openai/clip-vit-base-patch16"
const B32 = "openai/clip-vit-base-patch32"
const L14 = "openai/clip-vit-large-patch14"

const session = {
  authenticated: true,
  idle_expires_at: new Date(Date.now() + 30 * 60 * 1_000).toISOString(),
  absolute_expires_at: new Date(Date.now() + 8 * 60 * 60 * 1_000).toISOString(),
}

function catalog(
  activeModelId = B16,
  transition: Record<string, unknown> | null = null,
  maintenance = false,
) {
  return {
    active_model_id: activeModelId,
    maintenance,
    models: [
      {
        model_id: B16,
        display_name: "OpenAI CLIP ViT-B/16",
        revision: "fixture-revision",
        quality_passed: true,
        quality_reason: null as string | null,
        dimension: 512,
        prepared: true,
        reason: null,
      },
      {
        model_id: B32,
        display_name: "OpenAI CLIP ViT-B/32",
        revision: "fixture-revision",
        quality_passed: true,
        quality_reason: null as string | null,
        dimension: 512,
        prepared: false,
        reason: "Model assets are not prepared.",
      },
      {
        model_id: L14,
        display_name: "OpenAI CLIP ViT-L/14",
        revision: "fixture-revision",
        quality_passed: true,
        quality_reason: null as string | null,
        dimension: 768,
        prepared: true,
        reason: null,
      },
    ],
    transition,
  }
}

function transition(
  phase: string,
  processed: number,
  total: number,
  skipped = 0,
  error: string | null = null,
) {
  return {
    id: "switch-01",
    source_model_id: B16,
    target_model_id: L14,
    phase,
    processed,
    total,
    skipped,
    skip_reasons: skipped === 0 ? {} : { missing_crop: 1, corrupt_crop: 1 },
    error,
  }
}

async function installAuthenticatedShell(page: Page) {
  await page.route("**/api/settings/models/preflight?*", async (route) => {
    const modelId = new URL(route.request().url()).searchParams.get("model_id")
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        target_model_id: modelId,
        retained_count: 40,
        estimated_missing_count: 2,
        measured_crops_per_second: 10,
        measured_fixed_seconds: 20,
        estimated_seconds: 23.8,
        max_seconds: 900,
        eligible: true,
        reason: null,
      }),
    })
  })
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
    await route.fulfill({ contentType: "application/json", body: "[]" })
  })
  await page.route("**/api/settings", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        retention_days: 7,
        quota_bytes: 100_000_000_000,
        wall_slot_ids: [null, null, null, null],
      }),
    })
  })
}

async function openSettings(page: Page): Promise<void> {
  await page.goto("/")
  await page.getByRole("button", { name: "Cameras", exact: true }).click()
  await expect(page.getByRole("heading", { name: "Cameras" })).toBeVisible()
  await expect(page.getByRole("heading", { name: "Person search model" })).toBeVisible()
}

test.describe("model selector", () => {
  test("accepts a preflight response slower than the polling interval", async ({ page }) => {
    await installAuthenticatedShell(page)
    let requests = 0
    await page.route("**/api/settings/models/preflight?*", async (route) => {
      requests += 1
      await new Promise((resolve) => setTimeout(resolve, 6_000))
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          target_model_id: L14,
          retained_count: 40,
          estimated_missing_count: 0,
          measured_crops_per_second: 10,
          measured_fixed_seconds: 20,
          estimated_seconds: 24,
          max_seconds: 900,
          eligible: true,
          reason: null,
        }),
      })
    })
    await page.route(/\/api\/settings\/models(?:\?.*)?$/, (route) =>
      route.fulfill({ contentType: "application/json", body: JSON.stringify(catalog()) }),
    )
    await openSettings(page)
    await page.locator(`input[value="${L14}"]`).check()
    await expect(page.locator(".model-preflight")).toContainText("24", { timeout: 9_000 })
    expect(requests).toBe(1)
  })

  test("warns before applying, polls durable progress, and reports skipped crops", async ({
    page,
  }) => {
    await installAuthenticatedShell(page)
    let accepted = false
    let pollCount = 0
    const applyBodies: string[] = []
    await page.route(/\/api\/settings\/models(?:\/apply)?(?:\?.*)?$/, async (route) => {
      if (route.request().method() === "POST") {
        applyBodies.push(route.request().postData() ?? "")
        accepted = true
        await route.fulfill({
          status: 202,
          contentType: "application/json",
          body: JSON.stringify(catalog(B16, transition("queued", 0, 40), true)),
        })
        return
      }
      if (!accepted) {
        await route.fulfill({ contentType: "application/json", body: JSON.stringify(catalog()) })
        return
      }
      pollCount += 1
      const next =
        pollCount === 1
          ? catalog(B16, transition("reindexing", 14, 40, 2), true)
          : catalog(L14, transition("succeeded", 40, 40, 2), false)
      await route.fulfill({ contentType: "application/json", body: JSON.stringify(next) })
    })

    await openSettings(page)
    const unprepared = page.locator(`input[value="${B32}"]`)
    await expect(unprepared).toBeDisabled()
    await expect(page.locator(".model-option--disabled")).toContainText("not prepared")
    // biome-ignore lint/complexity/useLiteralKeys: TypeScript's process.env index signature requires this form.
    const screenshotPath = process.env["GW_MODEL_SELECTOR_SCREENSHOT"]
    if (screenshotPath !== undefined) {
      await page.screenshot({ fullPage: true, path: screenshotPath })
    }

    await page.locator(`input[value="${L14}"]`).check()
    await expect(
      page.locator(`label[for="model-option-${L14.replaceAll("/", "-")}"]`),
    ).toContainText("Quality approved")
    await expect(page.locator(".model-preflight")).toContainText("Expected skipped crops: 2")
    await page.getByRole("button", { name: "Apply model" }).click()
    await expect(page.getByRole("dialog")).toBeVisible()
    await expect(page.getByRole("dialog")).toContainText("pauses person analysis and search")
    expect(applyBodies).toEqual([])

    await page.getByRole("dialog").getByRole("button", { name: "Start model change" }).click()
    await expect.poll(() => applyBodies.length).toBe(1)
    expect(applyBodies[0]).toBe(JSON.stringify({ model_id: L14 }))
    await expect(page.locator(".model-settings__maintenance")).toBeVisible()
    await expect(page.locator(".model-transition")).toContainText("Re-embedding retained crops")
    await expect(page.locator(".model-transition")).toContainText(
      "14 of 40 retained crops processed",
    )
    await expect(page.locator(".model-transition")).toContainText("2 crops skipped")
    await expect(page.locator(".model-transition")).toContainText("Missing crop: 1")
    await expect(page.locator(".model-transition")).toContainText("Unreadable crop: 1")
    await expect(page.locator(".model-transition")).toContainText("Model change complete", {
      timeout: 4_000,
    })
  })

  test("reconciles a duplicate submission and makes controls usable after success", async ({
    page,
  }) => {
    await installAuthenticatedShell(page)
    let statusReads = 0
    let applyStarted = false
    let applyCalls = 0
    await page.route(/\/api\/settings\/models(?:\/apply)?(?:\?.*)?$/, async (route) => {
      if (route.request().method() === "POST") {
        applyCalls += 1
        applyStarted = true
        await route.fulfill({
          status: 409,
          contentType: "application/json",
          body: JSON.stringify({ detail: { code: "model_transition_conflict", message: "Busy." } }),
        })
        return
      }
      const next = applyStarted
        ? (() => {
            statusReads += 1
            return statusReads === 1
              ? catalog(B16, transition("reindexing", 8, 12), true)
              : catalog(L14, transition("succeeded", 12, 12), false)
          })()
        : catalog()
      await route.fulfill({ contentType: "application/json", body: JSON.stringify(next) })
    })

    await openSettings(page)
    await page.locator(`input[value="${L14}"]`).check()
    await page.getByRole("button", { name: "Apply model" }).click()
    await page.getByRole("dialog").getByRole("button", { name: "Start model change" }).click()
    await expect.poll(() => applyCalls).toBe(1)
    await expect(page.locator(".model-transition")).toContainText("Re-embedding retained crops")
    await expect(page.locator(".model-transition")).toContainText("Model change complete", {
      timeout: 4_000,
    })

    await page.locator(`input[value="${B16}"]`).check()
    await expect(page.getByRole("button", { name: "Apply model" })).toBeEnabled()
  })

  test("shows failed outcome after reload from durable status", async ({ page }) => {
    await installAuthenticatedShell(page)
    let reads = 0
    await page.route(/\/api\/settings\/models(?:\/apply)?(?:\?.*)?$/, async (route) => {
      reads += 1
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(
          catalog(B16, transition("failed", 7, 12, 0, "The previous model was restored."), false),
        ),
      })
    })

    await openSettings(page)
    await expect(page.locator(".model-transition")).toContainText("Model change failed")
    await expect(page.locator(".model-transition__error")).toHaveText(
      "The previous model was restored.",
    )
    expect(reads).toBeGreaterThanOrEqual(1)
    const initialReads = reads

    await page.reload()
    await page.getByRole("button", { name: "Cameras", exact: true }).click()
    await expect(page.getByRole("heading", { name: "Person search model" })).toBeVisible()
    await expect(page.locator(".model-transition")).toContainText("Model change failed")
    expect(reads).toBeGreaterThan(initialReads)
  })
})

test("quality-blocked package shows reason and cannot apply", async ({ page }) => {
  await installAuthenticatedShell(page)
  const blocked = catalog()
  const blockedModel = blocked.models[2]
  if (blockedModel === undefined) throw new Error("missing test model")
  blockedModel.quality_passed = false
  blockedModel.quality_reason = "product quality evidence missing"
  await page.route(/\/api\/settings\/models$/, async (route) => {
    await route.fulfill({ contentType: "application/json", body: JSON.stringify(blocked) })
  })
  await openSettings(page)
  await expect(page.locator(`input[value="${L14}"]`)).toBeDisabled()
  await expect(page.locator(`label[for="model-option-${L14.replaceAll("/", "-")}"]`)).toContainText(
    "product quality evidence missing",
  )
})

test("over-limit preflight blocks confirmation", async ({ page }) => {
  await installAuthenticatedShell(page)
  await page.route(/\/api\/settings\/models$/, async (route) => {
    await route.fulfill({ contentType: "application/json", body: JSON.stringify(catalog()) })
  })
  await page.route("**/api/settings/models/preflight?*", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        target_model_id: L14,
        retained_count: 10_000,
        estimated_missing_count: 0,
        measured_crops_per_second: 10,
        measured_fixed_seconds: 20,
        estimated_seconds: 1020,
        max_seconds: 900,
        eligible: false,
        reason: "estimate_exceeds_limit",
      }),
    })
  })
  await openSettings(page)
  await page.locator(`input[value="${L14}"]`).check()
  await expect(page.locator(".model-preflight")).toContainText("1020 seconds")
  await expect(page.getByRole("button", { name: "Apply model" })).toBeDisabled()
})

test("missing full-transition measurement prevents apply", async ({ page }) => {
  await installAuthenticatedShell(page)
  await page.route(/\/api\/settings\/models$/, async (route) => {
    await route.fulfill({ contentType: "application/json", body: JSON.stringify(catalog()) })
  })
  await page.route("**/api/settings/models/preflight?*", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        target_model_id: L14,
        retained_count: 40,
        estimated_missing_count: 2,
        measured_crops_per_second: null,
        measured_fixed_seconds: null,
        estimated_seconds: null,
        max_seconds: 900,
        eligible: false,
        reason: "throughput_unavailable",
      }),
    })
  })
  await openSettings(page)
  await page.locator(`input[value="${L14}"]`).check()
  await expect(page.locator(".model-preflight")).toContainText("unavailable")
  await expect(page.getByRole("button", { name: "Apply model" })).toBeDisabled()
})

test("estimate change at confirmation requires another review", async ({ page }) => {
  await installAuthenticatedShell(page)
  let preflightReads = 0
  let applyCalls = 0
  await page.route(/\/api\/settings\/models(?:\/apply)?$/, async (route) => {
    if (route.request().method() === "POST") applyCalls += 1
    await route.fulfill({ contentType: "application/json", body: JSON.stringify(catalog()) })
  })
  await page.route("**/api/settings/models/preflight?*", async (route) => {
    preflightReads += 1
    const estimate = preflightReads >= 3 ? 50 : 40
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        target_model_id: L14,
        retained_count: estimate,
        estimated_missing_count: 2,
        measured_crops_per_second: 10,
        measured_fixed_seconds: 20,
        estimated_seconds: 20 + (estimate - 2) / 10,
        max_seconds: 900,
        eligible: true,
        reason: null,
      }),
    })
  })
  await openSettings(page)
  await page.locator(`input[value="${L14}"]`).check()
  await expect(page.getByRole("button", { name: "Apply model" })).toBeEnabled()
  await page.getByRole("button", { name: "Apply model" }).click()
  await expect(page.getByRole("dialog")).toBeVisible()
  await page.getByRole("dialog").getByRole("button", { name: "Start model change" }).click()
  await expect(page.getByRole("dialog")).not.toBeVisible()
  await expect(page.getByRole("alert")).toContainText("estimate changed")
  await expect(page.locator(".model-preflight")).toContainText("Retained crops: 50")
  expect(applyCalls).toBe(0)
})
