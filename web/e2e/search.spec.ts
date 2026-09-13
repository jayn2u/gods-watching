import { existsSync, writeFileSync } from "node:fs"
import { expect, type Page, type Response, test } from "@playwright/test"

// Runs only inside the Task 16 real stack: the built web app and the real API share
// one origin, and the pipeline worker has published real appearances.
const password = process.env["GW_E2E_OPERATOR_PASSWORD"]
const cameraName = process.env["GW_E2E_CAMERA_NAME"]
const otherCameraName = process.env["GW_E2E_OTHER_CAMERA_NAME"]
const phase = process.env["GW_E2E_SEARCH_PHASE"] ?? "normal"
const outageMarker = process.env["GW_E2E_OUTAGE_MARKER"]
const recoveryMarker = process.env["GW_E2E_RECOVERY_MARKER"]
const TEXT_QUERY = "a person walking"

function isSearch(response: Response, needle?: string): boolean {
  const request = response.request()
  return (
    request.method() === "POST" &&
    new URL(response.url()).pathname === "/api/search" &&
    (needle === undefined || (request.postData() ?? "").includes(needle))
  )
}

async function signIn(page: Page): Promise<void> {
  await page.goto("/")
  await page.getByLabel("Operator ID").fill("operator")
  await page.getByLabel("Password").fill(password ?? "")
  await page.getByRole("button", { name: "Sign in" }).click()
  await page
    .getByRole("navigation", { name: "Primary" })
    .getByRole("button", { name: "Person search" })
    .click()
  await expect(page.getByRole("heading", { name: "Person search" })).toBeVisible()
}

function collectConsoleErrors(page: Page): string[] {
  const errors: string[] = []
  page.on("console", (message) => {
    if (message.type() === "error") {
      errors.push(message.text())
    }
  })
  page.on("pageerror", (error) => errors.push(error.message))
  return errors
}

test.describe("person search against the real API", () => {
  test.skip(
    password === undefined || cameraName === undefined || otherCameraName === undefined,
    "requires the Task 16 real stack environment",
  )

  test("text search, camera filter, detail, find similar, and back keep state", async ({
    page,
  }) => {
    test.skip(phase !== "normal", "normal-phase test")
    const consoleErrors = collectConsoleErrors(page)
    const camera = cameraName ?? ""
    await signIn(page)

    const query = page.getByLabel("Describe a person")
    await query.fill(TEXT_QUERY)
    await page.getByLabel(camera).check()
    const searchResponse = page.waitForResponse((response) => isSearch(response, TEXT_QUERY))
    await page.getByRole("button", { name: "Search", exact: true }).click()
    const response = await searchResponse
    expect(response.status()).toBe(200)
    const body = (await response.json()) as { results: Array<{ camera_name: string }> }
    expect(body.results.length).toBeGreaterThan(0)
    expect(new Set(body.results.map((result) => result.camera_name))).toEqual(new Set([camera]))

    await expect(page.getByRole("heading", { name: "Results" })).toBeVisible()
    await expect(page.getByRole("button", { name: "View details" })).toHaveCount(
      body.results.length,
    )
    await expect(page.getByAltText(/Person crop from /).first()).toBeVisible()

    await page.getByRole("button", { name: "View details" }).first().click()
    await expect(page.getByRole("heading", { name: camera })).toBeVisible()
    await expect(page.getByText("Detector confidence")).toBeVisible()
    await expect(page.getByText("Cosine similarity", { exact: true })).toBeVisible()

    const similarResponse = page.waitForResponse((candidate) =>
      isSearch(candidate, '"mode":"similar"'),
    )
    await page.getByRole("button", { name: "Find similar" }).click()
    expect((await similarResponse).status()).toBe(200)
    await expect(page.getByRole("heading", { name: "Similar appearances" })).toBeVisible()

    await page.getByRole("button", { name: "Back to results" }).click()
    await expect(page.getByRole("heading", { name: "Results" })).toBeVisible()
    await expect(query).toHaveValue(TEXT_QUERY)
    await expect(page.getByLabel(camera)).toBeChecked()
    expect(consoleErrors).toEqual([])
  })

  test("renders only the latest of two rapid searches", async ({ page }) => {
    test.skip(phase !== "normal", "normal-phase test")
    const first = cameraName ?? ""
    const second = otherCameraName ?? ""
    await signIn(page)

    const query = page.getByLabel("Describe a person")
    await query.fill(TEXT_QUERY)
    await page.getByLabel(first).check()
    await page.getByRole("button", { name: "Search", exact: true }).click()
    await page.getByLabel(first).uncheck()
    await page.getByLabel(second).check()
    const latest = page.waitForResponse(
      (response) => isSearch(response) && (response.request().postData() ?? "").includes(second),
    )
    await page.locator("form.search-form").evaluate((form) => {
      if (!(form instanceof HTMLFormElement)) {
        throw new Error("search form is missing")
      }
      form.requestSubmit()
    })
    const latestBody = (await (await latest).json()) as { results: Array<{ camera_name: string }> }
    expect(latestBody.results.length).toBeGreaterThan(0)

    const headings = page.locator(".search-result h3")
    await expect(headings).toHaveCount(latestBody.results.length)
    await page.waitForTimeout(750)
    await expect(headings).toHaveCount(latestBody.results.length)
    await expect(page.locator(".search-result h3", { hasText: first })).toHaveCount(0)
  })

  test("reports an inference outage and recovers on retry", async ({ page }) => {
    test.skip(phase !== "outage", "outage-phase test")
    test.setTimeout(300_000)
    await signIn(page)

    const query = page.getByLabel("Describe a person")
    await query.fill(TEXT_QUERY)
    await page.getByRole("button", { name: "Search", exact: true }).click()
    const alert = page.locator(".search-state[role='alert']")
    await expect(alert).toContainText("inference_unavailable")
    await expect(page.getByRole("button", { name: "Retry search" })).toBeVisible()

    writeFileSync(outageMarker ?? "", "outage observed\n")
    await expect
      .poll(() => existsSync(recoveryMarker ?? ""), { intervals: [1_000], timeout: 240_000 })
      .toBe(true)

    await page.getByRole("button", { name: "Retry search" }).click()
    await expect(page.getByRole("button", { name: "View details" }).first()).toBeVisible({
      timeout: 60_000,
    })
    await expect(alert).toHaveCount(0)
  })
})
