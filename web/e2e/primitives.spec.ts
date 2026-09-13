import { mkdir, writeFile } from "node:fs/promises"
import { isAbsolute, join, resolve } from "node:path"
import { expect, test } from "@playwright/test"

const widths = [375, 768, 1280, 1440] as const

function evidenceRoot(): string {
  const configured = process.env["GW_E2E_EVIDENCE_ROOT"]
  if (configured === undefined || configured === "" || !isAbsolute(configured)) {
    throw new Error("GW_E2E_EVIDENCE_ROOT must be an absolute directory.")
  }
  return resolve(configured)
}

test("showcase renders every primitive at the required widths", async ({ page }) => {
  const browserErrors: string[] = []
  page.on("console", (message) => {
    if (message.type() === "error") {
      browserErrors.push(message.text())
    }
  })
  page.on("pageerror", (error) => browserErrors.push(error.message))
  await page.goto("/showcase", { waitUntil: "networkidle" })
  await expect(page.getByRole("heading", { name: "Design primitives" })).toBeVisible()

  const referenceContract = await page.evaluate(async () => {
    await document.fonts.ready
    const root = getComputedStyle(document.documentElement)
    const header = document.querySelector<HTMLElement>(".showcase__masthead")
    const button = document.querySelector<HTMLElement>(".gw-button")
    const geometry = document.querySelector<HTMLElement>(".showcase__geometry")
    const panel = document.querySelector<HTMLElement>(".gw-panel")
    if (header === null || button === null || geometry === null || panel === null) {
      throw new Error("Showcase geometry targets are missing.")
    }
    const geometryStyle = getComputedStyle(geometry)
    const headerStyle = getComputedStyle(header)
    return {
      accent: root.getPropertyValue("--gw-color-accent").trim(),
      body_font: getComputedStyle(document.body).fontFamily,
      button_radius: getComputedStyle(button).borderRadius,
      font_resources: performance
        .getEntriesByType("resource")
        .map((entry) => entry.name)
        .filter((name) => name.includes("/fonts/"))
        .sort(),
      geometry_columns: geometryStyle.gridTemplateColumns,
      geometry_gap: geometryStyle.columnGap,
      header_background: headerStyle.backgroundColor,
      header_height: headerStyle.height,
      line: root.getPropertyValue("--gw-color-line").trim(),
      page: root.getPropertyValue("--gw-color-page").trim(),
      panel_radius: getComputedStyle(panel).borderRadius,
      wall_gap: root.getPropertyValue("--gw-space-wall").trim(),
      wall_left: root.getPropertyValue("--gw-wall-left").trim(),
      wall_right: root.getPropertyValue("--gw-wall-right").trim(),
    }
  })
  expect(referenceContract).toMatchObject({
    accent: "#9bdcff",
    body_font: '"DM Sans", ui-sans-serif, system-ui, sans-serif',
    button_radius: "2px",
    geometry_gap: "10px",
    header_background: "rgb(7, 16, 26)",
    header_height: "54px",
    line: "#21384b",
    page: "#03070c",
    panel_radius: "2px",
    wall_gap: "0.625rem",
    wall_left: "232px",
    wall_right: "268px",
  })
  expect(referenceContract.geometry_columns).toMatch(/^232px \d+(?:\.\d+)?px 268px$/)
  expect(referenceContract.font_resources).toHaveLength(2)
  expect(referenceContract.font_resources[0]).toMatch(/\/fonts\/dm-sans-latin\.woff2$/)
  expect(referenceContract.font_resources[1]).toMatch(/\/fonts\/space-grotesk-latin\.woff2$/)
  await writeFile(
    join(evidenceRoot(), "reference-fidelity.json"),
    `${JSON.stringify(referenceContract, null, 2)}\n`,
    "utf8",
  )

  for (const width of widths) {
    await page.setViewportSize({ width, height: 1000 })
    await page.screenshot({
      path: join(evidenceRoot(), `task-5-primitives-${width}.png`),
      fullPage: true,
    })
  }
  expect(browserErrors).toEqual([])
  await writeFile(
    join(evidenceRoot(), "browser-errors.json"),
    `${JSON.stringify({ errors: browserErrors }, null, 2)}\n`,
    "utf8",
  )
})

test("invalid, disabled, loading, reduced-motion, and dialog keyboard states are accessible", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" })
  await page.goto("/showcase", { waitUntil: "networkidle" })

  const invalidInput = page.getByLabel("Invalid camera name")
  await expect(invalidInput).toHaveAttribute("aria-invalid", "true")
  await expect(page.getByText("Enter a camera name.")).toBeVisible()
  await expect(page.getByRole("button", { name: "Disabled action" })).toBeDisabled()
  await expect(page.getByRole("button", { name: "Connecting" })).toHaveAttribute(
    "aria-busy",
    "true",
  )

  const motionDuration = await page
    .getByRole("button", { name: "Primary action" })
    .evaluate((element) => getComputedStyle(element).transitionDuration)
  expect(motionDuration).toBe("0s")

  const dialogTrigger = page.getByRole("button", { name: "Open dialog" })
  await dialogTrigger.focus()
  await dialogTrigger.press("Enter")
  const dialog = page.getByRole("dialog", { name: "Remove camera?" })
  await expect(dialog).toBeVisible()
  await expect(page.getByRole("button", { name: "Cancel" })).toBeFocused()
  await page.keyboard.press("Tab")
  await expect(page.getByRole("button", { name: "Remove" })).toBeFocused()
  await page.keyboard.press("Tab")
  await expect(page.getByRole("button", { name: "Cancel" })).toBeFocused()
  await page.keyboard.press("Escape")
  await expect(dialog).toBeHidden()
  await expect(dialogTrigger).toBeFocused()

  const accessibility = {
    dialog: "focus trapped; Escape closes; focus returns to trigger",
    disabled: true,
    invalid: await invalidInput.getAttribute("aria-invalid"),
    reduced_motion_transition_duration: motionDuration,
  }
  await mkdir(evidenceRoot(), { recursive: true })
  await writeFile(
    join(evidenceRoot(), "accessibility.json"),
    `${JSON.stringify(accessibility, null, 2)}\n`,
    "utf8",
  )
})
