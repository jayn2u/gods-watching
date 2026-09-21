import { mkdir, writeFile } from "node:fs/promises"
import { join, resolve } from "node:path"
import { expect, type Page, test } from "@playwright/test"

const CAMERA_A = "00000000-0000-0000-0000-000000000001"
const CAMERA_B = "00000000-0000-0000-0000-000000000002"

// This is a source-QA-only fixture. It is mounted directly into the feature and
// never enters the product API or a production screen.
const sourceFixture = {
  cameras: {
    kind: "ready" as const,
    cameras: [
      {
        camera_id: CAMERA_A,
        name: "North gate",
        source_host: "north-gate.fixture",
        source_port: 554,
        detection_enabled: true,
        detection_threshold: 0.5,
        version: 1,
        deleted_at: null,
      },
      {
        camera_id: CAMERA_B,
        name: "South gate",
        source_host: "south-gate.fixture",
        source_port: 554,
        detection_enabled: true,
        detection_threshold: 0.5,
        version: 1,
        deleted_at: null,
      },
    ],
  },
  first: {
    appearance_id: "appearance-source-a",
    camera_id: CAMERA_A,
    camera_name: "North gate",
    session_id: "session-source-a",
    track_id: 17,
    first_seen: "2026-09-08T09:00:00Z",
    last_seen: "2026-09-08T09:00:12Z",
    ended_at: "2026-09-08T09:00:12Z",
    representative_version: 3,
    bounding_box: { x_min: 10, y_min: 20, x_max: 100, y_max: 220 },
    source_width: 1920,
    source_height: 1080,
    detector_confidence: 0.94,
    crop_quality: 0.88,
    model_id: "clip",
    model_revision: "fixture-revision-1",
    similarity: 0.91,
  },
  second: {
    appearance_id: "appearance-source-b",
    camera_id: CAMERA_B,
    camera_name: "South gate",
    session_id: "session-source-b",
    track_id: 23,
    first_seen: "2026-09-08T08:42:00Z",
    last_seen: "2026-09-08T08:42:10Z",
    ended_at: "2026-09-08T08:42:10Z",
    representative_version: 2,
    bounding_box: { x_min: 30, y_min: 24, x_max: 120, y_max: 240 },
    source_width: 1920,
    source_height: 1080,
    detector_confidence: 0.89,
    crop_quality: 0.84,
    model_id: "clip",
    model_revision: "fixture-revision-1",
    similarity: 0.84,
  },
} as const

type FixtureBehavior =
  | "normal"
  | "stale"
  | "response-mismatch"
  | "inference-unavailable"
  | "crop-expired"
  | "unauthorized"

type FixtureCall = Readonly<{
  appearanceId?: string
  kind: string
  request?: Record<string, unknown>
}>

type FixtureObservation = Readonly<{
  calls: readonly FixtureCall[]
  createdUrls: number
  revokedUrls: number
  unauthorizedCount: number
}>

type FixtureWindow = Window & {
  __gwSearchSource?: {
    readonly calls: FixtureCall[]
    readonly createdUrls: number
    readonly revokedUrls: number
    readonly unauthorizedCount: number
    cleanup: () => void
    restore: () => void
    setBehavior: (behavior: FixtureBehavior) => void
  }
}

function artifactDirectory(testTitle: string): string {
  const root = process.env["GW_E2E_EVIDENCE_ROOT"] ?? join(process.cwd(), "test-results")
  const safeTitle = testTitle
    .replace(/[^a-z0-9]+/gi, "-")
    .replace(/^-|-$/gu, "")
    .toLowerCase()
  return join(resolve(root), "search-source", safeTitle)
}

async function mountSourceSearch(page: Page): Promise<void> {
  await page.goto("/showcase")
  await page.evaluate(
    async ({ cameras, first, second }) => {
      type DynamicModule = Record<string, unknown>
      type Harness = NonNullable<FixtureWindow["__gwSearchSource"]>

      const rootElement = document.getElementById("root")
      if (rootElement === null) {
        throw new Error("source-QA root is missing")
      }

      const dynamicImport = (specifier: string): Promise<DynamicModule> =>
        import(/* @vite-ignore */ specifier) as Promise<DynamicModule>
      const [reactModule, reactDomModule, searchModule, clientModule] = await Promise.all([
        dynamicImport("/@id/react"),
        dynamicImport("/@id/react-dom/client"),
        dynamicImport("/src/features/search/index.ts"),
        dynamicImport("/src/app/client.ts"),
      ])
      function moduleExport(module: DynamicModule, key: string): unknown {
        const direct = module[key]
        if (direct !== undefined) {
          return direct
        }
        const defaultExport = module["default"]
        return typeof defaultExport === "object" && defaultExport !== null
          ? (defaultExport as Record<string, unknown>)[key]
          : undefined
      }

      const createElement = moduleExport(reactModule, "createElement")
      const createRoot = moduleExport(reactDomModule, "createRoot")
      const SearchScreen = moduleExport(searchModule, "SearchScreen")
      const HttpError = moduleExport(clientModule, "HttpError")
      if (
        typeof createElement !== "function" ||
        typeof createRoot !== "function" ||
        typeof SearchScreen !== "function" ||
        typeof HttpError !== "function"
      ) {
        throw new Error(
          `source-QA modules did not expose the expected search surface (react=${Object.keys(reactModule).join(",")}, dom=${Object.keys(reactDomModule).join(",")}, search=${Object.keys(searchModule).join(",")}, client=${Object.keys(clientModule).join(",")})`,
        )
      }

      const fixtureWindow = window as FixtureWindow
      let behavior: FixtureBehavior = "normal"
      const calls: FixtureCall[] = []
      let unauthorizedCount = 0
      let createdUrls = 0
      let revokedUrls = 0
      const originalCreateObjectUrl = URL.createObjectURL.bind(URL)
      const originalRevokeObjectUrl = URL.revokeObjectURL.bind(URL)
      const jpegBytes = Uint8Array.from([
        0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00,
        0x01, 0x00, 0x01, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43, 0x00, 0x08, 0x06, 0x06, 0x07, 0x06,
        0x05, 0x08, 0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0a, 0x0c, 0x14, 0x0d, 0x0c, 0x0b, 0x0b,
        0x0c, 0x19, 0x12, 0x13, 0x0f, 0x14, 0x1d, 0x1a, 0x1f, 0x1e, 0x1d, 0x1a, 0x1c, 0x1c, 0x20,
        0x24, 0x2e, 0x27, 0x20, 0x22, 0x2c, 0x23, 0x1c, 0x1c, 0x28, 0x37, 0x29, 0x2c, 0x30, 0x31,
        0x34, 0x34, 0x34, 0x1f, 0x27, 0x39, 0x3d, 0x38, 0x32, 0x3c, 0x2e, 0x33, 0x34, 0x32, 0xff,
        0xc0, 0x00, 0x11, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x11, 0x00, 0x02, 0x11, 0x01, 0x03,
        0x11, 0x01, 0xff, 0xc4, 0x00, 0x14, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0xff, 0xda, 0x00, 0x08, 0x01, 0x01, 0x00,
        0x00, 0x3f, 0x00, 0xfb, 0xd2, 0x8a, 0x28, 0xa0, 0x0f, 0xff, 0xd9,
      ])

      function abortError(): DOMException {
        return new DOMException("The source-QA request was aborted.", "AbortError")
      }

      function waitFor(delay: number, signal: AbortSignal, ignoreAbort: boolean): Promise<void> {
        return new Promise((resolve, reject) => {
          const timer = window.setTimeout(resolve, delay)
          if (ignoreAbort) {
            return
          }
          const abort = () => {
            window.clearTimeout(timer)
            reject(abortError())
          }
          if (signal.aborted) {
            abort()
            return
          }
          signal.addEventListener("abort", abort, { once: true })
        })
      }

      function error(status: number, code: string, message: string): Error {
        return new (HttpError as new (status: number, code: string, message: string) => Error)(
          status,
          code,
          message,
        )
      }

      function resultFor(request: { mode: string; query?: string }) {
        if (request.mode === "similar") {
          return { mode: "similar", results: [second] }
        }
        if (request.mode === "text" && request.query?.toLowerCase().includes("new")) {
          return { mode: "text", results: [second] }
        }
        return { mode: request.mode, results: [first] }
      }

      const client = {
        search(request: Record<string, unknown>, signal: AbortSignal) {
          calls.push({ kind: "search", request })
          const behaviorAtStart = behavior
          if (behaviorAtStart === "unauthorized") {
            return waitFor(8, signal, false).then(() => {
              throw error(401, "session_expired", "The operator session expired.")
            })
          }
          if (behaviorAtStart === "inference-unavailable" && request["mode"] === "text") {
            return waitFor(8, signal, false).then(() => {
              throw error(503, "text_inference_unavailable", "Text inference is unavailable.")
            })
          }
          const query = typeof request["query"] === "string" ? request["query"] : undefined
          if (behaviorAtStart === "response-mismatch") {
            return waitFor(8, signal, false).then(() => ({ mode: "browse", results: [first] }))
          }
          const isOldStaleQuery = behaviorAtStart === "stale" && query?.includes("old")
          return waitFor(isOldStaleQuery ? 120 : 15, signal, behaviorAtStart === "stale").then(() =>
            resultFor({
              mode: String(request["mode"]),
              ...(query === undefined ? {} : { query }),
            }),
          )
        },
        getAppearance(appearanceId: string, signal: AbortSignal) {
          calls.push({ kind: "appearance", appearanceId })
          return waitFor(10, signal, false).then(() =>
            appearanceId === second.appearance_id ? second : first,
          )
        },
        getAppearanceCrop(appearanceId: string, signal: AbortSignal) {
          calls.push({ kind: "crop", appearanceId })
          const behaviorAtStart = behavior
          if (behaviorAtStart === "crop-expired" && appearanceId === first.appearance_id) {
            return waitFor(10, signal, false).then(() => {
              throw error(404, "appearance_expired", "The appearance crop has expired.")
            })
          }
          return waitFor(8, signal, false).then(() => ({
            blob: new Blob([jpegBytes], { type: "image/jpeg" }),
            etag: `fixture-${appearanceId}`,
            representative_version:
              appearanceId === second.appearance_id
                ? second.representative_version
                : first.representative_version,
          }))
        },
      }

      URL.createObjectURL = (blob: Blob): string => {
        createdUrls += 1
        return originalCreateObjectUrl(blob)
      }
      URL.revokeObjectURL = (url: string): void => {
        revokedUrls += 1
        originalRevokeObjectUrl(url)
      }

      const reactRoot = (
        createRoot as (container: Element) => {
          render: (element: unknown) => void
          unmount: () => void
        }
      )(rootElement)
      reactRoot.render(
        (createElement as (...args: unknown[]) => unknown)(SearchScreen, {
          cameras,
          client,
          onUnauthorized: () => {
            unauthorizedCount += 1
          },
        }),
      )
      const harness = {
        calls,
        cleanup: () => {
          reactRoot.unmount()
        },
        restore: () => {
          URL.createObjectURL = originalCreateObjectUrl
          URL.revokeObjectURL = originalRevokeObjectUrl
        },
        setBehavior: (nextBehavior) => {
          behavior = nextBehavior
        },
      } as Harness
      fixtureWindow.__gwSearchSource = harness
      Object.defineProperties(harness, {
        createdUrls: { enumerable: true, get: () => createdUrls },
        revokedUrls: { enumerable: true, get: () => revokedUrls },
        unauthorizedCount: { enumerable: true, get: () => unauthorizedCount },
      })
    },
    { cameras: sourceFixture.cameras, first: sourceFixture.first, second: sourceFixture.second },
  )
}

async function fixtureObservation(page: Page): Promise<FixtureObservation> {
  return page.evaluate(() => {
    const harness = (window as FixtureWindow).__gwSearchSource
    if (harness === undefined) {
      throw new Error("source-QA fixture is not mounted")
    }
    return {
      calls: harness.calls.map((call) => ({
        kind: call.kind,
        ...(call.appearanceId === undefined ? {} : { appearanceId: call.appearanceId }),
        ...(call.request === undefined ? {} : { request: call.request }),
      })),
      createdUrls: harness.createdUrls,
      revokedUrls: harness.revokedUrls,
      unauthorizedCount: harness.unauthorizedCount,
    }
  })
}

async function setFixtureBehavior(page: Page, behavior: FixtureBehavior): Promise<void> {
  await page.evaluate((nextBehavior) => {
    const harness = (window as FixtureWindow).__gwSearchSource
    if (harness === undefined) {
      throw new Error("source-QA fixture is not mounted")
    }
    harness.setBehavior(nextBehavior)
  }, behavior)
}

async function unmountSourceSearch(page: Page): Promise<FixtureObservation> {
  return page.evaluate(async () => {
    const harness = (window as FixtureWindow).__gwSearchSource
    if (harness === undefined) {
      return { calls: [], createdUrls: 0, revokedUrls: 0, unauthorizedCount: 0 }
    }
    harness.cleanup()
    await new Promise<void>((resolveCleanup) => window.setTimeout(resolveCleanup, 0))
    harness.restore()
    return {
      calls: harness.calls,
      createdUrls: harness.createdUrls,
      revokedUrls: harness.revokedUrls,
      unauthorizedCount: harness.unauthorizedCount,
    }
  })
}

async function saveObservation(
  directory: string,
  observation: FixtureObservation,
  consoleErrors: readonly string[],
) {
  await mkdir(directory, { recursive: true })
  await writeFile(
    join(directory, "fixture-observation.json"),
    `${JSON.stringify({ fixture: "data-source-qa-fixture", observation, consoleErrors }, null, 2)}\n`,
  )
}

test.describe("person search source fixture", () => {
  test("covers text, filters, detail, similar, back, keyboard, and reference widths", async ({
    page,
  }, testInfo) => {
    const directory = artifactDirectory(testInfo.title)
    await mkdir(directory, { recursive: true })
    const consoleErrors: string[] = []
    page.on("console", (message) => {
      if (message.type() === "error") {
        consoleErrors.push(message.text())
      }
    })
    page.on("pageerror", (error) => consoleErrors.push(error.message))

    let unmountedObservation: FixtureObservation | undefined
    try {
      await mountSourceSearch(page)
      await expect(page.getByRole("heading", { name: "Person search" })).toBeVisible()

      const query = page.getByLabel("Describe a person")
      await query.fill("a person in a red jacket with a bag")
      await page.getByLabel("North gate").check()
      await page.getByLabel("From (local, inclusive)").fill("2026-09-08T09:00")
      await page.getByLabel("To (local, inclusive)").fill("2026-09-08T10:00")
      await page.getByRole("button", { name: "Search" }).click()
      await expect(page.getByRole("heading", { name: "Results" })).toBeVisible()
      await expect(page.getByRole("button", { name: "View details" })).toHaveCount(1)
      await expect(page.getByAltText(/Person crop from North gate/)).toHaveCount(1)

      const afterSearch = await fixtureObservation(page)
      const searchCall = [...afterSearch.calls].reverse().find((call) => call.kind === "search")
      expect(searchCall?.request).toMatchObject({
        camera_ids: [CAMERA_A],
        mode: "text",
        query: "a person in a red jacket with a bag",
        sort: "similarity",
      })
      expect(searchCall?.request?.["from"]).toEqual(expect.stringMatching(/Z$/u))
      expect(searchCall?.request?.["to"]).toEqual(expect.stringMatching(/Z$/u))

      await page.getByRole("button", { name: "View details" }).click()
      await expect(page.getByRole("heading", { name: "North gate" })).toBeVisible()
      await expect(page.getByAltText(/Person crop from North gate/)).toHaveCount(1)
      await expect(page.getByText("Detector confidence")).toBeVisible()
      await expect(page.getByText("Cosine similarity", { exact: true })).toBeVisible()
      await expect(page.getByText("0.94")).toBeVisible()
      await expect(page.getByText("0.91")).toBeVisible()

      await page.getByRole("button", { name: "Find similar" }).click()
      await expect(page.getByRole("heading", { name: "Similar appearances" })).toBeVisible()
      await expect(page.getByText("Cosine ranked")).toBeVisible()
      const afterSimilar = await fixtureObservation(page)
      const similarCall = [...afterSimilar.calls]
        .reverse()
        .find((call) => call.kind === "search" && call.request?.["mode"] === "similar")
      expect(similarCall?.request).toMatchObject({
        appearance_id: sourceFixture.first.appearance_id,
        camera_ids: [CAMERA_A],
        mode: "similar",
        sort: "similarity",
      })

      await page.getByRole("button", { name: "Back to results" }).click()
      await expect(page.getByRole("heading", { name: "Results" })).toBeVisible()
      await expect(query).toHaveValue("a person in a red jacket with a bag")
      await expect(page.getByLabel("North gate")).toBeChecked()
      await expect(page.getByLabel("From (local, inclusive)")).toHaveValue("2026-09-08T09:00")

      await query.focus()
      await expect(query).toBeFocused()
      await page.keyboard.press("Tab")
      await expect(page.locator(":focus-visible")).toHaveCount(1)

      for (const width of [375, 768, 1280, 1440]) {
        await page.setViewportSize({ height: 900, width })
        await expect(
          page.evaluate(() => document.documentElement.scrollWidth),
        ).resolves.toBeLessThanOrEqual(width)
        const dateControls = await page
          .locator('input[type="datetime-local"]')
          .evaluateAll((inputs) => {
            const [from, to] = inputs.map((input) => input.getBoundingClientRect())
            if (from === undefined || to === undefined) {
              return false
            }
            return (
              from.right <= to.left ||
              to.right <= from.left ||
              from.bottom <= to.top ||
              to.bottom <= from.top
            )
          })
        expect(dateControls).toBe(true)
        await page.screenshot({
          fullPage: true,
          path: join(directory, `task-16-search-${width}.png`),
        })
      }
      expect(consoleErrors).toEqual([])
    } finally {
      unmountedObservation = await unmountSourceSearch(page)
      await saveObservation(directory, unmountedObservation, consoleErrors)
    }

    expect(unmountedObservation).toBeDefined()
    expect(unmountedObservation?.revokedUrls).toBeGreaterThanOrEqual(
      unmountedObservation?.createdUrls ?? 0,
    )
  })

  test("fences stale responses, supports cancellation, and reports recoverable failures", async ({
    page,
  }, testInfo) => {
    const directory = artifactDirectory(testInfo.title)
    await mkdir(directory, { recursive: true })
    const consoleErrors: string[] = []
    page.on("console", (message) => {
      if (message.type() === "error") {
        consoleErrors.push(message.text())
      }
    })
    page.on("pageerror", (error) => consoleErrors.push(error.message))

    let unmountedObservation: FixtureObservation | undefined
    try {
      await mountSourceSearch(page)
      const query = page.getByLabel("Describe a person")

      await setFixtureBehavior(page, "stale")
      await query.fill("old query")
      await page.getByRole("button", { name: "Search" }).click()
      await expect(page.getByRole("button", { name: "Cancel" })).toBeVisible()
      await query.fill("new query")
      await page.locator("form.search-form").evaluate((form) => {
        if (!(form instanceof HTMLFormElement)) {
          throw new Error("search source-QA form is missing")
        }
        form.requestSubmit()
      })
      await expect(page.locator(".search-result h3", { hasText: "South gate" })).toBeVisible()
      await page.waitForTimeout(150)
      await expect(page.locator(".search-result h3", { hasText: "North gate" })).toHaveCount(0)

      await query.fill("old cancelled query")
      await page.getByRole("button", { name: "Search" }).click()
      await expect(page.getByRole("button", { name: "Cancel" })).toBeVisible()
      await page.getByRole("button", { name: "Cancel" }).click()
      await expect(page.getByText("Waiting for a search")).toBeVisible()

      await setFixtureBehavior(page, "response-mismatch")
      await query.fill("a person")
      await page.getByRole("button", { name: "Search" }).click()
      await expect(page.locator(".search-state[role='alert']")).toContainText(
        "The search service returned an unexpected response. Retry the search.",
      )
      await expect(page.locator(".search-state[role='alert']")).not.toContainText(
        "search_response_mismatch",
      )

      await setFixtureBehavior(page, "inference-unavailable")
      await query.fill("a person")
      await page.getByRole("button", { name: "Search", exact: true }).click()
      await expect(page.locator(".search-state[role='alert']")).toContainText(
        "Person search is temporarily unavailable. Retry in a moment.",
      )
      await expect(page.getByRole("alert")).toHaveCount(1)
      await expect(page.getByRole("button", { name: "Retry search" })).toBeVisible()
      for (const width of [375, 768, 1280]) {
        await page.setViewportSize({ height: 900, width })
        await page.screenshot({
          fullPage: true,
          path: join(directory, `search-unavailable-${width}.png`),
        })
      }

      await setFixtureBehavior(page, "normal")
      await page.getByRole("button", { name: "Retry search" }).click()
      await expect(page.getByRole("button", { name: "View details" })).toHaveCount(1)
      await setFixtureBehavior(page, "crop-expired")
      await page.getByRole("button", { name: "View details" }).click()
      await expect(page.locator(".search-detail__crop--error")).toContainText(
        "This appearance crop has expired under the retention policy.",
      )
      await expect(page.getByRole("alert")).toHaveCount(1)
      await expect(page.getByRole("button", { name: "Retry detail" })).toBeVisible()
      for (const width of [375, 768, 1280]) {
        await page.setViewportSize({ height: 900, width })
        await page.screenshot({
          fullPage: true,
          path: join(directory, `crop-expired-${width}.png`),
        })
      }

      await setFixtureBehavior(page, "normal")
      await page.getByRole("button", { name: "Retry detail" }).click()
      await expect(page.getByAltText(/Person crop from North gate/)).toHaveCount(1)

      await setFixtureBehavior(page, "unauthorized")
      await page.getByRole("button", { name: "Find similar" }).click()
      await expect.poll(async () => (await fixtureObservation(page)).unauthorizedCount).toBe(1)
      expect(consoleErrors).toEqual([])
    } finally {
      unmountedObservation = await unmountSourceSearch(page)
      await saveObservation(directory, unmountedObservation, consoleErrors)
    }

    expect(unmountedObservation).toBeDefined()
    expect(unmountedObservation?.revokedUrls).toBeGreaterThanOrEqual(
      unmountedObservation?.createdUrls ?? 0,
    )
  })
})
