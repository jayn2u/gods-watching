/* biome-ignore-all lint/complexity/useLiteralKeys: process.env and API records require indexed access. */

import { access, mkdir, readFile, writeFile } from "node:fs/promises"
import { dirname, resolve } from "node:path"
import { expect, type Page, type Request, type Response, test } from "@playwright/test"

const phase = process.env["GW_TRAINING_SMOKE_PHASE"] ?? ""
const password = process.env["GW_E2E_OPERATOR_PASSWORD"]
const username = process.env["GW_E2E_OPERATOR_USERNAME"] ?? "admin"
const requestedJobId = process.env["GW_TRAINING_SMOKE_JOB_ID"]
const stateFile = process.env["GW_TRAINING_SMOKE_STATE_FILE"]
const POINTER_ACTION_TIMEOUT_MS = 15_000
let pointerDiagnosticSequence = 0

type DatasetStatus = Readonly<{
  registered: boolean
  valid: boolean
  reason: string | null
  snapshot: Readonly<{
    dataset_id: string
    fingerprint: string
    protocol: string
    image_count: number
    caption_count: number
    identity_count: number
    split_counts: Readonly<
      Record<string, Readonly<{ images: number; captions: number; identities: number }>>
    >
  }> | null
  supervisor: Readonly<{
    state: "validating" | "ready" | "unavailable"
    reason: string | null
    observed_at: string | null
    source_fingerprint: string | null
    dataset_fingerprint: string | null
  }>
}>

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

async function signIn(page: Page): Promise<void> {
  await page.goto("/")
  await page.getByLabel("Operator ID").fill(username)
  await page.getByLabel("Password").fill(password ?? "")
  await page.getByRole("button", { name: "Sign in" }).click()
  await expect(
    page
      .getByRole("navigation", { name: "Primary" })
      .getByRole("button", { name: "Person search" }),
  ).toBeVisible()
}

async function openTraining(page: Page): Promise<void> {
  await page
    .getByRole("navigation", { name: "Primary" })
    .getByRole("button", {
      name: "Training",
      exact: true,
    })
    .click()
  await expect(page.getByRole("heading", { name: "CLIP training" })).toBeVisible()
}

async function refreshDatasetStatus(page: Page): Promise<DatasetStatus> {
  const refreshButton = page.getByRole("button", { name: "Refresh status" })
  await expect(refreshButton).toBeEnabled({ timeout: 30_000 })
  const responsePromise = page
    .waitForRequest(
      (request) =>
        request.method() === "GET" && new URL(request.url()).pathname === "/api/training/datasets",
      { timeout: 30_000 },
    )
    .then((request) =>
      page.waitForResponse((response) => response.request() === request, { timeout: 30_000 }),
    )
  const [, response] = await Promise.all([
    refreshButton.click({ timeout: 15_000 }),
    responsePromise,
  ])
  expect(response.status()).toBe(200)
  const status: unknown = await response.json()
  if (!isRecord(status)) throw new Error("dataset status response was not an object")
  return status as DatasetStatus
}

async function waitForReadiness(page: Page): Promise<DatasetStatus> {
  const deadline = Date.now() + 7 * 60 * 1_000
  let sawDatasetValidation = false
  while (Date.now() < deadline) {
    const status = await refreshDatasetStatus(page)
    if (status.reason === "dataset_validating") sawDatasetValidation = true
    if (status.valid && status.snapshot !== null && status.supervisor.state === "ready") {
      const snapshot = status.snapshot
      const validationSummary = `${snapshot.image_count.toLocaleString()} images, ${snapshot.caption_count.toLocaleString()} captions, and ${snapshot.identity_count.toLocaleString()} identities passed validation.`
      await expect(page.getByRole("button", { name: "Refresh status" })).toBeEnabled({
        timeout: 30_000,
      })
      await expect(page.getByText("Validated", { exact: true })).toBeVisible({ timeout: 30_000 })
      await expect(page.getByText(validationSummary, { exact: true })).toBeVisible({
        timeout: 30_000,
      })
      await expect(page.getByText(snapshot.protocol, { exact: true })).toBeVisible({
        timeout: 30_000,
      })
      const evidence = {
        dataset_id: snapshot.dataset_id,
        fingerprint: snapshot.fingerprint,
        image_count: snapshot.image_count,
        caption_count: snapshot.caption_count,
        identity_count: snapshot.identity_count,
        split_counts: snapshot.split_counts,
        validation_was_observed: sawDatasetValidation,
        supervisor: status.supervisor,
      }
      await writeEvidence({ kind: "readiness", ...evidence })
      return status
    }
    if (
      !status.registered ||
      status.reason === "dataset_not_configured" ||
      status.reason === "dataset_invalid_or_unavailable"
    ) {
      const unavailable = status.reason ?? status.supervisor.reason ?? "training unavailable"
      throw new Error(`training runtime is unavailable: ${unavailable}`)
    }
    await page.waitForTimeout(3_000)
  }
  throw new Error("training dataset and supervisor did not become ready within seven minutes")
}

async function waitForFile(path: string, label: string): Promise<void> {
  await expect
    .poll(
      async () => {
        try {
          await access(path)
          return true
        } catch {
          return false
        }
      },
      { timeout: 90_000, message: `${label} was not created by the runtime controller` },
    )
    .toBe(true)
}

async function listTrainingJobIds(page: Page): Promise<string[]> {
  return page.evaluate(async () => {
    const response = await fetch("/api/training/jobs?limit=50", { credentials: "same-origin" })
    if (!response.ok) throw new Error(`training history returned HTTP ${response.status}`)
    const payload: unknown = await response.json()
    if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
      throw new Error("training history response was not an object")
    }
    const items = (payload as Record<string, unknown>)["items"]
    if (!Array.isArray(items)) throw new Error("training history response omitted its items")
    return items.map((item: unknown) => {
      if (typeof item !== "object" || item === null || Array.isArray(item)) {
        throw new Error("training history item was not an object")
      }
      const id = (item as Record<string, unknown>)["id"]
      if (typeof id !== "string") throw new Error("training history item omitted its identity")
      return id
    })
  })
}

type RuntimeCamera = Readonly<{ camera_id: string; name: string }>
const EXISTING_RUNTIME_CAMERAS: readonly RuntimeCamera[] = [
  { camera_id: "db694fa3-4401-4ea7-86f9-57dc5f9a2315", name: "Training QA Camera 1" },
  { camera_id: "4dc75938-21ec-48c5-a62b-dda5feb1e5f0", name: "Training QA Camera 2" },
  { camera_id: "80825949-7479-450d-9045-4cfcbbd3c041", name: "Training QA Camera 3" },
  { camera_id: "d71201a1-db43-4533-a6ce-f6cae387755f", name: "Training QA Camera 4" },
]
type VideoSample = Readonly<{
  ready_state: number
  width: number
  height: number
  current_time: number
  paused: boolean
}>
type DetectionPoll = Readonly<{
  camera_id: string
  box_count: number
  frame_age_seconds: number | null
}>
type RuntimeCropSearch = Readonly<{
  active_camera_id: string
  minimum_last_seen_after: string | null
  text_query: string
  text_result_count: number
  text_result_camera_ids: readonly string[]
  text_max_last_seen_age_seconds: number
  text_latest_last_seen_at: string
  seed_appearance_id: unknown
  similar_result_count: number
  similar_result_camera_ids: readonly string[]
  similar_max_last_seen_age_seconds: number
  similar_latest_last_seen_at: string
  text_screenshot: string | null
  image_screenshot: string | null
}>

function activeQACamera(cameras: readonly RuntimeCamera[]): RuntimeCamera {
  const camera = cameras.find((candidate) => candidate.name === "Training QA Camera 2")
  if (camera === undefined) throw new Error("the active QA camera was not registered")
  return camera
}

function localDateTimeValue(value: Date): string {
  const local = new Date(value.getTime() - value.getTimezoneOffset() * 60_000)
  return local.toISOString().slice(0, 16)
}

function resultCameraIds(results: readonly Record<string, unknown>[]): string[] {
  return results.map((result) => {
    const cameraId = result["camera_id"]
    if (typeof cameraId !== "string") throw new Error("appearance result omitted its camera ID")
    return cameraId
  })
}

function maxLastSeenAgeSeconds(results: readonly Record<string, unknown>[]): number {
  if (results.length === 0) throw new Error("appearance search returned no evidence rows")
  const now = Date.now()
  const ages = results.map((result) => {
    const lastSeen = result["last_seen"]
    if (typeof lastSeen !== "string") throw new Error("appearance result omitted last_seen")
    const timestamp = Date.parse(lastSeen)
    if (!Number.isFinite(timestamp)) throw new Error("appearance result had an invalid last_seen")
    return (now - timestamp) / 1_000
  })
  return Math.max(...ages)
}

function latestLastSeenAt(results: readonly Record<string, unknown>[]): string {
  if (results.length === 0) throw new Error("appearance search returned no evidence rows")
  const timestamps = results.map((result) => {
    const lastSeen = result["last_seen"]
    if (typeof lastSeen !== "string") throw new Error("appearance result omitted last_seen")
    const timestamp = Date.parse(lastSeen)
    if (!Number.isFinite(timestamp)) throw new Error("appearance result had an invalid last_seen")
    return timestamp
  })
  return new Date(Math.max(...timestamps)).toISOString()
}

function isSearchMode(
  response: import("@playwright/test").Response,
  mode: "text" | "similar",
): boolean {
  const request = response.request()
  return (
    request.method() === "POST" &&
    new URL(response.url()).pathname === "/api/search" &&
    (request.postData() ?? "").includes(`"mode":"${mode}"`)
  )
}

async function openCameraSettings(page: Page): Promise<void> {
  await page
    .getByRole("navigation", { name: "Primary" })
    .getByRole("button", { name: "Cameras", exact: true })
    .click()
  await expect(page.getByRole("heading", { name: "Cameras", exact: true })).toBeVisible()
  const cameraListLoaded = async (): Promise<boolean> =>
    (await page.locator("[data-camera-list]").count()) > 0 ||
    (await page
      .getByText("No cameras are configured. Add a source to begin authenticated ingest.", {
        exact: true,
      })
      .count()) > 0
  await expect.poll(cameraListLoaded, { timeout: 30_000 }).toBe(true)
}

async function registerRuntimeCamera(page: Page, index: number): Promise<RuntimeCamera> {
  const name = `Training QA Camera ${index}`
  const existingHeading = page.getByRole("heading", { name, exact: true })
  if ((await existingHeading.count()) > 0) {
    const existingCard = existingHeading
      .first()
      .locator("xpath=ancestor::article[@data-camera-id][1]")
    const cameraId = await existingCard.getAttribute("data-camera-id")
    if (cameraId === null) throw new Error(`registered camera ${name} omitted its identity`)
    return { camera_id: cameraId, name }
  }

  const sourceUrl = `rtsp://127.0.0.1:38554/camera-${index}`
  await page.getByRole("button", { name: "Add camera" }).click()
  await page.getByLabel("Camera name").fill(name)
  await page.getByLabel("RTSP source").fill(sourceUrl)
  const testResponsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/cameras/test",
    { timeout: 30_000 },
  )
  await page.getByRole("button", { name: "Test connection" }).click()
  const testResponse = await testResponsePromise
  expect(testResponse.status()).toBe(200)
  await expect(page.getByText("Connection verified", { exact: true })).toBeVisible({
    timeout: 30_000,
  })

  const createRequestPromise = page.waitForRequest(
    (request) => request.method() === "POST" && new URL(request.url()).pathname === "/api/cameras",
    { timeout: 30_000 },
  )
  const createResponsePromise = createRequestPromise.then((request) =>
    page.waitForResponse((response) => response.request() === request, { timeout: 30_000 }),
  )
  const [createResponse] = await Promise.all([
    createResponsePromise,
    page.getByRole("button", { name: "Create camera" }).click(),
  ])
  expect([200, 201]).toContain(createResponse.status())
  const cameraValue: unknown = await createResponse.json()
  if (!isRecord(cameraValue) || typeof cameraValue["camera_id"] !== "string") {
    throw new Error(`camera creation for ${name} omitted its identity`)
  }
  const camera = { camera_id: cameraValue["camera_id"], name }
  await expect(page.locator(`.camera-card[data-camera-id="${camera.camera_id}"]`)).toBeVisible({
    timeout: 30_000,
  })
  return camera
}

async function prepareRuntimeWall(page: Page, cameras: readonly RuntimeCamera[]): Promise<void> {
  await page
    .getByRole("navigation", { name: "Primary" })
    .getByRole("button", { name: "Live wall", exact: true })
    .click()
  await expect(page.getByRole("heading", { name: "Authenticated live wall" })).toBeVisible()
  for (const [index, camera] of cameras.entries()) {
    const slot = page.getByRole("combobox", { name: `Slot ${index + 1} camera` })
    if ((await slot.inputValue()) !== camera.camera_id) {
      const requestPromise = page.waitForRequest(
        (request) =>
          request.method() === "PATCH" && new URL(request.url()).pathname === "/api/settings",
        { timeout: 30_000 },
      )
      const responsePromise = requestPromise.then((request) =>
        page.waitForResponse((response) => response.request() === request, { timeout: 30_000 }),
      )
      const [response] = await Promise.all([responsePromise, slot.selectOption(camera.camera_id)])
      expect(response.status()).toBe(200)
    }
    await expect(page.locator("[data-wall-slot]").nth(index)).toHaveAttribute(
      "data-camera-id",
      camera.camera_id,
    )
  }
}

async function sampleRuntimeVideos(
  page: Page,
  cameras: readonly RuntimeCamera[],
  activeCamera: RuntimeCamera,
): Promise<
  Readonly<{
    before: Record<string, VideoSample>
    after: Record<string, VideoSample>
    detections: readonly DetectionPoll[]
    active_detection_camera_id: string
  }>
> {
  const detections: DetectionPoll[] = []
  page.on("response", (response) => {
    if (response.status() !== 200 || response.request().method() !== "GET") return
    const match = /^\/api\/live\/([^/]+)\/detections$/u.exec(new URL(response.url()).pathname)
    if (match === null) return
    void response
      .json()
      .then((body: unknown) => {
        if (!isRecord(body) || !Array.isArray(body["boxes"])) return
        const age = body["frame_age_seconds"]
        detections.push({
          camera_id: match[1] ?? "",
          box_count: body["boxes"].length,
          frame_age_seconds: typeof age === "number" ? age : null,
        })
      })
      .catch(() => undefined)
  })
  const sample = async (camera: RuntimeCamera, slotIndex: number): Promise<VideoSample> => {
    const slot = page.getByRole("region", { name: `Slot ${slotIndex + 1}`, exact: true })
    await expect(slot).toHaveAttribute("data-camera-id", camera.camera_id)
    await expect(slot.getByRole("status").filter({ hasText: /^Live$/u })).toBeVisible({
      timeout: 120_000,
    })
    const video = page.locator(`video[aria-label="${camera.name} live video"]`)
    await expect
      .poll(
        async () =>
          video.evaluate((element) => {
            if (!(element instanceof HTMLVideoElement)) return false
            return element.readyState >= 2 && element.videoWidth > 0 && element.videoHeight > 0
          }),
        { timeout: 60_000, message: `${camera.name} should decode live video frames` },
      )
      .toBe(true)
    return video.evaluate((element) => {
      if (!(element instanceof HTMLVideoElement)) throw new Error("camera video element is missing")
      return {
        ready_state: element.readyState,
        width: element.videoWidth,
        height: element.videoHeight,
        current_time: element.currentTime,
        paused: element.paused,
      }
    })
  }
  const before = Object.fromEntries(
    await Promise.all(
      cameras.map(
        async (camera, index) => [camera.camera_id, await sample(camera, index)] as const,
      ),
    ),
  ) as Record<string, VideoSample>
  await page.waitForTimeout(1_500)
  const after = Object.fromEntries(
    await Promise.all(
      cameras.map(
        async (camera, index) => [camera.camera_id, await sample(camera, index)] as const,
      ),
    ),
  ) as Record<string, VideoSample>
  for (const camera of cameras) {
    expect(after[camera.camera_id]?.ready_state).toBeGreaterThanOrEqual(2)
    expect(after[camera.camera_id]?.width).toBeGreaterThan(0)
    expect(after[camera.camera_id]?.height).toBeGreaterThan(0)
    expect(after[camera.camera_id]?.current_time).toBeGreaterThan(
      before[camera.camera_id]?.current_time ?? 0,
    )
  }
  await expect
    .poll(
      () =>
        detections.some(
          (item) =>
            item.camera_id === activeCamera.camera_id &&
            item.frame_age_seconds !== null &&
            item.frame_age_seconds <= 1.0 &&
            item.box_count > 0,
        ),
      {
        timeout: 20_000,
        message: `${activeCamera.name} should return fresh live detections with person boxes`,
      },
    )
    .toBe(true)
  return { before, after, detections, active_detection_camera_id: activeCamera.camera_id }
}

async function runRuntimeCropSearch(
  page: Page,
  activeCamera: RuntimeCamera,
  minimumLastSeenAfter?: string,
): Promise<RuntimeCropSearch> {
  await page
    .getByRole("navigation", { name: "Primary" })
    .getByRole("button", { name: "Person search", exact: true })
    .click()
  await expect(page.getByRole("heading", { name: "Person search" })).toBeVisible()
  const cameraFilters = page.locator("[data-search-camera-filters]")
  const activeCameraFilter = cameraFilters.getByLabel(activeCamera.name, { exact: true })
  await expect(activeCameraFilter).toBeVisible({ timeout: 30_000 })
  if (!(await activeCameraFilter.isChecked())) await activeCameraFilter.check()
  if (
    minimumLastSeenAfter !== undefined &&
    !Number.isFinite(new Date(minimumLastSeenAfter).getTime())
  ) {
    throw new Error("minimum searchable timestamp was invalid")
  }
  const fromDate = new Date()
  fromDate.setSeconds(0, 0)
  const toDate = new Date(fromDate.getTime() + 2 * 60_000)
  await page.getByLabel("From (local, inclusive)").fill(localDateTimeValue(fromDate))
  await page.getByLabel("To (local, inclusive)").fill(localDateTimeValue(toDate))
  const query = page.getByLabel("Describe a person")
  const textQueries = ["a person walking", "a person in casual clothing", "person"]
  let textResults: Array<Record<string, unknown>> = []
  let selectedTextQuery = ""
  for (const textQuery of textQueries) {
    await query.fill(textQuery)
    const responsePromise = page.waitForResponse((response) => isSearchMode(response, "text"), {
      timeout: 30_000,
    })
    await page.getByRole("button", { name: "Search", exact: true }).click()
    const response = await responsePromise
    expect(response.status()).toBe(200)
    const requestBodyText = response.request().postData()
    if (requestBodyText === null) throw new Error("text search request omitted its filters")
    const requestBody: unknown = JSON.parse(requestBodyText)
    if (!isRecord(requestBody)) throw new Error("text search request filters were malformed")
    expect(requestBody["camera_ids"]).toEqual([activeCamera.camera_id])
    expect(typeof requestBody["from"]).toBe("string")
    expect(typeof requestBody["to"]).toBe("string")
    const payload: unknown = await response.json()
    if (!isRecord(payload) || !Array.isArray(payload["results"])) {
      throw new Error("text search response omitted its appearance results")
    }
    textResults = payload["results"].filter(isRecord)
    selectedTextQuery = textQuery
    if (textResults.length > 0) break
    await page.waitForTimeout(1_000)
  }
  expect(textResults.length).toBeGreaterThan(0)
  const textResultCameraIds = resultCameraIds(textResults)
  const textMaxLastSeenAgeSeconds = maxLastSeenAgeSeconds(textResults)
  const textLatestLastSeenAt = latestLastSeenAt(textResults)
  expect(textResultCameraIds.every((cameraId) => cameraId === activeCamera.camera_id)).toBe(true)
  expect(textMaxLastSeenAgeSeconds).toBeLessThanOrEqual(65)
  if (minimumLastSeenAfter !== undefined) {
    expect(Date.parse(textLatestLastSeenAt)).toBeGreaterThan(Date.parse(minimumLastSeenAfter))
  }
  await expect(page.getByRole("heading", { name: "Results" })).toBeVisible()
  await expect(page.getByAltText(/Person crop from /).first()).toBeVisible({ timeout: 30_000 })
  const textScreenshot = await captureRuntimeScreenshot(page, "text-search")
  await page.getByRole("button", { name: "View details" }).first().click()
  await expect(page.getByRole("button", { name: "Find similar" })).toBeVisible()
  const similarResponsePromise = page.waitForResponse(
    (response) => isSearchMode(response, "similar"),
    { timeout: 30_000 },
  )
  await page.getByRole("button", { name: "Find similar" }).click()
  const similarResponse = await similarResponsePromise
  expect(similarResponse.status()).toBe(200)
  const similarRequestBodyText = similarResponse.request().postData()
  if (similarRequestBodyText === null) throw new Error("similar search request omitted its filters")
  const similarRequestBody: unknown = JSON.parse(similarRequestBodyText)
  if (!isRecord(similarRequestBody)) {
    throw new Error("similar search request filters were malformed")
  }
  expect(similarRequestBody["camera_ids"]).toEqual([activeCamera.camera_id])
  expect(typeof similarRequestBody["from"]).toBe("string")
  expect(typeof similarRequestBody["to"]).toBe("string")
  const similarPayload: unknown = await similarResponse.json()
  if (!isRecord(similarPayload) || !Array.isArray(similarPayload["results"])) {
    throw new Error("image-similarity search omitted its appearance results")
  }
  const similarResults = similarPayload["results"].filter(isRecord)
  expect(similarResults.length).toBeGreaterThan(0)
  const similarResultCameraIds = resultCameraIds(similarResults)
  const similarMaxLastSeenAgeSeconds = maxLastSeenAgeSeconds(similarResults)
  const similarLatestLastSeenAt = latestLastSeenAt(similarResults)
  expect(similarResultCameraIds.every((cameraId) => cameraId === activeCamera.camera_id)).toBe(true)
  expect(similarMaxLastSeenAgeSeconds).toBeLessThanOrEqual(65)
  if (minimumLastSeenAfter !== undefined) {
    expect(Date.parse(similarLatestLastSeenAt)).toBeGreaterThan(Date.parse(minimumLastSeenAfter))
  }
  await expect(page.getByRole("heading", { name: "Similar appearances" })).toBeVisible()
  const imageScreenshot = await captureRuntimeScreenshot(page, "image-search")
  return {
    active_camera_id: activeCamera.camera_id,
    minimum_last_seen_after: minimumLastSeenAfter ?? null,
    text_query: selectedTextQuery,
    text_result_count: textResults.length,
    text_result_camera_ids: textResultCameraIds,
    text_max_last_seen_age_seconds: textMaxLastSeenAgeSeconds,
    text_latest_last_seen_at: textLatestLastSeenAt,
    seed_appearance_id: textResults[0]?.["appearance_id"],
    similar_result_count: similarResults.length,
    similar_result_camera_ids: similarResultCameraIds,
    similar_max_last_seen_age_seconds: similarMaxLastSeenAgeSeconds,
    similar_latest_last_seen_at: similarLatestLastSeenAt,
    text_screenshot: textScreenshot,
    image_screenshot: imageScreenshot,
  }
}

async function captureRuntimeScreenshot(page: Page, name: string): Promise<string | null> {
  if (stateFile === undefined || stateFile === "") return null
  const statePath = resolve(stateFile)
  const stem = statePath.endsWith(".json") ? statePath.slice(0, -5) : statePath
  const destination = `${stem}-${name}.png`
  await mkdir(dirname(destination), { recursive: true })
  await page.screenshot({ path: destination })
  return destination
}

type PointerDragEvidence = Readonly<{
  initialRangeValue: number
  finalRangeValue: number
  screenshotPath: string | null
  diagnosticPath: string | null
}>

async function capturePreDragEvidence(
  page: Page,
  label: string,
  diagnostic: Record<string, unknown>,
): Promise<Readonly<{ screenshotPath: string | null; diagnosticPath: string | null }>> {
  if (stateFile === undefined || stateFile === "") {
    return { screenshotPath: null, diagnosticPath: null }
  }
  const statePath = resolve(stateFile)
  const stateStem = statePath.endsWith(".json") ? statePath.slice(0, -5) : statePath
  const labelSlug = label.toLowerCase().replace(/[^a-z0-9]+/g, "-")
  const sequence = String(++pointerDiagnosticSequence).padStart(2, "0")
  const screenshotPath = `${stateStem}-${labelSlug}-${sequence}-before-drag.png`
  const diagnosticPath = `${stateStem}-${labelSlug}-${sequence}-before-drag.json`
  await mkdir(dirname(screenshotPath), { recursive: true })
  await page.screenshot({ path: screenshotPath })
  await writeFile(diagnosticPath, `${JSON.stringify(diagnostic, null, 2)}\n`, { mode: 0o600 })
  return { screenshotPath, diagnosticPath }
}

async function dragSliderTo(
  page: Page,
  label: string,
  targetValue: number,
  onPointerReady: () => void,
  captureDiagnostics = true,
): Promise<PointerDragEvidence> {
  const slider = page.getByRole("slider", { name: `${label} slider` })
  await slider.scrollIntoViewIfNeeded({ timeout: POINTER_ACTION_TIMEOUT_MS })
  const box = await slider.boundingBox()
  if (box === null) {
    const evidence = captureDiagnostics
      ? await capturePreDragEvidence(page, label, {
          reason: "range input has no visible bounding box after scrollIntoViewIfNeeded",
        })
      : { screenshotPath: null, diagnosticPath: null }
    throw new Error(
      `${label} slider is not visible; diagnostic screenshot: ${evidence.screenshotPath ?? "unavailable"}`,
    )
  }
  const values = await slider.evaluate((element) => {
    if (!(element instanceof HTMLInputElement)) throw new Error(`${label} slider is not an input`)
    return {
      min: Number(element.min),
      max: Number(element.max),
      value: element.valueAsNumber,
    }
  })
  const inset = Math.min(8, box.width / 4)
  const trackWidth = box.width - 2 * inset
  const position = (value: number): number => {
    if (values.max <= values.min) throw new Error(`${label} slider has invalid bounds`)
    return box.x + inset + ((value - values.min) / (values.max - values.min)) * trackWidth
  }
  const y = box.y + box.height / 2
  const viewport = await page.evaluate(() => ({
    width: window.innerWidth,
    height: window.innerHeight,
  }))
  const pointerStart = { x: position(values.value), y }
  const pointerTarget = { x: position(targetValue), y }
  const isInViewport = (point: Readonly<{ x: number; y: number }>): boolean =>
    point.x >= 0 && point.x < viewport.width && point.y >= 0 && point.y < viewport.height
  const startHit = await slider.evaluate(
    (element, point) => document.elementFromPoint(point.x, point.y) === element,
    pointerStart,
  )
  const targetHit = await slider.evaluate(
    (element, point) => document.elementFromPoint(point.x, point.y) === element,
    pointerTarget,
  )
  const evidence = captureDiagnostics
    ? await capturePreDragEvidence(page, label, {
        viewport,
        bounding_box: box,
        initial_range_value: values.value,
        requested_range_value: targetValue,
        pointer_start: pointerStart,
        pointer_target: pointerTarget,
        pointer_start_hit_range: startHit,
        pointer_target_hit_range: targetHit,
      })
    : { screenshotPath: null, diagnosticPath: null }
  if (
    box.x < 0 ||
    box.y < 0 ||
    box.x + box.width > viewport.width ||
    box.y + box.height > viewport.height ||
    !isInViewport(pointerStart) ||
    !isInViewport(pointerTarget) ||
    !startHit ||
    !targetHit
  ) {
    throw new Error(
      `${label} slider pointer targets are outside the visible, hittable control; diagnostic screenshot: ${evidence.screenshotPath ?? "unavailable"}`,
    )
  }
  onPointerReady()
  await page.mouse.move(position(values.value), y)
  await page.mouse.down()
  await page.mouse.move(position(targetValue), y, { steps: 12 })
  await page.mouse.up()
  await expect
    .poll(() => slider.evaluate((element) => (element as HTMLInputElement).valueAsNumber), {
      timeout: 5_000,
      message: `${label} range value should change after pointer drag`,
    })
    .not.toBe(values.value)
  const finalRangeValue = await slider.evaluate(
    (element) => (element as HTMLInputElement).valueAsNumber,
  )
  return {
    initialRangeValue: values.value,
    finalRangeValue,
    screenshotPath: evidence.screenshotPath,
    diagnosticPath: evidence.diagnosticPath,
  }
}

type PreflightObservation = Readonly<{
  request: Promise<Readonly<{ request: Request }> | Readonly<{ error: unknown }>>
  response: Promise<Readonly<{ response: Response }> | Readonly<{ error: unknown }>>
}>

function observePreflight(page: Page): PreflightObservation {
  const request = page
    .waitForRequest(
      (candidate) =>
        candidate.method() === "POST" &&
        new URL(candidate.url()).pathname === "/api/training/preflight",
      { timeout: POINTER_ACTION_TIMEOUT_MS },
    )
    .then(
      (matchedRequest) => ({ request: matchedRequest }),
      (error: unknown) => ({ error }),
    )
  const response = request.then((result) => {
    if (!("request" in result)) return result
    return page
      .waitForResponse((candidate) => candidate.request() === result.request, {
        timeout: POINTER_ACTION_TIMEOUT_MS,
      })
      .then(
        (matchedResponse) => ({ response: matchedResponse }),
        (error: unknown) => ({ error }),
      )
  })
  return { request, response }
}

async function finishPreflightObservation(
  observation: PreflightObservation,
  label: string,
  screenshotPath: string | null,
): Promise<Record<string, unknown>> {
  const requestResult = await observation.request
  if (!("request" in requestResult)) {
    throw new Error(
      `${label} drag produced no preflight request within ${POINTER_ACTION_TIMEOUT_MS}ms: ${String(requestResult.error)}; diagnostic screenshot: ${screenshotPath ?? "unavailable"}`,
    )
  }
  const responseResult = await observation.response
  if (!("response" in responseResult)) {
    throw new Error(
      `${label} preflight produced no response within ${POINTER_ACTION_TIMEOUT_MS}ms: ${String(responseResult.error)}; diagnostic screenshot: ${screenshotPath ?? "unavailable"}`,
    )
  }
  const request = requestResult.request
  const response = responseResult.response
  expect(response.status()).toBe(200)
  const payload: unknown = request.postDataJSON()
  if (!isRecord(payload) || !isRecord(payload["config"])) {
    throw new Error("preflight request omitted the current training configuration")
  }
  return payload["config"]
}

type SliderPreflightEvidence = Readonly<{
  config: Record<string, unknown>
  displayedValue: string
  numericValue: number
  screenshotPath: string | null
  diagnosticPath: string | null
  outgoingConfigMatched: true
}>

function displayedConfigValue(label: string, value: number): string {
  return label === "Learning rate"
    ? value.toExponential(2)
    : value.toLocaleString(undefined, { maximumFractionDigits: 8 })
}

async function dragAndReadPreflight(
  page: Page,
  label: string,
  targetValue: number,
): Promise<SliderPreflightEvidence> {
  let observation: PreflightObservation | undefined
  const drag = await dragSliderTo(page, label, targetValue, () => {
    observation = observePreflight(page)
  })
  if (observation === undefined) throw new Error(`${label} preflight observer was not installed`)
  const config = await finishPreflightObservation(observation, label, drag.screenshotPath)
  const displayedValue = await page.getByRole("textbox", { name: label }).inputValue()
  const numericValue = Number(displayedValue)
  const fieldName = label === "Learning rate" ? "learning_rate" : "weight_decay"
  const requestValue = config[fieldName]
  if (typeof requestValue !== "number") {
    throw new Error(`${label} preflight request did not contain a numeric value`)
  }
  expect(displayedConfigValue(label, requestValue)).toBe(displayedValue)
  return {
    config,
    displayedValue,
    numericValue,
    screenshotPath: drag.screenshotPath,
    diagnosticPath: drag.diagnosticPath,
    outgoingConfigMatched: true,
  }
}

async function fillAndReadPreflight(
  page: Page,
  label: string,
  value: string,
): Promise<SliderPreflightEvidence> {
  const observation = observePreflight(page)
  const field = page.getByRole("textbox", { name: label })
  await field.fill(value)
  await field.press("Tab")
  const config = await finishPreflightObservation(observation, label, null)
  const displayedValue = await field.inputValue()
  const numericValue = Number(displayedValue)
  const fieldName = label === "Learning rate" ? "learning_rate" : "weight_decay"
  const requestValue = config[fieldName]
  if (typeof requestValue !== "number") {
    throw new Error(`${label} preflight request did not contain a numeric value`)
  }
  expect(displayedConfigValue(label, requestValue)).toBe(displayedValue)
  return {
    config,
    displayedValue,
    numericValue,
    screenshotPath: null,
    diagnosticPath: null,
    outgoingConfigMatched: true,
  }
}

async function verifyTrainingSliders(page: Page): Promise<Record<string, unknown>> {
  await expect(page.getByRole("button", { name: "Start training" })).toBeEnabled({
    timeout: 120_000,
  })
  const learningRate = page.getByRole("textbox", { name: "Learning rate" })
  const weightDecay = page.getByRole("textbox", { name: "Weight decay" })
  const learningRateSlider = page.getByRole("slider", { name: "Learning rate slider" })
  const weightDecaySlider = page.getByRole("slider", { name: "Weight decay slider" })
  const approvedLearningRateText = await learningRate.inputValue()
  const approvedWeightDecayText = await weightDecay.inputValue()
  const approvedLearningRate = Number(approvedLearningRateText)
  const approvedWeightDecay = Number(approvedWeightDecayText)
  const approvedLearningRatePosition = Number(
    await learningRateSlider.evaluate((element) => {
      if (!(element instanceof HTMLInputElement))
        throw new Error("learning-rate slider is not an input")
      return element.value
    }),
  )
  const approvedWeightDecayPosition = Number(
    await weightDecaySlider.evaluate((element) => {
      if (!(element instanceof HTMLInputElement))
        throw new Error("weight-decay slider is not an input")
      return element.value
    }),
  )
  const learningRateTarget = approvedLearningRatePosition === 680 ? 620 : 680
  const weightDecayTarget = approvedWeightDecay === 0.08 ? 0.04 : 0.08

  const changedLearningRateConfig = await dragAndReadPreflight(
    page,
    "Learning rate",
    learningRateTarget,
  )
  expect(changedLearningRateConfig.numericValue).not.toBe(approvedLearningRate)
  const learningRatePairedWeight = changedLearningRateConfig.config["weight_decay"]
  if (typeof learningRatePairedWeight !== "number") {
    throw new Error("learning-rate preflight omitted the current weight decay")
  }
  expect(displayedConfigValue("Weight decay", learningRatePairedWeight)).toBe(
    approvedWeightDecayText,
  )

  const changedWeightDecayConfig = await dragAndReadPreflight(
    page,
    "Weight decay",
    weightDecayTarget,
  )
  const changedLearningRate = changedLearningRateConfig.numericValue
  const changedWeightDecay = changedWeightDecayConfig.numericValue
  expect(changedWeightDecay).not.toBe(approvedWeightDecay)
  const weightDecayPairedLearningRate = changedWeightDecayConfig.config["learning_rate"]
  if (typeof weightDecayPairedLearningRate !== "number") {
    throw new Error("weight-decay preflight omitted the current learning rate")
  }
  expect(displayedConfigValue("Learning rate", weightDecayPairedLearningRate)).toBe(
    changedLearningRateConfig.displayedValue,
  )

  const restoredLearningRateConfig = await dragAndReadPreflight(
    page,
    "Learning rate",
    approvedLearningRatePosition,
  )
  const restoredLearningRate =
    restoredLearningRateConfig.numericValue === approvedLearningRate
      ? restoredLearningRateConfig
      : await fillAndReadPreflight(page, "Learning rate", approvedLearningRateText)
  expect(restoredLearningRate.numericValue).toBe(approvedLearningRate)

  const restoredConfig = await dragAndReadPreflight(
    page,
    "Weight decay",
    approvedWeightDecayPosition,
  )
  const restoredWeightDecay =
    restoredConfig.numericValue === approvedWeightDecay
      ? restoredConfig
      : await fillAndReadPreflight(page, "Weight decay", approvedWeightDecayText)
  expect(restoredWeightDecay.numericValue).toBe(approvedWeightDecay)
  const restoredPairedLearningRate = restoredWeightDecay.config["learning_rate"]
  if (typeof restoredPairedLearningRate !== "number") {
    throw new Error("restored weight-decay preflight omitted the learning rate")
  }
  expect(displayedConfigValue("Learning rate", restoredPairedLearningRate)).toBe(
    restoredLearningRate.displayedValue,
  )
  await expect(page.getByRole("button", { name: "Start training" })).toBeEnabled({
    timeout: 120_000,
  })

  return {
    learning_rate: {
      approved_default: approvedLearningRate,
      dragged_value: changedLearningRate,
      outgoing_config_matched: changedLearningRateConfig.outgoingConfigMatched,
      pre_drag_screenshot: changedLearningRateConfig.screenshotPath,
      pre_drag_diagnostic: changedLearningRateConfig.diagnosticPath,
      restored: restoredLearningRate.numericValue,
    },
    weight_decay: {
      approved_default: approvedWeightDecay,
      dragged_value: changedWeightDecay,
      outgoing_config_matched: changedWeightDecayConfig.outgoingConfigMatched,
      pre_drag_screenshot: changedWeightDecayConfig.screenshotPath,
      pre_drag_diagnostic: changedWeightDecayConfig.diagnosticPath,
      restored: restoredWeightDecay.numericValue,
    },
  }
}

async function setEpochsAndWaitForAdmission(page: Page, count: number): Promise<void> {
  const epochs = page.getByRole("textbox", { name: "Epochs" })
  await epochs.fill(String(count))
  await expect(page.getByRole("button", { name: "Start training" })).toBeEnabled({
    timeout: 120_000,
  })
}

async function startTraining(
  page: Page,
  epochs: number,
  readiness: DatasetStatus,
): Promise<string> {
  const preflightResponsePromise = page.waitForResponse((response) => {
    const request = response.request()
    if (
      request.method() !== "POST" ||
      new URL(response.url()).pathname !== "/api/training/preflight"
    ) {
      return false
    }
    const bodyText = request.postData()
    if (bodyText === null) return false
    try {
      const body: unknown = JSON.parse(bodyText)
      return isRecord(body) && isRecord(body["config"]) && body["config"]["epochs"] === epochs
    } catch {
      return false
    }
  })
  await setEpochsAndWaitForAdmission(page, epochs)
  const preflightResponse = await preflightResponsePromise
  expect(preflightResponse.status()).toBe(200)
  const preflightValue: unknown = await preflightResponse.json()
  if (
    !isRecord(preflightValue) ||
    typeof preflightValue["profile_identity"] !== "string" ||
    typeof preflightValue["required_bytes"] !== "number" ||
    typeof preflightValue["free_bytes"] !== "number"
  ) {
    throw new Error("training preflight omitted its admitted memory profile")
  }
  expect(preflightValue["admitted"]).toBe(true)
  const expectedProfileIdentity = process.env["GW_TRAINING_SMOKE_PROFILE_ID"]
  if (expectedProfileIdentity !== undefined) {
    expect(preflightValue["profile_identity"]).toBe(expectedProfileIdentity)
  }
  const responsePromise = page.waitForResponse((response) => {
    const request = response.request()
    return request.method() === "POST" && new URL(response.url()).pathname === "/api/training/jobs"
  })
  await page.getByRole("button", { name: "Start training" }).click()
  const response = await responsePromise
  expect(response.status()).toBe(202)
  const job: unknown = await response.json()
  if (!isRecord(job) || typeof job["id"] !== "string" || !isRecord(job["config"])) {
    throw new Error("training submit response omitted its job identity")
  }
  expect(job["config"]["epochs"]).toBe(epochs)
  await writeEvidence({
    kind: "job",
    job_id: job["id"],
    phase: job["phase"],
    config: job["config"],
    source_fingerprint: readiness.supervisor.source_fingerprint,
    dataset_fingerprint: readiness.snapshot?.fingerprint ?? null,
    memory_profile_identity: preflightValue["profile_identity"],
    required_bytes: preflightValue["required_bytes"],
    free_bytes: preflightValue["free_bytes"],
  })
  return job["id"]
}

async function openRun(page: Page, jobId: string): Promise<void> {
  await expect(
    page.getByRole("button", { name: new RegExp(`Run ${jobId.slice(0, 8)}`) }),
  ).toBeVisible({ timeout: 60_000 })
  await page.getByRole("button", { name: new RegExp(`Run ${jobId.slice(0, 8)}`) }).click()
  await expect(
    page.getByRole("heading", { name: new RegExp(`Run ${jobId.slice(0, 8)}`) }),
  ).toBeVisible()
}

async function waitForRunning(page: Page, jobId: string): Promise<void> {
  await expect(page.getByRole("button", { name: "Request cooperative cancel" })).toBeVisible({
    timeout: 180_000,
  })
  const runHeading = page.getByRole("heading", {
    name: new RegExp(`^Run ${jobId.slice(0, 8)}$`),
  })
  const runMain = runHeading.locator("xpath=ancestor::main[1]")
  await expect(runMain.getByRole("status").filter({ hasText: /^Training$/u })).toBeVisible({
    timeout: 180_000,
  })
}

type InterruptProgress = Readonly<{
  phase: string
  currentEpoch: number
  currentStep: number
  checkpointEpoch: number
  optimizerStep: number
  totalEpochs: number
  candidateModelId: string | null
}>

function hasCompleteCheckpointAndActiveWork(progress: InterruptProgress): boolean {
  return (
    progress.phase === "training" &&
    progress.checkpointEpoch >= 1 &&
    progress.optimizerStep >= 1 &&
    progress.currentStep >= 1 &&
    progress.currentEpoch >= progress.checkpointEpoch &&
    progress.currentEpoch < progress.totalEpochs
  )
}

async function readTrainingProgress(page: Page, jobId: string): Promise<InterruptProgress> {
  return page.evaluate(async (requestedJobId) => {
    const isRecord = (value: unknown): value is Record<string, unknown> =>
      typeof value === "object" && value !== null && !Array.isArray(value)
    const encodedId = encodeURIComponent(requestedJobId)
    const jobPath = `/api/training/jobs/${encodedId}`
    const jobResponse = await fetch(jobPath, { credentials: "same-origin" })
    const metricResponse = await fetch(`${jobPath}/metrics?limit=50`, {
      credentials: "same-origin",
    })
    if (!jobResponse.ok || !metricResponse.ok) {
      throw new Error("training progress endpoint could not be read")
    }
    const jobValue: unknown = await jobResponse.json()
    const metricValue: unknown = await metricResponse.json()
    if (!isRecord(jobValue) || !isRecord(jobValue["config"]) || !isRecord(metricValue)) {
      throw new Error("training progress response was malformed")
    }
    const phase = jobValue["phase"]
    const currentEpoch = jobValue["current_epoch"]
    const currentStep = jobValue["current_step"]
    const totalEpochs = jobValue["config"]["epochs"]
    const candidateModelId = jobValue["candidate_model_id"]
    const metricItems = metricValue["items"]
    if (
      typeof phase !== "string" ||
      typeof currentEpoch !== "number" ||
      typeof currentStep !== "number" ||
      typeof totalEpochs !== "number" ||
      (candidateModelId !== null &&
        candidateModelId !== undefined &&
        typeof candidateModelId !== "string") ||
      !Array.isArray(metricItems)
    ) {
      throw new Error("training progress response omitted epoch or step values")
    }
    let checkpointEpoch = 0
    let optimizerStep = 0
    for (const item of metricItems) {
      if (!isRecord(item)) continue
      const epoch = item["epoch"]
      const step = item["step"]
      if (typeof epoch === "number" && Number.isInteger(epoch)) {
        checkpointEpoch = Math.max(checkpointEpoch, epoch)
      }
      if (typeof step === "number" && Number.isInteger(step)) {
        optimizerStep = Math.max(optimizerStep, step)
      }
    }
    return {
      phase,
      currentEpoch,
      currentStep,
      checkpointEpoch,
      optimizerStep,
      totalEpochs,
      candidateModelId: typeof candidateModelId === "string" ? candidateModelId : null,
    }
  }, jobId)
}

async function waitForCompleteCheckpointAndActiveWork(
  page: Page,
  jobId: string,
): Promise<InterruptProgress> {
  let observedProgress: InterruptProgress | undefined
  const jobPath = `/api/training/jobs/${jobId}`
  const metricsPath = `${jobPath}/metrics`
  let checkpointEpoch = 0
  let optimizerStep = 0
  page.on("response", async (response) => {
    if (response.status() !== 200 || response.request().method() !== "GET") return
    const pathname = new URL(response.url()).pathname
    try {
      const payload: unknown = await response.json()
      if (pathname === jobPath && isRecord(payload) && isRecord(payload["config"])) {
        const phaseValue = payload["phase"]
        const currentEpochValue = payload["current_epoch"]
        const currentStepValue = payload["current_step"]
        const totalEpochsValue = payload["config"]["epochs"]
        const candidateModelId = payload["candidate_model_id"]
        if (
          typeof phaseValue === "string" &&
          typeof currentEpochValue === "number" &&
          Number.isInteger(currentEpochValue) &&
          typeof currentStepValue === "number" &&
          Number.isInteger(currentStepValue) &&
          (candidateModelId === null ||
            candidateModelId === undefined ||
            typeof candidateModelId === "string") &&
          typeof totalEpochsValue === "number" &&
          Number.isInteger(totalEpochsValue)
        ) {
          observedProgress = {
            phase: phaseValue,
            currentEpoch: currentEpochValue,
            currentStep: currentStepValue,
            checkpointEpoch,
            optimizerStep,
            totalEpochs: totalEpochsValue,
            candidateModelId: typeof candidateModelId === "string" ? candidateModelId : null,
          }
        }
      } else if (pathname === metricsPath && isRecord(payload) && Array.isArray(payload["items"])) {
        for (const item of payload["items"]) {
          if (
            isRecord(item) &&
            typeof item["epoch"] === "number" &&
            Number.isInteger(item["epoch"])
          ) {
            checkpointEpoch = Math.max(checkpointEpoch, item["epoch"])
          }
          if (
            isRecord(item) &&
            typeof item["step"] === "number" &&
            Number.isInteger(item["step"])
          ) {
            optimizerStep = Math.max(optimizerStep, item["step"])
          }
        }
        if (observedProgress !== undefined) {
          observedProgress = { ...observedProgress, checkpointEpoch, optimizerStep }
        }
      }
    } catch {
      // A later successful polling response supplies the job and metric evidence.
    }
  })

  await expect(
    page.locator(".training-chart__labels").getByText("E1", { exact: true }).first(),
  ).toBeVisible({
    timeout: 10 * 60_000,
  })
  await expect
    .poll(
      () => observedProgress !== undefined && hasCompleteCheckpointAndActiveWork(observedProgress),
      {
        timeout: 10 * 60_000,
        message: "wait for an epoch checkpoint while another epoch is still training",
      },
    )
    .toBe(true)
  if (observedProgress === undefined || !hasCompleteCheckpointAndActiveWork(observedProgress)) {
    throw new Error("training checkpoint evidence was not observed")
  }
  return observedProgress
}

async function cancelAndWait(page: Page): Promise<void> {
  await page.getByRole("button", { name: "Request cooperative cancel" }).click()
  await expect(page.getByText("Cancellation requested")).toBeVisible()
  await expect(page.getByText("Cancelled", { exact: true })).toBeVisible({ timeout: 120_000 })
}

async function writeEvidence(payload: Record<string, unknown>): Promise<void> {
  if (stateFile === undefined || stateFile === "") return
  const destination = resolve(stateFile)
  await mkdir(dirname(destination), { recursive: true })
  await writeFile(destination, `${JSON.stringify(payload, null, 2)}\n`, { mode: 0o600 })
}

test.describe("training against the real deployment", () => {
  test.skip(
    phase === "" || phase === "camera" || password === undefined,
    "set GW_TRAINING_SMOKE_PHASE and GW_E2E_OPERATOR_PASSWORD to run this real-stack scenario",
  )

  test(`real browser workflow: ${phase}`, async ({ page }) => {
    test.setTimeout(30 * 60 * 1_000)
    const browserErrors: string[] = []
    page.on("pageerror", (error) => browserErrors.push(error.message))
    await signIn(page)
    await openTraining(page)

    if (phase === "validate") {
      const status = await waitForReadiness(page)
      expect(status.snapshot).not.toBeNull()
      expect(status.supervisor.state).toBe("ready")
      const sliderEvidence = await verifyTrainingSliders(page)
      const snapshot = status.snapshot
      if (snapshot === null)
        throw new Error("ready training dataset omitted its validated snapshot")
      await writeEvidence({
        kind: "readiness",
        dataset_id: snapshot.dataset_id,
        fingerprint: snapshot.fingerprint,
        image_count: snapshot.image_count,
        caption_count: snapshot.caption_count,
        identity_count: snapshot.identity_count,
        split_counts: snapshot.split_counts,
        supervisor: status.supervisor,
        slider_pointer_interaction: sliderEvidence,
      })
    } else if (phase === "shortage") {
      await waitForReadiness(page)
      const refreshButton = page.getByRole("button", { name: "Refresh estimate" })
      await expect(refreshButton).toBeEnabled({ timeout: 30_000 })
      const preflightStartedAt = Date.now()
      const preflightResponsePromise = page.waitForResponse(
        (response) =>
          response.request().method() === "POST" &&
          new URL(response.url()).pathname === "/api/training/preflight",
        { timeout: 30_000 },
      )
      const [preflightResponse] = await Promise.all([
        preflightResponsePromise,
        refreshButton.click(),
      ])
      expect(preflightResponse.status()).toBe(200)
      const preflightRoundTripMs = Date.now() - preflightStartedAt
      const preflightValue: unknown = await preflightResponse.json()
      if (!isRecord(preflightValue) || preflightValue["admitted"] !== true) {
        throw new Error("GPU shortage proof requires a currently admitted calibrated profile")
      }
      const requiredBytes = preflightValue["required_bytes"]
      const freeBytes = preflightValue["free_bytes"]
      const profileIdentity = preflightValue["profile_identity"]
      if (
        typeof requiredBytes !== "number" ||
        typeof freeBytes !== "number" ||
        typeof profileIdentity !== "string"
      ) {
        throw new Error("admitted memory preflight omitted exact profile telemetry")
      }
      const startButton = page.getByRole("button", { name: "Start training" })
      await expect(startButton).toBeEnabled({ timeout: 30_000 })
      const jobIdsBefore = await listTrainingJobIds(page)
      const continueFile = process.env["GW_TRAINING_SMOKE_CONTINUE_FILE"]
      const pressureStateFile = process.env["GW_TRAINING_SMOKE_PRESSURE_STATE_FILE"]
      if (
        stateFile === undefined ||
        continueFile === undefined ||
        continueFile === "" ||
        pressureStateFile === undefined ||
        pressureStateFile === ""
      ) {
        throw new Error("shortage phase requires its evidence and controller signal files")
      }
      await writeEvidence({
        kind: "memory-pressure-ready",
        preflight: preflightValue,
        preflight_round_trip_ms: preflightRoundTripMs,
        job_ids_before: jobIdsBefore,
      })
      await waitForFile(resolve(continueFile), "CUDA pressure controller signal")
      const pressureValue: unknown = JSON.parse(await readFile(resolve(pressureStateFile), "utf-8"))
      if (!isRecord(pressureValue) || typeof pressureValue["free_bytes_after"] !== "number") {
        throw new Error("CUDA pressure evidence omitted its post-allocation free-memory reading")
      }
      expect(pressureValue["free_bytes_after"]).toBeLessThan(requiredBytes)
      await expect(startButton).toBeEnabled()
      const submitStartedAt = Date.now()
      const submitResponsePromise = page.waitForResponse(
        (response) =>
          response.request().method() === "POST" &&
          new URL(response.url()).pathname === "/api/training/jobs",
        { timeout: 30_000 },
      )
      const [submitResponse] = await Promise.all([submitResponsePromise, startButton.click()])
      expect(submitResponse.status()).toBe(409)
      const submitRoundTripMs = Date.now() - submitStartedAt
      const refusalPayload: unknown = await submitResponse.json()
      if (!isRecord(refusalPayload) || !isRecord(refusalPayload["detail"])) {
        throw new Error("memory refusal omitted its structured detail")
      }
      const refusal = refusalPayload["detail"]
      expect(refusal["code"]).toBe("training_memory_refused")
      expect(refusal["reason"]).toBe("insufficient_free_memory")
      expect(refusal["required_bytes"]).toBe(requiredBytes)
      expect(refusal["profile_identity"]).toBe(profileIdentity)
      expect(typeof refusal["free_bytes"]).toBe("number")
      expect(refusal["free_bytes"]).toBeLessThan(requiredBytes)
      const jobIdsAfter = await listTrainingJobIds(page)
      expect(jobIdsAfter).toEqual(jobIdsBefore)
      const refusalDialog = page.getByRole("dialog")
      await expect(refusalDialog).toBeVisible()
      await expect(
        refusalDialog.getByText("GPU memory is currently insufficient", { exact: true }),
      ).toBeVisible()
      await expect(
        refusalDialog.getByText(
          "The refusal applies to the immutable settings sent with this submit request.",
        ),
      ).toBeVisible()
      const refusalScreenshot = await captureRuntimeScreenshot(page, "gpu-shortage-popup")
      await writeEvidence({
        kind: "memory-shortage-refused",
        preflight: preflightValue,
        preflight_round_trip_ms: preflightRoundTripMs,
        pressure: pressureValue,
        submit_round_trip_ms: submitRoundTripMs,
        refusal,
        job_ids_before: jobIdsBefore,
        job_ids_after: jobIdsAfter,
        popup_screenshot: refusalScreenshot,
      })
      await refusalDialog.getByRole("button", { name: "Close details" }).click()
      await openCameraSettings(page)
      const cameras: RuntimeCamera[] = []
      for (const index of [1, 2, 3, 4]) cameras.push(await registerRuntimeCamera(page, index))
      const activeCamera = activeQACamera(cameras)
      await prepareRuntimeWall(page, cameras)
      const playback = await sampleRuntimeVideos(page, cameras, activeCamera)
      const wallScreenshot = await captureRuntimeScreenshot(page, "wall-live-under-shortage")
      const search = await runRuntimeCropSearch(page, activeCamera)
      await writeEvidence({
        kind: "memory-shortage-with-camera-search",
        preflight: preflightValue,
        preflight_round_trip_ms: preflightRoundTripMs,
        pressure: pressureValue,
        submit_round_trip_ms: submitRoundTripMs,
        refusal,
        job_ids_before: jobIdsBefore,
        job_ids_after: jobIdsAfter,
        playback,
        search,
        popup_screenshot: refusalScreenshot,
        wall_screenshot: wallScreenshot,
      })
    } else if (phase === "cancel" || phase === "interrupt" || phase === "finish") {
      const readiness = await waitForReadiness(page)
      const minimumLastSeenAfter = process.env["GW_TRAINING_SMOKE_MIN_LAST_SEEN_AFTER"]
      if (phase === "interrupt" && minimumLastSeenAfter === undefined) {
        throw new Error("interrupt phase requires the pre-training searchable baseline")
      }
      const jobId = await startTraining(page, phase === "interrupt" ? 5 : 1, readiness)
      if (phase === "cancel") {
        await waitForRunning(page, jobId)
        await cancelAndWait(page)
        await writeEvidence({ kind: "cancelled", job_id: jobId })
      } else if (phase === "interrupt") {
        const progress = await waitForCompleteCheckpointAndActiveWork(page, jobId)
        await waitForRunning(page, jobId)
        const cameras = EXISTING_RUNTIME_CAMERAS
        const activeCamera = activeQACamera(cameras)
        await prepareRuntimeWall(page, cameras)
        const playbackDuringTraining = await sampleRuntimeVideos(page, cameras, activeCamera)
        const wallScreenshotDuringTraining = await captureRuntimeScreenshot(
          page,
          "wall-live-during-training",
        )
        const searchDuringTraining = await runRuntimeCropSearch(
          page,
          activeCamera,
          minimumLastSeenAfter,
        )
        await openTraining(page)
        await openRun(page, jobId)
        await waitForRunning(page, jobId)
        let progressAfterCamera = await readTrainingProgress(page, jobId)
        await expect
          .poll(
            async () => {
              progressAfterCamera = await readTrainingProgress(page, jobId)
              return (
                progressAfterCamera.phase === "training" &&
                progressAfterCamera.currentEpoch < progressAfterCamera.totalEpochs &&
                (progressAfterCamera.currentEpoch > progress.currentEpoch ||
                  progressAfterCamera.currentStep > progress.currentStep ||
                  progressAfterCamera.optimizerStep > progress.optimizerStep)
              )
            },
            {
              timeout: 120_000,
              message: "training should keep updating while camera playback and search run",
            },
          )
          .toBe(true)
        await writeEvidence({
          kind: "interrupt-me",
          job_id: jobId,
          phase: progress.phase,
          current_epoch: progress.currentEpoch,
          current_step: progress.currentStep,
          checkpoint_epoch: progress.checkpointEpoch,
          optimizer_step: progress.optimizerStep,
          total_epochs: progress.totalEpochs,
          minimum_last_seen_after: minimumLastSeenAfter,
          camera_ids: cameras.map((camera) => camera.camera_id),
          playback_during_training: playbackDuringTraining,
          search_during_training: searchDuringTraining,
          wall_screenshot_during_training: wallScreenshotDuringTraining,
          progress_after_camera_search: progressAfterCamera,
        })
      } else {
        await expect(page.getByRole("heading", { name: "Final held-out comparison" })).toBeVisible({
          timeout: 30 * 60_000,
        })
        await expect(page.getByText("Manual model application remains blocked")).toBeVisible()
        await writeEvidence({ kind: "succeeded", job_id: jobId })
      }
    } else if (phase === "coexist") {
      if (requestedJobId === undefined) throw new Error("coexist phase requires a training job ID")
      const minimumLastSeenAfter = process.env["GW_TRAINING_SMOKE_MIN_LAST_SEEN_AFTER"]
      if (minimumLastSeenAfter === undefined) {
        throw new Error("coexist phase requires the pre-training searchable baseline")
      }
      const progressBefore = await readTrainingProgress(page, requestedJobId)
      expect(progressBefore.phase).toBe("training")
      expect(progressBefore.currentEpoch).toBeLessThan(progressBefore.totalEpochs)
      await openRun(page, requestedJobId)
      await waitForRunning(page, requestedJobId)
      const cameras = EXISTING_RUNTIME_CAMERAS
      const activeCamera = activeQACamera(cameras)
      await prepareRuntimeWall(page, cameras)
      const playbackDuringTraining = await sampleRuntimeVideos(page, cameras, activeCamera)
      const searchDuringTraining = await runRuntimeCropSearch(
        page,
        activeCamera,
        minimumLastSeenAfter,
      )
      await openTraining(page)
      await openRun(page, requestedJobId)
      await waitForRunning(page, requestedJobId)
      const progressAfter = await readTrainingProgress(page, requestedJobId)
      expect(progressAfter.phase).toBe("training")
      expect(progressAfter.currentEpoch).toBeLessThan(progressAfter.totalEpochs)
      expect(
        progressAfter.currentStep > progressBefore.currentStep ||
          progressAfter.currentEpoch > progressBefore.currentEpoch,
      ).toBe(true)
      await writeEvidence({
        kind: "camera-search-during-training",
        job_id: requestedJobId,
        minimum_last_seen_after: minimumLastSeenAfter,
        progress_before: progressBefore,
        progress_after: progressAfter,
        camera_ids: cameras.map((camera) => camera.camera_id),
        active_detection_camera_id: activeCamera.camera_id,
        playback_during_training: playbackDuringTraining,
        search_during_training: searchDuringTraining,
      })
    } else if (phase === "resume") {
      await waitForReadiness(page)
      if (requestedJobId === undefined) throw new Error("resume phase requires a training job ID")
      await openRun(page, requestedJobId)
      await expect(page.getByText("Interrupted · manual resume available")).toBeVisible({
        timeout: 120_000,
      })
      const responsePromise = page.waitForResponse((response) => {
        const request = response.request()
        return (
          request.method() === "POST" &&
          new URL(response.url()).pathname.endsWith(`/jobs/${requestedJobId}/resume`)
        )
      })
      await page.getByRole("button", { name: "Resume from complete checkpoint" }).click()
      const response = await responsePromise
      expect(response.status()).toBe(202)
      let resumedJob = await readTrainingProgress(page, requestedJobId)
      await expect
        .poll(
          async () => {
            resumedJob = await readTrainingProgress(page, requestedJobId)
            return resumedJob.phase === "succeeded" && resumedJob.candidateModelId !== null
          },
          {
            timeout: 30 * 60_000,
            message: "manual resume should finish evaluation and publish its candidate",
          },
        )
        .toBe(true)
      await expect(page.getByRole("heading", { name: "Final held-out comparison" })).toBeVisible()
      await expect(page.getByText("Manual model application remains blocked")).toBeVisible()
      await writeEvidence({
        kind: "resumed-succeeded",
        job_id: requestedJobId,
        phase: resumedJob.phase,
        candidate_model_id: resumedJob.candidateModelId,
        final_epoch: resumedJob.currentEpoch,
        final_step: resumedJob.currentStep,
      })
    } else if (phase === "candidate") {
      await waitForReadiness(page)
      if (requestedJobId === undefined)
        throw new Error("candidate phase requires a training job ID")
      await openRun(page, requestedJobId)
      let candidateJob = await readTrainingProgress(page, requestedJobId)
      await expect
        .poll(
          async () => {
            candidateJob = await readTrainingProgress(page, requestedJobId)
            return candidateJob.phase === "succeeded" && candidateJob.candidateModelId !== null
          },
          {
            timeout: 120_000,
            message: "candidate review requires a durably succeeded training run",
          },
        )
        .toBe(true)
      await expect(page.getByRole("heading", { name: "Final held-out comparison" })).toBeVisible()
      await expect(page.getByText("Manual model application remains blocked")).toBeVisible()
      await page
        .getByRole("button", { name: "Review candidate in the existing model selector" })
        .click()
      await expect(page.getByRole("heading", { name: "Person search model" })).toBeVisible()
      const candidate = page.getByRole("radio", {
        name: new RegExp(`CUHK-PEDES CLIP ${requestedJobId.slice(0, 8)}`),
      })
      await expect(candidate).toBeVisible()
      await expect(candidate).toBeDisabled()
      await expect(page.getByRole("radio", { name: /OpenAI CLIP ViT-B\/16/u })).toBeChecked()
      await expect(page.getByRole("button", { name: /apply/i })).toBeDisabled()
      await writeEvidence({
        kind: "candidate-quality-blocked",
        job_id: requestedJobId,
        candidate_model_id: candidateJob.candidateModelId,
        active_model_id: "openai/clip-vit-base-patch16",
        candidate_prepared: false,
        apply_enabled: false,
      })
    } else {
      throw new Error(`unsupported training smoke phase: ${phase}`)
    }

    expect(browserErrors).toEqual([])
  })
})

test("real camera registration, decoded playback, and text/image search", async ({ page }) => {
  test.skip(
    phase !== "camera" || password === undefined,
    "set GW_TRAINING_SMOKE_PHASE=camera and the operator password for actual camera QA",
  )
  test.setTimeout(45 * 60 * 1_000)
  const browserErrors: string[] = []
  page.on("pageerror", (error) => browserErrors.push(error.message))
  await signIn(page)
  await openCameraSettings(page)
  const cameras: RuntimeCamera[] = []
  for (const index of [1, 2, 3, 4]) cameras.push(await registerRuntimeCamera(page, index))
  const activeCamera = activeQACamera(cameras)
  await prepareRuntimeWall(page, cameras)
  const playback = await sampleRuntimeVideos(page, cameras, activeCamera)
  const wallScreenshot = await captureRuntimeScreenshot(page, "wall-live")
  const search = await runRuntimeCropSearch(page, activeCamera)
  expect(search.text_result_camera_ids.length).toBeGreaterThan(0)
  expect(
    search.text_result_camera_ids.every((cameraId) => cameraId === activeCamera.camera_id),
  ).toBe(true)
  expect(search.text_max_last_seen_age_seconds).toBeLessThanOrEqual(60)
  expect(search.similar_result_camera_ids.length).toBeGreaterThan(0)
  expect(
    search.similar_result_camera_ids.every((cameraId) => cameraId === activeCamera.camera_id),
  ).toBe(true)
  expect(search.similar_max_last_seen_age_seconds).toBeLessThanOrEqual(60)
  await writeEvidence({
    kind: "camera-runtime-baseline",
    cameras: cameras.map((camera) => ({ camera_id: camera.camera_id, name: camera.name })),
    active_detection_camera_id: activeCamera.camera_id,
    playback,
    search,
    wall_screenshot: wallScreenshot,
  })
  expect(browserErrors).toEqual([])
})
