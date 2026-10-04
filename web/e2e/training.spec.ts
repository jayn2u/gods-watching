import { mkdir } from "node:fs/promises"
import { join, resolve } from "node:path"
import { expect, type Page, type Route, test } from "@playwright/test"

const SESSION = {
  authenticated: true,
  idle_expires_at: new Date(Date.now() + 30 * 60 * 1_000).toISOString(),
  absolute_expires_at: new Date(Date.now() + 8 * 60 * 60 * 1_000).toISOString(),
}

const CONFIG = {
  epochs: 30,
  learning_rate: 1e-5,
  micro_batch_size: 16,
  weight_decay: 0.01,
  gradient_accumulation: 4,
  warmup_ratio: 0.05,
  seed: 42,
  early_stopping_patience: 5,
  gradient_clipping_norm: 1,
  mixed_precision: "fp16",
  gradient_checkpointing: true,
}

const HASH = "a".repeat(64)
const OBSERVED_AT = "2026-10-04T12:30:00Z"
const DATASET = {
  dataset_id: "cuhk-pedes",
  fingerprint: HASH,
  protocol: "cuhk-pedes-original-splits-v1",
  split_counts: {
    train: { images: 2, captions: 4, identities: 2 },
    val: { images: 1, captions: 2, identities: 1 },
    test: { images: 1, captions: 2, identities: 1 },
  },
  image_count: 4,
  caption_count: 8,
  identity_count: 4,
}

const EVALUATION = {
  dataset_sha256: HASH,
  dataset_split: "test",
  protocol: "cuhk-pedes-original-splits-v1",
  baseline_model_id: "openai/clip-vit-base-patch16",
  baseline_revision: "local-b16-revision",
  baseline_package_sha256: HASH,
  training_source_fingerprint: HASH,
  evaluation_code_revision: HASH,
  metric_definition: "text-to-image-macro-recall-v1",
  best_validation_epoch: 2,
  baseline: { recall_at_1: 0.25, recall_at_5: 0.5, recall_at_10: 0.75 },
  candidate: { recall_at_1: 0.5, recall_at_5: 0.75, recall_at_10: 1 },
  candidate_weights_sha256: HASH,
  package_sha256: HASH,
}

function trainingJob(phase: string, overrides: Readonly<Record<string, unknown>> = {}) {
  return {
    id: "8b3f3a95-1dbe-4d4f-97c7-1e4a72b98111",
    request_id: "27e4e6c0-7a30-414e-b1e1-2e27c3678873",
    phase,
    config: CONFIG,
    dataset: DATASET,
    current_epoch: phase === "interrupted" || phase === "training" ? 2 : 0,
    current_step: 12,
    owner_generation: 1,
    cancel_requested: phase === "cancelling",
    attempts: phase === "interrupted" ? 1 : 0,
    best_metric: 0.5,
    candidate_model_id: phase === "succeeded" ? "local/cuhk-pedes-candidate" : null,
    candidate_revision: phase === "succeeded" ? HASH : null,
    evaluation: phase === "succeeded" ? EVALUATION : null,
    error: null,
    created_at: "2026-10-04T12:00:00Z",
    updated_at: OBSERVED_AT,
    finished_at: phase === "succeeded" || phase === "interrupted" ? OBSERVED_AT : null,
    ...overrides,
  }
}

function preflight(freeBytes = 10 * 1024 ** 3, admitted = true) {
  return {
    admitted,
    training_peak_bytes: 3 * 1024 ** 3,
    reserve_bytes: 2 * 1024 ** 3,
    required_bytes: 5 * 1024 ** 3,
    free_bytes: freeBytes,
    profile_identity: HASH,
    observed_at: OBSERVED_AT,
    reason: admitted ? "admitted" : "insufficient_free_memory",
  }
}

type TrainingApiOptions = Readonly<{
  initialJobs?: readonly Record<string, unknown>[]
  datasetStatus?: Record<string, unknown>
  onPreflight?: (route: Route, body: unknown) => Promise<void>
  onSubmit?: (route: Route, body: unknown) => Promise<void>
  onCancel?: (route: Route, body: unknown) => Promise<void>
  onResume?: (route: Route, body: unknown) => Promise<void>
  onHistoryPage?: (route: Route, cursor: string | null) => Promise<void>
  onMetricsPage?: (route: Route, cursor: string | null) => Promise<void>
  onLogsPage?: (route: Route, cursor: string | null) => Promise<void>
  onGetJob?: (route: Route, jobId: string, job: unknown) => Promise<void>
}>

async function installTrainingApi(page: Page, options: TrainingApiOptions = {}) {
  const jobs = [...(options.initialJobs ?? [])]
  const preflightBodies: Array<Record<string, unknown>> = []
  const submitBodies: Array<Record<string, unknown>> = []
  const cancelBodies: Array<Record<string, unknown>> = []
  const resumeBodies: Array<Record<string, unknown>> = []
  let currentJob = trainingJob("starting")

  await page.route("**/api/session", async (route) => {
    if (route.request().method() === "DELETE") {
      await route.fulfill({ status: 204, body: "" })
      return
    }
    await route.fulfill({ contentType: "application/json", body: JSON.stringify(SESSION) })
  })
  await page.route("**/api/session/activity", async (route) => {
    await route.fulfill({ contentType: "application/json", body: JSON.stringify(SESSION) })
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
  await page.route("**/api/settings/models", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        active_model_id: "openai/clip-vit-base-patch16",
        maintenance: false,
        models: [
          {
            model_id: "openai/clip-vit-base-patch16",
            display_name: "OpenAI CLIP ViT-B/16",
            revision: "base-revision",
            quality_passed: true,
            quality_reason: null,
            dimension: 512,
            prepared: true,
            reason: null,
          },
          {
            model_id: "local/cuhk-pedes-candidate",
            display_name: "CUHK-PEDES candidate",
            revision: HASH,
            quality_passed: false,
            quality_reason: "product_crop_evidence_missing",
            dimension: 512,
            prepared: false,
            reason: "Model assets are not prepared. Run prepare.",
          },
        ],
        transition: null,
      }),
    })
  })

  const validDataset = {
    registered: true,
    valid: true,
    reason: null,
    snapshot: DATASET,
  }
  await page.route("**/api/training/**", async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const pathname = url.pathname
    const method = request.method()
    const body: unknown = request.postDataJSON()

    if (pathname === "/api/training/config" && method === "GET") {
      await route.fulfill({ contentType: "application/json", body: JSON.stringify(CONFIG) })
      return
    }
    if (pathname === "/api/training/datasets" && method === "GET") {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(options.datasetStatus ?? validDataset),
      })
      return
    }
    if (pathname === "/api/training/preflight" && method === "POST") {
      if (isRecord(body)) preflightBodies.push(body)
      if (options.onPreflight !== undefined) {
        await options.onPreflight(route, body)
      } else {
        await route.fulfill({ contentType: "application/json", body: JSON.stringify(preflight()) })
      }
      return
    }
    if (pathname === "/api/training/jobs" && method === "GET") {
      if (options.onHistoryPage !== undefined) {
        await options.onHistoryPage(route, url.searchParams.get("cursor"))
        return
      }
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({ items: jobs, next_cursor: null }),
      })
      return
    }
    if (pathname === "/api/training/jobs" && method === "POST") {
      if (isRecord(body)) submitBodies.push(body)
      if (options.onSubmit !== undefined) {
        await options.onSubmit(route, body)
      } else {
        currentJob = trainingJob("starting", {
          config: isRecord(body) && isRecord(body["config"]) ? body["config"] : CONFIG,
          request_id:
            isRecord(body) && typeof body["request_id"] === "string"
              ? body["request_id"]
              : currentJob.request_id,
        })
        jobs.unshift(currentJob)
        await route.fulfill({
          status: 202,
          contentType: "application/json",
          body: JSON.stringify(currentJob),
        })
      }
      return
    }
    if (pathname.endsWith("/metrics") && method === "GET") {
      if (options.onMetricsPage !== undefined) {
        await options.onMetricsPage(route, url.searchParams.get("cursor"))
        return
      }
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          items: [
            {
              epoch: 1,
              step: 6,
              training_loss: 0.8,
              validation_recall_at_1: 0.25,
              allocated_bytes: 1_000,
              reserved_bytes: 2_000,
              observed_at: OBSERVED_AT,
            },
            {
              epoch: 2,
              step: 12,
              training_loss: 0.4,
              validation_recall_at_1: 0.5,
              allocated_bytes: 1_200,
              reserved_bytes: 2_100,
              observed_at: OBSERVED_AT,
            },
          ],
          next_cursor: null,
        }),
      })
      return
    }
    if (pathname.endsWith("/logs") && method === "GET") {
      if (options.onLogsPage !== undefined) {
        await options.onLogsPage(route, url.searchParams.get("cursor"))
        return
      }
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          items: [
            {
              cursor: "log-1",
              level: "info",
              message: "epoch 2 validation finished",
              observed_at: OBSERVED_AT,
            },
          ],
          next_cursor: null,
        }),
      })
      return
    }
    const actionMatch = pathname.match(/\/api\/training\/jobs\/([^/]+)\/(cancel|resume)$/)
    if (actionMatch !== null && method === "POST") {
      const action = actionMatch[2]
      if (isRecord(body)) (action === "cancel" ? cancelBodies : resumeBodies).push(body)
      if (action === "cancel" && options.onCancel !== undefined) {
        await options.onCancel(route, body)
        return
      }
      if (action === "resume" && options.onResume !== undefined) {
        await options.onResume(route, body)
        return
      }
      currentJob = trainingJob(action === "cancel" ? "cancelling" : "training", {
        attempts: action === "resume" ? 2 : currentJob.attempts,
        current_epoch: action === "resume" ? 2 : currentJob.current_epoch,
      })
      const existingIndex = jobs.findIndex((item) => isRecord(item) && item["id"] === currentJob.id)
      if (existingIndex >= 0) jobs[existingIndex] = currentJob
      await route.fulfill({
        status: 202,
        contentType: "application/json",
        body: JSON.stringify(currentJob),
      })
      return
    }
    if (/\/api\/training\/jobs\/[0-9a-f-]+$/.test(pathname) && method === "GET") {
      const matchId = pathname.split("/").at(-1)
      const job = jobs.find((item) => isRecord(item) && item["id"] === matchId) ?? currentJob
      if (options.onGetJob !== undefined) {
        await options.onGetJob(route, matchId ?? "", job)
        return
      }
      await route.fulfill({ contentType: "application/json", body: JSON.stringify(job) })
      return
    }
    await route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({ detail: { code: "not_found", message: "not found" } }),
    })
  })

  return { preflightBodies, submitBodies, cancelBodies, resumeBodies, jobs }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

async function openTraining(page: Page): Promise<void> {
  await page.goto("/")
  await page.getByRole("button", { name: "Training", exact: true }).click()
  await expect(page.getByRole("heading", { name: "CLIP training" })).toBeVisible()
}

test.describe("training operator screen with mocked API contracts", () => {
  test("syncs the learning-rate number with low, default, and high logarithmic slider values", async ({
    page,
  }) => {
    const api = await installTrainingApi(page)
    await openTraining(page)
    const slider = page.getByRole("slider", { name: "Learning rate slider" })
    const number = page.getByRole("textbox", { name: "Learning rate" })
    await expect(slider).toHaveValue("500")
    await expect(number).toHaveValue("1.00e-5")
    await expect
      .poll(() =>
        api.preflightBodies.some(
          (body) => isRecord(body["config"]) && body["config"]["learning_rate"] === 1e-5,
        ),
      )
      .toBe(true)

    await slider.focus()
    await slider.press("Home")
    await expect(number).toHaveValue("1.00e-7")
    await expect
      .poll(() =>
        api.preflightBodies.some(
          (body) => isRecord(body["config"]) && body["config"]["learning_rate"] === 1e-7,
        ),
      )
      .toBe(true)

    await slider.press("End")
    await expect(number).toHaveValue("1.00e-3")
    await expect
      .poll(() =>
        api.preflightBodies.some(
          (body) => isRecord(body["config"]) && body["config"]["learning_rate"] === 1e-3,
        ),
      )
      .toBe(true)

    await number.fill("1e-4")
    await expect(slider).toHaveValue("750")
    await expect
      .poll(() =>
        api.preflightBodies.some(
          (body) => isRecord(body["config"]) && body["config"]["learning_rate"] === 1e-4,
        ),
      )
      .toBe(true)
  })

  test("syncs basic and advanced integer inputs with their sliders and preflight snapshots", async ({
    page,
  }) => {
    const api = await installTrainingApi(page)
    await openTraining(page)
    const epochs = page.getByRole("textbox", { name: "Epochs" })
    const epochsSlider = page.getByRole("slider", { name: "Epochs slider" })
    await epochs.fill("31")
    await expect(epochsSlider).toHaveValue("31")
    await expect
      .poll(() =>
        api.preflightBodies.some(
          (body) => isRecord(body["config"]) && body["config"]["epochs"] === 31,
        ),
      )
      .toBe(true)

    await page.getByText("Advanced settings", { exact: true }).click()
    const accumulation = page.getByRole("textbox", { name: "Gradient accumulation" })
    const accumulationSlider = page.getByRole("slider", { name: "Gradient accumulation slider" })
    await accumulation.fill("8")
    await expect(accumulationSlider).toHaveValue("8")
    await expect
      .poll(() =>
        api.preflightBodies.some(
          (body) => isRecord(body["config"]) && body["config"]["gradient_accumulation"] === 8,
        ),
      )
      .toBe(true)
  })

  test("preserves incomplete numeric text and blocks stale estimates until the value is repaired", async ({
    page,
  }) => {
    const api = await installTrainingApi(page)
    await openTraining(page)
    const number = page.getByRole("textbox", { name: "Learning rate" })
    await expect(page.getByRole("button", { name: "Start training" })).toBeEnabled()
    const priorRequests = api.preflightBodies.length

    await number.fill("1e-")
    await expect(number).toHaveValue("1e-")
    await expect(number).toHaveAttribute("aria-invalid", "true")
    await expect(page.getByRole("button", { name: "Start training" })).toBeDisabled()
    await page.waitForTimeout(250)
    expect(api.preflightBodies.length).toBe(priorRequests)

    await number.fill("1e-5")
    await expect(number).not.toHaveAttribute("aria-invalid", "true")
    await expect
      .poll(() => page.getByRole("button", { name: "Start training" }).isEnabled())
      .toBe(true)
  })

  test("shows a server-backed memory refusal dialog without creating a job", async ({ page }) => {
    const api = await installTrainingApi(page, {
      onPreflight: async (route) => {
        await route.fulfill({
          contentType: "application/json",
          body: JSON.stringify(preflight(4 * 1024 ** 3, false)),
        })
      },
    })
    await page.setViewportSize({ width: 375, height: 820 })
    await openTraining(page)
    const dialog = page.getByRole("dialog", { name: "GPU memory is currently insufficient" })
    await expect(dialog).toBeVisible()
    await expect(dialog).toContainText("Required")
    await expect(dialog).toContainText("Training peak")
    await expect(dialog).toContainText("3.00 GiB")
    await expect(dialog).toContainText("5.00 GiB")
    await expect(dialog).toContainText("4.00 GiB")
    await expect(dialog).toContainText("2.00 GiB")
    await expect(dialog.locator("time")).toHaveAttribute("datetime", OBSERVED_AT)
    const noHorizontalOverflow = await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    )
    expect(noHorizontalOverflow).toBe(true)
    const screenshotRoot = resolve(process.cwd(), "../output/training/task7-browser/screenshots")
    await mkdir(screenshotRoot, { recursive: true })
    await page.screenshot({
      path: join(screenshotRoot, "memory-refusal-375.png"),
      animations: "disabled",
    })

    const cancel = dialog.getByRole("button", { name: "Cancel" })
    const close = dialog.getByRole("button", { name: "Close details" })
    await expect(cancel).toBeFocused()
    await page.keyboard.press("Tab")
    await expect(close).toBeFocused()
    await page.keyboard.press("Tab")
    await expect(cancel).toBeFocused()
    await page.keyboard.press("Shift+Tab")
    await expect(close).toBeFocused()
    await page.keyboard.press("Escape")
    await expect(dialog).toHaveCount(0)
    await expect(page.getByRole("button", { name: "Refresh estimate" })).toBeFocused()
    expect(api.submitBodies).toHaveLength(0)
    await expect(page.getByRole("button", { name: "Start training" })).toBeDisabled()
  })

  test("shows final server recheck refusal details after an admitted estimate", async ({
    page,
  }) => {
    const api = await installTrainingApi(page, {
      onSubmit: async (route) => {
        await route.fulfill({
          status: 409,
          contentType: "application/json",
          body: JSON.stringify({
            detail: {
              code: "training_memory_refused",
              message: "private recheck details are not shown",
              reason: "insufficient_free_memory",
              training_peak_bytes: 3 * 1024 ** 3,
              reserve_bytes: 2 * 1024 ** 3,
              required_bytes: 5 * 1024 ** 3,
              free_bytes: 4 * 1024 ** 3,
              profile_identity: HASH,
              observed_at: OBSERVED_AT,
            },
          }),
        })
      },
    })
    await openTraining(page)
    await expect(page.getByRole("button", { name: "Start training" })).toBeEnabled()
    await page.getByRole("button", { name: "Start training" }).click()
    const dialog = page.getByRole("dialog", { name: "GPU memory is currently insufficient" })
    await expect(dialog).toBeVisible()
    await expect(dialog.locator("time")).toHaveAttribute("datetime", OBSERVED_AT)
    await expect(dialog).toContainText("5.00 GiB")
    await expect(dialog).toContainText("4.00 GiB")
    await expect(dialog).not.toContainText("private recheck details")
    expect(api.submitBodies).toHaveLength(1)
  })

  test("distinguishes an unsupported profile from current memory shortage", async ({ page }) => {
    await installTrainingApi(page, {
      onPreflight: async (route) => {
        await route.fulfill({
          status: 409,
          contentType: "application/json",
          body: JSON.stringify({
            detail: {
              code: "training_memory_refused",
              message: "private server message is not shown",
              reason: "memory_profile_unsupported",
              training_peak_bytes: 0,
              reserve_bytes: 2 * 1024 ** 3,
              required_bytes: 2 * 1024 ** 3,
              free_bytes: 12 * 1024 ** 3,
            },
          }),
        })
      },
    })
    await openTraining(page)
    const dialog = page.getByRole("dialog", {
      name: "This configuration has no calibrated memory profile",
    })
    await expect(dialog).toBeVisible()
    await expect(dialog).toContainText("Choose a supported batch size")
    await expect(dialog).toContainText("The server refusal did not include its observation time")
    await expect(dialog).not.toContainText("private server message")
  })

  test("keeps the start action disabled while dataset validation is running", async ({ page }) => {
    const api = await installTrainingApi(page, {
      datasetStatus: {
        registered: true,
        valid: false,
        reason: "dataset_validating",
        snapshot: null,
      },
    })
    await openTraining(page)
    await expect(
      page.getByRole("status").filter({ hasText: "The registered dataset is still being checked" }),
    ).toBeVisible()
    await expect(page.getByRole("button", { name: "Start training" })).toBeDisabled()
    expect(api.preflightBodies).toHaveLength(0)
    expect(api.submitBodies).toHaveLength(0)
  })

  test("aborts an older preflight and keeps the newest configuration estimate", async ({
    page,
  }) => {
    let preflightNumber = 0
    const observedFreeBytes: number[] = []
    await installTrainingApi(page, {
      onPreflight: async (route) => {
        preflightNumber += 1
        if (preflightNumber === 1) {
          await new Promise((resolvePromise) => setTimeout(resolvePromise, 650))
          try {
            await route.fulfill({
              contentType: "application/json",
              body: JSON.stringify(preflight(1 * 1024 ** 3)),
            })
          } catch {
            // The browser should have aborted this older response after the setting changed.
          }
          return
        }
        observedFreeBytes.push(7 * 1024 ** 3)
        await route.fulfill({
          contentType: "application/json",
          body: JSON.stringify(preflight(7 * 1024 ** 3)),
        })
      },
    })
    await openTraining(page)
    await expect.poll(() => preflightNumber).toBeGreaterThan(0)
    await page.getByRole("textbox", { name: "Epochs" }).fill("31")
    await expect.poll(() => observedFreeBytes.length).toBeGreaterThan(0)
    await expect(page.getByText("Observed", { exact: false })).toBeVisible()
    await expect(page.getByText("7.00 GiB", { exact: true })).toBeVisible()
    await expect(page.getByRole("button", { name: "Start training" })).toBeEnabled()
    await expect(page.getByRole("dialog")).toHaveCount(0)
  })

  test("submits one immutable request identity after a rapid duplicate click", async ({ page }) => {
    const api = await installTrainingApi(page, {
      onSubmit: async (route, body) => {
        await new Promise((resolvePromise) => setTimeout(resolvePromise, 350))
        const requestId =
          isRecord(body) && typeof body["request_id"] === "string" ? body["request_id"] : ""
        const submitted = trainingJob("starting", {
          config: isRecord(body) && isRecord(body["config"]) ? body["config"] : CONFIG,
          request_id: requestId,
        })
        await route.fulfill({
          status: 202,
          contentType: "application/json",
          body: JSON.stringify(submitted),
        })
      },
    })
    await openTraining(page)
    const start = page.getByRole("button", { name: "Start training" })
    await expect(start).toBeEnabled()
    await start.dblclick()
    await expect(page.getByRole("heading", { name: /Run 8b3f3a95/ })).toBeVisible()
    expect(api.submitBodies).toHaveLength(1)
    const requestId = api.submitBodies[0]?.["request_id"]
    expect(typeof requestId).toBe("string")
    expect(api.submitBodies[0]).toMatchObject({ dataset_id: "cuhk-pedes", config: CONFIG })
  })

  test("keeps the newest round-trip preflight when an older submitted snapshot is refused later", async ({
    page,
  }) => {
    let releaseSubmission: () => void = () => {}
    let useNewerEpochs30Estimate = false
    const submissionGate = new Promise<void>((resolve) => {
      releaseSubmission = resolve
    })
    const api = await installTrainingApi(page, {
      onPreflight: async (route, body) => {
        const requestConfig = isRecord(body) && isRecord(body["config"]) ? body["config"] : null
        const epochs = requestConfig === null ? undefined : requestConfig["epochs"]
        const freeBytes =
          epochs === 31 ? 7 * 1024 ** 3 : useNewerEpochs30Estimate ? 6 * 1024 ** 3 : 10 * 1024 ** 3
        await route.fulfill({
          contentType: "application/json",
          body: JSON.stringify(preflight(freeBytes)),
        })
      },
      onSubmit: async (route) => {
        await submissionGate
        await route.fulfill({
          status: 409,
          contentType: "application/json",
          body: JSON.stringify({
            detail: {
              code: "training_memory_refused",
              message: "the submitted snapshot was refused",
              reason: "insufficient_free_memory",
              training_peak_bytes: 3 * 1024 ** 3,
              reserve_bytes: 2 * 1024 ** 3,
              required_bytes: 5 * 1024 ** 3,
              free_bytes: 4 * 1024 ** 3,
              profile_identity: HASH,
              observed_at: OBSERVED_AT,
            },
          }),
        })
      },
    })
    await openTraining(page)
    const start = page.getByRole("button", { name: "Start training" })
    await expect(start).toBeEnabled()
    await start.click()
    await expect.poll(() => api.submitBodies.length).toBe(1)
    const epochs = page.getByRole("textbox", { name: "Epochs" })
    await epochs.fill("31")
    await expect
      .poll(() =>
        api.preflightBodies.some(
          (body) => isRecord(body["config"]) && body["config"]["epochs"] === 31,
        ),
      )
      .toBe(true)
    await expect(page.getByText("7.00 GiB", { exact: true })).toBeVisible()
    const previousPreflightCount = api.preflightBodies.length
    useNewerEpochs30Estimate = true
    await epochs.fill("30")
    await expect.poll(() => api.preflightBodies.length).toBeGreaterThan(previousPreflightCount)
    await expect(page.getByText("6.00 GiB", { exact: true })).toBeVisible()

    releaseSubmission()
    const dialog = page.getByRole("dialog", { name: "GPU memory is currently insufficient" })
    await expect(dialog).toBeVisible()
    await expect(dialog).toContainText("refusal applies to the immutable settings sent")
    await expect(dialog).toContainText("30 epochs · LR 1.00e-5 · micro batch 16")
    await dialog.getByRole("button", { name: "Cancel" }).click()
    await expect(epochs).toHaveValue("30")
    await expect(page.getByText("6.00 GiB", { exact: true })).toBeVisible()
    await expect(start).toBeEnabled()
  })

  test("refreshes history and keeps cancel and manual resume actions visible", async ({ page }) => {
    const interrupted = trainingJob("interrupted")
    const api = await installTrainingApi(page)
    await openTraining(page)
    const history = page.getByRole("heading", { name: "Training history" })
    await expect(history).toBeVisible()
    api.jobs.unshift(interrupted)
    await page.getByRole("button", { name: "Refresh", exact: true }).click()
    const runEntry = page.getByRole("button", { name: /2026.*Run 8b3f3a95/ })
    await expect(runEntry).toBeVisible()
    await runEntry.click()
    await expect(page.getByRole("heading", { name: /Run 8b3f3a95/ })).toBeVisible()
    await expect(page.getByText("Interrupted · manual resume available")).toBeVisible()
    await page.getByRole("button", { name: "Resume from complete checkpoint" }).click()
    await expect(page.getByText("Training", { exact: true })).toBeVisible()
    expect(api.resumeBodies).toHaveLength(1)
    expect(api.resumeBodies[0]?.["request_id"]).toBeTruthy()
  })

  test("releases the history load-more state when Refresh supersedes a delayed page", async ({
    page,
  }) => {
    const firstJob = trainingJob("failed", {
      id: "1b3f3a95-1dbe-4d4f-97c7-1e4a72b98111",
      created_at: "2026-10-03T12:00:00Z",
    })
    const secondJob = trainingJob("interrupted", {
      id: "2b3f3a95-1dbe-4d4f-97c7-1e4a72b98111",
      created_at: "2026-10-02T12:00:00Z",
    })
    let secondPageCalls = 0
    await installTrainingApi(page, {
      onHistoryPage: async (route, cursor) => {
        if (cursor === null) {
          await route.fulfill({
            contentType: "application/json",
            body: JSON.stringify({ items: [firstJob], next_cursor: "jobs-page-1" }),
          })
          return
        }
        secondPageCalls += 1
        if (secondPageCalls === 1) {
          await new Promise((resolvePromise) => setTimeout(resolvePromise, 1_200))
          try {
            await route.fulfill({
              contentType: "application/json",
              body: JSON.stringify({ items: [secondJob], next_cursor: null }),
            })
          } catch {
            // Refresh should abort this older page request.
          }
          return
        }
        await route.fulfill({
          contentType: "application/json",
          body: JSON.stringify({ items: [secondJob], next_cursor: null }),
        })
      },
    })
    await openTraining(page)
    await expect(page.getByRole("button", { name: /Run 1b3f3a95/ })).toBeVisible()
    await page.getByRole("button", { name: "Load older runs" }).click()
    await expect(page.getByRole("button", { name: "Loading…" })).toBeDisabled()
    await page.getByRole("button", { name: "Refresh", exact: true }).click()
    await expect(page.getByRole("button", { name: /Run 1b3f3a95/ })).toBeVisible()
    const loadMore = page.getByRole("button", { name: "Load older runs" })
    await expect(loadMore).toBeEnabled()
    await loadMore.click()
    await expect(page.getByRole("button", { name: /Run 2b3f3a95/ })).toBeVisible()
  })

  test("requests cooperative cancel and shows the server phase transition", async ({ page }) => {
    const api = await installTrainingApi(page, { initialJobs: [trainingJob("training")] })
    await openTraining(page)
    await page.getByRole("button", { name: /2026.*Run 8b3f3a95/ }).click()
    await expect(page.getByRole("button", { name: "Request cooperative cancel" })).toBeVisible()
    await page.getByRole("button", { name: "Request cooperative cancel" }).click()
    await expect(page.getByText("Cancellation requested")).toBeVisible()
    expect(api.cancelBodies).toHaveLength(1)
    expect(api.cancelBodies[0]?.["request_id"]).toBeTruthy()
  })

  test("shows exact memory metadata when manual resume is refused", async ({ page }) => {
    const interrupted = trainingJob("interrupted")
    await installTrainingApi(page, {
      initialJobs: [interrupted],
      onResume: async (route) => {
        await route.fulfill({
          status: 409,
          contentType: "application/json",
          body: JSON.stringify({
            detail: {
              code: "training_memory_refused",
              message: "private refusal text is not shown",
              reason: "insufficient_free_memory",
              training_peak_bytes: 3 * 1024 ** 3,
              reserve_bytes: 2 * 1024 ** 3,
              required_bytes: 5 * 1024 ** 3,
              free_bytes: 4 * 1024 ** 3,
              profile_identity: HASH,
              observed_at: OBSERVED_AT,
            },
          }),
        })
      },
    })
    await openTraining(page)
    await page.getByRole("button", { name: /2026.*Run 8b3f3a95/ }).click()
    await page.getByRole("button", { name: "Resume from complete checkpoint" }).click()
    const dialog = page.getByRole("dialog", { name: "GPU memory is currently insufficient" })
    await expect(dialog).toBeVisible()
    await expect(dialog.locator("time")).toHaveAttribute("datetime", OBSERVED_AT)
    await expect(dialog).toContainText("5.00 GiB")
    await expect(dialog).toContainText("4.00 GiB")
    await expect(dialog).not.toContainText("private refusal text")
  })

  test("follows the metric page cursor before rendering the epoch chart", async ({ page }) => {
    const metricCursors: Array<string | null> = []
    await installTrainingApi(page, {
      initialJobs: [trainingJob("training")],
      onMetricsPage: async (route, cursor) => {
        metricCursors.push(cursor)
        const epoch = cursor === null ? 1 : 2
        await route.fulfill({
          contentType: "application/json",
          body: JSON.stringify({
            items: [
              {
                epoch,
                step: epoch * 6,
                training_loss: epoch === 1 ? 0.8 : 0.4,
                validation_recall_at_1: epoch === 1 ? 0.25 : 0.5,
                allocated_bytes: 1_000 + epoch,
                reserved_bytes: 2_000 + epoch,
                observed_at: OBSERVED_AT,
              },
            ],
            next_cursor: cursor === null ? "metrics-page-1" : null,
          }),
        })
      },
    })
    await openTraining(page)
    await page.getByRole("button", { name: /Run 8b3f3a95/ }).click()
    await expect(page.locator(".training-chart__labels").first()).toContainText("E2")
    expect(metricCursors).toContain("metrics-page-1")
  })

  test("walks bounded telemetry pages, polls after the latest log cursor, and reloads the current tail", async ({
    page,
  }) => {
    const job = trainingJob("training")
    const metricCursors: Array<string | null> = []
    const logCursors: Array<string | null> = []
    let jobReads = 0
    const logRow = (number: number) => ({
      cursor: `log-${String(number).padStart(3, "0")}`,
      level: "info",
      message: `log message ${String(number).padStart(3, "0")}`,
      observed_at: OBSERVED_AT,
    })
    await installTrainingApi(page, {
      initialJobs: [job],
      onGetJob: async (route) => {
        jobReads += 1
        await route.fulfill({ contentType: "application/json", body: JSON.stringify(job) })
      },
      onMetricsPage: async (route, cursor) => {
        metricCursors.push(cursor)
        const items =
          cursor === null
            ? [
                {
                  epoch: 1,
                  step: 6,
                  training_loss: 0.8,
                  validation_recall_at_1: 0.25,
                  allocated_bytes: 1_000,
                  reserved_bytes: 2_000,
                  observed_at: OBSERVED_AT,
                },
              ]
            : [
                {
                  epoch: 2,
                  step: 12,
                  training_loss: 0.4,
                  validation_recall_at_1: 0.5,
                  allocated_bytes: 1_200,
                  reserved_bytes: 2_100,
                  observed_at: OBSERVED_AT,
                },
              ]
        await route.fulfill({
          contentType: "application/json",
          body: JSON.stringify({ items, next_cursor: cursor === null ? "metrics-page-1" : null }),
        })
      },
      onLogsPage: async (route, cursor) => {
        logCursors.push(cursor)
        if (cursor === null) {
          await route.fulfill({
            contentType: "application/json",
            body: JSON.stringify({
              items: Array.from({ length: 50 }, (_unused, index) => logRow(index + 1)),
              next_cursor: "logs-page-50",
            }),
          })
          return
        }
        if (cursor === "logs-page-50") {
          const pageItems =
            job.phase === "succeeded"
              ? [logRow(51), logRow(52), logRow(53)]
              : [logRow(51), logRow(52)]
          await route.fulfill({
            contentType: "application/json",
            body: JSON.stringify({ items: pageItems, next_cursor: null }),
          })
          return
        }
        if (cursor === "log-052") {
          job.phase = "succeeded"
          job.finished_at = OBSERVED_AT
          job.updated_at = OBSERVED_AT
          await route.fulfill({
            contentType: "application/json",
            body: JSON.stringify({ items: [logRow(53)], next_cursor: null }),
          })
          return
        }
        await route.fulfill({
          contentType: "application/json",
          body: JSON.stringify({ items: [], next_cursor: null }),
        })
      },
    })
    await openTraining(page)
    await page.getByRole("button", { name: /Run 8b3f3a95/ }).click()
    await expect(page.getByText("log message 053", { exact: true })).toBeVisible({ timeout: 8_000 })
    await expect(page.locator(".training-chart__labels").first()).toContainText("E2")
    await expect(page.getByText("Succeeded", { exact: true })).toBeVisible()
    expect(metricCursors).toContain("metrics-page-1")
    expect(logCursors).toContain("logs-page-50")
    expect(logCursors).toContain("log-052")
    await expect(page.locator(".training-log")).toHaveCount(50)
    const terminalReadCount = jobReads
    await page.waitForTimeout(2_300)
    expect(jobReads).toBe(terminalReadCount)

    await page.getByRole("button", { name: "Back to setup" }).click()
    await page.getByRole("button", { name: /Run 8b3f3a95/ }).click()
    await expect(page.getByText("log message 053", { exact: true })).toBeVisible()
    await expect(page.locator(".training-log")).toHaveCount(50)
  })

  test("shows final evaluation and sends the candidate link to the existing model selector", async ({
    page,
  }) => {
    const succeeded = trainingJob("succeeded")
    await installTrainingApi(page, { initialJobs: [succeeded] })
    await openTraining(page)
    await page.getByRole("button", { name: /2026.*Run 8b3f3a95/ }).click()
    await expect(page.getByRole("heading", { name: "Final held-out comparison" })).toBeVisible()
    await expect(page.getByRole("table", { name: "Final retrieval evaluation" })).toContainText(
      "50.0%",
    )
    await expect(page.getByText("Manual model application remains blocked")).toBeVisible()
    await expect(page.getByRole("button", { name: /apply/i })).toHaveCount(0)
    await page
      .getByRole("button", { name: "Review candidate in the existing model selector" })
      .click()
    await expect(page.getByRole("heading", { name: "Cameras" })).toBeVisible()
    await expect(page.getByRole("heading", { name: "Person search model" })).toBeVisible()
    await expect(page.getByText("CUHK-PEDES candidate")).toBeVisible()
  })

  test("keeps training usable at required widths and records screenshots", async ({ page }) => {
    await installTrainingApi(page, { initialJobs: [trainingJob("succeeded")] })
    await openTraining(page)
    const screenshotRoot = resolve(process.cwd(), "../output/training/task7-browser/screenshots")
    await mkdir(screenshotRoot, { recursive: true })
    for (const width of [375, 768, 1280, 1440]) {
      await page.setViewportSize({ width, height: 900 })
      await expect(page.getByRole("heading", { name: "CLIP training" })).toBeVisible()
      await expect(page.getByText("Admitted", { exact: true })).toBeVisible()
      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth > window.innerWidth,
      )
      expect(overflow, `unexpected horizontal overflow at ${width}px`).toBe(false)
      const learningRateTrackWidth = await page
        .getByRole("slider", { name: "Learning rate slider" })
        .evaluate((slider) => slider.getBoundingClientRect().width)
      expect(
        learningRateTrackWidth,
        `learning-rate slider is too narrow at ${width}px`,
      ).toBeGreaterThan(80)
      await page.screenshot({ path: join(screenshotRoot, `training-${width}.png`), fullPage: true })
      if (width >= 768) {
        await page.getByRole("heading", { name: "Training configuration" }).scrollIntoViewIfNeeded()
        await page.screenshot({ path: join(screenshotRoot, `training-${width}-controls.png`) })
      }
    }
    await page.setViewportSize({ width: 1280, height: 900 })
    await page.getByRole("heading", { name: "Training history" }).scrollIntoViewIfNeeded()
    await page.screenshot({ path: join(screenshotRoot, "training-history-1280.png") })
    await page.getByRole("button", { name: /2026.*Run 8b3f3a95/ }).click()
    await expect(page.getByRole("heading", { name: "Final held-out comparison" })).toBeVisible()
    await page.getByRole("heading", { name: "Final held-out comparison" }).scrollIntoViewIfNeeded()
    await page.screenshot({ path: join(screenshotRoot, "training-detail-1280.png") })
    await page
      .getByRole("button", { name: "Review candidate in the existing model selector" })
      .scrollIntoViewIfNeeded()
    await page.screenshot({ path: join(screenshotRoot, "training-candidate-gate-1280.png") })
  })
})
