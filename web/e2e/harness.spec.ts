import { randomUUID } from "node:crypto"
import { mkdir, writeFile } from "node:fs/promises"
import { createServer } from "node:http"
import { dirname, isAbsolute, join, resolve } from "node:path"
import type { TestInfo } from "@playwright/test"
import { expect, test } from "@playwright/test"

type NeutralServer = {
  close: () => Promise<void>
  id: string
  origin: string
}

function evidenceArtifactPath(testInfo: TestInfo, filename: string): string {
  const configuredRoot = process.env["GW_E2E_EVIDENCE_ROOT"]
  if (configuredRoot === undefined || configuredRoot === "") {
    return testInfo.outputPath(filename)
  }
  if (!isAbsolute(configuredRoot)) {
    throw new Error("GW_E2E_EVIDENCE_ROOT must be absolute when supplied.")
  }
  return join(resolve(configuredRoot), filename)
}

async function writeJson(path: string, value: object): Promise<void> {
  await mkdir(dirname(path), { recursive: true })
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`, "utf8")
}

async function startNeutralServer(): Promise<NeutralServer> {
  const id = randomUUID()
  const server = createServer((_request, response) => {
    response.writeHead(200, { "content-type": "text/html; charset=utf-8" })
    response.end(
      `<!doctype html><html><head><title>GW browser harness neutral page</title></head><body><main id="browser-harness-observation" data-fixture-id="${id}">Browser harness ready</main></body></html>`,
    )
  })

  await new Promise<void>((resolvePromise, reject) => {
    server.once("error", reject)
    server.listen({ host: "127.0.0.1", port: 0 }, () => {
      server.off("error", reject)
      resolvePromise()
    })
  })

  const address = server.address()
  if (address === null || typeof address === "string") {
    await new Promise<void>((resolvePromise, reject) => {
      server.close((error) => (error === undefined ? resolvePromise() : reject(error)))
    })
    throw new Error("Neutral browser fixture did not bind a TCP port.")
  }
  const port = address.port

  return {
    id,
    origin: `http://127.0.0.1:${port}`,
    close: async () => {
      await new Promise<void>((resolvePromise, reject) => {
        server.close((error) => (error === undefined ? resolvePromise() : reject(error)))
      })
    },
  }
}

test("Chromium launches and records navigation evidence from an owned neutral origin", async ({
  page,
}, testInfo) => {
  const fixture = await startNeutralServer()
  try {
    await page.goto(fixture.origin, { waitUntil: "domcontentloaded" })
    const title = await page.title()
    const domObservation = await page.locator("#browser-harness-observation").textContent()
    const fixtureId = await page
      .locator("#browser-harness-observation")
      .getAttribute("data-fixture-id")
    const browser = page.context().browser()
    if (browser === null) {
      throw new Error("Playwright did not expose the launched browser instance.")
    }
    if (domObservation !== "Browser harness ready" || fixtureId !== fixture.id) {
      throw new Error("Neutral page DOM observation did not match the server owned by this test.")
    }

    expect(title).toBe("GW browser harness neutral page")
    const screenshotPath = evidenceArtifactPath(testInfo, "harness-neutral-page.png")
    if (process.env["GW_HARNESS_TEST_SUPPRESS_REQUIRED_ARTIFACT"] !== "1") {
      await mkdir(dirname(screenshotPath), { recursive: true })
      await page.screenshot({ path: screenshotPath, fullPage: true })
      await writeJson(evidenceArtifactPath(testInfo, "browser-observation.json"), {
        browser_version: browser.version(),
        origin: fixture.origin,
        title,
        dom_observation: domObservation,
        fixture_id: fixtureId,
        screenshot: screenshotPath,
      })
    }
  } finally {
    await fixture.close()
  }
})

test("owned fixture servers receive distinct origins while live concurrently", async ({
  page,
}, testInfo) => {
  const [first, second] = await Promise.all([startNeutralServer(), startNeutralServer()])
  try {
    if (first.origin === second.origin) {
      throw new Error("Concurrent neutral fixtures were assigned the same origin.")
    }
    await page.goto(first.origin, { waitUntil: "domcontentloaded" })
    await expect(page.locator("#browser-harness-observation")).toHaveAttribute(
      "data-fixture-id",
      first.id,
    )
    await page.goto(second.origin, { waitUntil: "domcontentloaded" })
    await expect(page.locator("#browser-harness-observation")).toHaveAttribute(
      "data-fixture-id",
      second.id,
    )

    await writeJson(evidenceArtifactPath(testInfo, "isolation-observation.json"), {
      first_origin: first.origin,
      second_origin: second.origin,
      unique_origins: true,
    })
  } finally {
    await Promise.all([first.close(), second.close()])
  }
})
