import { expect, test } from "@playwright/test"

const authenticatedSession = {
  authenticated: true,
  idle_expires_at: "2026-09-07T12:30:00Z",
  absolute_expires_at: "2026-09-07T20:00:00Z",
}

test.describe("shell and session states", () => {
  test("shows the truthful unavailable state when the API cannot be reached", async ({ page }) => {
    await page.route("**/api/session", async (route) => {
      await route.abort("connectionrefused")
    })

    await page.goto("/")

    await expect(page.getByRole("heading", { name: "God’s Watching" })).toBeVisible()
    await expect(page.getByText("The session service is unavailable.")).toBeVisible()
    await expect(page.getByRole("button", { name: "Retry" })).toBeVisible()
  })

  test("supports the session contract, logout, and shell geometry", async ({ page }) => {
    let authenticated = false
    let loginBody = ""
    await page.route("**/api/session", async (route) => {
      const method = route.request().method()
      if (method === "GET") {
        await route.fulfill({
          contentType: "application/json",
          body: JSON.stringify(
            authenticated
              ? authenticatedSession
              : {
                  authenticated: false,
                  idle_expires_at: null,
                  absolute_expires_at: null,
                },
          ),
        })
        return
      }
      if (method === "POST") {
        authenticated = true
        loginBody = route.request().postData() ?? ""
        await route.fulfill({
          contentType: "application/json",
          body: JSON.stringify(authenticatedSession),
        })
        return
      }
      if (method === "DELETE") {
        authenticated = false
        await route.fulfill({ status: 204, body: "" })
        return
      }
      await route.fallback()
    })
    await page.route("**/api/cameras", async (route) => {
      await route.fulfill({ contentType: "application/json", body: "[]" })
    })
    await page.route("**/api/settings", async (route) => {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          retention_days: 7,
          quota_bytes: 10_000_000_000,
          wall_slot_ids: [null, null, null, null],
        }),
      })
    })
    await page.route("**/api/session/activity", async (route) => {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(authenticatedSession),
      })
    })

    await page.goto("/")
    await page.getByLabel("Operator ID").fill("operator")
    await page.getByLabel("Password").fill("a-password-longer-than-twelve")
    await page.getByRole("button", { name: "Sign in" }).click()
    expect(loginBody).toBe(JSON.stringify({ password: "a-password-longer-than-twelve" }))

    await expect(page.getByRole("button", { name: "Live wall" })).toBeVisible()
    await expect(page.getByText("0 of 4 assigned")).toBeVisible()
    await expect(page.locator("[data-shell-wall-grid]")).toHaveCSS(
      "grid-template-columns",
      /^\d+px \d+px$/,
    )

    await page.getByRole("button", { name: "Sign out" }).click()
    await expect(page.getByRole("button", { name: "Sign in" })).toBeVisible()
  })
})
