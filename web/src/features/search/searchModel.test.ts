import { describe, expect, it } from "vitest"
import { buildSearchRequest, localDateTimeToUtc, normalizeQuery } from "./searchModel"

describe("person search request model", () => {
  it("normalizes a printable description without inventing a language check", () => {
    expect(normalizeQuery("  a person　 with   a bag  ")).toBe("a person with a bag")
  })

  it("builds an explicit text request with inclusive UTC filters", () => {
    const result = buildSearchRequest({
      cameraIds: ["camera-a"],
      from: "2026-09-08T09:00",
      mode: "text",
      query: "a person with a red bag",
      to: "2026-09-08T10:00",
    })

    expect(result.kind).toBe("ok")
    if (result.kind !== "ok") {
      return
    }
    expect(result.request).toMatchObject({
      camera_ids: ["camera-a"],
      mode: "text",
      query: "a person with a red bag",
      sort: "similarity",
    })
    expect(result.request.from).toBe(localDateTimeToUtc("2026-09-08T09:00"))
    expect(result.request.to).toBe(localDateTimeToUtc("2026-09-08T10:00"))
  })

  it("requires explicit browse and never turns blank text into browse", () => {
    const blank = buildSearchRequest({ cameraIds: [], from: "", mode: "text", query: "  ", to: "" })
    expect(blank).toEqual({
      kind: "invalid",
      message: "Enter a person description before searching.",
    })

    const browse = buildSearchRequest({
      cameraIds: [],
      from: "",
      mode: "browse",
      query: "",
      to: "",
    })
    expect(browse).toEqual({
      kind: "ok",
      request: { limit: 30, mode: "browse", sort: "newest" },
    })
  })

  it("reports malformed local date-time values at the boundary", () => {
    expect(localDateTimeToUtc("2026-02-30T09:00")).toBeUndefined()
    expect(
      buildSearchRequest({
        cameraIds: [],
        from: "2026-02-30T09:00",
        mode: "text",
        query: "person",
        to: "",
      }),
    ).toEqual({ kind: "invalid", message: "Use a valid local date and time for the from filter." })
  })
})
