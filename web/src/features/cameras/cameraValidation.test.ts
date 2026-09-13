import { describe, expect, it } from "vitest"
import {
  testMetadataLabel,
  validateCameraName,
  validateOptionalSourceUrl,
  validateSourceUrl,
  validateThreshold,
} from "./cameraValidation"

describe("camera form boundary", () => {
  it("accepts a credential-bearing RTSP source without exposing it in metadata", () => {
    // Given: an operator-entered RTSP source with credentials.
    // When: the source crosses the form boundary and a probe response is rendered.
    const source = validateSourceUrl("rtsp://operator:secret@fixture:8554/live")
    const metadata = testMetadataLabel({
      source_host: "fixture",
      source_port: 8554,
      codec: "h264",
      width: 1920,
      height: 1080,
    })

    // Then: the request keeps the source while displayed probe metadata stays sanitized.
    expect(source).toEqual({ kind: "valid", value: "rtsp://operator:secret@fixture:8554/live" })
    expect(metadata).toContain("fixture")
    expect(metadata).not.toContain("secret")
  })

  it.each(["", "https://camera/live", "rtsp:///live", "rtsp://camera:0/live"])(
    "rejects malformed source %s",
    (value) => {
      // Given: a malformed or unsupported source.
      // When: the source crosses the form boundary.
      const result = validateSourceUrl(value)

      // Then: no source request value is produced.
      expect(result.kind).toBe("invalid")
    },
  )

  it("keeps an edit source empty when the operator makes no replacement", () => {
    // Given: an edit draft with no explicit source replacement.
    // When: the optional source crosses the form boundary.
    const result = validateOptionalSourceUrl("  ")

    // Then: the patch can omit source_url and preserve the working source.
    expect(result).toEqual({ kind: "valid", value: { kind: "keep", value: null } })
  })

  it.each(["", " ", "0", "1.0", "0.99"])("rejects invalid threshold %s", (value) => {
    // Given: a threshold outside the server contract.
    // When: the threshold crosses the form boundary.
    const result = validateThreshold(value)

    // Then: no threshold request value is produced.
    expect(result.kind).toBe("invalid")
  })

  it("trims a bounded camera name", () => {
    // Given: a valid camera name with surrounding whitespace.
    // When: the name crosses the form boundary.
    const result = validateCameraName("  East entrance  ")

    // Then: the typed name is normalized for the request.
    expect(result).toEqual({ kind: "valid", value: "East entrance" })
  })
})
