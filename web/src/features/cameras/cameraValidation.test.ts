import { describe, expect, it } from "vitest"
import {
  buildAuthenticatedSourceUrl,
  resolveSourceUrl,
  testMetadataLabel,
  validateCameraName,
  validateOptionalSourceUrl,
  validateRtspCredentials,
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

  it("keeps an edit source only when both replacement URL and credentials are blank", () => {
    // Given: the API omits the stored source URL from edit responses.
    expect(resolveSourceUrl("  ", "", "", "edit")).toEqual({ kind: "valid", value: null })

    // When: separate credentials are entered without a replacement URL.
    // Then: the operator must provide the source explicitly instead of silently dropping them.
    expect(resolveSourceUrl("", "operator", "secret", "edit")).toEqual({
      kind: "invalid",
      message: "Enter a replacement RTSP source URL when changing credentials.",
    })
  })

  it("percent-encodes raw RTSP credentials without double-encoding existing source text", () => {
    // Given: raw credentials containing RTSP userinfo delimiters and a literal percent sequence.
    const result = buildAuthenticatedSourceUrl(
      "rtsp://old:old-pass@fixture:8554/live?transport=tcp",
      "new#user/@",
      "raw%23 pass",
    )

    // Then: the new pair replaces the old pair and each raw character is encoded once.
    expect(result).toEqual({
      kind: "valid",
      value: "rtsp://new%23user%2F%40:raw%2523%20pass@fixture:8554/live?transport=tcp",
    })
  })

  it("preserves an embedded credential-bearing source when separate credentials are blank", () => {
    // Given: an existing URL is already complete and the separate fields are untouched.
    const source = "rtsp://operator:old%23pass@fixture:8554/live"

    // When: the shared source builder receives blank separate fields.
    const result = buildAuthenticatedSourceUrl(source, "", "")

    // Then: the original source is used verbatim for backward compatibility.
    expect(result).toEqual({ kind: "valid", value: source })
  })

  it("replaces both embedded credentials when only a new username is supplied", () => {
    // Given: an embedded pair and a username-only replacement.
    const result = buildAuthenticatedSourceUrl(
      "rtsp://old:old-pass@fixture:8554/live",
      "new-user",
      "",
    )

    // Then: the old password cannot be mixed into the new URL.
    expect(result).toEqual({ kind: "valid", value: "rtsp://new-user@fixture:8554/live" })
  })

  it("rejects a password without a username and allows a username with an empty password", () => {
    // Given: a password-only pair is ambiguous and must never inherit an embedded username.
    expect(validateRtspCredentials("", "secret")).toEqual({
      kind: "invalid",
      message: "Enter an RTSP username when providing a password.",
    })

    // Then: an explicit username-only credential is a valid replacement.
    expect(buildAuthenticatedSourceUrl("rtsp://old:old-pass@fixture/live", "new-user", "")).toEqual(
      { kind: "valid", value: "rtsp://new-user@fixture/live" },
    )
  })

  it("rejects a combined authenticated source over the 2048-character limit", () => {
    // Given: the source itself fits the limit but adding credentials would exceed it.
    const source = `rtsp://fixture/${"x".repeat(2028)}`

    // When: the new pair is built through the shared helper.
    const result = buildAuthenticatedSourceUrl(source, "operator", "secret")

    // Then: the bounded request is rejected before it reaches the API.
    expect(result).toEqual({
      kind: "invalid",
      message: "Authenticated source URLs must be 2048 characters or fewer.",
    })
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
