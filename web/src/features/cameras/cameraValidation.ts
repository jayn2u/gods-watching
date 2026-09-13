import type { CameraTestResponse } from "../../app/client"

type Validated<T> =
  | Readonly<{ kind: "valid"; value: T }>
  | Readonly<{ kind: "invalid"; message: string }>

export type SourceDraft = Readonly<{
  kind: "replacement" | "keep"
  value: string | null
}>

const SOURCE_URL_LIMIT = 2048

function invalid(message: string): Readonly<{ kind: "invalid"; message: string }> {
  return { kind: "invalid", message }
}

function hasControlCharacter(value: string): boolean {
  for (const character of value) {
    const codePoint = character.codePointAt(0)
    if (codePoint !== undefined && (codePoint <= 31 || codePoint === 127)) {
      return true
    }
  }
  return false
}

export function validateCameraName(rawValue: string): Validated<string> {
  const value = rawValue.trim()
  if (value.length === 0) {
    return invalid("Enter a camera name.")
  }
  if (value.length > 80) {
    return invalid("Camera names must be 80 characters or fewer.")
  }
  if (hasControlCharacter(value)) {
    return invalid("Camera names cannot contain control characters.")
  }
  return { kind: "valid", value }
}

export function validateThreshold(rawValue: string): Validated<number> {
  const value = Number(rawValue)
  if (!Number.isFinite(value) || value < 0.1 || value > 0.95) {
    return invalid("Detection threshold must be between 0.10 and 0.95.")
  }
  return { kind: "valid", value }
}

export function validateSourceUrl(rawValue: string): Validated<string> {
  const value = rawValue.trim()
  if (value.length === 0) {
    return invalid("Enter an RTSP source URL.")
  }
  if (value.length > SOURCE_URL_LIMIT) {
    return invalid("Source URLs must be 2048 characters or fewer.")
  }
  if (hasControlCharacter(value)) {
    return invalid("Source URLs cannot contain control characters.")
  }

  let parsed: URL
  try {
    parsed = new URL(value)
  } catch (error) {
    if (error instanceof TypeError) {
      return invalid("Enter a complete RTSP source URL.")
    }
    throw error
  }
  if (parsed.protocol !== "rtsp:") {
    return invalid("The source URL must use the rtsp:// scheme.")
  }
  if (parsed.hostname.length === 0) {
    return invalid("The RTSP source must include a host.")
  }
  if (parsed.port !== "") {
    const port = Number(parsed.port)
    if (!Number.isInteger(port) || port < 1 || port > 65535) {
      return invalid("The RTSP source port must be between 1 and 65535.")
    }
  }
  return { kind: "valid", value }
}

export function validateOptionalSourceUrl(rawValue: string): Validated<SourceDraft> {
  if (rawValue.trim() === "") {
    return { kind: "valid", value: { kind: "keep", value: null } }
  }
  const result = validateSourceUrl(rawValue)
  return result.kind === "invalid"
    ? result
    : { kind: "valid", value: { kind: "replacement", value: result.value } }
}

export function testMetadataLabel(result: CameraTestResponse): string {
  const port = result.source_port === null ? "default" : String(result.source_port)
  return `${result.source_host}:${port} · ${result.codec} · ${result.width}×${result.height}`
}
