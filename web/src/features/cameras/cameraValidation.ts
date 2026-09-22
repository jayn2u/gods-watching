import type { CameraTestResponse } from "../../app/client"

type Validated<T> =
  | Readonly<{ kind: "valid"; value: T }>
  | Readonly<{ kind: "invalid"; message: string }>

export type RtspCredentials = Readonly<{
  username: string
  password: string
}>

export type SourceDraftMode = "create" | "edit"

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

export function validateRtspCredentials(
  username: string,
  password: string,
): Validated<RtspCredentials> {
  if (password.length > 0 && username.length === 0) {
    return invalid("Enter an RTSP username when providing a password.")
  }
  if (hasControlCharacter(username) || hasControlCharacter(password)) {
    return invalid("RTSP credentials cannot contain control characters.")
  }
  return { kind: "valid", value: { username, password } }
}

export function buildAuthenticatedSourceUrl(
  rawSourceUrl: string,
  username: string,
  password: string,
): Validated<string> {
  const source = validateSourceUrl(rawSourceUrl)
  if (source.kind === "invalid") {
    return source
  }
  const credentials = validateRtspCredentials(username, password)
  if (credentials.kind === "invalid") {
    return credentials
  }
  if (credentials.value.username.length === 0 && credentials.value.password.length === 0) {
    return source
  }

  let parsed: URL
  try {
    parsed = new URL(source.value)
    parsed.username = encodeURIComponent(credentials.value.username)
    parsed.password = encodeURIComponent(credentials.value.password)
  } catch (error) {
    if (error instanceof URIError || error instanceof TypeError) {
      return invalid("Enter valid RTSP credentials.")
    }
    throw error
  }
  const value = parsed.href
  if (value.length > SOURCE_URL_LIMIT) {
    return invalid("Authenticated source URLs must be 2048 characters or fewer.")
  }
  return { kind: "valid", value }
}

export function resolveSourceUrl(
  rawSourceUrl: string,
  username: string,
  password: string,
  mode: SourceDraftMode,
): Validated<string | null> {
  const hasSeparateCredentials = username.length > 0 || password.length > 0
  if (rawSourceUrl.trim() === "") {
    if (mode === "create") {
      return validateSourceUrl(rawSourceUrl)
    }
    if (hasSeparateCredentials) {
      return invalid("Enter a replacement RTSP source URL when changing credentials.")
    }
    return { kind: "valid", value: null }
  }
  const source = buildAuthenticatedSourceUrl(rawSourceUrl, username, password)
  return source.kind === "invalid" ? source : { kind: "valid", value: source.value }
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
