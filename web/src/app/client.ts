import {
  type AppearanceResponse,
  type CameraCreateRequest,
  type CameraPatchRequest,
  type CameraTestRequest,
  type CameraTestResponse,
  type CropResponse,
  parseAppearanceResponse,
  parseCameraResponse,
  parseCameraTestResponse,
  parseSearchResponse,
  parseSettingsResponse,
  type SearchRequest,
  type SearchResponse,
} from "./clientDomain"
import type { CameraResponse, SettingsPatch, SettingsResponse } from "./clientTypes"

export type {
  AppearanceResponse,
  BoundingBox,
  BrowseSearchRequest,
  CameraCreateRequest,
  CameraPatchRequest,
  CameraTestRequest,
  CameraTestResponse,
  CropResponse,
  SearchFilters,
  SearchRequest,
  SearchResponse,
  SimilarSearchRequest,
  TextSearchRequest,
} from "./clientDomain"

export type SessionResponse = Readonly<{
  authenticated: boolean
  idle_expires_at: string | null
  absolute_expires_at: string | null
}>

export type { CameraResponse, SettingsPatch, SettingsResponse, WallSlotIds } from "./clientTypes"

type ErrorPayload = Readonly<{
  code: string
  message: string
}>

export class HttpError extends Error {
  readonly name = "HttpError"

  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly retryAfterSeconds?: number,
  ) {
    super(message)
  }
}

export class NetworkError extends Error {
  readonly name = "NetworkError"

  constructor(
    message: string,
    readonly cause: unknown,
  ) {
    super(message)
  }
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError"
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function unknownField(record: Record<string, unknown>, key: string): unknown {
  return record[key]
}

function stringField(record: Record<string, unknown>, key: string): string | undefined {
  const value = record[key]
  return typeof value === "string" ? value : undefined
}

function nullableStringField(
  record: Record<string, unknown>,
  key: string,
): string | null | undefined {
  const value = record[key]
  if (value === null) {
    return null
  }
  return typeof value === "string" ? value : undefined
}

function booleanField(record: Record<string, unknown>, key: string): boolean | undefined {
  const value = record[key]
  return typeof value === "boolean" ? value : undefined
}

function parseSession(value: unknown): SessionResponse {
  if (!isRecord(value)) {
    throw new Error("session response is not an object")
  }
  const authenticated = booleanField(value, "authenticated")
  const idleExpiresAt = nullableStringField(value, "idle_expires_at")
  const absoluteExpiresAt = nullableStringField(value, "absolute_expires_at")
  if (
    authenticated === undefined ||
    idleExpiresAt === undefined ||
    absoluteExpiresAt === undefined
  ) {
    throw new Error("session response has an invalid shape")
  }
  return {
    authenticated,
    idle_expires_at: idleExpiresAt,
    absolute_expires_at: absoluteExpiresAt,
  }
}

function parseCameras(value: unknown): readonly CameraResponse[] {
  if (!Array.isArray(value)) {
    throw new Error("camera list response is not an array")
  }
  return value.map(parseCameraResponse)
}

function parseErrorPayload(value: unknown): ErrorPayload | undefined {
  if (!isRecord(value)) {
    return undefined
  }
  const detailValue = unknownField(value, "detail")
  const detail = isRecord(detailValue) ? detailValue : unknownField(value, "error")
  if (!isRecord(detail)) {
    return undefined
  }
  const code = stringField(detail, "code")
  const message = stringField(detail, "message")
  return code === undefined || message === undefined ? undefined : { code, message }
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json()
  } catch (error) {
    if (error instanceof SyntaxError || error instanceof TypeError) {
      return undefined
    }
    throw error
  }
}

async function requestJson<T>(
  path: string,
  init: RequestInit,
  parse: (value: unknown) => T,
): Promise<T> {
  const headers = new Headers(init.headers)
  headers.set("Accept", "application/json")
  if (init.body !== undefined) {
    headers.set("Content-Type", "application/json")
  }
  let response: Response
  try {
    response = await fetch(path, { ...init, credentials: "same-origin", headers })
  } catch (error) {
    if (isAbortError(error)) {
      throw error
    }
    throw new NetworkError("The session service is unavailable.", error)
  }
  const payload = await readJson(response)
  if (!response.ok) {
    const detail = parseErrorPayload(payload)
    const retryAfterValue = response.headers.get("Retry-After")
    const retryAfterSeconds = retryAfterValue === null ? undefined : Number(retryAfterValue)
    throw new HttpError(
      response.status,
      detail?.code ?? "http_error",
      detail?.message ?? `Request failed with HTTP ${response.status}.`,
      Number.isFinite(retryAfterSeconds) ? retryAfterSeconds : undefined,
    )
  }
  return parse(payload)
}

async function requestNoContent(
  path: string,
  signal: AbortSignal,
  initialHeaders?: HeadersInit,
): Promise<void> {
  const headers = new Headers(initialHeaders)
  headers.set("Accept", "application/json")
  let response: Response
  try {
    response = await fetch(path, {
      method: "DELETE",
      credentials: "same-origin",
      headers,
      signal,
    })
  } catch (error) {
    if (isAbortError(error)) {
      throw error
    }
    throw new NetworkError("The session service is unavailable.", error)
  }
  if (!response.ok) {
    const payload = await readJson(response)
    const detail = parseErrorPayload(payload)
    throw new HttpError(
      response.status,
      detail?.code ?? "http_error",
      detail?.message ?? `Request failed with HTTP ${response.status}.`,
    )
  }
}

async function requestBlob(path: string, signal: AbortSignal): Promise<CropResponse> {
  let response: Response
  try {
    response = await fetch(path, { credentials: "same-origin", signal })
  } catch (error) {
    if (isAbortError(error)) {
      throw error
    }
    throw new NetworkError("The session service is unavailable.", error)
  }
  if (!response.ok) {
    const payload = await readJson(response)
    const detail = parseErrorPayload(payload)
    throw new HttpError(
      response.status,
      detail?.code ?? "http_error",
      detail?.message ?? `Request failed with HTTP ${response.status}.`,
    )
  }
  const contentType = response.headers.get("Content-Type")
  if (contentType === null || contentType.split(";", 1)[0]?.trim() !== "image/jpeg") {
    throw new Error("crop response has an invalid content type")
  }
  const etag = response.headers.get("ETag")
  const versionValue = Number(response.headers.get("X-Representative-Version"))
  return {
    blob: await response.blob(),
    ...(etag === null ? {} : { etag }),
    ...(Number.isInteger(versionValue) && versionValue > 0
      ? { representative_version: versionValue }
      : {}),
  }
}

function pathSegment(value: string): string {
  return encodeURIComponent(value)
}

export class ApiClient {
  getSession(signal: AbortSignal): Promise<SessionResponse> {
    return requestJson("/api/session", { method: "GET", signal }, parseSession)
  }

  login(username: string, password: string, signal: AbortSignal): Promise<SessionResponse> {
    return requestJson(
      "/api/session",
      { method: "POST", body: JSON.stringify({ username, password }), signal },
      parseSession,
    )
  }

  activity(signal: AbortSignal): Promise<SessionResponse> {
    return requestJson("/api/session/activity", { method: "POST", signal }, parseSession)
  }

  listCameras(signal: AbortSignal): Promise<readonly CameraResponse[]> {
    return requestJson("/api/cameras", { method: "GET", signal }, parseCameras)
  }

  search(request: SearchRequest, signal: AbortSignal): Promise<SearchResponse> {
    return requestJson(
      "/api/search",
      { method: "POST", body: JSON.stringify(request), signal },
      parseSearchResponse,
    )
  }

  getAppearance(appearanceId: string, signal: AbortSignal): Promise<AppearanceResponse> {
    return requestJson(
      `/api/appearances/${pathSegment(appearanceId)}`,
      { method: "GET", signal },
      parseAppearanceResponse,
    )
  }

  getAppearanceCrop(appearanceId: string, signal: AbortSignal): Promise<CropResponse> {
    return requestBlob(`/api/appearances/${pathSegment(appearanceId)}/crop`, signal)
  }

  createCamera(request: CameraCreateRequest, signal: AbortSignal): Promise<CameraResponse> {
    return requestJson(
      "/api/cameras",
      { method: "POST", body: JSON.stringify(request), signal },
      parseCameraResponse,
    )
  }

  testCamera(request: CameraTestRequest, signal: AbortSignal): Promise<CameraTestResponse> {
    return requestJson(
      "/api/cameras/test",
      { method: "POST", body: JSON.stringify(request), signal },
      parseCameraTestResponse,
    )
  }

  updateCamera(
    cameraId: string,
    request: CameraPatchRequest,
    expectedVersion: number,
    signal: AbortSignal,
  ): Promise<CameraResponse> {
    return requestJson(
      `/api/cameras/${pathSegment(cameraId)}`,
      {
        method: "PATCH",
        body: JSON.stringify(request),
        headers: { "X-Camera-Version": String(expectedVersion) },
        signal,
      },
      parseCameraResponse,
    )
  }

  deleteCamera(cameraId: string, expectedVersion: number, signal: AbortSignal): Promise<void> {
    return requestNoContent(`/api/cameras/${pathSegment(cameraId)}`, signal, {
      "X-Camera-Version": String(expectedVersion),
    })
  }

  getSettings(signal: AbortSignal): Promise<SettingsResponse> {
    return requestJson("/api/settings", { method: "GET", signal }, parseSettingsResponse)
  }

  patchSettings(patch: SettingsPatch, signal: AbortSignal): Promise<SettingsResponse> {
    return requestJson(
      "/api/settings",
      { method: "PATCH", body: JSON.stringify(patch), signal },
      parseSettingsResponse,
    )
  }

  logout(signal: AbortSignal): Promise<void> {
    return requestNoContent("/api/session", signal)
  }
}

export const apiClient = new ApiClient()
