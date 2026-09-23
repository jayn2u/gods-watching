import type {
  CameraResponse,
  ClipModelOption,
  ClipModelTransition,
  ModelSettingsResponse,
  ModelTransitionPhase,
  SettingsResponse,
  WallSlotIds,
} from "./clientTypes"

export type SearchFilters = Readonly<{
  camera_ids?: readonly string[]
  from?: string
  to?: string
  limit?: number
}>

export type TextSearchRequest = SearchFilters &
  Readonly<{
    mode: "text"
    query: string
    sort?: "similarity"
  }>

export type SimilarSearchRequest = SearchFilters &
  Readonly<{
    mode: "similar"
    appearance_id: string
    sort?: "similarity"
  }>

export type BrowseSearchRequest = SearchFilters &
  Readonly<{
    mode: "browse"
    sort?: "newest"
  }>

export type SearchRequest = TextSearchRequest | SimilarSearchRequest | BrowseSearchRequest

export type BoundingBox = Readonly<{
  x_min: number
  y_min: number
  x_max: number
  y_max: number
}>

export type AppearanceResponse = Readonly<{
  appearance_id: string
  camera_id: string
  camera_name: string
  session_id: string
  track_id: number
  first_seen: string
  last_seen: string
  ended_at: string | null
  representative_version: number
  bounding_box: BoundingBox
  source_width: number
  source_height: number
  detector_confidence: number
  crop_quality: number
  model_id: string
  model_revision: string
  similarity: number | null
}>

export type SearchResponse = Readonly<{
  mode: "text" | "similar" | "browse"
  results: readonly AppearanceResponse[]
}>

export type LiveDetectionBox = Readonly<{
  x1: number
  y1: number
  x2: number
  y2: number
  confidence: number
}>

export type LiveDetectionsResponse = Readonly<{
  camera_id: string
  camera_session_id: string | null
  frame_at: string | null
  frame_age_seconds: number | null
  width: number | null
  height: number | null
  boxes: readonly LiveDetectionBox[]
}>

export type CameraCreateRequest = Readonly<{
  name: string
  source_url: string
  detection_enabled?: boolean
  detection_threshold?: number
}>

export type CameraPatchRequest = Readonly<{
  name?: string
  source_url?: string
  detection_enabled?: boolean
  detection_threshold?: number
}>

export type CameraTestRequest = Readonly<{
  source_url: string
}>

export type CameraTestResponse = Readonly<{
  source_host: string
  source_port: number | null
  codec: string
  width: number
  height: number
}>

export type CropResponse = Readonly<{
  blob: Blob
  etag?: string
  representative_version?: number
}>

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

function numberField(record: Record<string, unknown>, key: string): number | undefined {
  const value = record[key]
  return typeof value === "number" && Number.isFinite(value) ? value : undefined
}

function integerField(record: Record<string, unknown>, key: string): number | undefined {
  const value = numberField(record, key)
  return value !== undefined && Number.isInteger(value) ? value : undefined
}

function boundedNumberField(
  record: Record<string, unknown>,
  key: string,
  minimum: number,
  maximum: number,
): number | undefined {
  const value = numberField(record, key)
  return value !== undefined && value >= minimum && value <= maximum ? value : undefined
}

function positiveIntegerField(record: Record<string, unknown>, key: string): number | undefined {
  const value = integerField(record, key)
  return value !== undefined && value > 0 ? value : undefined
}

function nonNegativeIntegerField(record: Record<string, unknown>, key: string): number | undefined {
  const value = integerField(record, key)
  return value !== undefined && value >= 0 ? value : undefined
}

function nullableNumberField(
  record: Record<string, unknown>,
  key: string,
  minimum = Number.NEGATIVE_INFINITY,
): number | null | undefined {
  const value = unknownField(record, key)
  if (value === null) {
    return null
  }
  return typeof value === "number" && Number.isFinite(value) && value >= minimum ? value : undefined
}

function nullablePositiveIntegerField(
  record: Record<string, unknown>,
  key: string,
): number | null | undefined {
  const value = nullableNumberField(record, key)
  return value === null || (value !== undefined && Number.isInteger(value) && value > 0)
    ? value
    : undefined
}

function parseBoundingBox(value: unknown): BoundingBox {
  if (!isRecord(value)) {
    throw new Error("appearance response has an invalid bounding box")
  }
  const xMin = integerField(value, "x_min")
  const yMin = integerField(value, "y_min")
  const xMax = positiveIntegerField(value, "x_max")
  const yMax = positiveIntegerField(value, "y_max")
  if (xMin === undefined || yMin === undefined || xMax === undefined || yMax === undefined) {
    throw new Error("appearance response has an invalid bounding box")
  }
  if (xMin < 0 || yMin < 0) {
    throw new Error("appearance response has an invalid bounding box")
  }
  return { x_min: xMin, y_min: yMin, x_max: xMax, y_max: yMax }
}

export function parseCameraResponse(value: unknown): CameraResponse {
  if (!isRecord(value)) {
    throw new Error("camera response is not an object")
  }
  const cameraId = stringField(value, "camera_id")
  const name = stringField(value, "name")
  const sourceHost = stringField(value, "source_host")
  const sourcePort = unknownField(value, "source_port")
  const detectionEnabled = unknownField(value, "detection_enabled")
  const detectionThreshold = boundedNumberField(value, "detection_threshold", 0.1, 0.95)
  const version = positiveIntegerField(value, "version")
  const deletedAt = nullableStringField(value, "deleted_at")
  if (
    cameraId === undefined ||
    name === undefined ||
    sourceHost === undefined ||
    (sourcePort !== null && (typeof sourcePort !== "number" || !Number.isInteger(sourcePort))) ||
    typeof detectionEnabled !== "boolean" ||
    detectionThreshold === undefined ||
    version === undefined ||
    deletedAt === undefined
  ) {
    throw new Error("camera response has an invalid shape")
  }
  return {
    camera_id: cameraId,
    name,
    source_host: sourceHost,
    source_port: sourcePort,
    detection_enabled: detectionEnabled,
    detection_threshold: detectionThreshold,
    version,
    deleted_at: deletedAt,
  }
}

export function parseSettingsResponse(value: unknown): SettingsResponse {
  if (!isRecord(value)) {
    throw new Error("settings response is not an object")
  }
  const retentionDays = positiveIntegerField(value, "retention_days")
  const quotaBytes = positiveIntegerField(value, "quota_bytes")
  const slots = unknownField(value, "wall_slot_ids")
  if (
    retentionDays === undefined ||
    quotaBytes === undefined ||
    !Array.isArray(slots) ||
    slots.length !== 4 ||
    !slots.every((slot) => slot === null || typeof slot === "string")
  ) {
    throw new Error("settings response has an invalid shape")
  }
  const wallSlotIds: WallSlotIds = [
    slots[0] ?? null,
    slots[1] ?? null,
    slots[2] ?? null,
    slots[3] ?? null,
  ]
  return { retention_days: retentionDays, quota_bytes: quotaBytes, wall_slot_ids: wallSlotIds }
}

const MODEL_TRANSITION_PHASES: readonly ModelTransitionPhase[] = [
  "queued",
  "preparing",
  "reindexing",
  "activating",
  "rolling_back",
  "succeeded",
  "failed",
]

function parseModelOption(value: unknown): ClipModelOption {
  if (!isRecord(value)) {
    throw new Error("model settings response has an invalid model")
  }
  const modelId = stringField(value, "model_id")
  const displayName = stringField(value, "display_name")
  const dimension = positiveIntegerField(value, "dimension")
  const prepared = unknownField(value, "prepared")
  const reason = nullableStringField(value, "reason")
  if (
    modelId === undefined ||
    modelId.length === 0 ||
    displayName === undefined ||
    displayName.length === 0 ||
    dimension === undefined ||
    typeof prepared !== "boolean" ||
    reason === undefined
  ) {
    throw new Error("model settings response has an invalid model")
  }
  return {
    model_id: modelId,
    display_name: displayName,
    dimension,
    prepared,
    reason,
  }
}

function parseSkipReasons(value: unknown): Readonly<Record<string, number>> {
  if (!isRecord(value)) {
    throw new Error("model settings response has an invalid transition")
  }
  const reasons: Record<string, number> = {}
  for (const [reason, count] of Object.entries(value)) {
    if (reason.length === 0 || typeof count !== "number" || !Number.isInteger(count) || count < 0) {
      throw new Error("model settings response has an invalid transition")
    }
    reasons[reason] = count
  }
  return reasons
}

function parseModelTransition(value: unknown): ClipModelTransition {
  if (!isRecord(value)) {
    throw new Error("model settings response has an invalid transition")
  }
  const id = stringField(value, "id")
  const sourceModelId = stringField(value, "source_model_id")
  const targetModelId = stringField(value, "target_model_id")
  const phase = unknownField(value, "phase")
  const processed = nonNegativeIntegerField(value, "processed")
  const total = nonNegativeIntegerField(value, "total")
  const skipped = nonNegativeIntegerField(value, "skipped")
  const error = nullableStringField(value, "error")
  if (
    id === undefined ||
    id.length === 0 ||
    sourceModelId === undefined ||
    sourceModelId.length === 0 ||
    targetModelId === undefined ||
    targetModelId.length === 0 ||
    !MODEL_TRANSITION_PHASES.includes(phase as ModelTransitionPhase) ||
    processed === undefined ||
    total === undefined ||
    skipped === undefined ||
    processed > total ||
    skipped > total ||
    error === undefined
  ) {
    throw new Error("model settings response has an invalid transition")
  }
  return {
    id,
    source_model_id: sourceModelId,
    target_model_id: targetModelId,
    phase: phase as ModelTransitionPhase,
    processed,
    total,
    skipped,
    skip_reasons: parseSkipReasons(unknownField(value, "skip_reasons")),
    error,
  }
}

export function parseModelSettingsResponse(value: unknown): ModelSettingsResponse {
  if (!isRecord(value)) {
    throw new Error("model settings response is not an object")
  }
  const activeModelId = stringField(value, "active_model_id")
  const maintenance = unknownField(value, "maintenance")
  const models = unknownField(value, "models")
  const transitionValue = unknownField(value, "transition")
  if (
    activeModelId === undefined ||
    activeModelId.length === 0 ||
    typeof maintenance !== "boolean" ||
    !Array.isArray(models) ||
    (transitionValue !== null && !isRecord(transitionValue))
  ) {
    throw new Error("model settings response has an invalid shape")
  }
  return {
    active_model_id: activeModelId,
    maintenance,
    models: models.map(parseModelOption),
    transition: transitionValue === null ? null : parseModelTransition(transitionValue),
  }
}

export function parseAppearanceResponse(value: unknown): AppearanceResponse {
  if (!isRecord(value)) {
    throw new Error("appearance response is not an object")
  }
  const appearanceId = stringField(value, "appearance_id")
  const cameraId = stringField(value, "camera_id")
  const cameraName = stringField(value, "camera_name")
  const sessionId = stringField(value, "session_id")
  const trackId = integerField(value, "track_id")
  const firstSeen = stringField(value, "first_seen")
  const lastSeen = stringField(value, "last_seen")
  const endedAt = nullableStringField(value, "ended_at")
  const representativeVersion = positiveIntegerField(value, "representative_version")
  const sourceWidth = positiveIntegerField(value, "source_width")
  const sourceHeight = positiveIntegerField(value, "source_height")
  const detectorConfidence = boundedNumberField(value, "detector_confidence", 0, 1)
  const cropQuality = boundedNumberField(value, "crop_quality", 0, Number.POSITIVE_INFINITY)
  const modelId = stringField(value, "model_id")
  const modelRevision = stringField(value, "model_revision")
  const similarity = unknownField(value, "similarity")
  if (
    appearanceId === undefined ||
    cameraId === undefined ||
    cameraName === undefined ||
    sessionId === undefined ||
    trackId === undefined ||
    firstSeen === undefined ||
    lastSeen === undefined ||
    endedAt === undefined ||
    representativeVersion === undefined ||
    sourceWidth === undefined ||
    sourceHeight === undefined ||
    detectorConfidence === undefined ||
    cropQuality === undefined ||
    modelId === undefined ||
    modelRevision === undefined ||
    (similarity !== null &&
      (typeof similarity !== "number" ||
        !Number.isFinite(similarity) ||
        similarity < -1 ||
        similarity > 1))
  ) {
    throw new Error("appearance response has an invalid shape")
  }
  return {
    appearance_id: appearanceId,
    camera_id: cameraId,
    camera_name: cameraName,
    session_id: sessionId,
    track_id: trackId,
    first_seen: firstSeen,
    last_seen: lastSeen,
    ended_at: endedAt,
    representative_version: representativeVersion,
    bounding_box: parseBoundingBox(unknownField(value, "bounding_box")),
    source_width: sourceWidth,
    source_height: sourceHeight,
    detector_confidence: detectorConfidence,
    crop_quality: cropQuality,
    model_id: modelId,
    model_revision: modelRevision,
    similarity: similarity,
  }
}

export function parseSearchResponse(value: unknown): SearchResponse {
  if (!isRecord(value)) {
    throw new Error("search response is not an object")
  }
  const mode = unknownField(value, "mode")
  const results = unknownField(value, "results")
  if ((mode !== "text" && mode !== "similar" && mode !== "browse") || !Array.isArray(results)) {
    throw new Error("search response has an invalid shape")
  }
  return { mode, results: results.map(parseAppearanceResponse) }
}

export function parseCameraTestResponse(value: unknown): CameraTestResponse {
  if (!isRecord(value)) {
    throw new Error("camera test response is not an object")
  }
  const sourceHost = stringField(value, "source_host")
  const sourcePort = unknownField(value, "source_port")
  const codec = stringField(value, "codec")
  const width = numberField(value, "width")
  const height = numberField(value, "height")
  if (
    sourceHost === undefined ||
    (sourcePort !== null && (typeof sourcePort !== "number" || !Number.isInteger(sourcePort))) ||
    codec === undefined ||
    width === undefined ||
    !Number.isInteger(width) ||
    width <= 0 ||
    height === undefined ||
    !Number.isInteger(height) ||
    height <= 0
  ) {
    throw new Error("camera test response has an invalid shape")
  }
  return { source_host: sourceHost, source_port: sourcePort, codec, width, height }
}

function parseLiveDetectionBox(value: unknown): LiveDetectionBox {
  if (!isRecord(value)) {
    throw new Error("live detection response has an invalid shape")
  }
  const x1 = numberField(value, "x1")
  const y1 = numberField(value, "y1")
  const x2 = numberField(value, "x2")
  const y2 = numberField(value, "y2")
  const confidence = boundedNumberField(value, "confidence", 0, 1)
  if (
    x1 === undefined ||
    y1 === undefined ||
    x2 === undefined ||
    y2 === undefined ||
    confidence === undefined ||
    x1 < 0 ||
    y1 < 0 ||
    x2 <= x1 ||
    y2 <= y1
  ) {
    throw new Error("live detection response has an invalid shape")
  }
  return { x1, y1, x2, y2, confidence }
}

export function parseLiveDetectionsResponse(value: unknown): LiveDetectionsResponse {
  if (!isRecord(value)) {
    throw new Error("live detection response has an invalid shape")
  }
  const cameraId = stringField(value, "camera_id")
  const cameraSessionId = nullableStringField(value, "camera_session_id")
  const frameAt = nullableStringField(value, "frame_at")
  const frameAgeSeconds = nullableNumberField(value, "frame_age_seconds", 0)
  const width = nullablePositiveIntegerField(value, "width")
  const height = nullablePositiveIntegerField(value, "height")
  const boxes = unknownField(value, "boxes")
  if (
    cameraId === undefined ||
    cameraId.length === 0 ||
    cameraSessionId === undefined ||
    frameAt === undefined ||
    frameAgeSeconds === undefined ||
    width === undefined ||
    height === undefined ||
    (width === null) !== (height === null) ||
    !Array.isArray(boxes)
  ) {
    throw new Error("live detection response has an invalid shape")
  }
  return {
    camera_id: cameraId,
    camera_session_id: cameraSessionId,
    frame_at: frameAt,
    frame_age_seconds: frameAgeSeconds,
    width,
    height,
    boxes: boxes.map(parseLiveDetectionBox),
  }
}
