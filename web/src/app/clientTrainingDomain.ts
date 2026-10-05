import type {
  TrainingConfig,
  TrainingDatasetSnapshot,
  TrainingDatasetStatus,
  TrainingEvaluationSummary,
  TrainingJobPage,
  TrainingJobPhase,
  TrainingJobResponse,
  TrainingLogEntry,
  TrainingLogPage,
  TrainingMemoryRefusal,
  TrainingMetric,
  TrainingMetricPage,
  TrainingPreflightResponse,
  TrainingRetrievalScores,
  TrainingSplitCounts,
  TrainingSupervisorStatus,
} from "./clientTypes"

const CONFIG_BOUNDS = {
  epochs: [1, 100],
  learning_rate: [1e-7, 1e-3],
  micro_batch_size: [2, 128],
  weight_decay: [0, 0.2],
  gradient_accumulation: [1, 32],
  warmup_ratio: [0, 0.3],
  seed: [0, 2_147_483_647],
  gradient_clipping_norm: [0.1, 10],
} as const

const SHA256_PATTERN = /^[0-9a-f]{64}$/
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
const TIMESTAMP_ZONE_PATTERN = /(?:Z|[+-]\d{2}:\d{2})$/i

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function valueAt(record: Record<string, unknown>, key: string): unknown {
  return record[key]
}

function stringAt(record: Record<string, unknown>, key: string): string | undefined {
  const value = valueAt(record, key)
  return typeof value === "string" ? value : undefined
}

function nullableStringAt(record: Record<string, unknown>, key: string): string | null | undefined {
  const value = valueAt(record, key)
  if (value === null) return null
  return typeof value === "string" ? value : undefined
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined
}

function integerInRange(value: unknown, minimum: number, maximum: number): number | undefined {
  const number = finiteNumber(value)
  return number !== undefined && Number.isInteger(number) && number >= minimum && number <= maximum
    ? number
    : undefined
}

function nonNegativeInteger(value: unknown): number | undefined {
  return integerInRange(value, 0, Number.MAX_SAFE_INTEGER)
}

function numberInRange(value: unknown, minimum: number, maximum: number): number | undefined {
  const number = finiteNumber(value)
  return number !== undefined && number >= minimum && number <= maximum ? number : undefined
}

function nullableNumberInRange(
  record: Record<string, unknown>,
  key: string,
  minimum: number,
  maximum: number,
): number | null | undefined {
  const value = valueAt(record, key)
  if (value === null) return null
  return numberInRange(value, minimum, maximum)
}

function isTimestamp(value: string): boolean {
  return TIMESTAMP_ZONE_PATTERN.test(value) && Number.isFinite(Date.parse(value))
}

function requiredTimestamp(record: Record<string, unknown>, key: string): string | undefined {
  const value = stringAt(record, key)
  return value !== undefined && isTimestamp(value) ? value : undefined
}

function isSha256(value: string | undefined): value is string {
  return value !== undefined && SHA256_PATTERN.test(value)
}

function parseMixedPrecision(value: unknown): TrainingConfig["mixed_precision"] | undefined {
  if (value === "fp16" || value === "fp32") return value
  return undefined
}

function parseJobPhase(value: unknown): TrainingJobPhase | undefined {
  switch (value) {
    case "starting":
    case "training":
    case "evaluating":
    case "publishing":
    case "succeeded":
    case "cancelling":
    case "cancelled":
    case "failed":
    case "interrupted":
      return value
    default:
      return undefined
  }
}

function invalid(name: string): never {
  throw new Error(`${name} response has an invalid shape`)
}

export function parseTrainingConfig(value: unknown): TrainingConfig {
  if (!isRecord(value)) invalid("training config")
  const epochs = integerInRange(valueAt(value, "epochs"), ...CONFIG_BOUNDS.epochs)
  const learningRate = numberInRange(
    valueAt(value, "learning_rate"),
    ...CONFIG_BOUNDS.learning_rate,
  )
  const microBatchSize = integerInRange(
    valueAt(value, "micro_batch_size"),
    ...CONFIG_BOUNDS.micro_batch_size,
  )
  const weightDecay = numberInRange(valueAt(value, "weight_decay"), ...CONFIG_BOUNDS.weight_decay)
  const gradientAccumulation = integerInRange(
    valueAt(value, "gradient_accumulation"),
    ...CONFIG_BOUNDS.gradient_accumulation,
  )
  const warmupRatio = numberInRange(valueAt(value, "warmup_ratio"), ...CONFIG_BOUNDS.warmup_ratio)
  const seed = integerInRange(valueAt(value, "seed"), ...CONFIG_BOUNDS.seed)
  const earlyStopping = valueAt(value, "early_stopping_patience")
  const earlyStoppingPatience = earlyStopping === null ? null : integerInRange(earlyStopping, 1, 20)
  const gradientClippingNorm = numberInRange(
    valueAt(value, "gradient_clipping_norm"),
    ...CONFIG_BOUNDS.gradient_clipping_norm,
  )
  const mixedPrecision = parseMixedPrecision(valueAt(value, "mixed_precision"))
  const gradientCheckpointing = valueAt(value, "gradient_checkpointing")
  if (
    epochs === undefined ||
    learningRate === undefined ||
    microBatchSize === undefined ||
    weightDecay === undefined ||
    gradientAccumulation === undefined ||
    warmupRatio === undefined ||
    seed === undefined ||
    earlyStoppingPatience === undefined ||
    gradientClippingNorm === undefined ||
    mixedPrecision === undefined ||
    typeof gradientCheckpointing !== "boolean"
  ) {
    invalid("training config")
  }
  return {
    epochs,
    learning_rate: learningRate,
    micro_batch_size: microBatchSize,
    weight_decay: weightDecay,
    gradient_accumulation: gradientAccumulation,
    warmup_ratio: warmupRatio,
    seed,
    early_stopping_patience: earlyStoppingPatience,
    gradient_clipping_norm: gradientClippingNorm,
    mixed_precision: mixedPrecision,
    gradient_checkpointing: gradientCheckpointing,
  }
}

function parseSplitCounts(value: unknown): TrainingSplitCounts {
  if (!isRecord(value)) invalid("training dataset split counts")
  const images = nonNegativeInteger(valueAt(value, "images"))
  const captions = nonNegativeInteger(valueAt(value, "captions"))
  const identities = nonNegativeInteger(valueAt(value, "identities"))
  if (images === undefined || captions === undefined || identities === undefined) {
    invalid("training dataset split counts")
  }
  return { images, captions, identities }
}

function parseDatasetSnapshot(value: unknown): TrainingDatasetSnapshot {
  if (!isRecord(value)) invalid("training dataset")
  const datasetId = valueAt(value, "dataset_id")
  const fingerprint = stringAt(value, "fingerprint")
  const protocol = stringAt(value, "protocol")
  const counts = valueAt(value, "split_counts")
  const imageCount = nonNegativeInteger(valueAt(value, "image_count"))
  const captionCount = nonNegativeInteger(valueAt(value, "caption_count"))
  const identityCount = nonNegativeInteger(valueAt(value, "identity_count"))
  if (
    datasetId !== "cuhk-pedes" ||
    !isSha256(fingerprint) ||
    protocol === undefined ||
    protocol.length === 0 ||
    protocol.length > 96 ||
    !isRecord(counts) ||
    imageCount === undefined ||
    captionCount === undefined ||
    identityCount === undefined
  ) {
    invalid("training dataset")
  }
  const train = parseSplitCounts(valueAt(counts, "train"))
  const val = parseSplitCounts(valueAt(counts, "val"))
  const test = parseSplitCounts(valueAt(counts, "test"))
  return {
    dataset_id: "cuhk-pedes",
    fingerprint,
    protocol,
    split_counts: { train, val, test },
    image_count: imageCount,
    caption_count: captionCount,
    identity_count: identityCount,
  }
}

export function parseTrainingDatasetStatus(value: unknown): TrainingDatasetStatus {
  if (!isRecord(value)) invalid("training dataset status")
  const registered = valueAt(value, "registered")
  const valid = valueAt(value, "valid")
  const reason = nullableStringAt(value, "reason")
  const snapshotValue = valueAt(value, "snapshot")
  const supervisorValue = valueAt(value, "supervisor")
  if (
    typeof registered !== "boolean" ||
    typeof valid !== "boolean" ||
    reason === undefined ||
    (snapshotValue !== null && !isRecord(snapshotValue)) ||
    !isRecord(supervisorValue)
  ) {
    invalid("training dataset status")
  }
  const snapshot = snapshotValue === null ? null : parseDatasetSnapshot(snapshotValue)
  if (valid && (!registered || snapshot === null)) invalid("training dataset status")
  const supervisorState = valueAt(supervisorValue, "state")
  const supervisorReason = nullableStringAt(supervisorValue, "reason")
  const supervisorObservedAt = nullableStringAt(supervisorValue, "observed_at")
  const supervisorSourceFingerprint = nullableStringAt(supervisorValue, "source_fingerprint")
  const supervisorDatasetFingerprint = nullableStringAt(supervisorValue, "dataset_fingerprint")
  if (
    (supervisorState !== "validating" &&
      supervisorState !== "ready" &&
      supervisorState !== "unavailable") ||
    supervisorReason === undefined ||
    supervisorObservedAt === undefined ||
    (supervisorObservedAt !== null && !isTimestamp(supervisorObservedAt)) ||
    supervisorSourceFingerprint === undefined ||
    (supervisorSourceFingerprint !== null && !isSha256(supervisorSourceFingerprint)) ||
    supervisorDatasetFingerprint === undefined ||
    (supervisorDatasetFingerprint !== null && !isSha256(supervisorDatasetFingerprint))
  ) {
    invalid("training supervisor status")
  }
  const supervisor: TrainingSupervisorStatus = {
    state: supervisorState,
    reason: supervisorReason,
    observed_at: supervisorObservedAt,
    source_fingerprint: supervisorSourceFingerprint,
    dataset_fingerprint: supervisorDatasetFingerprint,
  }
  return { registered, valid, reason, snapshot, supervisor }
}

export function parseTrainingPreflightResponse(value: unknown): TrainingPreflightResponse {
  if (!isRecord(value)) invalid("training preflight")
  const admitted = valueAt(value, "admitted")
  const trainingPeakBytes = nonNegativeInteger(valueAt(value, "training_peak_bytes"))
  const reserveBytes = nonNegativeInteger(valueAt(value, "reserve_bytes"))
  const requiredBytes = nonNegativeInteger(valueAt(value, "required_bytes"))
  const freeBytes = nonNegativeInteger(valueAt(value, "free_bytes"))
  const profileIdentity = stringAt(value, "profile_identity")
  const observedAt = requiredTimestamp(value, "observed_at")
  const reason = valueAt(value, "reason")
  if (
    typeof admitted !== "boolean" ||
    trainingPeakBytes === undefined ||
    reserveBytes === undefined ||
    requiredBytes === undefined ||
    freeBytes === undefined ||
    !isSha256(profileIdentity) ||
    observedAt === undefined ||
    (reason !== "admitted" && reason !== "insufficient_free_memory") ||
    admitted !== (reason === "admitted")
  ) {
    invalid("training preflight")
  }
  return {
    admitted,
    training_peak_bytes: trainingPeakBytes,
    reserve_bytes: reserveBytes,
    required_bytes: requiredBytes,
    free_bytes: freeBytes,
    profile_identity: profileIdentity,
    observed_at: observedAt,
    reason,
  }
}

export function parseTrainingMemoryRefusal(value: unknown): TrainingMemoryRefusal | null {
  if (!isRecord(value)) return null
  const code = stringAt(value, "code")
  const message = stringAt(value, "message")
  const reason = stringAt(value, "reason")
  const trainingPeakBytes = nonNegativeInteger(valueAt(value, "training_peak_bytes"))
  const reserveBytes = nonNegativeInteger(valueAt(value, "reserve_bytes"))
  const requiredBytes = nonNegativeInteger(valueAt(value, "required_bytes"))
  const freeBytes = nonNegativeInteger(valueAt(value, "free_bytes"))
  const observedAtValue = valueAt(value, "observed_at")
  const profileIdentityValue = valueAt(value, "profile_identity")
  const observedAt =
    observedAtValue === undefined ? undefined : requiredTimestamp(value, "observed_at")
  const profileIdentity =
    profileIdentityValue === undefined
      ? undefined
      : typeof profileIdentityValue === "string" && isSha256(profileIdentityValue)
        ? profileIdentityValue
        : null
  if (
    code === undefined ||
    message === undefined ||
    reason === undefined ||
    trainingPeakBytes === undefined ||
    reserveBytes === undefined ||
    requiredBytes === undefined ||
    freeBytes === undefined ||
    (observedAtValue !== undefined && observedAt === undefined) ||
    profileIdentity === null
  ) {
    return null
  }
  return {
    code,
    message,
    reason,
    training_peak_bytes: trainingPeakBytes,
    reserve_bytes: reserveBytes,
    required_bytes: requiredBytes,
    free_bytes: freeBytes,
    ...(observedAt === undefined ? {} : { observed_at: observedAt }),
    ...(profileIdentity === undefined ? {} : { profile_identity: profileIdentity }),
  }
}

function parseRetrievalScores(value: unknown): TrainingRetrievalScores {
  if (!isRecord(value)) invalid("training evaluation")
  const recallAt1 = numberInRange(valueAt(value, "recall_at_1"), 0, 1)
  const recallAt5 = numberInRange(valueAt(value, "recall_at_5"), 0, 1)
  const recallAt10 = numberInRange(valueAt(value, "recall_at_10"), 0, 1)
  if (recallAt1 === undefined || recallAt5 === undefined || recallAt10 === undefined) {
    invalid("training evaluation")
  }
  return { recall_at_1: recallAt1, recall_at_5: recallAt5, recall_at_10: recallAt10 }
}

function parseEvaluationSummary(value: unknown): TrainingEvaluationSummary {
  if (!isRecord(value)) invalid("training evaluation")
  const datasetSha256 = stringAt(value, "dataset_sha256")
  const datasetSplit = valueAt(value, "dataset_split")
  const protocol = stringAt(value, "protocol")
  const baselineModelId = stringAt(value, "baseline_model_id")
  const baselineRevision = stringAt(value, "baseline_revision")
  const baselinePackageSha256 = stringAt(value, "baseline_package_sha256")
  const trainingSourceFingerprint = stringAt(value, "training_source_fingerprint")
  const evaluationCodeRevision = stringAt(value, "evaluation_code_revision")
  const metricDefinition = stringAt(value, "metric_definition")
  const bestValidationEpoch = integerInRange(valueAt(value, "best_validation_epoch"), 1, 100_000)
  const candidateWeightsSha256 = stringAt(value, "candidate_weights_sha256")
  const packageSha256 = stringAt(value, "package_sha256")
  if (
    !isSha256(datasetSha256) ||
    datasetSplit !== "test" ||
    protocol === undefined ||
    protocol.length === 0 ||
    protocol.length > 96 ||
    baselineModelId === undefined ||
    baselineModelId.length === 0 ||
    baselineModelId.length > 128 ||
    baselineRevision === undefined ||
    baselineRevision.length === 0 ||
    baselineRevision.length > 128 ||
    !isSha256(baselinePackageSha256) ||
    !isSha256(trainingSourceFingerprint) ||
    !isSha256(evaluationCodeRevision) ||
    metricDefinition === undefined ||
    metricDefinition.length === 0 ||
    metricDefinition.length > 128 ||
    bestValidationEpoch === undefined ||
    !isSha256(candidateWeightsSha256) ||
    !isSha256(packageSha256)
  ) {
    invalid("training evaluation")
  }
  return {
    dataset_sha256: datasetSha256,
    dataset_split: "test",
    protocol,
    baseline_model_id: baselineModelId,
    baseline_revision: baselineRevision,
    baseline_package_sha256: baselinePackageSha256,
    training_source_fingerprint: trainingSourceFingerprint,
    evaluation_code_revision: evaluationCodeRevision,
    metric_definition: metricDefinition,
    best_validation_epoch: bestValidationEpoch,
    baseline: parseRetrievalScores(valueAt(value, "baseline")),
    candidate: parseRetrievalScores(valueAt(value, "candidate")),
    candidate_weights_sha256: candidateWeightsSha256,
    package_sha256: packageSha256,
  }
}

function nullableRequiredString(value: unknown): string | null | undefined {
  if (value === null) return null
  return typeof value === "string" && value.length > 0 ? value : undefined
}

export function parseTrainingJobResponse(value: unknown): TrainingJobResponse {
  if (!isRecord(value)) invalid("training job")
  const id = stringAt(value, "id")
  const requestId = stringAt(value, "request_id")
  const phase = parseJobPhase(valueAt(value, "phase"))
  const currentEpoch = nonNegativeInteger(valueAt(value, "current_epoch"))
  const currentStep = nonNegativeInteger(valueAt(value, "current_step"))
  const ownerGeneration = nonNegativeInteger(valueAt(value, "owner_generation"))
  const cancelRequested = valueAt(value, "cancel_requested")
  const attempts = nonNegativeInteger(valueAt(value, "attempts"))
  const bestMetric = nullableNumberInRange(value, "best_metric", 0, 1)
  const candidateModelId = nullableRequiredString(valueAt(value, "candidate_model_id"))
  const candidateRevision = nullableRequiredString(valueAt(value, "candidate_revision"))
  const evaluationValue = valueAt(value, "evaluation")
  const error = nullableStringAt(value, "error")
  const createdAt = requiredTimestamp(value, "created_at")
  const updatedAt = requiredTimestamp(value, "updated_at")
  const finishedAtValue = valueAt(value, "finished_at")
  const finishedAt = finishedAtValue === null ? null : requiredTimestamp(value, "finished_at")
  if (
    id === undefined ||
    !UUID_PATTERN.test(id) ||
    requestId === undefined ||
    !UUID_PATTERN.test(requestId) ||
    phase === undefined ||
    currentEpoch === undefined ||
    currentStep === undefined ||
    ownerGeneration === undefined ||
    typeof cancelRequested !== "boolean" ||
    attempts === undefined ||
    bestMetric === undefined ||
    candidateModelId === undefined ||
    candidateRevision === undefined ||
    (evaluationValue !== null && !isRecord(evaluationValue)) ||
    error === undefined ||
    createdAt === undefined ||
    updatedAt === undefined ||
    (finishedAtValue !== null && finishedAt === undefined)
  ) {
    invalid("training job")
  }
  return {
    id,
    request_id: requestId,
    phase,
    config: parseTrainingConfig(valueAt(value, "config")),
    dataset: parseDatasetSnapshot(valueAt(value, "dataset")),
    current_epoch: currentEpoch,
    current_step: currentStep,
    owner_generation: ownerGeneration,
    cancel_requested: cancelRequested,
    attempts,
    best_metric: bestMetric,
    candidate_model_id: candidateModelId,
    candidate_revision: candidateRevision,
    evaluation: evaluationValue === null ? null : parseEvaluationSummary(evaluationValue),
    error,
    created_at: createdAt,
    updated_at: updatedAt,
    finished_at: finishedAt ?? null,
  }
}

function parseCursor(record: Record<string, unknown>): string | null | undefined {
  const cursor = nullableStringAt(record, "next_cursor")
  return cursor !== undefined && (cursor === null || (cursor.length > 0 && cursor.length <= 160))
    ? cursor
    : undefined
}

export function parseTrainingJobPage(value: unknown): TrainingJobPage {
  if (!isRecord(value)) invalid("training job page")
  const items = valueAt(value, "items")
  const nextCursor = parseCursor(value)
  if (!Array.isArray(items) || nextCursor === undefined) invalid("training job page")
  return { items: items.map(parseTrainingJobResponse), next_cursor: nextCursor }
}

function parseMetric(value: unknown): TrainingMetric {
  if (!isRecord(value)) invalid("training metric")
  const epoch = nonNegativeInteger(valueAt(value, "epoch"))
  const step = nonNegativeInteger(valueAt(value, "step"))
  const trainingLoss = nullableNumberInRange(
    value,
    "training_loss",
    Number.NEGATIVE_INFINITY,
    Number.POSITIVE_INFINITY,
  )
  const validationRecall = nullableNumberInRange(value, "validation_recall_at_1", 0, 1)
  const allocatedBytes =
    valueAt(value, "allocated_bytes") === null
      ? null
      : nonNegativeInteger(valueAt(value, "allocated_bytes"))
  const reservedBytes =
    valueAt(value, "reserved_bytes") === null
      ? null
      : nonNegativeInteger(valueAt(value, "reserved_bytes"))
  const observedAt = requiredTimestamp(value, "observed_at")
  if (
    epoch === undefined ||
    step === undefined ||
    trainingLoss === undefined ||
    validationRecall === undefined ||
    allocatedBytes === undefined ||
    reservedBytes === undefined ||
    observedAt === undefined
  ) {
    invalid("training metric")
  }
  return {
    epoch,
    step,
    training_loss: trainingLoss,
    validation_recall_at_1: validationRecall,
    allocated_bytes: allocatedBytes,
    reserved_bytes: reservedBytes,
    observed_at: observedAt,
  }
}

function parseMetricPage(value: unknown): TrainingMetricPage {
  if (!isRecord(value)) invalid("training metric page")
  const items = valueAt(value, "items")
  const nextCursor = parseCursor(value)
  if (!Array.isArray(items) || nextCursor === undefined) invalid("training metric page")
  return { items: items.map(parseMetric), next_cursor: nextCursor }
}

function parseLogLevel(value: unknown): TrainingLogEntry["level"] | undefined {
  if (value === "debug" || value === "info" || value === "warning" || value === "error") {
    return value
  }
  return undefined
}

function parseLogEntry(value: unknown): TrainingLogEntry {
  if (!isRecord(value)) invalid("training log")
  const cursor = stringAt(value, "cursor")
  const level = parseLogLevel(valueAt(value, "level"))
  const message = stringAt(value, "message")
  const observedAt = requiredTimestamp(value, "observed_at")
  if (
    cursor === undefined ||
    cursor.length === 0 ||
    cursor.length > 160 ||
    level === undefined ||
    message === undefined ||
    message.length > 500 ||
    observedAt === undefined
  ) {
    invalid("training log")
  }
  return { cursor, level, message, observed_at: observedAt }
}

export function parseTrainingLogPage(value: unknown): TrainingLogPage {
  if (!isRecord(value)) invalid("training log page")
  const items = valueAt(value, "items")
  const nextCursor = parseCursor(value)
  if (!Array.isArray(items) || nextCursor === undefined) invalid("training log page")
  return { items: items.map(parseLogEntry), next_cursor: nextCursor }
}

export function parseTrainingMetricPageResponse(value: unknown): TrainingMetricPage {
  return parseMetricPage(value)
}
