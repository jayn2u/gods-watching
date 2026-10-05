export type CameraResponse = Readonly<{
  camera_id: string
  name: string
  source_host: string
  source_port: number | null
  detection_enabled: boolean
  detection_threshold: number
  version: number
  deleted_at: string | null
}>

export type WallSlotIds = readonly [string | null, string | null, string | null, string | null]

export type SettingsResponse = Readonly<{
  retention_days: number
  quota_bytes: number
  wall_slot_ids: WallSlotIds
}>

export type SettingsPatch = Readonly<{
  retention_days?: number
  quota_bytes?: number
  wall_slot_ids?: WallSlotIds
}>

export type ModelTransitionPhase =
  | "queued"
  | "preparing"
  | "reindexing"
  | "activating"
  | "rolling_back"
  | "succeeded"
  | "failed"

export type ClipModelOption = Readonly<{
  model_id: string
  display_name: string
  revision: string
  dimension: number
  prepared: boolean
  reason: string | null
  quality_passed: boolean
  quality_reason: string | null
}>

export type SwitchPreflight = Readonly<{
  target_model_id: string
  retained_count: number
  estimated_missing_count: number
  measured_crops_per_second: number | null
  measured_fixed_seconds: number | null
  estimated_seconds: number | null
  max_seconds: number
  eligible: boolean
  reason: string | null
}>

export type ClipModelTransition = Readonly<{
  id: string
  source_model_id: string
  target_model_id: string
  phase: ModelTransitionPhase
  processed: number
  total: number
  skipped: number
  skip_reasons: Readonly<Record<string, number>>
  error: string | null
}>

export type ModelSettingsResponse = Readonly<{
  active_model_id: string
  maintenance: boolean
  models: readonly ClipModelOption[]
  transition: ClipModelTransition | null
}>

export type TrainingConfig = Readonly<{
  epochs: number
  learning_rate: number
  micro_batch_size: number
  weight_decay: number
  gradient_accumulation: number
  warmup_ratio: number
  seed: number
  early_stopping_patience: number | null
  gradient_clipping_norm: number
  mixed_precision: "fp16" | "fp32"
  gradient_checkpointing: boolean
}>

export type TrainingSplitCounts = Readonly<{
  images: number
  captions: number
  identities: number
}>

export type TrainingDatasetSnapshot = Readonly<{
  dataset_id: "cuhk-pedes"
  fingerprint: string
  protocol: string
  split_counts: Readonly<Record<"train" | "val" | "test", TrainingSplitCounts>>
  image_count: number
  caption_count: number
  identity_count: number
}>

export type TrainingSupervisorStatus = Readonly<{
  state: "validating" | "ready" | "unavailable"
  reason: string | null
  observed_at: string | null
  source_fingerprint: string | null
  dataset_fingerprint: string | null
}>

export type TrainingDatasetStatus = Readonly<{
  registered: boolean
  valid: boolean
  reason: string | null
  snapshot: TrainingDatasetSnapshot | null
  supervisor: TrainingSupervisorStatus
}>

export type TrainingPreflightResponse = Readonly<{
  admitted: boolean
  training_peak_bytes: number
  reserve_bytes: number
  required_bytes: number
  free_bytes: number
  profile_identity: string
  observed_at: string
  reason: "admitted" | "insufficient_free_memory"
}>

export type TrainingRetrievalScores = Readonly<{
  recall_at_1: number
  recall_at_5: number
  recall_at_10: number
}>

export type TrainingEvaluationSummary = Readonly<{
  dataset_sha256: string
  dataset_split: "test"
  protocol: string
  baseline_model_id: string
  baseline_revision: string
  baseline_package_sha256: string
  training_source_fingerprint: string
  evaluation_code_revision: string
  metric_definition: string
  best_validation_epoch: number
  baseline: TrainingRetrievalScores
  candidate: TrainingRetrievalScores
  candidate_weights_sha256: string
  package_sha256: string
}>

export type TrainingJobPhase =
  | "starting"
  | "training"
  | "evaluating"
  | "publishing"
  | "succeeded"
  | "cancelling"
  | "cancelled"
  | "failed"
  | "interrupted"

export type TrainingJobResponse = Readonly<{
  id: string
  request_id: string
  phase: TrainingJobPhase
  config: TrainingConfig
  dataset: TrainingDatasetSnapshot
  current_epoch: number
  current_step: number
  owner_generation: number
  cancel_requested: boolean
  attempts: number
  best_metric: number | null
  candidate_model_id: string | null
  candidate_revision: string | null
  evaluation: TrainingEvaluationSummary | null
  error: string | null
  created_at: string
  updated_at: string
  finished_at: string | null
}>

export type TrainingJobPage = Readonly<{
  items: readonly TrainingJobResponse[]
  next_cursor: string | null
}>

export type TrainingMetric = Readonly<{
  epoch: number
  step: number
  training_loss: number | null
  validation_recall_at_1: number | null
  allocated_bytes: number | null
  reserved_bytes: number | null
  observed_at: string
}>

export type TrainingMetricPage = Readonly<{
  items: readonly TrainingMetric[]
  next_cursor: string | null
}>

export type TrainingLogEntry = Readonly<{
  cursor: string
  level: "debug" | "info" | "warning" | "error"
  message: string
  observed_at: string
}>

export type TrainingLogPage = Readonly<{
  items: readonly TrainingLogEntry[]
  next_cursor: string | null
}>

export type TrainingMemoryRefusal = Readonly<{
  code: string
  message: string
  reason: string
  training_peak_bytes: number
  reserve_bytes: number
  required_bytes: number
  free_bytes: number
  observed_at?: string
  profile_identity?: string
}>
