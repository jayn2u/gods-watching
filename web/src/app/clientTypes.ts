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
