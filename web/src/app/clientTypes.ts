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
