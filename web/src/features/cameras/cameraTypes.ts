import type { ApiClient, CameraResponse, SettingsResponse } from "../../app/client"
import type { CameraState } from "../../app/layout/AppShell"
import type { WallSettingsState } from "../wall/LiveWall"

export type CameraClient = Pick<
  ApiClient,
  "createCamera" | "testCamera" | "updateCamera" | "deleteCamera"
>

export type SettingsClient = Pick<ApiClient, "patchSettings">
export type ModelSettingsClient = Pick<
  ApiClient,
  "applyModel" | "getModelSettings" | "getModelPreflight"
>

export type RetentionSettingsProps = Readonly<{
  settings: WallSettingsState
  settingsClient: SettingsClient
  onSettingsSaved: (settings: SettingsResponse) => void
  onUnauthorized: () => void
}>

export type CameraScreenProps = Readonly<{
  cameras: CameraState
  client: CameraClient & ModelSettingsClient
  onRefresh: () => void
  onUnauthorized: () => void
  settings: WallSettingsState
  settingsClient: SettingsClient
  onSettingsSaved: (settings: SettingsResponse) => void
}>

export type CameraEditorMode = "create" | "edit"

export type CameraFormDraft = Readonly<{
  name: string
  sourceUrl: string
  sourceUsername: string
  sourcePassword: string
  threshold: string
  detectionEnabled: boolean
}>

export type CameraFieldErrors = Readonly<{
  name: string | undefined
  sourceUrl: string | undefined
  sourceUsername: string | undefined
  sourcePassword: string | undefined
  threshold: string | undefined
}>

export type CameraEditorProps = Readonly<{
  camera: CameraResponse | null
  client: CameraClient
  onClose: () => void
  onSaved: () => void
  onUnauthorized: () => void
  onRefresh: () => void
}>
