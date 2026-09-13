import { useState } from "react"
import { Button } from "../../components/Button"
import { Status } from "../../components/Status"
import { CameraScreen } from "../../features/cameras"
import { SearchScreen } from "../../features/search"
import { LiveWall, type WallSettingsState } from "../../features/wall/LiveWall"
import type { ApiClient, CameraResponse, SettingsResponse, WallSlotIds } from "../client"
import "./layout.css"

type Screen = "wall" | "search" | "cameras"

export type CameraState =
  | { readonly kind: "loading" }
  | { readonly kind: "ready"; readonly cameras: readonly CameraResponse[] }
  | { readonly kind: "error"; readonly message: string }

export type AppShellProps = {
  readonly cameras: CameraState
  readonly client: ApiClient
  readonly settings: WallSettingsState
  readonly savingWallSlots: boolean
  readonly onCamerasRefresh: () => void
  readonly onLogout: () => Promise<void>
  readonly onSessionExpired: () => void
  readonly onSettingsSaved: (settings: SettingsResponse) => void
  readonly onWallSlotsChange: (slots: WallSlotIds) => void
  readonly sessionNotice?: string | undefined
}

const NAV_ITEMS = [
  { id: "wall", label: "Live wall" },
  { id: "search", label: "Person search" },
  { id: "cameras", label: "Cameras" },
] as const

export function AppShell({
  cameras,
  client,
  onCamerasRefresh,
  onLogout,
  onSessionExpired,
  onSettingsSaved,
  onWallSlotsChange,
  savingWallSlots,
  sessionNotice,
  settings,
}: AppShellProps) {
  const [screen, setScreen] = useState<Screen>("wall")
  const [loggingOut, setLoggingOut] = useState(false)

  async function logout(): Promise<void> {
    setLoggingOut(true)
    await onLogout()
    setLoggingOut(false)
  }

  const cameraCount = cameras.kind === "ready" ? String(cameras.cameras.length) : "—"

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header__brand">
          <span>God’s Watching</span>
          <span className="app-header__brand-tag">RTSP</span>
        </div>
        <nav aria-label="Primary" className="app-nav">
          {NAV_ITEMS.map((item) => (
            <button
              aria-current={screen === item.id ? "page" : undefined}
              className={
                screen === item.id ? "app-nav__button app-nav__button--active" : "app-nav__button"
              }
              key={item.id}
              onClick={() => setScreen(item.id)}
              type="button"
            >
              {item.label}
            </button>
          ))}
        </nav>
        <div className="app-header__actions">
          <div className="app-header__meta">
            <Status tone="live">Session active</Status>
            <span>{cameraCount} configured</span>
            <span>Operator</span>
          </div>
          <Button disabled={loggingOut} onClick={() => void logout()}>
            Sign out
          </Button>
        </div>
      </header>
      {sessionNotice === undefined ? null : (
        <p className="app-shell__notice" role="alert">
          {sessionNotice}
        </p>
      )}
      <div className="app-body">
        <CameraTree cameras={cameras} />
        {screen === "wall" ? (
          <LiveWall
            cameras={cameras.kind === "ready" ? cameras.cameras : []}
            onSlotsChange={onWallSlotsChange}
            saving={savingWallSlots}
            settings={settings}
          />
        ) : null}
        {screen === "search" ? (
          <SearchScreen cameras={cameras} client={client} onUnauthorized={onSessionExpired} />
        ) : null}
        {screen === "cameras" ? (
          <CameraScreen
            cameras={cameras}
            client={client}
            onRefresh={onCamerasRefresh}
            onSettingsSaved={onSettingsSaved}
            onUnauthorized={onSessionExpired}
            settings={settings}
            settingsClient={client}
          />
        ) : null}
        <EventRail />
      </div>
      <footer className="app-footer">Research-only local console · no recording or playback</footer>
    </div>
  )
}

function CameraTree({ cameras }: { readonly cameras: CameraState }) {
  return (
    <aside className="camera-tree" aria-label="Camera tree">
      <div className="camera-tree__header">
        <span>Camera tree</span>
        <span>{cameras.kind === "ready" ? cameras.cameras.length : "—"}</span>
      </div>
      <div className="camera-tree__list">
        {cameras.kind === "loading" ? (
          <p className="app-empty">Loading camera configuration…</p>
        ) : null}
        {cameras.kind === "error" ? (
          <p className="app-empty app-empty--error">{cameras.message}</p>
        ) : null}
        {cameras.kind === "ready" && cameras.cameras.length === 0 ? (
          <p className="app-empty">No cameras are configured.</p>
        ) : null}
        {cameras.kind === "ready"
          ? cameras.cameras.map((camera) => (
              <button className="camera-tree__item" key={camera.camera_id} type="button">
                <span className="camera-tree__copy">
                  <strong>{camera.name}</strong>
                  <small>
                    {camera.source_host}
                    {camera.source_port === null ? "" : `:${camera.source_port}`} · configured
                  </small>
                </span>
                <Status tone={camera.detection_enabled ? "live" : "neutral"}>
                  {camera.detection_enabled ? "Ready" : "Paused"}
                </Status>
              </button>
            ))
          : null}
      </div>
      <div className="camera-tree__actions">
        <Button disabled>Fill wall</Button>
        <Button disabled>Add</Button>
      </div>
    </aside>
  )
}

function EventRail() {
  return (
    <aside className="event-rail" aria-label="Events">
      <div className="event-rail__header">Events</div>
      <div className="event-rail__empty">
        <Status tone="neutral">No event feed</Status>
        <p>Events will appear when authenticated live processing is connected.</p>
      </div>
    </aside>
  )
}
