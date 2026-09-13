import { useRef, useState } from "react"
import type { CameraResponse, WallSlotIds } from "../../app/client"
import { Button } from "../../components/Button"
import { Status, type StatusTone } from "../../components/Status"
import { type LiveStreamStatus, useLiveStream } from "./useLiveStream"

type WallSlotProps = Readonly<{
  index: number
  cameraId: WallSlotIds[number]
  camera: CameraResponse | undefined
  cameras: readonly CameraResponse[]
  disabled?: boolean
  onAssign: (index: number, cameraId: string | null) => void
}>

const STATUS_LABELS: Record<LiveStreamStatus, string> = {
  idle: "Idle",
  connecting: "Connecting",
  live: "Live",
  stale: "Stale",
  offline: "Offline",
}

const STATUS_TONES: Record<LiveStreamStatus, StatusTone> = {
  idle: "neutral",
  connecting: "warning",
  live: "live",
  stale: "warning",
  offline: "offline",
}

export function WallSlot({
  index,
  cameraId,
  camera,
  cameras,
  disabled = false,
  onAssign,
}: WallSlotProps) {
  const slotRef = useRef<HTMLElement>(null)
  const [fullscreenRequested, setFullscreenRequested] = useState(false)
  const { videoRef, snapshot } = useLiveStream(cameraId)
  const slotLabel = `Slot ${index + 1}`

  async function requestFullscreen(): Promise<void> {
    const element = slotRef.current
    if (element === null || typeof element.requestFullscreen !== "function") {
      return
    }
    setFullscreenRequested(true)
    try {
      await element.requestFullscreen()
    } catch {
      setFullscreenRequested(false)
    }
  }

  return (
    <section
      aria-label={slotLabel}
      className="wall-slot"
      data-camera-id={cameraId ?? undefined}
      data-fullscreen-requested={fullscreenRequested ? "true" : undefined}
      data-wall-slot
      ref={slotRef}
    >
      <div className="wall-slot__topline">
        <span className="wall-slot__number">#{String(index + 1).padStart(2, "0")}</span>
        <div className="wall-slot__camera">
          <strong>{camera?.name ?? "No camera selected"}</strong>
          <small>
            {camera === undefined
              ? cameraId === null
                ? "Choose a configured camera"
                : "Camera unavailable"
              : `${camera.source_host}${camera.source_port === null ? "" : `:${camera.source_port}`}`}
          </small>
        </div>
        <Status tone={STATUS_TONES[snapshot.status]}>{STATUS_LABELS[snapshot.status]}</Status>
      </div>
      <div className={`wall-slot__viewport wall-slot__viewport--${snapshot.status}`}>
        {cameraId === null ? null : (
          <video
            aria-label={`${camera?.name ?? slotLabel} live video`}
            autoPlay
            className="wall-slot__video"
            muted
            playsInline
            ref={videoRef}
          />
        )}
        <div className="wall-slot__overlay">
          <span>{cameraId === null ? "NO SIGNAL" : STATUS_LABELS[snapshot.status]}</span>
          <small>
            {snapshot.detail ??
              (cameraId === null ? "Assign a camera to this slot" : "Waiting for decoded video")}
          </small>
        </div>
      </div>
      <div className="wall-slot__controls">
        <label className="wall-slot__select-label">
          <span>{slotLabel} camera</span>
          <select
            aria-label={`${slotLabel} camera`}
            disabled={disabled}
            onChange={(event) =>
              onAssign(index, event.target.value === "" ? null : event.target.value)
            }
            value={cameraId ?? ""}
          >
            <option value="">No camera</option>
            {cameras.map((option) => (
              <option key={option.camera_id} value={option.camera_id}>
                {option.name}
              </option>
            ))}
          </select>
        </label>
        <Button
          aria-label={`Fullscreen ${slotLabel.toLowerCase()}`}
          onClick={() => void requestFullscreen()}
        >
          Fullscreen
        </Button>
      </div>
    </section>
  )
}
