import type { CameraResponse, SettingsResponse, WallSlotIds } from "../../app/client"
import { Button } from "../../components/Button"
import { Status } from "../../components/Status"
import { WallSlot } from "./WallSlot"
import "./wall.css"

export type WallSettingsState =
  | { readonly kind: "loading" }
  | { readonly kind: "ready"; readonly settings: SettingsResponse }
  | { readonly kind: "error"; readonly message: string }

type LiveWallProps = Readonly<{
  cameras: readonly CameraResponse[]
  settings: WallSettingsState
  saving: boolean
  onSlotsChange: (slots: WallSlotIds) => void
}>

const EMPTY_SLOTS: WallSlotIds = [null, null, null, null]
const SLOT_KEYS = ["slot-1", "slot-2", "slot-3", "slot-4"] as const

export function LiveWall({ cameras, settings, saving, onSlotsChange }: LiveWallProps) {
  const slots = settings.kind === "ready" ? settings.settings.wall_slot_ids : EMPTY_SLOTS
  const selectedCount = slots.filter((cameraId) => cameraId !== null).length

  function replaceSlot(index: number, cameraId: string | null): void {
    const next = [...slots] as [string | null, string | null, string | null, string | null]
    next[index] = cameraId
    if (cameraId !== null) {
      for (let other = 0; other < next.length; other += 1) {
        if (other !== index && next[other] === cameraId) {
          next[other] = null
        }
      }
    }
    onSlotsChange(next)
  }

  function fillWall(): void {
    const next: [string | null, string | null, string | null, string | null] = [
      null,
      null,
      null,
      null,
    ]
    cameras.slice(0, 4).forEach((camera, index) => {
      next[index] = camera.camera_id
    })
    onSlotsChange(next)
  }

  function assignedLabel(): string {
    return `${selectedCount} of 4 assigned`
  }

  function wallDescription(): string {
    if (settings.kind === "error") {
      return settings.message
    }
    return "Choose a configured camera for each slot. Live means decoded frames are advancing."
  }

  return (
    <main aria-labelledby="wall-heading" className="wall-main">
      <div className="wall-main__heading">
        <div>
          <p className="app-kicker">Live wall</p>
          <h1 id="wall-heading">Authenticated live wall</h1>
        </div>
        <Status tone={selectedCount === 0 ? "neutral" : "live"}>{assignedLabel()}</Status>
      </div>
      {settings.kind === "loading" ? (
        <p className="wall-main__notice" role="status">
          Loading durable wall placement…
        </p>
      ) : null}
      {settings.kind === "error" ? (
        <p className="wall-main__notice wall-main__notice--error" role="alert">
          {wallDescription()}
        </p>
      ) : null}
      <div className="wall-main__toolbar">
        <p>{wallDescription()}</p>
        <Button
          disabled={saving || cameras.length === 0 || settings.kind !== "ready"}
          onClick={fillWall}
        >
          Fill wall
        </Button>
      </div>
      <div className="wall-grid" data-shell-wall-grid>
        {slots.map((cameraId, index) => (
          <WallSlot
            camera={cameras.find((candidate) => candidate.camera_id === cameraId)}
            cameraId={cameraId}
            cameras={cameras}
            disabled={saving || settings.kind !== "ready"}
            index={index}
            key={SLOT_KEYS[index]}
            onAssign={replaceSlot}
          />
        ))}
      </div>
    </main>
  )
}
