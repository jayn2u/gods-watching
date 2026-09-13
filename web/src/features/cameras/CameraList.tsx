import type { CameraResponse } from "../../app/client"
import { Button, Panel, Status } from "../../components"

type CameraListProps = Readonly<{
  cameras: readonly CameraResponse[]
  onAdd: () => void
  onDelete: (camera: CameraResponse, trigger: HTMLButtonElement) => void
  onEdit: (camera: CameraResponse) => void
}>

export function CameraList({ cameras, onAdd, onDelete, onEdit }: CameraListProps) {
  return (
    <Panel
      actions={
        <Button onClick={onAdd} variant="primary">
          Add camera
        </Button>
      }
      eyebrow="Configured sources"
      title="Camera editor"
    >
      {cameras.length === 0 ? (
        <p className="camera-list__empty">
          No cameras are configured. Add a source to begin authenticated ingest.
        </p>
      ) : (
        <div className="camera-list" data-camera-list>
          {cameras.map((camera) => (
            <article
              className="camera-card"
              data-camera-id={camera.camera_id}
              key={camera.camera_id}
            >
              <div className="camera-card__header">
                <div className="camera-card__title">
                  <h3>{camera.name}</h3>
                  <p>
                    {camera.source_host}
                    {camera.source_port === null ? "" : `:${camera.source_port}`}
                  </p>
                </div>
                <Status tone={camera.detection_enabled ? "neutral" : "warning"}>
                  {camera.detection_enabled ? "Detection on" : "Detection paused"}
                </Status>
              </div>
              <dl className="camera-card__details">
                <div>
                  <dt>Threshold</dt>
                  <dd>{camera.detection_threshold.toFixed(2)}</dd>
                </div>
                <div>
                  <dt>Configuration version</dt>
                  <dd>{camera.version}</dd>
                </div>
              </dl>
              <div className="camera-card__actions">
                <Button onClick={() => onEdit(camera)}>Edit</Button>
                <Button onClick={(event) => onDelete(camera, event.currentTarget)} variant="danger">
                  Delete
                </Button>
              </div>
            </article>
          ))}
        </div>
      )}
    </Panel>
  )
}
