import { useRef, useState } from "react"
import type { CameraResponse } from "../../app/client"
import { Panel, Status } from "../../components"
import { RetentionSettings } from "../settings/RetentionSettings"
import { CameraEditor } from "./CameraEditor"
import { CameraList } from "./CameraList"
import type { CameraScreenProps } from "./cameraTypes"
import { DeleteCameraDialog } from "./DeleteCameraDialog"
import "./cameras.css"

type EditorTarget = CameraResponse | null | undefined

export function CameraScreen({
  cameras,
  client,
  onRefresh,
  onSettingsSaved,
  onUnauthorized,
  settings,
  settingsClient,
}: CameraScreenProps) {
  const [editorTarget, setEditorTarget] = useState<EditorTarget>(undefined)
  const [deleteTarget, setDeleteTarget] = useState<CameraResponse | null>(null)
  const deleteTriggerRef = useRef<HTMLButtonElement | null>(null)
  const editorCamera =
    editorTarget === undefined || editorTarget === null || cameras.kind !== "ready"
      ? editorTarget
      : (cameras.cameras.find((camera) => camera.camera_id === editorTarget.camera_id) ??
        editorTarget)

  const currentDeleteTarget =
    deleteTarget === null
      ? null
      : cameras.kind === "ready"
        ? (cameras.cameras.find((camera) => camera.camera_id === deleteTarget.camera_id) ??
          deleteTarget)
        : deleteTarget

  return (
    <main aria-labelledby="camera-heading" className="camera-settings-main">
      <div className="camera-settings__heading">
        <div>
          <p className="app-kicker">Configuration</p>
          <h1 id="camera-heading">Cameras</h1>
        </div>
        {cameras.kind === "ready" ? (
          <Status>{`${cameras.cameras.length} configured`}</Status>
        ) : null}
      </div>
      <div className="camera-settings__layout">
        <div className="camera-settings__primary">
          {editorTarget !== undefined ? (
            <CameraEditor
              camera={editorCamera ?? null}
              client={client}
              key={
                editorCamera === null || editorCamera === undefined ? "new" : editorCamera.camera_id
              }
              onClose={() => setEditorTarget(undefined)}
              onRefresh={onRefresh}
              onSaved={() => {
                setEditorTarget(undefined)
                onRefresh()
              }}
              onUnauthorized={onUnauthorized}
            />
          ) : cameras.kind === "ready" ? null : (
            <Panel title="Camera editor">
              {cameras.kind === "loading" ? (
                <Status>Loading camera configuration…</Status>
              ) : (
                <p className="camera-settings__error" role="alert">
                  {cameras.message}
                </p>
              )}
            </Panel>
          )}
          {cameras.kind === "ready" ? (
            <CameraList
              cameras={cameras.cameras}
              onAdd={() => setEditorTarget(null)}
              onDelete={(camera, trigger) => {
                deleteTriggerRef.current = trigger
                setDeleteTarget(camera)
              }}
              onEdit={(camera) => setEditorTarget(camera)}
            />
          ) : null}
        </div>
        <div className="camera-settings__secondary">
          <RetentionSettings
            onSettingsSaved={onSettingsSaved}
            onUnauthorized={onUnauthorized}
            settings={settings}
            settingsClient={settingsClient}
          />
          {cameras.kind === "ready" ? (
            <Panel eyebrow="Source boundary" title="Operational notes" variant="quiet">
              <ul className="camera-notes">
                <li>Only sanitized host and port metadata is returned to this browser.</li>
                <li>Detection enabled is a configuration choice, not an ingest health signal.</li>
                <li>Appearance history follows retention after a camera is removed.</li>
              </ul>
            </Panel>
          ) : null}
        </div>
      </div>
      <DeleteCameraDialog
        camera={currentDeleteTarget}
        client={client}
        onClose={() => setDeleteTarget(null)}
        onRefresh={onRefresh}
        onRemoved={() => {
          setDeleteTarget(null)
          onRefresh()
        }}
        onUnauthorized={onUnauthorized}
        returnFocusRef={deleteTriggerRef}
      />
    </main>
  )
}
