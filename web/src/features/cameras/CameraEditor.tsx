import { useEffect, useRef, useState } from "react"
import {
  type CameraCreateRequest,
  type CameraPatchRequest,
  type CameraResponse,
  type CameraTestResponse,
  HttpError,
  isAbortError,
  NetworkError,
} from "../../app/client"
import { Button, Input, Panel, Status } from "../../components"
import type {
  CameraEditorMode,
  CameraEditorProps,
  CameraFieldErrors,
  CameraFormDraft,
} from "./cameraTypes"
import {
  buildAuthenticatedSourceUrl,
  resolveSourceUrl,
  testMetadataLabel,
  validateCameraName,
  validateRtspCredentials,
  validateThreshold,
} from "./cameraValidation"

const EMPTY_FIELD_ERRORS: CameraFieldErrors = {
  name: undefined,
  sourceUrl: undefined,
  sourceUsername: undefined,
  sourcePassword: undefined,
  threshold: undefined,
}

type PendingAction = "test" | "save" | null

function initialDraft(camera: CameraResponse | null): CameraFormDraft {
  return camera === null
    ? {
        name: "",
        sourceUrl: "",
        sourceUsername: "",
        sourcePassword: "",
        threshold: "0.50",
        detectionEnabled: true,
      }
    : {
        name: camera.name,
        sourceUrl: "",
        sourceUsername: "",
        sourcePassword: "",
        threshold: String(camera.detection_threshold),
        detectionEnabled: camera.detection_enabled,
      }
}

function apiErrorMessage(error: unknown, action: "save" | "test"): string {
  if (error instanceof HttpError) {
    if (error.code === "authentication_failed") {
      return "RTSP authentication failed. Enter the stream's RTSP username and password, then test again."
    }
    if (error.status === 409 && action === "save") {
      return "This camera changed elsewhere. The latest row was requested; review your draft before trying again."
    }
    return `${error.code}: ${error.message}`
  }
  if (error instanceof NetworkError) {
    return error.message
  }
  if (error instanceof Error) {
    return error.message
  }
  return action === "test"
    ? "The camera source could not be tested. Try again."
    : "The camera could not be saved. Try again."
}

export function CameraEditor({
  camera,
  client,
  onClose,
  onRefresh,
  onSaved,
  onUnauthorized,
}: CameraEditorProps) {
  const mode: CameraEditorMode = camera === null ? "create" : "edit"
  const [draft, setDraft] = useState<CameraFormDraft>(() => initialDraft(camera))
  const [fieldErrors, setFieldErrors] = useState<CameraFieldErrors>(EMPTY_FIELD_ERRORS)
  const [formError, setFormError] = useState<string | undefined>(undefined)
  const [pendingAction, setPendingAction] = useState<PendingAction>(null)
  const [testResult, setTestResult] = useState<CameraTestResponse | null>(null)
  const requestGeneration = useRef(0)
  const activeController = useRef<AbortController | null>(null)

  useEffect(() => {
    return () => {
      requestGeneration.current += 1
      activeController.current?.abort()
    }
  }, [])

  function invalidateConnectionTest(): void {
    setTestResult(null)
    if (pendingAction !== "test") {
      return
    }
    requestGeneration.current += 1
    activeController.current?.abort()
    activeController.current = null
    setPendingAction(null)
  }

  function updateDraft(
    field: "name" | "sourceUrl" | "sourceUsername" | "sourcePassword" | "threshold",
    value: string,
  ): void {
    setDraft((current) => ({ ...current, [field]: value }))
    setFieldErrors((current) => ({ ...current, [field]: undefined }))
    setFormError(undefined)
    if (field === "sourceUrl" || field === "sourceUsername" || field === "sourcePassword") {
      invalidateConnectionTest()
    }
  }

  function updateDetectionEnabled(value: boolean): void {
    setDraft((current) => ({ ...current, detectionEnabled: value }))
    setFormError(undefined)
  }

  function beginRequest(action: Exclude<PendingAction, null>): Readonly<{
    controller: AbortController
    generation: number
  }> {
    activeController.current?.abort()
    const controller = new AbortController()
    activeController.current = controller
    const generation = requestGeneration.current + 1
    requestGeneration.current = generation
    setPendingAction(action)
    setFormError(undefined)
    return { controller, generation }
  }

  function isCurrentRequest(generation: number, controller: AbortController): boolean {
    return generation === requestGeneration.current && !controller.signal.aborted
  }

  async function testConnection(): Promise<void> {
    if (pendingAction !== null) {
      return
    }
    const credentials = validateRtspCredentials(draft.sourceUsername, draft.sourcePassword)
    if (credentials.kind === "invalid") {
      setFieldErrors((current) => ({
        ...current,
        sourceUsername: credentials.message,
        sourcePassword: undefined,
      }))
      setTestResult(null)
      return
    }
    const source = buildAuthenticatedSourceUrl(
      draft.sourceUrl,
      credentials.value.username,
      credentials.value.password,
    )
    if (source.kind === "invalid") {
      setFieldErrors((current) => ({ ...current, sourceUrl: source.message }))
      setTestResult(null)
      return
    }

    const request = beginRequest("test")
    try {
      const result = await client.testCamera(
        { source_url: source.value },
        request.controller.signal,
      )
      if (!isCurrentRequest(request.generation, request.controller)) {
        return
      }
      setTestResult(result)
    } catch (error) {
      if (!isCurrentRequest(request.generation, request.controller) || isAbortError(error)) {
        return
      }
      if (error instanceof HttpError && error.status === 401) {
        onUnauthorized()
        return
      }
      setTestResult(null)
      setFormError(apiErrorMessage(error, "test"))
    } finally {
      if (request.generation === requestGeneration.current) {
        activeController.current = null
        setPendingAction(null)
      }
    }
  }

  function validatedForm():
    | Readonly<{
        kind: "valid"
        name: string
        sourceUrl: string | null
        threshold: number
      }>
    | Readonly<{ kind: "invalid"; errors: CameraFieldErrors }> {
    const name = validateCameraName(draft.name)
    const threshold = validateThreshold(draft.threshold)
    const credentials = validateRtspCredentials(draft.sourceUsername, draft.sourcePassword)
    const source =
      credentials.kind === "invalid" && draft.sourceUrl.trim() !== ""
        ? { kind: "valid" as const, value: null }
        : resolveSourceUrl(draft.sourceUrl, draft.sourceUsername, draft.sourcePassword, mode)
    const errors: CameraFieldErrors = {
      name: name.kind === "invalid" ? name.message : undefined,
      sourceUrl: source.kind === "invalid" ? source.message : undefined,
      sourceUsername: credentials.kind === "invalid" ? credentials.message : undefined,
      sourcePassword: undefined,
      threshold: threshold.kind === "invalid" ? threshold.message : undefined,
    }
    if (
      name.kind === "invalid" ||
      credentials.kind === "invalid" ||
      source.kind === "invalid" ||
      threshold.kind === "invalid"
    ) {
      return { kind: "invalid", errors }
    }
    const sourceUrl =
      source.kind === "valid" && typeof source.value === "string" ? source.value : null
    return {
      kind: "valid",
      name: name.value,
      sourceUrl,
      threshold: threshold.value,
    }
  }

  async function saveCamera(): Promise<void> {
    if (pendingAction !== null) {
      return
    }
    const form = validatedForm()
    if (form.kind === "invalid") {
      setFieldErrors(form.errors)
      return
    }

    const request = beginRequest("save")
    try {
      if (mode === "create" && form.sourceUrl !== null) {
        const createRequest: CameraCreateRequest = {
          name: form.name,
          source_url: form.sourceUrl,
          detection_enabled: draft.detectionEnabled,
          detection_threshold: form.threshold,
        }
        await client.createCamera(createRequest, request.controller.signal)
      } else if (mode === "edit" && camera !== null) {
        const currentCamera = camera
        const patch: CameraPatchRequest =
          form.sourceUrl === null
            ? {
                name: form.name,
                detection_enabled: draft.detectionEnabled,
                detection_threshold: form.threshold,
              }
            : {
                name: form.name,
                source_url: form.sourceUrl,
                detection_enabled: draft.detectionEnabled,
                detection_threshold: form.threshold,
              }
        await client.updateCamera(
          currentCamera.camera_id,
          patch,
          currentCamera.version,
          request.controller.signal,
        )
      }
      if (!isCurrentRequest(request.generation, request.controller)) {
        return
      }
      onSaved()
    } catch (error) {
      if (!isCurrentRequest(request.generation, request.controller) || isAbortError(error)) {
        return
      }
      if (error instanceof HttpError && error.status === 401) {
        onUnauthorized()
        return
      }
      if (error instanceof HttpError && error.status === 409 && mode === "edit") {
        onRefresh()
      }
      setFormError(apiErrorMessage(error, "save"))
    } finally {
      if (request.generation === requestGeneration.current) {
        activeController.current = null
        setPendingAction(null)
      }
    }
  }

  return (
    <Panel
      eyebrow={mode === "create" ? "Register source" : "Edit configuration"}
      title={mode === "create" ? "Add camera" : "Edit camera"}
    >
      <form
        className="camera-editor"
        noValidate
        onSubmit={(event) => {
          event.preventDefault()
          void saveCamera()
        }}
      >
        <p className="camera-editor__copy">
          {mode === "create"
            ? "Register an RTSP source for the authenticated research wall."
            : "Update the working camera configuration. A stored source URL is never returned to this form."}
        </p>
        <div className="camera-editor__fields">
          <Input
            autoComplete="off"
            disabled={pendingAction === "save"}
            error={fieldErrors.name}
            label="Camera name"
            maxLength={80}
            onChange={(event) => updateDraft("name", event.target.value)}
            value={draft.name}
          />
          <div className="camera-editor__source-fields">
            <Input
              autoComplete="off"
              disabled={pendingAction === "save"}
              error={fieldErrors.sourceUrl}
              hint={
                mode === "edit"
                  ? "Leave empty to keep the current source. Enter a replacement when changing the RTSP server credentials."
                  : "Optional RTSP server credentials are separate from app sign-in and stay server-side."
              }
              label={mode === "edit" ? "Replacement RTSP source (optional)" : "RTSP source"}
              onChange={(event) => updateDraft("sourceUrl", event.target.value)}
              placeholder="rtsp://host:8554/path"
              type="text"
              value={draft.sourceUrl}
            />
            <Input
              autoComplete="off"
              disabled={pendingAction === "save"}
              error={fieldErrors.sourceUsername}
              hint="Optional RTSP server credential, separate from app sign-in."
              label="RTSP username (optional)"
              onChange={(event) => updateDraft("sourceUsername", event.target.value)}
              type="text"
              value={draft.sourceUsername}
            />
            <Input
              autoComplete="new-password"
              disabled={pendingAction === "save"}
              error={fieldErrors.sourcePassword}
              hint="Optional RTSP server credential, separate from app sign-in."
              label="RTSP password (optional)"
              onChange={(event) => updateDraft("sourcePassword", event.target.value)}
              type="password"
              value={draft.sourcePassword}
            />
          </div>
          <label className="camera-editor__toggle">
            <input
              checked={draft.detectionEnabled}
              disabled={pendingAction === "save"}
              onChange={(event) => updateDetectionEnabled(event.target.checked)}
              type="checkbox"
            />
            <span>Detection enabled</span>
          </label>
          <Input
            disabled={pendingAction === "save"}
            error={fieldErrors.threshold}
            inputMode="decimal"
            label="Detection threshold"
            max="0.95"
            min="0.10"
            onChange={(event) => updateDraft("threshold", event.target.value)}
            step="0.01"
            type="number"
            value={draft.threshold}
          />
        </div>
        {testResult === null ? null : (
          <p className="camera-editor__test-result" role="status">
            <Status tone="live">Connection verified</Status>
            <span>{testMetadataLabel(testResult)}</span>
          </p>
        )}
        {formError === undefined ? null : (
          <p className="camera-editor__error" role="alert">
            {formError}
          </p>
        )}
        <div className="camera-editor__actions">
          <Button disabled={pendingAction === "save"} onClick={onClose}>
            Cancel
          </Button>
          <Button
            disabled={pendingAction === "save"}
            loading={pendingAction === "test"}
            onClick={() => void testConnection()}
          >
            Test connection
          </Button>
          <Button loading={pendingAction === "save"} type="submit" variant="primary">
            {mode === "create" ? "Create camera" : "Save camera"}
          </Button>
        </div>
      </form>
    </Panel>
  )
}
