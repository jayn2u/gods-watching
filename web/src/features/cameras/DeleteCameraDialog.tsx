import type { RefObject } from "react"
import { useEffect, useRef, useState } from "react"
import { type CameraResponse, HttpError, isAbortError, NetworkError } from "../../app/client"
import { Dialog, Status } from "../../components"
import type { CameraClient } from "./cameraTypes"

type DeleteCameraDialogProps = Readonly<{
  camera: CameraResponse | null
  client: CameraClient
  onClose: () => void
  onRefresh: () => void
  onRemoved: () => void
  onUnauthorized: () => void
  returnFocusRef: RefObject<HTMLButtonElement | null>
}>

function apiErrorMessage(error: unknown): string {
  if (error instanceof HttpError) {
    if (error.status === 409) {
      return "This camera changed elsewhere. The latest list was requested; review it before trying again."
    }
    return `${error.code}: ${error.message}`
  }
  if (error instanceof NetworkError) {
    return error.message
  }
  if (error instanceof Error) {
    return error.message
  }
  return "The camera could not be removed. Try again."
}

export function DeleteCameraDialog({
  camera,
  client,
  onClose,
  onRefresh,
  onRemoved,
  onUnauthorized,
  returnFocusRef,
}: DeleteCameraDialogProps) {
  const [pending, setPending] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | undefined>(undefined)
  const requestGeneration = useRef(0)
  const activeController = useRef<AbortController | null>(null)
  const open = camera !== null

  useEffect(() => {
    if (open) {
      return
    }
    requestGeneration.current += 1
    activeController.current?.abort()
    setPending(false)
    setErrorMessage(undefined)
  }, [open])

  useEffect(() => {
    return () => {
      requestGeneration.current += 1
      activeController.current?.abort()
    }
  }, [])

  async function removeCamera(): Promise<void> {
    if (pending || camera === null) {
      return
    }
    activeController.current?.abort()
    const controller = new AbortController()
    activeController.current = controller
    const generation = requestGeneration.current + 1
    requestGeneration.current = generation
    setPending(true)
    setErrorMessage(undefined)

    try {
      await client.deleteCamera(camera.camera_id, camera.version, controller.signal)
      if (generation !== requestGeneration.current || controller.signal.aborted) {
        return
      }
      onRemoved()
    } catch (error) {
      if (generation !== requestGeneration.current || isAbortError(error)) {
        return
      }
      if (error instanceof HttpError && error.status === 401) {
        onUnauthorized()
        return
      }
      if (error instanceof HttpError && error.status === 409) {
        onRefresh()
      }
      setErrorMessage(apiErrorMessage(error))
    } finally {
      if (generation === requestGeneration.current) {
        activeController.current = null
        setPending(false)
      }
    }
  }

  function closeDialog(): void {
    requestGeneration.current += 1
    activeController.current?.abort()
    setPending(false)
    onClose()
  }

  return (
    <Dialog
      confirmLabel="Remove camera"
      description="Removing this camera cancels its ingest. Existing appearance history stays under its archived camera label until retention removes it."
      onClose={closeDialog}
      onConfirm={() => void removeCamera()}
      open={open}
      returnFocusRef={returnFocusRef}
      title="Remove camera?"
    >
      <p>
        The row disappears only after the service confirms a successful 204 response. Existing crops
        and metadata remain searchable until the retention policy removes them.
      </p>
      {pending ? <Status>Removing camera…</Status> : null}
      {errorMessage === undefined ? null : <p role="alert">{errorMessage}</p>}
    </Dialog>
  )
}
