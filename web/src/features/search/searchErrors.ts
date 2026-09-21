import { HttpError, NetworkError } from "../../app/client"

const OPERATOR_ERROR_MESSAGES: Readonly<Record<string, string>> = {
  appearance_expired: "This appearance crop has expired under the retention policy.",
  appearance_not_found: "This appearance is no longer available. Return to the results and retry.",
  inference_unavailable: "Person search is temporarily unavailable. Retry in a moment.",
  invalid_text: "The person description could not be searched. Shorten it and try again.",
  text_inference_unavailable: "Person search is temporarily unavailable. Retry in a moment.",
  unknown_camera:
    "A selected camera is no longer available. Refresh the camera list and try again.",
} as const

export function describeSearchError(
  error: unknown,
  onUnauthorized: () => void,
): string | undefined {
  if (error instanceof CropVersionMismatchError) {
    return error.message
  }
  if (error instanceof HttpError) {
    if (error.status === 401) {
      onUnauthorized()
      return undefined
    }
    return (
      OPERATOR_ERROR_MESSAGES[error.code] ?? "Search could not be completed. Retry the request."
    )
  }
  if (error instanceof NetworkError) {
    return "Search is unavailable. Check the session service and retry."
  }
  return "Search could not be completed. Retry the request."
}

export class CropVersionMismatchError extends Error {
  readonly name = "CropVersionMismatchError"

  constructor() {
    super("The representative crop changed while it was loading. Retry to load a matching crop.")
  }
}
