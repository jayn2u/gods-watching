import { HttpError, NetworkError } from "../../app/client"

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
    return `${error.code}: ${error.message}`
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
