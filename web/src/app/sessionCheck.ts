import { isAbortError, type SessionResponse } from "./client"
import type { RequestFence } from "./requestFence"

export type SessionCheckHandlers = Readonly<{
  onResponse: (response: SessionResponse) => void
  onError: (error: unknown) => void
}>

/**
 * Starts a latest-wins session check. A result that settles after the fence was
 * cancelled (login, logout, expiry) or restarted is discarded, even when the
 * response body was already read before the abort.
 */
export function startSessionCheck(
  fence: RequestFence,
  getSession: (signal: AbortSignal) => Promise<SessionResponse>,
  handlers: SessionCheckHandlers,
): Promise<void> {
  const { generation, signal } = fence.start()
  return getSession(signal).then(
    (response) => {
      if (fence.isCurrent(generation)) {
        handlers.onResponse(response)
      }
    },
    (error: unknown) => {
      if (fence.isCurrent(generation) && !isAbortError(error)) {
        handlers.onError(error)
      }
    },
  )
}
