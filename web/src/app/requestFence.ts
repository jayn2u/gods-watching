import { isAbortError } from "./client"

export type RequestToken = Readonly<{
  generation: number
  signal: AbortSignal
}>

export type FencedRequestHandlers<T> = Readonly<{
  onResponse: (response: T) => void
  onError: (error: unknown) => void
}>

/** Owns one latest-wins request and its cancellation signal per feature surface. */
export class RequestFence {
  #controller: AbortController | null = null
  #generation = 0

  start(): RequestToken {
    this.#controller?.abort()
    this.#generation += 1
    const controller = new AbortController()
    this.#controller = controller
    return { generation: this.#generation, signal: controller.signal }
  }

  isCurrent(generation: number): boolean {
    return generation === this.#generation && this.#controller !== null
  }

  cancel(): void {
    this.#controller?.abort()
    this.#controller = null
    this.#generation += 1
  }

  dispose(): void {
    this.cancel()
  }
}

/**
 * Starts a latest-wins request on the fence. A result that settles after the
 * fence was cancelled or restarted is discarded, even when the response body
 * was already read before the abort.
 */
export function startFencedRequest<T>(
  fence: RequestFence,
  request: (signal: AbortSignal) => Promise<T>,
  handlers: FencedRequestHandlers<T>,
): Promise<void> {
  const { generation, signal } = fence.start()
  return request(signal).then(
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
