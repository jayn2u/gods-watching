export type RequestToken = Readonly<{
  generation: number
  signal: AbortSignal
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
