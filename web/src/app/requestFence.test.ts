import { describe, expect, it } from "vitest"
import { RequestFence } from "./requestFence"

describe("latest request ownership", () => {
  it("aborts and invalidates an older request when a newer one starts", () => {
    const fence = new RequestFence()
    const first = fence.start()
    const second = fence.start()

    expect(first.signal.aborted).toBe(true)
    expect(fence.isCurrent(first.generation)).toBe(false)
    expect(fence.isCurrent(second.generation)).toBe(true)
  })

  it("invalidates a pending request when the surface is cancelled or unmounted", () => {
    const fence = new RequestFence()
    const pending = fence.start()

    fence.cancel()

    expect(pending.signal.aborted).toBe(true)
    expect(fence.isCurrent(pending.generation)).toBe(false)
  })
})
