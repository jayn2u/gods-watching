import { describe, expect, it, vi } from "vitest"
import { RequestFence, startFencedRequest } from "./requestFence"

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<T>((onResolve, onReject) => {
    resolve = onResolve
    reject = onReject
  })
  return { promise, resolve, reject }
}

function handlers() {
  return { onResponse: vi.fn(), onError: vi.fn() }
}

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

describe("fenced request callbacks", () => {
  it("applies the response of the current request", async () => {
    const fence = new RequestFence()
    const pending = deferred<string>()
    const callbacks = handlers()

    const settled = startFencedRequest(fence, () => pending.promise, callbacks)
    pending.resolve("current")
    await settled

    expect(callbacks.onResponse).toHaveBeenCalledWith("current")
    expect(callbacks.onError).not.toHaveBeenCalled()
  })

  it("discards a late response after login, logout, or expiry cancels the fence", async () => {
    const fence = new RequestFence()
    const pending = deferred<string>()
    const callbacks = handlers()

    const settled = startFencedRequest(fence, () => pending.promise, callbacks)
    fence.cancel()
    pending.resolve("late")
    await settled

    expect(callbacks.onResponse).not.toHaveBeenCalled()
  })

  it("discards a late failure after the fence is cancelled", async () => {
    const fence = new RequestFence()
    const pending = deferred<string>()
    const callbacks = handlers()

    const settled = startFencedRequest(fence, () => pending.promise, callbacks)
    fence.cancel()
    pending.reject(new Error("service unavailable"))
    await settled

    expect(callbacks.onError).not.toHaveBeenCalled()
  })

  it("keeps a newer refresh when an older response resolves after it", async () => {
    const fence = new RequestFence()
    const older = deferred<string>()
    const newer = deferred<string>()
    const stale = handlers()
    const current = handlers()

    const olderSettled = startFencedRequest(fence, () => older.promise, stale)
    const newerSettled = startFencedRequest(fence, () => newer.promise, current)
    newer.resolve("newer")
    older.resolve("older")
    await Promise.all([olderSettled, newerSettled])

    expect(current.onResponse).toHaveBeenCalledWith("newer")
    expect(stale.onResponse).not.toHaveBeenCalled()
  })

  it("discards an older failure superseded by a retry", async () => {
    const fence = new RequestFence()
    const older = deferred<string>()
    const newer = deferred<string>()
    const stale = handlers()
    const current = handlers()

    const olderSettled = startFencedRequest(fence, () => older.promise, stale)
    const newerSettled = startFencedRequest(fence, () => newer.promise, current)
    older.reject(new Error("service unavailable"))
    newer.resolve("newer")
    await Promise.all([olderSettled, newerSettled])

    expect(stale.onError).not.toHaveBeenCalled()
    expect(current.onResponse).toHaveBeenCalledOnce()
  })

  it("ignores abort errors from the current request", async () => {
    const fence = new RequestFence()
    const callbacks = handlers()

    await startFencedRequest(
      fence,
      () => Promise.reject(new DOMException("aborted", "AbortError")),
      callbacks,
    )

    expect(callbacks.onError).not.toHaveBeenCalled()
  })
})
