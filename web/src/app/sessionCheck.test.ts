import { describe, expect, it, vi } from "vitest"
import type { SessionResponse } from "./client"
import { RequestFence } from "./requestFence"
import { startSessionCheck } from "./sessionCheck"

const AUTHENTICATED: SessionResponse = {
  authenticated: true,
  idle_expires_at: "2026-09-13T00:30:00Z",
  absolute_expires_at: "2026-09-13T12:00:00Z",
}

function deferredSession() {
  let resolve!: (response: SessionResponse) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<SessionResponse>((onResolve, onReject) => {
    resolve = onResolve
    reject = onReject
  })
  return { promise, resolve, reject }
}

function handlers() {
  return { onResponse: vi.fn(), onError: vi.fn() }
}

describe("session check ownership", () => {
  it("applies the response of the current check", async () => {
    const fence = new RequestFence()
    const pending = deferredSession()
    const callbacks = handlers()

    const settled = startSessionCheck(fence, () => pending.promise, callbacks)
    pending.resolve(AUTHENTICATED)
    await settled

    expect(callbacks.onResponse).toHaveBeenCalledWith(AUTHENTICATED)
    expect(callbacks.onError).not.toHaveBeenCalled()
  })

  it("discards a late response after login or expiry cancels the check", async () => {
    const fence = new RequestFence()
    const pending = deferredSession()
    const callbacks = handlers()

    const settled = startSessionCheck(fence, () => pending.promise, callbacks)
    fence.cancel()
    pending.resolve(AUTHENTICATED)
    await settled

    expect(callbacks.onResponse).not.toHaveBeenCalled()
  })

  it("discards a late failure after the check is cancelled", async () => {
    const fence = new RequestFence()
    const pending = deferredSession()
    const callbacks = handlers()

    const settled = startSessionCheck(fence, () => pending.promise, callbacks)
    fence.cancel()
    pending.reject(new Error("service unavailable"))
    await settled

    expect(callbacks.onError).not.toHaveBeenCalled()
  })

  it("discards an older check superseded by a retry", async () => {
    const fence = new RequestFence()
    const first = deferredSession()
    const second = deferredSession()
    const stale = handlers()
    const current = handlers()

    const firstSettled = startSessionCheck(fence, () => first.promise, stale)
    const secondSettled = startSessionCheck(fence, () => second.promise, current)
    first.resolve(AUTHENTICATED)
    second.reject(new Error("service unavailable"))
    await Promise.all([firstSettled, secondSettled])

    expect(stale.onResponse).not.toHaveBeenCalled()
    expect(current.onError).toHaveBeenCalledOnce()
  })

  it("ignores abort errors from the current check", async () => {
    const fence = new RequestFence()
    const callbacks = handlers()

    await startSessionCheck(
      fence,
      () => Promise.reject(new DOMException("aborted", "AbortError")),
      callbacks,
    )

    expect(callbacks.onError).not.toHaveBeenCalled()
  })
})
