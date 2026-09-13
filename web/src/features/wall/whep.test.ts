import { afterEach, describe, expect, it, vi } from "vitest"
import { connectWhep } from "./whep"

const cameraId = "00000000-0000-0000-0000-000000000001"
const resourceId = "00000000-0000-4000-8000-000000000002"

class FakePeerConnection {
  localDescription: { readonly type: string; readonly sdp: string } | null = null
  ontrack: ((event: { streams: readonly MediaStream[]; track: MediaStreamTrack }) => void) | null =
    null
  onicecandidate: ((event: { candidate: { candidate: string } | null }) => void) | null = null
  onconnectionstatechange: (() => void) | null = null
  oniceconnectionstatechange: (() => void) | null = null
  connectionState = "new"
  iceConnectionState = "new"

  addTransceiver(): void {}

  async createOffer(): Promise<{ readonly type: "offer"; readonly sdp: string }> {
    return { type: "offer", sdp: "offer" }
  }

  async setLocalDescription(): Promise<void> {
    this.localDescription = {
      type: "offer",
      sdp: [
        "v=0",
        "a=ice-ufrag:ufrag",
        "a=ice-pwd:password",
        "m=video 9 UDP/TLS/RTP/SAVPF 96",
        "a=mid:0",
      ].join("\r\n"),
    }
    queueMicrotask(() => {
      this.onicecandidate?.({
        candidate: { candidate: "candidate:1 1 UDP 1 127.0.0.1 9 typ host" },
      })
      this.onicecandidate?.({ candidate: null })
    })
  }

  async setRemoteDescription(): Promise<void> {}

  close(): void {
    this.connectionState = "closed"
    this.iceConnectionState = "closed"
  }
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe("connectWhep", () => {
  it("rejects a late response after external cancellation and deletes its resource once", async () => {
    const requests: Array<{ readonly method: string }> = []
    let releasePost = (_response: Response): void => {
      throw new Error("POST response gate was not initialized")
    }
    let postStarted = (): void => {
      throw new Error("POST start gate was not initialized")
    }
    const postStartedPromise = new Promise<void>((resolve) => {
      postStarted = resolve
    })
    const delayedPost = new Promise<Response>((resolve) => {
      releasePost = resolve
    })
    vi.stubGlobal("window", { location: { origin: "http://localhost" } })
    vi.stubGlobal("RTCPeerConnection", FakePeerConnection)
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const method = init?.method ?? "GET"
        requests.push({ method })
        if (method === "POST") {
          postStarted()
          return delayedPost
        }
        return new Response(null, { status: method === "DELETE" ? 200 : 204 })
      }),
    )

    const controller = new AbortController()
    const connectionPromise = connectWhep(cameraId, {
      signal: controller.signal,
      onConnectionState: () => undefined,
      onStream: () => undefined,
    })
    await postStartedPromise
    controller.abort()
    releasePost(
      new Response("answer", {
        status: 201,
        headers: { Location: `/api/live/${cameraId}/whep/${resourceId}` },
      }),
    )

    await expect(connectionPromise).rejects.toMatchObject({ name: "AbortError" })
    expect(requests.filter((request) => request.method === "DELETE")).toHaveLength(1)
  })

  it("cleans up when external cancellation happens during remote description", async () => {
    const requests: Array<{ readonly method: string }> = []
    let abortDuringRemoteDescription: (() => void) | null = null
    class AbortDuringRemoteDescriptionPeerConnection extends FakePeerConnection {
      async setRemoteDescription(): Promise<void> {
        abortDuringRemoteDescription?.()
      }
    }
    vi.stubGlobal("window", { location: { origin: "http://localhost" } })
    vi.stubGlobal("RTCPeerConnection", AbortDuringRemoteDescriptionPeerConnection)
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const method = init?.method ?? "GET"
        requests.push({ method })
        if (method === "POST") {
          return new Response("answer", {
            status: 201,
            headers: { Location: `/api/live/${cameraId}/whep/${resourceId}` },
          })
        }
        return new Response(null, { status: method === "DELETE" ? 200 : 204 })
      }),
    )

    const controller = new AbortController()
    abortDuringRemoteDescription = () => controller.abort()
    const connectionPromise = connectWhep(cameraId, {
      signal: controller.signal,
      onConnectionState: () => undefined,
      onStream: () => undefined,
    })

    await expect(connectionPromise).rejects.toMatchObject({ name: "AbortError" })
    expect(requests.filter((request) => request.method === "DELETE")).toHaveLength(1)
  })

  it("preserves AbortError when cancellation cleanup receives an unsuccessful delete", async () => {
    const requests: Array<{ readonly method: string }> = []
    let postStarted = (): void => {
      throw new Error("POST start gate was not initialized")
    }
    const postStartedPromise = new Promise<void>((resolve) => {
      postStarted = resolve
    })
    let releasePost = (_response: Response): void => {
      throw new Error("POST response gate was not initialized")
    }
    const delayedPost = new Promise<Response>((resolve) => {
      releasePost = resolve
    })
    vi.stubGlobal("window", { location: { origin: "http://localhost" } })
    vi.stubGlobal("RTCPeerConnection", FakePeerConnection)
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const method = init?.method ?? "GET"
        requests.push({ method })
        if (method === "POST") {
          postStarted()
          return delayedPost
        }
        if (method === "DELETE") {
          return new Response(null, { status: 500 })
        }
        return new Response(null, { status: 204 })
      }),
    )

    const controller = new AbortController()
    const connectionPromise = connectWhep(cameraId, {
      signal: controller.signal,
      onConnectionState: () => undefined,
      onStream: () => undefined,
    })
    await postStartedPromise
    controller.abort()
    releasePost(
      new Response("answer", {
        status: 201,
        headers: { Location: `/api/live/${cameraId}/whep/${resourceId}` },
      }),
    )

    await expect(connectionPromise).rejects.toMatchObject({ name: "AbortError" })
    expect(requests.filter((request) => request.method === "DELETE")).toHaveLength(1)
  })

  it("trickles valid SDP fragments and deletes one application resource", async () => {
    const requests: Array<{
      readonly url: string
      readonly method: string
      readonly body: string
    }> = []
    vi.stubGlobal("window", { location: { origin: "http://localhost" } })
    vi.stubGlobal("RTCPeerConnection", FakePeerConnection)
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input)
        const method = init?.method ?? "GET"
        requests.push({ url, method, body: typeof init?.body === "string" ? init.body : "" })
        if (method === "POST") {
          return new Response("answer", {
            status: 201,
            headers: { Location: `/api/live/${cameraId}/whep/${resourceId}` },
          })
        }
        return new Response(null, { status: method === "DELETE" ? 200 : 204 })
      }),
    )

    const connection = await connectWhep(cameraId, {
      signal: new AbortController().signal,
      onConnectionState: () => undefined,
      onStream: () => undefined,
    })
    await new Promise<void>((resolve) => queueMicrotask(resolve))
    await connection.stop()

    const patchBodies = requests
      .filter((request) => request.method === "PATCH")
      .map((request) => request.body)
    expect(patchBodies).toHaveLength(2)
    expect(patchBodies[0]).toContain("a=ice-ufrag:ufrag")
    expect(patchBodies[0]).toContain("a=ice-pwd:password")
    expect(patchBodies[0]).toContain("m=video 9 UDP/TLS/RTP/SAVPF 96")
    expect(patchBodies[0]).toContain("a=mid:0")
    expect(patchBodies[0]).toContain("a=candidate:1 1 UDP 1 127.0.0.1 9 typ host")
    expect(patchBodies[1]).toContain("a=end-of-candidates")
    expect(requests.filter((request) => request.method === "DELETE")).toHaveLength(1)
  })

  it("surfaces an unsuccessful resource delete and still attempts it once", async () => {
    let deleteAttempts = 0
    vi.stubGlobal("window", { location: { origin: "http://localhost" } })
    vi.stubGlobal("RTCPeerConnection", FakePeerConnection)
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const method = init?.method ?? "GET"
        if (method === "POST") {
          return new Response("answer", {
            status: 201,
            headers: { Location: `/api/live/${cameraId}/whep/${resourceId}` },
          })
        }
        if (method === "DELETE") {
          deleteAttempts += 1
          return new Response(null, { status: 500 })
        }
        return new Response(null, { status: 204 })
      }),
    )

    const connection = await connectWhep(cameraId, {
      signal: new AbortController().signal,
      onConnectionState: () => undefined,
      onStream: () => undefined,
    })

    const stopPromise = connection.stop()
    await expect(stopPromise).rejects.toMatchObject({ name: "WhepCleanupError", status: 500 })
    await expect(connection.stop()).rejects.toMatchObject({ name: "WhepCleanupError", status: 500 })
    expect(deleteAttempts).toBe(1)
  })

  it("surfaces a rejected resource delete without creating another attempt", async () => {
    let deleteAttempts = 0
    vi.stubGlobal("window", { location: { origin: "http://localhost" } })
    vi.stubGlobal("RTCPeerConnection", FakePeerConnection)
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const method = init?.method ?? "GET"
        if (method === "POST") {
          return new Response("answer", {
            status: 201,
            headers: { Location: `/api/live/${cameraId}/whep/${resourceId}` },
          })
        }
        if (method === "DELETE") {
          deleteAttempts += 1
          throw new TypeError("network unavailable")
        }
        return new Response(null, { status: 204 })
      }),
    )

    const connection = await connectWhep(cameraId, {
      signal: new AbortController().signal,
      onConnectionState: () => undefined,
      onStream: () => undefined,
    })

    await expect(connection.stop()).rejects.toMatchObject({ name: "WhepCleanupError" })
    expect(deleteAttempts).toBe(1)
  })
})
