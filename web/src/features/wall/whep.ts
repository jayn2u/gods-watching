export type WhepState = "connecting" | "live" | "stale" | "offline" | "error"

export type WhepConnection = Readonly<{
  peerConnection: RTCPeerConnection
  stop: () => Promise<void>
}>

type WhepOptions = Readonly<{
  signal: AbortSignal
  onConnectionState: (state: RTCPeerConnectionState | RTCIceConnectionState) => void
  onStream: (stream: MediaStream) => void
}>

class WhepError extends Error {
  readonly name = "WhepError"
}

export class WhepCleanupError extends Error {
  readonly name = "WhepCleanupError"

  constructor(
    message: string,
    readonly status: number | null,
    readonly cause: unknown = undefined,
  ) {
    super(message)
  }
}

export async function connectWhep(cameraId: string, options: WhepOptions): Promise<WhepConnection> {
  const controller = new AbortController()
  const abort = () => controller.abort()
  options.signal.addEventListener("abort", abort, { once: true })
  if (options.signal.aborted) {
    controller.abort()
  }
  const peerConnection = new RTCPeerConnection({ iceServers: [] })
  const pendingCandidates: string[] = []
  let resourcePath: string | null = null
  let stopped = false
  let stopPromise: Promise<void> | null = null
  let cleanupPromise: Promise<void> | null = null
  let deletedResourcePath: string | null = null
  let deletePromise: Promise<void> | null = null

  peerConnection.addTransceiver("video", { direction: "recvonly" })
  peerConnection.ontrack = (event) => {
    options.onStream(event.streams[0] ?? new MediaStream([event.track]))
  }
  peerConnection.onconnectionstatechange = () => {
    options.onConnectionState(peerConnection.connectionState)
  }
  peerConnection.oniceconnectionstatechange = () => {
    options.onConnectionState(peerConnection.iceConnectionState)
  }

  const sendCandidate = (fragment: string): void => {
    if (resourcePath === null) {
      pendingCandidates.push(fragment)
      return
    }
    void patchWhep(resourcePath, fragment, controller.signal).catch(() => {
      if (!stopped) {
        options.onConnectionState("failed")
      }
    })
  }
  peerConnection.onicecandidate = (event) => {
    sendCandidate(toIceFragment(peerConnection, event.candidate))
  }

  try {
    const offer = await createOffer(peerConnection, controller.signal)
    const response = await postWhep(cameraId, offer, controller.signal)
    resourcePath = validateResourcePath(cameraId, response.headers.get("Location"))
    if (stopped) {
      await deleteResource(resourcePath)
      throw abortError()
    }
    throwIfAborted(controller.signal)
    const answer = await response.text()
    throwIfAborted(controller.signal)
    if (answer.trim() === "") {
      throw new WhepError("The live service returned an empty SDP answer.")
    }
    await peerConnection.setRemoteDescription({ type: "answer", sdp: answer })
    throwIfAborted(controller.signal)
    for (const fragment of pendingCandidates.splice(0)) {
      sendCandidate(fragment)
    }
  } catch (error) {
    const cleanupFailure = await cleanupConnection().then(
      () => null,
      (cleanupError: unknown) => cleanupError,
    )
    options.signal.removeEventListener("abort", abort)
    if (isAbortError(error) || options.signal.aborted) {
      throw abortError()
    }
    if (cleanupFailure !== null) {
      throw cleanupFailure
    }
    throw error
  }

  const stop = (): Promise<void> => {
    if (stopPromise !== null) {
      return stopPromise
    }
    stopped = true
    controller.abort()
    stopPromise = cleanupConnection().finally(() => {
      options.signal.removeEventListener("abort", abort)
    })
    return stopPromise
  }

  function deleteResource(path: string | null): Promise<void> {
    if (path === null || deletedResourcePath === path) {
      return deletePromise ?? Promise.resolve()
    }
    deletedResourcePath = path
    deletePromise = deleteWhep(path)
    return deletePromise
  }

  function cleanupConnection(): Promise<void> {
    if (cleanupPromise !== null) {
      return cleanupPromise
    }
    cleanupPromise = (async () => {
      peerConnection.ontrack = null
      peerConnection.onicecandidate = null
      peerConnection.onconnectionstatechange = null
      peerConnection.oniceconnectionstatechange = null
      peerConnection.close()
      await deleteResource(resourcePath)
    })()
    return cleanupPromise
  }

  return { peerConnection, stop }
}

async function createOffer(
  peerConnection: RTCPeerConnection,
  signal: AbortSignal,
): Promise<string> {
  const offer = await peerConnection.createOffer()
  if (signal.aborted) {
    throw abortError()
  }
  await peerConnection.setLocalDescription(offer)
  const localDescription = peerConnection.localDescription
  if (localDescription?.sdp === undefined || localDescription.sdp.trim() === "") {
    throw new WhepError("The browser did not produce a local SDP offer.")
  }
  return localDescription.sdp
}

async function postWhep(cameraId: string, offer: string, signal: AbortSignal): Promise<Response> {
  const response = await fetch(`/api/live/${encodeURIComponent(cameraId)}/whep`, {
    method: "POST",
    credentials: "same-origin",
    headers: { Accept: "application/sdp", "Content-Type": "application/sdp" },
    body: offer,
    signal,
  })
  if (response.status !== 201) {
    throw new WhepError(`The live service returned HTTP ${response.status}.`)
  }
  return response
}

async function patchWhep(path: string, fragment: string, signal: AbortSignal): Promise<void> {
  const response = await fetch(path, {
    method: "PATCH",
    credentials: "same-origin",
    headers: { "Content-Type": "application/trickle-ice-sdpfrag" },
    body: fragment,
    signal,
  })
  if (!response.ok) {
    throw new WhepError(`The live service rejected ICE candidates (HTTP ${response.status}).`)
  }
}

async function deleteWhep(path: string): Promise<void> {
  let response: Response
  try {
    response = await fetch(path, {
      method: "DELETE",
      credentials: "same-origin",
      headers: { Accept: "application/json" },
    })
  } catch (error: unknown) {
    throw new WhepCleanupError("The live service cleanup request failed.", null, error)
  }
  if (!response.ok) {
    throw new WhepCleanupError(
      `The live service rejected cleanup (HTTP ${response.status}).`,
      response.status,
    )
  }
}

function validateResourcePath(cameraId: string, location: string | null): string {
  if (location === null) {
    throw new WhepError("The live service did not return a resource location.")
  }
  let resource: URL
  try {
    resource = new URL(location, window.location.origin)
  } catch {
    throw new WhepError("The live service returned an invalid resource location.")
  }
  const expectedPrefix = `/api/live/${encodeURIComponent(cameraId)}/whep/`
  if (
    resource.origin !== window.location.origin ||
    resource.pathname.startsWith(expectedPrefix) === false ||
    isResourceIdentifier(resource.pathname.slice(expectedPrefix.length)) === false ||
    resource.search !== "" ||
    resource.hash !== ""
  ) {
    throw new WhepError("The live service returned an unsafe resource location.")
  }
  return resource.pathname
}

function isResourceIdentifier(value: string): boolean {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value)
}

function toIceFragment(
  peerConnection: RTCPeerConnection,
  candidate: RTCIceCandidate | null,
): string {
  const sdp = peerConnection.localDescription?.sdp ?? ""
  const mid = sdp.match(/^a=mid:([^\r\n]+)/m)?.[1]
  const ufrag = sdp.match(/^a=ice-ufrag:([^\r\n]+)/m)?.[1]
  const pwd = sdp.match(/^a=ice-pwd:([^\r\n]+)/m)?.[1]
  const mediaLine = sdp.match(/^m=video[^\r\n]*/m)?.[0]
  const candidateLine =
    candidate === null || candidate.candidate.trim() === ""
      ? "a=end-of-candidates"
      : `a=${candidate.candidate}`
  const lines = [
    ufrag === undefined ? null : `a=ice-ufrag:${ufrag}`,
    pwd === undefined ? null : `a=ice-pwd:${pwd}`,
    mediaLine ?? "m=video 9 UDP/TLS/RTP/SAVPF 96",
    mid === undefined ? null : `a=mid:${mid}`,
    candidateLine,
  ]
  return `${lines.filter((line): line is string => line !== null).join("\r\n")}\r\n`
}

function abortError(): DOMException {
  return new DOMException("The live connection was cancelled.", "AbortError")
}

function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) {
    throw abortError()
  }
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError"
}
