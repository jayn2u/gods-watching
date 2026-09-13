import { type RefObject, useEffect, useRef, useState } from "react"
import { connectWhep, WhepCleanupError, type WhepConnection } from "./whep"

export type LiveStreamStatus = "idle" | "connecting" | "live" | "stale" | "offline"

export type LiveStreamSnapshot = Readonly<{
  status: LiveStreamStatus
  detail?: string
}>

export type LiveStreamResult = Readonly<{
  videoRef: RefObject<HTMLVideoElement | null>
  snapshot: LiveStreamSnapshot
}>

const STALE_AFTER_MS = 2_000
const OFFLINE_AFTER_MS = 10_000
const WATCH_INTERVAL_MS = 500

export function useLiveStream(cameraId: string | null): LiveStreamResult {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [snapshot, setSnapshot] = useState<LiveStreamSnapshot>(() =>
    cameraId === null ? { status: "idle" } : { status: "connecting" },
  )

  useEffect(() => {
    let active = true
    let connection: WhepConnection | null = null
    let frameCallbackId: number | null = null
    let watchTimer: number | null = null
    let statsTimer: number | null = null
    let lastFrameAt: number | null = null
    let sawFrame = false
    let terminal = false
    let previousFramesDecoded = 0
    const controller = new AbortController()

    setSnapshot(cameraId === null ? { status: "idle" } : { status: "connecting" })
    if (cameraId === null) {
      return () => undefined
    }

    const video = videoRef.current
    const markFrame = (): void => {
      if (!active) {
        return
      }
      sawFrame = true
      lastFrameAt = Date.now()
      setSnapshot({ status: "live" })
      if (video !== null && typeof video.requestVideoFrameCallback === "function") {
        frameCallbackId = video.requestVideoFrameCallback(() => markFrame())
      }
    }
    const attachStream = (nextStream: MediaStream): void => {
      if (video !== null) {
        video.srcObject = nextStream
        void video.play().catch(() => undefined)
      }
      if (video !== null && typeof video.requestVideoFrameCallback === "function") {
        frameCallbackId = video.requestVideoFrameCallback(() => markFrame())
      }
    }
    const onConnectionState = (state: RTCPeerConnectionState | RTCIceConnectionState): void => {
      if (!active) {
        return
      }
      if (state === "failed" || state === "closed") {
        terminal = true
        setSnapshot({ status: "offline", detail: "Connection ended" })
        if (statsTimer !== null) {
          window.clearInterval(statsTimer)
          statsTimer = null
        }
        if (connection !== null) {
          const endedConnection = connection
          connection = null
          stopConnection(endedConnection)
        }
        return
      }
      if (state === "disconnected" && sawFrame === false) {
        setSnapshot({ status: "stale", detail: "Waiting for a decoded frame" })
      }
    }
    const stopConnection = (target: WhepConnection): void => {
      void target.stop().catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") {
          return
        }
        if (active && error instanceof WhepCleanupError) {
          setSnapshot({ status: "offline", detail: "Live connection cleanup failed" })
          return
        }
        if (active) {
          setSnapshot({ status: "offline", detail: "Live connection cleanup failed" })
        }
      })
    }
    const pollStats = async (): Promise<void> => {
      const peerConnection = connection?.peerConnection
      if (!active || peerConnection === undefined) {
        return
      }
      try {
        const reports = await peerConnection.getStats()
        reports.forEach((report) => {
          if (
            report.type === "inbound-rtp" &&
            report.kind === "video" &&
            typeof report.framesDecoded === "number" &&
            report.framesDecoded > previousFramesDecoded
          ) {
            previousFramesDecoded = report.framesDecoded
            markFrame()
          }
        })
      } catch {
        if (active && !terminal) {
          setSnapshot((current) => (current.status === "live" ? { status: "stale" } : current))
        }
      }
    }
    const checkFreshness = (): void => {
      if (!active || terminal) {
        return
      }
      if (lastFrameAt === null) {
        return
      }
      const age = Date.now() - lastFrameAt
      if (age >= OFFLINE_AFTER_MS) {
        setSnapshot({ status: "offline", detail: "No decoded frames for 10 seconds" })
      } else if (age >= STALE_AFTER_MS) {
        setSnapshot({ status: "stale", detail: "No decoded frames for 2 seconds" })
      }
    }
    void connectWhep(cameraId, {
      signal: controller.signal,
      onConnectionState,
      onStream: attachStream,
    })
      .then((nextConnection) => {
        if (!active) {
          stopConnection(nextConnection)
          return
        }
        connection = nextConnection
        if (terminal) {
          connection = null
          stopConnection(nextConnection)
          return
        }
        if (lastFrameAt === null) {
          lastFrameAt = Date.now()
        }
        statsTimer = window.setInterval(() => void pollStats(), WATCH_INTERVAL_MS)
      })
      .catch((error: unknown) => {
        if (active && !(error instanceof DOMException && error.name === "AbortError")) {
          setSnapshot({ status: "offline", detail: "Live service unavailable" })
        }
      })

    watchTimer = window.setInterval(checkFreshness, WATCH_INTERVAL_MS)
    return () => {
      active = false
      controller.abort()
      if (watchTimer !== null) {
        window.clearInterval(watchTimer)
      }
      if (statsTimer !== null) {
        window.clearInterval(statsTimer)
      }
      if (frameCallbackId !== null && video !== null) {
        video.cancelVideoFrameCallback(frameCallbackId)
      }
      if (video !== null) {
        video.pause()
        video.srcObject = null
      }
      if (connection !== null) {
        stopConnection(connection)
      }
    }
  }, [cameraId])

  return { videoRef, snapshot }
}
