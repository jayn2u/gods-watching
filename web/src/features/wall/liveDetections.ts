import { useEffect, useRef, useState } from "react"
import {
  apiClient,
  isAbortError,
  type LiveDetectionBox,
  type LiveDetectionsResponse,
} from "../../app/client"
import type { LiveStreamStatus } from "./useLiveStream"

export const LIVE_DETECTIONS_POLL_MS = 200
const LIVE_DETECTIONS_MAX_AGE_MS = 1_000

export type LiveDetectionSnapshot = Readonly<{
  cameraId: string | null
  width: number | null
  height: number | null
  boxes: readonly LiveDetectionBox[]
}>

export const EMPTY_LIVE_DETECTION_SNAPSHOT: LiveDetectionSnapshot = {
  cameraId: null,
  width: null,
  height: null,
  boxes: [],
}

type DetectionResponseOwnership = Readonly<{
  response: LiveDetectionsResponse
  requestedCameraId: string
  currentCameraId: string | null
  effectGeneration: number
  currentEffectGeneration: number
  requestGeneration: number
  lastAppliedGeneration: number
}>

export function detectionExpiryAt(receivedAt: number, frameAgeSeconds: number | null): number {
  const serverAgeMs = Math.max(0, (frameAgeSeconds ?? 0) * 1_000)
  return receivedAt + Math.max(0, LIVE_DETECTIONS_MAX_AGE_MS - serverAgeMs)
}

export function isDetectionFresh(now: number, expiresAt: number): boolean {
  return now < expiresAt
}

export function canApplyDetectionResponse({
  response,
  requestedCameraId,
  currentCameraId,
  effectGeneration,
  currentEffectGeneration,
  requestGeneration,
  lastAppliedGeneration,
}: DetectionResponseOwnership): boolean {
  return (
    requestedCameraId === currentCameraId &&
    response.camera_id === currentCameraId &&
    effectGeneration === currentEffectGeneration &&
    requestGeneration > lastAppliedGeneration
  )
}

function monotonicNow(): number {
  return typeof performance !== "undefined" && typeof performance.now === "function"
    ? performance.now()
    : Date.now()
}

function isDocumentVisible(): boolean {
  return typeof document === "undefined" || document.visibilityState !== "hidden"
}

function clearTimer(timer: number | null): void {
  if (timer !== null) {
    window.clearTimeout(timer)
  }
}

export function useLiveDetections(
  cameraId: string | null,
  detectionEnabled: boolean,
  streamStatus: LiveStreamStatus,
): LiveDetectionSnapshot {
  const [visible, setVisible] = useState(isDocumentVisible)
  const [snapshot, setSnapshot] = useState<LiveDetectionSnapshot>(EMPTY_LIVE_DETECTION_SNAPSHOT)
  const generationRef = useRef(0)

  useEffect(() => {
    if (typeof document === "undefined") {
      return
    }
    const onVisibilityChange = (): void => {
      setVisible(isDocumentVisible())
    }
    document.addEventListener("visibilitychange", onVisibilityChange)
    return () => document.removeEventListener("visibilitychange", onVisibilityChange)
  }, [])

  useEffect(() => {
    setSnapshot(EMPTY_LIVE_DETECTION_SNAPSHOT)
    generationRef.current += 1
    const currentGeneration = generationRef.current
    if (cameraId === null || !detectionEnabled || streamStatus !== "live" || !visible) {
      return
    }

    let active = true
    let inFlight = false
    let requestGeneration = 0
    let lastAppliedGeneration = 0
    let pollTimer: number | null = null
    let expiryTimer: number | null = null
    const controller = new AbortController()

    const clearDetections = (): void => {
      clearTimer(expiryTimer)
      expiryTimer = null
      if (active) {
        setSnapshot(EMPTY_LIVE_DETECTION_SNAPSHOT)
      }
    }
    const scheduleExpiry = (expiresAt: number): void => {
      clearTimer(expiryTimer)
      const delay = Math.max(0, expiresAt - monotonicNow())
      expiryTimer = window.setTimeout(() => {
        expiryTimer = null
        if (!active) {
          return
        }
        if (isDetectionFresh(monotonicNow(), expiresAt)) {
          scheduleExpiry(expiresAt)
          return
        }
        setSnapshot(EMPTY_LIVE_DETECTION_SNAPSHOT)
      }, delay)
    }
    const poll = (): void => {
      if (!active || inFlight) {
        return
      }
      inFlight = true
      const thisRequestGeneration = ++requestGeneration
      void apiClient
        .getLiveDetections(cameraId, controller.signal)
        .then((response) => {
          const receivedAt = monotonicNow()
          if (
            !active ||
            !canApplyDetectionResponse({
              response,
              requestedCameraId: cameraId,
              currentCameraId: cameraId,
              effectGeneration: currentGeneration,
              currentEffectGeneration: generationRef.current,
              requestGeneration: thisRequestGeneration,
              lastAppliedGeneration,
            })
          ) {
            return
          }
          lastAppliedGeneration = thisRequestGeneration
          if (response.width === null || response.height === null || response.boxes.length === 0) {
            clearDetections()
            return
          }
          const expiresAt = detectionExpiryAt(receivedAt, response.frame_age_seconds)
          if (!isDetectionFresh(receivedAt, expiresAt)) {
            clearDetections()
            return
          }
          setSnapshot({
            cameraId: response.camera_id,
            width: response.width,
            height: response.height,
            boxes: response.boxes,
          })
          scheduleExpiry(expiresAt)
        })
        .catch((error: unknown) => {
          if (active && !isAbortError(error)) {
            clearDetections()
          }
        })
        .finally(() => {
          inFlight = false
        })
    }

    poll()
    pollTimer = window.setInterval(poll, LIVE_DETECTIONS_POLL_MS)
    return () => {
      active = false
      controller.abort()
      if (pollTimer !== null) {
        window.clearInterval(pollTimer)
      }
      clearTimer(expiryTimer)
    }
  }, [cameraId, detectionEnabled, streamStatus, visible])

  return snapshot
}
