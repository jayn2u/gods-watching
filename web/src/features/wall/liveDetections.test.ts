import { describe, expect, it } from "vitest"
import { parseLiveDetectionsResponse } from "../../app/clientDomain"
import { LiveDetectionOverlay } from "./LiveDetectionOverlay"
import { canApplyDetectionResponse, detectionExpiryAt, isDetectionFresh } from "./liveDetections"

const validResponse = {
  camera_id: "camera-1",
  camera_session_id: "session-1",
  frame_at: "2026-09-22T08:00:00Z",
  frame_age_seconds: 0.25,
  width: 1920,
  height: 1080,
  boxes: [{ x1: 120.5, y1: 80, x2: 420, y2: 700, confidence: 0.91 }],
}

describe("live detection response parsing", () => {
  it("keeps source-pixel boxes and nullable frame metadata", () => {
    expect(parseLiveDetectionsResponse(validResponse)).toEqual(validResponse)
    expect(
      parseLiveDetectionsResponse({
        ...validResponse,
        camera_session_id: null,
        frame_at: null,
        frame_age_seconds: null,
        width: null,
        height: null,
        boxes: [],
      }),
    ).toEqual({
      ...validResponse,
      camera_session_id: null,
      frame_at: null,
      frame_age_seconds: null,
      width: null,
      height: null,
      boxes: [],
    })
  })

  it.each([
    ["x1", { boxes: [{ ...validResponse.boxes[0], x1: Number.POSITIVE_INFINITY }] }],
    ["width", { width: Number.NaN }],
    ["confidence", { boxes: [{ ...validResponse.boxes[0], confidence: 1.1 }] }],
  ])("rejects non-finite or out-of-range %s", (_field, change) => {
    expect(() => parseLiveDetectionsResponse({ ...validResponse, ...change })).toThrow(
      /live detection response has an invalid shape/,
    )
  })
})

describe("live detection freshness", () => {
  it("subtracts the server frame age from the one-second client freshness window", () => {
    const expiresAt = detectionExpiryAt(10_000, 0.25)

    expect(expiresAt).toBe(10_750)
    expect(isDetectionFresh(expiresAt - 1, expiresAt)).toBe(true)
    expect(isDetectionFresh(expiresAt, expiresAt)).toBe(false)
  })
})

describe("live detection response ownership", () => {
  it("drops a response from a switched camera or an older poll generation", () => {
    const response = parseLiveDetectionsResponse(validResponse)

    expect(
      canApplyDetectionResponse({
        response,
        requestedCameraId: "camera-1",
        currentCameraId: "camera-2",
        effectGeneration: 3,
        currentEffectGeneration: 3,
        requestGeneration: 3,
        lastAppliedGeneration: 2,
      }),
    ).toBe(false)
    expect(
      canApplyDetectionResponse({
        response,
        requestedCameraId: "camera-1",
        currentCameraId: "camera-1",
        effectGeneration: 1,
        currentEffectGeneration: 1,
        requestGeneration: 1,
        lastAppliedGeneration: 2,
      }),
    ).toBe(false)
    expect(
      canApplyDetectionResponse({
        response,
        requestedCameraId: "camera-1",
        currentCameraId: "camera-1",
        effectGeneration: 3,
        currentEffectGeneration: 3,
        requestGeneration: 3,
        lastAppliedGeneration: 2,
      }),
    ).toBe(true)
    expect(
      canApplyDetectionResponse({
        response,
        requestedCameraId: "camera-1",
        currentCameraId: "camera-1",
        effectGeneration: 3,
        currentEffectGeneration: 3,
        requestGeneration: 4,
        lastAppliedGeneration: 3,
      }),
    ).toBe(true)
  })
})

describe("live detection SVG mapping", () => {
  it("uses the same source-pixel cover mapping as the video", () => {
    const element = LiveDetectionOverlay({
      width: 1920,
      height: 1080,
      boxes: validResponse.boxes,
    })
    const svgProps = element?.props as {
      viewBox: string
      preserveAspectRatio: string
      pointerEvents: string
      children:
        | { props: { x: number; y: number; width: number; height: number } }
        | readonly [
            {
              props: {
                x: number
                y: number
                width: number
                height: number
                stroke: string
                strokeWidth: number
                vectorEffect: string
              }
            },
          ]
    }
    const rectangle = Array.isArray(svgProps.children) ? svgProps.children[0] : svgProps.children

    expect(svgProps.viewBox).toBe("0 0 1920 1080")
    expect(svgProps.preserveAspectRatio).toBe("xMidYMid slice")
    expect(svgProps.pointerEvents).toBe("none")
    expect(rectangle.props).toMatchObject({
      x: 120.5,
      y: 80,
      width: 299.5,
      height: 620,
      stroke: "#22d36f",
      strokeWidth: 3,
      vectorEffect: "non-scaling-stroke",
    })
  })
})
