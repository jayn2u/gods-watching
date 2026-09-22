import type { LiveDetectionBox } from "../../app/client"

type LiveDetectionOverlayProps = Readonly<{
  width: number
  height: number
  boxes: readonly LiveDetectionBox[]
}>

export function LiveDetectionOverlay({ width, height, boxes }: LiveDetectionOverlayProps) {
  if (boxes.length === 0) {
    return null
  }
  return (
    <svg
      aria-hidden="true"
      className="wall-slot__detection-overlay"
      focusable="false"
      height="100%"
      preserveAspectRatio="xMidYMid slice"
      pointerEvents="none"
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
    >
      {boxes.map((box) => (
        <rect
          fill="none"
          height={box.y2 - box.y1}
          key={`${box.x1}-${box.y1}-${box.x2}-${box.y2}-${box.confidence}`}
          stroke="#22d36f"
          strokeWidth={3}
          vectorEffect="non-scaling-stroke"
          width={box.x2 - box.x1}
          x={box.x1}
          y={box.y1}
        />
      ))}
    </svg>
  )
}
