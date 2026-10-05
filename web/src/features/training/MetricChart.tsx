export type MetricPoint = Readonly<{
  label: string
  value: number
}>

type MetricChartProps = Readonly<{
  title: string
  points: readonly MetricPoint[]
  formatValue?: (value: number) => string
}>

const WIDTH = 360
const HEIGHT = 150
const PADDING = 18

function pointCoordinates(points: readonly MetricPoint[]): string {
  const values = points.map((point) => point.value)
  const minimum = Math.min(...values)
  const maximum = Math.max(...values)
  const span = maximum - minimum || 1
  return points
    .map((point, index) => {
      const x = PADDING + (index / Math.max(1, points.length - 1)) * (WIDTH - PADDING * 2)
      const y = HEIGHT - PADDING - ((point.value - minimum) / span) * (HEIGHT - PADDING * 2)
      return `${x},${y}`
    })
    .join(" ")
}

function defaultFormat(value: number): string {
  return value.toFixed(3)
}

export function MetricChart({ title, points, formatValue = defaultFormat }: MetricChartProps) {
  if (points.length === 0) {
    return (
      <section className="training-chart" aria-label={title}>
        <h3>{title}</h3>
        <p className="training-muted">No measurements have been recorded yet.</p>
      </section>
    )
  }
  const values = points.map((point) => point.value)
  const minimum = Math.min(...values)
  const maximum = Math.max(...values)
  return (
    <figure className="training-chart">
      <figcaption>{title}</figcaption>
      <div className="training-chart__bounds" aria-hidden="true">
        <span>{formatValue(maximum)}</span>
        <span>{formatValue(minimum)}</span>
      </div>
      <svg
        aria-label={`${title} by epoch`}
        className="training-chart__svg"
        preserveAspectRatio="none"
        role="img"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      >
        <title>{title} by epoch</title>
        <line x1={PADDING} x2={WIDTH - PADDING} y1={PADDING} y2={PADDING} />
        <line x1={PADDING} x2={WIDTH - PADDING} y1={HEIGHT / 2} y2={HEIGHT / 2} />
        <line x1={PADDING} x2={WIDTH - PADDING} y1={HEIGHT - PADDING} y2={HEIGHT - PADDING} />
        <polyline points={pointCoordinates(points)} />
      </svg>
      <div className="training-chart__labels" aria-hidden="true">
        <span>{points[0]?.label}</span>
        <span>{points.at(-1)?.label}</span>
      </div>
    </figure>
  )
}
