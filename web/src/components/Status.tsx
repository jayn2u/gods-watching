import "./surfaces.css"

export type StatusTone = "neutral" | "live" | "warning" | "offline"

type StatusProps = {
  announce?: boolean
  children: string
  tone?: StatusTone
}

export function Status({ announce = true, children, tone = "neutral" }: StatusProps) {
  return (
    <span
      className={`gw-status gw-status--${tone}`}
      role={announce ? (tone === "offline" ? "alert" : "status") : undefined}
    >
      <span aria-hidden="true" className="gw-status__dot" />
      {children}
    </span>
  )
}
