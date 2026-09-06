import "./surfaces.css"

export type StatusTone = "neutral" | "live" | "warning" | "offline"

type StatusProps = {
  children: string
  tone?: StatusTone
}

export function Status({ children, tone = "neutral" }: StatusProps) {
  return (
    <span className={`gw-status gw-status--${tone}`} role={tone === "offline" ? "alert" : "status"}>
      <span aria-hidden="true" className="gw-status__dot" />
      {children}
    </span>
  )
}
