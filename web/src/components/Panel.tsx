import type { ReactNode } from "react"
import "./surfaces.css"

type PanelProps = {
  actions?: ReactNode
  children: ReactNode
  eyebrow?: string
  title: string
  variant?: "default" | "quiet"
}

export function Panel({ actions, children, eyebrow, title, variant = "default" }: PanelProps) {
  return (
    <section className={`gw-panel gw-panel--${variant}`}>
      <header className="gw-panel__header">
        <div>
          {eyebrow === undefined ? null : <p className="gw-panel__eyebrow">{eyebrow}</p>}
          <h2 className="gw-panel__title">{title}</h2>
        </div>
        {actions === undefined ? null : <div className="gw-panel__actions">{actions}</div>}
      </header>
      <div className="gw-panel__body">{children}</div>
    </section>
  )
}
