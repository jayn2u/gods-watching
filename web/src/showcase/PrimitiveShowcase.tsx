import { useRef, useState } from "react"
import { Button, Dialog, Input, Panel, Status } from "../components"
import "./showcase.css"

export function PrimitiveShowcase() {
  const [dialogOpen, setDialogOpen] = useState(false)
  const dialogTriggerRef = useRef<HTMLButtonElement>(null)

  return (
    <main className="showcase">
      <header className="showcase__masthead">
        <div className="showcase__brand">
          <span aria-hidden="true" className="showcase__brand-mark" />
          <span>Gods Watching</span>
        </div>
        <Status tone="live">System ready</Status>
      </header>

      <div className="showcase__content">
        <div className="showcase__intro">
          <p className="showcase__kicker">Reference system / R4</p>
          <h1>Design primitives</h1>
          <p>
            Accessible controls for the live monitoring surface, extracted from the exact navy and
            cyan reference palette.
          </p>
        </div>

        <div className="showcase__grid">
          <Panel eyebrow="Controls" title="Buttons">
            <div className="showcase__stack">
              <div className="showcase__row">
                <Button variant="primary">Primary action</Button>
                <Button>Secondary action</Button>
                <Button variant="danger">Danger action</Button>
              </div>
              <div className="showcase__row">
                <Button disabled>Disabled action</Button>
                <Button loading variant="primary">
                  Connecting
                </Button>
              </div>
            </div>
          </Panel>

          <Panel eyebrow="Form" title="Inputs">
            <div className="showcase__stack">
              <Input hint="Visible to operators." label="Camera name" placeholder="Entrance" />
              <Input error="Enter a camera name." label="Invalid camera name" value="" readOnly />
              <Input disabled label="Disabled field" value="Unavailable" readOnly />
            </div>
          </Panel>

          <Panel eyebrow="Feedback" title="Statuses">
            <div className="showcase__status-list">
              <Status>Idle</Status>
              <Status tone="live">Live</Status>
              <Status tone="warning">Reconnecting</Status>
              <Status tone="offline">Offline</Status>
            </div>
          </Panel>

          <Panel
            actions={
              <Button onClick={() => setDialogOpen(true)} ref={dialogTriggerRef}>
                Open dialog
              </Button>
            }
            eyebrow="Overlay"
            title="Dialog"
            variant="quiet"
          >
            <p className="showcase__panel-copy">
              Modal focus begins on the safe action, remains contained, and returns here on close.
            </p>
          </Panel>

          <div className="showcase__geometry-panel">
            <Panel eyebrow="Geometry" title="Product shell contract">
              <div aria-label="Desktop product columns" className="showcase__geometry" role="img">
                <span>232</span>
                <span>minmax(0, 1fr)</span>
                <span>268</span>
              </div>
              <p className="showcase__caption">54px header · 10px wall gap · 2px radius</p>
            </Panel>
          </div>

          <Panel eyebrow="Scope" title="Approved deviations" variant="quiet">
            <ul className="showcase__scope-list">
              <li>Recording and playback controls excluded</li>
              <li>Scanner and attribute filters excluded</li>
              <li>English search and global retention added</li>
            </ul>
          </Panel>
        </div>
      </div>

      <Dialog
        confirmLabel="Remove"
        description="This action removes the selected camera configuration."
        onClose={() => setDialogOpen(false)}
        onConfirm={() => setDialogOpen(false)}
        open={dialogOpen}
        returnFocusRef={dialogTriggerRef}
        title="Remove camera?"
      >
        The primitive supplies semantics and behavior; product data is intentionally absent.
      </Dialog>
    </main>
  )
}
