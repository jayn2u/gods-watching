import { type ReactNode, type RefObject, useEffect, useId, useRef } from "react"
import { Button } from "./Button"
import "./surfaces.css"

type DialogProps = {
  children: ReactNode
  confirmLabel: string
  description: string
  onClose: () => void
  onConfirm: () => void
  open: boolean
  returnFocusRef: RefObject<HTMLButtonElement | null>
  title: string
}

export function Dialog(props: DialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null)
  const cancelRef = useRef<HTMLButtonElement>(null)
  const confirmRef = useRef<HTMLButtonElement>(null)
  const titleId = useId()
  const descriptionId = useId()

  useEffect(() => {
    const dialog = dialogRef.current
    if (dialog === null) {
      return
    }
    if (props.open) {
      dialog.showModal()
      cancelRef.current?.focus()
      return
    }
    if (dialog.open) {
      dialog.close()
      props.returnFocusRef.current?.focus()
    }
  }, [props.open, props.returnFocusRef])

  return (
    <dialog
      aria-describedby={descriptionId}
      aria-labelledby={titleId}
      className="gw-dialog"
      onCancel={(event) => {
        event.preventDefault()
        props.onClose()
      }}
      onPointerDown={(event) => {
        if (event.target === dialogRef.current) {
          props.onClose()
        }
      }}
      onKeyDown={(event) => {
        if (event.key !== "Tab") {
          return
        }
        if (event.shiftKey && document.activeElement === cancelRef.current) {
          event.preventDefault()
          confirmRef.current?.focus()
        } else if (!event.shiftKey && document.activeElement === confirmRef.current) {
          event.preventDefault()
          cancelRef.current?.focus()
        }
      }}
      ref={dialogRef}
    >
      <div className="gw-dialog__surface">
        <div className="gw-dialog__marker" />
        <h2 className="gw-dialog__title" id={titleId}>
          {props.title}
        </h2>
        <p className="gw-dialog__description" id={descriptionId}>
          {props.description}
        </p>
        <div className="gw-dialog__content">{props.children}</div>
        <div className="gw-dialog__actions">
          <Button onClick={props.onClose} ref={cancelRef}>
            Cancel
          </Button>
          <Button onClick={props.onConfirm} ref={confirmRef} variant="danger">
            {props.confirmLabel}
          </Button>
        </div>
      </div>
    </dialog>
  )
}
