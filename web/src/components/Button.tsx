import type { ComponentPropsWithRef, ReactNode } from "react"
import "./primitives.css"

export type ButtonVariant = "primary" | "secondary" | "danger"

type ButtonProps = ComponentPropsWithRef<"button"> & {
  children: ReactNode
  loading?: boolean
  variant?: ButtonVariant
}

export function Button({
  children,
  className = "",
  disabled = false,
  loading = false,
  ref,
  type = "button",
  variant = "secondary",
  ...buttonProps
}: ButtonProps) {
  const classes = ["gw-button", `gw-button--${variant}`, className].filter(Boolean).join(" ")

  return (
    <button
      {...buttonProps}
      aria-busy={loading}
      className={classes}
      disabled={disabled || loading}
      ref={ref}
      type={type}
    >
      {loading ? <span aria-hidden="true" className="gw-button__spinner" /> : null}
      <span>{children}</span>
    </button>
  )
}
