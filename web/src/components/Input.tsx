import { type InputHTMLAttributes, useId } from "react"
import "./primitives.css"

type InputProps = Omit<InputHTMLAttributes<HTMLInputElement>, "id"> & {
  error?: string
  hint?: string
  label: string
}

export function Input({ error, hint, label, ...inputProps }: InputProps) {
  const generatedId = useId()
  const hintId = `${generatedId}-hint`
  const errorId = `${generatedId}-error`
  const describedBy = error === undefined ? (hint === undefined ? undefined : hintId) : errorId

  return (
    <div className="gw-field">
      <label className="gw-field__label" htmlFor={generatedId}>
        {label}
      </label>
      <input
        {...inputProps}
        aria-describedby={describedBy}
        aria-invalid={error === undefined ? undefined : true}
        className="gw-input"
        id={generatedId}
      />
      {error === undefined ? null : (
        <span className="gw-field__error" id={errorId}>
          {error}
        </span>
      )}
      {error === undefined && hint !== undefined ? (
        <span className="gw-field__hint" id={hintId}>
          {hint}
        </span>
      ) : null}
    </div>
  )
}
