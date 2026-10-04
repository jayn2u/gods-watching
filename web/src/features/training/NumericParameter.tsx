import { useEffect, useId, useRef, useState } from "react"
import {
  logarithmicPercent,
  logarithmicValueAt,
  type NumericBounds,
  parseNumericDraft,
  roundTrainingStep,
} from "./trainingModel"

export type NumericParameterProps = Readonly<{
  fieldKey: string
  label: string
  value: number
  min: number
  max: number
  step: number
  integer?: boolean
  logScale?: boolean
  description?: string
  disabled?: boolean
  formatValue?: (value: number) => string
  onChange: (value: number) => void
  onDraftValidityChange?: (fieldKey: string, valid: boolean) => void
}>

function defaultFormat(value: number): string {
  return String(value)
}

export function NumericParameter({
  description,
  disabled = false,
  fieldKey,
  formatValue = defaultFormat,
  integer = false,
  label,
  logScale = false,
  max,
  min,
  onChange,
  onDraftValidityChange,
  step,
  value,
}: NumericParameterProps) {
  const inputId = useId()
  const descriptionId = useId()
  const errorId = useId()
  const [draft, setDraft] = useState(() => formatValue(value))
  const [error, setError] = useState<string | undefined>(undefined)
  const inputFocused = useRef(false)
  const bounds: NumericBounds = { min, max, integer }

  useEffect(() => {
    if (!inputFocused.current) setDraft(formatValue(value))
  }, [formatValue, value])

  function setDraftValue(next: string): void {
    setDraft(next)
    const result = parseNumericDraft(next, bounds)
    if (result.kind === "valid") {
      setError(undefined)
      onDraftValidityChange?.(fieldKey, true)
      onChange(result.value)
      return
    }
    const message =
      result.kind === "incomplete" ? "Finish entering a value in the allowed range." : result.reason
    setError(message)
    onDraftValidityChange?.(fieldKey, false)
  }

  function updateFromSlider(position: number): void {
    const rawValue = logScale ? logarithmicValueAt(position, min, max) : position
    const nextValue = roundTrainingStep(rawValue, min, step)
    setDraft(formatValue(nextValue))
    setError(undefined)
    onDraftValidityChange?.(fieldKey, true)
    onChange(nextValue)
  }

  const rangeValue = logScale
    ? String(Math.round(logarithmicPercent(value, min, max) * 1_000))
    : String(value)

  return (
    <div
      className={`training-parameter${error === undefined ? "" : " training-parameter--invalid"}`}
    >
      <div className="training-parameter__heading">
        <label htmlFor={inputId}>{label}</label>
        <output htmlFor={inputId}>{formatValue(value)}</output>
      </div>
      {description === undefined ? null : (
        <p className="training-parameter__description" id={descriptionId}>
          {description}
        </p>
      )}
      <input
        aria-label={`${label} slider`}
        aria-describedby={description === undefined ? undefined : descriptionId}
        className="training-parameter__range"
        disabled={disabled}
        max={logScale ? 1_000 : max}
        min={logScale ? 0 : min}
        onChange={(event) =>
          updateFromSlider(
            logScale
              ? Number(event.currentTarget.value) / 1_000
              : Number(event.currentTarget.value),
          )
        }
        step={logScale ? 1 : step}
        type="range"
        value={rangeValue}
      />
      <input
        aria-describedby={
          [
            description === undefined ? undefined : descriptionId,
            error === undefined ? undefined : errorId,
          ]
            .filter((item) => item !== undefined)
            .join(" ") || undefined
        }
        aria-invalid={error === undefined ? undefined : true}
        autoComplete="off"
        className="training-parameter__number"
        disabled={disabled}
        id={inputId}
        inputMode={integer ? "numeric" : "decimal"}
        onBlur={() => {
          inputFocused.current = false
          if (error === undefined) setDraft(formatValue(value))
        }}
        onChange={(event) => setDraftValue(event.currentTarget.value)}
        onFocus={() => {
          inputFocused.current = true
        }}
        type="text"
        value={draft}
      />
      {error === undefined ? null : (
        <p className="training-parameter__error" id={errorId} role="alert">
          {error}
        </p>
      )}
    </div>
  )
}
