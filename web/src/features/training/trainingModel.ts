import type { TrainingConfig, TrainingJobPhase, TrainingMemoryRefusal } from "../../app/clientTypes"

export type NumericBounds = Readonly<{
  min: number
  max: number
  integer?: boolean
}>

export type NumericDraftResult =
  | Readonly<{ kind: "valid"; value: number }>
  | Readonly<{ kind: "incomplete" }>
  | Readonly<{ kind: "invalid"; reason: string }>

export type TrainingRefusalPresentation = Readonly<{
  kind: "memory_shortage" | "unsupported_profile" | "dataset_changed" | "unknown"
  title: string
  message: string
}>

export const DEFAULT_TRAINING_CONFIG: TrainingConfig = Object.freeze({
  epochs: 30,
  learning_rate: 1e-5,
  micro_batch_size: 16,
  weight_decay: 0.01,
  gradient_accumulation: 4,
  warmup_ratio: 0.05,
  seed: 42,
  early_stopping_patience: 5,
  gradient_clipping_norm: 1,
  mixed_precision: "fp16",
  gradient_checkpointing: true,
})

export const TRAINING_PARAMETER_BOUNDS = Object.freeze({
  epochs: Object.freeze({ min: 1, max: 100, step: 1, integer: true }),
  learning_rate: Object.freeze({ min: 1e-7, max: 1e-3, step: 1e-7 }),
  micro_batch_size: Object.freeze({ min: 2, max: 128, step: 1, integer: true }),
  weight_decay: Object.freeze({ min: 0, max: 0.2, step: 0.001 }),
  gradient_accumulation: Object.freeze({ min: 1, max: 32, step: 1, integer: true }),
  warmup_ratio: Object.freeze({ min: 0, max: 0.3, step: 0.01 }),
  seed: Object.freeze({ min: 0, max: 2_147_483_647, step: 1, integer: true }),
  early_stopping_patience: Object.freeze({ min: 1, max: 20, step: 1, integer: true }),
  gradient_clipping_norm: Object.freeze({ min: 0.1, max: 10, step: 0.1 }),
})

export function logarithmicValueAt(position: number, minimum: number, maximum: number): number {
  const boundedPosition = Math.max(0, Math.min(1, position))
  return minimum * (maximum / minimum) ** boundedPosition
}

export function logarithmicPercent(value: number, minimum: number, maximum: number): number {
  const boundedValue = Math.max(minimum, Math.min(maximum, value))
  return Math.log(boundedValue / minimum) / Math.log(maximum / minimum)
}

function decimalPlaces(value: number): number {
  const [coefficient = "", exponentText] = value.toString().toLowerCase().split("e")
  const fractionLength = coefficient.split(".")[1]?.length ?? 0
  const exponent = exponentText === undefined ? 0 : Number(exponentText)
  return Math.max(0, fractionLength - exponent)
}

export function roundTrainingStep(value: number, minimum: number, step: number): number {
  const precision = Math.min(20, Math.max(decimalPlaces(minimum), decimalPlaces(step)))
  const quantized = minimum + Math.round((value - minimum) / step) * step
  return Number(quantized.toFixed(precision))
}

function displayNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : String(Number(value.toPrecision(6)))
}

export function parseNumericDraft(text: string, bounds: NumericBounds): NumericDraftResult {
  const normalized = text.trim()
  if (
    normalized === "" ||
    /^[+-]?(?:\d+\.?\d*|\.\d+)[eE][+-]?$/.test(normalized) ||
    /^[+-]?(?:\.)?$/.test(normalized)
  ) {
    return { kind: "incomplete" }
  }
  const value = Number(normalized)
  if (!Number.isFinite(value)) {
    return { kind: "invalid", reason: "Enter a valid number." }
  }
  if (bounds.integer && !Number.isInteger(value)) {
    return { kind: "invalid", reason: "Enter a whole number." }
  }
  if (value < bounds.min || value > bounds.max) {
    return {
      kind: "invalid",
      reason: `Enter a value from ${displayNumber(bounds.min)} to ${displayNumber(bounds.max)}.`,
    }
  }
  return { kind: "valid", value }
}

export function freezeTrainingConfig(config: TrainingConfig): TrainingConfig {
  return Object.freeze({ ...config })
}

export class TrainingGeneration {
  private current = 0

  next(): number {
    this.current += 1
    return this.current
  }

  isCurrent(generation: number): boolean {
    return generation === this.current
  }

  invalidate(): void {
    this.current += 1
  }
}

export function canCancelTrainingJob(phase: TrainingJobPhase): boolean {
  return (
    phase === "starting" || phase === "training" || phase === "evaluating" || phase === "publishing"
  )
}

export function canResumeTrainingJob(phase: TrainingJobPhase): boolean {
  return phase === "interrupted"
}

export function classifyTrainingRefusal(
  refusal: Pick<TrainingMemoryRefusal, "code" | "reason">,
): TrainingRefusalPresentation {
  if (refusal.code !== "training_memory_refused") {
    return {
      kind: "unknown",
      title: "Training was not admitted",
      message:
        "The server returned an unrecognized admission result. Refresh the estimate and try again.",
    }
  }
  switch (refusal.reason) {
    case "insufficient_free_memory":
      return {
        kind: "memory_shortage",
        title: "GPU memory is currently insufficient",
        message:
          "The server refused this run to preserve memory for live inference. Reduce the micro batch size or change the approved precision and checkpointing settings.",
      }
    case "memory_profile_unsupported":
      return {
        kind: "unsupported_profile",
        title: "This configuration has no calibrated memory profile",
        message:
          "The server cannot safely estimate this GPU and configuration combination. Choose a supported batch size, precision, or checkpointing setting and refresh the estimate.",
      }
    case "registered_dataset_fingerprint_mismatch":
      return {
        kind: "dataset_changed",
        title: "The registered dataset changed",
        message:
          "The CUHK-PEDES snapshot no longer matches the server's validated dataset. Refresh dataset status before starting another run.",
      }
    default:
      return {
        kind: "unknown",
        title: "Training was not admitted",
        message:
          "The server returned an unrecognized admission result. Refresh the estimate or contact the operator.",
      }
  }
}
