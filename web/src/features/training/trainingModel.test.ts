import { describe, expect, it } from "vitest"
import {
  parseTrainingConfig,
  parseTrainingMemoryRefusal,
  parseTrainingPreflightResponse,
} from "../../app/clientTrainingDomain"
import {
  classifyTrainingRefusal,
  DEFAULT_TRAINING_CONFIG,
  freezeTrainingConfig,
  logarithmicPercent,
  logarithmicValueAt,
  parseNumericDraft,
  roundTrainingStep,
  TrainingGeneration,
} from "./trainingModel"

describe("training configuration controls", () => {
  it("maps the learning-rate slider logarithmically across the approved range", () => {
    expect(logarithmicValueAt(0, 1e-7, 1e-3)).toBe(1e-7)
    expect(logarithmicValueAt(0.5, 1e-7, 1e-3)).toBeCloseTo(1e-5, 14)
    expect(logarithmicValueAt(1, 1e-7, 1e-3)).toBe(1e-3)
    expect(logarithmicPercent(1e-5, 1e-7, 1e-3)).toBeCloseTo(0.5, 12)
    expect(roundTrainingStep(logarithmicValueAt(0.5, 1e-7, 1e-3), 1e-7, 1e-7)).toBe(1e-5)
  })

  it("keeps incomplete numeric text distinct from valid values", () => {
    expect(parseNumericDraft("1e-", { min: 1e-7, max: 1e-3 })).toEqual({ kind: "incomplete" })
    expect(parseNumericDraft("", { min: 1, max: 100, integer: true })).toEqual({
      kind: "incomplete",
    })
    expect(parseNumericDraft("1.5", { min: 1, max: 100, integer: true })).toEqual({
      kind: "invalid",
      reason: "Enter a whole number.",
    })
    expect(parseNumericDraft("101", { min: 1, max: 100, integer: true })).toEqual({
      kind: "invalid",
      reason: "Enter a value from 1 to 100.",
    })
  })

  it("freezes an independent submitted snapshot with the approved defaults", () => {
    expect(DEFAULT_TRAINING_CONFIG).toMatchObject({
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

    const editable = { ...DEFAULT_TRAINING_CONFIG }
    const submitted = freezeTrainingConfig(editable)
    editable.epochs = 60

    expect(submitted.epochs).toBe(30)
    expect(Object.isFrozen(submitted)).toBe(true)
  })

  it("fences older request generations after a newer estimate starts", () => {
    const generations = new TrainingGeneration()
    const first = generations.next()
    const second = generations.next()

    expect(generations.isCurrent(first)).toBe(false)
    expect(generations.isCurrent(second)).toBe(true)
    generations.invalidate()
    expect(generations.isCurrent(second)).toBe(false)
  })

  it("validates API configuration and memory decisions from unknown JSON", () => {
    expect(parseTrainingConfig(DEFAULT_TRAINING_CONFIG)).toEqual(DEFAULT_TRAINING_CONFIG)
    expect(() => parseTrainingConfig({ ...DEFAULT_TRAINING_CONFIG, epochs: 101 })).toThrow()

    const estimate = {
      admitted: true,
      training_peak_bytes: 3,
      reserve_bytes: 2,
      required_bytes: 5,
      free_bytes: 10,
      profile_identity: "b".repeat(64),
      observed_at: "2026-10-04T12:30:00Z",
      reason: "admitted",
    }
    expect(parseTrainingPreflightResponse(estimate)).toEqual(estimate)
    const refusedEstimate = {
      ...estimate,
      admitted: false,
      free_bytes: 4,
      reason: "insufficient_free_memory",
    }
    expect(parseTrainingPreflightResponse(refusedEstimate)).toEqual(refusedEstimate)

    const refusal = {
      code: "training_memory_refused",
      message: "training would exceed currently available GPU memory",
      reason: "insufficient_free_memory",
      training_peak_bytes: 3,
      reserve_bytes: 2,
      required_bytes: 5,
      free_bytes: 4,
      profile_identity: "b".repeat(64),
      observed_at: "2026-10-04T12:30:00Z",
    }
    expect(parseTrainingMemoryRefusal(refusal)).toEqual(refusal)
    expect(
      parseTrainingMemoryRefusal({ ...refusal, reason: "memory_profile_unsupported" })?.reason,
    ).toBe("memory_profile_unsupported")
    expect(parseTrainingMemoryRefusal({ ...refusal, observed_at: "today" })).toBeNull()
    expect(parseTrainingMemoryRefusal({ ...refusal, profile_identity: "untrusted" })).toBeNull()
  })

  it("classifies only known server refusal reasons into safe operator copy", () => {
    const base = {
      code: "training_memory_refused",
      message: "server message",
      reason: "insufficient_free_memory",
      training_peak_bytes: 3,
      reserve_bytes: 2,
      required_bytes: 5,
      free_bytes: 4,
    }
    expect(classifyTrainingRefusal(base)).toMatchObject({ kind: "memory_shortage" })
    expect(
      classifyTrainingRefusal({ ...base, reason: "memory_profile_unsupported" }),
    ).toMatchObject({ kind: "unsupported_profile" })
    expect(
      classifyTrainingRefusal({ ...base, reason: "registered_dataset_fingerprint_mismatch" }),
    ).toMatchObject({ kind: "dataset_changed" })
    expect(
      classifyTrainingRefusal({ ...base, reason: "unrecognized_private_reason" }),
    ).toMatchObject({
      kind: "unknown",
    })
  })
})
