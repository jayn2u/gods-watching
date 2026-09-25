import { describe, expect, it } from "vitest"
import { parseModelSettingsResponse, parseSwitchPreflight } from "./clientDomain"

const modelCatalog = {
  active_model_id: "openai/clip-vit-base-patch16",
  maintenance: false,
  models: [
    {
      model_id: "openai/clip-vit-base-patch16",
      display_name: "OpenAI CLIP ViT-B/16",
      revision: "fixture-revision",
      quality_passed: true,
      quality_reason: null,
      dimension: 512,
      prepared: true,
      reason: null,
    },
    {
      model_id: "openai/clip-vit-base-patch32",
      display_name: "OpenAI CLIP ViT-B/32",
      revision: "fixture-revision",
      quality_passed: true,
      quality_reason: null,
      dimension: 512,
      prepared: false,
      reason: "Model assets are not prepared.",
    },
    {
      model_id: "openai/clip-vit-large-patch14",
      display_name: "OpenAI CLIP ViT-L/14",
      revision: "fixture-revision",
      quality_passed: true,
      quality_reason: null,
      dimension: 768,
      prepared: true,
      reason: null,
    },
  ],
  transition: null,
} as const

const transition = {
  id: "switch-01",
  source_model_id: "openai/clip-vit-base-patch16",
  target_model_id: "openai/clip-vit-large-patch14",
  phase: "reindexing",
  processed: 14,
  total: 40,
  skipped: 2,
  skip_reasons: { missing_crop: 1, corrupt_crop: 1 },
  error: null,
} as const

describe("parseModelSettingsResponse", () => {
  it("parses the catalog and durable transition fields", () => {
    expect(parseModelSettingsResponse({ ...modelCatalog, transition })).toEqual({
      ...modelCatalog,
      transition,
    })
  })

  it("rejects an invalid phase or skip count instead of trusting server data", () => {
    expect(() =>
      parseModelSettingsResponse({
        ...modelCatalog,
        transition: { ...transition, phase: "downloading" },
      }),
    ).toThrow("model settings response has an invalid transition")
    expect(() =>
      parseModelSettingsResponse({
        ...modelCatalog,
        transition: { ...transition, skip_reasons: { corrupt_crop: -1 } },
      }),
    ).toThrow("model settings response has an invalid transition")
  })
})

describe("parseSwitchPreflight", () => {
  const preview = {
    target_model_id: "custom/one",
    retained_count: 40,
    estimated_missing_count: 2,
    measured_crops_per_second: 10,
    measured_fixed_seconds: 20,
    estimated_seconds: 23.8,
    max_seconds: 900,
    eligible: true,
    reason: null,
  }
  it("accepts a measured eligible preview", () => {
    expect(parseSwitchPreflight(preview)).toEqual(preview)
  })
  it("rejects inconsistent counts and unmeasured eligibility", () => {
    expect(() => parseSwitchPreflight({ ...preview, estimated_missing_count: 41 })).toThrow()
    expect(() => parseSwitchPreflight({ ...preview, measured_crops_per_second: null })).toThrow()
  })
})
