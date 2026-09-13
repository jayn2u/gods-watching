import { describe, expect, it } from "vitest"
import { parseQuotaGb, parseRetentionDays } from "./retentionValidation"

describe("parseRetentionDays", () => {
  it("parses a positive whole number", () => {
    // Given: an operator-entered retention value.
    // When: the value crosses the form boundary.
    const result = parseRetentionDays(" 21 ")

    // Then: the typed day count is available to the request builder.
    expect(result).toEqual({ kind: "valid", value: 21 })
  })

  it.each(["", "0", "1.5", "-2", "1e3", "days"])(
    "rejects malformed or non-positive input %s",
    (value) => {
      // Given: an invalid retention value.
      // When: the value crosses the form boundary.
      const result = parseRetentionDays(value)

      // Then: no request value is produced.
      expect(result.kind).toBe("invalid")
    },
  )
})

describe("parseQuotaGb", () => {
  it("converts decimal gigabytes using decimal bytes", () => {
    // Given: a positive decimal quota.
    // When: the value crosses the form boundary.
    const result = parseQuotaGb("100.25")

    // Then: the request uses 1,000,000,000 bytes per GB.
    expect(result).toEqual({ kind: "valid", value: 100_250_000_000 })
  })

  it.each(["", "0", "-1", "1e2", "1.2.3", "NaN"])(
    "rejects malformed or non-positive quota %s",
    (value) => {
      // Given: an invalid quota value.
      // When: the value crosses the form boundary.
      const result = parseQuotaGb(value)

      // Then: no request value is produced.
      expect(result.kind).toBe("invalid")
    },
  )
})
