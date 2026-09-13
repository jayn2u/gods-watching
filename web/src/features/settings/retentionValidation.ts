export const BYTES_PER_DECIMAL_GB = 1_000_000_000

export type ParsedRetention<T> =
  | Readonly<{ kind: "valid"; value: T }>
  | Readonly<{ kind: "invalid"; message: string }>

export function parseRetentionDays(rawValue: string): ParsedRetention<number> {
  const value = rawValue.trim()
  if (!/^\d+$/u.test(value)) {
    return { kind: "invalid", message: "Retention days must be a positive whole number." }
  }
  const days = Number(value)
  return Number.isSafeInteger(days) && days > 0
    ? { kind: "valid", value: days }
    : { kind: "invalid", message: "Retention days must be a positive whole number." }
}

export function parseQuotaGb(rawValue: string): ParsedRetention<number> {
  const value = rawValue.trim()
  if (!/^(?:\d+(?:\.\d+)?|\.\d+)$/u.test(value)) {
    return { kind: "invalid", message: "Quota must be a positive decimal number of GB." }
  }
  const gigabytes = Number(value)
  if (!Number.isFinite(gigabytes) || gigabytes <= 0) {
    return { kind: "invalid", message: "Quota must be a positive decimal number of GB." }
  }
  const bytes = gigabytes * BYTES_PER_DECIMAL_GB
  return Number.isSafeInteger(bytes)
    ? { kind: "valid", value: bytes }
    : { kind: "invalid", message: "Quota is outside the supported positive range." }
}
