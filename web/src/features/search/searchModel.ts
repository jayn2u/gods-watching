import type {
  BrowseSearchRequest,
  SearchFilters,
  SearchRequest,
  SimilarSearchRequest,
  TextSearchRequest,
} from "../../app/client"

export type SearchMode = "text" | "browse"

export type SearchDraft = Readonly<{
  cameraIds: readonly string[]
  from: string
  mode: SearchMode
  query: string
  to: string
}>

export type SearchBuildResult =
  | Readonly<{ kind: "ok"; request: SearchRequest }>
  | Readonly<{ kind: "invalid"; message: string }>

const SEARCH_LIMIT = 30

export function normalizeQuery(value: string): string {
  return value.normalize("NFKC").trim().split(/\s+/u).join(" ")
}

export function localDateTimeToUtc(value: string): string | undefined {
  if (value === "") {
    return undefined
  }
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})$/u.exec(value)
  if (match === null) {
    return undefined
  }
  const year = Number(match[1])
  const month = Number(match[2])
  const day = Number(match[3])
  const hour = Number(match[4])
  const minute = Number(match[5])
  if (year < 1000 || month < 1 || month > 12 || hour > 23 || minute > 59) {
    return undefined
  }
  const date = new Date(year, month - 1, day, hour, minute)
  if (
    date.getFullYear() !== year ||
    date.getMonth() !== month - 1 ||
    date.getDate() !== day ||
    date.getHours() !== hour ||
    date.getMinutes() !== minute
  ) {
    return undefined
  }
  return date.toISOString()
}

function buildFilters(draft: SearchDraft): SearchFilters | Readonly<{ error: string }> {
  const from = localDateTimeToUtc(draft.from)
  if (draft.from !== "" && from === undefined) {
    return { error: "Use a valid local date and time for the from filter." }
  }
  const to = localDateTimeToUtc(draft.to)
  if (draft.to !== "" && to === undefined) {
    return { error: "Use a valid local date and time for the to filter." }
  }
  return {
    ...(draft.cameraIds.length === 0 ? {} : { camera_ids: draft.cameraIds }),
    ...(from === undefined ? {} : { from }),
    limit: SEARCH_LIMIT,
    ...(to === undefined ? {} : { to }),
  }
}

export function buildSearchRequest(draft: SearchDraft): SearchBuildResult {
  const filters = buildFilters(draft)
  if ("error" in filters) {
    return { kind: "invalid", message: filters.error }
  }
  if (draft.mode === "browse") {
    const request: BrowseSearchRequest = { ...filters, mode: "browse", sort: "newest" }
    return { kind: "ok", request }
  }
  const query = normalizeQuery(draft.query)
  if (query === "") {
    return { kind: "invalid", message: "Enter a person description before searching." }
  }
  const request: TextSearchRequest = { ...filters, mode: "text", query, sort: "similarity" }
  return { kind: "ok", request }
}

export function buildSimilarRequest(
  appearanceId: string,
  draft: Omit<SearchDraft, "mode" | "query">,
): SearchBuildResult {
  const filters = buildFilters({ ...draft, mode: "browse", query: "" })
  if ("error" in filters) {
    return { kind: "invalid", message: filters.error }
  }
  const request: SimilarSearchRequest = {
    ...filters,
    appearance_id: appearanceId,
    mode: "similar",
    sort: "similarity",
  }
  return { kind: "ok", request }
}

export function formatLocalTime(value: string): string {
  const date = new Date(value)
  return Number.isNaN(date.getTime())
    ? "Time unavailable"
    : new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date)
}

export function formatScore(value: number | null): string {
  return value === null ? "Unavailable" : value.toFixed(2)
}
