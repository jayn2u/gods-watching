import { useCallback, useEffect, useRef, useState } from "react"
import {
  type AppearanceResponse,
  isAbortError,
  type SearchRequest,
  type SearchResponse,
} from "../../app/client"
import type { CameraState } from "../../app/layout/AppShell"
import { RequestFence } from "../../app/requestFence"
import { Button } from "../../components/Button"
import { Panel } from "../../components/Panel"
import { Status } from "../../components/Status"
import { SearchDetail } from "./SearchDetail"
import { SearchForm } from "./SearchForm"
import { SearchResults } from "./SearchResults"
import type { SearchClient } from "./searchClient"
import { CropVersionMismatchError, describeSearchError } from "./searchErrors"
import {
  buildSearchRequest,
  buildSimilarRequest,
  type SearchDraft,
  type SearchMode,
} from "./searchModel"
import type { DetailState, ResultsState } from "./searchTypes"
import "./search.css"

export type SearchScreenProps = Readonly<{
  cameras: CameraState
  client: SearchClient
  onUnauthorized: () => void
}>

type SearchTarget = "results" | "similar"

function nextResultsState(request: SearchRequest, response: SearchResponse): ResultsState {
  if (request.mode !== response.mode) {
    return {
      kind: "error",
      message: "search_response_mismatch: the service returned a different search mode.",
      request,
    }
  }
  return response.results.length === 0
    ? { kind: "empty", request }
    : { kind: "ready", request, results: response.results }
}

function initialDraft(): SearchDraft {
  return { cameraIds: [], from: "", mode: "text", query: "", to: "" }
}

export function SearchScreen({ cameras, client, onUnauthorized }: SearchScreenProps) {
  const [draft, setDraft] = useState<SearchDraft>(initialDraft)
  const [formError, setFormError] = useState<string | undefined>(undefined)
  const [results, setResults] = useState<ResultsState>({ kind: "idle" })
  const [similar, setSimilar] = useState<ResultsState>({ kind: "idle" })
  const [detail, setDetail] = useState<DetailState>({ kind: "closed" })
  const [view, setView] = useState<"results" | "detail">("results")
  const searchFence = useRef(new RequestFence())
  const detailFence = useRef(new RequestFence())

  const runSearch = useCallback(
    (request: SearchRequest, target: SearchTarget): void => {
      const token = searchFence.current.start()
      if (target === "results") {
        detailFence.current.cancel()
        setDetail({ kind: "closed" })
        setSimilar({ kind: "idle" })
        setView("results")
        setResults({ kind: "loading", request })
      } else {
        setSimilar({ kind: "loading", request })
      }
      void client
        .search(request, token.signal)
        .then((response) => {
          if (!searchFence.current.isCurrent(token.generation)) {
            return
          }
          const next = nextResultsState(request, response)
          if (target === "results") {
            setResults(next)
          } else {
            setSimilar(next)
          }
        })
        .catch((error: unknown) => {
          if (!searchFence.current.isCurrent(token.generation) || isAbortError(error)) {
            return
          }
          const message = describeSearchError(error, onUnauthorized)
          if (message === undefined) {
            return
          }
          const next: ResultsState = { kind: "error", message, request }
          if (target === "results") {
            setResults(next)
          } else {
            setSimilar(next)
          }
        })
    },
    [client, onUnauthorized],
  )

  const loadDetail = useCallback(
    (seed: AppearanceResponse): void => {
      const token = detailFence.current.start()
      setView("detail")
      setSimilar({ kind: "idle" })
      setDetail({ kind: "loading", seed })
      void client
        .getAppearance(seed.appearance_id, token.signal)
        .then(async (appearance) => {
          if (!detailFence.current.isCurrent(token.generation)) {
            return
          }
          const crop = await client.getAppearanceCrop(appearance.appearance_id, token.signal)
          if (!detailFence.current.isCurrent(token.generation)) {
            return
          }
          if (
            crop.representative_version !== undefined &&
            crop.representative_version !== appearance.representative_version
          ) {
            throw new CropVersionMismatchError()
          }
          const url = URL.createObjectURL(crop.blob)
          setDetail({
            kind: "ready",
            appearance,
            crop: {
              etag: crop.etag,
              representativeVersion: crop.representative_version,
              url,
            },
          })
        })
        .catch((error: unknown) => {
          if (!detailFence.current.isCurrent(token.generation) || isAbortError(error)) {
            return
          }
          const message = describeSearchError(error, onUnauthorized)
          if (message !== undefined) {
            setDetail({ kind: "error", appearance: seed, message })
          }
        })
    },
    [client, onUnauthorized],
  )

  useEffect(() => {
    return () => {
      searchFence.current.dispose()
      detailFence.current.dispose()
    }
  }, [])

  useEffect(() => {
    return () => {
      if (detail.kind === "ready") {
        URL.revokeObjectURL(detail.crop.url)
      }
    }
  }, [detail])

  const submit = useCallback((): void => {
    const built = buildSearchRequest(draft)
    if (built.kind === "invalid") {
      setFormError(built.message)
      return
    }
    setFormError(undefined)
    runSearch(built.request, "results")
  }, [draft, runSearch])

  const browse = useCallback((): void => {
    const built = buildSearchRequest({ ...draft, mode: "browse" })
    if (built.kind === "invalid") {
      setFormError(built.message)
      return
    }
    setFormError(undefined)
    runSearch(built.request, "results")
  }, [draft, runSearch])

  const findSimilar = useCallback((): void => {
    const appearance =
      detail.kind === "loading" || detail.kind === "ready" || detail.kind === "error"
        ? detail.kind === "loading"
          ? detail.seed
          : detail.appearance
        : undefined
    if (appearance === undefined) {
      return
    }
    const built = buildSimilarRequest(appearance.appearance_id, {
      cameraIds: draft.cameraIds,
      from: draft.from,
      to: draft.to,
    })
    if (built.kind === "invalid") {
      setFormError(built.message)
      return
    }
    setFormError(undefined)
    runSearch(built.request, "similar")
  }, [detail, draft.cameraIds, draft.from, draft.to, runSearch])

  const cancelSearch = useCallback((): void => {
    searchFence.current.cancel()
    setResults((current) => (current.kind === "loading" ? { kind: "idle" } : current))
    setSimilar((current) => (current.kind === "loading" ? { kind: "idle" } : current))
  }, [])

  const toggleCamera = useCallback((cameraId: string): void => {
    setDraft((current) => ({
      ...current,
      cameraIds: current.cameraIds.includes(cameraId)
        ? current.cameraIds.filter((selected) => selected !== cameraId)
        : [...current.cameraIds, cameraId],
    }))
  }, [])

  const pending = results.kind === "loading" || similar.kind === "loading"

  return (
    <main aria-labelledby="search-heading" className="search-main">
      <div className="search-main__heading">
        <div>
          <p className="app-kicker">Retrieval console</p>
          <h1 id="search-heading">Person search</h1>
          <p>
            Search committed person crops with an English description, camera scope, and local time
            window.
          </p>
        </div>
        <Status tone={pending ? "warning" : "neutral"}>
          {pending ? "Request pending" : "Ready"}
        </Status>
      </div>
      <Panel eyebrow="Describe and filter" title="Search controls">
        <SearchForm
          cameras={cameras}
          draft={draft}
          error={formError}
          onBrowse={browse}
          onCameraToggle={toggleCamera}
          onCancel={cancelSearch}
          onFromChange={(from) => setDraft((current) => ({ ...current, from }))}
          onModeChange={(mode: SearchMode) => setDraft((current) => ({ ...current, mode }))}
          onQueryChange={(query) => setDraft((current) => ({ ...current, query }))}
          onSubmit={submit}
          onToChange={(to) => setDraft((current) => ({ ...current, to }))}
          pending={pending}
        />
      </Panel>
      {view === "detail" ? (
        <SearchDetail
          client={client}
          detail={detail}
          onBack={() => {
            detailFence.current.cancel()
            setDetail({ kind: "closed" })
            setSimilar({ kind: "idle" })
            setView("results")
          }}
          onFindSimilar={findSimilar}
          onOpen={loadDetail}
          onRetry={() => {
            if (detail.kind === "error") {
              loadDetail(detail.appearance)
            }
          }}
          onRetrySimilar={() => {
            if (similar.kind === "error") {
              runSearch(similar.request, "similar")
            }
          }}
          onUnauthorized={onUnauthorized}
          similar={similar}
        />
      ) : (
        <ResultsPanel
          client={client}
          onOpen={loadDetail}
          onRetry={(request) => runSearch(request, "results")}
          onUnauthorized={onUnauthorized}
          state={results}
        />
      )}
    </main>
  )
}

function ResultsPanel({
  client,
  onOpen,
  onRetry,
  onUnauthorized,
  state,
}: Readonly<{
  client: SearchClient
  onOpen: (appearance: AppearanceResponse) => void
  onRetry: (request: SearchRequest) => void
  onUnauthorized: () => void
  state: ResultsState
}>) {
  return (
    <Panel eyebrow="Committed appearances" title="Results">
      {state.kind === "idle" ? (
        <div className="search-state">
          <Status tone="neutral">Waiting for a search</Status>
          <p>Enter a description and choose Search, or explicitly browse the newest appearances.</p>
        </div>
      ) : null}
      {state.kind === "loading" ? (
        <div aria-busy="true" className="search-state">
          <Status tone="warning">Searching committed appearances</Status>
          <p>Results will replace this pending state when the service responds.</p>
        </div>
      ) : null}
      {state.kind === "error" ? (
        <div className="search-state" role="alert">
          <Status tone="offline">Search unavailable</Status>
          <p>{state.message}</p>
          <Button onClick={() => onRetry(state.request)} type="button">
            Retry search
          </Button>
        </div>
      ) : null}
      {state.kind === "empty" ? (
        <div className="search-state">
          <Status tone="neutral">No appearances found</Status>
          <p>The service returned no committed crops for these filters.</p>
        </div>
      ) : null}
      {state.kind === "ready" ? (
        <>
          <div className="search-results__summary">
            <Status tone="live">
              {state.request.mode === "browse" ? "Newest first" : "Cosine ranked"}
            </Status>
            <span>{state.results.length} appearance(s)</span>
          </div>
          <SearchResults
            client={client}
            onOpen={onOpen}
            onUnauthorized={onUnauthorized}
            results={state.results}
          />
        </>
      ) : null}
    </Panel>
  )
}
