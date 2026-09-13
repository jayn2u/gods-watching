import { useEffect, useRef, useState } from "react"
import type { AppearanceResponse } from "../../app/client"
import { isAbortError } from "../../app/client"
import { Button } from "../../components/Button"
import type { SearchClient } from "./searchClient"
import { describeSearchError } from "./searchErrors"
import { formatLocalTime, formatScore } from "./searchModel"

type CropState =
  | Readonly<{ kind: "loading" }>
  | Readonly<{ kind: "ready"; url: string }>
  | Readonly<{ kind: "error"; message: string }>

export type SearchResultsProps = Readonly<{
  client: SearchClient
  onOpen: (appearance: AppearanceResponse) => void
  onUnauthorized: () => void
  results: readonly AppearanceResponse[]
}>

export function SearchResults({ client, onOpen, onUnauthorized, results }: SearchResultsProps) {
  return (
    <ul aria-label="Search results" className="search-results">
      {results.map((appearance) => (
        <ResultCard
          appearance={appearance}
          client={client}
          key={appearance.appearance_id}
          onOpen={onOpen}
          onUnauthorized={onUnauthorized}
        />
      ))}
    </ul>
  )
}

export function ResultCard({
  appearance,
  client,
  onOpen,
  onUnauthorized,
}: Readonly<{
  appearance: AppearanceResponse
  client: SearchClient
  onOpen: (appearance: AppearanceResponse) => void
  onUnauthorized: () => void
}>) {
  return (
    <li className="search-result">
      <AppearanceCrop appearance={appearance} client={client} onUnauthorized={onUnauthorized} />
      <div className="search-result__body">
        <div className="search-result__title-row">
          <h3>{appearance.camera_name}</h3>
          <span className="search-result__track">Track {appearance.track_id}</span>
        </div>
        <p className="search-result__time">{formatLocalTime(appearance.first_seen)}</p>
        <dl className="search-result__scores">
          <div>
            <dt>Detector confidence</dt>
            <dd>{formatScore(appearance.detector_confidence)}</dd>
          </div>
          <div>
            <dt>Cosine similarity</dt>
            <dd>{formatScore(appearance.similarity)}</dd>
          </div>
        </dl>
        <Button onClick={() => onOpen(appearance)} type="button">
          View details
        </Button>
      </div>
    </li>
  )
}

function AppearanceCrop({
  appearance,
  client,
  onUnauthorized,
}: Readonly<{
  appearance: AppearanceResponse
  client: SearchClient
  onUnauthorized: () => void
}>) {
  const [state, setState] = useState<CropState>({ kind: "loading" })
  const urlRef = useRef<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    setState({ kind: "loading" })
    void client
      .getAppearanceCrop(appearance.appearance_id, controller.signal)
      .then((response) => {
        if (
          response.representative_version !== undefined &&
          response.representative_version !== appearance.representative_version
        ) {
          setState({ kind: "error", message: "The crop was updated. Open details to retry." })
          return
        }
        const url = URL.createObjectURL(response.blob)
        const previousUrl = urlRef.current
        if (previousUrl !== null) {
          URL.revokeObjectURL(previousUrl)
        }
        urlRef.current = url
        setState({ kind: "ready", url })
      })
      .catch((error: unknown) => {
        if (isAbortError(error)) {
          return
        }
        const message = describeSearchError(error, onUnauthorized)
        if (message !== undefined) {
          setState({ kind: "error", message })
        }
      })

    return () => {
      controller.abort()
      const url = urlRef.current
      if (url !== null) {
        URL.revokeObjectURL(url)
        urlRef.current = null
      }
    }
  }, [appearance.appearance_id, appearance.representative_version, client, onUnauthorized])

  if (state.kind === "loading") {
    return (
      <div
        aria-label="Loading crop"
        className="search-result__crop search-result__crop--loading"
        role="status"
      />
    )
  }
  if (state.kind === "error") {
    return (
      <div className="search-result__crop search-result__crop--error" role="status">
        <span>Crop unavailable</span>
        <small>{state.message}</small>
      </div>
    )
  }
  return (
    <div className="search-result__crop">
      <img
        alt={`Person crop from ${appearance.camera_name} at ${formatLocalTime(appearance.first_seen)}`}
        height="400"
        loading="lazy"
        src={state.url}
        width="320"
      />
    </div>
  )
}
