import type { AppearanceResponse } from "../../app/client"
import { Button } from "../../components/Button"
import { Panel } from "../../components/Panel"
import { Status } from "../../components/Status"
import { SearchResults } from "./SearchResults"
import type { SearchClient } from "./searchClient"
import { formatLocalTime, formatScore } from "./searchModel"
import type { DetailState, ResultsState } from "./searchTypes"

export type SearchDetailProps = Readonly<{
  client: SearchClient
  detail: DetailState
  onBack: () => void
  onFindSimilar: () => void
  onOpen: (appearance: AppearanceResponse) => void
  onRetry: () => void
  onRetrySimilar: () => void
  onUnauthorized: () => void
  similar: ResultsState
}>

export function SearchDetail({
  client,
  detail,
  onBack,
  onFindSimilar,
  onOpen,
  onRetry,
  onRetrySimilar,
  onUnauthorized,
  similar,
}: SearchDetailProps) {
  if (detail.kind === "closed") {
    return null
  }
  const appearance =
    detail.kind === "ready"
      ? detail.appearance
      : detail.kind === "error"
        ? detail.appearance
        : detail.seed
  const similarLoading = similar.kind === "loading"

  return (
    <Panel
      actions={
        <Button onClick={onBack} type="button">
          Back to results
        </Button>
      }
      eyebrow="Appearance detail"
      title={appearance.camera_name}
    >
      <div className="search-detail">
        <div className="search-detail__media">
          {detail.kind === "loading" ? (
            <div
              aria-label="Loading appearance detail"
              className="search-detail__crop search-detail__crop--loading"
              role="status"
            />
          ) : null}
          {detail.kind === "error" ? (
            <div className="search-detail__crop search-detail__crop--error" role="alert">
              <Status announce={false} tone="offline">
                Crop unavailable
              </Status>
              <p>{detail.message}</p>
              <Button onClick={onRetry} type="button">
                Retry detail
              </Button>
            </div>
          ) : null}
          {detail.kind === "ready" ? (
            <img
              alt={`Person crop from ${appearance.camera_name} at ${formatLocalTime(appearance.first_seen)}`}
              className="search-detail__crop"
              height="640"
              src={detail.crop.url}
              width="512"
            />
          ) : null}
        </div>
        <div className="search-detail__copy">
          <div className="search-detail__actions">
            <Button
              disabled={similarLoading}
              loading={similarLoading}
              onClick={onFindSimilar}
              type="button"
            >
              Find similar
            </Button>
          </div>
          <dl className="search-detail__metadata">
            <div>
              <dt>Camera</dt>
              <dd>{appearance.camera_name}</dd>
            </div>
            <div>
              <dt>First seen</dt>
              <dd>{formatLocalTime(appearance.first_seen)}</dd>
            </div>
            <div>
              <dt>Last seen</dt>
              <dd>{formatLocalTime(appearance.last_seen)}</dd>
            </div>
            <div>
              <dt>Track</dt>
              <dd>{appearance.track_id}</dd>
            </div>
            <div>
              <dt>Detector confidence</dt>
              <dd>{formatScore(appearance.detector_confidence)}</dd>
            </div>
            <div>
              <dt>Cosine similarity</dt>
              <dd>{formatScore(appearance.similarity)}</dd>
            </div>
            <div>
              <dt>Representative</dt>
              <dd>Version {appearance.representative_version}</dd>
            </div>
            <div>
              <dt>Crop quality</dt>
              <dd>{appearance.crop_quality.toFixed(2)}</dd>
            </div>
          </dl>
          <SimilarResults
            client={client}
            onOpen={onOpen}
            onRetry={onRetrySimilar}
            onUnauthorized={onUnauthorized}
            state={similar}
          />
        </div>
      </div>
    </Panel>
  )
}

function SimilarResults({
  client,
  onOpen,
  onRetry,
  onUnauthorized,
  state,
}: Readonly<{
  client: SearchClient
  onOpen: (appearance: AppearanceResponse) => void
  onRetry: () => void
  onUnauthorized: () => void
  state: ResultsState
}>) {
  if (state.kind === "idle") {
    return (
      <p className="search-detail__similar-hint">
        Find similar uses this committed appearance crop.
      </p>
    )
  }
  if (state.kind === "loading") {
    return (
      <div className="search-detail__similar-status" aria-busy="true">
        <Status tone="neutral">Finding similar appearances</Status>
      </div>
    )
  }
  if (state.kind === "error") {
    return (
      <div className="search-detail__similar-status" role="alert">
        <p>{state.message}</p>
        <Button onClick={onRetry} type="button">
          Retry similar search
        </Button>
      </div>
    )
  }
  if (state.kind === "empty") {
    return (
      <div className="search-detail__similar-status">
        <Status tone="neutral">No similar appearances</Status>
        <p>The service returned no committed appearances for this crop.</p>
      </div>
    )
  }
  return (
    <section aria-labelledby="similar-heading" className="search-detail__similar">
      <div className="search-detail__similar-heading">
        <h3 id="similar-heading">Similar appearances</h3>
        <Status tone="live">Cosine ranked</Status>
      </div>
      <SearchResults
        client={client}
        onOpen={onOpen}
        onUnauthorized={onUnauthorized}
        results={state.results}
      />
    </section>
  )
}
