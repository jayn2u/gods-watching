import { useCallback, useEffect, useRef, useState } from "react"
import type { TrainingJobResponse } from "../../app/client"
import { type ApiClient, HttpError, isAbortError } from "../../app/client"
import { Button, Panel, Status } from "../../components"

const PAGE_SIZE = 50

type TrainingHistoryProps = Readonly<{
  client: ApiClient
  onUnauthorized: () => void
  onSelect: (jobId: string) => void
  refreshToken: number
}>

function phaseLabel(phase: TrainingJobResponse["phase"]): string {
  switch (phase) {
    case "starting":
      return "Starting"
    case "training":
      return "Training"
    case "evaluating":
      return "Evaluating"
    case "publishing":
      return "Publishing candidate"
    case "succeeded":
      return "Succeeded"
    case "cancelling":
      return "Cancelling"
    case "cancelled":
      return "Cancelled"
    case "failed":
      return "Failed"
    case "interrupted":
      return "Interrupted"
  }
}

function dateLabel(value: string): string {
  return new Date(value).toLocaleString()
}

export function TrainingHistory({
  client,
  onSelect,
  onUnauthorized,
  refreshToken,
}: TrainingHistoryProps) {
  const [items, setItems] = useState<readonly TrainingJobResponse[]>([])
  const [nextCursor, setNextCursor] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState<string | undefined>(undefined)
  const generation = useRef(0)
  const controllerRef = useRef<AbortController | null>(null)
  const loadingMoreRef = useRef(false)
  const loadMoreOwner = useRef(0)

  const refresh = useCallback((): void => {
    loadMoreOwner.current += 1
    loadingMoreRef.current = false
    setLoadingMore(false)
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    const requestGeneration = generation.current + 1
    generation.current = requestGeneration
    setLoading(true)
    setError(undefined)
    void client
      .listTrainingJobs(PAGE_SIZE, null, controller.signal)
      .then((page) => {
        if (requestGeneration !== generation.current || controller.signal.aborted) return
        setItems(page.items)
        setNextCursor(page.next_cursor)
      })
      .catch((requestError: unknown) => {
        if (requestGeneration !== generation.current || isAbortError(requestError)) return
        if (requestError instanceof HttpError && requestError.status === 401) {
          onUnauthorized()
          return
        }
        setError("Training history could not be loaded. Retry to refresh the saved jobs.")
      })
      .finally(() => {
        if (requestGeneration === generation.current) {
          setLoading(false)
          if (controllerRef.current === controller) controllerRef.current = null
        }
      })
  }, [client, onUnauthorized])

  useEffect(() => {
    refresh()
    return () => {
      generation.current += 1
      controllerRef.current?.abort()
      controllerRef.current = null
    }
  }, [refresh])

  useEffect(() => {
    if (refreshToken > 0) refresh()
  }, [refresh, refreshToken])

  function loadMore(): void {
    if (nextCursor === null || loadingMoreRef.current) return
    loadingMoreRef.current = true
    const requestOwner = loadMoreOwner.current + 1
    loadMoreOwner.current = requestOwner
    const controller = new AbortController()
    controllerRef.current?.abort()
    controllerRef.current = controller
    const requestGeneration = generation.current + 1
    generation.current = requestGeneration
    setLoadingMore(true)
    void client
      .listTrainingJobs(PAGE_SIZE, nextCursor, controller.signal)
      .then((page) => {
        if (requestGeneration !== generation.current || controller.signal.aborted) return
        setItems((current) => [...current, ...page.items])
        setNextCursor(page.next_cursor)
      })
      .catch((requestError: unknown) => {
        if (requestGeneration !== generation.current || isAbortError(requestError)) return
        if (requestError instanceof HttpError && requestError.status === 401) {
          onUnauthorized()
          return
        }
        setError("Older training history could not be loaded. Retry the page.")
      })
      .finally(() => {
        if (requestOwner === loadMoreOwner.current) {
          loadingMoreRef.current = false
          setLoadingMore(false)
          if (controllerRef.current === controller) controllerRef.current = null
        }
      })
  }

  return (
    <Panel eyebrow="Durable jobs" title="Training history">
      <div className="training-history">
        <div className="training-history__heading">
          <p className="training-muted">
            Saved settings, progress, and final evaluations reload from the server.
          </p>
          <Button disabled={loading} onClick={refresh} variant="secondary">
            Refresh
          </Button>
        </div>
        {error === undefined ? null : (
          <p className="training-form__error" role="alert">
            {error}
          </p>
        )}
        {loading && items.length === 0 ? (
          <p className="training-muted" role="status">
            Loading saved runs…
          </p>
        ) : null}
        {!loading && items.length === 0 && error === undefined ? (
          <p className="training-muted">No training runs have been submitted.</p>
        ) : null}
        <div className="training-history__list">
          {items.map((job) => (
            <button
              className="training-history__item"
              key={job.id}
              onClick={() => onSelect(job.id)}
              type="button"
            >
              <span className="training-history__item-title">
                <strong>{dateLabel(job.created_at)}</strong>
                <Status
                  tone={
                    job.phase === "succeeded"
                      ? "live"
                      : job.phase === "failed"
                        ? "offline"
                        : "warning"
                  }
                >
                  {phaseLabel(job.phase)}
                </Status>
              </span>
              <span className="training-history__item-summary">
                {job.config.epochs} epochs · batch {job.config.micro_batch_size} ×{" "}
                {job.config.gradient_accumulation}
                {job.current_epoch > 0 ? ` · epoch ${job.current_epoch}/${job.config.epochs}` : ""}
              </span>
              <span className="training-history__item-id">
                Run {job.id.slice(0, 8)} · attempt {job.attempts}
              </span>
            </button>
          ))}
        </div>
        {nextCursor === null ? null : (
          <div className="training-history__more">
            <Button disabled={loadingMore} onClick={loadMore} variant="secondary">
              {loadingMore ? "Loading…" : "Load older runs"}
            </Button>
          </div>
        )}
      </div>
    </Panel>
  )
}
