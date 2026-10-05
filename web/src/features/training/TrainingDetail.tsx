import { useEffect, useRef, useState } from "react"
import type {
  ApiClient,
  TrainingJobResponse,
  TrainingLogEntry,
  TrainingMemoryRefusal,
  TrainingMetric,
} from "../../app/client"
import { HttpError, isAbortError } from "../../app/client"
import { parseTrainingMemoryRefusal } from "../../app/clientTrainingDomain"
import { Button, Dialog, Panel, Status } from "../../components"
import { MetricChart } from "./MetricChart"
import {
  canCancelTrainingJob,
  canResumeTrainingJob,
  classifyTrainingRefusal,
  type TrainingRefusalPresentation,
} from "./trainingModel"

const METRIC_LIMIT = 200
const LOG_LIMIT = 50
const ACTIVE_PHASES = new Set<TrainingJobResponse["phase"]>([
  "starting",
  "training",
  "evaluating",
  "publishing",
  "cancelling",
])
const METRIC_PAGE_WALK_LIMIT = 5
const METRIC_WINDOW_LIMIT = METRIC_LIMIT * METRIC_PAGE_WALK_LIMIT
const LOG_PAGE_WALK_LIMIT = 20

type TrainingDetailProps = Readonly<{
  client: ApiClient
  jobId: string
  onBack: () => void
  onCandidate: () => void
  onUnauthorized: () => void
  onChanged: () => void
}>

type LoadState =
  | Readonly<{ kind: "loading" }>
  | Readonly<{ kind: "ready"; job: TrainingJobResponse }>
  | Readonly<{ kind: "error"; message: string }>

type RefusalDialogState = Readonly<{
  refusal: TrainingMemoryRefusal
  presentation: TrainingRefusalPresentation
}> | null

type MetricWindow = Readonly<{
  items: readonly TrainingMetric[]
  nextCursor: string | null
}>

type LogWindow = Readonly<{
  items: readonly TrainingLogEntry[]
  nextCursor: string | null
  lastEntryCursor: string | null
}>

async function loadMetricWindow(
  client: ApiClient,
  jobId: string,
  cursor: string | null,
  signal: AbortSignal,
): Promise<MetricWindow> {
  let nextCursor = cursor
  let pageCount = 0
  let items: TrainingMetric[] = []
  while (pageCount < METRIC_PAGE_WALK_LIMIT) {
    const page = await client.getTrainingMetrics(jobId, METRIC_LIMIT, nextCursor, signal)
    items = [...items, ...page.items].slice(-METRIC_WINDOW_LIMIT)
    nextCursor = page.next_cursor
    pageCount += 1
    if (nextCursor === null || page.items.length === 0) break
  }
  return { items, nextCursor }
}

async function loadLogWindow(
  client: ApiClient,
  jobId: string,
  cursor: string | null,
  signal: AbortSignal,
): Promise<LogWindow> {
  let nextCursor = cursor
  let lastEntryCursor: string | null = null
  let pageCount = 0
  let items: TrainingLogEntry[] = []
  while (pageCount < LOG_PAGE_WALK_LIMIT) {
    const page = await client.getTrainingLogs(jobId, LOG_LIMIT, nextCursor, signal)
    items = [...items, ...page.items].slice(-LOG_LIMIT)
    const lastPageEntry = page.items.at(-1)
    if (lastPageEntry !== undefined) lastEntryCursor = lastPageEntry.cursor
    nextCursor = page.next_cursor
    pageCount += 1
    if (nextCursor === null || page.items.length === 0) break
  }
  return { items, nextCursor, lastEntryCursor }
}

function phaseLabel(phase: TrainingJobResponse["phase"]): string {
  switch (phase) {
    case "starting":
      return "Starting"
    case "training":
      return "Training"
    case "evaluating":
      return "Evaluating held-out data"
    case "publishing":
      return "Publishing candidate"
    case "succeeded":
      return "Succeeded"
    case "cancelling":
      return "Cancellation requested"
    case "cancelled":
      return "Cancelled"
    case "failed":
      return "Failed"
    case "interrupted":
      return "Interrupted · manual resume available"
  }
}

function formatBytes(bytes: number | null): string {
  return bytes === null ? "—" : `${(bytes / 1024 ** 3).toFixed(2)} GiB`
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function timestamp(value: string): string {
  return new Date(value).toLocaleString()
}

function elapsed(job: TrainingJobResponse): string {
  const start = Date.parse(job.created_at)
  const end = job.finished_at === null ? Date.now() : Date.parse(job.finished_at)
  const totalSeconds = Math.max(0, Math.floor((end - start) / 1_000))
  const hours = Math.floor(totalSeconds / 3_600)
  const minutes = Math.floor((totalSeconds % 3_600) / 60)
  const seconds = totalSeconds % 60
  return `${hours}h ${minutes}m ${seconds}s`
}

function requestErrorMessage(error: unknown): string {
  if (error instanceof HttpError && error.status === 401)
    return "Your session expired. Sign in again."
  return "The saved training run could not be loaded. Refresh the run details and try again."
}

function safeJobError(error: string | null): string | undefined {
  if (error === null) return undefined
  switch (error) {
    case "training_memory_refused":
    case "memory_profile_unsupported":
      return "The server refused the memory admission. Review the settings and request a new estimate."
    case "training_dataset_unavailable":
    case "training_dataset_changed":
      return "The registered CUHK-PEDES dataset is unavailable or changed. Refresh its validation status."
    case "training_gpu_unavailable":
    case "training_supervisor_unavailable":
      return "GPU or training supervisor status became unavailable. The run did not continue."
    case "training_oom":
    case "out_of_memory":
      return "GPU memory ran out during training. The last complete checkpoint remains available when one was saved."
    default:
      return "The training service recorded a failure. Review the sanitized log entries below for more detail."
  }
}

function memoryRefusal(error: unknown): TrainingMemoryRefusal | null {
  if (!(error instanceof HttpError)) return null
  const payload = error.payload
  const detail =
    typeof payload === "object" && payload !== null && !Array.isArray(payload)
      ? Reflect.get(payload, "detail")
      : undefined
  return parseTrainingMemoryRefusal(detail ?? payload)
}

function appendMetrics(
  current: readonly TrainingMetric[],
  incoming: readonly TrainingMetric[],
): readonly TrainingMetric[] {
  const byIdentity = new Map<string, TrainingMetric>()
  for (const metric of current) {
    byIdentity.set(`${metric.epoch}:${metric.step}:${metric.observed_at}`, metric)
  }
  for (const metric of incoming) {
    byIdentity.set(`${metric.epoch}:${metric.step}:${metric.observed_at}`, metric)
  }
  return [...byIdentity.values()].slice(-METRIC_WINDOW_LIMIT)
}

function appendLogs(
  current: readonly TrainingLogEntry[],
  incoming: readonly TrainingLogEntry[],
): readonly TrainingLogEntry[] {
  const seen = new Set(current.map((entry) => entry.cursor))
  return [...current, ...incoming.filter((entry) => !seen.has(entry.cursor))].slice(-LOG_LIMIT)
}

function trainingActionErrorMessage(error: unknown): string {
  if (error instanceof HttpError) {
    if (error.status === 401) return "Your session expired. Sign in again."
    if (error.code === "training_job_conflict") {
      return "Another training process still owns the GPU. The run status will refresh automatically."
    }
    if (error.code === "training_job_state_conflict") {
      return "The run phase changed before the action was accepted. Refresh the saved status."
    }
    if (error.code === "training_memory_refused") {
      return "The server refused resume because the current GPU memory admission changed. Review the training setup before retrying."
    }
  }
  return "The requested training action was not accepted. Refresh the run status and try again."
}

export function TrainingDetail({
  client,
  jobId,
  onBack,
  onCandidate,
  onChanged,
  onUnauthorized,
}: TrainingDetailProps) {
  const [loadState, setLoadState] = useState<LoadState>({ kind: "loading" })
  const [metrics, setMetrics] = useState<readonly TrainingMetric[]>([])
  const [logs, setLogs] = useState<readonly TrainingLogEntry[]>([])
  const [nextMetricCursor, setNextMetricCursor] = useState<string | null>(null)
  const [nextLogCursor, setNextLogCursor] = useState<string | null>(null)
  const [telemetryMessage, setTelemetryMessage] = useState<string | undefined>(undefined)
  const [actionError, setActionError] = useState<string | undefined>(undefined)
  const [refusalDialog, setRefusalDialog] = useState<RefusalDialogState>(null)
  const [actionBusy, setActionBusy] = useState(false)
  const [loadingMoreMetrics, setLoadingMoreMetrics] = useState(false)
  const [loadingMoreLogs, setLoadingMoreLogs] = useState(false)
  const loadController = useRef<AbortController | null>(null)
  const loadGeneration = useRef(0)
  const actionController = useRef<AbortController | null>(null)
  const loadLatest = useRef<() => void>(() => {})
  const actionInFlight = useRef(false)
  const resumeButtonRef = useRef<HTMLButtonElement | null>(null)
  const metricCursor = useRef<string | null>(null)
  const logContinuationCursor = useRef<string | null>(null)
  const lastLogCursor = useRef<string | null>(null)
  const metricPageController = useRef<AbortController | null>(null)
  const logPageController = useRef<AbortController | null>(null)
  const metricPageOwner = useRef(0)
  const logPageOwner = useRef(0)
  const metricPageBusy = useRef(false)
  const logPageBusy = useRef(false)

  useEffect(() => {
    let mounted = true
    let timer: number | null = null
    loadController.current?.abort()
    setLoadState({ kind: "loading" })
    setMetrics([])
    setLogs([])
    setNextMetricCursor(null)
    setNextLogCursor(null)
    setTelemetryMessage(undefined)
    metricCursor.current = null
    logContinuationCursor.current = null
    lastLogCursor.current = null

    async function load(): Promise<void> {
      if (!mounted) return
      const controller = new AbortController()
      loadController.current = controller
      const generation = loadGeneration.current + 1
      loadGeneration.current = generation
      const metricRequestCursor = metricCursor.current
      const logRequestCursor = logContinuationCursor.current ?? lastLogCursor.current
      const [jobResult, metricResult, logResult] = await Promise.allSettled([
        client.getTrainingJob(jobId, controller.signal),
        metricPageBusy.current
          ? Promise.resolve({ items: [], nextCursor: metricCursor.current })
          : loadMetricWindow(client, jobId, metricRequestCursor, controller.signal),
        logPageBusy.current
          ? Promise.resolve({
              items: [],
              nextCursor: logContinuationCursor.current,
              lastEntryCursor: null,
            })
          : loadLogWindow(client, jobId, logRequestCursor, controller.signal),
      ])
      if (!mounted || generation !== loadGeneration.current || controller.signal.aborted) return
      if (jobResult.status === "rejected") {
        if (isAbortError(jobResult.reason)) return
        if (jobResult.reason instanceof HttpError && jobResult.reason.status === 401)
          onUnauthorized()
        setLoadState({ kind: "error", message: requestErrorMessage(jobResult.reason) })
      } else {
        setLoadState({ kind: "ready", job: jobResult.value })
        if (metricResult.status === "fulfilled") {
          metricCursor.current = metricResult.value.nextCursor
          setNextMetricCursor(metricResult.value.nextCursor)
          if (metricRequestCursor === null) {
            setMetrics(metricResult.value.items)
          } else if (metricResult.value.items.length > 0) {
            setMetrics((current) => appendMetrics(current, metricResult.value.items))
          }
        }
        if (logResult.status === "fulfilled") {
          logContinuationCursor.current = logResult.value.nextCursor
          if (logResult.value.lastEntryCursor !== null) {
            lastLogCursor.current = logResult.value.lastEntryCursor
          }
          setNextLogCursor(logResult.value.nextCursor)
          if (logRequestCursor === null) {
            setLogs(logResult.value.items)
          } else if (logResult.value.items.length > 0) {
            setLogs((current) => appendLogs(current, logResult.value.items))
          }
        }
        if (metricResult.status === "rejected" || logResult.status === "rejected") {
          const telemetryError =
            metricResult.status === "rejected"
              ? metricResult.reason
              : logResult.status === "rejected"
                ? logResult.reason
                : undefined
          if (telemetryError instanceof HttpError && telemetryError.status === 401) onUnauthorized()
          setTelemetryMessage("Some bounded metric or log history is temporarily unavailable.")
        } else {
          setTelemetryMessage(undefined)
        }
        if (ACTIVE_PHASES.has(jobResult.value.phase)) {
          timer = window.setTimeout(() => void load(), 2_000)
        }
      }
      if (loadController.current === controller) loadController.current = null
    }

    loadLatest.current = () => void load()
    void load()
    return () => {
      mounted = false
      loadGeneration.current += 1
      if (timer !== null) window.clearTimeout(timer)
      loadController.current?.abort()
      loadController.current = null
      actionController.current?.abort()
      actionController.current = null
      metricPageOwner.current += 1
      metricPageController.current?.abort()
      metricPageController.current = null
      logPageOwner.current += 1
      logPageController.current?.abort()
      logPageController.current = null
    }
  }, [client, jobId, onUnauthorized])

  async function act(action: "cancel" | "resume"): Promise<void> {
    if (actionBusy || actionInFlight.current || loadState.kind !== "ready") return
    const permitted =
      action === "cancel"
        ? canCancelTrainingJob(loadState.job.phase)
        : canResumeTrainingJob(loadState.job.phase)
    if (!permitted) return
    actionInFlight.current = true
    setActionBusy(true)
    setActionError(undefined)
    actionController.current?.abort()
    const controller = new AbortController()
    actionController.current = controller
    try {
      const requestId = crypto.randomUUID()
      const updated =
        action === "cancel"
          ? await client.cancelTrainingJob(jobId, requestId, controller.signal)
          : await client.resumeTrainingJob(jobId, requestId, controller.signal)
      if (controller.signal.aborted) return
      setLoadState({ kind: "ready", job: updated })
      setRefusalDialog(null)
      onChanged()
      loadLatest.current()
    } catch (error) {
      if (isAbortError(error)) return
      if (error instanceof HttpError && error.status === 401) onUnauthorized()
      const refusal = action === "resume" ? memoryRefusal(error) : null
      if (refusal === null) {
        setActionError(trainingActionErrorMessage(error))
      } else {
        setActionError(undefined)
        setRefusalDialog({ refusal, presentation: classifyTrainingRefusal(refusal) })
      }
    } finally {
      if (actionController.current === controller) actionController.current = null
      actionInFlight.current = false
      setActionBusy(false)
    }
  }

  async function loadMoreMetricHistory(): Promise<void> {
    const cursor = metricCursor.current
    if (cursor === null || metricPageBusy.current) return
    metricPageBusy.current = true
    setLoadingMoreMetrics(true)
    metricPageOwner.current += 1
    const owner = metricPageOwner.current
    metricPageController.current?.abort()
    const controller = new AbortController()
    metricPageController.current = controller
    try {
      const window = await loadMetricWindow(client, jobId, cursor, controller.signal)
      if (owner !== metricPageOwner.current || controller.signal.aborted) return
      setMetrics((current) => appendMetrics(current, window.items))
      metricCursor.current = window.nextCursor
      setNextMetricCursor(window.nextCursor)
    } catch (error) {
      if (isAbortError(error)) return
      if (error instanceof HttpError && error.status === 401) onUnauthorized()
      setTelemetryMessage(
        "Newer metric history could not be loaded. The current bounded page remains visible.",
      )
    } finally {
      if (owner === metricPageOwner.current) {
        metricPageBusy.current = false
        setLoadingMoreMetrics(false)
        if (metricPageController.current === controller) metricPageController.current = null
      }
    }
  }

  async function loadMoreLogHistory(): Promise<void> {
    const cursor = logContinuationCursor.current
    if (cursor === null || logPageBusy.current) return
    logPageBusy.current = true
    setLoadingMoreLogs(true)
    logPageOwner.current += 1
    const owner = logPageOwner.current
    logPageController.current?.abort()
    const controller = new AbortController()
    logPageController.current = controller
    try {
      const window = await loadLogWindow(client, jobId, cursor, controller.signal)
      if (owner !== logPageOwner.current || controller.signal.aborted) return
      setLogs((current) => appendLogs(current, window.items))
      logContinuationCursor.current = window.nextCursor
      setNextLogCursor(window.nextCursor)
      if (window.lastEntryCursor !== null) lastLogCursor.current = window.lastEntryCursor
    } catch (error) {
      if (isAbortError(error)) return
      if (error instanceof HttpError && error.status === 401) onUnauthorized()
      setTelemetryMessage(
        "Newer logs could not be loaded. The current bounded page remains visible.",
      )
    } finally {
      if (owner === logPageOwner.current) {
        logPageBusy.current = false
        setLoadingMoreLogs(false)
        if (logPageController.current === controller) logPageController.current = null
      }
    }
  }

  if (loadState.kind === "loading") {
    return (
      <main className="training-main" aria-busy="true">
        <p role="status">Loading saved training run…</p>
      </main>
    )
  }
  if (loadState.kind === "error") {
    return (
      <main className="training-main">
        <div className="training-detail__toolbar">
          <Button onClick={onBack} variant="secondary">
            Back to setup
          </Button>
        </div>
        <p className="training-form__error" role="alert">
          {loadState.message}
        </p>
        <Button onClick={() => loadLatest.current()} variant="secondary">
          Refresh run
        </Button>
      </main>
    )
  }

  const { job } = loadState
  const lossPoints = metrics.flatMap((metric) =>
    metric.training_loss === null
      ? []
      : [{ label: `E${metric.epoch}`, value: metric.training_loss }],
  )
  const recallPoints = metrics.flatMap((metric) =>
    metric.validation_recall_at_1 === null
      ? []
      : [{ label: `E${metric.epoch}`, value: metric.validation_recall_at_1 }],
  )
  const latestMetric = metrics.at(-1)
  const evaluation = job.evaluation

  return (
    <main className="training-main" aria-labelledby="training-detail-heading">
      <div className="training-detail__toolbar">
        <Button onClick={onBack} variant="secondary">
          Back to setup
        </Button>
        <span className="training-detail__updated">
          Updated <time dateTime={job.updated_at}>{timestamp(job.updated_at)}</time>
        </span>
      </div>
      <div className="training-detail__heading">
        <div>
          <p className="app-kicker">Saved training job</p>
          <h1 id="training-detail-heading">Run {job.id.slice(0, 8)}</h1>
        </div>
        <Status
          tone={job.phase === "succeeded" ? "live" : job.phase === "failed" ? "offline" : "warning"}
        >
          {phaseLabel(job.phase)}
        </Status>
      </div>
      {actionError === undefined ? null : (
        <p className="training-form__error" role="alert">
          {actionError}
        </p>
      )}
      {telemetryMessage === undefined ? null : (
        <p className="training-notice" role="status">
          {telemetryMessage}
        </p>
      )}
      <div className="training-detail__grid">
        <Panel eyebrow="Progress" title="Current run">
          <dl className="training-progress-facts">
            <div>
              <dt>Epoch</dt>
              <dd>
                {job.current_epoch} / {job.config.epochs}
              </dd>
            </div>
            <div>
              <dt>Step</dt>
              <dd>{job.current_step.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Attempt</dt>
              <dd>{job.attempts}</dd>
            </div>
            <div>
              <dt>Elapsed</dt>
              <dd>{elapsed(job)}</dd>
            </div>
            <div>
              <dt>Best validation Recall@1</dt>
              <dd>{job.best_metric === null ? "—" : formatPercent(job.best_metric)}</dd>
            </div>
            <div>
              <dt>Last allocated memory</dt>
              <dd>{formatBytes(latestMetric?.allocated_bytes ?? null)}</dd>
            </div>
            <div>
              <dt>Last reserved memory</dt>
              <dd>{formatBytes(latestMetric?.reserved_bytes ?? null)}</dd>
            </div>
          </dl>
          <div className="training-detail__actions">
            {canCancelTrainingJob(job.phase) ? (
              <Button disabled={actionBusy} onClick={() => void act("cancel")} variant="danger">
                {actionBusy ? "Sending request…" : "Request cooperative cancel"}
              </Button>
            ) : null}
            {canResumeTrainingJob(job.phase) ? (
              <Button
                disabled={actionBusy}
                onClick={() => void act("resume")}
                ref={resumeButtonRef}
              >
                {actionBusy ? "Checking admission…" : "Resume from complete checkpoint"}
              </Button>
            ) : null}
          </div>
          {safeJobError(job.error) === undefined ? null : (
            <p className="training-form__error" role="status">
              {safeJobError(job.error)}
            </p>
          )}
        </Panel>

        <Panel eyebrow="Submitted snapshot" title="Immutable configuration">
          <dl className="training-progress-facts">
            <div>
              <dt>Epochs</dt>
              <dd>{job.config.epochs}</dd>
            </div>
            <div>
              <dt>Learning rate</dt>
              <dd>{job.config.learning_rate.toExponential(2)}</dd>
            </div>
            <div>
              <dt>Micro batch</dt>
              <dd>{job.config.micro_batch_size}</dd>
            </div>
            <div>
              <dt>Weight decay</dt>
              <dd>{job.config.weight_decay}</dd>
            </div>
            <div>
              <dt>Accumulation</dt>
              <dd>{job.config.gradient_accumulation}</dd>
            </div>
            <div>
              <dt>Warmup</dt>
              <dd>{formatPercent(job.config.warmup_ratio)}</dd>
            </div>
            <div>
              <dt>Seed</dt>
              <dd>{job.config.seed}</dd>
            </div>
            <div>
              <dt>Early stopping</dt>
              <dd>{job.config.early_stopping_patience ?? "Off"}</dd>
            </div>
            <div>
              <dt>Clip norm</dt>
              <dd>{job.config.gradient_clipping_norm}</dd>
            </div>
            <div>
              <dt>Precision</dt>
              <dd>{job.config.mixed_precision.toUpperCase()}</dd>
            </div>
            <div>
              <dt>Gradient checkpointing</dt>
              <dd>{job.config.gradient_checkpointing ? "On" : "Off"}</dd>
            </div>
          </dl>
          <p className="training-muted">
            Dataset fingerprint <code>{job.dataset.fingerprint.slice(0, 16)}…</code>
          </p>
        </Panel>

        <Panel eyebrow="Stored telemetry" title="Training metrics">
          <div className="training-chart-grid">
            <MetricChart title="Training loss" points={lossPoints} />
            <MetricChart
              title="Validation Recall@1"
              points={recallPoints}
              formatValue={formatPercent}
            />
          </div>
          <p className="training-muted">
            Metrics load in forward pages of {METRIC_LIMIT}; the chart retains at most{" "}
            {METRIC_WINDOW_LIMIT} points.
          </p>
          {nextMetricCursor === null ? null : (
            <div className="training-detail__actions">
              <Button
                disabled={loadingMoreMetrics}
                onClick={() => void loadMoreMetricHistory()}
                variant="secondary"
              >
                {loadingMoreMetrics ? "Loading…" : "Load newer metrics"}
              </Button>
            </div>
          )}
        </Panel>

        <Panel eyebrow="Evaluation" title="Final held-out comparison">
          {evaluation === null ? (
            <p className="training-muted">
              Final CUHK-PEDES evaluation is not available for this run yet.
            </p>
          ) : (
            <>
              <p className="training-muted">
                Test split · {evaluation.protocol} · best validation epoch{" "}
                {evaluation.best_validation_epoch}
              </p>
              <table className="training-evaluation-table">
                <caption className="sr-only">Final retrieval evaluation</caption>
                <thead>
                  <tr className="training-evaluation-table__row--header">
                    <th scope="col">Model</th>
                    <th scope="col">Recall@1</th>
                    <th scope="col">Recall@5</th>
                    <th scope="col">Recall@10</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th scope="row">Baseline · {evaluation.baseline_model_id}</th>
                    <td>{formatPercent(evaluation.baseline.recall_at_1)}</td>
                    <td>{formatPercent(evaluation.baseline.recall_at_5)}</td>
                    <td>{formatPercent(evaluation.baseline.recall_at_10)}</td>
                  </tr>
                  <tr>
                    <th scope="row">Candidate · {job.candidate_model_id ?? "pending"}</th>
                    <td>{formatPercent(evaluation.candidate.recall_at_1)}</td>
                    <td>{formatPercent(evaluation.candidate.recall_at_5)}</td>
                    <td>{formatPercent(evaluation.candidate.recall_at_10)}</td>
                  </tr>
                </tbody>
              </table>
              <dl className="training-provenance">
                <div>
                  <dt>Dataset SHA-256</dt>
                  <dd>
                    <code>{evaluation.dataset_sha256}</code>
                  </dd>
                </div>
                <div>
                  <dt>Candidate weights SHA-256</dt>
                  <dd>
                    <code>{evaluation.candidate_weights_sha256}</code>
                  </dd>
                </div>
                <div>
                  <dt>Package SHA-256</dt>
                  <dd>
                    <code>{evaluation.package_sha256}</code>
                  </dd>
                </div>
                <div>
                  <dt>Evaluation code</dt>
                  <dd>
                    <code>{evaluation.evaluation_code_revision.slice(0, 16)}…</code>
                  </dd>
                </div>
              </dl>
            </>
          )}
          <div className="training-apply-gate" role="status">
            <strong>Manual model application remains blocked</strong>
            <p>
              CUHK-PEDES scores do not establish product-crop quality. Product retrieval cases, GPU
              preparation, and the existing full model-transition preflight must pass first.
            </p>
          </div>
          {job.candidate_model_id === null ? null : (
            <Button onClick={onCandidate} variant="secondary">
              Review candidate in the existing model selector
            </Button>
          )}
        </Panel>

        <Panel eyebrow="Sanitized output" title="Training logs">
          <div className="training-log-list" aria-live="polite">
            {logs.length === 0 ? (
              <p className="training-muted">No log entries are available.</p>
            ) : null}
            {logs.map((entry) => (
              <div className="training-log" key={entry.cursor}>
                <time dateTime={entry.observed_at}>{timestamp(entry.observed_at)}</time>
                <Status
                  tone={
                    entry.level === "error"
                      ? "offline"
                      : entry.level === "warning"
                        ? "warning"
                        : "neutral"
                  }
                >
                  {entry.level}
                </Status>
                <p>{entry.message}</p>
              </div>
            ))}
          </div>
          {nextLogCursor === null ? null : (
            <div className="training-detail__actions">
              <Button
                disabled={loadingMoreLogs}
                onClick={() => void loadMoreLogHistory()}
                variant="secondary"
              >
                {loadingMoreLogs ? "Loading…" : "Load newer logs"}
              </Button>
            </div>
          )}
          <p className="training-muted">Showing at most {LOG_LIMIT} log lines at a time.</p>
        </Panel>
      </div>
      <Dialog
        confirmLabel="Close details"
        description={refusalDialog?.presentation.message ?? "Training resume admission details."}
        onClose={() => setRefusalDialog(null)}
        onConfirm={() => setRefusalDialog(null)}
        open={refusalDialog !== null}
        returnFocusRef={resumeButtonRef}
        title={refusalDialog?.presentation.title ?? "Training resume admission"}
      >
        {refusalDialog === null ? null : (
          <div className="training-refusal-details">
            {refusalDialog.presentation.kind === "memory_shortage" ||
            refusalDialog.presentation.kind === "unsupported_profile" ? (
              <dl className="training-memory-facts">
                <div>
                  <dt>Training peak</dt>
                  <dd>{formatBytes(refusalDialog.refusal.training_peak_bytes)}</dd>
                </div>
                <div>
                  <dt>Inference reserve</dt>
                  <dd>{formatBytes(refusalDialog.refusal.reserve_bytes)}</dd>
                </div>
                <div>
                  <dt>Required</dt>
                  <dd>{formatBytes(refusalDialog.refusal.required_bytes)}</dd>
                </div>
                <div>
                  <dt>Available</dt>
                  <dd>{formatBytes(refusalDialog.refusal.free_bytes)}</dd>
                </div>
              </dl>
            ) : null}
            {refusalDialog.refusal.observed_at === undefined ? (
              <p className="training-muted">
                The server refusal did not include its observation time.
              </p>
            ) : (
              <p className="training-muted">
                Server observed{" "}
                <time dateTime={refusalDialog.refusal.observed_at}>
                  {timestamp(refusalDialog.refusal.observed_at)}
                </time>
              </p>
            )}
            {refusalDialog.refusal.profile_identity === undefined ? null : (
              <p className="training-muted">
                Memory profile <code>{refusalDialog.refusal.profile_identity.slice(0, 16)}…</code>
              </p>
            )}
          </div>
        )}
      </Dialog>
    </main>
  )
}
