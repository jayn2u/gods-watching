import { useCallback, useEffect, useRef, useState } from "react"
import type {
  ApiClient,
  TrainingConfig,
  TrainingDatasetStatus,
  TrainingMemoryRefusal,
  TrainingPreflightResponse,
} from "../../app/client"
import { HttpError, isAbortError, NetworkError } from "../../app/client"
import { parseTrainingMemoryRefusal } from "../../app/clientTrainingDomain"
import { Button, Dialog } from "../../components"
import { TrainingDetail } from "./TrainingDetail"
import { TrainingForm } from "./TrainingForm"
import { TrainingHistory } from "./TrainingHistory"
import {
  classifyTrainingRefusal,
  DEFAULT_TRAINING_CONFIG,
  freezeTrainingConfig,
  TrainingGeneration,
  type TrainingRefusalPresentation,
} from "./trainingModel"
import "./training.css"

type DatasetState =
  | Readonly<{ kind: "loading" }>
  | Readonly<{ kind: "ready"; status: TrainingDatasetStatus }>
  | Readonly<{ kind: "error"; message: string }>

type AdmissionState =
  | Readonly<{
      kind: "idle"
      configurationKey: string
      configurationGeneration: number
    }>
  | Readonly<{
      kind: "checking"
      configurationKey: string
      configurationGeneration: number
    }>
  | Readonly<{
      kind: "ready"
      configurationKey: string
      configurationGeneration: number
      response: TrainingPreflightResponse
    }>
  | Readonly<{
      kind: "refused"
      configurationKey: string
      configurationGeneration: number
      refusal: TrainingMemoryRefusal
      presentation: TrainingRefusalPresentation
    }>
  | Readonly<{
      kind: "unavailable"
      configurationKey: string
      configurationGeneration: number
      message: string
    }>

type RefusalDialogState = Readonly<{
  refusal: TrainingMemoryRefusal
  presentation: TrainingRefusalPresentation
  configSnapshot: TrainingConfig
  source: "preflight" | "submission"
}> | null

type TrainingScreenProps = Readonly<{
  client: ApiClient
  onUnauthorized: () => void
  onOpenModelSelector: () => void
}>

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function recordValue(value: unknown, key: string): unknown {
  return isRecord(value) ? value[key] : undefined
}

function memoryRefusal(error: unknown): TrainingMemoryRefusal | null {
  if (!(error instanceof HttpError)) return null
  const detail = recordValue(error.payload, "detail")
  return parseTrainingMemoryRefusal(detail ?? error.payload)
}

function safeTrainingError(error: unknown): string {
  if (error instanceof NetworkError) {
    return "The training service could not be reached. The server will make the final GPU admission decision."
  }
  if (error instanceof HttpError) {
    if (error.status === 401) return "Your session expired. Sign in again."
    switch (error.code) {
      case "training_dataset_unavailable":
        return "The operator-configured CUHK-PEDES dataset is unavailable. Refresh dataset status after it has been checked."
      case "training_supervisor_unavailable":
        return "The training supervisor or GPU status is unavailable. Refresh the estimate before trying again."
      case "training_request_conflict":
        return "The request identity was already used with different settings. Refresh the screen before submitting again."
      case "training_job_conflict":
        return "Another training run currently owns the GPU. Refresh the saved run status."
      default:
        return error.status >= 500
          ? "The training service is temporarily unavailable. Try again after its status recovers."
          : "The training request was not accepted. Review the configuration and refresh the estimate."
    }
  }
  return "The training request could not be completed. Refresh the estimate and try again."
}

function datasetMessage(state: DatasetState): string {
  if (state.kind === "loading") return "Checking the operator-configured CUHK-PEDES dataset…"
  if (state.kind === "error") return state.message
  const { status } = state
  if (status.valid && status.registered && status.snapshot !== null) {
    return `${status.snapshot.image_count.toLocaleString()} images, ${status.snapshot.caption_count.toLocaleString()} captions, and ${status.snapshot.identity_count.toLocaleString()} identities passed validation.`
  }
  switch (status.reason) {
    case "dataset_validating":
      return "The registered dataset is still being checked. Refresh status in a moment; training stays disabled until validation finishes."
    case "dataset_invalid_or_unavailable":
    case "dataset_unavailable":
      return "The registered dataset could not be validated. Check the operator-configured CUHK-PEDES source, then refresh status."
    case "registered_dataset_fingerprint_mismatch":
      return "The registered dataset changed since this run was recorded. Refresh status before starting a new run."
    default:
      return "The registered CUHK-PEDES dataset is not ready. Refresh status after the server finishes validation."
  }
}

function formatBytes(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(2)} GiB`
}

function refusalFromEstimate(response: TrainingPreflightResponse): TrainingMemoryRefusal {
  return {
    code: "training_memory_refused",
    message: "The server declined this memory admission.",
    reason: response.reason,
    training_peak_bytes: response.training_peak_bytes,
    reserve_bytes: response.reserve_bytes,
    required_bytes: response.required_bytes,
    free_bytes: response.free_bytes,
    observed_at: response.observed_at,
    profile_identity: response.profile_identity,
  }
}

function safeObservedTime(value: string): string {
  return new Date(value).toLocaleString()
}

export function TrainingScreen({
  client,
  onOpenModelSelector,
  onUnauthorized,
}: TrainingScreenProps) {
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_TRAINING_CONFIG)
  const [configLoading, setConfigLoading] = useState(true)
  const [configReady, setConfigReady] = useState(false)
  const [dataset, setDataset] = useState<DatasetState>({ kind: "loading" })
  const [admission, setAdmission] = useState<AdmissionState>({
    kind: "idle",
    configurationKey: "",
    configurationGeneration: 0,
  })
  const [draftErrors, setDraftErrors] = useState<ReadonlySet<string>>(() => new Set())
  const [dialog, setDialog] = useState<RefusalDialogState>(null)
  const [submissionMessage, setSubmissionMessage] = useState<string | undefined>(undefined)
  const [starting, setStarting] = useState(false)
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)
  const [historyRefreshToken, setHistoryRefreshToken] = useState(0)
  const [estimateRefreshToken, setEstimateRefreshToken] = useState(0)
  const [configurationRequestGeneration, setConfigurationRequestGeneration] = useState(0)
  const estimateRefreshButtonRef = useRef<HTMLButtonElement | null>(null)
  const startInFlight = useRef(false)
  const configController = useRef<AbortController | null>(null)
  const datasetController = useRef<AbortController | null>(null)
  const datasetGeneration = useRef(0)
  const preflightController = useRef<AbortController | null>(null)
  const preflightGeneration = useRef(new TrainingGeneration())
  const submitController = useRef<AbortController | null>(null)
  const configurationRequestGenerationRef = useRef(0)

  const advanceConfigurationRequestGeneration = useCallback((): number => {
    const generation = configurationRequestGenerationRef.current + 1
    configurationRequestGenerationRef.current = generation
    setConfigurationRequestGeneration(generation)
    return generation
  }, [])

  const loadConfig = useCallback((): void => {
    advanceConfigurationRequestGeneration()
    configController.current?.abort()
    const controller = new AbortController()
    configController.current = controller
    setConfigLoading(true)
    void client
      .getTrainingConfig(controller.signal)
      .then((response) => {
        if (controller.signal.aborted) return
        setConfig(freezeTrainingConfig(response))
        setDraftErrors(new Set())
        setConfigReady(true)
      })
      .catch((error: unknown) => {
        if (isAbortError(error)) return
        if (error instanceof HttpError && error.status === 401) onUnauthorized()
        setConfigReady(false)
      })
      .finally(() => {
        if (configController.current === controller) {
          configController.current = null
          setConfigLoading(false)
        }
      })
  }, [advanceConfigurationRequestGeneration, client, onUnauthorized])

  const refreshDataset = useCallback((): void => {
    advanceConfigurationRequestGeneration()
    datasetController.current?.abort()
    const controller = new AbortController()
    datasetController.current = controller
    const generation = datasetGeneration.current + 1
    datasetGeneration.current = generation
    setDataset({ kind: "loading" })
    void client
      .getTrainingDatasetStatus(controller.signal)
      .then((status) => {
        if (generation !== datasetGeneration.current || controller.signal.aborted) return
        setDataset({ kind: "ready", status })
      })
      .catch((error: unknown) => {
        if (generation !== datasetGeneration.current || isAbortError(error)) return
        if (error instanceof HttpError && error.status === 401) onUnauthorized()
        setDataset({
          kind: "error",
          message: safeTrainingError(error),
        })
      })
      .finally(() => {
        if (datasetController.current === controller) datasetController.current = null
      })
  }, [advanceConfigurationRequestGeneration, client, onUnauthorized])

  useEffect(() => {
    loadConfig()
    refreshDataset()
    return () => {
      configController.current?.abort()
      configController.current = null
      datasetGeneration.current += 1
      datasetController.current?.abort()
      datasetController.current = null
      preflightGeneration.current.invalidate()
      preflightController.current?.abort()
      preflightController.current = null
      submitController.current?.abort()
      submitController.current = null
    }
  }, [loadConfig, refreshDataset])

  const datasetStatus = dataset.kind === "ready" ? dataset.status : null
  const datasetReady =
    datasetStatus?.registered === true && datasetStatus.valid && datasetStatus.snapshot !== null
  const configurationKey = `${JSON.stringify(config)}:${datasetStatus?.snapshot?.fingerprint ?? ""}:${estimateRefreshToken}`
  const currentConfigurationKey = useRef(configurationKey)
  currentConfigurationKey.current = configurationKey
  const draftsValid = draftErrors.size === 0

  useEffect(() => {
    preflightController.current?.abort()
    preflightController.current = null
    setDialog(null)
    if (!configReady || !datasetReady || !draftsValid) {
      preflightGeneration.current.invalidate()
      setAdmission({
        kind: "idle",
        configurationKey,
        configurationGeneration: configurationRequestGeneration,
      })
      return
    }
    const controller = new AbortController()
    preflightController.current = controller
    const generation = preflightGeneration.current.next()
    setAdmission({
      kind: "checking",
      configurationKey,
      configurationGeneration: configurationRequestGeneration,
    })
    const timeout = window.setTimeout(() => {
      void client
        .preflightTraining(freezeTrainingConfig(config), controller.signal)
        .then((response) => {
          if (!preflightGeneration.current.isCurrent(generation) || controller.signal.aborted)
            return
          if (response.admitted) {
            setAdmission({
              kind: "ready",
              configurationKey,
              configurationGeneration: configurationRequestGeneration,
              response,
            })
            return
          }
          const refusal = refusalFromEstimate(response)
          const presentation = classifyTrainingRefusal(refusal)
          setAdmission({
            kind: "refused",
            configurationKey,
            configurationGeneration: configurationRequestGeneration,
            refusal,
            presentation,
          })
          setDialog({
            refusal,
            presentation,
            configSnapshot: freezeTrainingConfig(config),
            source: "preflight",
          })
        })
        .catch((error: unknown) => {
          if (!preflightGeneration.current.isCurrent(generation) || isAbortError(error)) return
          if (error instanceof HttpError && error.status === 401) onUnauthorized()
          const refusal = memoryRefusal(error)
          if (refusal !== null) {
            const presentation = classifyTrainingRefusal(refusal)
            setAdmission({
              kind: "refused",
              configurationKey,
              configurationGeneration: configurationRequestGeneration,
              refusal,
              presentation,
            })
            setDialog({
              refusal,
              presentation,
              configSnapshot: freezeTrainingConfig(config),
              source: "preflight",
            })
            return
          }
          setAdmission({
            kind: "unavailable",
            configurationKey,
            configurationGeneration: configurationRequestGeneration,
            message: safeTrainingError(error),
          })
        })
        .finally(() => {
          if (preflightController.current === controller) preflightController.current = null
        })
    }, 180)
    return () => {
      window.clearTimeout(timeout)
      controller.abort()
      if (preflightController.current === controller) preflightController.current = null
      preflightGeneration.current.invalidate()
    }
  }, [
    client,
    config,
    configReady,
    configurationRequestGeneration,
    configurationKey,
    datasetReady,
    draftsValid,
    onUnauthorized,
  ])

  function onConfigChange(nextConfig: TrainingConfig): void {
    advanceConfigurationRequestGeneration()
    setConfig(nextConfig)
    setDialog(null)
    setSubmissionMessage(undefined)
  }

  function onDraftValidityChange(fieldKey: string, valid: boolean): void {
    advanceConfigurationRequestGeneration()
    setDraftErrors((current) => {
      if (current.has(fieldKey) === !valid) return current
      const next = new Set(current)
      if (valid) next.delete(fieldKey)
      else next.add(fieldKey)
      return next
    })
    if (!valid) setDialog(null)
  }

  const estimateIsCurrent =
    admission.configurationKey === configurationKey &&
    admission.configurationGeneration === configurationRequestGeneration
  const estimate = admission.kind === "ready" && estimateIsCurrent ? admission.response : null
  const estimateLoading = admission.kind === "checking" && estimateIsCurrent
  let estimateMessage: string | undefined
  if (!configReady || dataset.kind === "loading") {
    estimateMessage = "Waiting for server configuration and dataset validation."
  } else if (!datasetReady) {
    estimateMessage = datasetMessage(dataset)
  } else if (!draftsValid) {
    estimateMessage = "Complete numeric entries to refresh the memory estimate."
  } else if (admission.kind === "unavailable" && estimateIsCurrent) {
    estimateMessage = admission.message
  } else if (admission.kind === "refused" && estimateIsCurrent) {
    estimateMessage = admission.presentation.message
  } else if (
    estimateLoading ||
    admission.configurationKey !== configurationKey ||
    admission.configurationGeneration !== configurationRequestGeneration
  ) {
    estimateMessage = "Checking the current settings against available GPU memory…"
  }
  const canStart =
    configReady &&
    datasetReady &&
    draftsValid &&
    admission.kind === "ready" &&
    estimateIsCurrent &&
    admission.response.admitted &&
    !starting

  async function startTraining(): Promise<void> {
    if (startInFlight.current || !canStart) return
    startInFlight.current = true
    setStarting(true)
    setDialog(null)
    const submittedRequestGeneration = configurationRequestGenerationRef.current
    const submittedConfig = freezeTrainingConfig(config)
    const submittedConfigurationKey = configurationKey
    const requestId = crypto.randomUUID()
    const controller = new AbortController()
    submitController.current?.abort()
    submitController.current = controller
    try {
      const job = await client.submitTraining(submittedConfig, requestId, controller.signal)
      if (controller.signal.aborted) return
      setHistoryRefreshToken((current) => current + 1)
      setSelectedJobId(job.id)
    } catch (error) {
      if (isAbortError(error)) return
      if (error instanceof HttpError && error.status === 401) onUnauthorized()
      const refusal = memoryRefusal(error)
      if (refusal !== null) {
        const presentation = classifyTrainingRefusal(refusal)
        if (
          currentConfigurationKey.current === submittedConfigurationKey &&
          configurationRequestGenerationRef.current === submittedRequestGeneration
        ) {
          setAdmission({
            kind: "refused",
            configurationKey: submittedConfigurationKey,
            configurationGeneration: submittedRequestGeneration,
            refusal,
            presentation,
          })
        }
        setSubmissionMessage(undefined)
        setDialog({
          refusal,
          presentation,
          configSnapshot: submittedConfig,
          source: "submission",
        })
      } else {
        const message = safeTrainingError(error)
        if (
          currentConfigurationKey.current === submittedConfigurationKey &&
          configurationRequestGenerationRef.current === submittedRequestGeneration
        ) {
          setAdmission({
            kind: "unavailable",
            configurationKey: submittedConfigurationKey,
            configurationGeneration: submittedRequestGeneration,
            message,
          })
        } else {
          setSubmissionMessage(message)
        }
      }
    } finally {
      if (submitController.current === controller) submitController.current = null
      startInFlight.current = false
      setStarting(false)
    }
  }

  if (selectedJobId !== null) {
    return (
      <TrainingDetail
        client={client}
        jobId={selectedJobId}
        onBack={() => setSelectedJobId(null)}
        onCandidate={onOpenModelSelector}
        onChanged={() => setHistoryRefreshToken((current) => current + 1)}
        onUnauthorized={onUnauthorized}
      />
    )
  }

  return (
    <main className="training-main" aria-labelledby="training-heading">
      <div className="training-page-heading">
        <div>
          <p className="app-kicker">Operator workspace</p>
          <h1 id="training-heading">CLIP training</h1>
        </div>
        <p>
          Train a CUHK-PEDES candidate while the current search and live analysis model remains
          active.
        </p>
      </div>
      {configLoading ? (
        <p className="training-muted" role="status">
          Loading server training defaults…
        </p>
      ) : null}
      {!configLoading && !configReady ? (
        <div className="training-notice" role="alert">
          <p>
            Server training configuration could not be loaded. Retry to use the server-owned
            defaults.
          </p>
          <Button onClick={loadConfig} variant="secondary">
            Retry configuration
          </Button>
        </div>
      ) : null}
      {!configReady ? null : (
        <TrainingForm
          canStart={canStart}
          config={config}
          dataset={datasetStatus}
          datasetLoading={dataset.kind === "loading"}
          datasetMessage={datasetMessage(dataset)}
          draftsValid={draftsValid}
          estimate={estimate}
          estimateLoading={estimateLoading}
          estimateMessage={estimateMessage}
          onConfigChange={onConfigChange}
          onDraftValidityChange={onDraftValidityChange}
          onRefreshDataset={refreshDataset}
          onRefreshEstimate={() => {
            advanceConfigurationRequestGeneration()
            setEstimateRefreshToken((current) => current + 1)
          }}
          onStart={() => void startTraining()}
          estimateRefreshButtonRef={estimateRefreshButtonRef}
          starting={starting}
        />
      )}
      {submissionMessage === undefined ? null : (
        <p className="training-form__error" role="alert">
          {submissionMessage}
        </p>
      )}
      {!configReady ? null : (
        <TrainingHistory
          client={client}
          onSelect={setSelectedJobId}
          onUnauthorized={onUnauthorized}
          refreshToken={historyRefreshToken}
        />
      )}
      <p className="training-footer-note">
        Submitted settings are saved with each run. A completed candidate is never applied
        automatically; product-crop evidence and the existing model transition checks remain
        required.
      </p>
      <Dialog
        confirmLabel="Close details"
        description={dialog?.presentation.message ?? "Training admission details."}
        onClose={() => setDialog(null)}
        onConfirm={() => setDialog(null)}
        open={dialog !== null}
        returnFocusRef={estimateRefreshButtonRef}
        title={dialog?.presentation.title ?? "Training admission"}
      >
        {dialog === null ? null : (
          <div className="training-refusal-details">
            <p className="training-muted">
              {dialog.source === "submission"
                ? "The refusal applies to the immutable settings sent with this submit request."
                : "Settings checked by this memory preflight:"}{" "}
              {dialog.configSnapshot.epochs} epochs · LR{" "}
              {dialog.configSnapshot.learning_rate.toExponential(2)} · micro batch{" "}
              {dialog.configSnapshot.micro_batch_size}
            </p>
            {dialog.presentation.kind === "memory_shortage" ||
            dialog.presentation.kind === "unsupported_profile" ? (
              <dl className="training-memory-facts">
                <div>
                  <dt>Training peak</dt>
                  <dd>{formatBytes(dialog.refusal.training_peak_bytes)}</dd>
                </div>
                <div>
                  <dt>Inference reserve</dt>
                  <dd>{formatBytes(dialog.refusal.reserve_bytes)}</dd>
                </div>
                <div>
                  <dt>Required</dt>
                  <dd>{formatBytes(dialog.refusal.required_bytes)}</dd>
                </div>
                <div>
                  <dt>Available</dt>
                  <dd>{formatBytes(dialog.refusal.free_bytes)}</dd>
                </div>
              </dl>
            ) : null}
            {dialog.refusal.observed_at === undefined ? (
              <p className="training-muted">
                The server refusal did not include its observation time.
              </p>
            ) : (
              <p className="training-muted">
                Server observed{" "}
                <time dateTime={dialog.refusal.observed_at}>
                  {safeObservedTime(dialog.refusal.observed_at)}
                </time>
              </p>
            )}
            {dialog.refusal.profile_identity === undefined ? null : (
              <p className="training-muted">
                Memory profile <code>{dialog.refusal.profile_identity.slice(0, 16)}…</code>
              </p>
            )}
          </div>
        )}
      </Dialog>
    </main>
  )
}
