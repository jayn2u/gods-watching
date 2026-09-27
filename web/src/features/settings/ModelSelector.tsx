import { useEffect, useRef, useState } from "react"
import {
  type ClipModelOption,
  type ClipModelTransition,
  HttpError,
  isAbortError,
  type ModelSettingsResponse,
  NetworkError,
  type SwitchPreflight,
} from "../../app/client"
import { Button, Dialog, Panel, Status } from "../../components"
import type { ModelSettingsClient } from "../cameras/cameraTypes"
import "./settings.css"

type ModelSettingsState =
  | { readonly kind: "loading" }
  | { readonly kind: "ready"; readonly response: ModelSettingsResponse }
  | { readonly kind: "error"; readonly message: string }

type ModelSelectorProps = Readonly<{
  client: ModelSettingsClient
  onUnauthorized: () => void
}>

const POLL_INTERVAL_MS = 1_000

const PREPARED_REASON_MESSAGES: Readonly<Record<string, string>> = {
  asset_missing: "Model files are missing. Run prepare.",
  asset_corrupt: "Model files are corrupt. Run prepare.",
  identity_marker_missing: "Model identity metadata is missing. Run prepare.",
  identity_marker_invalid: "Model identity metadata is invalid. Run prepare.",
  identity_marker_mismatch: "Model identity does not match the prepared registry. Run prepare.",
  registry_lock_mismatch: "The prepared model does not match the registry lock. Run prepare.",
  lock_mismatch: "The prepared model does not match the registry lock. Run prepare.",
  lock_unreadable: "The model lock could not be read. Run prepare.",
  lock_invalid: "The model lock is invalid. Run prepare.",
}

function preparedReasonMessage(reason: string | null): string {
  if (reason === null || reason.trim() === "") {
    return "Model assets are not prepared. Run prepare."
  }
  const normalized = reason.trim()
  return (
    PREPARED_REASON_MESSAGES[normalized] ??
    (/^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$/.test(normalized)
      ? "Model assets are unavailable. Run prepare."
      : normalized)
  )
}

function isActiveTransition(transition: ClipModelTransition | null): boolean {
  return transition !== null && transition.phase !== "succeeded" && transition.phase !== "failed"
}

function phaseLabel(phase: ClipModelTransition["phase"]): string {
  switch (phase) {
    case "queued":
      return "Queued"
    case "preparing":
      return "Preparing model"
    case "reindexing":
      return "Re-embedding retained crops"
    case "activating":
      return "Activating model"
    case "rolling_back":
      return "Recovering model"
    case "succeeded":
      return "Model change complete"
    case "failed":
      return "Model change failed"
  }
}

function statusTone(phase: ClipModelTransition["phase"]): "live" | "warning" | "offline" {
  if (phase === "succeeded") {
    return "live"
  }
  if (phase === "failed") {
    return "offline"
  }
  return "warning"
}

function progressLabel(transition: ClipModelTransition): string {
  if (transition.phase === "preparing") {
    return "Preparing the selected model for use…"
  }
  if (transition.phase === "activating") {
    return "Applying the new model…"
  }
  if (transition.phase === "rolling_back") {
    return "Restoring search and analysis safely…"
  }
  if (transition.total === 0) {
    return "No retained crops need re-embedding."
  }
  return `${transition.processed} of ${transition.total} retained crops processed.`
}

function skipReasonLabel(reason: string): string {
  switch (reason) {
    case "missing_crop":
      return "Missing crop"
    case "corrupt_crop":
      return "Unreadable crop"
    default:
      return reason.replaceAll("_", " ")
  }
}

function selectionForResponse(response: ModelSettingsResponse): string {
  const transition = response.transition
  if (transition !== null && transition.phase === "failed") {
    return response.active_model_id
  }
  if (transition !== null && isActiveTransition(transition)) {
    return transition.target_model_id
  }
  return response.active_model_id
}

function apiErrorMessage(error: unknown, operation: "load" | "apply"): string {
  if (error instanceof HttpError) {
    if (error.status === 409) {
      return "Another model change is already in progress. The current status is shown below."
    }
    if (error.status === 422) {
      return "That model is not prepared for this deployment. Run prepare and try again."
    }
    if (error.status >= 500) {
      return "The model service returned an error. Try again shortly."
    }
  }
  if (error instanceof NetworkError) {
    return operation === "load"
      ? "Model settings are unavailable. Check the session service and retry."
      : "The model change could not be started. Check the session service and try again."
  }
  return operation === "load"
    ? "Model settings could not be loaded. Try again."
    : "The model change could not be started. Try again."
}

function optionDescription(option: ClipModelOption): string {
  return `${option.dimension}-dimensional embeddings · revision ${option.revision}`
}

export function ModelSelector({ client, onUnauthorized }: ModelSelectorProps) {
  const [state, setState] = useState<ModelSettingsState>({ kind: "loading" })
  const [selectedModelId, setSelectedModelId] = useState("")
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [applying, setApplying] = useState(false)
  const [preflight, setPreflight] = useState<SwitchPreflight | null>(null)
  const [preflightLoading, setPreflightLoading] = useState(false)
  const confirmationPreflight = useRef<SwitchPreflight | null>(null)
  const preflightGeneration = useRef(0)
  const preflightInFlight = useRef(
    new Map<
      string,
      { readonly generation: number; readonly promise: Promise<SwitchPreflight | null> }
    >(),
  )
  const applyInFlight = useRef(false)
  const [formError, setFormError] = useState<string | undefined>(undefined)
  const requestGeneration = useRef(0)
  const activeController = useRef<AbortController | null>(null)
  const pollTimeout = useRef<number | null>(null)
  const selectionTouched = useRef(false)
  const applyTriggerRef = useRef<HTMLButtonElement | null>(null)

  function clearPoll(): void {
    if (pollTimeout.current !== null) {
      window.clearTimeout(pollTimeout.current)
      pollTimeout.current = null
    }
  }

  function beginRequest(): { readonly controller: AbortController; readonly generation: number } {
    activeController.current?.abort()
    const controller = new AbortController()
    const generation = requestGeneration.current + 1
    requestGeneration.current = generation
    activeController.current = controller
    return { controller, generation }
  }

  function isCurrent(generation: number, controller: AbortController): boolean {
    return generation === requestGeneration.current && !controller.signal.aborted
  }

  function acceptResponse(response: ModelSettingsResponse, preserveSelection = false): void {
    setState({ kind: "ready", response })
    setFormError(undefined)
    if (!preserveSelection && !selectionTouched.current) {
      setSelectedModelId(selectionForResponse(response))
    }
  }

  function schedulePoll(response: ModelSettingsResponse): void {
    clearPoll()
    if (!response.maintenance && !isActiveTransition(response.transition)) {
      return
    }
    pollTimeout.current = window.setTimeout(() => {
      pollTimeout.current = null
      void refreshStatus()
    }, POLL_INTERVAL_MS)
  }

  async function refreshStatus(): Promise<void> {
    const { controller, generation } = beginRequest()
    try {
      const response = await client.getModelSettings(controller.signal)
      if (!isCurrent(generation, controller)) {
        return
      }
      acceptResponse(response)
      schedulePoll(response)
    } catch (error) {
      if (!isCurrent(generation, controller) || isAbortError(error)) {
        return
      }
      if (error instanceof HttpError && error.status === 401) {
        onUnauthorized()
        return
      }
      setState({ kind: "error", message: apiErrorMessage(error, "load") })
    } finally {
      if (activeController.current === controller) {
        activeController.current = null
      }
    }
  }

  const refreshStatusRef = useRef<() => Promise<void>>(() => Promise.resolve())
  refreshStatusRef.current = refreshStatus

  useEffect(() => {
    void refreshStatusRef.current()
    return () => {
      requestGeneration.current += 1
      activeController.current?.abort()
      activeController.current = null
      if (pollTimeout.current !== null) {
        window.clearTimeout(pollTimeout.current)
        pollTimeout.current = null
      }
    }
  }, [])

  async function refreshPreflight(modelId: string): Promise<SwitchPreflight | null> {
    const existing = preflightInFlight.current.get(modelId)
    if (existing?.generation === preflightGeneration.current) return existing.promise

    const generation = ++preflightGeneration.current
    setPreflightLoading(true)
    const request = (async () => {
      try {
        const result = await client.getModelPreflight(modelId, new AbortController().signal)
        if (generation === preflightGeneration.current) setPreflight(result)
        return generation === preflightGeneration.current ? result : null
      } catch (error) {
        if (generation === preflightGeneration.current) {
          setPreflight(null)
          setFormError(
            error instanceof HttpError && error.status === 401
              ? "Session expired. Sign in again."
              : "Model preflight is unavailable. Try again.",
          )
          if (error instanceof HttpError && error.status === 401) onUnauthorized()
        }
        return null
      } finally {
        if (generation === preflightGeneration.current) setPreflightLoading(false)
      }
    })()
    const inFlight = { generation, promise: request }
    preflightInFlight.current.set(modelId, inFlight)
    const clearInFlight = () => {
      if (preflightInFlight.current.get(modelId) === inFlight) {
        preflightInFlight.current.delete(modelId)
      }
    }
    void request.then(clearInFlight, clearInFlight)
    return request
  }

  const refreshPreflightRef = useRef<(modelId: string) => Promise<SwitchPreflight | null>>(() =>
    Promise.resolve(null),
  )
  refreshPreflightRef.current = refreshPreflight

  const pollCandidate =
    state.kind === "ready"
      ? state.response.models.find((model) => model.model_id === selectedModelId)
      : undefined
  const shouldPollPreflight =
    state.kind === "ready" &&
    selectedModelId !== "" &&
    selectedModelId !== state.response.active_model_id &&
    !state.response.maintenance &&
    !isActiveTransition(state.response.transition) &&
    pollCandidate?.prepared === true &&
    pollCandidate.quality_passed

  useEffect(() => {
    if (!shouldPollPreflight) return
    const timer = window.setInterval(() => {
      void refreshPreflightRef.current(selectedModelId)
    }, 5_000)
    return () => window.clearInterval(timer)
  }, [selectedModelId, shouldPollPreflight])

  function chooseModel(modelId: string): void {
    if (
      state.kind !== "ready" ||
      applying ||
      state.response.maintenance ||
      isActiveTransition(state.response.transition)
    ) {
      return
    }
    selectionTouched.current = true
    setSelectedModelId(modelId)
    setPreflight(null)
    setFormError(undefined)
    if (modelId !== state.response.active_model_id) void refreshPreflight(modelId)
  }

  function selectedOption(): ClipModelOption | undefined {
    if (state.kind !== "ready") {
      return undefined
    }
    return state.response.models.find((model) => model.model_id === selectedModelId)
  }

  async function openConfirmation(): Promise<void> {
    const option = selectedOption()
    if (
      state.kind !== "ready" ||
      option === undefined ||
      !option.prepared ||
      !option.quality_passed ||
      selectedModelId === state.response.active_model_id ||
      applying ||
      state.response.maintenance ||
      isActiveTransition(state.response.transition)
    ) {
      return
    }
    const refreshed = await refreshPreflight(selectedModelId)
    if (refreshed === null || !refreshed.eligible) return
    confirmationPreflight.current = refreshed
    setFormError(undefined)
    setConfirmOpen(true)
  }

  async function applySelectedModel(): Promise<void> {
    const option = selectedOption()
    if (
      state.kind !== "ready" ||
      applyInFlight.current ||
      option === undefined ||
      !option.prepared ||
      !option.quality_passed ||
      selectedModelId === state.response.active_model_id ||
      applying ||
      state.response.maintenance ||
      isActiveTransition(state.response.transition)
    ) {
      return
    }

    applyInFlight.current = true
    try {
      const refreshed = await refreshPreflight(selectedModelId)
      if (refreshed === null || !refreshed.eligible) {
        setConfirmOpen(false)
        return
      }
      const confirmed = confirmationPreflight.current
      if (
        confirmed === null ||
        refreshed.estimated_seconds !== confirmed.estimated_seconds ||
        refreshed.retained_count !== confirmed.retained_count ||
        refreshed.estimated_missing_count !== confirmed.estimated_missing_count
      ) {
        setFormError("The estimate changed. Review the updated preflight and confirm again.")
        setConfirmOpen(false)
        return
      }
      setConfirmOpen(false)
      clearPoll()
      setApplying(true)
      setFormError(undefined)
      const targetModelId = selectedModelId
      const { controller, generation } = beginRequest()
      try {
        const response = await client.applyModel(targetModelId, controller.signal)
        if (!isCurrent(generation, controller)) {
          return
        }
        selectionTouched.current = false
        acceptResponse(response)
        schedulePoll(response)
      } catch (error) {
        if (!isCurrent(generation, controller) || isAbortError(error)) {
          return
        }
        if (error instanceof HttpError && error.status === 401) {
          onUnauthorized()
          return
        }
        setFormError(apiErrorMessage(error, "apply"))
        setApplying(false)
        if (error instanceof HttpError && error.status === 409) {
          void refreshStatus()
        } else if (
          error instanceof NetworkError ||
          (error instanceof HttpError && error.status >= 500)
        ) {
          setFormError("The model change request could not be confirmed. Refreshing its status…")
          void refreshStatus()
        }
      } finally {
        if (isCurrent(generation, controller)) {
          setApplying(false)
        }
        if (activeController.current === controller) {
          activeController.current = null
        }
      }
    } finally {
      applyInFlight.current = false
    }
  }

  const response = state.kind === "ready" ? state.response : undefined
  const transition = response?.transition ?? null
  const activeTransition = isActiveTransition(transition)
  const maintenance = response?.maintenance ?? false
  const selected = selectedOption()
  const canApply =
    response !== undefined &&
    selected?.prepared === true &&
    selected.quality_passed === true &&
    preflight?.target_model_id === selected.model_id &&
    preflight.eligible &&
    !preflightLoading &&
    selected.model_id !== response.active_model_id &&
    !applying &&
    !maintenance &&
    !activeTransition

  return (
    <Panel eyebrow="Retrieval" title="Person search model">
      <div className="model-settings">
        <p className="settings-copy">
          Choose the model used for person search and image similarity. Models are prepared during
          deployment; changing the model re-embeds retained person crops.
        </p>
        {state.kind === "loading" ? (
          <div className="settings-state" aria-busy="true">
            <Status>Loading model catalog…</Status>
          </div>
        ) : null}
        {state.kind === "error" ? (
          <div className="model-settings__error" role="alert">
            <p>{state.message}</p>
            <Button onClick={() => void refreshStatus()}>Retry</Button>
          </div>
        ) : null}
        {response === undefined ? null : (
          <>
            <fieldset
              className="model-options"
              disabled={applying || maintenance || activeTransition}
            >
              <legend className="model-options__legend">Available models</legend>
              {response.models.map((model) => {
                const inputId = `model-option-${model.model_id.replaceAll(/[^a-zA-Z0-9_-]/g, "-")}`
                const reasonId = `${inputId}-reason`
                return (
                  <label
                    className={`model-option${model.model_id === selectedModelId ? " model-option--selected" : ""}${model.prepared && model.quality_passed ? "" : " model-option--disabled"}`}
                    htmlFor={inputId}
                    key={model.model_id}
                  >
                    <input
                      aria-describedby={
                        model.prepared && model.quality_passed ? undefined : reasonId
                      }
                      checked={model.model_id === selectedModelId}
                      disabled={!model.prepared || !model.quality_passed}
                      id={inputId}
                      name="clip-model"
                      onChange={() => chooseModel(model.model_id)}
                      type="radio"
                      value={model.model_id}
                    />
                    <span className="model-option__copy">
                      <span className="model-option__title">
                        <strong>{model.display_name}</strong>
                        {model.model_id === response.active_model_id ? (
                          <Status announce={false} tone="live">
                            Active
                          </Status>
                        ) : null}
                      </span>
                      <span className="model-option__meta">{optionDescription(model)}</span>
                      {model.prepared && model.quality_passed ? (
                        <span className="model-option__quality">Quality approved</span>
                      ) : null}
                      {model.prepared && !model.quality_passed ? (
                        <span className="model-option__reason" id={reasonId}>
                          Quality blocked:{" "}
                          {model.quality_reason ?? "Required quality evidence is unavailable."}
                        </span>
                      ) : null}
                      {model.prepared ? null : (
                        <span className="model-option__reason" id={reasonId}>
                          Unavailable: {preparedReasonMessage(model.reason)}
                        </span>
                      )}
                    </span>
                  </label>
                )
              })}
            </fieldset>
            {selected !== undefined &&
            selected.model_id !== response.active_model_id &&
            selected.prepared &&
            selected.quality_passed ? (
              <div className="model-preflight" role="status">
                {preflightLoading ? <p>Refreshing model preflight…</p> : null}
                {preflight?.target_model_id === selected.model_id ? (
                  <>
                    <p>
                      Retained crops: {preflight.retained_count}. Expected skipped crops:{" "}
                      {preflight.estimated_missing_count}.
                    </p>
                    <p>
                      Estimated pause:{" "}
                      {preflight.estimated_seconds === null
                        ? "unavailable"
                        : `${Math.ceil(preflight.estimated_seconds)} seconds`}{" "}
                      (limit {preflight.max_seconds} seconds).
                    </p>
                    <p>
                      {preflight.eligible
                        ? "Ready to confirm."
                        : `Unavailable: ${preflight.reason?.replaceAll("_", " ") ?? "preflight failed"}`}
                    </p>
                    {preflight.eligible ? (
                      <Button onClick={() => void refreshPreflight(selected.model_id)}>
                        Refresh estimate
                      </Button>
                    ) : null}
                  </>
                ) : null}
              </div>
            ) : null}
            {maintenance ? (
              <p className="model-settings__maintenance" role="status">
                Person analysis and search are paused while this model change completes.
              </p>
            ) : null}
            {transition === null ? null : <TransitionStatus transition={transition} />}
            {formError === undefined ? null : (
              <p className="model-settings__error" role="alert">
                {formError}
              </p>
            )}
            <div className="model-settings__actions">
              <Button
                disabled={!canApply}
                onClick={() => void openConfirmation()}
                ref={applyTriggerRef}
                loading={applying}
                variant="primary"
              >
                Apply model
              </Button>
            </div>
          </>
        )}
      </div>
      <Dialog
        confirmLabel="Start model change"
        description="Changing the retrieval model pauses person analysis and search until retained crops are ready."
        onClose={() => setConfirmOpen(false)}
        onConfirm={() => void applySelectedModel()}
        open={confirmOpen}
        returnFocusRef={applyTriggerRef}
        title="Pause person analysis and search?"
      >
        {preflight !== null ? (
          <p>
            Retained crops: {preflight.retained_count}; expected skipped crops:{" "}
            {preflight.estimated_missing_count}; estimated pause:{" "}
            {Math.ceil(preflight.estimated_seconds ?? 0)} seconds.
          </p>
        ) : null}
        <p>
          Transition state stays durable while the service restores a safe committed model. The
          settings page will show the outcome when recovery finishes.
        </p>
      </Dialog>
    </Panel>
  )
}

function TransitionStatus({ transition }: { readonly transition: ClipModelTransition }) {
  const progressValue =
    transition.total === 0 ? 0 : Math.min(transition.processed, transition.total)
  return (
    <div className="model-transition" aria-live="polite">
      <div className="model-transition__header">
        <Status tone={statusTone(transition.phase)}>{phaseLabel(transition.phase)}</Status>
        <span className="model-transition__target">{transition.target_model_id}</span>
      </div>
      <p className="model-transition__progress-copy">{progressLabel(transition)}</p>
      {transition.phase === "reindexing" || transition.phase === "activating" ? (
        <progress
          aria-label="Model change progress"
          className="model-transition__progress"
          max={Math.max(transition.total, 1)}
          value={progressValue}
        />
      ) : null}
      {transition.skipped > 0 ? (
        <div className="model-transition__skips">
          <strong>{transition.skipped} crops skipped</strong>
          <ul>
            {Object.entries(transition.skip_reasons).map(([reason, count]) => (
              <li key={reason}>
                {skipReasonLabel(reason)}: {count}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {transition.error === null ? null : (
        <p className="model-transition__error" role="alert">
          {transition.error}
        </p>
      )}
    </div>
  )
}
