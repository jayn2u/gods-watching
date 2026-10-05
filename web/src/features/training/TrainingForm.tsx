import type { RefObject } from "react"
import type {
  TrainingConfig,
  TrainingDatasetStatus,
  TrainingPreflightResponse,
} from "../../app/client"
import { Button, Panel, Status } from "../../components"
import { NumericParameter } from "./NumericParameter"
import { TRAINING_PARAMETER_BOUNDS } from "./trainingModel"

export type TrainingFormProps = Readonly<{
  config: TrainingConfig
  dataset: TrainingDatasetStatus | null
  datasetLoading: boolean
  datasetMessage: string
  estimate: TrainingPreflightResponse | null
  estimateLoading: boolean
  estimateMessage: string | undefined
  canStart: boolean
  starting: boolean
  draftsValid: boolean
  onConfigChange: (config: TrainingConfig) => void
  onDraftValidityChange: (fieldKey: string, valid: boolean) => void
  onStart: () => void
  onRefreshDataset: () => void
  onRefreshEstimate: () => void
  estimateRefreshButtonRef: RefObject<HTMLButtonElement | null>
}>

function formatBytes(bytes: number): string {
  const gib = bytes / 1024 ** 3
  return `${gib.toFixed(2)} GiB`
}

function configNumberLabel(value: number): string {
  return value.toLocaleString(undefined, { maximumFractionDigits: 8 })
}

export function TrainingForm({
  canStart,
  config,
  dataset,
  datasetLoading,
  datasetMessage,
  draftsValid,
  estimate,
  estimateLoading,
  estimateMessage,
  onConfigChange,
  onDraftValidityChange,
  onRefreshDataset,
  onRefreshEstimate,
  onStart,
  estimateRefreshButtonRef,
  starting,
}: TrainingFormProps) {
  function update<K extends keyof TrainingConfig>(field: K, value: TrainingConfig[K]): void {
    onConfigChange({ ...config, [field]: value })
  }

  const snapshot = dataset?.snapshot ?? null
  const isDatasetValid = dataset?.registered === true && dataset.valid && snapshot !== null

  return (
    <div className="training-builder">
      <Panel eyebrow="Registered input" title="CUHK-PEDES dataset">
        <div className="training-dataset">
          <div className="training-dataset__status">
            <Status tone={datasetLoading ? "neutral" : isDatasetValid ? "live" : "warning"}>
              {datasetLoading
                ? "Checking validation"
                : isDatasetValid
                  ? "Validated"
                  : "Unavailable"}
            </Status>
            <Button disabled={datasetLoading} onClick={onRefreshDataset} variant="secondary">
              Refresh status
            </Button>
          </div>
          <p
            className={isDatasetValid ? "training-muted" : "training-notice"}
            role={isDatasetValid ? undefined : "status"}
          >
            {datasetMessage}
          </p>
          {snapshot === null ? null : (
            <>
              <dl className="training-dataset__facts">
                <div>
                  <dt>Protocol</dt>
                  <dd>{snapshot.protocol}</dd>
                </div>
                <div>
                  <dt>Images</dt>
                  <dd>{snapshot.image_count.toLocaleString()}</dd>
                </div>
                <div>
                  <dt>Captions</dt>
                  <dd>{snapshot.caption_count.toLocaleString()}</dd>
                </div>
                <div>
                  <dt>Identities</dt>
                  <dd>{snapshot.identity_count.toLocaleString()}</dd>
                </div>
              </dl>
              <table className="training-split-table">
                <caption className="sr-only">Dataset split counts</caption>
                <thead>
                  <tr className="training-split-table__row--header">
                    <th scope="col">Split</th>
                    <th scope="col">Images</th>
                    <th scope="col">Captions</th>
                    <th scope="col">Identities</th>
                  </tr>
                </thead>
                <tbody>
                  {(["train", "val", "test"] as const).map((split) => (
                    <tr key={split}>
                      <th scope="row">{split === "val" ? "Validation" : split}</th>
                      <td>{snapshot.split_counts[split].images.toLocaleString()}</td>
                      <td>{snapshot.split_counts[split].captions.toLocaleString()}</td>
                      <td>{snapshot.split_counts[split].identities.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="training-dataset__fingerprint">
                Snapshot <code>{snapshot.fingerprint.slice(0, 16)}…</code>
              </p>
            </>
          )}
          <p className="training-copy">
            The dataset root is configured by the operator. This screen does not upload files or
            accept a host path.
          </p>
        </div>
      </Panel>

      <Panel eyebrow="Immutable at submit" title="Training configuration">
        <div className="training-form">
          <div className="training-form__section">
            <h3>Basic settings</h3>
            <div className="training-form__parameters">
              <NumericParameter
                fieldKey="epochs"
                integer
                label="Epochs"
                max={TRAINING_PARAMETER_BOUNDS.epochs.max}
                min={TRAINING_PARAMETER_BOUNDS.epochs.min}
                onChange={(value) => update("epochs", value)}
                onDraftValidityChange={onDraftValidityChange}
                step={TRAINING_PARAMETER_BOUNDS.epochs.step}
                value={config.epochs}
                formatValue={configNumberLabel}
              />
              <NumericParameter
                fieldKey="learning_rate"
                label="Learning rate"
                logScale
                max={TRAINING_PARAMETER_BOUNDS.learning_rate.max}
                min={TRAINING_PARAMETER_BOUNDS.learning_rate.min}
                onChange={(value) => update("learning_rate", value)}
                onDraftValidityChange={onDraftValidityChange}
                step={TRAINING_PARAMETER_BOUNDS.learning_rate.step}
                value={config.learning_rate}
                description="Logarithmic slider from 0.0000001 to 0.001."
                formatValue={(value) => value.toExponential(2)}
              />
              <NumericParameter
                fieldKey="micro_batch_size"
                integer
                label="Micro batch size"
                max={TRAINING_PARAMETER_BOUNDS.micro_batch_size.max}
                min={TRAINING_PARAMETER_BOUNDS.micro_batch_size.min}
                onChange={(value) => update("micro_batch_size", value)}
                onDraftValidityChange={onDraftValidityChange}
                step={TRAINING_PARAMETER_BOUNDS.micro_batch_size.step}
                value={config.micro_batch_size}
                description="Controls the contrastive negative pool in each training step."
              />
              <NumericParameter
                fieldKey="weight_decay"
                label="Weight decay"
                max={TRAINING_PARAMETER_BOUNDS.weight_decay.max}
                min={TRAINING_PARAMETER_BOUNDS.weight_decay.min}
                onChange={(value) => update("weight_decay", value)}
                onDraftValidityChange={onDraftValidityChange}
                step={TRAINING_PARAMETER_BOUNDS.weight_decay.step}
                value={config.weight_decay}
              />
            </div>
          </div>

          <details className="training-form__advanced">
            <summary>Advanced settings</summary>
            <div className="training-form__parameters">
              <NumericParameter
                fieldKey="gradient_accumulation"
                integer
                label="Gradient accumulation"
                max={TRAINING_PARAMETER_BOUNDS.gradient_accumulation.max}
                min={TRAINING_PARAMETER_BOUNDS.gradient_accumulation.min}
                onChange={(value) => update("gradient_accumulation", value)}
                onDraftValidityChange={onDraftValidityChange}
                step={TRAINING_PARAMETER_BOUNDS.gradient_accumulation.step}
                value={config.gradient_accumulation}
                description="Raises the optimizer update batch; it does not increase the per-step negative pool."
              />
              <NumericParameter
                fieldKey="warmup_ratio"
                label="Warmup ratio"
                max={TRAINING_PARAMETER_BOUNDS.warmup_ratio.max}
                min={TRAINING_PARAMETER_BOUNDS.warmup_ratio.min}
                onChange={(value) => update("warmup_ratio", value)}
                onDraftValidityChange={onDraftValidityChange}
                step={TRAINING_PARAMETER_BOUNDS.warmup_ratio.step}
                value={config.warmup_ratio}
              />
              <NumericParameter
                fieldKey="seed"
                integer
                label="Random seed"
                max={TRAINING_PARAMETER_BOUNDS.seed.max}
                min={TRAINING_PARAMETER_BOUNDS.seed.min}
                onChange={(value) => update("seed", value)}
                onDraftValidityChange={onDraftValidityChange}
                step={TRAINING_PARAMETER_BOUNDS.seed.step}
                value={config.seed}
              />
              <div className="training-parameter">
                <label className="training-check" htmlFor="training-early-stopping-enabled">
                  <input
                    checked={config.early_stopping_patience !== null}
                    id="training-early-stopping-enabled"
                    onChange={(event) => {
                      update("early_stopping_patience", event.currentTarget.checked ? 5 : null)
                      onDraftValidityChange("early_stopping_patience", true)
                    }}
                    type="checkbox"
                  />
                  <span>Early stopping</span>
                </label>
                {config.early_stopping_patience === null ? (
                  <p className="training-parameter__description">Disabled</p>
                ) : (
                  <NumericParameter
                    fieldKey="early_stopping_patience"
                    integer
                    label="Patience (epochs)"
                    max={TRAINING_PARAMETER_BOUNDS.early_stopping_patience.max}
                    min={TRAINING_PARAMETER_BOUNDS.early_stopping_patience.min}
                    onChange={(value) => update("early_stopping_patience", value)}
                    onDraftValidityChange={onDraftValidityChange}
                    step={TRAINING_PARAMETER_BOUNDS.early_stopping_patience.step}
                    value={config.early_stopping_patience}
                  />
                )}
              </div>
              <NumericParameter
                fieldKey="gradient_clipping_norm"
                label="Gradient clipping norm"
                max={TRAINING_PARAMETER_BOUNDS.gradient_clipping_norm.max}
                min={TRAINING_PARAMETER_BOUNDS.gradient_clipping_norm.min}
                onChange={(value) => update("gradient_clipping_norm", value)}
                onDraftValidityChange={onDraftValidityChange}
                step={TRAINING_PARAMETER_BOUNDS.gradient_clipping_norm.step}
                value={config.gradient_clipping_norm}
              />
              <label className="training-select-field" htmlFor="training-mixed-precision">
                <span>Mixed precision</span>
                <select
                  id="training-mixed-precision"
                  onChange={(event) => {
                    const value = event.currentTarget.value
                    if (value === "fp16" || value === "fp32") update("mixed_precision", value)
                  }}
                  value={config.mixed_precision}
                >
                  <option value="fp16">FP16</option>
                  <option value="fp32">FP32</option>
                </select>
              </label>
              <label className="training-check" htmlFor="training-gradient-checkpointing">
                <input
                  checked={config.gradient_checkpointing}
                  id="training-gradient-checkpointing"
                  onChange={(event) =>
                    update("gradient_checkpointing", event.currentTarget.checked)
                  }
                  type="checkbox"
                />
                <span>Gradient checkpointing</span>
              </label>
            </div>
          </details>

          <section className="training-config-summary" aria-labelledby="training-summary-heading">
            <strong id="training-summary-heading">Run summary</strong>
            <span>
              {config.epochs} epochs · LR {config.learning_rate.toExponential(2)}
            </span>
            <span>
              Micro batch {config.micro_batch_size} × accumulation {config.gradient_accumulation} =
              effective batch {config.micro_batch_size * config.gradient_accumulation}
            </span>
            <span>{config.mixed_precision.toUpperCase()} · AdamW · warmup + cosine schedule</span>
          </section>

          <div className="training-estimate" aria-live="polite" aria-busy={estimateLoading}>
            <div className="training-estimate__heading">
              <strong>Live memory preflight</strong>
              <div className="training-estimate__actions">
                <Status
                  tone={estimateLoading ? "warning" : estimate?.admitted ? "live" : "neutral"}
                >
                  {estimateLoading ? "Checking" : estimate?.admitted ? "Admitted" : "Waiting"}
                </Status>
                <Button
                  disabled={estimateLoading || !draftsValid}
                  onClick={onRefreshEstimate}
                  ref={estimateRefreshButtonRef}
                  variant="secondary"
                >
                  Refresh estimate
                </Button>
              </div>
            </div>
            {estimate === null ? (
              <p className="training-muted">
                {estimateMessage ?? "A current estimate is required before starting."}
              </p>
            ) : (
              <>
                <dl className="training-memory-facts">
                  <div>
                    <dt>Training peak</dt>
                    <dd>{formatBytes(estimate.training_peak_bytes)}</dd>
                  </div>
                  <div>
                    <dt>Inference reserve</dt>
                    <dd>{formatBytes(estimate.reserve_bytes)}</dd>
                  </div>
                  <div>
                    <dt>Required</dt>
                    <dd>{formatBytes(estimate.required_bytes)}</dd>
                  </div>
                  <div>
                    <dt>Available now</dt>
                    <dd>{formatBytes(estimate.free_bytes)}</dd>
                  </div>
                </dl>
                <p className="training-muted">
                  Observed{" "}
                  <time dateTime={estimate.observed_at}>
                    {new Date(estimate.observed_at).toLocaleString()}
                  </time>
                </p>
              </>
            )}
          </div>

          {!draftsValid ? (
            <p className="training-form__error" role="alert">
              Complete every numeric value before refreshing the estimate or starting training.
            </p>
          ) : null}
          <div className="training-form__actions">
            <Button disabled={!canStart || starting} onClick={onStart}>
              {starting ? "Submitting…" : "Start training"}
            </Button>
          </div>
        </div>
      </Panel>
    </div>
  )
}
