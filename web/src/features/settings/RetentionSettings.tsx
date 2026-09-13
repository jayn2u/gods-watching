import { useEffect, useRef, useState } from "react"
import { HttpError, isAbortError, NetworkError, type SettingsResponse } from "../../app/client"
import { Button, Input, Panel, Status } from "../../components"
import type { RetentionSettingsProps } from "../cameras/cameraTypes"
import { BYTES_PER_DECIMAL_GB, parseQuotaGb, parseRetentionDays } from "./retentionValidation"
import "./settings.css"

type Draft = Readonly<{
  days: string
  quotaGb: string
}>

type FieldErrors = Readonly<{
  days: string | undefined
  quotaGb: string | undefined
}>

const EMPTY_ERRORS: FieldErrors = { days: undefined, quotaGb: undefined }

function draftFromSettings(settings: SettingsResponse): Draft {
  return {
    days: String(settings.retention_days),
    quotaGb: String(settings.quota_bytes / BYTES_PER_DECIMAL_GB),
  }
}

function apiErrorMessage(error: unknown): string {
  if (error instanceof HttpError) {
    return `${error.code}: ${error.message}`
  }
  if (error instanceof NetworkError) {
    return error.message
  }
  if (error instanceof Error) {
    return error.message
  }
  return "The retention settings could not be saved. Try again."
}

export function RetentionSettings({
  onSettingsSaved,
  onUnauthorized,
  settings,
  settingsClient,
}: RetentionSettingsProps) {
  const [draft, setDraft] = useState<Draft>({ days: "", quotaGb: "" })
  const [dirty, setDirty] = useState(false)
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>(EMPTY_ERRORS)
  const [formError, setFormError] = useState<string | undefined>(undefined)
  const [saving, setSaving] = useState(false)
  const requestGeneration = useRef(0)
  const activeController = useRef<AbortController | null>(null)

  useEffect(() => {
    if (settings.kind !== "ready" || dirty) {
      return
    }
    setDraft(draftFromSettings(settings.settings))
  }, [dirty, settings])

  useEffect(() => {
    return () => {
      requestGeneration.current += 1
      activeController.current?.abort()
    }
  }, [])

  function updateDraft(field: "days" | "quotaGb", value: string): void {
    setDirty(true)
    setFormError(undefined)
    setFieldErrors((current) => ({ ...current, [field]: undefined }))
    setDraft((current) => ({ ...current, [field]: value }))
  }

  async function saveSettings(): Promise<void> {
    if (saving || settings.kind !== "ready") {
      return
    }
    const days = parseRetentionDays(draft.days)
    const quota = parseQuotaGb(draft.quotaGb)
    const nextFieldErrors: FieldErrors = {
      days: days.kind === "invalid" ? days.message : undefined,
      quotaGb: quota.kind === "invalid" ? quota.message : undefined,
    }
    setFieldErrors(nextFieldErrors)
    if (days.kind === "invalid" || quota.kind === "invalid") {
      return
    }

    activeController.current?.abort()
    const controller = new AbortController()
    activeController.current = controller
    const generation = requestGeneration.current + 1
    requestGeneration.current = generation
    setSaving(true)
    setFormError(undefined)

    try {
      const saved = await settingsClient.patchSettings(
        { retention_days: days.value, quota_bytes: quota.value },
        controller.signal,
      )
      if (generation !== requestGeneration.current || controller.signal.aborted) {
        return
      }
      setSaving(false)
      setDirty(false)
      setDraft(draftFromSettings(saved))
      setFieldErrors(EMPTY_ERRORS)
      onSettingsSaved(saved)
    } catch (error) {
      if (generation !== requestGeneration.current || isAbortError(error)) {
        return
      }
      setSaving(false)
      if (error instanceof HttpError && error.status === 401) {
        onUnauthorized()
        return
      }
      setFormError(apiErrorMessage(error))
    } finally {
      if (generation === requestGeneration.current) {
        activeController.current = null
        setSaving(false)
      }
    }
  }

  return (
    <Panel eyebrow="Global policy" title="Retention settings">
      {settings.kind === "loading" ? (
        <div className="settings-state">
          <Status>Loading retention settings…</Status>
        </div>
      ) : null}
      {settings.kind === "error" ? (
        <p className="settings-state settings-state--error" role="alert">
          {settings.message}
        </p>
      ) : null}
      {settings.kind === "ready" ? (
        <form
          className="settings-form"
          noValidate
          onSubmit={(event) => {
            event.preventDefault()
            void saveSettings()
          }}
        >
          <p className="settings-copy">
            Keep committed person crops for a positive number of days and within a decimal GB
            budget. The server defaults to 7 days and 100 GB.
          </p>
          <div className="settings-form__fields">
            <Input
              error={fieldErrors.days}
              inputMode="numeric"
              label="Retention days"
              min="1"
              onChange={(event) => updateDraft("days", event.target.value)}
              step="1"
              type="number"
              value={draft.days}
            />
            <Input
              error={fieldErrors.quotaGb}
              inputMode="decimal"
              label="Managed quota (decimal GB)"
              min="0.000000001"
              onChange={(event) => updateDraft("quotaGb", event.target.value)}
              step="any"
              type="number"
              value={draft.quotaGb}
            />
          </div>
          <p className="settings-accounting">
            Managed storage includes physical crop files, including pending garbage collection, plus
            PostgreSQL application relations and indexes. Model assets, logs, and WAL are separate
            operational storage.
          </p>
          {formError === undefined ? null : (
            <p className="settings-form__error" role="alert">
              {formError}
            </p>
          )}
          <div className="settings-form__actions">
            <Button loading={saving} type="submit" variant="primary">
              Save retention
            </Button>
          </div>
        </form>
      ) : null}
    </Panel>
  )
}
