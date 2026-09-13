import type { CameraState } from "../../app/layout/AppShell"
import { Button } from "../../components/Button"
import { Input } from "../../components/Input"
import type { SearchDraft, SearchMode } from "./searchModel"

export type SearchFormProps = Readonly<{
  cameras: CameraState
  draft: SearchDraft
  error: string | undefined
  onBrowse: () => void
  onCameraToggle: (cameraId: string) => void
  onCancel: () => void
  onFromChange: (value: string) => void
  onModeChange: (mode: SearchMode) => void
  onQueryChange: (value: string) => void
  onSubmit: () => void
  onToChange: (value: string) => void
  pending: boolean
}>

export function SearchForm({
  cameras,
  draft,
  error,
  onBrowse,
  onCameraToggle,
  onCancel,
  onFromChange,
  onModeChange,
  onQueryChange,
  onSubmit,
  onToChange,
  pending,
}: SearchFormProps) {
  const inlineQueryError =
    draft.mode === "text" && error?.startsWith("Enter a person description") ? error : undefined

  return (
    <form
      className="search-form"
      onSubmit={(event) => {
        event.preventDefault()
        onSubmit()
      }}
    >
      <Input
        autoComplete="off"
        error={inlineQueryError}
        hint="Describe clothing and context in English. The service validates supported text and token limits."
        label="Describe a person"
        onChange={(event) => onQueryChange(event.currentTarget.value)}
        placeholder="a person in a black T-shirt, with a red bag"
        value={draft.query}
      />
      <div className="search-form__filters">
        <fieldset className="search-filter-group">
          <legend>Search mode and sort</legend>
          <label className="search-choice">
            <input
              checked={draft.mode === "text"}
              name="search-mode"
              onChange={() => onModeChange("text")}
              type="radio"
            />
            <span>Description · cosine similarity</span>
          </label>
          <label className="search-choice">
            <input
              checked={draft.mode === "browse"}
              name="search-mode"
              onChange={() => onModeChange("browse")}
              type="radio"
            />
            <span>Browse · newest first</span>
          </label>
        </fieldset>
        <fieldset className="search-filter-group">
          <legend>Camera filters</legend>
          <CameraChoices
            cameras={cameras}
            selectedIds={draft.cameraIds}
            onToggle={onCameraToggle}
          />
        </fieldset>
        <div className="search-form__dates">
          <Input
            hint="Converted to UTC for the inclusive server filter."
            label="From (local, inclusive)"
            onChange={(event) => onFromChange(event.currentTarget.value)}
            type="datetime-local"
            value={draft.from}
          />
          <Input
            hint="Results overlap this end time, inclusive."
            label="To (local, inclusive)"
            onChange={(event) => onToChange(event.currentTarget.value)}
            type="datetime-local"
            value={draft.to}
          />
        </div>
      </div>
      {error !== undefined && inlineQueryError === undefined ? (
        <p className="search-form__error" role="alert">
          {error}
        </p>
      ) : null}
      <div className="search-form__actions">
        <Button loading={pending} type="submit" variant="primary">
          Search
        </Button>
        <Button disabled={pending} onClick={onBrowse} type="button">
          Browse latest
        </Button>
        {pending ? (
          <Button onClick={onCancel} type="button">
            Cancel
          </Button>
        ) : null}
      </div>
    </form>
  )
}

function CameraChoices({
  cameras,
  onToggle,
  selectedIds,
}: Readonly<{
  cameras: CameraState
  onToggle: (cameraId: string) => void
  selectedIds: readonly string[]
}>) {
  if (cameras.kind === "loading") {
    return <p className="search-filter-group__hint">Loading camera filters…</p>
  }
  if (cameras.kind === "error") {
    return (
      <p className="search-filter-group__hint search-filter-group__hint--error">
        {cameras.message}
      </p>
    )
  }
  if (cameras.cameras.length === 0) {
    return (
      <p className="search-filter-group__hint">
        No cameras are configured; all cameras are included.
      </p>
    )
  }
  return (
    <div className="search-choice-list" data-search-camera-filters>
      {cameras.cameras.map((camera) => (
        <label className="search-choice" key={camera.camera_id}>
          <input
            checked={selectedIds.includes(camera.camera_id)}
            onChange={() => onToggle(camera.camera_id)}
            type="checkbox"
          />
          <span>{camera.name}</span>
        </label>
      ))}
    </div>
  )
}
