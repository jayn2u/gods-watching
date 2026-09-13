import type { AppearanceResponse, SearchRequest } from "../../app/client"

export type ResultsState =
  | Readonly<{ kind: "idle" }>
  | Readonly<{ kind: "loading"; request: SearchRequest }>
  | Readonly<{ kind: "ready"; request: SearchRequest; results: readonly AppearanceResponse[] }>
  | Readonly<{ kind: "empty"; request: SearchRequest }>
  | Readonly<{ kind: "error"; request: SearchRequest; message: string }>

export type CropAsset = Readonly<{
  etag: string | undefined
  representativeVersion: number | undefined
  url: string
}>

export type DetailState =
  | Readonly<{ kind: "closed" }>
  | Readonly<{ kind: "loading"; seed: AppearanceResponse }>
  | Readonly<{ kind: "ready"; appearance: AppearanceResponse; crop: CropAsset }>
  | Readonly<{ kind: "error"; appearance: AppearanceResponse; message: string }>
