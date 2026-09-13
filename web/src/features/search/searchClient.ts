import type { ApiClient } from "../../app/client"

export type SearchClient = Pick<ApiClient, "search" | "getAppearance" | "getAppearanceCrop">
