import { useCallback, useEffect, useRef, useState } from "react"
import { Status } from "../components/Status"
import type { WallSettingsState } from "../features/wall/LiveWall"
import { AuthScreen, type LoginOutcome } from "./auth/AuthScreen"
import {
  apiClient,
  HttpError,
  isAbortError,
  NetworkError,
  type SessionResponse,
  type SettingsResponse,
  type WallSlotIds,
} from "./client"
import { AppShell, type CameraState } from "./layout/AppShell"
import { RequestFence, startFencedRequest } from "./requestFence"
import "./app.css"

type SessionState =
  | { readonly kind: "loading" }
  | { readonly kind: "anonymous"; readonly notice?: string | undefined }
  | {
      readonly kind: "authenticated"
      readonly session: SessionResponse
      readonly notice?: string | undefined
    }

const SESSION_UNAVAILABLE = "The session service is unavailable."
const SESSION_EXPIRED = "Your session expired. Sign in again."

function sessionErrorMessage(error: unknown, operation: "check" | "login"): string {
  if (error instanceof NetworkError) {
    return SESSION_UNAVAILABLE
  }
  if (error instanceof HttpError) {
    if (operation === "login" && error.status === 401) {
      return "Invalid operator ID or password."
    }
    if (error.status === 401) {
      return SESSION_EXPIRED
    }
    if (error.status === 429) {
      const retryAfter = error.retryAfterSeconds
      return retryAfter === undefined
        ? "Too many sign-in attempts. Try again shortly."
        : `Too many sign-in attempts. Try again in ${Math.max(1, Math.ceil(retryAfter))} seconds.`
    }
    if (error.status >= 500) {
      return "The session service returned an error. Try again shortly."
    }
  }
  return operation === "login"
    ? "Unable to sign in right now. Try again."
    : "Unable to check the current session. Try again."
}

function cameraErrorMessage(error: unknown): string {
  if (error instanceof NetworkError) {
    return "Camera configuration is unavailable. Check the session service and retry."
  }
  if (error instanceof HttpError && error.status === 401) {
    return SESSION_EXPIRED
  }
  return "Camera configuration could not be loaded."
}

function settingsErrorMessage(error: unknown): string {
  if (error instanceof NetworkError) {
    return "Wall placement is unavailable. Check the session service and retry."
  }
  if (error instanceof HttpError && error.status === 401) {
    return SESSION_EXPIRED
  }
  return "Wall placement could not be loaded."
}

function LoadingScreen() {
  return (
    <main className="session-status" aria-busy="true">
      <div className="session-status__panel">
        <p className="session-status__kicker">Live RTSP monitoring</p>
        <h1>God’s Watching</h1>
        <Status tone="neutral">Checking session</Status>
        <p>Verifying the local operator session…</p>
      </div>
    </main>
  )
}

export function App() {
  const [session, setSession] = useState<SessionState>({ kind: "loading" })
  const [cameras, setCameras] = useState<CameraState>({ kind: "loading" })
  const [settings, setSettings] = useState<WallSettingsState>({ kind: "loading" })
  const [savingWallSlots, setSavingWallSlots] = useState(false)
  const sessionFence = useRef(new RequestFence())
  const logoutRequest = useRef<AbortController | null>(null)
  const activityRequest = useRef<AbortController | null>(null)
  const cameraFence = useRef(new RequestFence())
  const wallSlotRequest = useRef<AbortController | null>(null)
  const settingsRequest = useRef<AbortController | null>(null)
  const settingsGeneration = useRef(0)
  const lastActivityAt = useRef(0)

  const checkSession = useCallback(() => {
    setSession({ kind: "loading" })
    void startFencedRequest(sessionFence.current, (signal) => apiClient.getSession(signal), {
      onResponse: (response) => {
        setSession(
          response.authenticated
            ? { kind: "authenticated", session: response }
            : { kind: "anonymous" },
        )
      },
      onError: (error) => {
        setSession({ kind: "anonymous", notice: sessionErrorMessage(error, "check") })
      },
    })
  }, [])

  useEffect(() => {
    const fence = sessionFence.current
    checkSession()
    return () => fence.cancel()
  }, [checkSession])

  const login = useCallback(async (username: string, password: string): Promise<LoginOutcome> => {
    const controller = new AbortController()
    try {
      const response = await apiClient.login(username, password, controller.signal)
      if (!response.authenticated) {
        return { ok: false, message: "Authentication was not accepted. Try again." }
      }
      sessionFence.current.cancel()
      setSession({ kind: "authenticated", session: response })
      return { ok: true }
    } catch (error: unknown) {
      if (isAbortError(error)) {
        return { ok: false, message: "Sign-in was cancelled. Try again." }
      }
      const message = sessionErrorMessage(error, "login")
      if (error instanceof NetworkError) {
        setSession({ kind: "anonymous", notice: message })
      }
      return { ok: false, message }
    }
  }, [])

  const logout = useCallback(async (): Promise<void> => {
    logoutRequest.current?.abort()
    const controller = new AbortController()
    logoutRequest.current = controller
    try {
      await apiClient.logout(controller.signal)
      sessionFence.current.cancel()
      setSession({ kind: "anonymous" })
    } catch (error: unknown) {
      if (isAbortError(error)) {
        return
      }
      if (error instanceof HttpError && error.status === 401) {
        sessionFence.current.cancel()
        setSession({ kind: "anonymous" })
        return
      }
      setSession((current) => {
        if (current.kind !== "authenticated") {
          return current
        }
        return {
          kind: "authenticated",
          session: current.session,
          notice: "Sign out failed. Try again.",
        }
      })
    } finally {
      if (logoutRequest.current === controller) {
        logoutRequest.current = null
      }
    }
  }, [])

  const authenticated = session.kind === "authenticated"

  const onSessionExpired = useCallback((): void => {
    sessionFence.current.cancel()
    logoutRequest.current?.abort()
    logoutRequest.current = null
    activityRequest.current?.abort()
    activityRequest.current = null
    cameraFence.current.cancel()
    settingsRequest.current?.abort()
    settingsRequest.current = null
    wallSlotRequest.current?.abort()
    wallSlotRequest.current = null
    settingsGeneration.current += 1
    setSavingWallSlots(false)
    setSession({ kind: "anonymous", notice: SESSION_EXPIRED })
  }, [])

  const loadCameras = useCallback((): void => {
    if (!authenticated) {
      cameraFence.current.cancel()
      setCameras({ kind: "loading" })
      return
    }
    setCameras({ kind: "loading" })
    void startFencedRequest(cameraFence.current, (signal) => apiClient.listCameras(signal), {
      onResponse: (response) => setCameras({ kind: "ready", cameras: response }),
      onError: (error) => {
        if (error instanceof HttpError && error.status === 401) {
          onSessionExpired()
          return
        }
        setCameras({ kind: "error", message: cameraErrorMessage(error) })
      },
    })
  }, [authenticated, onSessionExpired])

  const refreshCameras = loadCameras

  useEffect(() => {
    if (!authenticated) {
      activityRequest.current?.abort()
      activityRequest.current = null
      lastActivityAt.current = 0
      return
    }

    function recordActivity(): void {
      const now = Date.now()
      if (activityRequest.current !== null || now - lastActivityAt.current < 60_000) {
        return
      }
      lastActivityAt.current = now
      const controller = new AbortController()
      activityRequest.current = controller
      void apiClient
        .activity(controller.signal)
        .then((response) => {
          if (!response.authenticated) {
            onSessionExpired()
            return
          }
          setSession((current) => {
            if (current.kind !== "authenticated") {
              return current
            }
            return { kind: "authenticated", session: response, notice: current.notice }
          })
        })
        .catch((error: unknown) => {
          if (isAbortError(error)) {
            return
          }
          if (error instanceof HttpError && error.status === 401) {
            onSessionExpired()
          }
        })
        .finally(() => {
          if (activityRequest.current === controller) {
            activityRequest.current = null
          }
        })
    }

    window.addEventListener("pointerdown", recordActivity, { passive: true })
    window.addEventListener("keydown", recordActivity)
    return () => {
      window.removeEventListener("pointerdown", recordActivity)
      window.removeEventListener("keydown", recordActivity)
      activityRequest.current?.abort()
      activityRequest.current = null
    }
  }, [authenticated, onSessionExpired])

  const updateWallSlots = useCallback(
    (wallSlotIds: WallSlotIds): void => {
      if (settings.kind !== "ready") {
        return
      }
      const previous = settings
      wallSlotRequest.current?.abort()
      const controller = new AbortController()
      wallSlotRequest.current = controller
      const generation = settingsGeneration.current + 1
      settingsGeneration.current = generation
      setSavingWallSlots(true)
      setSettings({
        kind: "ready",
        settings: { ...settings.settings, wall_slot_ids: wallSlotIds },
      })
      void apiClient
        .patchSettings({ wall_slot_ids: wallSlotIds }, controller.signal)
        .then((response) => {
          if (generation !== settingsGeneration.current || controller.signal.aborted) {
            return
          }
          setSettings({ kind: "ready", settings: response })
        })
        .catch((error: unknown) => {
          if (generation !== settingsGeneration.current || isAbortError(error)) {
            return
          }
          if (error instanceof HttpError && error.status === 401) {
            onSessionExpired()
            return
          }
          setSettings(previous)
        })
        .finally(() => {
          if (generation === settingsGeneration.current && wallSlotRequest.current === controller) {
            wallSlotRequest.current = null
            setSavingWallSlots(false)
          }
        })
    },
    [onSessionExpired, settings],
  )

  useEffect(() => {
    const fence = cameraFence.current
    loadCameras()
    return () => fence.cancel()
  }, [loadCameras])

  const onSettingsSaved = useCallback((response: SettingsResponse): void => {
    settingsGeneration.current += 1
    settingsRequest.current?.abort()
    settingsRequest.current = null
    wallSlotRequest.current?.abort()
    wallSlotRequest.current = null
    setSavingWallSlots(false)
    setSettings({ kind: "ready", settings: response })
  }, [])

  useEffect(() => {
    if (!authenticated) {
      settingsRequest.current?.abort()
      settingsRequest.current = null
      wallSlotRequest.current?.abort()
      wallSlotRequest.current = null
      settingsGeneration.current += 1
      setSavingWallSlots(false)
      setSettings({ kind: "loading" })
      return
    }
    settingsRequest.current?.abort()
    const controller = new AbortController()
    settingsRequest.current = controller
    const generation = settingsGeneration.current + 1
    settingsGeneration.current = generation
    setSettings({ kind: "loading" })
    void apiClient
      .getSettings(controller.signal)
      .then((response) => {
        if (generation !== settingsGeneration.current || controller.signal.aborted) {
          return
        }
        setSettings({ kind: "ready", settings: response })
      })
      .catch((error: unknown) => {
        if (generation !== settingsGeneration.current || isAbortError(error)) {
          return
        }
        if (error instanceof HttpError && error.status === 401) {
          onSessionExpired()
          return
        }
        setSettings({ kind: "error", message: settingsErrorMessage(error) })
      })
      .finally(() => {
        if (settingsRequest.current === controller) {
          settingsRequest.current = null
        }
      })
    return () => {
      controller.abort()
      if (settingsRequest.current === controller) {
        settingsRequest.current = null
      }
    }
  }, [authenticated, onSessionExpired])

  if (session.kind === "loading") {
    return <LoadingScreen />
  }
  if (session.kind === "anonymous") {
    return (
      <AuthScreen
        notice={session.notice}
        onLogin={login}
        onRetry={checkSession}
        unavailable={session.notice === SESSION_UNAVAILABLE}
      />
    )
  }
  return (
    <AppShell
      cameras={cameras}
      client={apiClient}
      onCamerasRefresh={refreshCameras}
      onLogout={logout}
      onSessionExpired={onSessionExpired}
      onSettingsSaved={onSettingsSaved}
      onWallSlotsChange={updateWallSlots}
      savingWallSlots={savingWallSlots}
      sessionNotice={session.notice}
      settings={settings}
    />
  )
}
