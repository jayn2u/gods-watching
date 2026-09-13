import { type FormEvent, useState } from "react"
import { Button } from "../../components/Button"
import { Input } from "../../components/Input"
import { Panel } from "../../components/Panel"
import "./auth.css"

export type LoginOutcome = { readonly ok: true } | { readonly ok: false; readonly message: string }

type AuthScreenProps = {
  readonly onLogin: (password: string) => Promise<LoginOutcome>
  readonly notice?: string | undefined
  readonly onRetry?: () => void
  readonly unavailable?: boolean
}

export function AuthScreen({ onLogin, notice, onRetry, unavailable = false }: AuthScreenProps) {
  const [operatorId, setOperatorId] = useState("")
  const [password, setPassword] = useState("")
  const [formError, setFormError] = useState<string | undefined>(undefined)
  const [submitting, setSubmitting] = useState(false)

  async function submit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault()
    if (unavailable) {
      onRetry?.()
      return
    }
    if (operatorId.trim() === "") {
      setFormError("Enter an operator ID.")
      return
    }
    if (password.length < 12) {
      setFormError("Password must be at least 12 characters.")
      return
    }
    setFormError(undefined)
    setSubmitting(true)
    try {
      const outcome = await onLogin(password)
      if (!outcome.ok) {
        setFormError(outcome.message)
        return
      }
      setPassword("")
    } finally {
      setSubmitting(false)
    }
  }

  const error = formError ?? notice

  return (
    <main className="auth-screen">
      <div className="auth-screen__content">
        <div className="auth-screen__intro">
          <p className="auth-screen__kicker">Live RTSP monitoring</p>
          <h1>God’s Watching</h1>
        </div>
        <Panel title="Sign in to continue" variant="default">
          <form className="auth-form" noValidate onSubmit={(event) => void submit(event)}>
            <p className="auth-form__description">
              The console is locked until an operator authenticates. Every wall view is attributed
              to your operator session.
            </p>
            <Input
              autoComplete="username"
              label="Operator ID"
              onChange={(event) => setOperatorId(event.target.value)}
              placeholder="operator"
              value={operatorId}
            />
            <Input
              autoComplete="current-password"
              error={formError?.startsWith("Password") ? formError : undefined}
              label="Password"
              onChange={(event) => setPassword(event.target.value)}
              placeholder="Enter your operator password"
              type="password"
              value={password}
            />
            {error === undefined ? null : (
              <p className="auth-form__error" role="alert">
                {error}
              </p>
            )}
            <Button loading={submitting} type="submit" variant="primary">
              {unavailable ? "Retry" : "Sign in"}
            </Button>
            <p className="auth-form__hint">Sessions expire after 30 minutes idle.</p>
          </form>
        </Panel>
        <p className="auth-screen__footer">
          Research-only local console · authenticated live monitoring
        </p>
      </div>
    </main>
  )
}
