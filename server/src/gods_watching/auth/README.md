# Authentication service boundary

`AuthService` owns the database transaction for each operation. Construct it with the existing `Database` transaction provider, a `SqlAlchemyCredentialStore`, and a `SessionRevocationHook` implementation when 12c wires the WHEP manager:

```python
auth = AuthService(
    transactions=database,
    revocation_hook=whep_manager.close_session,
)
```

`AuthService` accepts the configured operator identifier as `operator_username` (default `admin`, supplied in production from `GW_OPERATOR_USERNAME`). `LoginAttempt` accepts the identifier, the password, and request peer metadata. The identifier is compared case-insensitively after stripping surrounding whitespace, in constant time, and a mismatch is indistinguishable from a wrong password.

`sync_password()` makes the configured password authoritative at process start: it creates the credential when the singleton row is absent and replaces it — revoking every session — when the stored hash no longer matches. Use `initialize_password()` instead when the stored credential must win.

 `AuthService.login()` canonicalizes the address, applies the five failures per 60 seconds throttle, verifies the Argon2id hash, and returns a `LoginAccepted` containing a `SessionToken`. Only `SessionToken.raw` is suitable for setting the HttpOnly cookie; its `repr` and `str` are redacted. The database receives only the SHA-256 digest in `sessions.token_hash`.

`AuthService.authenticate(token, user_action=False)` checks both expiry limits. Set `user_action=True` only for search, camera/settings mutations, or the CSRF-protected activity endpoint. Passive polling, WHEP keepalives, and live viewing must leave it false. The HTTP layer should rate-limit activity requests to one refresh per minute after real pointer or keyboard activity.

`logout()` revokes only the supplied session. `replace_password()` and `replace_password_file()` replace the credential and revoke every session in one database transaction. A same-password replacement returns `changed=False` and emits no revocation. `cleanup_loop()` performs expiry cleanup every five seconds. Every committed revocation is awaited through `SessionRevocationHook` after transaction commit, so 12c can close all WHEP resources owned by the session.

The credential adapter expects the migration-owned singleton table below. Migration `0002_operator_credentials` creates this exact transactional relation after the Task 4 storage foundation:

```sql
CREATE TABLE operator_credentials (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    password_hash text NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
```

The service does not claim HTTP, CLI, or real media integration. Origin helpers (`require_same_origin`) must protect state-changing routes, and `canonical_client_ip` must receive the direct peer plus the configured trusted gateway before consuming forwarded headers.
