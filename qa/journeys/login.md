---
id: login
title: Sign in and protect operator access
setup:
tags: authentication, session
---
## Preconditions
The application is available. The harness supplies the operator credentials directly to the agent.

## Goal
Use sign-in and sign-out to confirm that operator access is accepted only for valid credentials.

## Expected Outcomes
- [E1] An incorrect password is rejected with a visible message and does not create an authenticated session.
- [E2] The supplied valid operator credentials open the authenticated console.
- [E3] Signing out returns to the sign-in screen, and opening a protected page requires signing in again.
