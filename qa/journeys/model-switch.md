---
id: model-switch
title: Change the person search model
setup: login, cameras
tags: models, search
---
## Preconditions
The harness has registered the four fixture cameras and waited for them to stream. More than one prepared CLIP model is available.

## Goal
Change the model used for person search and confirm that the active model and search remain in a safe, usable state.

## Expected Outcomes
- [E1] The Cameras page lists the prepared CLIP models and marks the active model.
- [E2] Choosing another prepared model shows a preflight with a retained crop count and pause estimate, or an actionable reason that blocks the change.
- [E3] If the change is allowed, confirming it shows durable progress that survives a page reload and finishes with the new model active and person search usable.
- [E4] If the change is blocked, the active model and search results remain unchanged.
