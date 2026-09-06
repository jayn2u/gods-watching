# Gods Watching Design System

## 1. Direction and source

This system is extracted from the attached **Gods Watching** reference (`R4` in the project plan). The visual contract is a dense, operational NVR surface: near-black navy depth, thin blue-gray rules, crisp cyan action color, compact geometry, and restrained motion. The memorable material is the panel gradient from a blue-black illuminated top edge into an almost-black base over the page's upper-left cyan glow.

The reference is visual guidance only. Its bundled demo, inline implementation, placeholder video, fake data, and mock callbacks are not reusable product code.

## 2. Color and material tokens

All color usage must resolve through `web/src/styles/tokens.css`.

- Canvas: `page #03070c`, `page-top #050b12`, `page-glow #12324d`.
- Content: `text #eaf5ff`, `muted #8daabd`, `footer #6f899b`.
- Action: `accent #9bdcff`, `accent-ink #03101a`.
- Structure: `line #21384b`, `input-border #31516a`, `secondary-border #44637a`.
- Panels: `rgba(13,29,44,.96)` to `rgba(6,14,23,.98)`; inputs `#040b12`; secondary controls/cards `#08131e`.
- Status: success `#a6c6d8`, danger `#ff9ca6` on `#24131a`, inactive `#526e82`.
- Overlay: `rgba(3,7,12,.74)`.

Panels use a one-pixel rule, the two-stop reference gradient, and a subtle black shadow. Radius is `2px` for buttons, fields, panels, and dialogs. No pill-shaped containers, ornamental gradients, glass blur, or oversized shadows.

## 3. Typography

Self-hosted WOFF2 files are required so the interface works offline.

- Body and controls: **DM Sans**, 400/500/600.
- Brand, headings, labels, and statuses: **Space Grotesk**, 500/600.
- Body: 14px/1.5; compact text: 12px/1.4; micro labels: 11px/1.25.
- Showcase title: responsive 28–36px/1.05. Product screen headings remain compact and follow the reference.
- Labels may use uppercase with `.08em` tracking; sentence content stays mixed case.

## 4. Geometry and responsive layout

- Product header: exactly `54px` high with `0 18px` padding.
- Desktop live wall: `232px minmax(0, 1fr) 268px` columns.
- Wall: 2×2 with exactly `10px` gap.
- Spacing scale: 4, 8, 10, 12, 16, 20, 24, 32, 48px.
- At 768–1199px the product shell stacks the right rail below the main wall; camera navigation becomes a horizontal/contained region.
- At 375–767px every operation remains in DOM order, controls become full-width where needed, dialog margins are 16px, and no horizontal scrolling is permitted.
- The isolated primitive showcase uses one column at 375px, two at 768px, and a three-column specimen grid at 1280/1440px.

## 5. Reusable primitives and states

- `Button`: primary, secondary, danger; normal, hover, active, focus-visible, disabled, loading. Loading keeps its label, exposes `aria-busy`, and cannot be activated.
- `Input`: visible label, optional hint, invalid description; normal, hover, focus-visible, disabled, invalid. Invalid uses `aria-invalid` and `aria-describedby`.
- `Panel`: semantic titled region with optional eyebrow and actions; default and quiet variants.
- `Status`: neutral, live/success, warning, offline/error; calm text plus a dot. Polite changes use `role=status`; errors use `role=alert`.
- `Dialog`: native modal semantics, title/description association, initial focus, Tab/Shift+Tab containment, Escape close, backdrop close, and focus return to the trigger. Destructive actions use the danger button.

Focus uses a two-pixel cyan outline with a two-pixel dark offset. Disabled controls remain legible at reduced opacity and use the native `disabled` contract. Error meaning is never communicated by color alone.

## 6. Motion

Interactive color, border, opacity, and transform transitions use `120ms` ease-out. Buttons translate down by one pixel while pressed. Dialog entry uses opacity plus a four-pixel vertical transform over `160ms`. Under `prefers-reduced-motion: reduce`, animations and transitions are removed and scrolling is immediate.

## 7. Accessibility constraints

- Minimum target size is 36×36px for compact operational controls and 44px on narrow screens.
- Text and controls maintain WCAG AA contrast on their declared surfaces.
- Landmarks, headings, labels, error messages, and status announcements are explicit.
- Keyboard order follows the visual order. No positive `tabindex` is allowed.
- Dialog background content is blocked by the native modal layer while open.
- Loading feedback preserves the button's accessible name and sets `aria-busy=true`.

Primary review personas are a keyboard-only control-room operator, a low-vision operator using 200% zoom, and a motion-sensitive operator. The showcase must demonstrate their required states before product screens are built.

## 8. Scope decisions, exclusions, and debt

Approved deviations from R4:

- Exclude all REC/record controls and recording state.
- Exclude timeline, seeking, playback, and “open in wall” playback actions.
- Exclude subnet scanner/discovery controls.
- Exclude person attribute filters.
- Add English natural-language person search.
- Add global retention settings in Cameras.

No demo camera/person data, video placeholders, or fake backend success behavior may enter product screens. Task 5 only establishes primitives and the isolated showcase; login, wall, search, and camera product screens are intentionally owned by later tasks.

Accepted debt: the showcase validates primitive behavior and responsive composition, while full product-screen pixel comparison and production Lighthouse audits occur after tasks 15–17 render the complete routes. Local font files are the only reference-derived assets used here.
