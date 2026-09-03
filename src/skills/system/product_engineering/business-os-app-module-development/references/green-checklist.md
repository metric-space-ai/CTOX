# Green Checklist

Use this before claiming a Business OS app is done.

- Every list/selection column follows the Canonical Column Grammar from
  `design-guide.md`: header (kicker+title left, action icons top-right) →
  collapsed filter tray (dropdowns, ✓-chips, reset, active-dot) → counted
  view-switcher band (`Name (n)`, zeros included) → recessed element well
  under a divider → one-line footer.
- Every element-listing view (left column AND main view) has the cards ↔
  compact-list toggle; the toggle sits IN the filterbar row between search and
  the filter-tray toggle, using only the canonical glyphs (stacked rects /
  three lines); shards are pure selectors (no inline expansion).
- Every view band offers at least two real views; a band with a single counted
  tab is a stray chip — a lone count belongs in the pane footer
  (module_static_check enforces this).
- Element actions are collected top-right icons (pencil/trash/✓/✗/▶); no
  text buttons in the content flow; no manual refresh button; no standing
  status badges; sync notices float as toasts.
- Module CSS AND module markup are fetched with the module's own JS
  cache-buster and the module version is three-part semver.
- Any sandboxed `srcdoc` iframe assigns only on content change and re-applies
  once via a load watchdog (first assignment can be swallowed at mount).
- Interaction proof, not just render proof: scroll a filled list well past the
  fold, select a lower element — the scroll offset stays and the selection
  applies (selection = in-place class flip, never a list rebuild). Repeat the
  interaction after a data refresh re-render.

- The target directory is correct for runtime or source mode.
- Three relevant shipped Business OS apps were chosen and inspected.
- The app is vanilla HTML/CSS/browser ESM with no build step.
- Runtime `module.json` sets `"icon"` to `icon.svg` or `icon.png` and the
  module directory contains that local file. PNG icons are square, 60–1024
  px, and no larger than 512 KiB.
- Runtime `module.json` sets root `launch_kind` to `desktop-app`, writes the
  canonical `presentation` object (minimum 640×480), and keeps
  `layout.shell: windowed` only as a compatibility hint.
- The app is usable at its minimum width and responds to its window container,
  not only to the browser viewport.
- `IMPECCABLE_PREFLIGHT` passed with Product register plus the CTOX root
  `PRODUCT.md`, `DESIGN.md`, and `.impeccable/design.json` context.
- Routine controls remain compact and neutral. At most one real, domain-named
  AI/automation action is visually dominant per visible work surface.
- Real mouse dragging resizes the floating window and every visible
  `.ctox-column-resizer`; no direct style mutation is accepted as proof.
- At 360px the shell uses a mobile app sheet with Start, version/status,
  Source/Versions, close/back, chat, and task switching reachable.
- Each two-/three-pane layout preserves a visible stack/tab/drawer and return
  path for panes that cannot remain side by side.
- The app has at most one compact app-level command/header row. It does not
  repeat shell-owned app identity/version/source chrome or stack hero, metrics,
  date-strip, and filter headers before the work surface.
- `index.js` exports `mount(ctx)`.
- `mount(ctx)` loads `index.html` into `ctx.host` or renders an equivalent
  primary UI into `ctx.host`; it does not assume the shell preloaded the
  fragment.
- `index.css` is loaded by the module or otherwise available through the app contract.
- App records use declared module collections through
  `ctx.db.collection('<declared-collection-name>')`.
- Every declared app-owned collection has reviewed `data.read` and every
  required `data.write` grant for the intended tenant actor/role.
- The browser smoke runs as that same tenant actor/role; a local synthetic
  admin smoke is not accepted as a substitute.
- No legacy DB fallback exists: no `ctx.db[name]`, `ctx.db.collections`, direct
  `ctx.db.<collection>` property access, cached DB facade, raw IndexedDB, HTTP,
  or app-owned sync path.
- App code does not call `ctx.db.registerSchemas`; schema registration comes
  from module metadata and the Business OS shell/native peer.
- Runtime app collection names are scoped to the module id.
- `schema.js`, `collections.schema.json`, and record helper outputs agree on
  collection names, schema versions, required fields, and property types.
- Every collection version above 0 has all intermediate JSON
  `migration_strategies`; persisted schemas were never edited in place.
- Automation uses `ctx.commandBus.dispatch(...)`.
- Chat/AI actions use `business_os.chat.task` with `payload.record_snapshot`; real ticket lifecycle actions use `ctox.ticket.*`.
- Automation results that return `task_id` or `command_id` are visible and
  clickable from the originating record, opening the CTOX Flow/Queue focus via
  `ctox.businessOs.focusTask` and `#ctox?...`.
- The app does not include a generic "Report to CTOX" / "An CTOX melden" /
  queue / AI / command-bus button unless that automation was requested or is a
  real workflow with a trackable result.
- The UI has no decorative panes or dead controls.
- Optional/secondary controls, filters, forms, and reference panels are hidden by
  default and revealed on demand (collapsible pane, `<details>`, `[hidden]`, or
  `⋯` menu); the default view is the minimal common case.
- Any detail/inspector pane that is hidden by default auto-reveals when a record
  is selected (`visible = hasSelection && !userCollapsed`); it never leaves a
  selection showing nothing.
- Sections/cards render only when they have data; there are no empty "not set"
  placeholder cards.
- Any left/right column inside the app contains real workflow content and is not
  an empty copy of shell context/topics.
- `index.css` uses Business OS theme tokens for surfaces, borders, and text,
  does not force `color-scheme`, and does not define root Business OS tokens.
- No color-bearing CSS declaration hard-codes hex/rgb theme colors; everything
  resolves through tokens or `color-mix(...)` over tokens.
- The UI is built from `shared/base.css` kit classes (pane header with
  kicker/title and `.ctox-pane-actions`, kit controls, `.ctox-table`,
  `.ctox-fields`, `.ctox-badge`, `.ctox-modal`, `.ctox-empty`) instead of
  app-local rebuilds; header primary actions are `.ctox-pane-icon` icon
  buttons with `aria-label`/`title` and `ctx.getActionIcon` glyphs.
- The app was visually checked in light and dark theme at desktop and narrow
  viewport sizes; text, buttons, cards, dialogs, and bottom actions remain
  readable and do not overlap.
- Long German, English, and unbreakable technical app names remain within a
  fixed two-line desktop icon cell and keep the full accessible name/tooltip.
- The app was visually checked against one custom-brand fixture, proving it
  consumes shell tokens instead of hard-coded root palettes.
- Every record row/card/tree node exposes `data-context-record-id`/`-record-type`/`-label` (or at least a `data-*-id`) so a right-click hands the agent the record.
- Browser proof right-clicks a record and confirms the "Chat to CTOX" popover opens showing that record (its label/id), proving the agent receives the click target.
- The empty state lets the user create at least one primary business record.
- Primary Create/New/Add controls are clicked in the real Business OS shell and
  reveal a usable dialog, form, or save flow.
- Hidden modals, drawers, and overlays really stop intercepting clicks when hidden.
- Core workflows implemented in the UI actually work.
- For booking, parking, scheduling, shift, availability, or date/slot domains,
  the common claim/release/book path works in one click from the visible
  calendar/date/slot view.
- Resource/date apps enforce domain conflicts in that one-click path, for
  example one vehicle/person/asset cannot be booked into two overlapping slots.
- Any changed local ESM helper export is loaded through a fresh helper URL
  (for example a versioned helper filename), and a browser reload proves the
  module does not fail on stale helper imports.
- Tests cover record helper behavior and automation payloads.
- `ctox business-os app validate <module-id> --installed` or `--source` passes.
- `ctox business-os app smoke <module-id> --installed` passes.
- Imported modules remain absent from the live catalog until static validation,
  installed validation, browser smoke, and catalog refresh all pass. Failure,
  timeout, cancellation, or provider outage removes any temporary smoke
  visibility and cannot return `live=true`.
- The completed command result contains `live=true`, source revision/snapshot
  evidence, `asset_revision`, catalog revision, and catalog fingerprint.
- If the app began standalone, the port removed app-owned persistence/sync and
  production code now uses shell-provided `ctx.db` and `ctx.commandBus`.
- For an imported source, the immutable entry point was rendered first and its
  visual/interaction inventory is attached to the result evidence.
- Source and mounted screenshots were compared at matching desktop and narrow
  viewports; recognizable composition, typography, density, motion, canvas or
  WebGL output, audio, and primary controls remain equivalent.
- Every source workflow has interaction proof. No starter sample record,
  generic CRUD label, placeholder panel, dead control, static imitation, or
  undocumented partial implementation remains.
- No service lifecycle command was used during the app build:
  no `ctox stop/start/upgrade`, `launchctl`, `systemctl`, bootout, disable, or
  daemon restart.
