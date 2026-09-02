# Business OS Shell v2 Host Contract

Use this contract for every Business OS app except the reserved `desktop`
surface. Shell v2 owns the host; an app owns only its workspace in `ctx.host`.

## Manifest and ownership

- Set root `launch_kind` to `desktop-app` and provide canonical root
  `presentation` with `window`, `maximized`, and `focus` modes.
- Keep `layout.shell: "windowed"`, set `layout.shell_contract: "v2"`, and set
  `layout.shell_geometry_contract: "business-os-v2-global-1"`.
- The shell owns the frame, icon, title, close, drag/resize, lifecycle controls,
  taskbar, account, chat, drawers, focus/maximize, and the mobile sheet.
- The app renders only its responsive work surface into `ctx.host`; it must not
  duplicate shell chrome or depend on undocumented shell globals.
- Use only shell-provided `ctx.db`, `ctx.sync`, `ctx.commandBus`, lifecycle,
  drawer, icon, and navigation facades. Never add HTTP, localStorage,
  IndexedDB, app-owned RxDB, or shell-global fallbacks.

## Data permissions

- Every collection used by `ctx.db.collection(name)` must be declared in
  `module.json`, `collections.schema.json`, and `schema.js` with the same name
  and schema version.
- Declaring a collection does not grant access. Before an app may be live, the
  intended actor or role must have reviewed `data.read` and every required
  `data.write` grant for each app-owned collection.
- Validate the permission facade with the same tenant actor/role used by the
  acceptance browser. A local admin smoke is not evidence for a managed web
  actor when their session/governance contexts differ.
- A permission failure during `mount(ctx)` is a release blocker. Do not catch
  it and render a fake success state.

## Import publication gate

- Imported app files are build artifacts until the full gate passes. Directory
  presence, a parseable manifest, HTTP 200 assets, or a local render must never
  be reported as `live`.
- The catalog must keep an importer-generated module hidden before its bounded
  browser-smoke gate begins and must hide it again immediately if smoke fails,
  times out, the worker aborts, or the provider becomes unavailable.
- Only the core may record passed import-smoke evidence and publish the module.
  App code and the coding agent must not fabricate or edit gate evidence.
- A completed result needs static validation, installed-module validation,
  browser smoke, catalog refresh, `live=true`, asset revision, catalog revision,
  fingerprint, and source revision/snapshot evidence.

## Interaction and release proof

- Blank shell header space remains draggable; app controls do not cover it.
- The shell provides exactly one frame and close control. The app does not add
  outer window borders, resize rails, or another close button.
- Verify 640×480 and the 360 px mobile sheet in light, dark, and one custom
  brand. Test close/reopen, drag, resize, Source/Versions/Coding, primary
  workflows, and reload persistence.
- HTML, CSS, JavaScript, icons, manifest, and registry/catalog data must come
  from one immutable generation. A mutable path with a new query string is not
  an atomic release.
- Final proof runs in the real tenant Shell v2 host with the intended actor and
  includes console/network inspection. Standalone and localhost runs are
  supporting evidence only.

Do not call the app complete or production-ready while any item above is
missing or red.
