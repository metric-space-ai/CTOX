# Standalone To Business OS Porting

Use this when a user starts with a standalone vanilla app and asks an agent to
port it into CTOX Business OS.

## Portable Shape

Write standalone apps so the Business OS port is mostly wiring, not a rewrite:

```js
export async function mount(ctx) {
  const root = ctx.host;
  const records = ctx.db.collection('your_module_records');
  const cleanup = [];
  // render, subscribe, dispatch commands
  return () => cleanup.forEach((fn) => fn());
}
```

The app may run standalone by passing a mock `ctx`, but production Business OS
gets the real `ctx` from the shell.

Required `ctx` boundary:

- `ctx.host`: DOM element owned by the shell.
- `ctx.db.collection(name)`: declared module collection handle.
- `ctx.commandBus.dispatch(command)`: automation/CTOX task dispatch.
- `ctx.preferences.theme`: current `dark` or `light` mode.
- `ctx.preferences.branding`: active workspace token payload when available.

## Standalone Rules

- Use browser ESM and relative imports only.
- Keep persistence calls behind `ctx.db.collection(...)`.
- Keep automation behind `ctx.commandBus.dispatch(...)`.
- Use Business OS tokens in CSS even in standalone mode.
- Load `assets/standalone/business-os-tokens.css` in standalone previews to
  mimic the default shell tokens.
- Use `assets/standalone/mock-business-os-context.mjs` for local demos and
  tests, then remove the mock from the Business OS runtime bundle.

## Porting Steps

1. Start from the pre-materialized App Starter V2 files: `module.json`,
   `collections.schema.json`, `schema.js`, `index.html`, `index.css`,
   `index.js`, `icon.svg`, and focused tests. Keep their valid Shell-V2 wiring,
   then replace their visible placeholder UI, sample records, and generic CRUD
   semantics completely.
2. Move the standalone app's root render into `mount(ctx)`.
3. Replace direct storage, localStorage, IndexedDB, REST, or in-memory stores
   with `ctx.db.collection('<module_scoped_collection>')`.
4. Replace automation/follow-up calls with `ctx.commandBus.dispatch(...)`.
5. Add record right-click annotations to every row/card/tree node.
6. Set root `launch_kind` to `"desktop-app"`, write the canonical root
   `presentation` object, and retain `layout.shell: "windowed"` only as the
   compatibility hint.
7. Validate with `ctox business-os app validate <module-id> --installed` or
   `--source`, then smoke in the real shell.

## Import Fidelity Gate

Before implementation, render the immutable source entry point at one desktop
and one narrow viewport. Record:

- recognizable composition, typography, controls, labels, and content density;
- pointer, keyboard, canvas/WebGL, animation, audio, game, import/export, and
  full-screen behavior that exists in the source;
- state held in localStorage, IndexedDB, memory, workers, or external APIs;
- embedded data URLs, shaders, workers, binary assets, and third-party runtime
  dependencies that must be extracted or adapted.

Port every inventory item or record a concrete incompatibility and fail the
job. Compare source and mounted screenshots at matching viewports and exercise
the same primary interactions. Do not replace a distinctive source with a
generic list/detail dashboard, decorative mock, or static canvas. Large
single-file apps should extract embedded scripts, workers, assets, and shaders
into maintainable module files while preserving behavior.

## What Not To Port

- Package manager setup, bundlers, dev servers, framework bootstraps.
- Full HTML documents with `<html>`, `<head>`, `<body>`, scripts, or styles.
- App-owned auth, HTTP APIs, database sync, or server storage.
- Standalone mock data as production persistence.
- CSS root palettes or forced `color-scheme`.
- Starter labels, sample records, placeholder panels, or controls that are not
  part of the imported application.

## Acceptance Proof

Before claiming the port is done:

- Standalone mock still mounts for local inspection.
- Business OS mount works with shell-provided `ctx`.
- Records persist through declared collections.
- Automation commands persist through `business_commands`.
- Visual proof covers light, dark, and one custom-brand fixture.
- Source and mounted screenshots are compared at matching desktop and narrow
  viewports, and every source interaction in the inventory has passing proof.
