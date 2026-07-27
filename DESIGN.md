---
name: CTOX Business OS
description: A compact operational instrument for governed business work.
colors:
  light-workspace: "#f1f3f4"
  light-surface: "#f8f9fa"
  light-surface-subtle: "#e7eaec"
  light-line: "#c6ccd0"
  light-text: "#20262b"
  light-text-strong: "#11171c"
  light-muted: "#65717b"
  light-accent: "#237c74"
  light-accent-soft: "#d8ebe8"
  light-accent-foreground: "#f6fbfa"
  dark-workspace: "#15191d"
  dark-surface: "#1b2025"
  dark-surface-subtle: "#22282e"
  dark-line: "#394149"
  dark-text: "#d9dfe4"
  dark-text-strong: "#eff2f4"
  dark-muted: "#98a3ad"
  dark-accent: "#62b8ac"
  dark-accent-soft: "#23443f"
  dark-accent-foreground: "#0d1b18"
  danger: "#b64a43"
  warning: "#9b6b12"
  success: "#237a4b"
typography:
  headline:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif'
    fontSize: "18px"
    fontWeight: 650
    lineHeight: "1.25"
    letterSpacing: "-0.01em"
  title:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif'
    fontSize: "14px"
    fontWeight: 650
    lineHeight: "1.3"
    letterSpacing: "normal"
  body:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif'
    fontSize: "13px"
    fontWeight: 400
    lineHeight: "1.45"
    letterSpacing: "normal"
  label:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif'
    fontSize: "12px"
    fontWeight: 600
    lineHeight: "1.2"
    letterSpacing: "0.01em"
  mono:
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'
    fontSize: "12px"
    fontWeight: 450
    lineHeight: "1.4"
    letterSpacing: "normal"
rounded:
  control: "3px"
  surface: "4px"
  overlay: "6px"
  pill: "999px"
spacing:
  hairline: "1px"
  xxs: "2px"
  xs: "4px"
  sm: "6px"
  md: "8px"
  lg: "12px"
  xl: "16px"
  xxl: "24px"
components:
  button-utility:
    backgroundColor: "{colors.light-surface}"
    textColor: "{colors.light-text}"
    typography: "{typography.label}"
    rounded: "{rounded.control}"
    height: "28px"
    padding: "0 8px"
  button-run:
    backgroundColor: "{colors.light-accent}"
    textColor: "{colors.light-accent-foreground}"
    typography: "{typography.label}"
    rounded: "{rounded.control}"
    height: "34px"
    padding: "0 14px"
  input-compact:
    backgroundColor: "{colors.light-surface-subtle}"
    textColor: "{colors.light-text}"
    typography: "{typography.body}"
    rounded: "{rounded.control}"
    height: "30px"
    padding: "0 8px"
  pane:
    backgroundColor: "{colors.light-surface}"
    textColor: "{colors.light-text}"
    rounded: "{rounded.surface}"
    padding: "0"
  status-chip:
    backgroundColor: "{colors.light-surface-subtle}"
    textColor: "{colors.light-muted}"
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    height: "20px"
    padding: "0 7px"
---

# Design System: CTOX Business OS

## Overview

**Creative North Star: "The Operational Instrument"**

CTOX Business OS should feel like a precise instrument built for repeated
professional work. Its spatial model is stable, controls are compact, and the
interface recedes while a person operates records, queues, documents,
approvals, and automations. Ableton Live is a reference for density, immediacy,
and functional color, not a skin to copy.

Routine controls remain visually quiet. Each app may elevate one signature
automation through a clearly named Run Control, but it does not redesign
search, filter, sort, import, edit, save, tables, context menus, or status.
Windowed, maximized, light, and dark modes use the same information
architecture and component vocabulary.

The system explicitly rejects glossy dark SaaS dashboards, landing-page
composition inside authenticated apps, generic AI interfaces that hide real
work, and app-specific reinventions of common controls.

**Key Characteristics:**

- compact operational density
- flat connected work surfaces
- restrained color with one functional accent
- stable pane and window geometry
- explicit selection, permission, sync, and run state
- keyboard-first interaction with visible focus

**The One Surface Rule.** An app opens directly into its real workbench. It
never starts with a hero, decorative metric mosaic, or marketing explanation.

**The Container Rule.** Responsive behavior follows the app container, not the
browser viewport. Compact, standard, and wide presentations preserve the same
task and selection.

**The Signature Automation Rule.** "Hero" describes functional emphasis, not
landing-page composition. A visible work surface may contain exactly one
prominent, domain-named AI/automation control when it dispatches a real typed
command and exposes approval, queue, progress, result, failure, abort, and retry
state. Routine create, import, filter, sort, edit, save, export, navigation, and
window controls stay compact and neutral.

**The Responsive Shell Rule.** Desktop windows remain freely resizable down to
640×480. At a shell width of 600px or below, the window becomes a mobile app
sheet rather than pretending to remain a 640px floating window. The supported
mobile floor is 360px. Start, app identity, version/status, source/version
actions, close/back, chat, and task switching remain reachable.

## Colors

The palette uses lightly tinted neutral surfaces and one muted teal accent.
Light and dark are equivalent operational renderings, not separate art
directions.

### Primary

- **Operational Teal:** the active accent for focus, selected controls, links,
  and the single signature Run Control.
- **Quiet Teal Field:** the low-emphasis accent surface for selected rows,
  active filters, and permission-aware context.

### Neutral

- **Paper Graphite:** the light workspace and surface family. It is cool enough
  to separate work regions without looking clinical.
- **Instrument Graphite:** the dark workspace and surface family. It uses
  visible tonal steps and crisp separators instead of glow or glass.
- **Working Ink:** normal and strong text roles.
- **Operational Muted:** metadata, secondary labels, timestamps, and inactive
  controls.
- **Mechanical Line:** one-pixel pane, row, and control separators.

### Semantic

- **Danger:** destructive action, failed run, rejected write, or invalid state.
- **Warning:** approval needed, stale data, or attention without failure.
- **Success:** completed run, verified data, or successful persistence.

**The Functional Color Rule.** Accent color communicates action, selection, or
state and occupies less than ten percent of a routine work surface. It is never
ambient decoration.

**The Theme Parity Rule.** A semantic role has the same meaning in both themes.
Theme switching never moves controls, changes available actions, or hides
state.

## Typography

**Display Font:** system sans-serif stack

**Body Font:** system sans-serif stack

**Label/Mono Font:** system monospace stack for identifiers, code, timestamps,
and technical values only

**Character:** Native, quiet, and highly legible. Hierarchy comes from weight,
alignment, and restrained scale rather than oversized headings or decorative
display type.

### Hierarchy

- **Headline** (650, 18px, 1.25): rare app- or major-surface heading.
- **Title** (650, 14px, 1.3): pane headings, inspector titles, and selected
  record titles.
- **Body** (400, 13px, 1.45): operational copy, field values, and supporting
  detail. Prose is capped at 65 to 75 characters per line.
- **Label** (600, 12px, 1.2): buttons, tabs, filters, column headings, and
  compact status.
- **Mono** (450, 12px, 1.4): IDs, hashes, code, machine state, and timestamps.

**The No Display Voice Rule.** UI labels, buttons, data, and routine headings
never use a display font or oversized marketing scale.

**The Expansion Rule.** German and English labels must fit or wrap without
hiding the operative noun, state, or action.

## Elevation

The workspace is flat by default. Depth is communicated through tonal layering
and one-pixel boundaries. Shadows are reserved for objects that genuinely sit
above work: movable windows, popovers, context menus, and blocking overlays.

### Shadow Vocabulary

- **Window Lift:** a broad, low-opacity shadow that separates a movable window
  from the desktop without producing a glow.
- **Popover Lift:** a tighter shadow for context menus, tooltips, and transient
  overlays.
- **No Workspace Shadow:** panes, tables, cards, toolbars, inputs, and selected
  rows remain shadowless.

**The Flat-by-Default Rule.** If a stationary pane needs a shadow to be
understood, its separator and surface hierarchy are wrong.

**The No Glass Rule.** Backdrop blur, translucent glass cards, radial gradients,
and ambient glow are prohibited as default shell or app styling.

## Components

Common components are platform primitives. Apps compose them and add only
domain-specific visualization or their signature automation.

### Buttons

- **Shape:** precise corners with a small control radius (3px).
- **Utility:** compact, neutral, 28px high, and placed next to the object or
  state it affects.
- **Run Control:** 34px high, solid accent, named with a domain verb, and limited
  to one dominant action per visible work surface.
- **Hover / Focus:** a tonal shift on hover and a clearly visible focus ring.
  Disabled, loading, error, and active states are mandatory.
- **Ghost:** borderless only when hierarchy remains obvious; hover receives a
  neutral surface, not an accent glow.

### Chips

- **Style:** pill geometry is limited to filters, tags, and compact status.
- **State:** unselected chips remain neutral; selected chips use the quiet
  accent field and keep readable text contrast.

### Cards / Containers

- **Corner Style:** restrained surface radius (4px).
- **Background:** connected surfaces use the surface roles and one-pixel lines.
- **Shadow Strategy:** none inside a workspace.
- **Border:** mechanical one-pixel separator.
- **Internal Padding:** 8px to 12px for dense work, 16px only for focused forms
  or empty states.
- **Use:** cards are allowed for true repeated domain items or inspector
  sections. They never wrap an entire page or nest decoratively.

### Inputs / Fields

- **Style:** compact 30px controls, subtle surface, one-pixel border, 3px radius.
- **Focus:** visible accent border plus focus ring.
- **Error / Disabled:** semantic state remains readable in both themes and does
  not rely on color alone.

### Navigation

- **Style:** compact shell chrome, stable module placement, and restrained
  selected state.
- **Responsive:** navigation becomes a shared drawer below the compact boundary;
  apps do not invent mobile navigation.
- **State:** active module, active record, and pending work remain distinguishable.

### Window Header

- **Desktop:** one 32px mechanical row containing app icon/name, compact
  version and lifecycle state, Source and Versions actions, then
  Minimize/Maximize/Close.
- **Narrow window:** labels yield before functionality; icon-only actions keep
  tooltips and accessible names.
- **Mobile sheet:** a 44px touch row remains below the shell topbar. Resize
  handles disappear because the presentation mode changed, not because the app
  lost responsiveness.

### Responsive Panes

- **Wide:** two or three real workflow panes may remain simultaneously visible
  and use shell-owned `.ctox-column-resizer` handles.
- **Compact:** panes stack or become a shared tab/drawer/master-detail flow.
  Hiding a pane is permitted only when a visible control and deterministic
  return path can restore its work.
- **Mobile:** preserve the main task, selection, drafts, permissions, run state,
  and access to secondary panes. Root overflow without a usable pane path is a
  failure.

### Data Workbench

Tables, lists, timelines, trees, editors, and inspectors share row rhythm,
selection treatment, scroll ownership, loading, empty, error, and permission
states. Filters stay next to the data they affect. Bulk actions appear next to
selection state.

### Context Menu

The shell renders and positions the menu. Apps register semantic targets and
optional domain actions. Pointer, keyboard, record, field, selection,
permission, and delegation use one contract.

### Run Control

The signature component progresses through ready, approval needed, queued,
running, completed, failed, and retryable states. It exposes command and run
identity, progress, abort or retry, and result access. It is never a decorative
call-to-action.

### Loading

The full-app loading shell is automatically derived from the real static
`index.html` and `index.css`. In-pane skeletons represent data that is still
loading after mount and must never replace the app-level derivation.

## Do's and Don'ts

### Do:

- **Do** build routine UI from the existing canonical `.ctox-*` kit names.
- **Do** use one-pixel separators, 3px control radii, 4px surface radii, and
  connected panes.
- **Do** keep import, search, filter, sort, edit, save, tables, dialogs, context
  menus, and status consistent across apps.
- **Do** expose real records, queues, documents, timelines, or automations on
  the first screen.
- **Do** reserve the solid accent for focus, selection, and one signature Run
  Control.
- **Do** preserve selected record, scroll position, draft state, permissions,
  and run evidence across window state changes.
- **Do** support keyboard operation, visible focus, Reduced Motion, and WCAG
  2.2 AA contrast.
- **Do** test light, dark, German, English, compact, standard, and wide states.
- **Do** test real mouse resize, pane resize, 360px mobile sheet, coarse-pointer
  hit areas, long app names, and chat/taskbar coexistence.

### Don't:

- **Don't** create glossy dark SaaS dashboards with radial gradients, ambient
  glow, glass cards, oversized radii, or heavy workspace shadows.
- **Don't** use landing-page composition inside authenticated apps: no heroes,
  marketing copy, large metric mosaics, or decorative card grids.
- **Don't** build generic AI interfaces that foreground chat while hiding
  records, scope, permissions, commands, or results.
- **Don't** reinvent import, search, filter, sort, edit, save, context menus,
  dialogs, tables, or run status inside an app.
- **Don't** achieve density by sacrificing focus visibility, contrast, keyboard
  access, label clarity, or readable state.
- **Don't** use color as decoration or as the only carrier of meaning.
- **Don't** use nested cards, colored side stripes, gradient text, glassmorphism,
  or custom scrollbars.
- **Don't** hand-author a full-app loading skeleton. Keep the static app layout
  representative so the shell can derive it.
- **Don't** promote ordinary controls into large colored buttons or cards.
- **Don't** hide a left/right pane on a smaller container without a visible
  tab, drawer, stack, or back route to the same workflow.
