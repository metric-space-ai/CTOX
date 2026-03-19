---
name: bios-interface-bootstrap
description: Use when the CTO-Agent needs to build, replace or rewire the BIOS interface from a stable start template with SQLite-backed runtime data.
---

# BIOS Interface Bootstrap

This skill exists so the BIOS interface does not restart from zero on every installation or redesign.

Use it when:

- the owner asks for a new BIOS interface
- the current BIOS page must be reworked without losing its basic shell
- the agent needs a stable bootstrap template before owner-specific shaping
- runtime data should come from SQLite instead of hardcoded mockup values

## Required sources

Read these first:

1. `../../../contracts/bios/bios.json`
2. `../../../contracts/homepage/homepage-policy.json`
3. `../../../src/pages.rs`
4. `../../../src/app.rs`
5. `../../../src/runtime_db.rs`
6. `./references/bios-interface-template-contract.md`
7. `./references/bios-template-sqlite.sql`
8. `./assets/bios-start-template.html`

If the runtime exists, also inspect:

9. `../../../runtime/cto_agent.db`

## Core rules

- BIOS is a visible website surface, not an internal-only prompt artifact.
- Keep a stable shell: top status bar, tab row, left control rail, right work area.
- Use contracts for constitutional truth and SQLite for live operational state.
- Do not surface raw secrets or superpassword values in the template.
- Root credential setup must remain a dedicated protected form path.
- BIOS freeze must stay visible and must block normal mutation paths.
- Terminal remains the permanent fallback even when BIOS becomes the preferred trust surface.
- Start from the bundled template and data contract before inventing a fresh layout.
- Prefer named panels and explicit adapters over inline hardcoded state blobs.

## Start template contract

- HTML scaffold: `./assets/bios-start-template.html`
- Live snapshot route: `/api/bios/template-data`
- SQLite source of truth: `runtime/cto_agent.db`
- Primary runtime tables:
  - `owner_trust`
  - `focus_state`
  - `tasks`
  - `bios_dialogue`
  - `memory_items`
  - `memory_summaries`
  - `learning_entries`
  - `resources`
  - `skills`
  - `homepage_revisions`
  - `bios_uploads`
- Primary contract files:
  - `contracts/bios/bios.json`
  - `contracts/homepage/homepage-policy.json`
  - organigram and root-auth contracts already loaded by the Rust app layer

## Build pattern

1. Review the current BIOS page and the bundled start template before changing structure.
2. Preserve the shell and tab model unless the owner explicitly asks to replace it.
3. Map constitutional fields from contracts and operational fields from SQLite into the shared snapshot shape.
4. Keep write paths explicit: BIOS update, BIOS chat, brain access, root auth, freeze.
5. Mask or summarize sensitive values instead of dumping them raw into the UI.
6. If the layout changes materially, update the template resource instead of leaving the improvement only in a one-off page patch.

## Never do

- Never rebuild the BIOS UI from a blank page if the start template already fits the task.
- Never couple the interface to a single hardcoded mockup dataset.
- Never let branding lock or freeze controls drift out of the visible root section.
