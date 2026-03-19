# Chrome Extension Implementation Roadmap

## Purpose

This roadmap is the working reference for the extension rebuild.

It translates the existing product documents into a staged implementation plan so the project can move in a controlled way:

- [README.md](/Users/michaelwelsch/Dokumente%20-%20MacBook%20Air%20von%20Michael/Dokumente%20-%20MacBook%20Air%20von%20Michael/local_ai_tunes/fuck-api-train-local-ai/README.md)
- [UI_UX_DESIGN_GUIDE.md](/Users/michaelwelsch/Dokumente%20-%20MacBook%20Air%20von%20Michael/Dokumente%20-%20MacBook%20Air%20von%20Michael/local_ai_tunes/fuck-api-train-local-ai/UI_UX_DESIGN_GUIDE.md)
- [COOP_SYNC_ARCHITECTURE.md](/Users/michaelwelsch/Dokumente%20-%20MacBook%20Air%20von%20Michael/Dokumente%20-%20MacBook%20Air%20von%20Michael/local_ai_tunes/fuck-api-train-local-ai/COOP_SYNC_ARCHITECTURE.md)

This document is intentionally high level.
It defines workstreams, sequence, and boundaries.

## Product Goal

The extension should feel like:

- a browser-native capability workshop
- a Craft and bundle system
- a sharing and lineage system

It should not feel like:

- a generic LLM chat wrapper
- a dev console
- a bundle of unrelated settings pages

## Main Workstreams

## 1. Sidepanel Product Surface

Goal:

- turn the sidepanel into the primary use surface for Crafts

Work:

- keep the Craft list as the primary structure
- introduce a clear `Installed` plus `Available` flow
- keep the primary action as `Run`
- replace `Craft`/`Inspect`/chat-like patterns with `Open Bundle`
- remove full bundle editing from the sidepanel
- keep sync and peer state compact

Explicit note:

- the dev header with test buttons stays during development
- add a setting to show or hide the dev header
- default should be hidden for normal product usage once the toggle exists

## 2. Bundle Tab

Goal:

- create one dedicated bundle tab for inspection and editing

Work:

- add one stable bundle entrypoint from the sidepanel
- consolidate the current split across `Training Data`, `Navigator`, and related views
- define tab structure:
  - `Overview`
  - `Data`
  - `Model`
  - `Tools`
  - `Skillset`
  - optional `Lineage`
- keep source, ownership, sync state, and editability visible in one stable header

## 3. Sync UX And State Model

Goal:

- make peer sync understandable without explanatory clutter

Work:

- rename sync language toward `session code` or `share code`
- separate `discoverable`, `shared`, `available`, `subscribed`, `collaborative`, `forked`
- let remote Crafts appear in `Available` before local install
- add `Subscribe`, `Remove`, `Fork`, and `Freeze`
- add owner-side share mode selection:
  - `view`
  - `collaborate`

## 4. Co-Op Control Plane

Goal:

- prepare the sync architecture for collaboration without breaking the current data plane

Work:

- add replicated control-plane collections
- define claim, run, and event schemas
- add revision metadata conventions for mutable artifacts
- enforce scope-based write gating in app logic

Initial scopes:

- `dataset`
- `tools`
- `skillset`
- `training`
- `weights_promotion`

## 5. Training Architecture Hardening

Goal:

- make current and future training safe for co-op and later distributed workflows

Work:

- separate candidate training outputs from active weights
- move toward explicit promotion of trained results
- keep active training exclusive at first
- reserve space for later distributed worker orchestration

## 6. Data Model Stability And Migration Safety

Goal:

- avoid later destructive migrations

Work:

- preserve current chunked bundle storage
- extend through new collections and additive metadata
- avoid changing existing Craft ids and share ids
- keep old bundle snapshots readable
- avoid replacing the current artifact transport format

## 7. Visual Cleanup Across Themes

Goal:

- reduce noise without removing theme choice

Work:

- keep the three themes
- unify component hierarchy and interaction behavior across themes
- reduce chip, frame, and card overuse
- remove persistent explanatory clutter
- make hover and inspect the main path for extra detail

Important:

- this is not a theme removal task
- this is a component and hierarchy cleanup task

## Phased Sequence

## Phase 0. Documentation And Architectural Guardrails

Status:

- in progress

Deliverables:

- UI/UX guide
- co-op sync architecture
- implementation roadmap

## Phase 1. Minimal Product Restructure

Priority:

- highest

Deliverables:

- sidepanel simplified around Craft list and run flow
- `Open Bundle` entrypoint
- dev header toggle in options
- sidepanel no longer acts as dataset editor or bundle editor
- `Available` region introduced

## Phase 2. Bundle Tab Consolidation

Priority:

- high

Deliverables:

- one dedicated bundle tab
- existing training-data and navigator content migrated into bundle structure
- stable bundle header and tab model

## Phase 3. Sync State Expansion

Priority:

- high

Deliverables:

- `Subscribe`
- `Remove`
- `Freeze`
- owner-side `view` vs `collaborate`
- explicit remote state badges and actions

## Phase 4. Co-Op Foundations

Priority:

- high, but after the basic product surfaces are clean

Deliverables:

- `collab_claims`
- `agent_runs`
- `agent_events`
- artifact revision metadata
- scope-based single-writer rules

## Phase 5. Training Safety

Priority:

- medium-high

Deliverables:

- candidate weights artifacts
- explicit promote action
- training job tracking

## Phase 6. Advanced Co-Op And Distributed Work

Priority:

- later

Deliverables:

- proposal-only parallel dataset work
- possible row-level dataset operations
- distributed training workers
- broader collaboration ergonomics

## What Is In Scope Now

- restructuring the sidepanel
- creating the bundle tab
- improving sync state clarity
- preparing co-op schema and artifact conventions
- adding a dev-header visibility setting

## What Is Not In Scope For Immediate Implementation

- full concurrent dataset editing
- agent freeform chat with each other
- CRDT-style tool editing
- full distributed training execution
- replacing the current chunked sync architecture

## Immediate Task List

## A. Product Surface Refactor

- define the new sidepanel information architecture
- remove bundle-editing responsibilities from the sidepanel
- keep only compact run feedback in the sidepanel
- add `Open Bundle`
- split the list into `Installed` and `Available`

## B. Dev Header Control

- add an option setting `showDevHeader`
- default new installs to hidden once the setting exists
- keep the current dev header code path available during development

## C. Bundle Tab Definition

- define URL and routing behavior for a Craft bundle tab
- map current options sections into bundle tabs
- design the stable bundle header

## D. Sync State Model

- extend share metadata with permission mode
- define subscribe/remove/freeze actions
- define remote local-install state separate from raw remote visibility

## E. Co-Op Data Contract

- define new RxDB collections for claims, runs, and events
- define artifact revision metadata fields
- define allowed write scopes and lock rules

## F. Training Safety Contract

- define candidate artifact ids and active artifact ids
- define promote flow
- define training job lifecycle records

## G. UI Cleanup Pass

- simplify component usage across themes
- reduce persistent labels and notes
- move nonessential explanation into hover or inspect patterns

## Dependencies

- the sidepanel refactor should happen before visual cleanup
- the bundle tab should be defined before migrating more editor surfaces
- the sync state model should be defined before owner-side collaborate mode is exposed
- co-op collections should be added before collaborative write mode is considered production-ready
- training candidate/promotion flow should be in place before any distributed training work

## Success Criteria

We are on track when:

- the sidepanel reads as a capability browser, not a chat panel
- the bundle has one stable home
- share and subscribe states are unambiguous
- the sync architecture remains additive, not destructive
- co-op can be added without reworking the chunked data plane
- future training orchestration can build on explicit job and candidate artifacts

## Working Rule

Any implementation change should be mapped back to one of the workstreams in this roadmap.
If a proposed change does not clearly support one of them, it should probably wait.
