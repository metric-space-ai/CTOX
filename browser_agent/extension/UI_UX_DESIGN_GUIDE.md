# Chrome Extension UI/UX Design Guide

## Scope

This guide applies only to the Chrome extension UI.
Training scripts, local experiments, reference code, and development helpers are out of scope.

The goal is to keep the shipped product aligned with the thesis in `README.md`:

- a `Craft` is a portable capability bundle
- the product is centered on browser actions, local capability building, and community sharing
- the product must not feel like a generic LLM chat wrapper

## Product Thesis

The extension is not a chat client with extra panels.
It is a browser-native workshop for:

- building narrow capabilities
- running them in browser context
- inspecting the full bundle
- sharing them with peers
- subscribing to other people's bundles
- collaborating or forking with visible lineage

Every UI decision should reinforce those ideas.

## Hard UX Rules

These rules are non-negotiable.

1. The first impression must be calm, clean, and directional.
2. Meaning must come from layout, hierarchy, spacing, typography, and state, not from helper copy.
3. Persistent explanatory text should be rare. If more explanation is needed, use hover, inspect, or an explicit detail view.
4. Do not decorate every element with borders, chips, badges, or framed callouts.
5. Do not default to subtitles, micro-descriptions, or programmer-style comments under controls.
6. Do not make the sidepanel look like a conversation UI.
7. Free text input is secondary unless the selected Craft truly requires it.
8. Browser action, bundle state, and lineage must be legible without reading paragraphs.

## Vocabulary

Use product nouns that fit the thesis.

Prefer:

- Craft
- Bundle
- Run
- Result
- Data
- Model
- Tools
- Skillset
- Share
- Subscribe
- Collaborate
- Fork
- Freeze
- Lineage

Avoid as primary UI language:

- Chat
- Conversation
- Assistant
- Reply
- Composer
- Message thread
- Wrapper-like "ask the AI" phrasing

`Prompt` is valid inside the dataset editor because prompt-to-JSON data is a real artifact.
It should not be the dominant noun of the main product surface.

## Core Object Model

A Craft is always one bundle.

The bundle contains:

- task definition
- dataset
- compiled model weights
- browser tools
- skillset and resources
- lineage and sync metadata

Important product truth:

- the model weights are the compiled result of the dataset
- tools and skillset are not optional extras
- share, subscribe, collaborate, and fork all apply to the bundle first
- a user may edit one part, but the semantic unit is still the full bundle

## Main Surfaces

### 1. Sidepanel

The sidepanel is for use, quick status, and fast entry.

It should do only these jobs:

- show installed Crafts
- show available remote Crafts in the same flow
- let the user run a Craft
- show compact run/result feedback
- show compact sync and peer presence
- offer one action to open the Bundle tab

It should not be the full bundle inspector.
It should not contain the full bundle overview.
It should not turn into a chat workspace.

### 2. Bundle Tab

Bundle inspection and bundle editing live in a dedicated tab, never in the sidepanel.

This tab is the home for:

- Overview
- Data
- Model
- Tools
- Skillset
- Lineage or Versions

The `Overview` tab is the most important one.
It must show all bundle parts together so the bundle concept is understood at a glance.

### 3. Options

Options is the control surface.

It should remain responsible for:

- providers
- model slots
- signaling server and session code
- share permissions
- published Crafts
- peer management

Options is not the place for daily bundle editing.

## Sidepanel Information Architecture

The sidepanel should show one continuous Craft list with two visual regions:

- Installed
- Available

Those regions should feel continuous, not like different products.
`Available` sits directly after `Installed` in the same scroll flow and uses a lighter visual treatment.

Each row should answer these questions immediately:

- What is this Craft?
- Where does it come from?
- Can I run it now?
- Is it local, shared, subscribed, collaborative, or forked?

Each row should not try to explain the whole system.

The primary row actions should be:

- Run
- Open Bundle
- Subscribe or Remove for available remote Crafts

The sidepanel should not lead with:

- large textareas
- reply panels
- agent brief panels
- chat-like output containers

## Bundle Tab Structure

The Bundle tab should use a stable tab layout.

Required tabs:

- Overview
- Data
- Model
- Tools
- Skillset

Recommended additional tab:

- Lineage

Rules:

- Keep the Bundle header stable across all tabs
- Keep source, ownership, and sync state visible in the header
- Do not restate the same explanation in each tab
- Show dependencies visually instead of describing them in text

The `Overview` tab should make the bundle relationship obvious:

- dataset feeds model
- tools belong to the same capability
- skillset belongs to the same capability
- lineage and sync belong to the same capability

## Sync, Share, Subscribe, Collaborate

### Session Setup

The current random private code should be treated as a local-only default.
When a user wants to share, they replace it with their own deliberate session code.

Do not frame this as account security UI.
This is a peer session boundary, not a login password.

Preferred language:

- session code
- share code
- sync code

Avoid leading with:

- password

### Discovery vs Sharing

Connection and publication are separate concepts.

- using the same signaling server and session code makes peers discoverable
- no Craft is shared until the owner explicitly shares it
- share permissions are per Craft

The owner should always be able to see:

- which peers are visible
- which peers are live
- which Crafts are shared
- which peers have subscribed
- which peers have collaborative write access

## Share Permission Modes

Each shared Craft needs an explicit permission mode.

### View Mode

Remote peers may:

- see the Craft in `Available`
- subscribe to it
- receive upstream updates
- inspect the bundle

Remote peers may not modify the shared lineage directly.
If they edit any bundle part, that edit creates a fork automatically.

### Collaborate Mode

Remote peers may:

- subscribe to the Craft
- edit the shared lineage
- sync those edits back to the shared Craft

This is true co-op development.
It should be visibly different from a normal subscription.

Rules for collaborate mode:

- collaborators are editing the same lineage
- ownership remains visible
- share permissions remain owner-controlled
- a collaborator can still choose to fork out into an independent lineage
- a collaborator fork stops collaborative syncing for that new fork

## Remote Craft States

The product needs a strict and visible state model.

### Available

The Craft is visible from a peer but not yet installed locally.

### Subscribed

The Craft is installed locally as a synced read-only mirror.

Rules:

- upstream updates flow in
- local edits are not allowed in place
- first modification auto-forks
- the user may optionally create a freeze fork without editing

### Collaborative

The Craft is installed locally with write access to the shared lineage.

Rules:

- upstream updates flow in
- local edits sync back
- this is not a fork
- explicit `Fork` creates a new independent lineage

### Forked

The Craft is a local independent lineage derived from another Craft.

Rules:

- no upstream sync
- full local edit rights
- origin and ancestry remain visible

### Removed

The user can remove a subscribed or available remote Craft from the local UI.
Removal should feel reversible and lightweight, not destructive.

## State Transitions

The intended transitions are:

- Available -> Subscribe -> Subscribed
- Available -> Subscribe -> Collaborative
- Subscribed -> Modify -> Forked
- Subscribed -> Freeze -> Forked
- Collaborative -> Fork -> Forked
- Subscribed or Collaborative -> Remove -> Available or hidden remote entry

The UI must make these transitions legible through state and action labels, not through paragraphs.

## Visual Language

The extension must visually communicate `browser capability system`, not `chat console`.

The visual language should favor:

- compact rows
- strong hierarchy
- few dominant actions
- calm surfaces
- dense but readable bundle summaries
- minimal status markers
- hover for detail

The visual language should avoid:

- chat bubbles
- large message panes
- permanent instructional labels
- stacked badges everywhere
- decorative frames around every action
- equally loud controls

## Copy Rules

Copy should be short and functional.

Good:

- Available
- Subscribed
- Collaborative
- Forked
- Shared
- Open Bundle
- Run
- Freeze
- Fork

Bad:

- This craft is currently in a synced read-only state and can be modified only after...
- Click here to open the screen where you can inspect the different bundle parts...

If the user needs that level of explanation, the layout is not doing enough work.

Use hover for:

- permission explanations
- source details
- lineage detail
- sync timestamps
- precise behavior of fork and collaborate modes

## Empty States and Guidance

Empty states should exist, but they should not lecture the user.

Good empty states:

- No Crafts yet
- No remote Crafts available
- No dataset yet

Bad empty states:

- multi-sentence mini documentation blocks
- warnings that restate the whole system

## Consequences For The Current Extension

These implementation consequences follow from this guide:

1. The sidepanel should move away from `reply`, `prompt`, and `agent brief` as dominant visual patterns.
2. The sidepanel should prioritize Craft rows, run actions, compact status, and `Open Bundle`.
3. Full bundle inspection must move into a dedicated extension tab.
4. Sync UI must distinguish `discoverable`, `shared`, `subscribed`, and `collaborative`.
5. Remote Crafts should appear in an `Available` region contiguous with installed Crafts.
6. Subscribed read-only Crafts must auto-fork on modification.
7. Collaborative Crafts must support shared editing without hiding ownership and lineage.

## Decision Summary

The current agreed product decisions are:

- remote users subscribe first
- modifying a subscribed read-only Craft auto-forks
- a freeze fork is optional without editing
- the owner can share a Craft in read-only or collaborative write mode
- the full bundle overview lives in a dedicated tab, not in the sidepanel
- remote Crafts should appear in an available region attached to the installed list
- the guide applies to the Chrome extension only
