# Cooperative Sync Architecture

## Scope

This document defines the long-term sync architecture for the Chrome extension.

It is meant to avoid destructive redesigns after release.
The goal is to keep current user data stable while leaving room for:

- read-only subscription
- collaborative editing
- agent-to-agent coordination
- later distributed training workflows

This is an architecture decision document, not a promise to implement every part immediately.

## Core Decision

The current data plane stays.

We do not throw away the existing RxDB + WebRTC + chunked artifact model.
Instead, we keep the current large-payload transport and add a small collaborative control plane on top.

That split is the key decision:

- bundle data and heavy artifacts stay in chunked artifact storage
- coordination, claims, runs, and training orchestration live in small replicated control collections

## Existing Data Plane To Preserve

The current extension already has the right base for bundle transport:

- `shared_crafts`
- `shared_asset_chunks`
- `presence`
- `local_crafts`
- `local_craft_artifacts`
- `local_artifact_chunks`

These are defined in [shared/craft-sync.js](/Users/michaelwelsch/Dokumente%20-%20MacBook%20Air%20von%20Michael/Dokumente%20-%20MacBook%20Air%20von%20Michael/local_ai_tunes/fuck-api-train-local-ai/shared/craft-sync.js).

This existing model must remain the durable foundation because it already gives us:

- local-first storage
- chunked-at-rest artifact persistence
- chunked bundle replication
- remote bundle hydration
- stable local Craft identity

## Why The Current Model Is Not Enough For Co-Op

Today, collaborative writing is unsafe because several important artifacts are written as whole snapshots:

- training data artifact payload contains the full `samples` array
- tool script artifact payload contains the full script payload
- the newest artifact write wins

That is acceptable for local-only editing and for read-only subscribe + fork.
It is not enough for simultaneous collaborative writers.

The risk is not the chunking format.
The risk is the write model.

## Architectural Principle

We separate the system into two layers.

### 1. Data Plane

This layer stores and replicates heavy capability data:

- Craft bundle metadata
- training dataset snapshots
- tool scripts
- browser capability bundles
- weights
- policy bundles
- future checkpoints

These payloads continue to use the current chunked artifact mechanism.

### 2. Control Plane

This layer stores and replicates small coordination data:

- who is working on what
- which agent run owns a scope
- what each run intends to change
- whether a write is still valid
- training job status
- worker availability

This layer must stay small, structured, and high-frequency.
It must never carry heavy bundle blobs.

## Required Future Collections

These collections are the planned additive control-plane schema.
They should be introduced before public release or very early, even if some remain dormant at first.

### `collab_claims`

Purpose:

- lease-based ownership of a write scope
- conflict prevention
- authoritative write gating

Examples of scopes:

- `dataset`
- `tools`
- `skillset`
- `training`
- optional `craft`

Minimum fields:

- `id`
- `craftId`
- `scope`
- `ownerDeviceId`
- `ownerName`
- `runId`
- `mode`
- `status`
- `baseRevision`
- `last_seen`
- `expires_at`

Rules:

- this is the authoritative lock/lease collection
- a claim expires if heartbeats stop
- writes without a valid matching claim are rejected by app logic

### `agent_runs`

Purpose:

- visible state for collaborative agent work
- run lifecycle and intent

Minimum fields:

- `id`
- `craftId`
- `scope`
- `ownerDeviceId`
- `ownerName`
- `status`
- `goal`
- `baseRevision`
- `started_at`
- `updated_at`
- `completed_at`

Status examples:

- `planning`
- `running`
- `needs_input`
- `blocked`
- `completed`
- `failed`
- `cancelled`

### `agent_events`

Purpose:

- structured inter-agent and UI communication
- not freeform chat

Minimum fields:

- `id`
- `craftId`
- `runId`
- `scope`
- `ownerDeviceId`
- `type`
- `payload`
- `created_at`

Event examples:

- `intent_declared`
- `scope_claimed`
- `scope_released`
- `proposal_created`
- `write_started`
- `write_committed`
- `write_rejected`
- `conflict_detected`
- `run_finished`

### `training_jobs`

Purpose:

- future-proof training orchestration
- candidate runs separate from active bundle weights

Minimum fields:

- `id`
- `craftId`
- `ownerDeviceId`
- `ownerName`
- `status`
- `mode`
- `baseDatasetRevision`
- `baseWeightsRevision`
- `targetArtifactId`
- `created_at`
- `updated_at`

Status examples:

- `queued`
- `preparing`
- `running`
- `paused`
- `completed`
- `failed`
- `promoted`

### `training_workers`

Purpose:

- future distributed training worker presence
- capability and availability advertisement

Minimum fields:

- `id`
- `jobId`
- `deviceId`
- `name`
- `capabilities`
- `status`
- `last_seen`

This collection can stay unused until distributed training becomes real.
It is still worth planning now.

## Additive Collection Policy

The current data plane collections must not be redefined around co-op.
New collaboration features should prefer:

- new collections
- new artifact kinds
- additive `meta` fields

over:

- replacing current payload structures
- changing current bundle transport format
- rewriting existing Craft identities

## Artifact Conventions To Adopt Now

Even before full co-op is implemented, every mutable artifact should move toward a stable revision contract.

This does not require a new RxDB collection schema because `payload` and `meta` are already open objects.

Every mutable artifact write should eventually include:

- `meta.revision`
- `meta.baseRevision`
- `meta.scope`
- `meta.updatedByDeviceId`
- `meta.updatedByRunId`
- `meta.updatedAt`
- `meta.writeMode`

Recommended `writeMode` values:

- `local`
- `subscribe_materialize`
- `collab_commit`
- `fork_commit`
- `training_candidate`
- `promotion`

This keeps future migrations small because the artifact table shape stays the same.

## Bundle Rules

The semantic unit is still the Craft bundle.
But concurrency is governed per scope.

Bundle scopes:

- `dataset`
- `tools`
- `skillset`
- `training`
- `weights_promotion`

Rules:

- bundle transport remains whole-bundle compatible
- active editing claims are per scope
- a bundle may have multiple active claims only if scopes are compatible
- training must stay exclusive at first

## Subscribe, Collaborate, Fork

The sync product model must distinguish:

- `view`
- `collaborate`

### View

- read-only mirror
- receives updates
- local modification auto-forks

### Collaborate

- shared lineage write access
- writes are allowed only with valid scope claims
- local explicit fork exits the collaborative lineage

This means collaboration is not “free editing.”
It is “editing through claims and revisions.”

## Agent Communication Model

Agent-to-agent communication is useful, but it is not the source of truth.

The authoritative controls are:

- claims
- revisions
- write gates

Agent communication is an advisory coordination layer.

### What agents may use communication for

- announcing intent
- declaring which scope they want
- publishing proposals
- announcing phase changes
- reporting completion

### What agents must not rely on

- natural-language agreements as the only conflict prevention
- unrestricted simultaneous writes to the same scope

Therefore:

- use structured event docs, not freeform chat messages
- runtime enforces the claims
- agents consume events as context

## Scope Strategy

### Initial Safe Strategy

- `dataset`: single writer
- `tools`: single writer
- `skillset`: single writer
- `training`: single writer
- `weights_promotion`: single writer

Multiple agents may still run in parallel if they are non-writing or proposal-only.

Examples:

- one agent writes dataset while another only reviews
- one agent writes tools while another proposes dataset changes

### Later Strategy

Only the dataset scope should be considered for true parallel writes later.

That should happen only after introducing:

- row-level stable sample ids everywhere
- operation-level dataset changes
- deterministic merge rules

Tools and skillset should remain claim-based exclusive much longer.

## Dataset Strategy

Current dataset storage is snapshot-oriented.
That is acceptable for now, but future co-op must not depend on rewriting the full dataset forever.

The target path is:

1. keep the current snapshot artifact as the materialized dataset
2. later add optional dataset operations or row-level records
3. derive the materialized snapshot from those operations when needed

This means future parallel co-op can be introduced without breaking old bundles.

Important preparation:

- all dataset rows must keep stable ids
- agent edits should be representable as add/update/delete operations against row ids
- revision numbers must be tracked on the dataset artifact

## Weights And Training Strategy

Active weights must not be overwritten directly by every training run.

This is the long-term rule:

- training jobs produce candidate artifacts
- active bundle weights are updated only by explicit promotion

That separation is required for:

- collaborative safety
- future distributed training
- rollback
- auditability

Recommended artifact kinds:

- keep the active weights artifact
- add immutable candidate checkpoint artifacts per job and step

Example id patterns:

- active: `capability-weights:{craftId}`
- candidate: `capability-weights:{craftId}:job:{jobId}:step:{step}`

This can be introduced without redesigning the artifact tables.

## Distributed Training Preparation

The architecture should allow future training across multiple peers without requiring a new sync model.

That means:

- job state lives in `training_jobs`
- workers advertise themselves in `training_workers`
- checkpoints travel as chunked artifacts
- promotion to active weights remains explicit

The extension does not need to implement full distributed training now.
It only needs to avoid boxing itself out of it.

The important design choice is that training coordination belongs in the control plane, while checkpoints and model data stay in the data plane.

## Conflict Policy

The current generic conflict handler is acceptable as a transport fallback.
It is not enough as collaborative product logic.

Collaborative correctness must come from app-layer rules:

- valid claim required
- write must match expected `baseRevision`
- invalid write becomes proposal or conflict event, not silent overwrite

In other words:

- RxDB conflict resolution keeps replication alive
- application rules keep collaboration correct

## Migration Policy

To minimize migration pain after release:

1. Do not replace current bundle storage.
2. Do not replace current chunking.
3. Extend through new collections and new artifact conventions.
4. Keep old bundle snapshots readable forever.
5. Treat future op-level or distributed features as additive.

The highest-risk migrations would be:

- changing existing Craft ids
- changing current share ids
- replacing chunked artifact storage with a different payload model
- making current bundles unreadable without backfill

Avoid all of those.

## What Should Be Implemented Early

These are the early architecture steps worth doing before broad release:

1. Add share mode semantics: `view` vs `collaborate`.
2. Add artifact revision metadata conventions.
3. Introduce small control-plane collections for claims and run status.
4. Keep current data plane untouched.
5. Move active training result writes toward candidate-then-promote.

## What Can Wait

These can wait until product demand proves them necessary:

- row-level dataset co-authoring
- CRDT-like merges
- free concurrent tool editing
- full distributed multi-worker training
- agent-to-agent autonomous negotiation beyond structured events

## Decision Summary

The architecture decision is:

- preserve the current RxDB chunked data plane
- add an additive replicated control plane for collaboration
- use claims and revisions as the authoritative co-op mechanism
- use structured agent events as a coordination layer
- keep training outputs as candidates before promotion
- design for future distributed training without rebuilding storage later
