# Fuck API, Train Local AI

This Chrome extension is a local-first workshop for building, sharing, forking, and tuning small AI capabilities directly in the browser.

It is not trying to become another "one API app for everything."
It is trying to make narrow, useful capabilities cheap to build, easy to share, and easy to run locally.

## The message

The message of this project is simple:

- use strong remote models when they help you build
- train the useful part locally
- share capabilities with other people as community assets, not as locked platform features

The point is not "never use APIs."
The point is "do not make the final capability depend on an API forever."

The training flow is staged on purpose:

- use stronger models or agents to write datasets, tool code, and bundle policy
- train the narrow supervised core locally first
- only if supervised training plateaus, refine with browser rollouts plus preference or lightweight RL stages

## Community first

This project is not only about local inference. It is also about community distribution.

The social model is closer to old private file sharing than to app-store distribution:

- peers can publish capabilities directly to other peers
- other people can browse those capabilities, copy them, fork them, and modify them locally
- the community can build lineages of capabilities instead of waiting for a central platform to ship them

The difference is that these are not pirated media files.
They are community-created AI assets.

That is the core idea:
not just "run AI locally," but "let people own, exchange, remix, and improve useful capabilities together."

## Open-source AI, not just open weights

This project argues for something stronger than open-weight AI.

Open weights alone are not enough if the useful part of the system still lives somewhere else:
- in private prompts
- in hidden training data
- in locked evaluation sets
- in platform-only workflows

The unit that matters here is the capability bundle:
- task definition
- prompt-to-JSON training data
- tool scripts
- bundle policy, reward, and judge specs
- lineage
- evaluation context
- tuned weights

Those pieces stay together on purpose.
If you share a capability, you share the whole bundle.
That is what makes model migration, re-training, auditing, and capability merging possible.

If the community has the training data, tool scripts, bundle policy, and tuning result, people can:
- move a capability to a newer base model
- fork it for a new domain
- merge ideas from several related capabilities
- audit what the capability was actually trained to do

That is the difference between "open weight AI" and "open-source AI."

## What is a Craft?

A `Craft` is one capability.

Examples:
- a JSON extractor
- a classifier
- a browser workflow helper
- a small structured task model for one specific job

Each Craft can carry:
- a name and summary
- a progress stage such as `Task Spec`, `Golden Pairs`, `Seed Collection`, `Canary`, `Pilot`, `Training`, or `Evaluation`
- dataset status and lineage metadata
- sharing state
- fork ancestry

In practice, a Craft is the object you create, revise, share, fork, and eventually train.

## High-level workflow

At a high level, the workflow looks like this:

1. Create a new Craft in the side panel.
2. Describe what the capability should do.
3. Let an agent help revise prompt-to-JSON training samples.
4. Save those samples as the local dataset for that Craft.
5. Start a local training run for the target model.
6. Share the Craft with peers or fork a remote Craft into your own local lineage.

## How the sharing model works today

The current extension already has a real peer-to-peer sharing model.

Today, you can:
- keep a Craft private
- publish a Craft to peers
- see remote Crafts from other peers
- open remote Crafts as read-only mirrors
- fork remote Crafts locally
- keep your modifications in your own lineage

That last part matters.
A remote Craft does not become silently editable in place.
It stays linked to its origin until you fork it locally.

That keeps lineage understandable:
- origin stays origin
- forks stay forks
- community improvements can branch instead of overwriting each other

## What is implemented today

Published Crafts already travel as complete capability bundles.

That means a shared Craft can include:
- the Craft definition itself
- the training dataset stored for that Craft
- the tool scripts attached to that Craft
- the current weight payload for that Craft

If a trained adapter exists, that trained adapter is bundled.
If no adapter has been trained yet, the bundle still carries the target-model reference and bundle metadata so peers can continue from the same starting point.

This matters because the app is not trying to share only a label or a prompt.
It is trying to share a capability that another peer can inspect, fork, retrain, and modify without depending on a private backend.

The storage rule is strict:
the bundle parts are chunked from the moment they are stored locally.
They are not kept as one big raw payload and chunked only later during sync.

## App structure

### Sidepanel

The Sidepanel is the working surface:

- create Crafts
- use a Craft
- open a Craft for revision
- inspect and edit training samples
- start a local training run
- fork a remote Craft into a local draft

### Options

The Options page is the control surface:

- configure providers and API keys
- assign models to slots
- inspect Craft lineage in the Navigator
- configure peer sync, signaling, display name, and token/password
- publish or unpublish local Crafts

## UI/UX guide

See `UI_UX_DESIGN_GUIDE.md` for the Chrome extension interaction model, state model, and visual rules.

## Co-op sync architecture

See `COOP_SYNC_ARCHITECTURE.md` for the long-term sync, collaboration, and distributed-training architecture decisions for the Chrome extension.

### Planned distributed inference and training rollout

The first distributed-compute milestone should not split one model run across several peers.
It should offload a whole inference or training job to one selected peer that already has the synced Craft data and the base model locally.

The working assumptions for that first stage are:

- peers in the sync session already share the relevant Craft artifacts
- each participating peer already has the local base model it wants to offer
- each peer advertises one offered model in settings
- that offered model is used for both remote inference and remote training at first
- trusted peers only; remote execution is an explicit opt-in capability

The architectural split should be:

- replicated collections for discovery, worker availability, claims, jobs, and progress metadata
- a separate direct WebRTC RPC channel for the actual job execution
- no prompt payloads, hidden states, gradients, or token-step traffic stored in RxDB collections

The staged implementation plan is:

1. Add worker settings so a peer can opt in to remote compute and advertise the single model it offers.
2. Add small replicated worker/job collections for discovery, availability, lease ownership, and status reporting.
3. Add a dedicated peer RPC path over WebRTC for `run_job`, `job_progress`, `job_result`, `job_error`, and `job_cancel`.
4. Route full remote inference jobs to one selected peer while keeping the existing local inference path unchanged.
5. Route full remote training jobs to one selected peer, returning only metrics, manifests, and adapter artifacts.
6. Add timeout, cancellation, version checks, trusted-peer gating, and local fallback behavior.
7. Consider split-model inference or split training only after the whole-job remote path is stable and measurable.

This order matters.
Whole-job remote execution is mostly additive and can reuse the current local runtime on the worker side.
Model-split execution is a later feature because it requires new runtime contracts, new cache/session handling, and likely new model artifacts.

### Rollout guardrails for remote and shared compute

The first remote-compute rollout caused app-wide regressions because it touched the core startup and persistence path too early.
That must not happen again.

For this project, the following files and concerns are treated as `core-risk`:

- `shared/craft-sync.js`
- RxDB schema registration, migrations, and database boot
- `options.js` boot and settings persistence
- `sidepanel.js` setup persistence
- `bg/service_worker.js` and offscreen bootstrap

Rules for any future remote/shared inference or training work:

1. No new remote/shared-compute feature may ship directly in the default startup path.
2. New remote/shared-compute code must stay behind a hard rollout flag until a full browser smoke path passes.
3. New RxDB collections or schema changes for remote/shared compute must not be added to the normal app boot unless the rollout flag is enabled.
4. `ignoreDuplicate` and `closeDuplicates` are not acceptable production fixes for RxDB boot issues here.
5. If `createRxDatabase()` succeeds and any later init step fails, the database must be closed before retrying.
6. When a core-risk change is made, the minimum smoke checklist is:
   - open Options
   - save setup/provider settings
   - create the starter Craft
   - open the Sidepanel
7. If any core-risk path breaks, rollback comes before further feature work or symptom-only patching.

Current status:

- remote whole-job inference/training groundwork exists in partial form
- the rollout is intentionally disabled in the app until it can be reintroduced behind stricter isolation and live-smoke validation
- split-model inference or split training is explicitly out of scope until the whole-job remote path is stable

## Implementation roadmap

See `IMPLEMENTATION_ROADMAP.md` for the staged rebuild plan and task structure for the Chrome extension.

## Technical architecture

### Local storage and sync

The extension uses:

- `RxDB` for collection management and replication
- `Dexie` as the browser storage backend
- `WebRTC` replication for peer-to-peer sync
- `SimplePeer` through the RxDB WebRTC connection handler

The browser database maintains several logical collections.

Peer-to-peer replicated today:

- published shared Crafts
- shared asset chunks
- peer presence

Local working state:

- local Crafts
- local Craft artifact manifests
- local artifact chunks for per-Craft training data, tool scripts, saved weights, and persisted UI state

Shared Crafts now embed a capability bundle manifest.
The heavy bundle sections are stored as chunk references, and the chunk rows are replicated separately through the shared asset channel.
That keeps both local storage and WebRTC replication on the same chunk-first model.

### Signaling server and token/password

Peer discovery is coordinated through a signaling server.

In the current implementation:

- the default signaling server is `wss://signaling.rxdb.info/`
- users can configure one or more signaling URLs
- a token/password can be set for the sync session
- the token is attached to the signaling URL and acts as a practical room boundary for peers using the same server settings

Presence heartbeats are also replicated, so the UI can show:

- visible peers
- live links
- remote Crafts
- whether sync is currently live or just showing cached remote data

### Share and fork semantics

Sharing is explicit.
A local Craft remains private until you mark it as shared.

When a peer receives a shared Craft:

- it appears as a remote mirror
- it stays read-only while linked to the remote peer
- it carries the bundled training data, tool scripts, and weights with it
- it can be forked locally
- the fork keeps lineage metadata such as origin, parent, and fork depth
- the bundled artifacts can be materialized locally again for further editing and retraining

This is important because it makes the community model understandable.
You can copy ideas from others without erasing authorship or origin history.
It also means the received capability was transported in the same chunked form the sender stored locally, instead of being rebuilt from one oversized source blob.

### Model slots

The app separates model roles on purpose:

- `Agent`: plans dataset revisions and capability improvements
- `Batch`: meant for cheaper large-scale prompt/JSON work
- `Vision`: meant for browser or visual tasks
- `Target`: the local model slot that should eventually run the capability

That separation keeps the system honest.
Not every model needs to do everything.

### Provider layer

The provider layer is configurable.
Built-in provider support includes:

- OpenAI
- Azure OpenAI
- Anthropic
- OpenRouter
- DeepSeek
- Groq
- Cerebras
- Ollama
- custom OpenAI-compatible endpoints
- Local Qwen for the target runtime

There is one important implementation detail:

- the `Target` slot is local and uses `Local Qwen`
- the current structured `Agent` planning path requires a remote model
- the strict `Agent` slot selection is narrower than the general provider catalog

So the architecture is provider-flexible, but the current agent loop is still opinionated.

### What the agent actually does

The current agent does not run an open-ended autonomous tool loop during training.

Instead, it performs a narrower and more useful job:

- it receives the Craft summary, stage, gaps, and current prompt-to-JSON samples
- it can use the configured provider stack instead of forcing one API vendor
- it returns structured CRUD operations over those samples
- it proposes which examples to add, update, or delete
- it keeps the dataset shape stable for the target task

That means the remote model is used as a training-data planner, not as the final runtime.
The final capability is meant to live in the bundle, not in the API.

### Local runtime and training

The local target path uses:

- an offscreen extension document for model execution
- `transformers.js`
- ONNX Runtime Web backends
- packaged ORT WASM assets
- a WebGPU execution plan for the local Qwen runtime

The current local target runtime supports `Qwen3.5-0.8B`, `Qwen3.5-2B`, and `Qwen3.5-4B` in the local slot.

The local training loop:

- builds a dataset from the Craft's stored prompt-to-JSON samples
- keeps the Craft's tool scripts alongside that dataset as part of the same capability bundle
- encodes those examples in the browser
- runs a small adapter training loop locally
- evaluates base vs adapted accuracy
- saves the trained adapter weights back into the Craft bundle as chunked-at-rest artifact data
- records runtime, dataset, and metric summaries for the run

In other words, the extension is not just a local chat wrapper.
It already contains a browser-native path for tuning a small task-specific capability.

## A precise way to read the project

If you want the short version, it is this:

`Fuck API, Train Local AI` means:

- use APIs as scaffolding, not as permanent dependency
- treat capabilities as community bundles that can be shared, forked, and merged
- move from platform-owned AI features to peer-owned capability lineages
- move from open-weight rhetoric to actually portable, inspectable, re-trainable capability bundles

That is the thesis.
The current extension already implements that bundle model in the app: weights, training data, and tool scripts are treated as one shared capability package instead of separate hidden pieces.
