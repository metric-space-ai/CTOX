# CTOX Architecture

## System Shape

CTOX consists of four main layers:

- the CTOX orchestration layer
- Codex as the current execution engine
- the CTOX proxy server
- the standalone local inference engine

The persistent system is CTOX. Codex is the current optimized execution core inside that system.

## CTOX Orchestration Layer

The orchestration layer covers the parts of the system that persist, route, supervise, recover, and verify work across multiple execution slices.

Core responsibilities:

- persistent service and control-plane behavior
- external communication routing
- durable queue, plan, schedule, and follow-up execution
- long-run mission control
- long-context memory and continuity management
- context optimization and recovery logic
- governance, verification, and assurance
- runtime mediation around the execution engine

Core modules:

- `src/service.rs`
  Service loop, HTTP control surface, pending work, background loops, and prompt dispatch.
- `src/channels.rs`
  Multi-channel communication substrate, routing state, sender policy, leasing, acking, and thread context.
- `src/queue.rs`
  Durable queue tasks and queue management commands.
- `src/plan.rs`
  Persistent multi-step plans, step lifecycle, and step emission.
- `src/schedule.rs`
  Time-based work emission and recurring task management.
- `src/lcm.rs`
  Long-context memory, continuity documents, mission state, retrieval, compaction, and verification persistence.
- `src/context_health.rs`
  Context scoring, failure-memory checks, repetition detection, and repair guidance.
- `src/mission_governor.rs`
  Loop governance for repeated blockers and forced repair/replan slices.
- `src/follow_up.rs`
  Post-slice follow-up decisions.
- `src/review.rs`
  Completion review logic.
- `src/verification.rs`
  Verification runs, claims, assurance, and closure-blocking evidence.

## Execution Layer

CTOX currently uses Codex as the execution engine for bounded work slices.

This means:

- CTOX prepares the context, mission contract, runtime settings, and policy environment
- `codex-exec` performs the bounded agent run
- CTOX persists the outcome, evaluates follow-up, and continues the larger mission

The execution engine is intentionally treated as a component. The orchestration system around it remains CTOX-owned.

## Proxy Server

The proxy layer exists so clients can use one stable OpenAI-style surface while CTOX mediates backend differences.

Main responsibilities:

- stable `responses` entrypoint
- backend-specific request rewriting
- routing to chat, embeddings, STT, and TTS backends
- telemetry and live switching
- runtime recovery and readiness handling
- optional web-search augmentation

Core modules:

- `src/inference/gateway.rs`
- `src/inference/supervisor.rs`
- `src/inference/runtime_env.rs`
- `src/inference/runtime_plan.rs`

## Standalone Inference Engine

The local inference engine lives separately under `engine/candle/`.

That engine is not the orchestration system and not the proxy. It is the local model-serving substrate used when CTOX runs on-host local inference instead of or alongside API-backed models.

## Feature Placement Rule

Use this rule when deciding whether a new feature belongs in CTOX or in the execution engine:

- put the feature in CTOX if it changes persistence, routing, scheduling, continuity, governance, verification, communication, proxying, web-path policy, or host-side runtime control
- put the feature in Codex if it changes execution semantics inside the bounded agent run itself
