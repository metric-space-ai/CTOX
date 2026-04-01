# CTO-Agent Repo Map

This repository contains the first bootstrap layer of the CTO-Agent.

Priority order:

1. `src/main.rs`
   The CLI entrypoint and command surface. This is the fastest way to see which subsystems exist and how operators trigger them.
2. `src/service.rs`
   The detached service loop: HTTP control surface, serial prompt execution, channel routing, the explicit survival control plane, timeout continuation, mission idle watchdog, and queue pressure containment.
3. `src/inference/chat.rs`
   The active chat-turn orchestration. It assembles the live prompt from LCM state plus continuity documents plus the visible governance snapshot, invokes Codex, persists turns, and refreshes continuity best effort after reply.
4. `src/lcm.rs`
   The long-context memory layer: persisted messages, compaction, hierarchical summaries, continuity documents (`narrative`, `anchors`, `focus`), forgotten-line tracking, and retrieval helpers.
5. `src/channels.rs`
   The shared communication and durable queue substrate for inbound/outbound messages, routing state, thread context lookup, sender policy, and queue-task persistence.
6. `src/plan.rs`, `src/queue.rs`, `src/schedule.rs`, `src/follow_up.rs`
   The durable work-orchestration layer around Codex: explicit plans, queue tasks, recurring schedules, and end-of-turn follow-up decisions.
7. `src/inference/supervisor.rs` and `src/inference/`
   Runtime supervision and local model execution policy: backend watchdogs, proxy/bootstrap logic, runtime env handling, model manifests, and runtime planning.
8. `contracts/history/`, `contracts/models/`, `contracts/clean_room_bootstrap_manifest.json`
   The human-readable canonical history and the model/runtime policy contracts. These are important policy inputs, but the live operational state now primarily lives in `runtime/*.db`.

Rules:

- Keep the Rust structure close to current Codex-shaped layers: thin CLI entry, service/router layer, memory/state layer, and inference/supervisor layer.
- Do not reintroduce stale references to `src/app.rs`, `src/contracts.rs`, or `src/supervisor.rs` as if they were still the active architecture.
- Treat `src/lcm.rs` plus `src/inference/chat.rs` as the canonical context-management path for active turns.
- Treat `src/channels.rs` plus `src/queue.rs` / `src/plan.rs` / `src/schedule.rs` as the canonical durable execution pipeline.
- Michael Welsch is the creator of this agent; the origin story must not be rewritten into a self-created myth.
- When architecture, governance or identity shifts materially, append a short honest entry to `contracts/history/creation-ledger.md`.
- The default Kleinhirn model is `gpt-oss-20b`, but the installation may be explicitly switched to `Qwen3.5-35B-A3B` via model policy; this is the canonical local vision/browser path.
- Local Kleinhirn upgrades are allowed when the host has materially more resources; this path stays separate from Grosshirn procurement.
- Real browser work should be modeled through reviewed browser capabilities and compact artifact outputs, not by shoving raw browser traces into the main agent context.
- Repeated browser-backed tasks must pass through the specialist-model pipeline before promotion into a small specialist model or deterministic worker.

Agent-loop governance principles:

- Treat CTOX as an always-on multi-turn agent, not as a one-shot code generator. Protect loop quality and mission continuity over local convenience hacks.
- The default stance is non-interference: do not add hidden control loops that second-guess the model's own multi-turn reasoning unless the mechanism is truly required for loop survival or safety.
- Only explicit `survival` or `safety` mechanisms may autonomously interrupt, enqueue, reroute, pause, or block work in the background.
- Today this allowed class includes queue pressure containment, timeout continuation, mission idle watchdog, sender authority boundaries, and secret-input boundaries. Treat any expansion of that class as a material governance change.
- Every autonomous background mechanism must be first-class in the runtime, persist visible governance events, and be rendered back into the live agent context. CTOX must be able to see what acted, why it acted, and what it changed.
- Do not add silent autonomous behavior based only on heuristics extracted from agent prose, such as keyword spotting, sentiment, completion tone, or mini-DSL parsing, unless the mechanism is explicitly approved as part of the survival control plane.
- Heuristics are allowed as advisory diagnostics. They are not allowed to become hidden mission truth or hidden action triggers by default.
- Avoid agent-hostile control patterns such as forced JSON or mini-language output formats whose primary purpose is to let the harness parse an answer and automatically derive real actions from it.
- If structured output is needed for a tool contract, keep the contract narrow, explicit, user-visible in the prompt, and limited to the tool invocation itself. Do not turn the general conversation loop into a parser target.
- Prefer explicit tool calls, explicit durable state mutations, and explicit agent-declared outcomes over prose inference. The system should store what was declared, what was directly observed, and what was only inferred as distinct categories.
- Do not let review, follow-up, context-health, or mission-governor heuristics silently enqueue or rewrite work. Outside the survival or safety class, those systems should be advisory unless the agent explicitly invokes them.
- When a mechanism exists because the loop would otherwise collapse operationally, harden it. Make it idempotent, rate-limited where appropriate, deduplicated, and persistent. Survival mechanisms should be few, durable, and boring.
- When editing `src/service.rs`, `src/inference/chat.rs`, `src/follow_up.rs`, `src/context_health.rs`, `src/mission_governor.rs`, `src/review.rs`, or `src/verification.rs`, check whether the change increases hidden determinism over the loop. If it does, justify it against these principles or redesign it.
- Do not reintroduce automatic completion gating, automatic repair task creation, or automatic closure inference from free-form prose without explicit human approval and a matching visible governance design.
