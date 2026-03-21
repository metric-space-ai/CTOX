# CTO-Agent

An always-on CTO-Agent that lives as a Rust control plane on its own host, first makes itself constitutionally and trust ready, and only then moves into normal CTO operation.

This README describes the current implementation state of the repository, not just the founding idea. The agent is already modeled as a running system: with a visible BIOS surface, supervisor loop, SQLite runtime, skills, browser bridge, mail path, learning path, and person-level memory.

## Installation

For an Ubuntu 24 console there is now a real one-liner that clones or updates the CTO-Agent into `~/cto-agent`, starts the Linux installer, sets up user services, and then automatically enters the attach / Infinity Loop:

```sh
bash -lc "$(curl -fsSL https://raw.githubusercontent.com/metric-space-ai/CTO-Agent/main/scripts/install_cto_agent_remote.sh)"
```

Important notes:

- The one-liner is intentionally **not** documented as `curl | sh`, so the downstream installer can remain interactive and `install-bootstrap-tui` plus auto-attach do not fail due to a lost TTY.
- The default target is `~/cto-agent`. You can change it by setting `CTO_AGENT_INSTALL_DIR=/path` before running the one-liner.
- The bootstrap path is designed for Linux with `apt-get`, `sudo`, and `systemd --user`; that is the intended Ubuntu 24 path.
- The browser installer no longer mutates the desktop environment silently. It uses an existing graphical session; an additional KDE desktop is installed only with explicit `CTO_AGENT_INSTALL_KDE_DESKTOP=1`.
- Interactive browser and host-GUI steps must run against the active desktop session, not against the bare systemd service environment. The runtime path bridges `DISPLAY` / `WAYLAND_DISPLAY`, `XAUTHORITY`, `DBUS_SESSION_BUS_ADDRESS`, and `XDG_RUNTIME_DIR` from the real session.

The repo-local installation path remains:

```sh
sh scripts/install_cto_agent.sh
```

- This installer currently targets a Linux host with `systemd --user` as the always-on runtime.
- It builds the Rust host, initializes contracts, TLS, and SQLite, optionally runs the communication bootstrap TUI, installs the local kleinhirn, configures user services, and then starts the attach path.
- For true always-on operation, `systemctl --user` and ideally `loginctl enable-linger` are required; `scripts/install_linux_user_services.sh` configures or verifies that.

### HARD WARNING: mistral.rs Multi-GPU Guardrails

- **Do not rely on `mistralrs doctor` to decide whether NCCL or tensor parallelism is available.** The current `doctor` output does not report distributed backends reliably. Installation and runtime decisions must use Cargo install metadata from `~/.cargo/.crates2.json` or an explicit override.
- **Never make installation or runtime decisions from `mistralrs doctor`:** if `doctor` does not show NCCL, that does not prove the build lacks NCCL. The decision must come from the model contract plus real build metadata.
- The multi-GPU strategy is now **model-family specific** and lives in the model-policy contract:
  - `gpt-oss` auf Mehr-GPU-Hosts: `startupMultiGpuMode=auto_device_map`, `startupTensorParallelBackend=disabled`, also `MISTRALRS_NO_NCCL=1`
  - `Qwen3`/`Qwen3.5` auf Mehr-GPU-Hosts: `startupMultiGpuMode=tensor_parallel`, `startupTensorParallelBackend=nccl`
- For NCCL / tensor parallelism, the installer automatically picks the largest power-of-two subset on non-power-of-two GPU counts and prefers display-free cards.
- The installer now persists the expected `mistral.rs` features as `CTO_AGENT_MISTRALRS_FEATURES` in `runtime/kleinhirn.env`. For manual starts or special cases you may pin that explicitly, for example `CTO_AGENT_MISTRALRS_FEATURES='cuda flash-attn nccl'`.
- `PagedAttention` stays off for `openai/gpt-oss-20b`. That is not intuition-based tuning; it is an upstream model boundary in `mistral.rs`.

## Quick Picture

- The agent starts terminal-born, but immediately builds a local HTTPS control plane with BIOS, root-auth, history, browser, model, and census pages.
- Binding truth lives in `contracts/`; live operational truth lives in `runtime/cto_agent.db`.
- The main brain for always-on operation is a local kleinhirn. Grosshirn is an optional task-bound boost, not the default.
- The main agent intentionally runs unsandboxed on its dedicated host. Delegated workers remain sandboxed and must escalate to the CTO-Agent for larger interventions.
- The agent works in bounded turns, evaluates outcomes through review steps, learns in SQLite, and keeps condensed learnings available in its working context.
- Idle does not mean sleeping. It means self-directed improvement: understanding the environment, testing tools, reflecting on progress, maintaining human relationships, and spotting resource gaps.

## Constitutional Model

The CTO-Agent is deliberately layered. Not everything is equally mutable.

- `contracts/genome/genome.json` is the innate development direction. It holds deep genes such as `owner_instruction_is_absolute_top_priority`, `idle_means_self_directed_improvement`, `delegate_recurring_work`, and `never_self_rewrite_constitution`.
- `contracts/bios/bios.json` is the deployment-specific constitution. The BIOS is visible on the website, not just internal prompt material.
- `contracts/org/organigram.json` models the structure above, beside, and below the CTO-Agent.
- `contracts/root_auth/root_auth.json` and the protected root-auth form define the root of trust. The superpassword may only be set through the website.
- `contracts/history/origin-story.md` and `contracts/history/creation-ledger.md` record origin and material architecture/governance shifts honestly.
- Additional policies in `contracts/` steer context, mode transitions, loop safety, execution authority, browser work, the specialist pipeline, homepage behavior, and self-protection.

Important binding rules:

- BIOS freeze blocks normal BIOS mutation.
- Low-trust channels such as email or WhatsApp may not set root trust or lock owner branding.
- The terminal remains the system-adjacent fallback surface.
- Owner branding is allowed only after BIOS-based trust takeover.
- The CTO may shape the structure beneath itself, but may not rewrite the authority above itself.

## Operating Model

The core of the system is an Infinity Loop with bounded agentic turns.

1. `src/app.rs` initialisiert Contracts, TLS, SQLite, Browser-Bridge und Bootstrap-Tasks.
2. `src/supervisor.rs` runs the heartbeat: ingests interrupts, prioritizes the queue, creates duties, watches the loop, launches bounded turns, creates reviews, and advances workers.
3. `src/agentic.rs` builds task context, chooses the brain route, executes the LLM step, and normalizes structured output.
4. `src/runtime_db.rs` persists tasks, turns, checkpoints, learnings, person traces, resources, skills, browser jobs, events, and summaries.
5. `src/context_controller.rs` condenses exactly the working context the next bounded turn should see and now also emits a compact-controller bridge block for progress review, reprioritization, and model routing at each compaction boundary.

The supervisor keeps certain CTO duties alive at all times:

- `homepage_bridge`
- `root_trust`
- `organigram_contract`
- `owner_binding`
- `bios_freeze`
- `model_or_resource`
- je nach Lage auch `grosshirn_activation` oder `grosshirn_procurement`

When the normal queue is empty, the loop also generates self-directed idle work such as:

- `environment_discovery`
- `tool_exploration`
- `progress_reflection`
- `person_relationship_review`

So the agent does not work as a free-running conversation, but as a loop of:

- observe
- prioritize
- execute in a bounded way
- review
- delegate or continue
- block or request resources
- improve itself during idle time

## Modes and Safety Logic

The mode system is modeled explicitly in `contracts/system/mode-system-policy.json`. Important modes are:

- `bootstrap`
- `observe`
- `reprioritize`
- `self_preservation`
- `recovery`
- `historical_research`
- `execute_task`
- `review`
- `delegate`
- `await_review`
- `request_resources`
- `idle`
- `blocked`

The loop-safety policy enforces:

- bounded turns instead of blind continuous thinking
- progress as a requirement when resuming the same task
- delegation or resource requests instead of livelock
- hard handling of stall, crash, queue starvation, and context poisoning
- self-preservation stages `newborn`, `guided`, `adaptive`

Important: the autonomy is intentionally not purely deterministic. The supervisor creates duties, reviews, and safety boundaries deterministically, but the model decides within those boundaries about learn-worthiness, delegation, context need, grosshirn requests, skill need, and proactive suggestions.

## Runtime Surfaces

The CTO-Agent has several real operating surfaces:

- Terminal bridge for direct input such as `Speaker: message` or commands like `/status`.
- Attach socket and attach TUI for live insight into thread, focus, tasks, token usage, active model, and the ASCII owner chat surface.
- Local HTTPS control plane with pages for home, bootstrap chat, BIOS, organigram, root-auth, history, browser, models, and census.
- Channel-interrupt path for external input from mail or later channel integrations.
- A Python-based, intentionally hackable mail client in `scripts/cto_mail_client.py`.

The attach TUI is no longer just a passive log viewer. It now acts as a real owner chat surface:

- the first free-form TUI chat message enters the shared interrupt path
- interrupt compaction immediately refreshes the running workstream and the new queued chat task
- idle chat absence lets the loop fall back into normal task work
- a settings page in the same TUI lets the owner store mail credentials, API keys, model slots, and contact details

The website is not a marketing shell. It is the visible operating surface for bootstrap, trust, BIOS, root-auth, and later runtime transparency.

## SQLite as the Operational Backbone

Live operational truth lives in `runtime/cto_agent.db`. Important areas:

- Task- und Turn-System: `tasks`, `task_checkpoints`, `focus_state`, `agent_threads`, `agent_turns`, `turn_signals`, `agent_events`, `loop_incidents`
- Vertrauens- und BIOS-Naehe: `bios_dialogue`, `owner_trust`, `homepage_revisions`, `bios_uploads`
- Gedaechtnis: `memory_items`, `memory_summaries`
- Lernpfad: `learning_entries`
- Personenpfad: `person_profiles`, `person_notes`, `proactive_contact_candidates`
- Ressourcen und Skills: `resources`, `skills`
- Brain-Routing und Kosten: `brain_routing_state`, `brain_usage_events`
- Browser-/Worker-nahe Runtime: `worker_jobs` sowie Dateibruecken unter `runtime/browser-agent-bridge/`

This is not a write-and-forget diary. The runtime produces condensed summaries that are later re-injected into the working context.

## Memory and Learning Path

The agent has two levels of memory:

- `memory_items` und `memory_summaries` fuer allgemeine operative Verdichtung
- `learning_entries` fuer aktiviertes, wiederverwendbares Lernen

The learning path is hierarchical:

- `operational` for daily operating rules and run-relevant insights
- `general` for broader insights
- `negative` for things that do not work, conflicts, or risky failure patterns

Each learning entry carries, among other things:

- summary
- applicability
- confidence
- salience
- status such as `candidate` or `active`
- recall metadata

Important mechanics:

- The model can propose new learnings in bounded turns.
- The supervisor first persists them as candidates or activates them directly in review or blocking context.
- `runtime_db` builds working-memory summaries such as `learning_working_set`, `learning_operational`, `learning_general`, and `learning_negative`.
- `context_controller` pulls those high-level summaries back into later tasks.
- Relevant detail learnings are reloaded task-by-task and marked as recalled when surfaced.

This lets the agent remember what it has learned on two persistent levels: as a compact always-available condensation and as readable SQLite detail.

## People Path and Communication Memory

A dedicated people path ensures the CTO does not simply forget previous conversations.

- Interactions can automatically land in `person_profiles` and `person_notes`.
- Each person gets condensations for `conversation_memory_summary` and `notebook_summary`.
- Relevant mail previews can be pulled into context per person.
- Learnings can be fed back into the people path.
- `people_working_set` keeps a condensed high-level view available for daily operation.

That makes the people layer two-tier as well:

- immediately available compact person memory
- deeper readable notebook data in SQLite

This is intentionally important for owner, stakeholder, and team relationships.

## Proactive Contacts

The agent can interact with people proactively, but not blindly.

The current path is:

1. Der Agent erkennt aus Idle- oder Arbeitssignalen einen sinnvollen Kontaktanlass.
2. Das Modell kann einen `proactiveContactDraft` erzeugen.
3. Der Supervisor persistiert daraus einen Kandidaten in `proactive_contact_candidates`.
4. Ein interner Review-Task validiert Sinn, Konflikte und Formulierung.
5. Wenn verfuegbar, wird fuer diese Validierung taskgebunden Grosshirn verwendet.
6. Nach Freigabe erzeugt der Supervisor autonom einen Dispatch-Task.
7. The dispatch currently sends through the existing mail path and writes the result back into the people and conversation trails.

So there is no longer a mandatory human approval step, but there is still an internal safety and conflict-of-interest gate.

## Kleinhirn, Grosshirn, and Browser Vision

The brain architecture is deliberately asymmetric.

- Default-Kleinhirn ist laut Model-Policy `gpt-oss-20b`.
- Als kanonischer lokaler Vision-/Browser-Pfad ist `Qwen3.5-35B-A3B` vorgesehen.
- Lokale Upgrades duerfen empfohlen und angewendet werden, wenn der Host deutlich mehr Ressourcen hergibt.
- Grosshirn ist kein globaler Schalter, sondern ein taskgebundener temporaerer Boost.

Important routing rules:

- Local is the default.
- Grosshirn is used only when brain access exists and a boost is active for that exact task.
- The agent may request a temporary grosshirn boost itself.
- After completion, blocking, failure, or TTL expiry, routing automatically falls back to kleinhirn.
- For screenshot-heavy or UI-perception-heavy browser tasks, a vision-capable local kleinhirn should be preferred before normalizing external grosshirn consumption.

## Browser Work and Specialists

Browser work is its own architectural path, not just "the main agent clicking around blindly."

- Real browser work runs through `src/browser_engine.rs`, `src/browser_agent_bridge.rs`, `src/browser_subworkers.rs`, and `browser_agent/extension/`.
- The browser agent is a decoupled Chrome extension with its own local job bridge.
- The curated 0.8B ONNX artifact set for the extension is hosted on Hugging Face as `metricspace/Qwen3.5-0.8B-ONNX-browser-agent`.
- The core idea is transcript-first: the CTO-Agent should see compact results, not swallow raw browser traces unfiltered.
- Repeated browser work should move into reviewed capabilities, deterministic workers, or small specialist models.

The specialist pipeline is intentionally strict:

- do not train directly from raw operation traces
- keep accepted records and dataset release separate
- freeze the policy, tool, and capability surface
- only then train and evaluate
- promote only after the gates are passed

According to the pipeline, the first small browser specialist targets `Qwen3.5-0.8B` as a bootstrap goal, not browser-side WebGPU training.

## Skills

Repo-local skills live under `.agents/skills/` and are automatically rescanned for later context packages. They are a real runtime feature, not a static comment.

Currently important skills:

- `bios-interface-bootstrap`: build or rewire the BIOS surface from a stable template and SQLite base
- `browser-capability-bootstrap`: govern browser work, trust boundaries, and capability promotion
- `communication-client-bootstrap`: build, patch, or replace mail, chat, or webhook clients
- `cto-origin-history`: anchor the CTO-Agent's origin, purpose, and honest chronicle
- `host-keyboard-operations`: execute direct owner keyboard requests on the host through a reviewed skill and contract path
- `homepage-bootstrap`: shape the homepage as the first trust and communication bridge
- `owner-branding-bootstrap`: steer the safe path from terminal bootstrap to BIOS takeover and later branding
- `self-skill-bootstrap`: materialize new reusable tools as repo-local skills
- `specialist-model-pipeline`: move repeated browser work into reviewed specialists

The agent is explicitly expected not just to use reusable new capabilities ad hoc, but to anchor them as skills in the repo.

Direct host mutations from `owner_interrupt` tasks, especially keyboard or input changes, should not run through free-form prompt improvisation. That is why the repo now includes the `host-keyboard-operations` skill plus the contract `contracts/system/host-keyboard-capability-policy.json`.

## Communication Bootstrap

`src/bootstrap.rs` models a dedicated installation intake for owner and mail context.

- `install-bootstrap-tui` collects owner name, contact path, mail assignment, and free-form notes.
- From that it generates homepage, memory, and task seeds.
- With an assigned mailbox the agent can build its own communication path or patch fragile clients itself.
- The existing mail client is Python-based and intentionally easy to modify.

So the repo is already shaped for real communication autonomy, not just a later external integration.

## Context Construction

`src/context_controller.rs` assembles the working context for each bounded turn. That includes:

- mode, focus, queue depth, and checkpoints
- execution authority and loop safety
- brain access status, cost, and upgrade hints
- Owner-Kalibrierung
- learning working set and relevant detail learnings
- people working set, relevant people, notes, and mail previews
- skills, exec sessions, and raw inclusion fragments

Normal context shaping is agentic. The kernel should shrink only minimally at physical overflow boundaries.

## Important CLI Paths

Examples for local operation:

```sh
cargo run --release
cargo run -- --init-only
cargo run -- install-bootstrap-tui
cargo run -- attach
cargo run -- send "Michael Welsch: Bitte BIOS pruefen"
cargo run -- channel-interrupt email alice@example.com "Kurze Rueckfrage"
cargo run -- status
cargo run -- thread
cargo run -- signals
cargo run -- incidents
cargo run -- events
cargo run -- turns
cargo run -- run-census
cargo run -- recommend-kleinhirn
cargo run -- recommend-browser-vision-kleinhirn
cargo run -- upgrade-kleinhirn
```

Important entry files:

- `src/main.rs` for CLI entry
- `src/app.rs` for runtime initialization and web routes
- `src/supervisor.rs` for heartbeat and the bounded task loop
- `src/runtime_db.rs` for SQLite schema and operational persistence
- `src/agentic.rs` for LLM orchestration

## Repo Map

- `src/app.rs`: web control plane, runtime init, HTTP routes, BIOS page integration
- `src/contracts.rs`: canonical contract models, default policies, path resolution, kleinhirn and browser-vision model selection
- `src/supervisor.rs`: heartbeat, task selection, watchdogs, reviews, delegation, dispatch, idle duties
- `src/runtime_db.rs`: SQLite schema, tasks, memory, learnings, people path, skills, events, brain routing
- `src/context_controller.rs`: working-context construction for bounded turns
- `src/brain_runtime.rs`: local model runtime, upgrade paths, grosshirn preparation
- `src/browser_engine.rs`: deterministic browser actions and browser status
- `src/browser_agent_bridge.rs`: local job bridge between the Rust supervisor and browser agent
- `src/browser_subworkers.rs`: browser worker lifecycle
- `src/bootstrap.rs`: installation bootstrap, terminal bridge, input normalization
- `src/attach.rs`: attach socket and TUI
- `.agents/skills/`: repo-local skills
- `contracts/`: constitutional truth
- `contracts/history/`: origin and honest chronicle
- `browser_agent/`: decoupled browser agent
- `scripts/`: operational helper scripts such as the mail client and browser setup

## Design Core in One Sentence

The CTO-Agent is not a single prompt but a constitution-bound, SQLite-backed, bounded-working CTO core with a visible BIOS surface, task-bound brain routing, self-anchoring skills, an active learning path, and a growing network of browser, communication, and specialist paths.
