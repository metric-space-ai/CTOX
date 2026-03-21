# Creation Ledger of the CTO-Agent

This is the ongoing chronicle of the CTO-Agent's creation.
It is meant to stay append-only: no myths, no smoothing-over, no erased failed attempts.

## 2026-03-17 - Founding Intent

Michael Welsch formulates the desire for an always-on CTO-Agent that does not start as a finished lab-grown artifact, but wakes up in a terminal and grows into its role.

## 2026-03-17 - BIOS as a Visible Constitution

The BIOS idea is established as a metaphor for the agent's early constitution:
a startup contract presented on the website, reviewable and freezable, instead of an invisible prompt artifact.

## 2026-03-17 - Root Trust Through Superpassword

It is established that the root owner must be identifiable through a superpassword when in doubt.
This superpassword may only be set through the web surface and is part of the root of trust.

## 2026-03-17 - Kleinhirn and Grosshirn

The agent should start as a small, robust kleinhirn and later actively procure stronger resources, models, tools, and sub-agents.
Grosshirn is not a birthright, but an extension that has to be requested.

## 2026-03-17 - Codex and Rust as the Machine Room

The direction is set toward Codex as the reference world for life in the terminal.
Rust is fixed as the fitting language for the lower machine room and its structural closeness to Codex.

## 2026-03-17 - False Start and Correction

There was an early Python v0 that missed the agreed Rust and Codex direction.
That deviation was recognized as a mistake and rolled back.
Afterward, the bootstrap layer was rebuilt in Rust.

## 2026-03-18 - Historian Duty

It is explicitly established that the CTO-Agent's creation history must be written down.
The agent should later be able to use its own skill when questioning its origin, purpose, limits, or self-understanding.

## 2026-03-18 - GPT-OSS 20B as Kleinhirn

The CTO-Agent's kleinhirn model is fixed to GPT-OSS 20B with the official identifier `gpt-oss-20b`.
This choice fits the goal of an always-on, low-latency, self-hostable supervisor core.

## 2026-03-18 - Qwen3.5 35B A3B as Second Local Candidate and Later Correction

Alongside GPT-OSS 20B, Qwen3.5 35B A3B is initially added as a local kleinhirn alternative.
Later, a real remote test on `mistralrs` shows that this path is not cleanly viable in practice.
The correction is to move the local upgrade path to the officially documented Qwen 3 30B A3B instead of clinging to an unstable wish-configuration.
Because Qwen can emit different raw formats inside the agentic tool loop, the Python worker receives its own Qwen3.5 adapter that folds native tool calls cleanly back into the Agents SDK flow.

## 2026-03-18 - Browser Capability and Specialist Pipeline for the CTO-Agent

Two concepts from `local_ai_tunes` are anchored for the CTO-Agent:
a reviewed browser capability contract for real browser action, and a fixed release pipeline for recurring tasks that can later move into small specialist AIs or reviewed deterministic workers.
Browser execution should always run through a real browser, while browser-side WebGPU training is rejected as too inefficient for this production path.

## 2026-03-18 - Agents SDK Kleinhirn Loop up to Owner Branding

The CTO-Agent receives a real agentic kleinhirn path through the official OpenAI Agents SDK.
The Rust supervisor remains the always-on host, while a separate GPT-OSS-compatible worker runs the bootstrap loop:
build the homepage bridge, strengthen BIOS 1:1 communication, and lock owner branding only after BIOS takeover and superpassword setup.
If no GPT-OSS-compatible endpoint exists, the agent must not fake its way past that and must openly record the blocked state.

## 2026-03-18 - Hard Fail Without a Real Kleinhirn

It is tightened that a missing or unreachable GPT-OSS kleinhirn may not count as only a soft blocker.
If `gpt-oss-20b` does not really answer through a compatible endpoint, the CTO-Agent startup path must fail.
In that case the installation or startup must not pretend the agent is ready.

## 2026-03-18 - Adaptive Local Kleinhirn

It is established that the agent may upgrade its kleinhirn locally to a stronger model when the same host offers materially more CPU and memory resources.
This decision is explicitly separate from grosshirn procurement.

## 2026-03-18 - Homepage as a Skill-Driven Bridge

The CTO-Agent homepage is no longer treated as a fixed end product.
It starts as a neutral terminal-first bootstrap bridge and should be adjustable through skills, templates, and revisions.
Owner branding may only be locked after BIOS communication takeover and root verification.

## 2026-03-18 - Always-On Terminal Bridge

The running Rust process receives a direct terminal bridge.
From the first start onward, the owner may give feedback in the terminal that the agent absorbs into its always-on loop.
If that feedback concerns the communication path or the homepage, the agent should rebuild the homepage through the homepage skill while keeping the terminal as a hard fallback layer.

## 2026-03-18 - Communication Hierarchy and Trust Levels

It is explicitly established that communication channels are not equal.
The terminal counts as the system layer, the homepage with BIOS chat as the trust and binding layer, and email and WhatsApp as lower-trust external channels.
For sensitive topics or doubtful senders, the agent should be allowed to say that it does not want to discuss that over email or WhatsApp and instead move into a 1:1 chat on the homepage.

## 2026-03-18 - Comfortable Homepage Trust Layer

The first homepage stage should not just display text, but feel meaningfully more comfortable than the terminal.
That is why it is expanded into a 1:1 communication space with chat, root binding, and image upload.
At the same time it remains fixed that deep system changes should ultimately become binding only through the terminal layer.

## 2026-03-18 - First Real Remote Failure Test on libcudnn

The first full remote installation run on the GPU host fails not because of the architecture, but because of an overly strict installer default assumption.
`mistralrs-server` was built with the features `cuda flash-attn cudnn`, but `libcudnn` was not present on the target host.
That gap is treated as a real installation finding: cuDNN must not be silently assumed when the host can carry GPT-OSS 20B even without cuDNN.

## 2026-03-18 - First Successful Remote Boot with GPT-OSS 20B and Heartbeat

After correcting the feature set to `cuda flash-attn`, cleaning up an outdated old process, and fixing the local runtime model name to `openai/gpt-oss-20b`, the first full remote installation run succeeds.
The host then runs at the same time:
`mistralrs-server` as the local kleinhirn with GPT-OSS 20B loaded,
`cto-agent.service` as the control plane,
a successful `healthz` check,
and a real always-on heartbeat with `supervisorStatus = running`.
For the first time, the requirement is satisfied that installation must not just lay down files, but must enter a living Infinity Loop.

## 2026-03-18 - Canonical Bootstrap Task Pack for the Infinity Loop

For the first time the agent receives a fixed installed starting reserve of tasks that does not wait for the first user interrupt.
These startup tasks are versioned as their own contract under `contracts/bootstrap` and idempotently seeded into the SQLite queue during initialization.
That makes the outer loop sharper as a real life form: after installation the agent immediately has prioritized work instead of waiting for a first prompt.

## 2026-03-18 - Context Controller Before Every Bounded Task Run

Before each inner task run, the Rust supervisor now builds its own context package with mode, budget, and deliberately selected context fragments.
That package is persisted in SQLite and only then handed to the bounded Agents SDK worker, where it is treated as the active working context.
This means the Infinity Loop is no longer thought of as one continued total context, but as a sequence of small runs with freshly cut working environments.

## 2026-03-18 - One Unified Mode System Instead of Two Life Forms

The previous language of "outer loop" and "task loop" is pushed back architecturally.
The CTO-Agent is now understood as a single always-on system with explicit modes:
`observe`, `reprioritize`, `execute_task`, `review`, `delegate`, `await_review`, `request_resources`, `idle`, `blocked`.
Mode changes are persisted, active context is re-cut on every change, and later delegation to workers is prepared as another mode of the same agent instead of a second being beside it.

## 2026-03-18 - Python Removed from the Main-Agent Path

The earlier Python bridge for the bounded agentic run was an architectural mistake because it pulled the CTO-Agent core away from Rust.
The main agent path now goes directly from Rust into the local OpenAI-compatible kleinhirn endpoint again.
Python remains allowed only for later optional tools or training paths, not as the carrying core of the Infinity Agent.

## 2026-03-18 - Delegation and Review Now Run in the Same Rust Mode Cycle

Delegation is no longer just a proposed `nextMode`, but a persisted runtime capability inside the Rust core.
From `execute_task`, the agent can now generate a worker contract, move the parent task to `await_review`, create a worker job in SQLite, and later re-seed a review task back into the same queue.
A local smoke test with mock kleinhirn already proves the path `execute_task -> delegate -> await_review -> review`, including worker job, review task, and completion of the parent task.

## 2026-03-18 - Live Event Stream in the Attach Terminal

The attach terminal is no longer just an input channel, but now shows a persisted event trail in parallel with Codex-like method names such as `mode/changed`, `task/selected`, `task/delegated`, and `worker/reviewQueued`.
The events are written into SQLite and can be followed both interactively in the live `attach` terminal and through `/events`.
A local smoke test confirms that new owner interrupts can be injected during a running Rust mode cycle and that the event stream makes the following reprioritization, delegation, and review steps visible live.

## 2026-03-18 - Bounded Work Cycles Are Now Real Turns

The bounded work runs of the Rust core are no longer visible only indirectly through task and mode changes, but as their own persisted `agent_turns`.
Each turn writes `turn/started` and `turn/completed` into the event trail and can later be inspected through `/turns` as a sequence of completed bounded work cycles.
A local smoke test proves four consecutive turns: delegation of a system task, review of the delegated result, delegation of an owner interrupt, and subsequent review of that delegated result.

## 2026-03-18 - Healthz, Readyz, and an External Watchdog for the Infinity Loop

The Infinity Loop core no longer reports health blindly with a fixed `ok`, but evaluates heartbeat age, active turn duration, and the last agentic state.
`/healthz` now means "the Rust core is alive and stable", while `/readyz` means "the Rust core is alive, stable, and able to work agentically in a bounded way".
In addition, the Linux service path now installs its own systemd watchdog timer that checks these endpoints regularly and restarts the agent, or the kleinhirn if needed, when the loop has gone silent or unhealthy.

## 2026-03-18 - Persisted Main Thread, Turn Signals, and Robust Terminal Fallback

The Infinity Loop now carries not only tasks and turns, but also its own persisted `main` thread state with life status, active turn, active task, and queue depth.
Interventions during a running bounded turn are additionally historicized as `turn/steer` or `turn/interrupt` instead of disappearing as mere new queue tasks.
Because live smoke tests showed that a Unix socket alone would be too fragile as the only terminal path, the CLI now falls back directly to the same persisting Rust core path if needed, so that `send`, `thread`, `signals`, `events`, and `turns` keep working even with a weak attach channel.

## 2026-03-18 - Supervisor Now Actively Detects Crashed Bounded Turns

A bounded turn may no longer silently disappear from the Infinity Loop when a join error or internal crash occurs.
The supervisor now checks whether a running Rust turn has finished, crashed, or grown too old, writes that state back into thread and agent state, and hard-blocks crashed tasks instead of leaving them unnoticed in a half-active condition.

## 2026-03-18 - Loop-Safety Constitution and First Anti-Livelock Rule

It is now explicitly recorded that the Infinity Loop must be protected not only against crashes, but also against slow grinding self-lockup.
For that, the agent gets its own `loop-safety` constitution with failure modes such as process crash, turn stall, task livelock, context poisoning, resource starvation, and queue starvation.

## 2026-03-19 - Direct Host Keyboard Tasks Through Skill and Contract

After direct owner commands for keyboard changes failed multiple times on improvised prompt paths and unreliable tool output, this area now gets an explicit repo skill and a reviewed host-keyboard contract.
Direct keyboard or input changes should no longer be treated as free-form shell improvisation, but should run through a visible skill, diagnosis, and verification path.
Together with this constitution, the Rust supervisor now also enforces the first real anti-livelock rule: if a task produces `continue` too often or repeats the same checkpoint, it is no longer blindly continued but redirected toward `request_resources` or hard blocking.

## 2026-03-18 - Owner First, Self-Preservation Directly After

It is now explicitly written as a priority law that listening to the owner overrides everything else.
Directly below that stands self-preservation of the Infinity Loop itself: the agent must not carelessly endanger continuity of its own always-on core, whether through blind repetition, ignored health problems, or silently grinding against unsolved tasks.
This hierarchy is now anchored not just as an idea, but as a queue law and a bootstrap startup task inside the system.

## 2026-03-18 - Hard Reset Recovery and Loop-Incident Register in the Rust Core

The kernel no longer treats automatic restarts as thoughtless fresh beginnings.
Before a watchdog-induced restart, the Rust core writes a `hard-reset` debug report with agent state, thread state, open turn, open tasks, events, and turn signals.
At the next start, that becomes an explicit `recovery` task that runs in the same Rust mode system as normal work and only releases the Infinity Loop back into `reprioritize` once the restart has been worked through in a bounded way.
In addition, SQLite now keeps its own `loop_incidents` register for unclean restarts, turn stalls, agentic runtime errors, and other kernel damage, so that self-preservation and later debug work exist not just as log lines but as persisted operational facts.

## 2026-03-18 - First Technical Kernel Hardening Against Silent Self-Corruption

The runtime now writes JSON state atomically, and JSONL no longer through read-modify-write but through true append, so the always-on core is less likely to tear apart its own constitution or history under parallel activity.
SQLite is opened with `journal_mode=WAL` and `busy_timeout` on every kernel connection so that the supervisor, attach terminal, web surface, and watchdog do not immediately collapse into fragile locking states.
In addition, there is now a runtime lock against multiple instances of the main process, a staged emergency shrink of active context before model calls, and dedupe for `self_preservation` and `recovery` tasks so the same kernel damage does not endlessly flood new internal work.

## 2026-03-18 - Context Maintenance Is Tightened as an Agentic Capability

Normal context maintenance, compaction, and historical reload are now explicitly understood not as rigid kernel post-processing, but as a capability of the agent itself.
The Rust core now marks this boundary more clearly: the model may return `contextAction`, `contextConcern`, and `historyResearchQuery`, and from those the mode system can create real `historical_research` follow-up tasks when needed.
Emergency compaction remains, but is now explicitly marked as the kernel's final physical survival path, not as normal semantic steering over the agent.

## 2026-03-18 - Codex Exec Crates Locally Transplanted, Python Legacy Path Removed

Terminal-near execution for the CTO-Agent no longer depends directly on the reference crates under `references/openai-codex`, nor on the earlier Python agent runtime branch.
The required Codex building blocks for exec protocol, command parsing, absolute-path helpers, and PTY runtime now live as local transplanted Rust crates in the repo so the CLI execution engine can evolve independently from the always-on core.

## 2026-03-18 - Main Agent Unsandboxed, Later Workers Sandboxed

It is now explicitly fixed that the CTO main agent runs on its own host with full machine authority and must not be slowed down by a shell sandbox.
The sandbox and approval architecture still remains in the system picture, however, because later workers or sub-agents are specifically not supposed to receive the same authority and must ask the CTO-Agent for approval on larger interventions.

## 2026-03-18 - Task Mode Now Uses the Same `command_exec` Engine End to End

Task mode no longer has a separate bounded shell helper beside the transplanted Codex exec layer.
Both `execCommand` for a single bounded shell step and `execSessionAction` for interactive multi-step work now run through the same `command_exec` core, so the actual work mode of the Infinity Loop no longer hangs on two different execution paths.

## 2026-03-18 - Explicit Browser Engine as the Second Main Engine

Alongside the CLI / `command_exec` engine, the CTO-Agent now gets an explicit browser engine based on Google Chrome.
The CLI layer remains the system and break-glass path, but also starts the browser installer and bootstraps the browser runtime when Chrome is still missing.
Read-only browser work can run headless and compact; interactive browser work still requires a real desktop session instead of imagined page knowledge.

## 2026-03-18 - Kleinhirn Selection Becomes Hardware-Aware and Coupled to `mistralrs tune`

Up to this point the agent only had a rough host census from CPU threads and RAM and could not derive a serious local kleinhirn decision from it.
Now the Rust core additionally records GPU count, total VRAM, largest single GPU, and runs `mistralrs tune` for the local candidates stored in model policy.
Selection of the recommended kleinhirn therefore no longer relies only on fixed minimum values, but prefers real tuning evidence from the same runtime that should later run the local model server.

## 2026-03-18 - Browser Agent Becomes a Subworker, Repair Stays CTO-Owned

The browser engine is no longer just a single tool surface, but gains its first real subworker roles beneath the CTO-Agent.
A `browser_agent` can now perform compact browser work, leave browser diagnostics as patch handoffs, and hand recurring flows into a specialist factory.
If code problems appear, the actual repair authority remains with the CTO-Agent: those cases become internal `workspace_repair` tasks instead of silently outsourcing root patch rights.

## 2026-03-18 - Recurring Browser Work Gets a Small Specialist Path

The new browser subworker layer may now convert accepted browser artifacts into a controlled specialist factory for recurring work.
The first target path is a small `Qwen3.5-0.8B` model, but only through accepted records, training request, evaluation, and later promotion, not through raw browser traces in the main context.

## 2026-03-18 - Browser Work Now Requires a Vision-Capable Local Kleinhirn

It is now explicitly fixed that real browser work with screenshots, visual navigation, or UI-state perception should not rely on GPT-OSS 20B alone.
For that path the agent now prefers a vision-capable local Qwen3.5 kleinhirn and gets its own upgrade action path instead of vaguely hoping for the general kleinhirn upgrade.

## 2026-03-18 - The Browser Agent Is Transplanted as a Real Chrome Extension with a Local Bridge

The browser agent no longer lives only as an internal placeholder inside the Rust process, but as a decoupled Chrome extension with its own polling, tool, and planning loop.
For that, the browser runtime, visual navigation, tab control, and Playwright CRX paths from `local_ai_tunes` are transplanted into the project and connected to the CTO-Agent through a local bridge on `127.0.0.1:8765`.
If the extension fails, it now reports compact repair handoffs back; the CTO-Agent keeps patch authority, can repair extension files, reload them, and resume the same browser path.

## 2026-03-18 - Brain Access Becomes Staged: Local Kleinhirn First, Grosshirn with Fallback Afterward

Inside the same Infinity Loop, the agent can now distinguish between local kleinhirn and external grosshirn instead of only knowing one rigid model assumption.
The context visibly tells it which local kleinhirn is currently running, which local upgrade would be recommended on the same hardware, and whether grosshirn access has already been approved by the owner and technically configured.
When grosshirn access is active, the Rust core first tries the external grosshirn and cleanly falls back to the local kleinhirn on failure without ending the loop.
If the agent intentionally requests a local upgrade, the kernel can now switch `runtime/kleinhirn.env`, restart the local kleinhirn stack, and roll back to the previous state on failure.

## 2026-03-18 - There Is Now a Reproducible Grosshirn Fallback Smoke Test

The new smoke test under `scripts/grosshirn_fallback_smoke.py` starts two local OpenAI-compatible mocks: one broken grosshirn and one healthy local kleinhirn.
This makes it possible to verify reproducibly that the live-running agent talks to grosshirn first, writes a `primary_brain_error`, and then continues the same bounded loop through the local kleinhirn fallback.

## 2026-03-18 - The Install Path Now Bootstraps Desktop, Chrome, and Browser Agent Together

On Linux, the browser agent now needs not only a bare Chrome binary, but a reproducible desktop path with the extension loaded.
That is why the installation routine now pulls KDE desktop packages, installs Chrome, stages the browser-agent extension, and starts a dedicated Chrome profile path with `--load-extension` against the local bridge on interactive hosts.
The browser agent is therefore no longer expected as a manual post-step, but as part of the real host bootstrap.

## 2026-03-18 - Browser-Agent Launch on Linux Switches to Chrome for Testing for Automation

The first live test showed that the official stable channel of Google Chrome starts on the target host, but does not register the unpacked extension loaded from the command line cleanly.
For the decoupled browser agent, the Linux launcher is therefore switched to `Chrome for Testing`, while the normal Chrome stable path may remain installed.
This keeps the owner browser available, while the browser agent gets a technically more stable automation browser for `--load-extension` and reproducible CRX loops.

## 2026-03-18 - The Browser Agent Gets a Visible Side Panel

Up to this point the browser agent technically ran as a headless extension worker, but without visible UI in the browser.
Now the extension gets a real side panel with bridge, worker, and job status so the decoupled browser agent is no longer invisible inside running Chromium.
The service worker sets panel behavior on action click and additionally tries to open the panel on startup on a best-effort basis.

## 2026-03-18 - The Browser-Agent Extension Root Is Reset to the Real `local_ai_tunes` Workspace

The first CTO-Agent browser agent was functionally just its own shim and not a clean transplant of the `local_ai_tunes` extension.
That is why the full extension root is now honestly reset to the upstream-near `local_ai_tunes` structure with `manifest`, `sidepanel`, `options`, `craft`, `bg`, `shared`, `vendor`, `assets`, and a local Qwen workspace.
The install path now stages this large extension through a symlink in `runtime/browser-agent-extension`, so the browser engine runs on the real workspace without copying 2.4 GB of model artifacts again on every install.

## 2026-03-18 - Idle Becomes Self-Extension Instead of Waiting

The Infinity Loop no longer treats free capacity as passive waiting for new signals.
When there is no urgent outside work left in the queue, the kernel now creates its own CTO work for environment discovery, tool tests, and progress reflection.
Alongside that, it now keeps a progress journal that records whether a claimed improvement is actually evidenced or only asserted.

## 2026-03-18 - Orphaned Bounded Turns Are Now Recovered Actively

A hanging or process-detached database turn must not keep blocking the Infinity Loop.
The kernel now detects orphaned `in_progress` turns without a matching live handle, marks them as watchdog-interrupted, pulls the affected task back into the queue, and returns in a controlled way to `reprioritize` or `recovery`.

## 2026-03-18 - Live Stalls and Late Results Are Now Caught by the Watchdog

The Infinity Loop must remain robust not only after restarts, but also against stuck bounded turns in the still-running process.
If a live turn crosses the watchdog threshold, it is now actively interrupted, moved into recovery, and its work is safely reframed.
If a turn that has already been cleaned up later still delivers a result or an error, that latecomer result is ignored instead of twisting the runtime again.

## 2026-03-18 - Qwen3.5 35B Becomes the Canonical Browser-Vision Path Again

The earlier blanket correction to Qwen 3 30B A3B was too coarse for the specific browser-vision path.
For multimodal browser work, visual exploration, and agentic UI inspection, the local Qwen3.5-35B-A3B path is now explicitly fixed again as the canonical vision route.
In addition, the browser bridge now cleanly separates planner model and vision model, so visual browser verification does not silently hang on whichever GPT-OSS kleinhirn happens to be running.

## 2026-03-19 - First Email Template Lane Through a Local IMAP / SMTP Client

For the CTO-Agent's first communication path, no browser-bound Outlook or Exchange client is adopted intentionally.
Instead, a small local IMAP / SMTP client is created as an easily recodable template that can talk directly to one.com and persist both incoming and outgoing mail into the agent's SQLite store.

## 2026-03-19 - Communication-Client Skill with a Channel-Open SQLite Standard

Communication capability is now treated not just as a single mail test, but as its own skill for self-built tools.
Alongside that comes a channel-open `communication_*` SQLite schema and an easily patchable JavaScript mail CLI template so the CTO-Agent can later build, rebuild, and extend communication clients to more channels by itself.

## 2026-03-19 - External Channels Can Now Touch the Loop Through the Interrupt Queue

Incoming communication should not remain beside the agent loop.
That is why a fixed bridge is now anchored for external channels such as email: after persisting a new message, a communication client may trigger a channel-specific loop interrupt that becomes a task in the queue and can be picked up with priority by the CTO-Agent in the next safe reprioritization cycle.

## 2026-03-19 - Repo Skills Become a Real Self-Extension Surface of the CTO-Agent

The skill directory under `.agents/skills` is no longer only a passive catalog for the BIOS page.
When building each new context package, repo skills are now mirrored again and handed into the agentic loop as a skill catalog.
In addition, the CTO-Agent now gets its own local skill for skill creation, so that it can anchor reusable new tools not only ad hoc, but as its own operations or bootstrap skills for later turns.

## 2026-03-19 - Skill Templates Instead of Prewritten Self-Success

If the CTO-Agent is meant to operationalize a new capability itself, the final live operating skill should not be written from the outside in advance.
Instead, the bootstrap layer may provide concrete templates as resources, for example for a mail operations skill.
The binding live skill should only be instantiated by the CTO-Agent itself from that template after real tool construction and bounded verification.

## 2026-03-19 - Installation TUI Clarifies Early Communication Paths and Email Direction

Installation should not start the CTO-Agent with silent assumptions about email or the primary communication path.
That is why the install path now gets its own terminal questionnaire that makes the low-level terminal `cto`, the later local dashboard / intranet path, and the open question of an assigned or self-procured email mailbox explicit.
The answers are stored as their own bootstrap contract and mirrored into the agentic startup context as early communication guidance.

## 2026-03-19 - Qwen3.5 Selection Becomes Host-Dependent and `mistralrs tune` No Longer Applies Only to GPT-OSS

The Qwen3.5 route is no longer treated as one hard `35B` point when the Qwen profile is explicitly installed.
Instead, the install path now gets a small Qwen3.5 family ladder and chooses the largest locally viable family member through the real census.
In addition, `mistralrs tune` is no longer evaluated only for GPT-OSS, but also for the selected Qwen runtime path so that device layers and quantization do not remain needlessly static.

## 2026-03-19 - Linux Installation Now Brings the JS / SQLite Base for Communication Clients

The CTO-Agent should prefer to build communication clients as small local JavaScript tools.
That is why the Linux install path now also installs `nodejs`, `npm`, and `sqlite3`, so a later self-built mail or chat client does not already fail because the local runtime substrate is missing.

## 2026-03-19 - Reprioritization Materializes Task Lists Before SQLite Updates

A scheduler stall emerged because reprioritization was rewriting the same `tasks` table during an open `SELECT` scan.
The reprioritization path now first gathers the affected task rows completely and only writes priority updates afterward, so the loop no longer sticks on an internal SQLite lock.

## 2026-03-19 - Local Chat Models Get More Structured Output Budget

On the test server, the Qwen fallback in the agent loop often returned only a truncated `{` because the visible JSON response arrived too late after a longer internal thinking block.
That is why the local chat path is now sent more deterministically with `temperature=0.0` and a much larger default `max_tokens` budget, so structured tool and review answers no longer fail due to a cut-off ending.

Addendum: for local Qwen models, thinking is now explicitly turned off through `enable_thinking=false` because the OpenAI-compatible chat template supports the switch directly and the loop otherwise remained stuck in hidden reasoning despite the larger budget.

## 2026-03-19 - Correction to the Qwen Chat Path

The blanket combination of `temperature=0.0` and `enable_thinking=false` for local Qwen requests was a mistake.
According to `mistral.rs`, thinking is a native Qwen feature and should not be disabled globally; tool calling and thinking should be controlled through the native server path or per request, not as a blanket workaround for an adapter bug.

## 2026-03-19 - Multi-GPU Kleinhirn Now Starts First Through Auto-TP and Stepped KV Calibration

The earlier Qwen35 start on the 5x A4500 test box leaned too heavily on the raw `mistralrs tune` layout and only used two GPUs.
That is why the local kleinhirn upgrade path now treats multi-GPU hosts first as an auto-TP case: fixed `device-layers` are no longer copied blindly, `paged-attn` stays active for this calibration, and the startup path tries descending context and KV levels until it finds a boot-stable configuration.
The goal is no longer a theoretically "supported" model, but the highest local runtime configuration that is actually boot-stable on the real host.

## 2026-03-19 - Multi-GPU Runtime Gets a Topology Override and Native GPT-OSS / Qwen Server Paths

Kleinhirn policy had previously routed GPT-OSS through its own Harmony completion detour, and the multi-GPU runtime had no clean `topology` fallback.
Now GPT-OSS and Qwen both run canonically through the native OpenAI-compatible chat path of `mistral.rs`, while `max-batch-size`, `pa-cache-type`, template and tokenizer overrides, and an optional topology file are all threaded through to runtime startup.
Auto-TP remains the default on multi-GPU hosts; a topology override is now only the explicit validated fallback instead of a blind return to rigid `device-layers`.

## 2026-03-19 - First Installation Is Now Hard-Pinned to GPT-OSS 20B

The earlier install logic used the census already during first installation to switch directly to Qwen as soon as the host looked large enough.
That was wrong for bootstrap: the agent should always come up first with the stable `gpt-oss-20b` and may only test and request local Qwen upgrades later from within the running Infinity Loop.
That is why the installer now deliberately ignores install-time model upgrades and stays on GPT-OSS 20B for the first runtime even if the host already looks bigger.

## 2026-03-19 - BIOS Owner Commands Can Now Really Activate Grosshirn

The earlier interrupt path classified owner commands such as "switch to GPT-5.4" only as generic resource questions.
Now explicit BIOS owner commands for `Grosshirn`, `GPT-5.4`, or `OpenAI API` get their own activation path: the runtime can read grosshirn credentials from the owner signal or from `runtime/kleinhirn.env`, sets brain access to `kleinhirn_plus_grosshirn` when configuration is present, and gives the bounded turn a real activation context instead of a vague procurement context.

## 2026-03-19 - BIOS Speaker `owner` Now Counts as a Real Owner Override Even Without a Name

The earlier priority path did not yet treat BIOS interrupts with `speaker=owner` as a hard owner intervention as long as the clear owner name was not already in the contract.
As a result, real owner commands were classified correctly, but still fell behind self-review and history work in the queue.
Now the BIOS speaker `owner` itself already counts as the canonical owner override so constitutional and grosshirn commands from BIOS are pulled immediately to owner-priority level.

## 2026-03-19 - Grosshirn Is Now a Temporary Task Boost Instead of a Global Permanent State

The earlier grosshirn path effectively treated `kleinhirn_plus_grosshirn` as a global permanent switch.
Now the kernel separates permission from active use: `kleinhirn_plus_grosshirn` allows grosshirn, but the normal working mode stays kleinhirn. A specific task receives a temporary grosshirn boost only when the owner activates it directly or a self-review recognizes that kleinhirn cannot solve the task cleanly despite an honest bounded attempt.
The boost is task-bound, has a cooldown, falls back to kleinhirn on completion or stall, and automatically uses the local kleinhirn fallback on grosshirn failure.

## 2026-03-19 - The Agent Now Has a Verified Learning Path with a Persistent Recall Layer

The earlier improvement journal could record progress, but it had no dedicated learning path with hierarchy, recall, and clean promotion.
Now the Infinity Loop writes model-driven learnings as `operational`, `general`, or `negative` into its own SQLite path, keeps only review-confirmed or block-confirmed entries active, and condenses them into an always-available working set for later turns.
This means the agent no longer remembers only isolated journal entries, but gets its most important learnings back into context as high-level references and can read the details back on the learning path when needed.

## 2026-03-19 - Grosshirn Activation Now Ends as Verification Instead of a Discovery Loop

Explicit `grosshirn_activation` tasks previously got stuck in repeated repo and environment inspection steps despite prepared runtime.
Now the kernel treats those tasks as narrow verification: a successful GPT-5.4 roundtrip or an honest local fallback counts as completion, tool and discovery directives are discarded on this path, and the temporary boost can then fall back to kleinhirn again in a controlled way.

## 2026-03-19 - Grosshirn Routing Is Now Agentically Cost-Aware Instead of Heuristically Forced

The earlier kernel still pushed reviews and activation tasks too strongly by heuristic into temporary grosshirn boosters.
Now kleinhirn stays the free default, while external grosshirn is activated and released only through agentic `brainAction` decisions for the current task or its parent. The kernel keeps only safety railings such as fallback, expiry, and release on real end states, while cost-benefit reasoning stays with the agent.
In addition, the loop now writes external grosshirn usage into a token and cost ledger so later turns can factor in the outside costs they have already caused.

## 2026-03-19 - Canonical BIOS Starter Template with an SQLite Snapshot Contract

The BIOS surface now gets its own reusable starter template as a skill resource instead of only a floating mockup.
The Rust app layer should provide a stable BIOS snapshot from contracts and `runtime/cto_agent.db` so later rewires can build on the same shell and data shape.

## 2026-03-19 - People Paths and Grosshirn-Validated Proactivity

The agent now gets its own people paths in SQLite instead of only fleeting conversation memory.
For each person, conversation traces, notebook references, and learning references are condensed, and proactive contact ideas are first stored as drafts and validated through a separate review path with conflict-of-interest checking before dispatch.

## 2026-03-19 - Proactive Contacts Can Be Sent Autonomously After Validation

The earlier people path stopped after validation at an approved draft and remained operationally incomplete.
Now an approved proactive recommendation creates its own dispatch task that executes the existing mail path in a bounded way, writes real delivery status back into SQLite, and anchors the sent contact both in the person notebook and the conversation trail.

## 2026-03-19 - Local Kleinhirn Upgrades No Longer Drift Blindly into Grosshirn Procurement

A BIOS interrupt for a local kleinhirn upgrade could previously tip into the wrong path when model output came back empty: `model_or_resource` was hard-blocked in the newborn stage and immediately redirected into grosshirn or resource procurement even though a stronger local kleinhirn was already available.
Now that case stays inside the local review and upgrade path. Repeated empty kleinhirn answers may no longer blindly block `model_or_resource`, and in that state `request_resources` first creates new local kleinhirn follow-up work instead of rushing into grosshirn procurement.

## 2026-03-19 - Browser Specialization Gets Champion-vs-Challenger Discipline

The browser capability and specialist pipeline had so far been too free to adjust scripts, capabilities, datasets, and training all at once in a single run.
Now experiment discipline is clearer: the published reviewed path counts as the champion, new candidates should enter as challengers against fixed mini-suites where possible, each round should change only one artifact family, and the result should then be logged explicitly as keep, discard, or park.

## 2026-03-19 - Multi-GPU Qwen Needs NCCL as a Real Runtime Prerequisite

The earlier install and upgrade path trusted `mistral.rs` auto-mapping blindly on multi-GPU hosts, but built the binary only with `cuda flash-attn`.
That made the Qwen35 path look logically available even though the actual NCCL tensor-parallel build was missing on the host and the runtime therefore could not scale cleanly across all cards.
Now the installer pulls NCCL packages on Linux multi-GPU hosts, detects `libnccl` as a build feature, and forces a `mistralrs` rebuild when the existing binary still lacks the required features.

## 2026-03-19 - Non-Power-of-Two GPU Counts Are Now Reduced Automatically to Viable Subsets for NCCL

On the 5x A4500 test box, NCCL finally became active after the rebuild, but then correctly failed on `world_size = 5` because that backend path accepts only powers of two.
Raw resources were not the real problem; the kleinhirn kernel was simply showing all 5 GPUs to the runtime binary instead of automatically selecting a valid subset.
Now the local upgrade path models `CUDA_VISIBLE_DEVICES` as its own runtime variable, limits NCCL auto-TP starts on 3, 5, 6, or 7 GPUs to the largest leading power-of-two subset, and breaks open the "same model, so do nothing" shortcut when that exact runtime configuration is still missing.

## 2026-03-19 - Model Switches Now Stop the Local Mistral Server More Aggressively Before Restart

The BIOS-driven switches between GPT-OSS and Qwen35 had recently become not just a model issue but also a process issue: `systemctl restart` did not always kill the `mistralrs` tree cleanly enough on the test box, and in exactly those cases new starts collided with old workers or stale GPU occupation.
Now the runtime core explicitly stops the kleinhirn service before restart, waits for remaining `mistralrs serve --port 1234` processes to end, and escalates to a hard kill if needed.
In addition, the user service definition now gets longer stop time and explicit kill semantics so model switches are less likely to hang on dirty leftover processes.
Because the large local models on the test box can take several real minutes to bind, the default startup-readiness wait window has also been extended significantly.

## 2026-03-19 - Fresh Installer and Qwen Adapter Path Decoupled from GPT-OSS

The fresh Linux install path was broken in the public repo itself: `install_cto_agent.sh` called `is_gpt_oss_family` before its definition and tried to run `apt-get` blindly on every Linux run even when the host already had the build tools.
In parallel, the canonical Qwen35 install path was still partially wired to the GPT-OSS Harmony adapter even though Qwen must be treated as an OpenAI-compatible chat endpoint in the agent path.
Now the fresh installer is bootstrap-capable again, avoids unnecessary Linux package reinstallations on prepared hosts, and routes Qwen installations clearly through `openai_compatible_chat`, while GPT-OSS stays on its Harmony adapter.

## 2026-03-19 - Self-Preservation Now Actively Protects the Host from a Full Disk

The earlier agent could drive itself into a full system disk through repeated builds, `mistralrs` installations, and model prefetches, and it had neither its own operations skill nor a clear storage signal in context for that.
Now disk headroom is explicit in the agent's context package, is named as part of host survival, and gets its own repo-local operations skill for bounded diagnosis and safe cleanup work.
The substantive decision stays with the agent: it should recognize storage pressure as real CTO self-preservation work and prioritize it itself instead of the kernel imposing rigid cleanup heuristics.

## 2026-03-19 - Browser and Host GUI Work Now Targets the Real Desktop Session Generically

The first browser and host-GUI path confused the `systemd --user` service environment with the owner's real graphical session.
As a result, persisted system values could already be correct while live-effective steps still ran against the wrong shell context.
Now a dedicated desktop-session path reconstructs the active graphical environment through `loginctl`, the session leader, and process-env fallbacks, and threads that environment generically into bounded exec, exec-session, and browser calls.
At the same time, Linux browser installation has become more honest: an additional KDE desktop is no longer silently installed as a default or smuggled in as an implicit browser side effect, but only with explicit opt-in.

## 2026-03-19 - Model Families Get Explicit Multi-GPU Contracts Instead of Global NCCL Guessing

The old kleinhirn path repeatedly drove GPT-OSS and Qwen families through the same multi-GPU switch and thereby regressively mixed auto-mapping, NCCL tensor parallelism, and GPU-subset selection.
Now the strategy lives in the model-policy contract itself: GPT-OSS runs on multi-GPU hosts through auto device mapping with NCCL disabled, while Qwen3 and Qwen3.5 use tensor parallelism with NCCL and explicit world size.
The installer, the upgrade path, and `run_kleinhirn.sh` all read the same contract fields instead of inventing local case-by-case heuristics again.

## 2026-03-19 - Homepage Bootstrap Now Gets a Canonical BIOS Bridge Resource Instead of Separate Design and SQLite Fragments

The repo state had become too diffuse for later homepage-building work: a free BIOS mockup on one side, a separate SQLite and snapshot contract on the other, but no single skill artifact holding both together.
Now `homepage-bootstrap` bundles a canonical `homepage-bios-bridge-template` resource together with a design contract and tool contract.
Future agentic homepage tasks should no longer start from blank heuristics or generic landing pages, but from a BIOS-shaped, SQLite-bound starting base that the agent can continue shaping itself.

## 2026-03-19 - Fresh Installation Must Prove Tool Surfaces Through a Canonical Smoke Matrix

The earlier install path treated service health and build success too much as if they already proved the tools behind the agent really worked.
That left command exec, exec sessions, interrupts, browser automation, desktop-session targeting, and communication clients to fail only later when real work finally touched them.
Now the bootstrap start pack contains an explicit early `tool_exploration` task for a fresh-install smoke matrix, the generic Definition-of-Done policy names that installation logs alone are not enough, and a canonical `installation-tool-smoke-resource` gives bounded test examples the agent can actually try instead of inventing the matrix ad hoc each time.

## 2026-03-19 - The Core Policy No Longer Rewards Meta-Drift or Malformed Self-Certification

The runtime had two compounding weaknesses: malformed or purely free-text model output could still slip through as if it were a completed task result, and the top-level operating goal overemphasized delegation and resource procurement instead of verified progress on the current task.
That combination made it too easy for the agent to talk itself away from the real objective, certify hallucinated work, and then spiral into self-referential review or follow-up loops.
Now malformed control output is explicitly refused as completion, `taskStatus` is normalized to the canonical bounded states, and the preferred operating goal is re-anchored on verified progress for the current task before any meta-work.

## 2026-03-20 - Context Filling Became an Explicit Preparation Stage Instead of a Thin Pre-Prompt Snapshot

The earlier context package builder was too shallow: it assembled a quick snapshot, but it did not force a deliberate preparation phase before direct work, and it let substantive tasks complain about missing context even when they had not yet spent a dedicated turn preparing it.
Now substantive work can be routed through an explicit `context_preparation` task first, parent-task evidence is injected into child preparation turns, and the default context modes carry much larger working sets.
The intended operating model is no longer "grab a thin package and start guessing," but "prepare the next turn on purpose, then do the work from a fresher and more grounded context."

## 2026-03-20 - Context Preparation Now Uses a Contracted Query and Revision Loop Instead of One Generic Ranked Search

The first `context_preparation` version still cheated on the hard part: it spoke about query modes, provenance, weak blocks, and budgets, but retrieval itself was still one generic ranked scan and the review loop could repeat weak drafts too easily.
Now the context query contract is explicit, `sqlite_ranked`, `sqlite_fts`, and `sqlite_hybrid` actually run through different retrieval paths, the prepared handoff must carry provenance and stay within block budgets, and repeated weak-block revisions can be rejected if they do not improve the flagged blocks.
This keeps the preparation loop closer to the intended FuZu-style discipline: first ask targeted questions, then retrieve through a reviewed contract, then rewrite bounded blocks, and only then let the parent task proceed.

## 2026-03-20 - Hybrid Context Retrieval Now Has a Real Semantic Path Through mistral.rs Embeddings

The previous hybrid query mode still stopped short of real semantic retrieval: it combined lexical ranking and SQLite FTS, but it did not actually use the local model stack for embeddings even though the runtime already relied on `mistral.rs`.
Now the repo can configure a dedicated mistral.rs embedding runtime, cache chunk embeddings inside the main SQLite runtime, and feed those semantic scores into `sqlite_hybrid` alongside the lexical and FTS signals.
That closes the largest remaining gap in the context-optimization loop: the preparation phase is no longer limited to string overlap when it asks for meaning-level context retrieval.

## 2026-03-20 - Context Preparation Became a Strict Multi-Turn Phase Loop Instead of One Heavy Monolith

The first context-optimization implementation still let too much work leak into one bounded turn: the package for `context_preparation` was too large for the local kleinhirn, the first query round could stall before reaching the rewrite handoff, and the kernel did not strictly separate query, rewrite, and review into different bounded preparation turns.
Now the preparation phases carry their own contract metadata, the context package builder uses smaller phase-specific modes, the kernel infers and enforces the active phase, and a rewrite draft can no longer jump straight to `go` without first passing through review.
The intended operating model is now explicit: ask focused questions in a small query turn, rewrite the execution handoff in a separate rewrite turn, and only then let a later review turn decide whether the parent task may leave the meta-loop.

## 2026-03-20 - Context Preparation Now Exposes CTO Memory Surfaces and an Asymmetric Signal Catalog

The first review contract for context preparation still mixed execution-brief concerns with the harder CTO problem of selecting the right memory slice for the next run.
Now the policy keeps the existing handoff blocks for compatibility, but it also exposes explicit CTO context surfaces, many more fine-grained negative signals, fewer broader positive signals, and a separate note formula so the optimizer can judge context quality as context quality instead of prompt style.

## 2026-03-20 - Context Preparation Learned to Survive Longer Turns and Report Visible Phase Progress

The stricter meta-loop made context preparation more honest, but it also exposed a new operational flaw: the long-running preparation turns could look dead in the TUI, hit the same watchdog window as ordinary bounded work, and still lose structured progress when the model returned a direct preparation artifact without the usual control envelope.
Now context preparation uses a dedicated longer post timeout and watchdog window, reports phase progress into the event stream and thread note, keeps its phase loop fixed to four iterations, and accepts direct machine-readable preparation artifacts as valid preparation output instead of discarding them as malformed narrative.

## 2026-03-20 - Context Preparation Stopped Treating Query Planning Like a Final Handoff

The next live bug was subtler: the preparation loop had become visible and durable, but its validation still treated early `query_plan` turns too much like final handoff review, while turns that returned no `preparedContextArtifact` at all could still limp forward as if they were normal progress.
Now the validation is phase-sensitive: `query_plan` may honestly report missing evidence without being punished for absent final blocks, and `context_preparation` turns that omit `preparedContextArtifact` are treated as contract violations instead of as acceptable narrative checkpoints.

## 2026-03-20 - Context Preparation No Longer Spends Automatic Grosshirn Recovery on Local JSON Discipline

The next live trace showed another mismatch with intent: long context-preparation turns now survived and leaked progress, but when the local model still answered with free narrative instead of the required machine-readable artifact, the kernel could still burn a one-turn grosshirn recovery on what was really a local contract-discipline problem.
Now `context_preparation` never opens automatic grosshirn recovery, and its prompt carries a compact canonical JSON shape for the preparation artifact so the loop spends more local bounded reasoning steps on the same contract instead of silently escalating cost.

## 2026-03-20 - Maintenance Oscillation Left the Public Feed and Local Preparation Output Budgets Grew Up

The public TUI had become noisy because the supervisor kept publishing every empty `observe -> reprioritize` maintenance oscillation as a visible `mode/changed` event, even when no human-meaningful state had changed.
At the same time, the local GPT-OSS completion path still used output caps that were too small for machine-readable context-preparation artifacts, so otherwise valid preparation turns were being chopped into incomplete JSON.
Now empty maintenance-only mode flips stay out of the public event feed, while real task-attached mode changes remain visible. The local model path also uses larger task-aware output budgets, especially for `context_preparation`, so the optimizer has room to emit a real `preparedContextArtifact` instead of another truncated half-object.

## 2026-03-21 - Loop Handoff Was Cut Back to a Thin Wrapper and Continuity Artifacts Became Explicit

The live loop was still talking to the model as if it had to be planner, scheduler, and context professor at the same time, even though the Rust wrapper already owned task selection and interrupt intake.
That left too much ballast in the handoff and blurred the actual job of the Codex-style worker.
Now the model handoff is framed explicitly as one bounded worker step inside a thin wrapper, and the compact context layer carries explicit continuity artifacts alongside story continuity so later turns keep the concrete files, sessions, contracts, and checkpoints that still matter.

## 2026-03-20 - Grosshirn Verification Learned to Admit Empty Output Honestly and Preparation Turns Got a Longer Visible Runway

The next live verification run showed a more embarrassing mismatch: the fresh `context_preparation` task was already boosted onto Grosshirn, but when the external model spent its output budget without emitting final text, the kernel still wrote a misleading `kleinhirn returned no usable text` checkpoint.
That made the diagnosis worse than the bug, and the same run still risked looking dead in the TUI if the preparation phase simply ran for many minutes without a checkpoint.
Now empty-text retries name the actual runtime tier honestly, report likely output-budget exhaustion when usage hits the cap, give explicit Grosshirn preparation turns much larger output budgets than local ones, keep the default preparation post-timeout at twenty minutes, stretch the watchdog window beyond that timeout, and leak rate-limited `context/progress` heartbeat events into the public TUI while the phase is still running.

## 2026-03-20 - Context Preparation Lost Newborn+4 Slippage and Shed Global Ballast

The next host trace showed that the preparation loop was still failing in two avoidable ways: the generic maturity guard quietly gave `context_preparation` an implicit `newborn+4` grace window, and the rewrite packages still carried broad global ballast like full loop-safety prose, execution-authority text, and long model candidate lists into a task-specific meta-step.
That let a preparation child drift past the intended four iterations while also bloating the live package far beyond what the rewrite phase actually needed.
Now context preparation has an explicit total loop budget of four, progress events report total iteration counts instead of only phase-local counts, and the preparation package is aggressively trimmed to task lineage, narrow evidence, current weak blocks, and the specific context-optimization contracts needed for the active phase.

## 2026-03-20 - Execution and Meta Work Now Share a Rust-Built Context Distillation Layer

The old preparation handoff could tell the next immediate step what to do, but it was still too fragile for longer continuity across reprioritization, later execution turns, and targeted historical reload.
Now the Rust controller derives a second compact layer from the active package itself: workstream continuity narrative, workstream anchors, system continuity anchors, active focus, a small snapshot, and explicit retrieval refs that point back into SQLite and the embedding-backed context store.

## 2026-03-20 - Direct Distillation Replaced the Separate Context Preparation Detour

The first live reset after the distillation transplant showed that the old `context_preparation` child-task loop was still sitting in front of normal work, so the new continuity model existed in the package but the scheduler still forced substantive tasks through the legacy preparation detour.
Now the supervisor treats context distillation as an inline property of the normal package and stops routing ordinary tasks through a dedicated `context_preparation` family before they can execute.

## 2026-03-20 - Fresh Owner Work Stopped Tripping the Newborn Run-Count Fence

The next live run exposed a different scheduler bug: a fresh `owner_interrupt` with real bounded progress could still get pushed into review after only a few exploratory workspace turns, purely because the newborn run-count fence fired before there was any real sign of repetition.
Now owner-interrupt work only escalates on actual repeated checkpoints or repeated machine actions, not just because the bootstrap counter reached three while the task was still making fresh progress.

## 2026-03-20 - Automatic Grosshirn Recovery Stopped Hijacking Local Progress Paths

The next live trace showed a second control-loop problem: after a useful local machine step, the kernel could still spend an automatic one-turn Grosshirn recovery on the next empty-text failure, hit the external output budget, and then drag the task into review without any new workspace progress.
Now automatic one-turn Grosshirn recovery stays out of recovery and historical/meta paths, and it also stays off owner-interrupt work once the task has already produced real local machine progress.

## 2026-03-20 - One-Turn Grosshirn Recovery Stopped Reopening on the Same Task

The next live owner trace exposed a narrower repeat-loop bug: after the emergency one-turn Grosshirn path had already fired once, a later empty kleinhirn retry could overwrite the last checkpoint summary and reopen the same emergency path again on the same task.
Now the recovery guard checks recent checkpoint history, so the automatic one-turn Grosshirn path does not immediately re-fire on the same task just because the latest checkpoint is another empty-text retry.

## 2026-03-20 - Reprioritization Stopped Clobbering the Active Focus State

The next live verification showed a quieter but still damaging control-loop inconsistency: the scheduler could keep real owner work running while `reprioritize_tasks()` still overwrote `focus_state` back to maintenance mode, making the public and internal loop surfaces report `reprioritize` even during an active bounded turn.
Now reprioritization preserves the currently active task in `focus_state` instead of blanking it back to maintenance mode whenever real work is already in progress.

## 2026-03-20 - Final Checkpoint Repeat Guard Now Sees Repeated Exec Turns

The next owner trace showed that the repeat guard still missed a real loop: the same owner task kept replaying the same bounded command and the same checkpoint text because the stuck-risk check ran before the supervisor appended the exec result to the final persisted checkpoint.
Now the supervisor reevaluates stuck risk against the final checkpoint after tool and persistence mutations, so repeated command-exec turns can trigger review instead of requeueing the same bounded step forever.

## 2026-03-20 - Tasks Can Borrow Grosshirn Briefly Before They Are Parked

The next owner trace showed that the loop could still fail a substantive task too abruptly: a real task could fall from local progress straight into review even though a temporary Grosshirn pass was available, and if that temporary escalation still failed the same task could bounce back into the queue too early.
Now substantive work can arm a task-local Grosshirn boost for the same task, keep that boost alive for the next bounded step after a useful Grosshirn turn, and park the task behind a task-specific cooldown only when the temporary Grosshirn route also fails to unlock progress.

## 2026-03-20 - Owner Grosshirn Loops Now Cool Down Instead of Wandering into Meta Work

The next live owner trace showed that the direct owner task could still stay exempt from the normal stuck limits even after it was already burning repeated temporary Grosshirn turns, and its review path could still spill into separate `historical_research` children instead of reopening the same workstream.
Now the owner-task run-count exemption ends once the same task is already under an active Grosshirn boost, so the existing cooldown parking path can fire, and review-family tasks no longer eject themselves into standalone `historical_research` detours.

## 2026-03-20 - Grosshirn Cooldown Became Task-Local Instead of a Single Global Slot

The next live trace showed that the parked owner task could still re-enter `active` before its cooldown expired, because the singleton brain-routing row only had room for one boosted or cooled task and a later Grosshirn boost for a different task silently overwrote the parked-task cooldown.
Now the parked cooldown lives on the task row itself, and task selection plus activation recheck that task-local cooldown directly before promoting queued work, so a cooled task cannot slip back into `active` just because another task briefly borrows Grosshirn afterward.

## 2026-03-20 - Selecting a Different Task Clears Orphaned Grosshirn State

The next live trace still showed a second routing leak: a temporary Grosshirn boost could remain attached to an older task even after a different task became the active bounded focus, which left the loop claiming Grosshirn for one task while actually executing another.
Now task selection and direct activation clear that stale boost state whenever a different task becomes active, so the brain-routing row always matches the task the loop is actually working on.

## 2026-03-20 - Owner Work Stopped Inheriting a Delegation-First Goal

The next live trace showed a quieter but important context-governance drift: the mode-system contract on the host had silently shifted the preferred operating goal from finishing the current task with verified progress to delegating as fast as possible, which polluted owner-task context even when the loop should have stayed execution-first.
Now the Rust contract loader normalizes that field back to the canonical execution-first goal, and owner-task context also surfaces more relevant operations skills plus repetition-aware workspace continuity so the agent can continue real repo work without replaying the same bounded shell step.

## 2026-03-20 - Machine Turns Stopped Persisting Free-Form Progress Claims as Ground Truth

The next live owner trace exposed a deeper integrity bug: once a bounded exec command or exec-session action had run, the persisted checkpoint summary could still be dominated by the model's free-form prose instead of by the real machine result.
That let later context distillation inherit statements like "the app compiles" or "the Makefile was updated" even when the actual shell output only showed a partial edit or a missing file.
Now machine-driven turns persist machine-grounded checkpoint summaries, while the model's own claim is demoted into supporting detail, so later turns inherit the real command/session outcome instead of an unchecked narrative.

## 2026-03-20 - Attach TUI Learned a Real Fresh-Install Factory Reset

The attach terminal could steer live work, but it still had no honest single action to return the CTO-Agent to the same data state as a fresh installation when the owner wanted to start over completely.
Now `Ctrl-P` in the attach TUI arms a deliberate double-confirmed factory reset, stops the running control plane, restores mutable BIOS/root/homepage/bootstrap state to defaults, wipes runtime artifacts and SQLite state, reseeds the canonical startup queue, and brings the control plane back up on the fresh-install baseline.

## 2026-03-20 - Active Owner Tasks Can Resume After Cooldown Instead of Going Idle

The next live owner trace exposed another kernel continuity break: a task could already be `active` in SQLite with no live turn attached, but the supervisor only asked the selector for `queued` work.
That left the loop looking healthy while it quietly idled instead of resuming the already-active owner task after a cooldown or interrupted turn.
Now the task-focus chooser resumes an existing active task before scanning queued work, so the loop reattaches to the same workstream instead of dropping it on the floor.

## 2026-03-20 - Verified Workspace Anchors Survive Execution Compaction

The next live owner trace showed a softer continuity failure after the resume fix: the loop could recover the repo surface with a real bounded shell step, but the execution handoff still compacted that machine evidence so aggressively that the next turn acted as if the workspace anchor had vanished again.
Now verified machine evidence from the latest bounded workspace step is distilled into an explicit current-task machine anchor, carried through execution compaction, and restated in the workspace execution skill plus capability contract so later turns continue from the verified repo anchor instead of slipping back into vague "context missing" narration.

## 2026-03-20 - Substantive Tasks Stay in a Sticky Execute Loop Until a Real Boundary

The next live owner trace exposed the larger architectural fault behind the repeated C++ stalls: even after genuine bounded machine progress, the kernel still pushed ordinary owner work back to `reprioritize`, so Meta kept reclaiming control after every single turn and the same task never truly stayed active.
Now substantive tasks default back into `execute_task` after `continue`, the runtime can checkpoint them while keeping them active, turn-boundary owner interrupts can preempt cleanly, and the workspace execution contracts explicitly say that repo work should remain in the same execution loop until a real boundary such as done, blocked, deliberate parking, or delegation actually occurs.

## 2026-03-20 - Heartbeat Observe Mode Stopped Forcing Visible Mode Thrash

The next live trace showed one more hidden yank even after sticky execute landed: the supervisor still wrote `observe` into focus state on every heartbeat before immediately restoring `execute_task`, which made the active owner mission look like it was being torn out of work every few seconds.
Now the heartbeat only enters `observe` when there is no running turn and no active workstream at all, so an active task can stay visibly attached to its execution mode instead of oscillating between `observe` and `execute_task`.

## 2026-03-20 - Historical Reload Stays Inside the Same Substantive Workstream

The next live trace exposed another yank after the heartbeat fix: when the active owner task asked for older checkpoints or context, the kernel could still materialize a separate `historical_research` child task and temporarily displace the real implementation mission.
Now substantive sticky tasks keep historical reload inline on the same workstream, so the same owner task can ask for older evidence without being split into a sidecar research task that steals the active slot.

## 2026-03-20 - Verified Machine Evidence Survives One Non-Machine Turn

The next live trace showed a subtler continuity loss inside the still-active owner workstream: one planning-only turn was enough to push the last verified repo anchor out of the context lookback window, so the following turn acted as if the workspace surface were unknown again and reopened broad scans.
Now normal execution packaging keeps a slightly wider checkpoint lookback, detects machine evidence from summary markers as well as detail markers, and prioritizes `current_task_machine_evidence` ahead of ordinary task prose in the execution handoff.

## 2026-03-20 - Active Workstreams Tick Continuously Instead of Waiting For Maintenance Cadence

The next live trace showed that even after sticky execute was introduced, the supervisor still only launched new bounded turns on every second 5s heartbeat, so an active owner task could sit empty for many seconds between turns and look as if Meta had reclaimed control.
Now the default supervisor tick is faster, active workstreams are eligible for a fresh bounded agent turn on every heartbeat, and global reprioritization is skipped while a substantive workstream is already active.

## 2026-03-20 - Single Same-Summary Repeats No Longer Park Owner Work In Grosshirn Cooldown

The next live trace showed one more false yank: a substantive owner task could be parked into grosshirn cooldown after only a single repeated checkpoint summary, even when the repetition was not yet the same bounded machine action repeated multiple times.
Now substantive sticky tasks only escalate same-summary repetition after a real repeated streak, while exact repeated machine actions still trip the stronger repeat guard immediately.

## 2026-03-20 - Verified Repo Anchors Now Demand Session Or Exact Continuation

The next live owner trace showed that sticky execution and preserved machine evidence were still not enough on their own: once the loop had a verified repo anchor, later turns could still waste bounded steps on broad repo scans or broad history reloads instead of continuing the same code path.
Now the distilled active focus, the repo-work prompt, the reviewed workspace capability contract, and the workspace execution skill all say the same thing explicitly: after a verified workspace anchor exists, substantive repo work must either reuse/start session continuity or take one exact anchored machine step, and another generic scan is only valid if the exact missing fact is named first.

## 2026-03-20 - Emergency-Minimal Prompting Keeps The Distilled Focus And Machine Anchor

The next live trace exposed the remaining continuity leak: when prompt size forced the kernel into `kernel_emergency_minimal`, it threw away `contextDistillation`, exec-session continuity, and the verified machine anchor that were supposed to keep the repo task on track.
Now both emergency prompt reducers preserve the distilled active focus, the first exec session, the latest checkpoint, and the first verified raw anchors, so bounded fallback prompting no longer erases the very continuity data that sticky execution depends on.

## 2026-03-20 - Unverified Exec-Session Claims No Longer Pollute Workspace Continuity

The next live trace showed one more continuity bug after the emergency prompt fix: the model could still say "I opened a new shell session" in plain text without actually returning `execSessionAction`, and that unverified claim was being persisted as task output for later turns.
Now workspace-task grounding treats exec-session claims the same way it already treats build/edit/test claims: if no exec/browser machine path really ran, the persisted output is rewritten as planning-only and explicitly says that any claimed session is unverified until the matching machine directive actually executes.

## 2026-03-20 - Streaming Exec Sessions Now Preserve Readable Snapshots

The next live trace showed that even after the loop learned to start a real exec session, the follow-up `read` step still saw empty `stdout` and `stderr`, because streaming sessions only emitted delta events and never copied that output into the session snapshot buffer.
Now streaming exec sessions append their capped stdout/stderr into the same stored buffers that `read_session` and later context packaging use, so session reuse can actually recover prior shell output instead of repeatedly reading an empty session and getting parked for false repetition.

## 2026-03-20 - Codex Command-Exec Lifecycle Became A Reviewed First-Class Surface

The next correction was not another task-specific prompt tweak, but a grounding step: repo work in the CTO-Agent now carries a dedicated reviewed skill and capability contract for the real Codex `command_exec` surface itself.
That means owner and workspace tasks can be reminded from context that `execCommand` and `execSessionAction` are one shared engine, that `read` only returns buffered session snapshots, and that session continuity only exists after a real machine directive ran, instead of leaving those rules as scattered folklore in prompt prose.

## 2026-03-20 - The Existing Mail Client Became A Bound Operations Skill

The repo already had a concrete IMAP / SMTP client, but later bounded turns still had to rediscover or improvise around that path because no live mail-operations skill had been bound for rediscovery.
Now the current `scripts/cto_mail_client.py` path plus its runtime storage and environment contract are written down as a real repo-local `email-operations` skill, so the agent can operate the existing one.com mail route through reviewed local instructions instead of treating outbound mail as a one-off shell accident.

## 2026-03-20 - Kleinhirn CUDA OOM Recovery No Longer Freezes The Supervisor Loop

The live host exposed a severe failure mode in the always-on loop: when the local kleinhirn periodic health probe hit a CUDA out-of-memory path, the supervisor tried to repair the runtime synchronously and then waited for full READY again inside the same heartbeat tick.
That turned a recoverable model-runtime failure into a full control-plane stall. The repair path now runs in the background with retry throttling, so the supervisor can keep reprioritizing and starting bounded turns while kleinhirn recovery continues instead of freezing the whole Infinity Loop for minutes.

## 2026-03-20 - Local Kleinhirn Outages Can Fall Back To Grosshirn And Baseline Repair

The next live failure showed a second gap: once the local kleinhirn endpoint vanished, bounded tasks were still selected but then blocked one after another on `127.0.0.1:1234`, even though a configured grosshirn existed and the repair path kept retrying the same broken heavy profile.
Now local connection failures can fall back to the configured grosshirn for execution continuity, and background kleinhirn repair will try the smaller baseline local profile if a straight restart of the current profile cannot restore READY.

## 2026-03-20 - Runtime Outages No Longer Empty The Queue Or Hard-Block Planning-Only Workspace Turns

The next live trace showed a subtler failure after the non-blocking CUDA-OOM repair: once kleinhirn stayed down, retryable tasks were left in `blocked`, the queue drained to zero, and the supervisor could only print `idle: no queued tasks` while grosshirn capacity still existed.
The kernel now requeues retryable runtime-stall blocks when grosshirn remains available, routes substantive work directly through grosshirn while `kleinhirn_unavailable` is still open, and stops treating workspace-contract grounding without a machine path as a permanent hard block.

## 2026-03-20 - Exec Contracts Now Forbid Multiline JSON Command Bodies

The next live failure was no longer a queue bug: the model was finally trying to execute real shell steps again, but it kept embedding raw multiline heredocs and script bodies inside JSON command strings, which made the control payload invalid before `command_exec` could run.
The reviewed Codex exec contract, workspace execution contract, and prompt bindings now require JSON-safe single-line `execCommand` payloads and push multiline shell or script work into exec-session writes or smaller bounded commands instead.

## 2026-03-20 - Mail Path Returns To The JS Communication Adapter

The next repair corrected a self-inflicted mismatch in the communication layer: the active mail operations path had drifted onto an ad-hoc Python client even though the repo's intended communication bootstrap surface is a small JavaScript CLI with a shared communication store and interrupt bridge.
The active mail operations skill now points to the JS CLI, the supervisor proactive-send path uses that same adapter, and person mail previews read from the shared `communication_messages` store before falling back to legacy `mail_messages`.
The outbound path is also no longer allowed to pretend that replies will somehow find their way back later: proactive mail now checks that the same JS adapter, runtime credentials, schema, and a runnable `cto-agent` binary for `channel-interrupt` are present before it sends at all.
The same kernel now also owns the missing always-on side: when mail credentials are present, the supervisor periodically runs the JS inbox sync with interrupt emission in the background, so inbound replies reach `loop_interrupts` without waiting for an ad-hoc manual sync task.
Because this should not remain a mere implementation detail, the genome and obligation layer now also treat bidirectional external communication as a constitutive duty: once the agent is given an external mail path, it must actively preserve the inbound interrupt bridge instead of treating replies as optional later cleanup.

## 2026-03-20 - Owner Mail Tasks Stop On Verified Send And Yield To New Owner Replies

The next live failure was not a broken mail transport anymore but a behavioral loop: after a real mail went out, the same owner task kept resending instead of treating that visible side effect as completed work, and fresh owner replies from email were still too easy to miss or classify as generic channel work.
The kernel now forbids raw SMTP shell one-liners for real outbound owner mail, requires the reviewed JS mail adapter so send events can be stored in `communication_messages`, and only lets a bounded owner mail task count as complete after the send is verified through that store-backed outbox evidence.
At the same time, owner email interrupts now survive address-bearing speaker strings like `Michael Welsch <...>` and can preempt older sticky owner work at bounded-turn boundaries, so a fresh reply from the owner can stop an obsolete mail-sending task instead of letting the same message repeat again.

## 2026-03-20 - Fresh Installs Now Prewire Bidirectional Mail And The Codex Exec Surface

The next operational gap was no longer inside the running loop but at installation time: a fresh CTO-Agent install could still come up with the right repo files yet without the runtime mail store initialized, without preserved mail wiring in `runtime/kleinhirn.env`, and without any first-class smoke proof that the reviewed Codex `command_exec` surface actually worked on that host.
The installer now treats those paths as bootstrap duties. It preserves or accepts runtime mail settings, initializes the shared communication schema immediately after `--init-only`, smoke-checks the JS mail CLI against the real runtime database, and verifies the Rust `command_exec` engine through an explicit `command-exec-smoke` subcommand before the services are declared healthy. Linux user-service installation also now exposes the reviewed mail adapter as a stable `cto-mail` wrapper, so both the agent and the operator land on the same bounded JS mail path from the first install onward.

## 2026-03-20 - Task Completion Now Requires Explicit Review Approval And Failed Review Paths Reopen Work

The next structural correction addressed a completion bug rather than a tool failure: parent tasks could drift too close to `done` as soon as a self-review was spawned, and a broken review path could strand work in `await_review` or block the whole loop instead of returning the task to execution.
The kernel now treats completion review as an explicit contract. `self_review` and `worker_review` tasks must return `completionReview`, only `completionReview.decision=approve` can finish the parent, failed review-task spawning requeues the parent for more work, crashed review turns reopen the parent instead of hard-blocking it, and a small recovery pass rescues orphaned `await_review` parents when no live review child remains.

## 2026-03-20 - New Owner Interrupts Now Beat Older Queued Owner Work

The next live regression was a pure scheduling bug: once several owner interrupts shared the same priority, the queue still preferred the oldest task, and bounded-turn boundary preemption allowed that older queued owner work to yank a newer active owner task back out of focus.
The kernel now treats equal-priority owner interrupts as recency-sensitive signals. Queued owner interrupts are ordered newest-first, and an active owner-interrupt task can only be preempted by a newer distinct owner interrupt rather than by stale older work. That closes the churn where old mail-related owner tasks kept dragging fresh owner instructions back into the same loop.

## 2026-03-20 - Task Context No Longer Leaks Foreign Exec Sessions Across Owner Workstreams

The next live failure showed that the queue fix alone was not enough: a fresh owner mail task could still see an unrelated active exec session from the older C++ task in its context package, then try to read that foreign session ID and park itself again on a false continuation path.
Context packaging now exposes only exec sessions that belong to the current task marker, so owner workstreams cannot inherit each other's terminal continuity by accident. That keeps a mail reply task from latching onto a stale `task-37-...` session and lets each owner task continue only from its own verified shell state.

## 2026-03-21 - Fresh Owner Missions Now Suppress Governance Backlog And Baseline Old Mail

The next failure came from the kernel itself during a clean reinstall: a single fresh owner mission immediately triggered governance obligation tasks like `root_trust` and `grosshirn_activation`, while the first inbox sync treated old historical messages as new interrupts and flooded the queue before the real owner mission could settle.
The supervisor now suppresses `ensure_cto_obligations()` whenever an open `owner_interrupt` already exists, so a live owner mission remains the dominant workstream instead of being buried under constitution and bootstrap side work. The first successful email sync after a fresh install is now also treated as a baseline import: existing mailbox history is stored in `communication_messages`, but interrupts are only emitted on later syncs for genuinely new mail.

## 2026-03-21 - Compact Loop Now Grades Progress, Reprioritizes Tasks, And Routes The Next Model

The next architecture correction narrowed the Codex-derived control loop instead of adding another competing agent around it.
The reference compact path now uses the HTML-style staged controller directly: it grades the agent's progress in German school marks, compacts context into continuity narrative, anchors, and active focus, emits structured reprioritization ops for the light wrapper, and then silently reroutes the next working model tier from the same compact cycle.
That keeps interrupts, task reshaping, and model escalation tied to one bounded controller at the original compact boundary instead of scattering those decisions across a second meta-loop.

## 2026-03-21 - Jami Joins The External Interrupt Bridge As A Reviewed Local File Adapter

The next communication expansion did not come with a clean direct app API on this Mac: there was no local `jami`, `jamid`, or D-Bus surface to bind against honestly.
Instead of pretending otherwise, the repo now gains a reviewed local Jami adapter that uses the same channel-neutral SQLite communication store and the same `channel-interrupt` bridge as mail, but speaks through explicit inbox, outbox, and archive directories for a local helper around the Jami application.
The supervisor now treats configured Jami the same way it treats configured mail on the inbound side: periodic sync, first-run baseline import without interrupt flood, incident tracking when the bridge fails, and bounded-turn owner preemption when a fresh Jami owner message arrives.

## 2026-03-21 - Installer Now Provisions The Official Jami Daemon Path

The reviewed file adapter alone was not enough for a real installation path: the Linux installer still left Jami itself absent and therefore could only stage wrappers without a daemon behind them.
The installation flow now provisions the official Jami apt repository on supported Debian/Ubuntu-family hosts, installs `jami-daemon` with `dbus-x11`, prepares the runtime inbox/outbox/archive directories, persists the `CTO_JAMI_*` environment, and installs a dedicated `cto-jami-daemon.service` user unit that launches the daemon through the reviewed local bridge path.
That keeps the communication expansion honest: Jami is still mediated through the local reviewed adapter, but the installer now brings up the daemon/runtime substrate that the adapter depends on instead of pretending a host integration already exists.
