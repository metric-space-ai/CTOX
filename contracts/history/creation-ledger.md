# Creation Ledger

## 2026-03-23
- CTOX was refactored so its conversational loop can run as a detached local background service started with `ctox start` and stopped with `ctox stop`.
- The TUI was changed from owning the loop itself to attaching to the running service, with loop control exposed in Settings.
- Linux installs now provision a persistent `systemd --user` `ctox.service`, and `ctox start` or `ctox stop` now map to enable-and-start vs stop-and-disable semantics on hosts where that unit is installed.
- CTOX now keeps a parallel local sidecar set alive beside the chat backend: one embeddings server, one STT server, one TTS server, plus the local responses proxy.
- CTOX now has a first-class `schedule` channel: cron-style recurring jobs live in SQLite, emit normal inbound messages, and are routed through the same serial coil and service loop as TUI, Jami, and email.
- Shared auxiliary GPUs are no longer treated as all-or-nothing for chat inference: CTOX now reduces chat layer placement on GPUs that also host embedding, STT, or TTS backends, and drops out of even NCCL sharding when that asymmetric placement is required.
- The short-lived Qwen3-TTS `vllm-omni` detour was removed again; CTOX is back to a single `mistralrs` / `ctox-vllm-serve` runtime path for chat, embeddings, STT, and the current native TTS backend.

## 2026-03-24
- CTOX gained explicit `plan` and `follow-up` tools around Codex: longer owner requests can now be drafted into compact plan artifacts, optionally persisted as tracked steps, and evaluated at turn end without changing the Codex core.
- Plan steps now carry explicit lifecycle state (`pending`, `queued`, `completed`, `blocked`, `failed`) so blocked or failed long-running work is no longer hidden inside transient chat output.
- CTOX now ships a bundled `plan-orchestrator` system skill with explicit contracts for when long-running work must be persisted instead of handled as an in-memory todo list.
- CTOX now also exposes an explicit `queue` tool and `queue-orchestrator` skill: Codex can read and edit the durable inbound task queue itself instead of relying on hidden wrapper logic for follow-up work.
- The native `mistralrs` TTS path was switched from the temporary Dia backend to `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, and the chat launcher now prefers live per-GPU auxiliary VRAM measurements when rebalancing LLM layer placement on shared GPUs.

## 2026-03-25
- CTOX gained eight bundled admin system skills for infrastructure discovery, reliability work, controlled change execution, security posture review, recovery assurance, incident response, automation engineering, and operations insight.
- These admin skills were integrated as additional skill-layer operating knowledge on top of the existing CTOX queue, plan, schedule, follow-up, and service-loop substrate rather than as a second autonomous execution loop.
- The host-observability side of CTOX's admin surface is now anchored in concrete local tooling such as `htop`, `btop`, `top`, `ps`, `vmstat`, `iostat`, `ss`, `journalctl`, `systemctl`, `df`, `du`, `curl`, and `nvidia-smi`.
- The eight admin skills now share one SQLite evidence kernel (`discovery_run`, `discovery_capture`, `discovery_entity`, `discovery_relation`, `discovery_evidence`) keyed by `skill_key`, so discovery, reliability, incident, change, security, recovery, automation, and insight runs persist into one interoperable local source of truth.
- The bundled admin helper scripts are now explicitly open, inspectable resources for Codex rather than hidden black-box function calls; each skill can reuse, patch, or bypass them while still writing back into the same SQLite layer.
- CTOX now also has a canonical ops-template skill that defines the shared invariants, section layout, and refinement escalation ladder for the eight ops skills, so later refinement can adapt helpers and editable sectors without silently breaking family consistency or the shared SQLite kernel.
- The ops skill family now carries an explicit operator-feedback contract: every user-facing answer must say whether the result is proposed, prepared, executed, or blocked, and must separate internal persistence details from the operator-visible outcome.
- CTOX now also injects a dedicated `queue-cleanup` system-skill turn when the in-memory service queue crosses a pressure threshold, so the next execution slice can diagnose and contain queue floods before normal work continues.
- CTOX mail routing now treats only the configured owner email address as an instruction-bearing inbound email source; non-owner email is blocked from the execution queue, and the service now syncs configured email inboxes directly into the shared inbound path.

## 2026-03-26
- CTOX auxiliary model policy now distinguishes explicit `[GPU]` and `[CPU]` variants in the TUI and proxy configuration instead of assuming every embedding, STT, and TTS sidecar has CUDA available.
- The embedding sidecar can now keep `Qwen/Qwen3-Embedding-0.6B` on a CPU-only mistralrs path when a host has no GPU.
- The CPU STT fallback now switches to a multilingual `faster-whisper` path, and the CPU TTS fallback is now standardized on Speaches-backed Piper voices so CTOX installs without GPUs can still transcribe and synthesize speech locally in languages such as German and French without carrying multiple CPU TTS families.
- Owner-facing communication is now constrained more tightly: email replies on existing threads must preserve the thread subject, and CTOX should not promise future high-impact work without also creating an explicit durable follow-up or review record in queue or plan state.
- Owner-visible turns now carry a runtime follow-up safety net: if CTOX leaves a task explicitly open or a turn fails before closure, CTOX creates a durable queue continuation/recovery task instead of silently dropping continuity.
- The ops skill family now hardens the same operator-facing contract across discovery, reliability, incident, change, security, recovery, automation, and insight: every reply must separate `proposed`, `prepared`, `executed`, and `blocked`, and open multi-step work must point to durable next-work state instead of vague promises.
- Owner communication now explicitly depends on both thread continuity and the recent owner-relevant cross-channel communication state, so email replies should no longer act as if only the latest inbound line exists.
- Blocked owner-visible work now has to name the exact missing inputs or approvals, tell the owner how to unblock it, and gets an automatic recurring review schedule instead of relying on long-running waits or hopeful timeouts.
- CTOX now has the first delivery-family layer alongside the ops-family layer: a canonical delivery template plus `service-deployment` and `secret-management`, so local installs can be treated as bounded deployment work with credential classification instead of being mistaken for external integrations or vague blocker mail.
- The synchronous continuity refresh was removed from the pre-reply critical path of chat turns; CTOX now answers first and refreshes continuity best-effort afterward so long-running delivery turns do not appear hung before they can report completion.
- Email authorization is now role-based instead of owner-only: the configured domain may contact CTOX for support and account-management work, explicit admin mail profiles may authorize admin work, and sudo authority is tracked separately from plain admin rights; secret-bearing input still stays TUI-only.
- 2026-03-26: The delivery family now treats verification as a separate layered acceptance step instead of conflating “service started” with “deployment succeeded”; deployments that only reach a web UI or listener but fail authenticated/admin verification must stay `needs_repair` or `blocked` until the higher verification layer passes.
- CTOX chat turns now cap how much raw recent conversation is rendered into a live prompt and explicitly prefer compact recent summaries plus the newest unsummarized items, so long operator histories no longer bloat Codex turns into avoidable timeouts.
- CTOX runtime failures such as a `codex-exec` timeout are now surfaced back to the operator as a shaped `blocked` assistant reply and immediately paired with a durable recovery follow-up, instead of leaving the turn silent or leaking raw runtime event streams into user-facing status.
- CTOX now distinguishes a normal base chat model from an optional stronger boost model; the proxy can grant temporary boost leases with explicit reasons and automatic expiry, and the TUI now surfaces the base model, active model, and boost state instead of hiding model escalation.
- Owner communication now treats historical context as an explicit capability rather than a prompt dump: inbound mail wrappers point CTOX to active communication search tools (`ctox channel history`, `ctox channel search`, `ctox lcm-grep`), and the communication skill family now requires reconstructing relevant prior state before replying.
