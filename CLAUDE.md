# CTOX Development Guide (Instructions for Claude)

## MANDATORY: Read Before Answering Anything Architectural

Before answering **any** question about CTOX's architecture, scope, what code does, what is "old" / "new" / "dead" / "expected", or before proposing refactors — you **must** first read **every** `*.md` file in the repository root:

- `README.md` — public product description, installation, feature scope
- `AGENTS.md` — internal agent architecture (harness flow, inference engine, proxy, SQLite stores, compact policy, file layout)
- `HARNESS.md` — harness-specific notes
- `CLAUDE.md` — this file

Do **not** rely on grep/memory/assumptions about what a given subsystem is supposed to be — the architecture docs in root are the source of truth. Specifically: `tools/agent-runtime/` is a **hard-fork** of the OpenAI Codex runtime (now called `ctox-core`) and is executed **in-process** via `InProcessAppServerClient`, not as an external `codex-exec` subprocess. Any remaining `codex-exec` references in CTOX production code are leftover to be cleaned up, not the intended architecture.

## Operator Guardrails (hard rules)

- **No compilation on the operator machine.** Never invoke `cargo build`, `cargo test`, `cargo check`, `cargo clippy`, `cargo run`, `cargo fmt`, or anything that walks the target/ tree. This applies even to "just a quick sanity check" — the operator machine is not a build host. All build/test validation goes through the GitHub Actions CI/CD pipeline (`.github/workflows/ci.yml`, `.github/workflows/release.yml`). Triggering a release: push a tag `vX.Y.Z` on `main`. Inspect runs with `gh run list` / `gh run view`.
- **No unsolicited branches or worktrees.** Work directly on `main` in the origin checkout. Do not create `claude/*` branches or `.claude/worktrees/*` directories without an explicit request. The existing intended workflow is commit-to-main + push; branches are only for explicitly-requested PRs.
- **No global env-var controls for runtime state.** Runtime configuration belongs in typed `AppConfig` / `engine.env` / `runtime_env::env_or_config(root, ...)`. Do not add new process-environment toggles for production behavior. Tests that need host-state overrides must write to the test-root's `engine.env`, not `std::env::set_var`. The allowlist in `src/execution/models/runtime_env.rs::process_env_override_allowed` is frozen — do not extend it without explicit approval.

## What CTOX Is (one-paragraph orientation)

CTOX is an AI agent system for autonomous server and DevOps work. It combines (1) an orchestration layer with mission queue, continuity tracking, governance and communication routing, (2) an in-process inference engine based on a hard-fork of the OpenAI Codex agent runtime (`tools/agent-runtime/` = `ctox-core`), (3) a local proxy gateway that normalizes model APIs to an OpenAI Responses surface, and (4) an optional on-host model-serving engine (`tools/model-runtime/`). TUI uses ratatui + crossterm. Persistence: a single consolidated SQLite file `runtime/ctox.db` for all core state (mission queue, tickets, governance, secrets, LCM, continuity, verification, knowledge/skillbooks/runbooks) — all paths resolved through `src/paths.rs`; tool-owned stores (`ticket_local.db`, `ctox_scraping.db`, `documents/ctox_doc.db`) stay as separate files; plus `engine.env` and `runtime/inference_runtime.json`. Rust toolchain: 1.93. Full architecture is in `AGENTS.md` — read it before making architectural claims.

## TUI Surface (orientation only)

- `src/ui/tui/mod.rs` — App state, event loop, key handling (~7.1K lines)
- `src/ui/tui/render.rs` — All ratatui rendering (~3.8K lines)
- Three pages: `Chat`, `Skills`, `Settings` (enum at `mod.rs:365`)
- Settings sub-views: `Model`, `Communication`, `Secrets`, `Update`
- Layout: Header (7 lines) + Tabs (1 line) + Page content + Status bar

When the user asks for a TUI layout or rendering change, they are the ones who run the local snapshot/smoke tools if they want to preview it — you propose code edits, they verify on their machine or via CI.

## File Layout (key entry points)

| Path | Purpose |
|------|---------|
| `src/main.rs` | CLI entry point, `run-once` command, mission loop |
| `src/ui/tui/` | Terminal UI |
| `src/context/lcm.rs` | Long-context memory engine |
| `src/context/compact.rs` | Compact policy (emergency + adaptive) |
| `src/execution/agent/direct_session.rs` | `PersistentSession` + in-process inference |
| `src/execution/agent/turn_loop.rs` | Turn planning, context rendering, continuity refresh |
| `src/execution/models/` | Model registry, adapters, runtime control, `runtime_env` gate |
| `src/execution/responses/` | Proxy gateway |
| `src/mission/` | Queue, tickets, plans, communication, review |
| `src/service/` | systemd service daemon |
| `tools/agent-runtime/` | Hard-fork of codex agent runtime (ctox-core) — integrated source tree, not a dependency |
| `tools/model-runtime/` | Local model serving engine — integrated source tree |
| `src/paths.rs` | Single source of truth for all runtime DB paths |
| `src/service/db_migration.rs` | One-shot merge of legacy `cto_agent.db` + `ctox_lcm.db` into `ctox.db` |
| `runtime/ctox.db` | Consolidated core state (mission, tickets, governance, secrets, LCM, knowledge, verification) |
| `runtime/engine.env` | Persisted operator settings |
| `runtime/inference_runtime.json` | Persisted runtime-state projection |
