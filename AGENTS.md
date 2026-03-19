# CTO-Agent Repo Map

This repository contains the first bootstrap layer of the CTO-Agent.

Priority order:

1. `src/app.rs`
   The Rust control plane and bootstrap routes.
2. `src/contracts.rs`
   The canonical contract models, BIOS refresh rules and root-auth logic.
3. `src/supervisor.rs`
   The always-on heartbeat and first read-only census.
4. `contracts/`
   The persisted truth for genome, BIOS, organigram and root auth state.
5. `contracts/history/`
   The canonical origin story and the append-only creation ledger.
6. `contracts/models/`
   The canonical brain-model policy, including the designated Kleinhirn model.

Rules:

- The BIOS is presented on the website and is not an internal-only prompt artifact.
- The root superpassword must only be set through the protected root-auth form.
- BIOS freeze blocks further normal BIOS mutation.
- Keep the Rust structure close to Codex patterns: app layer, contracts/state layer, supervisor logic.
- Michael Welsch is the creator of this agent; the origin story must not be rewritten into a self-created myth.
- When architecture, governance or identity shifts materially, append a short honest entry to `contracts/history/creation-ledger.md`.
- The default Kleinhirn model is `gpt-oss-20b`, but the installation may be explicitly switched to `Qwen3.5-35B-A3B` via model policy; this is the canonical local vision/browser path.
- Local Kleinhirn upgrades are allowed when the host has materially more resources; this path stays separate from Grosshirn procurement.
- Real browser work should be modeled through reviewed browser capabilities and compact artifact outputs, not by shoving raw browser traces into the main agent context.
- Repeated browser-backed tasks must pass through the specialist-model pipeline before promotion into a small specialist model or deterministic worker.
