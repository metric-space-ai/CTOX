# CTOX Test Layout

This directory holds CTOX-level tests that exercise the system from the outside.

- Keep fast unit tests close to the module they verify under `src/...`.
- Use `tests/integration/` for binary-driven and cross-module regression tests.
- Use `tests/harness/` for shared helpers that create isolated `CTOX_ROOT` workspaces.
- Use `tests/fixtures/` for request payloads, sample prompts, and input artifacts.
- Use `tests/golden/` for stable expected outputs such as continuity docs or web extracts.
- Use `tests/scenarios/` for longer multi-slice flows that are heavier than normal integration tests.

Do not confuse this with model qualification:

- `tests/` checks deterministic CTOX platform behavior.
- `qualification/` checks whether a concrete execution engine is good enough for CTOX.
- `benchmarks/` are for longer stress and comparison runs.

Current scope:

- `integration/service_surface.rs` verifies the service control-plane surface on a fresh runtime.
- `integration/durable_queue.rs` verifies queue persistence and baseline plan ingestion.
- `integration/channels_core.rs` verifies inbound routing, leasing, acking, and thread context.
- `integration/plan_core.rs` verifies plan emission, auto-advance, blocking, retry, and completion.
- `integration/follow_up_core.rs` verifies end-of-turn follow-up decisions.
- `integration/schedule_core.rs` verifies schedule run-now, pause, and resume behavior.
- `integration/governance_core.rs` verifies governance inventory and snapshot surfaces.
- `integration/lcm_core.rs` verifies message persistence and continuity document refresh.
- `integration/context_health_core.rs` verifies context-health warnings and repair signals.
- `integration/scrape_core.rs` verifies scrape target registration, source modules, and the four scrape web endpoints.
- `integration/verification_core.rs` verifies verification assurance, runs, and claims surfaces.

Opt-in live gates:

- `integration/proxy_e2e.rs` runs `scripts/ctox_proxy_validate.py` when `CTOX_PROXY_E2E_MODELS` is set.
- `integration/qualification_gate.rs` runs `scripts/ctox_model_qualify.py` when `CTOX_QUALIFY_MODEL` is set.

Useful commands:

- Deterministic core suite:
  `cargo test --test service_surface --test durable_queue --test channels_core --test follow_up_core --test schedule_core --test governance_core --test lcm_core --test context_health_core --test plan_core --test scrape_core --test verification_core`
- Proxy load/switch gate:
  `CTOX_PROXY_E2E_MODELS="openai/gpt-oss-20b,Qwen/Qwen3.5-35B-A3B" cargo test --test proxy_e2e`
- Model suitability gate:
  `CTOX_QUALIFY_MODEL="openai/gpt-oss-20b" CTOX_QUALIFY_SCENARIOS="minimal_ctox_stability,continuity_recall" cargo test --test qualification_gate`

Benchmarks stay separate under `benchmarks/` when they are intentionally kept as long-running stress or evaluation paths.
