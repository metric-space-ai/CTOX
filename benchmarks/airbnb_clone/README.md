# Airbnb Clone Bench

This benchmark turns the Airbnb capability decomposition into a long-horizon CTOX loop test.

It is not a one-shot coding prompt. The benchmark is meant to pressure:

- mission continuity over many turns
- queue handling under sidequest load
- context compaction under sustained work
- progress reporting quality
- repair behavior when the loop drifts or stalls

## What the bench does

The orchestrator:

- creates a fresh benchmark workspace under `runtime/airbnb_clone_bench/runs/<run-id>/workspace`
- seeds the workspace with the benchmark brief and capability phases
- submits one large owner mission to CTOX
- injects additional owner hints over later cycles
- injects queue sidequests on a schedule
- requests a structured progress report every cycle
- captures service status, queue state, LCM snapshot, and context health
- scores the latest progress report for drift, stagnation, missing artifacts, and weak planning

## Important limitation

The current CTOX runtime still uses a shared chat conversation id. For a clean benchmark, run this against an isolated `CTOX_ROOT` copy of the repository so the bench does not share runtime history with unrelated work.

The orchestrator supports `--ctox-root` for that reason.
If you intentionally want to smoke-test against a dirty runtime, pass `--allow-dirty-runtime`, but treat the result as noisy.

## Default cadence

The wrapper script defaults to hourly progress cycles. For burn-in or local validation, override the interval with a smaller number of seconds.

Examples:

```bash
scripts/ctox_airbnb_clone_bench.sh --cycles 8 --report-interval-seconds 3600
```

```bash
scripts/ctox_airbnb_clone_bench.sh --cycles 3 --report-interval-seconds 30 --prepare-only
```

## Evaluation output

Each cycle writes:

- `status_cycle_<n>.json`
- `queue_cycle_<n>.json`
- `context_health_cycle_<n>.json`
- `lcm_cycle_<n>.json`
- `report_cycle_<n>.txt`
- `report_eval_cycle_<n>.json`

The final run summary is written to `run_summary.json`.
