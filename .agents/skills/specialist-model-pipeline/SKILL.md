---
name: specialist-model-pipeline
description: Use when the CTO-Agent identifies a repeated browser-backed task that should be externalized into a small specialist model or reviewed deterministic capability.
---

# Specialist Model Pipeline

This skill defines how repeated browser-backed tasks become a specialist model or reviewed worker.

It follows the same high-level logic as `local_ai_tunes`: quality first, scale second, and never train directly from raw accepted batch rows.

## Required sources

Read these first:

1. `../../../contracts/pipelines/specialist-model-pipeline.json`
2. `../../../contracts/browser/browser-capability-policy.json`
3. `../../../contracts/history/creation-ledger.md`

## Mandatory sequence

1. Freeze the task spec and reviewed browser capability surface.
2. Collect accepted records from real repeated runs.
3. Run a dedicated dataset release step that converts those records into the exact target runtime contract.
4. Train only from the released dataset artifact.
5. Evaluate against the frozen baseline path.
6. Promote only after the gates pass.

## CTO-specific interpretation

- The specialist exists to keep the main CTO-Agent context clean.
- The specialist should be narrow, boring and repeatable.
- Browser-backed specialists still execute through the real browser runtime.
- Browser execution and training are separate concerns.
- For this CTO-Agent path, browser WebGPU training is considered too inefficient.

## Experiment discipline

- Treat the currently released reviewed path as the champion by default.
- Test new tool, capability, dataset or training changes as challengers before promotion whenever a bounded challenger eval is possible.
- Prefer one artifact family per experiment round: tool scripts, browser capabilities, training rows, training config or bundle policy.
- Carry a short explicit hypothesis for each challenger and judge it against a fixed mini-suite when possible.
- Record a keep, discard or park decision after smoke, eval or training, instead of letting challenger outcomes stay implicit.
- Prefer keep only when the challenger is measurably better or equally strong and simpler.

## Never do

- Never train directly from raw operational transcripts.
- Never treat “accepted records” as already ready for training.
- Never promote a specialist without a release artifact and evaluation report.
