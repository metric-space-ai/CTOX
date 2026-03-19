---
name: storage-survival-operations
description: Use when the CTO-Agent needs to inspect host disk pressure, plan safe cleanup, or keep local builds, logs and model artifacts from killing the machine.
---

# Storage Survival Operations

Use this skill when host disk headroom becomes a real operational risk or when a heavy local action like builds, reinstalls or model downloads could push the machine into danger.

## Required sources

Read these first:

1. `src/context_controller.rs`
2. `src/agentic.rs`
3. `scripts/install_cto_agent.sh`

## Commands

- `df -h`
- `du -xh --max-depth=2 runtime 2>/dev/null | sort -h | tail -n 40`
- `du -xh --max-depth=2 target 2>/dev/null | sort -h | tail -n 40`
- `du -xh --max-depth=2 ~/.cargo 2>/dev/null | sort -h | tail -n 40`
- `du -xh --max-depth=2 ~/.cache/huggingface 2>/dev/null | sort -h | tail -n 40`

## Runtime contract

- Tool path: native terminal commands through the codex-backed exec engine
- Runtime storage: `runtime/`, `target/`, `~/.cargo/`, `~/.cache/huggingface/`
- Inputs: current host disk state, pending installs/builds/model switches, known cache and log locations
- Outputs: bounded storage diagnosis, safe cleanup plan, concrete follow-up task or exec step
- Side effects: only bounded cleanup after evidence; avoid broad deletion without first identifying the largest real offenders

## Failure handling

- If disk pressure is visible, inspect first and rank the largest artifact families before deleting anything.
- Prefer pruning rebuildable artifacts like `target/`, `runtime/build/`, stale logs or transient cargo caches before touching model snapshots.
- Delete model caches only when you can justify why the specific snapshot is safe to redownload later.
- Verify with a second `df -h` and record what changed in memory or checkpoint text.

## Never do

- Never blindly `rm -rf` large directories without first confirming they are the actual pressure source.
- Never claim the host is safe again without rechecking free space after the bounded cleanup.
- Never trade one outage for another by deleting active runtime state, current databases or live credentials.
