# Agent D — Port `gated_delta_net` (Qwen3.5 GDN linear-attention layer)

## Goal

Port the Gated DeltaNet kernel from the dflash reference into
`ctox-engine-cuda`. This is **unique to Qwen3.5 hybrid** — 48 of the
64 target layers are GDN (linear-attention), not FullAttention. No
other model in our scope uses it, but this model is our entire
first-pass target, so this kernel is on the critical path.

Complexity note: this is the **hardest** kernel in Phase 2. Unlike
the others which are generic primitives, GDN has stateful recurrence
with a per-step intermediate snapshot that the speculative-decode
rollback depends on. Budget ~2× the time of the other agent tasks.

## Context (read first)

1. `tools/model-runtime/cuda/src/kernels/rmsnorm.rs` — template.
2. `tools/model-runtime/cuda/kernels/rmsnorm.cu` — template.
3. `tools/model-runtime/cuda/agent_tasks/README.md` — conventions.
4. Reference (read fully before writing a line of code):
   ```
   ssh metricspace@192.168.178.113 \
       "cat /home/metricspace/dflash-ref/dflash/src/gated_delta_net_kernel.cu"
   ```
   If the filename differs, grep for `gated_delta_net`:
   ```
   ssh metricspace@192.168.178.113 \
       "grep -rln 'gated_delta_net' /home/metricspace/dflash-ref/dflash/src/"
   ```

## Scope

* `kernels/gated_delta_net.cu` — port the reference kernel verbatim.
  Preserve the `extern "C"` entry point name from the reference so
  call-site wiring later is trivial. Preserve the per-layer SSM
  intermediate capture that the DFlash fast-rollback path relies on.
* `src/kernels/gated_delta_net.rs` — Rust wrapper.
  `launch_gated_delta_net_bf16(device, input, state_in, weights, ...parent_ids, state_out, output) -> Result<()>`.
  The exact signature comes from the reference — transcribe it
  faithfully.
* Integration test `gated_delta_net_step_vs_ref_dump` — load
  reference-dumped inputs (we already have a protocol for this; see
  `tools/model-runtime/cli/src/bin/draft_diff_bench.rs` for an
  example of loading `.f32.bin` dumps from `/tmp/dflash_diff`). Run
  our kernel, diff against reference output. Tolerance `max_abs <
  1e-4` for bf16 (given the ~2ULP noise floor of bf16 itself).

## Non-negotiable invariants

1. **Preserve parent_ids handling** — the reference's `parent_ids`
   parameter lets the kernel reconstruct the SSM recurrence along
   non-linear DDTree paths. Without this, tree-verify acceptance
   breaks.
2. **Preserve ssm_intermediate capture** — the reference writes the
   per-step intermediate SSM state to a capture buffer so fast-
   rollback can restore on partial accepts. This buffer is sized
   `[S_v, S_v, H_v, max_verify_tokens]` f16. Must be the same shape
   and dtype or the stepper blows up.
3. Do **not** attempt to "improve" the kernel. Byte-equivalent port
   only. Optimization comes after we've run end-to-end and have
   profiling data.

## License / attribution

z-lab/lucebox-hub is MIT. Copy the license header from the original
`.cu` file into the ported version verbatim.

## Validation

1. Build: `cargo build --release -p ctox-engine-cuda --features cuda`.
2. Test: `cargo test -p ctox-engine-cuda --features cuda --release --
    --ignored --nocapture gated_delta_net_step_vs_ref_dump`. Must
   pass with the stated tolerance.
3. Commit: `feat(cuda): port gated_delta_net from dflash reference`.

Report back commit SHA, test diff, and note any reference-side
helpers (like `f16_to_f32_widen` — we already know about that one)
that this port depends on. If a helper is missing, flag it and stub
out — don't block on it, note it for a follow-up task.
