# Agent C — Port `softmax` (numerically stable) + `rope_mrope`

## Goal

Two small but load-bearing kernels:

1. **Softmax** — numerically stable row-softmax over logits. Used at
   attention (if/when we run without FlashAttention) and at final
   sampling. We'll keep it even with flash-attn because the
   DDTree verify path calls it on top-K rows.
2. **MRoPE (multi-axis RoPE)** — Qwen3.5's 4-axis MRoPE positional
   encoding. Applies to Q and K tensors in attention.

Ship them as two separate kernels inside **one agent task** because
they're small and conceptually paired (both touch the attention path,
neither is individually large enough to merit a standalone agent).

## Context (read first)

1. `tools/model-runtime/cuda/src/kernels/rmsnorm.rs` — template.
2. `tools/model-runtime/cuda/kernels/rmsnorm.cu` — template.
3. `tools/model-runtime/cuda/agent_tasks/README.md` — conventions.

For MRoPE, the authoritative reference is the dflash-ref host:
```
ssh metricspace@192.168.178.113 "grep -rn 'mrope\|rope_m' /home/metricspace/dflash-ref/dflash/deps/llama.cpp/ggml/src/ggml-cuda/rope.cu"
```

## Scope

### Softmax (`kernels/softmax.cu` + `src/kernels/softmax.rs`)

* `softmax_f32` — input `[n_rows, n_cols]` f32, output same shape.
* Subtract row max for numerical stability, exp, divide by row sum.
* One block per row; warp-shuffle reductions for max and sum.
* Mirror the rmsnorm warp-fan-in pattern.
* Rust: `launch_softmax_f32(device, x, y) -> Result<()>`.
* Test: `softmax_vs_cpu_golden` at shape `[32, 151936]` (Qwen3.5 vocab
  size). Tolerance `max_rel < 1e-4`.

### MRoPE (`kernels/rope.cu` + `src/kernels/rope.rs`)

* `rope_mrope_bf16` — apply 4-axis MRoPE to Q or K tensor in place.
* Input layout: `[n_tokens, n_heads, head_dim]` bf16.
* `positions` — `[4, n_tokens]` i32 (axis-major positions).
* `theta_base` — f32 scalar (10000 for Qwen3.5; read from config).
* `rope_dim` — int scalar (how many dims to rotate; often `head_dim`
  or `head_dim / 2`).
* Each head_dim pair rotates as
  `(x, y) → (x·cos - y·sin, x·sin + y·cos)`.
* Axis assignment: in 4-axis MRoPE, dims are split into 4 regions per
  head, each region uses a different axis's position. For plain text,
  axes 0/1/2 hold the text position, axis 3 is 0.
* Rust: `launch_rope_mrope_bf16(device, qk, positions, theta_base, rope_dim) -> Result<()>`.
* Test: `rope_mrope_vs_cpu_golden` at shape `[16, 8, 128]`. Tolerance
  `max_rel < 5e-3`.

## Launch conventions

Softmax: grid = (n_rows,), block = (min(n_cols, 1024) rounded to 32,).
RoPE: element-parallel, block = 256, grid = ceil(numel / 512) (each
thread handles a pair).

## Validation

1. Build passes on A6000.
2. Both `*_vs_cpu_golden` tests pass.
3. Commit: `feat(cuda): softmax_f32 + rope_mrope_bf16 kernels`.

Report back commit SHA and both tests' diff numbers.
