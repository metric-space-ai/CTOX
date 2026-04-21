# Agent B — Port `silu_mul_fused` (SwiGLU MLP activation)

## Goal

Fused SiLU-and-multiply kernel for the MLP block gate. This is the
second step of the SwiGLU nonlinearity:
`y = silu(gate) * up` where `silu(x) = x * sigmoid(x)`. Fusing avoids
two separate full reads+writes of the hidden tensor.

## Context (read first)

1. `/Users/michaelwelsch/Documents/ctox/tools/model-runtime/cuda/src/kernels/rmsnorm.rs`
   — template Rust wrapper. Mirror its structure.
2. `/Users/michaelwelsch/Documents/ctox/tools/model-runtime/cuda/kernels/rmsnorm.cu`
   — template CUDA kernel.
3. `/Users/michaelwelsch/Documents/ctox/tools/model-runtime/cuda/agent_tasks/README.md`
   — enforced conventions.

## Scope

* `kernels/silu_mul.cu` — single `.cu` file, two entry points:
    * `silu_mul_f32` — takes f32 in, f32 out
    * `silu_mul_bf16` — takes `__nv_bfloat16` in, `__nv_bfloat16` out
* `src/kernels/silu_mul.rs`:
    * `launch_silu_mul_f32(device, gate, up, y) -> Result<()>`
    * `launch_silu_mul_bf16(device, gate, up, y) -> Result<()>`
* Integration test `silu_mul_vs_cpu_golden` — shape
  `[n_tokens=8, intermediate_dim=13824]` (Qwen3.5-27B MLP hidden
  dim). Tolerance `max_rel < 1e-3` for f32, `< 5e-3` for bf16.

## Math

```
silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
y[i]    = silu(gate[i]) * up[i]
```

Per-element op, memory-bound. Launch 1 element per thread,
`block_dim = 256`, `grid_dim = ceil(numel / 256)`.

Use `expf()` for f32. For bf16, cast to f32 for the sigmoid, multiply,
cast back (the activation precision for SwiGLU doesn't need bf16 math
inside — only the memory representation).

## Signatures

```rust
pub fn launch_silu_mul_f32(
    device: &Arc<DeviceContext>,
    gate: &CudaTensor<f32>,
    up:   &CudaTensor<f32>,
    y:    &mut CudaTensor<f32>,
) -> Result<()>;
pub fn launch_silu_mul_bf16(
    device: &Arc<DeviceContext>,
    gate: &CudaTensor<half::bf16>,
    up:   &CudaTensor<half::bf16>,
    y:    &mut CudaTensor<half::bf16>,
) -> Result<()>;
```

Validate: `gate.shape() == up.shape() == y.shape()`, same dtype.

## Validation

1. `cargo build --release -p ctox-engine-cuda --features cuda` on A6000.
2. `cargo test -p ctox-engine-cuda --features cuda --release --
    --ignored --nocapture silu_mul_vs_cpu_golden` must pass.
3. Commit: `feat(cuda): silu_mul fused kernel (f32 + bf16)`.

Report back commit SHA and test diff numbers.
