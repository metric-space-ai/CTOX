# Agent A — Port `mmq_q4k` (Q4_K_M mat-vec-quantized matmul)

## Goal

Port the Q4_K_M matrix-vector matmul kernel from ggml-cuda into
`ctox-engine-cuda` following the `rmsnorm` template. This is **the
single biggest performance win** — our current candle-based decode
is ~10× slower than the ggml-cuda reference, and mmq_q4k is where
~60 % of that gap lives.

## Context (read these first)

You have no memory of prior conversations. Start by reading:

1. `/Users/michaelwelsch/Documents/ctox/tools/model-runtime/cuda/src/kernels/rmsnorm.rs` —
   template Rust wrapper. Mirror its structure exactly (OnceLock
   cache, shape validation, no stream sync, ignored CPU-diff test).
2. `/Users/michaelwelsch/Documents/ctox/tools/model-runtime/cuda/kernels/rmsnorm.cu` —
   template CUDA kernel. Mirror the `extern "C"` entry-point
   convention and the header comment block.
3. `/Users/michaelwelsch/Documents/ctox/tools/model-runtime/cuda/build.rs` —
   the build pipeline. When you add `kernels/mmq_q4k.cu` the script
   automatically picks it up and generates `MMQ_Q4K_PTX`.
4. Reference implementation — fetch from the A6000 host (see "Where to
   get the reference" below). Target llama.cpp's `ggml-cuda/mmq.cu`
   and `ggml-cuda/mmvq.cu` — the mat-mat (batched) and mat-vec
   (decode) paths respectively.

## Scope

* `kernels/mmq_q4k.cu` — single `.cu` file exposing `extern "C"`
  entry points. For the decode hot path we only need the MV (batch
  size 1 query × N columns) variant. Naming:
    * `mmvq_q4k_f32_out` — matrix-vector q4k × f32 → f32
    * `mmvq_q4k_f16_out` — same but output f16 (for activations)
* `src/kernels/mmq_q4k.rs` — Rust wrapper with
  `launch_mmvq_q4k_f32(...)` and `launch_mmvq_q4k_f16(...)`.
* Integration test `mmvq_q4k_vs_cpu_golden` — generates a small
  Q4_K_M block (256 elements per block; total 16 blocks = 4096
  columns) on CPU, runs GPU kernel, compares against a naive CPU
  Q4_K_M dequant + matmul. Tolerance `max_rel < 1e-2` because Q4
  quantization's dequant error is the floor.
* **Do not** port the batched `mmq_*` variant in this task — leave
  `TODO` at the top of the `.cu` stating the mat-mat path comes
  later (needed for prefill, but our decode hot path is mat-vec).

## Q4_K_M block layout (enforce this exactly)

Each block covers 256 logical elements in 144 bytes:

```
struct block_q4_K {
    half   d;        // super-block scale
    half   dmin;     // super-block minimum
    uint8_t scales[12];  // 6-bit per sub-block scale + min (packed)
    uint8_t qs[128];     // 4-bit quantized values (2 per byte)
};
```

Eight sub-blocks × 32 elements/sub-block = 256. Per-sub-block:
`val = d * scale * q - dmin * min_scale`. See llama.cpp's
`dequantize_q4_K` in `ggml/src/ggml-cuda/vecdotq.cuh` for the
definitive dequant math.

## Signatures

```rust
// A: [K, N] Q4_K_M quantized (K/256 blocks per column, N columns)
//    stored as raw bytes — CudaTensor<i8> with shape = [col_bytes × N]
// x: [K]     f32  input vector
// y: [N]     f32  output vector (= A^T · x)
pub fn launch_mmvq_q4k_f32(
    device: &Arc<DeviceContext>,
    a_q4k: &CudaTensor<i8>,  // packed block bytes; we lie about dtype
    k: usize,
    n: usize,
    x: &CudaTensor<f32>,
    y: &mut CudaTensor<f32>,
) -> Result<()>;

pub fn launch_mmvq_q4k_f16(...) -> Result<()>; // same but y is half::f16
```

Note: `CudaTensor<i8>` as the carrier for Q4_K_M byte-packed data is a
deliberate abuse. We track "this is Q4K bytes" out-of-band via a
comment on the tensor at the call site; the wrapper here just
asserts `a_q4k.numel() == (k / 256) * n * 144`.

## Launch config

Mirror ggml-cuda's mmvq pattern:
* `grid_dim = (n / BLOCK_N_PER_BLK, 1, 1)` where `BLOCK_N_PER_BLK = 2`
  (two output columns per thread block).
* `block_dim = (32, BLOCK_N_PER_BLK, 1)` — one warp per output column.
* `shared_mem = 0` (all partial reductions live in registers +
  warp-shuffle).

## Where to get the reference

The A6000 host has the full dflash fork:
```
ssh metricspace@192.168.178.113 "cat /home/metricspace/dflash-ref/dflash/deps/llama.cpp/ggml/src/ggml-cuda/mmvq.cu"
```
Read `vec_dot_q4_K_q8_1` and the `mul_mat_vec_q` template; port
literally. Licensing (MIT) is compatible — add the llama.cpp
copyright header to `kernels/mmq_q4k.cu` verbatim.

## Validation

1. Build: `cargo build --release -p ctox-engine-cuda --features cuda`
   on the A6000 (via ssh alias). Must pass.
2. Test: `cargo test -p ctox-engine-cuda --features cuda --release --
    --ignored --nocapture mmvq_q4k_vs_cpu_golden`. Must pass with
   `max_rel < 1e-2`.
3. Commit on `main` with message:
   `feat(cuda): port mmvq_q4k (Q4_K_M mat-vec) from ggml-cuda`.

Report back: (a) the commit SHA you landed, (b) the test output line
showing `max_rel` and `max_abs`, (c) any open TODOs in the `.cu` or
`.rs` file and why.

## Out of scope

* Batched `mmq` (mat-mat) — leave a TODO comment.
* Other quantization formats (Q8_0, Q6_K) — separate tasks.
* Activation quantization to q8_1 on the input side — assume x is
  already f32 and do the inner product as
  `dequant(a_block) · x_block` rather than going through the ggml
  q8_1-quantize-x path. (That's a later optimization.)
