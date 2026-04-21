# Agent E — GGUF weight loader (`gguf.rs`)

## Goal

Parse a GGUF file (llama.cpp's container format) from disk and emit
a map of `(tensor_name, CudaTensor<T>)` handles, with weights
uploaded to device in their native dtype (including Q4_K_M packed
blocks). This is the weight-loading half of the bare-metal stack —
kernels consume `CudaTensor`, this is how they get populated.

No CUDA kernel work. Pure Rust + cudarc. Independent of the kernel
agents so it runs fully in parallel.

## Context (read first)

1. `tools/model-runtime/cuda/src/tensor.rs` — the `CudaTensor<T>`
   type you're producing. Note `from_host` and `zeros` are the only
   constructors; you'll need `from_host` for uploading the loaded
   bytes.
2. `tools/model-runtime/cuda/src/dtype.rs` — the `DType` enum.
   `DType::Q4K` is the tag for Q4_K_M-packed weights.
3. GGUF spec: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>
4. Reference parsers — many exist. Pick one:
   * `llama.cpp/gguf-py/gguf/` (Python; readable)
   * `https://crates.io/crates/gguf` (don't add as dep; use as reference)

## Scope

* `src/gguf.rs` under `tools/model-runtime/cuda/src/gguf.rs` (new
  module; wire into `lib.rs` behind the `cuda` feature).
* Public API:
  ```rust
  pub struct GgufTensor {
      pub name: String,
      pub dtype: crate::dtype::DType,
      pub shape: Vec<usize>,
      // Owned device buffer. For Q4_K_M, bytes-packed via
      // CudaTensor<i8> as the carrier (same abuse as in the mmq
      // agent task — tag is out-of-band via `dtype`).
      pub buf: GgufBuf,
  }

  pub enum GgufBuf {
      F32(CudaTensor<f32>),
      F16(CudaTensor<half::f16>),
      Bf16(CudaTensor<half::bf16>),
      Q4K(CudaTensor<i8>),  // byte-packed; 144 bytes per 256 elements
      I32(CudaTensor<i32>),
      I8(CudaTensor<i8>),
  }

  pub fn load_gguf<P: AsRef<Path>>(
      device: &Arc<DeviceContext>,
      path: P,
  ) -> Result<HashMap<String, GgufTensor>>;
  ```
* Parse the GGUF header (`GGUF` magic, version=3, tensor/metadata
  counts), skip the metadata (we only care about tensors), read each
  tensor descriptor, mmap the data region, upload each tensor to
  device via `CudaTensor::from_host` in its native dtype.
* Support only the dtypes enumerated above. Other GGML dtypes
  (Q2_K, Q3_K, Q5_K, Q6_K, Q8_0) — return
  `Err(anyhow!("unsupported ggml dtype {:?} for tensor {}", t, name))`
  and let the caller deal with it.

## Performance notes

* Use `memmap2::Mmap` for the data region — 27B Q4_K_M is ~15 GB, we
  don't want to `std::fs::read` that.
* Per-tensor upload is `stream.memcpy_htod(host_slice, &mut device_slice)`.
  Host slice comes from the mmap + offset.
* No streaming; fine to serialize uploads. The bottleneck is PCIe,
  not CPU, and 15 GB of 27B weights uploads in ~3 seconds on x8
  Gen4.

## Validation

* Integration test `load_gguf_27b_q4km_smoke` (ignored by default,
  needs the Qwen3.5-27B GGUF on the host at
  `/home/metricspace/dflash-ref/dflash/models/Qwen3.5-27B-Q4_K_M.gguf`).
  Loads the file, verifies:
    1. Tensor count is 851 (known from the reference
       `[target] target loaded: 851 tensors` log line).
    2. `token_embd.weight` shape starts with `[151936, 5120]`
       (vocab × hidden).
    3. First attention layer's `blk.0.attn_q.weight` is `DType::Q4K`
       with correct byte footprint.
  Run: `cargo test -p ctox-engine-cuda --features cuda --release --
    --ignored --nocapture load_gguf_27b_q4km_smoke`.
* `cargo build --release -p ctox-engine-cuda --features cuda` passes.
* Commit: `feat(cuda): GGUF loader with Q4_K_M + half-precision support`.

Report: commit SHA, test output showing tensor count + one spot-check
tensor's shape & dtype, any GGML dtypes you had to stub.
