//! Kernel registry — one Rust wrapper function per fused CUDA op.
//!
//! Each kernel sits in its own module (e.g. `rmsnorm_rope_qkv`,
//! `flash_attn_v2`, `mmq_q4k`). The module provides:
//!
//!   * The PTX blob, either compiled at build time from a `.cu`
//!     source via `nvcc` + `include_bytes!`, or imported from the
//!     reference's pre-built `.ptx` files during the transition.
//!   * A `launch(...)` Rust function taking `&DeviceContext`,
//!     `&CudaTensor<...>` inputs, `&mut CudaTensor<...>` outputs,
//!     and explicit launch config (grid/block, shared memory size).
//!   * NO implicit sync. Callers sync at phase boundaries (end of
//!     prefill ubatch, end of verify forward, etc.).
//!
//! No kernels yet — this is the registry placeholder. First real
//! kernel: `rmsnorm_rope_qkv_fused` (sits at the top of every
//! attention layer, fuses three otherwise-separate ops, high ROI
//! for candle → bare-metal migration).
//!
//! ## Launch config convention
//!
//! `LaunchCfg { grid: (x, y, z), block: (x, y, z), shared_mem: n_bytes }`.
//! Every kernel wrapper computes this internally from tensor shapes
//! and a few tuning constants. Tuning constants are hard-coded per
//! SM capability (`sm_80`, `sm_86`, `sm_89`, `sm_90`) via cfg-gated
//! constants — we don't do runtime auto-tuning.

// (intentionally empty for now)
