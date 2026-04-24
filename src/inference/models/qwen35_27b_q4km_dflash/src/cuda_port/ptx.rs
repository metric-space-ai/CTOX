//! Embedded PTX blobs compiled from `vendor/ggml-cuda/*.cu` by
//! `build.rs`.
//!
//! `build.rs` runs nvcc once per vendored `.cu` file that the Rust
//! dispatcher port actually references, writing the resulting PTX
//! (plain-text nvcc assembly) to `$OUT_DIR/<stem>.ptx`. This module
//! pulls those blobs in via `include_str!` so the final Rust binary
//! ships the kernels as inline constants — no external `.so` to
//! link, no runtime file lookup.
//!
//! As each op is ported (see `cuda_port::ops::*`), a matching entry
//! is added here. During the transitional phase — before every op
//! is ported — this module is populated incrementally.

/// Lookup: kernel-source-file stem → PTX text.
///
/// Populated at build time by `build.rs`. Accessors below return
/// `None` while a file is not yet wired into the PTX compile step.
pub fn ptx_for_stem(stem: &str) -> Option<&'static str> {
    match stem {
        // Populated incrementally as each op dispatcher lands in
        // cuda_port::ops. First entry ships once norm.rs does.
        _ => None,
    }
}

/// Return every PTX blob currently embedded in the binary.
pub fn all_embedded_ptx() -> &'static [(&'static str, &'static str)] {
    &[]
}
