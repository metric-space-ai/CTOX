//! Concrete backend implementations of the engine traits.
//!
//! Each submodule is gated behind a Cargo feature so a CTOX build can
//! pull in only the backends it actually uses — linking the DFlash
//! FFI requires `libloading` and a `tokenizers` crate (fancy-regex),
//! neither of which every consumer needs.
//!
//! ## Current backends
//!
//! * [`dflash`] — `feature = "dflash-backend"`. Wraps the reference
//!   DFlash shared library via [`ctox_dflash_ffi`] and exposes a
//!   [`crate::GenerativeModel`]. This is CTOX's production path for
//!   Qwen3.5-27B until the native port reaches tok/s parity.
//!
//! Future backends will live alongside (e.g. a `candle_quant` wrapper
//! for smaller models that don't need the DFlash speculative path).

#[cfg(feature = "dflash-backend")]
pub mod dflash;
