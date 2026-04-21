//! Element dtypes supported by `CudaTensor`.
//!
//! Kept minimal: the engine only needs what the target + draft models
//! actually store or compute in. Adding a new dtype is a
//! one-enum-variant + one-trait-impl change; don't add speculatively.

use half::{bf16, f16};

/// Runtime dtype tag, stored on every `CudaTensor` for shape/dispatch
/// sanity checks and for bench/trace output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 16-bit brain float — default activation dtype on Ampere+.
    Bf16,
    /// 16-bit IEEE float — used by the reference's KV cache and mask
    /// tensors.
    F16,
    /// 32-bit IEEE float — norms + logits + a few activation paths
    /// that need the range.
    F32,
    /// 8-bit signed int — TurboQuant KV cache quantization.
    I8,
    /// 32-bit signed int — token ids, positions, parent_ids.
    I32,
    /// Q4_K_M GGUF quantized weight block (256-wide group with
    /// 8 scales + 8 mins + 4-bit values). Byte-addressed; the
    /// mmq/mmvq kernels know how to unpack.
    Q4K,
}

impl DType {
    /// Size in bytes of one logical element. For packed formats
    /// (Q4K) this is the block footprint divided by the number of
    /// logical elements per block.
    pub fn element_size_bytes(self) -> usize {
        match self {
            DType::Bf16 | DType::F16 => 2,
            DType::F32 | DType::I32 => 4,
            DType::I8 => 1,
            // Q4_K_M: 144 bytes per 256 elements = 0.5625 bytes/el.
            // Expressed as a numerator/denominator so callers can do
            // exact sizing; we don't expose a fractional byte here.
            // Call `block_bytes_for_elements` instead.
            DType::Q4K => {
                panic!("Q4K has sub-byte element size; use block_bytes_for_elements")
            }
        }
    }

    /// Byte count for storing `n_elements` of this dtype. Handles the
    /// packed block formats correctly.
    pub fn block_bytes_for_elements(self, n_elements: usize) -> usize {
        match self {
            DType::Q4K => {
                // Q4_K_M: 256 elements per 144-byte block.
                let blocks = n_elements.div_ceil(256);
                blocks * 144
            }
            other => n_elements * other.element_size_bytes(),
        }
    }
}

/// Compile-time binding from a Rust scalar type to its runtime
/// `DType` tag. Implemented only for the types we actually store
/// on device.
pub trait DTypeTrait: Sized + Copy + bytemuck::Pod + 'static {
    const DTYPE: DType;
}

impl DTypeTrait for bf16 {
    const DTYPE: DType = DType::Bf16;
}
impl DTypeTrait for f16 {
    const DTYPE: DType = DType::F16;
}
impl DTypeTrait for f32 {
    const DTYPE: DType = DType::F32;
}
impl DTypeTrait for i8 {
    const DTYPE: DType = DType::I8;
}
impl DTypeTrait for i32 {
    const DTYPE: DType = DType::I32;
}
