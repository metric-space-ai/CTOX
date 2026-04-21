//! Model constants mirroring `dflash_megakernel_decode.cu` and
//! `dflash_megakernel_prefill.cu`. MUST stay in lockstep with the
//! `.cu` sources — the kernels read them from `__device__ constexpr
//! int` values baked into their TUs, so if you bump one here without
//! bumping the other the buffer allocations will silently mis-size
//! and the kernel will write out of bounds.

// ── Core dims
pub const HIDDEN_SIZE: usize = 1024;
pub const INTERMEDIATE_SIZE: usize = 3584;
pub const NUM_LAYERS: usize = 24;
pub const VOCAB_SIZE: usize = 248_320;

// ── Full-Attention dims
pub const FA_NUM_Q_HEADS: usize = 8;
pub const FA_NUM_KV_HEADS: usize = 2;
pub const FA_HEAD_DIM: usize = 256;
pub const FA_Q_SIZE: usize = FA_NUM_Q_HEADS * FA_HEAD_DIM; // 2048
pub const FA_GATE_SIZE: usize = FA_Q_SIZE;
pub const FA_QPROJ_SIZE: usize = FA_Q_SIZE + FA_GATE_SIZE; // 4096
pub const FA_KV_SIZE: usize = FA_NUM_KV_HEADS * FA_HEAD_DIM; // 512
pub const FA_ROTARY_DIM: usize = 64;

// ── DeltaNet dims
pub const DN_NUM_HEADS: usize = 16;
pub const DN_KEY_DIM: usize = 128;
pub const DN_VALUE_DIM: usize = 128;
pub const DN_CONV_KERNEL: usize = 4;
pub const DN_QK_SIZE: usize = DN_NUM_HEADS * DN_KEY_DIM; // 2048
pub const DN_V_SIZE: usize = DN_NUM_HEADS * DN_VALUE_DIM; // 2048
pub const DN_CONV_CHANNELS: usize = DN_QK_SIZE + DN_QK_SIZE + DN_V_SIZE; // 6144

// ── Runtime caps
/// Ring capacity for each Full-Attention layer's KV cache. Reference
/// bakes this in at 2048; anything longer overflows the ring. Hard-
/// coded in both prefill.cu and kernel.cu — DO NOT change here
/// without rebuilding both .cu TUs.
pub const MAX_SEQ_LEN: usize = 2048;

/// Layer-type pattern for Qwen3.5-0.8B: every 4th layer is Full
/// Attention, the other three are DeltaNet. 18 DN + 6 FA = 24.
pub const LAYER_PATTERN: [u8; NUM_LAYERS] = [
    0, 0, 0, 1, //
    0, 0, 0, 1, //
    0, 0, 0, 1, //
    0, 0, 0, 1, //
    0, 0, 0, 1, //
    0, 0, 0, 1, //
];

/// How many layers of each type this config has.
pub const N_DN_LAYERS: usize = 18;
pub const N_FA_LAYERS: usize = 6;

/// Layer type constants matching the packed `LayerWeights.layer_type`
/// field expected by the kernel.
pub const LAYER_TYPE_DELTANET: i32 = 0;
pub const LAYER_TYPE_FULL_ATTENTION: i32 = 1;
