//! GGUF weight loader.
//!
//! Parses a GGUF v3 container (llama.cpp's weight format) from disk
//! and emits a `HashMap<String, GgufTensor>` of device-resident
//! weights.
//!
//! Scope:
//!   * Header: magic `GGUF`, version 3, tensor/metadata counts.
//!   * Metadata: parsed only enough to locate `general.alignment`;
//!     values are otherwise skipped (we don't surface metadata from
//!     this loader — that's the tokenizer / config layer's job).
//!   * Tensor descriptors: name, n_dims, dims, ggml dtype, offset.
//!   * Tensor data: mmap'd, sliced per-tensor, uploaded to the GPU
//!     via `CudaTensor::from_host`. Byte-packed types (Q4_K_M) go
//!     through a `CudaTensor<i8>` of the raw block bytes.
//!
//! Supported dtypes:
//!   * F32  (ggml_type 0)
//!   * F16  (ggml_type 1)
//!   * Q4_K (ggml_type 12 — 144 B / 256 elems)
//!   * I8   (ggml_type 24)
//!   * I32  (ggml_type 26)
//!   * BF16 (ggml_type 30)
//!
//! Any other ggml dtype returns an `unsupported ggml dtype` error.
//!
//! Performance note: the data region for a 27B Q4_K_M model is ~15 GB,
//! so we mmap the file rather than slurping it into a `Vec<u8>`. Each
//! per-tensor upload is a single `cudaMemcpy` H2D via cudarc's
//! `memcpy_stod`; PCIe — not the CPU-side parse — is the bottleneck.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use half::{bf16, f16};
use memmap2::Mmap;

use crate::device::DeviceContext;
use crate::dtype::DType;
use crate::tensor::CudaTensor;

/// Magic bytes: ASCII "GGUF" stored little-endian as u32 = 0x46554747.
const GGUF_MAGIC: u32 = 0x4655_4747;

/// Supported GGUF container version. We intentionally pin v3 — earlier
/// versions (v1/v2) used different tensor-info encodings and v1 used
/// u32 counts instead of u64.
const GGUF_VERSION: u32 = 3;

/// Default tensor-data alignment when `general.alignment` is absent.
/// Matches the llama.cpp default.
const DEFAULT_ALIGNMENT: u64 = 32;

/// One device-resident tensor loaded from a GGUF file.
pub struct GgufTensor {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub buf: GgufBuf,
}

/// Device storage for a GGUF tensor, tagged by element type.
///
/// Q4_K (and any other sub-byte packed format we add later) lives in
/// an `i8` tensor whose length equals the block-byte count — the
/// unpacking kernels know how to read it.
pub enum GgufBuf {
    F32(CudaTensor<f32>),
    F16(CudaTensor<f16>),
    Bf16(CudaTensor<bf16>),
    /// Byte-packed Q4_K_M blocks: 144 B per 256-element block.
    Q4K(CudaTensor<i8>),
    I32(CudaTensor<i32>),
    I8(CudaTensor<i8>),
}

/// GGML tensor type codes (subset — only the ones we parse or reject).
///
/// Values come straight from `ggml.h`. We spell them out here to avoid
/// re-linking ggml just for the enum.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
    Unknown(u32),
}

impl GgmlType {
    fn from_u32(v: u32) -> Self {
        match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2K,
            11 => GgmlType::Q3K,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            15 => GgmlType::Q8K,
            16 => GgmlType::IQ2XXS,
            17 => GgmlType::IQ2XS,
            18 => GgmlType::IQ3XXS,
            19 => GgmlType::IQ1S,
            20 => GgmlType::IQ4NL,
            21 => GgmlType::IQ3S,
            22 => GgmlType::IQ2S,
            23 => GgmlType::IQ4XS,
            24 => GgmlType::I8,
            25 => GgmlType::I16,
            26 => GgmlType::I32,
            27 => GgmlType::I64,
            28 => GgmlType::F64,
            29 => GgmlType::IQ1M,
            30 => GgmlType::BF16,
            other => GgmlType::Unknown(other),
        }
    }
}

/// GGUF metadata value-type codes. Used while skipping metadata KVs.
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
enum GgufValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl GgufValueType {
    fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => GgufValueType::U8,
            1 => GgufValueType::I8,
            2 => GgufValueType::U16,
            3 => GgufValueType::I16,
            4 => GgufValueType::U32,
            5 => GgufValueType::I32,
            6 => GgufValueType::F32,
            7 => GgufValueType::Bool,
            8 => GgufValueType::String,
            9 => GgufValueType::Array,
            10 => GgufValueType::U64,
            11 => GgufValueType::I64,
            12 => GgufValueType::F64,
            other => return Err(anyhow!("unknown gguf value type {}", other)),
        })
    }

    fn scalar_bytes(self) -> Option<usize> {
        Some(match self {
            GgufValueType::U8 | GgufValueType::I8 | GgufValueType::Bool => 1,
            GgufValueType::U16 | GgufValueType::I16 => 2,
            GgufValueType::U32 | GgufValueType::I32 | GgufValueType::F32 => 4,
            GgufValueType::U64 | GgufValueType::I64 | GgufValueType::F64 => 8,
            GgufValueType::String | GgufValueType::Array => return None,
        })
    }
}

/// Minimal forward-only cursor over the mmap'd file bytes.
///
/// All reads are little-endian per the GGUF spec.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn skip(&mut self, n: usize) -> Result<()> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| anyhow!("cursor skip overflow"))?;
        if end > self.buf.len() {
            return Err(anyhow!(
                "cursor skip out of range: {} + {} > {}",
                self.pos,
                n,
                self.buf.len()
            ));
        }
        self.pos = end;
        Ok(())
    }

    fn read_u8(&mut self) -> Result<u8> {
        if self.pos >= self.buf.len() {
            return Err(anyhow!("read_u8 past end"));
        }
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_u32(&mut self) -> Result<u32> {
        if self.pos + 4 > self.buf.len() {
            return Err(anyhow!("read_u32 past end"));
        }
        let mut b = [0u8; 4];
        b.copy_from_slice(&self.buf[self.pos..self.pos + 4]);
        self.pos += 4;
        Ok(u32::from_le_bytes(b))
    }

    fn read_u64(&mut self) -> Result<u64> {
        if self.pos + 8 > self.buf.len() {
            return Err(anyhow!("read_u64 past end"));
        }
        let mut b = [0u8; 8];
        b.copy_from_slice(&self.buf[self.pos..self.pos + 8]);
        self.pos += 8;
        Ok(u64::from_le_bytes(b))
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        if self.pos + len > self.buf.len() {
            return Err(anyhow!(
                "string length {} would exceed buf (pos={}, total={})",
                len,
                self.pos,
                self.buf.len()
            ));
        }
        let bytes = &self.buf[self.pos..self.pos + len];
        self.pos += len;
        // GGUF strings are UTF-8 without a NUL terminator.
        String::from_utf8(bytes.to_vec()).map_err(|e| anyhow!("non-utf8 gguf string: {}", e))
    }

    /// Skip a GGUF metadata value of the given type. Recursive for
    /// arrays.
    fn skip_value(&mut self, ty: GgufValueType) -> Result<()> {
        match ty {
            GgufValueType::String => {
                let len = self.read_u64()? as usize;
                self.skip(len)?;
            }
            GgufValueType::Array => {
                let elem_ty = GgufValueType::from_u32(self.read_u32()?)?;
                let n = self.read_u64()? as usize;
                if let Some(sz) = elem_ty.scalar_bytes() {
                    self.skip(
                        n.checked_mul(sz)
                            .ok_or_else(|| anyhow!("array byte count overflow: {} * {}", n, sz))?,
                    )?;
                } else {
                    for _ in 0..n {
                        self.skip_value(elem_ty)?;
                    }
                }
            }
            other => {
                let sz = other
                    .scalar_bytes()
                    .expect("scalar types handled; non-scalars matched above");
                self.skip(sz)?;
            }
        }
        Ok(())
    }

    /// Read and (optionally) capture a GGUF metadata value. We capture
    /// only the handful of values we care about — specifically
    /// `general.alignment` (a u32) — and otherwise skip.
    fn read_value_maybe_u32(&mut self, ty: GgufValueType) -> Result<Option<u32>> {
        match ty {
            GgufValueType::U32 => Ok(Some(self.read_u32()?)),
            other => {
                self.skip_value(other)?;
                Ok(None)
            }
        }
    }
}

/// Result of `load_gguf_lenient` — both the successfully-uploaded
/// tensors and the list of tensors whose ggml dtype isn't in the
/// loader's supported set.
pub struct GgufLoad {
    pub tensors: HashMap<String, GgufTensor>,
    /// `(tensor_name, unsupported_dtype_description)` pairs. Entries
    /// correspond to tensors that were parsed out of the file but
    /// skipped at upload time because their ggml dtype isn't in the
    /// loader's supported set (Q2_K, Q3_K, Q5_K, Q6_K, Q8_0, …). The
    /// total descriptor count equals `tensors.len() + unsupported.len()`.
    pub unsupported: Vec<(String, String)>,
    /// Total tensor-descriptor count from the GGUF header (==
    /// `tensors.len() + unsupported.len()`). Stored explicitly so
    /// callers can cross-check against the header even if some
    /// tensors were skipped.
    pub total_descriptors: usize,
}

/// Parse a GGUF v3 container and upload every tensor to the GPU.
///
/// Strict mode: returns `Err` the moment an unsupported ggml dtype
/// is encountered. For models containing non-supported quant types
/// (Q6_K output heads, Q2_K/Q3_K/Q5_K/Q8_0, etc.) use
/// [`load_gguf_lenient`] instead.
///
/// Returns a map keyed by tensor name. Order of upload is the file's
/// declaration order.
pub fn load_gguf<P: AsRef<Path>>(
    device: &Arc<DeviceContext>,
    path: P,
) -> Result<HashMap<String, GgufTensor>> {
    let load = load_gguf_impl(device, path.as_ref(), true)?;
    Ok(load.tensors)
}

/// Lenient variant of [`load_gguf`]: tensors with unsupported ggml
/// dtypes are logged (at `tracing::warn!`), left un-uploaded, and
/// recorded in the returned `unsupported` list. All supported
/// tensors are still uploaded; parse errors (bad header, truncated
/// file) still hard-fail.
pub fn load_gguf_lenient<P: AsRef<Path>>(device: &Arc<DeviceContext>, path: P) -> Result<GgufLoad> {
    load_gguf_impl(device, path.as_ref(), false)
}

fn load_gguf_impl(device: &Arc<DeviceContext>, path: &Path, strict: bool) -> Result<GgufLoad> {
    let file = File::open(path).with_context(|| format!("open gguf file {}", path.display()))?;
    // SAFETY: we mmap read-only; caller must not truncate the file
    // underneath us. The cudarc H2D copy reads from this mapping
    // synchronously (memcpy_stod is a blocking cudaMemcpy), so no
    // after-drop risk.
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("mmap gguf file {}", path.display()))?;

    let mut cur = Cursor::new(&mmap);

    // --- Header -----------------------------------------------------
    let magic = cur.read_u32()?;
    if magic != GGUF_MAGIC {
        return Err(anyhow!(
            "not a gguf file: magic=0x{:08x} (expected 0x{:08x})",
            magic,
            GGUF_MAGIC
        ));
    }
    let version = cur.read_u32()?;
    if version != GGUF_VERSION {
        return Err(anyhow!(
            "unsupported gguf version {} (expected {})",
            version,
            GGUF_VERSION
        ));
    }
    let tensor_count = cur.read_u64()? as usize;
    let metadata_kv_count = cur.read_u64()? as usize;

    // --- Metadata ---------------------------------------------------
    // We only care about one key: `general.alignment`. Everything else
    // is skipped in-place.
    let mut alignment: u64 = DEFAULT_ALIGNMENT;
    for _ in 0..metadata_kv_count {
        let key = cur.read_string()?;
        let ty = GgufValueType::from_u32(cur.read_u32()?)?;
        if key == "general.alignment" {
            if let Some(v) = cur.read_value_maybe_u32(ty)? {
                alignment = v as u64;
            }
        } else {
            cur.skip_value(ty)?;
        }
    }
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(anyhow!(
            "invalid gguf alignment: {} (must be nonzero power of two)",
            alignment
        ));
    }

    // --- Tensor descriptors ----------------------------------------
    struct Descriptor {
        name: String,
        shape: Vec<usize>,
        ggml: GgmlType,
        offset: u64,
    }
    let mut descriptors: Vec<Descriptor> = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = cur.read_string()?;
        let n_dims = cur.read_u32()? as usize;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(cur.read_u64()? as usize);
        }
        let ggml_ty = GgmlType::from_u32(cur.read_u32()?);
        let offset = cur.read_u64()?;
        descriptors.push(Descriptor {
            name,
            shape,
            ggml: ggml_ty,
            offset,
        });
    }

    // --- Align to data region --------------------------------------
    // Tensor data begins at the next `alignment`-aligned offset after
    // the tensor-info section.
    let hdr_end = cur.pos() as u64;
    let data_start = align_up(hdr_end, alignment) as usize;
    if data_start > mmap.len() {
        return Err(anyhow!(
            "gguf data region start ({}) exceeds file size ({})",
            data_start,
            mmap.len()
        ));
    }
    let data_region = &mmap[data_start..];

    // --- Upload each tensor ----------------------------------------
    let mut out = HashMap::with_capacity(tensor_count);
    let mut unsupported: Vec<(String, String)> = Vec::new();
    for d in descriptors {
        let Descriptor {
            name,
            shape,
            ggml,
            offset,
        } = d;

        // GGUF shapes are stored in reverse of row-major dimension
        // order. llama.cpp keeps them in "ggml order" (fastest-moving
        // dim first). Reverse here so the resulting shape matches the
        // logical `[rows, cols, ...]` convention that the rest of the
        // engine expects (`token_embd.weight` => [vocab, hidden]).
        let mut shape_row_major = shape.clone();
        shape_row_major.reverse();

        let n_elems: usize = shape_row_major.iter().product();
        let offset = offset as usize;

        let tensor = match ggml {
            GgmlType::F32 => {
                let slice = typed_slice::<f32>(data_region, offset, n_elems)?;
                let t = CudaTensor::from_host(device.clone(), shape_row_major.clone(), slice)
                    .with_context(|| format!("upload F32 tensor {}", name))?;
                GgufTensor {
                    name: name.clone(),
                    dtype: DType::F32,
                    shape: shape_row_major,
                    buf: GgufBuf::F32(t),
                }
            }
            GgmlType::F16 => {
                let slice = typed_slice::<f16>(data_region, offset, n_elems)?;
                let t = CudaTensor::from_host(device.clone(), shape_row_major.clone(), slice)
                    .with_context(|| format!("upload F16 tensor {}", name))?;
                GgufTensor {
                    name: name.clone(),
                    dtype: DType::F16,
                    shape: shape_row_major,
                    buf: GgufBuf::F16(t),
                }
            }
            GgmlType::BF16 => {
                let slice = typed_slice::<bf16>(data_region, offset, n_elems)?;
                let t = CudaTensor::from_host(device.clone(), shape_row_major.clone(), slice)
                    .with_context(|| format!("upload BF16 tensor {}", name))?;
                GgufTensor {
                    name: name.clone(),
                    dtype: DType::Bf16,
                    shape: shape_row_major,
                    buf: GgufBuf::Bf16(t),
                }
            }
            GgmlType::I32 => {
                let slice = typed_slice::<i32>(data_region, offset, n_elems)?;
                let t = CudaTensor::from_host(device.clone(), shape_row_major.clone(), slice)
                    .with_context(|| format!("upload I32 tensor {}", name))?;
                GgufTensor {
                    name: name.clone(),
                    dtype: DType::I32,
                    shape: shape_row_major,
                    buf: GgufBuf::I32(t),
                }
            }
            GgmlType::I8 => {
                let slice = typed_slice::<i8>(data_region, offset, n_elems)?;
                let t = CudaTensor::from_host(device.clone(), shape_row_major.clone(), slice)
                    .with_context(|| format!("upload I8 tensor {}", name))?;
                GgufTensor {
                    name: name.clone(),
                    dtype: DType::I8,
                    shape: shape_row_major,
                    buf: GgufBuf::I8(t),
                }
            }
            GgmlType::Q4K => {
                // Q4_K_M: 256-wide blocks of 144 bytes each. The last
                // axis must be a multiple of 256 in llama.cpp's quant
                // layout (otherwise this is a config bug upstream).
                let last = *shape_row_major
                    .last()
                    .ok_or_else(|| anyhow!("Q4K tensor {} has zero dimensions", name))?;
                if last % 256 != 0 {
                    return Err(anyhow!(
                        "Q4K tensor {} last-dim {} not a multiple of 256",
                        name,
                        last
                    ));
                }
                let byte_len = DType::Q4K.block_bytes_for_elements(n_elems);
                let slice = byte_slice_as_i8(data_region, offset, byte_len)
                    .with_context(|| format!("slice Q4K bytes for {}", name))?;
                // Build a 1-D [byte_len] i8 tensor of the raw packed
                // bytes; shape at the `GgufTensor` level remains the
                // logical element shape.
                let t = CudaTensor::from_host(device.clone(), vec![byte_len], slice)
                    .with_context(|| format!("upload Q4K tensor {}", name))?;
                GgufTensor {
                    name: name.clone(),
                    dtype: DType::Q4K,
                    shape: shape_row_major,
                    buf: GgufBuf::Q4K(t),
                }
            }
            unsupported_ty => {
                if strict {
                    return Err(anyhow!(
                        "unsupported ggml dtype {:?} for tensor {}",
                        unsupported_ty,
                        name
                    ));
                }
                tracing::warn!(
                    tensor = %name,
                    dtype = ?unsupported_ty,
                    "skipping tensor with unsupported ggml dtype"
                );
                unsupported.push((name.clone(), format!("{:?}", unsupported_ty)));
                continue;
            }
        };

        if out.insert(name.clone(), tensor).is_some() {
            return Err(anyhow!("duplicate tensor name in gguf: {}", name));
        }
    }

    Ok(GgufLoad {
        tensors: out,
        unsupported,
        total_descriptors: tensor_count,
    })
}

/// Round `v` up to the next multiple of `align`. `align` must be a
/// power of two (checked in `load_gguf`).
fn align_up(v: u64, align: u64) -> u64 {
    (v + (align - 1)) & !(align - 1)
}

/// Reinterpret a region of `data` as `&[T]`, checking length and (best-
/// effort) alignment. Unsafe-free since we go through `bytemuck`.
fn typed_slice<T>(data: &[u8], offset: usize, n_elems: usize) -> Result<&[T]>
where
    T: bytemuck::Pod,
{
    let elem_size = std::mem::size_of::<T>();
    let byte_len = n_elems
        .checked_mul(elem_size)
        .ok_or_else(|| anyhow!("byte length overflow: {} * {}", n_elems, elem_size))?;
    let end = offset
        .checked_add(byte_len)
        .ok_or_else(|| anyhow!("offset+len overflow: {} + {}", offset, byte_len))?;
    if end > data.len() {
        return Err(anyhow!(
            "tensor slice out of bounds: [{}, {}) vs data len {}",
            offset,
            end,
            data.len()
        ));
    }
    let bytes = &data[offset..end];
    bytemuck::try_cast_slice::<u8, T>(bytes).map_err(|e| {
        anyhow!(
            "cast {} bytes to [{}; {}]: {}",
            byte_len,
            std::any::type_name::<T>(),
            n_elems,
            e
        )
    })
}

/// Reinterpret a region of `data` as `&[i8]` via a sign-punning cast.
/// Used only for the Q4_K byte-packed payload upload path.
fn byte_slice_as_i8(data: &[u8], offset: usize, byte_len: usize) -> Result<&[i8]> {
    let end = offset
        .checked_add(byte_len)
        .ok_or_else(|| anyhow!("offset+len overflow: {} + {}", offset, byte_len))?;
    if end > data.len() {
        return Err(anyhow!(
            "Q4K slice out of bounds: [{}, {}) vs data len {}",
            offset,
            end,
            data.len()
        ));
    }
    // Transmuting &[u8] -> &[i8] is safe: same size, same alignment,
    // and every bit pattern is a valid i8. bytemuck covers this.
    Ok(bytemuck::cast_slice::<u8, i8>(&data[offset..end]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_rounds_correctly() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
        assert_eq!(align_up(63, 32), 64);
        assert_eq!(align_up(64, 32), 64);
    }

    #[test]
    fn ggml_type_roundtrip_known_codes() {
        assert_eq!(GgmlType::from_u32(0), GgmlType::F32);
        assert_eq!(GgmlType::from_u32(1), GgmlType::F16);
        assert_eq!(GgmlType::from_u32(12), GgmlType::Q4K);
        assert_eq!(GgmlType::from_u32(24), GgmlType::I8);
        assert_eq!(GgmlType::from_u32(26), GgmlType::I32);
        assert_eq!(GgmlType::from_u32(30), GgmlType::BF16);
        matches!(GgmlType::from_u32(9999), GgmlType::Unknown(9999));
    }

    /// Integration test: parses the 27B Q4_K_M GGUF on the A6000 host
    /// and checks tensor count, a known tensor's shape, and the
    /// byte-length invariant for a Q4_K block-packed weight.
    ///
    /// Uses the lenient loader because the "Q4_K_M" variant produced
    /// by llama.cpp mixes in a handful of Q6_K weights (notably the
    /// output head) which this loader intentionally doesn't support.
    ///
    /// Ignored by default — requires the file to exist and a working
    /// CUDA device.
    #[test]
    #[ignore]
    fn load_gguf_27b_q4km_smoke() {
        let path = "/home/metricspace/dflash-ref/dflash/models/Qwen3.5-27B-Q4_K_M.gguf";
        let device = Arc::new(DeviceContext::new(0).expect("init CUDA device 0"));
        let load = load_gguf_lenient(&device, path).expect("load gguf");

        eprintln!(
            "parsed {} total descriptors, uploaded {} tensors, skipped {} unsupported",
            load.total_descriptors,
            load.tensors.len(),
            load.unsupported.len()
        );
        for (name, ty) in &load.unsupported {
            eprintln!("  unsupported: {} ({})", name, ty);
        }
        assert_eq!(
            load.total_descriptors, 851,
            "expected 851 tensor descriptors in 27B Q4_K_M"
        );
        assert_eq!(
            load.tensors.len() + load.unsupported.len(),
            load.total_descriptors,
            "tensors + unsupported should sum to total"
        );

        // Spot check 1: token embedding shape. The Qwen3.5-27B model
        // in /home/metricspace has vocab 248320; hidden dim should be
        // 5120. (An earlier Qwen build had vocab 151936; we check only
        // the hidden-dim invariant to stay robust to vocab resizes.)
        let tok_embd = load
            .tensors
            .get("token_embd.weight")
            .expect("token_embd.weight not found");
        eprintln!(
            "token_embd.weight: dtype={:?} shape={:?}",
            tok_embd.dtype, tok_embd.shape
        );
        assert_eq!(
            tok_embd.shape.len(),
            2,
            "token_embd.weight should be 2-D, got {:?}",
            tok_embd.shape
        );
        assert_eq!(
            tok_embd.shape[1], 5120,
            "token_embd.weight hidden-dim {} != 5120",
            tok_embd.shape[1]
        );
        assert!(
            tok_embd.shape[0] > 0,
            "token_embd.weight vocab dim must be positive"
        );

        // Spot check 2: pick any Q4K tensor and verify the packed
        // byte-count invariant: n_blocks * 144 B where n_blocks is
        // n_elems / 256. The task description pointed at
        // `blk.0.attn_q.weight`, but this specific Qwen3.5-27B fuses
        // attention into `blk.*.attn_qkv.weight` (which is Q5_K —
        // unsupported). We fall back to scanning for any Q4K tensor.
        let q4k_samples: Vec<(&String, &GgufTensor)> = load
            .tensors
            .iter()
            .filter(|(_, t)| t.dtype == DType::Q4K)
            .collect();
        assert!(
            !q4k_samples.is_empty(),
            "expected at least one Q4K tensor in the 27B Q4_K_M file"
        );
        eprintln!("found {} Q4K tensors", q4k_samples.len());
        // Check the first one deterministically (by sorted name) so
        // the test output is stable across HashMap iteration orders.
        let mut sorted: Vec<_> = q4k_samples;
        sorted.sort_by_key(|(n, _)| (*n).clone());
        let (q_name, q) = sorted[0];
        eprintln!("spot-check Q4K tensor {}: shape={:?}", q_name, q.shape);
        let n_elems: usize = q.shape.iter().product();
        assert!(
            n_elems % 256 == 0,
            "Q4K tensor {} n_elems {} not divisible by 256",
            q_name,
            n_elems
        );
        let expected_bytes = (n_elems / 256) * 144;
        match q.buf {
            GgufBuf::Q4K(ref t) => {
                assert_eq!(
                    t.numel(),
                    expected_bytes,
                    "Q4K byte count mismatch for {}",
                    q_name
                );
            }
            _ => panic!("{} marked Q4K but buf variant is wrong", q_name),
        }
    }
}
