//! `CudaTensor<T>` — owned device buffer + shape/stride/dtype.
//!
//! Intentionally NOT an op-bearing type. Operations are kernel
//! launches that accept `&CudaTensor` inputs and an `&mut CudaTensor`
//! output. This keeps the model-forward code explicit about what
//! happens (no hidden `to_dtype` casts, no lazy graphs, no overload
//! dispatch) which is the whole reason we're leaving candle.
//!
//! Only row-major (C-order) storage is supported. If we ever need
//! column-major views, add a `layout: Layout` field — don't overload
//! `stride` to fake it.

use std::marker::PhantomData;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use cudarc::driver::{CudaSlice, DeviceRepr};

use crate::device::DeviceContext;
use crate::dtype::{DType, DTypeTrait};

/// Shape = N-dimensional row-major extents. Stored as a small Vec so
/// we don't hard-cap rank; in practice everything the engine uses is
/// rank-1 to rank-4.
pub type Shape = Vec<usize>;

/// Stride = element-level stride per axis (not byte stride). For
/// a non-transposed tensor, stride[i] = product of shape[i+1..].
pub type Stride = Vec<usize>;

/// Owned CUDA tensor. `T` is the element scalar type (binds at the
/// type level to a runtime `DType` via `DTypeTrait`).
pub struct CudaTensor<T: DTypeTrait + DeviceRepr> {
    buf: CudaSlice<T>,
    shape: Shape,
    stride: Stride,
    device: Arc<DeviceContext>,
    _marker: PhantomData<T>,
}

impl<T: DTypeTrait + DeviceRepr> CudaTensor<T> {
    /// Allocate zeroed storage for `shape`.
    pub fn zeros(device: Arc<DeviceContext>, shape: Shape) -> Result<Self> {
        let n_elems = shape.iter().product::<usize>();
        let stream = device.raw().default_stream();
        let buf = stream
            .alloc_zeros::<T>(n_elems)
            .with_context(|| format!("alloc_zeros({} elems) on device {}", n_elems, device.ordinal()))?;
        let stride = default_stride(&shape);
        Ok(Self {
            buf,
            shape,
            stride,
            device,
            _marker: PhantomData,
        })
    }

    /// Upload host slice into a fresh device tensor with the given
    /// shape. `host.len()` must match `shape.iter().product()`.
    pub fn from_host(
        device: Arc<DeviceContext>,
        shape: Shape,
        host: &[T],
    ) -> Result<Self> {
        let n_elems = shape.iter().product::<usize>();
        if host.len() != n_elems {
            return Err(anyhow!(
                "from_host: host.len()={} != shape.product()={}",
                host.len(),
                n_elems
            ));
        }
        let stream = device.raw().default_stream();
        let buf = stream
            .memcpy_stod(host)
            .with_context(|| format!("memcpy_stod {} elems → device", n_elems))?;
        let stride = default_stride(&shape);
        Ok(Self {
            buf,
            shape,
            stride,
            device,
            _marker: PhantomData,
        })
    }

    /// Number of logical elements (product of shape).
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Runtime dtype tag.
    pub fn dtype(&self) -> DType {
        T::DTYPE
    }

    /// Shape accessor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Stride accessor.
    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    /// Device handle.
    pub fn device(&self) -> &Arc<DeviceContext> {
        &self.device
    }

    /// Raw device buffer — for kernel launches. Do not store this
    /// outside of a single kernel dispatch; the `CudaSlice` is
    /// tied to this tensor's lifetime.
    pub fn buf(&self) -> &CudaSlice<T> {
        &self.buf
    }

    /// Mutable raw buffer for writing kernel output.
    pub fn buf_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.buf
    }

    /// Download to host. Used by bench tooling, diff checkers, and
    /// logits readout at the end of forward. NOT a hot-path call —
    /// implies a stream sync.
    pub fn to_host(&self) -> Result<Vec<T>> {
        let stream = self.device.raw().default_stream();
        let v = stream
            .memcpy_dtov(&self.buf)
            .with_context(|| format!("memcpy_dtov {} elems", self.numel()))?;
        Ok(v)
    }
}

impl<T: DTypeTrait + DeviceRepr> std::fmt::Debug for CudaTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaTensor")
            .field("dtype", &T::DTYPE)
            .field("shape", &self.shape)
            .field("stride", &self.stride)
            .field("dev", &self.device.ordinal())
            .finish()
    }
}

/// Default row-major stride for a shape. For shape `[a, b, c]` returns
/// `[b*c, c, 1]`.
fn default_stride(shape: &[usize]) -> Stride {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

#[cfg(test)]
mod tests {
    use super::default_stride;

    #[test]
    fn stride_matches_row_major() {
        assert_eq!(default_stride(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(default_stride(&[5]), vec![1]);
        assert_eq!(default_stride(&[]), Vec::<usize>::new());
    }
}
