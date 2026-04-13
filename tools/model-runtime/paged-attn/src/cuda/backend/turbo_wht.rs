use crate::cuda::backend::slice_ptr;
#[cfg(feature = "cuda")]
use crate::cuda::ffi;
use candle_core as candle;
use candle_core::{DType, Result, Shape, Tensor};

#[derive(Debug, Clone)]
struct TurboRotate {
    forward: bool,
}

impl candle::CustomOp1 for TurboRotate {
    fn name(&self) -> &'static str {
        "turbo-rotate"
    }

    fn cpu_fwd(
        &self,
        _: &candle::CpuStorage,
        _: &candle::Layout,
    ) -> Result<(candle::CpuStorage, Shape)> {
        candle::bail!("turbo rotate is not implemented on CPU")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        src: &candle::CudaStorage,
        layout: &candle::Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use core::ffi::c_void;

        let dev = src.device();
        let shape = layout.shape().clone();
        let elem_count = shape.elem_count();
        if elem_count % 128 != 0 {
            candle::bail!("turbo rotate expects element count multiple of 128, got {elem_count}");
        }

        let dtype = src.dtype();
        let (src_ptr, _src_guard) = match dtype {
            DType::F16 => slice_ptr(src.as_cuda_slice::<half::f16>()?, layout.start_offset()),
            DType::BF16 => slice_ptr(src.as_cuda_slice::<half::bf16>()?, layout.start_offset()),
            DType::F32 => slice_ptr(src.as_cuda_slice::<f32>()?, layout.start_offset()),
            _ => candle::bail!("turbo rotate unsupported storage dtype {dtype:?}"),
        };

        match dtype {
            DType::F16 => {
                let out = unsafe { dev.alloc::<half::f16>(elem_count) }?;
                let (out_ptr, out_guard) = out.device_ptr(out.stream());
                unsafe {
                    ffi::turbo_rotate_f16(
                        src_ptr as *const c_void,
                        out_ptr as *mut c_void,
                        elem_count as i64,
                        dev.cuda_stream().cu_stream(),
                        self.forward,
                    );
                }
                drop(out_guard);
                Ok((
                    candle::CudaStorage::wrap_cuda_slice(out, dev.clone()),
                    shape,
                ))
            }
            DType::BF16 => {
                let out = unsafe { dev.alloc::<half::bf16>(elem_count) }?;
                let (out_ptr, out_guard) = out.device_ptr(out.stream());
                unsafe {
                    ffi::turbo_rotate_bf16(
                        src_ptr as *const c_void,
                        out_ptr as *mut c_void,
                        elem_count as i64,
                        dev.cuda_stream().cu_stream(),
                        self.forward,
                    );
                }
                drop(out_guard);
                Ok((
                    candle::CudaStorage::wrap_cuda_slice(out, dev.clone()),
                    shape,
                ))
            }
            DType::F32 => {
                let out = unsafe { dev.alloc::<f32>(elem_count) }?;
                let (out_ptr, out_guard) = out.device_ptr(out.stream());
                unsafe {
                    ffi::turbo_rotate_f32(
                        src_ptr as *const c_void,
                        out_ptr as *mut c_void,
                        elem_count as i64,
                        dev.cuda_stream().cu_stream(),
                        self.forward,
                    );
                }
                drop(out_guard);
                Ok((
                    candle::CudaStorage::wrap_cuda_slice(out, dev.clone()),
                    shape,
                ))
            }
            other => candle::bail!("turbo rotate unsupported dtype {other:?}"),
        }
    }
}

pub fn turbo_rotate(t: &Tensor, forward: bool) -> Result<Tensor> {
    t.apply_op1(TurboRotate { forward })
}
