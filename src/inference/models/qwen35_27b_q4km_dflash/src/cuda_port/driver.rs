//! Raw CUDA Driver API FFI bindings — minimum subset needed to load
//! PTX modules and launch kernels from Rust without linking against
//! `libggml-cuda.so`.
//!
//! Only the OS CUDA runtime (`libcuda.so`, part of the NVIDIA driver
//! stack) is linked. That is the operating-system boundary, not a
//! library dependency per CLAUDE.md's Inference-Engine Architecture
//! rules.
//!
//! Naming follows the upstream CUDA driver API verbatim; Rust error
//! codes mirror the C enum values so existing sample code translates
//! one-to-one.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use libc::{c_char, c_int, c_uint, c_void, size_t};

// ─── Opaque handle types ──────────────────────────────────────────
//
// All are `*mut c_void` / opaque-pointer-sized on the C side. We
// box them in `#[repr(transparent)]` newtypes so the type system
// stops us from accidentally mixing a CUmodule with a CUfunction.

pub type CUresult = c_int;

/// `CUDA_SUCCESS`.
pub const CUDA_SUCCESS: CUresult = 0;

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct CUmodule(pub *mut c_void);

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct CUfunction(pub *mut c_void);

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct CUstream(pub *mut c_void);

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct CUdeviceptr(pub u64);

// ─── Minimum subset of the Driver API ─────────────────────────────

#[link(name = "cuda")]
extern "C" {
    /// `cuInit(flags)` — one-time driver init. Safe to call repeatedly.
    pub fn cuInit(Flags: c_uint) -> CUresult;

    /// `cuModuleLoadData(*mut CUmodule, const void *image)` — load a
    /// module from an in-memory PTX or cubin image (NUL-terminated
    /// PTX is the common case).
    pub fn cuModuleLoadData(
        module: *mut CUmodule,
        image: *const c_void,
    ) -> CUresult;

    /// `cuModuleUnload(CUmodule)`
    pub fn cuModuleUnload(module: CUmodule) -> CUresult;

    /// `cuModuleGetFunction(*mut CUfunction, CUmodule, const char *)`
    /// — look up a kernel function by its mangled C++ name.
    pub fn cuModuleGetFunction(
        function: *mut CUfunction,
        module: CUmodule,
        name: *const c_char,
    ) -> CUresult;

    /// `cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX,
    ///                 blockDimY, blockDimZ, sharedMemBytes, stream,
    ///                 kernelParams, extra)`
    pub fn cuLaunchKernel(
        f: CUfunction,
        gridDimX: c_uint,
        gridDimY: c_uint,
        gridDimZ: c_uint,
        blockDimX: c_uint,
        blockDimY: c_uint,
        blockDimZ: c_uint,
        sharedMemBytes: c_uint,
        hStream: CUstream,
        kernelParams: *const *const c_void,
        extra: *const *const c_void,
    ) -> CUresult;

    /// `cuGetErrorString(CUresult, const char **)` — human-readable
    /// error for a failed driver-API call.
    pub fn cuGetErrorString(
        error: CUresult,
        pStr: *mut *const c_char,
    ) -> CUresult;

    /// `cuMemAlloc(*mut CUdeviceptr, bytesize)` — device allocation.
    pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: size_t) -> CUresult;

    /// `cuMemFree(CUdeviceptr)`
    pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;

    /// `cuStreamSynchronize(CUstream)` — block until the stream's
    /// queued work is done.
    pub fn cuStreamSynchronize(hStream: CUstream) -> CUresult;
}

/// Pretty-print a `CUresult` error via `cuGetErrorString`.
pub fn error_string(result: CUresult) -> String {
    if result == CUDA_SUCCESS {
        return "CUDA_SUCCESS".to_string();
    }
    let mut raw: *const c_char = std::ptr::null();
    let rc = unsafe { cuGetErrorString(result, &mut raw) };
    if rc != CUDA_SUCCESS || raw.is_null() {
        return format!("CUresult({result}) (cuGetErrorString failed)");
    }
    let cstr = unsafe { std::ffi::CStr::from_ptr(raw) };
    cstr.to_string_lossy().into_owned()
}
