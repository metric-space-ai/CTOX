//! Build script for `ctox-qwen35-27b-q4km-dflash`.
//!
//! Links the crate against ggml + ggml-cuda. Two paths:
//!
//!  1. **GGML_LIB_DIR set** — link against pre-built ggml `.so` files in
//!     that directory. This is the dev-box / CI path: the lucebox-built
//!     `build/deps/llama.cpp/ggml/src/` tree has all four libraries
//!     (`libggml-base.so`, `libggml.so`, `libggml-cpu.so`, and
//!     `libggml-cuda.so` in `ggml-cuda/`). We also compile
//!     `vendor/ggml-cuda/f16_convert.cu` so
//!     `dflash27b_launch_{f16,bf16}_to_f32` resolve without the reference's
//!     `libdflash27b.a`.
//!
//!  2. **GGML_LIB_DIR unset** — emit a warning and no link directives.
//!     `cargo check` still passes on a host without the build tree, so the
//!     Rust surface compiles and trips no link step.
//!
//! The crate intentionally does NOT compile the 62 `.cu` files in
//! `vendor/ggml-cuda/` from source. That path is documented in
//! `vendor/llama-cpp.version` and handled by the lucebox reference build;
//! making the crate self-host-compile would pull in ~6K lines of CMake
//! logic (GGML_* CACHE vars, backend dispatch, HIP/ROCm detection, etc.)
//! from llama.cpp's `ggml/src/CMakeLists.txt`. The version pin in
//! `vendor/llama-cpp.version` plus the GGML_LIB_DIR contract is the
//! stable integration point.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=GGML_LIB_DIR");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CTOX_CUDA_SM");

    let cuda_feature = env::var("CARGO_FEATURE_CUDA").is_ok();
    let ggml_lib_dir = env::var("GGML_LIB_DIR").ok();

    // `cargo check` on a non-CUDA host without GGML_LIB_DIR: just compile
    // the Rust surface and bail out of linker setup.
    if !cuda_feature && ggml_lib_dir.is_none() {
        println!(
            "cargo:warning=ctox-qwen35-27b-q4km-dflash: cuda feature off and GGML_LIB_DIR unset \
             — skipping native compile + link (host-only build). Runtime will fail."
        );
        return;
    }

    if let Some(dir) = ggml_lib_dir {
        link_ggml(&PathBuf::from(dir));
    } else {
        println!(
            "cargo:warning=ctox-qwen35-27b-q4km-dflash: cuda feature on but GGML_LIB_DIR unset. \
             Set GGML_LIB_DIR to the lucebox build tree's \
             `build/deps/llama.cpp/ggml/src/` path. Link step will fail."
        );
    }

    compile_f16_convert();

    // Linux + CUDA needs libstdc++ because nvcc-emitted objects reference
    // C++ runtime symbols (`__cxa_guard_*`, `__gxx_personality_v0`). On
    // macOS + Metal we don't hit this path; the warning above fires first.
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

/// Emit `cargo:rustc-link-*` directives for the pre-built ggml `.so` set.
fn link_ggml(base: &PathBuf) {
    println!("cargo:rustc-link-search=native={}", base.display());
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-cpu");

    // ggml-cuda lives one level deeper on the lucebox tree.
    let cuda_subdir = base.join("ggml-cuda");
    if cuda_subdir.is_dir() {
        println!("cargo:rustc-link-search=native={}", cuda_subdir.display());
        println!("cargo:rustc-link-lib=dylib=ggml-cuda");
        println!("cargo:rustc-link-lib=dylib=cudart");
    } else {
        println!(
            "cargo:warning=ctox-qwen35-27b-q4km-dflash: expected `ggml-cuda/` subdir under {} \
             but it is missing. Link step will fail if --features=cuda.",
            base.display()
        );
    }
}

/// Compile `vendor/ggml-cuda/f16_convert.cu` into a static archive and
/// emit link directives. Byte-for-byte copy of `lucebox/dflash/src/f16_convert.cu`.
///
/// `NVCC` env var overrides the binary location; `CTOX_CUDA_SM` picks the
/// SM capability (default 86 — matches lucebox's hardcoded `CUDA_ARCHITECTURES "86"`).
fn compile_f16_convert() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let src = manifest.join("vendor/ggml-cuda/f16_convert.cu");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let obj = out_dir.join("f16_convert.o");
    let lib = out_dir.join("libctox_f16_convert.a");

    println!("cargo:rerun-if-changed={}", src.display());

    if !src.exists() {
        println!(
            "cargo:warning=ctox-qwen35-27b-q4km-dflash: vendor/ggml-cuda/f16_convert.cu \
             missing at {} — skipping (host-only build)",
            src.display()
        );
        return;
    }

    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".into());
    let sm = env::var("CTOX_CUDA_SM").unwrap_or_else(|_| "86".into());

    let nvcc_status = Command::new(&nvcc)
        .args([
            "--compile",
            "-arch",
            &format!("sm_{sm}"),
            "-std=c++17",
            "-O3",
            "-Xcompiler",
            "-fPIC",
            "-o",
        ])
        .arg(&obj)
        .arg(&src)
        .status();

    match nvcc_status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            println!(
                "cargo:warning=ctox-qwen35-27b-q4km-dflash: nvcc failed for f16_convert.cu: \
                 exit {s} — skipping"
            );
            return;
        }
        Err(e) => {
            println!(
                "cargo:warning=ctox-qwen35-27b-q4km-dflash: nvcc not available ({e}) — \
                 skipping f16_convert.cu compile. Fine for `cargo check` on non-CUDA hosts."
            );
            return;
        }
    }

    let ar = env::var("AR").unwrap_or_else(|_| "ar".into());
    let ar_status = Command::new(&ar)
        .args(["rcs"])
        .arg(&lib)
        .arg(&obj)
        .status();
    match ar_status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            println!(
                "cargo:warning=ctox-qwen35-27b-q4km-dflash: ar failed building \
                 libctox_f16_convert.a: exit {s}"
            );
            return;
        }
        Err(e) => {
            println!(
                "cargo:warning=ctox-qwen35-27b-q4km-dflash: ar not available ({e}) — \
                 skipping archive step"
            );
            return;
        }
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=ctox_f16_convert");
}
