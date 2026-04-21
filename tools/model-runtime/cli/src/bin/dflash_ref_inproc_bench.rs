//! In-process reference bench: dlopen the reference's `libdflash_run_lib.so`
//! and invoke its `dflash_run_main_c(argc, argv)` entry directly — no
//! subprocess, no file I/O for prompt (still internal to test_dflash
//! which reads prompt.bin, but everything else stays in one process).
//!
//! This is the concrete answer to "why not just use the reference kernels
//! in-process": one dlopen, one extern "C" call.

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    /// Path to the built libdflash_run_lib.so.
    #[arg(long, default_value = "/home/metricspace/dflash-ref/dflash/build/libdflash_run_lib.so")]
    lib: PathBuf,

    /// Path to the Q4_K_M GGUF target.
    #[arg(long, default_value = "/home/metricspace/dflash-ref/dflash/models/Qwen3.5-27B-Q4_K_M.gguf")]
    target_gguf: PathBuf,

    /// Path to the DFlash draft safetensors.
    #[arg(long, default_value = "/home/metricspace/dflash-ref/dflash/models/draft/model.safetensors")]
    draft_st: PathBuf,

    /// Prompt token ids file (little-endian i32).
    #[arg(long, default_value = "/tmp/humeval_prompt.bin")]
    prompt_bin: PathBuf,

    /// Output token ids file (little-endian i32) — written by library.
    #[arg(long, default_value = "/tmp/ref_inproc_out.bin")]
    out_bin: PathBuf,

    /// How many new tokens to generate.
    #[arg(long, default_value_t = 256)]
    n_tokens: usize,

    /// Enable DDTree tree verify.
    #[arg(long)]
    ddtree: bool,

    /// DDTree budget.
    #[arg(long, default_value_t = 22)]
    ddtree_budget: usize,

    /// Max KV-cache context (passed as --max-ctx=N to the library).
    #[arg(long, default_value_t = 4096)]
    max_ctx: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // dlopen the reference library.
    // Use libloading via direct dlopen system calls to avoid an extra crate
    // dependency — Rust's std::env sets LD_LIBRARY_PATH but libdl::dlopen
    // handles the actual load.
    let lib = unsafe { libloading::Library::new(&args.lib) }
        .with_context(|| format!("dlopen {:?}", args.lib))?;

    // Symbol lookup: `extern "C" int dflash_run_main_c(int argc,
    //                 const char *const *argv)`
    let entry: libloading::Symbol<
        unsafe extern "C" fn(c_int, *const *const c_char) -> c_int,
    > = unsafe { lib.get(b"dflash_run_main_c\0") }
        .context("dlsym dflash_run_main_c")?;

    // Build argv as if we were invoking test_dflash at the CLI:
    //   <prog> <target.gguf> <draft.st> <prompt.bin> <n_gen> <out.bin>
    //        [--ddtree --ddtree-budget=B] [--max-ctx=N]
    let mut argv_strings: Vec<CString> = Vec::new();
    argv_strings.push(CString::new("dflash-ref-inproc")?);
    argv_strings.push(CString::new(args.target_gguf.to_str().ok_or(anyhow!("target path not utf8"))?)?);
    argv_strings.push(CString::new(args.draft_st.to_str().ok_or(anyhow!("draft path not utf8"))?)?);
    argv_strings.push(CString::new(args.prompt_bin.to_str().ok_or(anyhow!("prompt path not utf8"))?)?);
    argv_strings.push(CString::new(format!("{}", args.n_tokens))?);
    argv_strings.push(CString::new(args.out_bin.to_str().ok_or(anyhow!("out path not utf8"))?)?);
    if args.ddtree {
        argv_strings.push(CString::new("--ddtree")?);
        argv_strings.push(CString::new(format!("--ddtree-budget={}", args.ddtree_budget))?);
    }
    argv_strings.push(CString::new(format!("--max-ctx={}", args.max_ctx))?);

    // Raw pointer array.
    let mut argv_ptrs: Vec<*const c_char> =
        argv_strings.iter().map(|s| s.as_ptr()).collect();
    argv_ptrs.push(std::ptr::null());

    eprintln!(
        "invoking dflash_run_main_c with {} args ({})",
        argv_strings.len(),
        argv_strings
            .iter()
            .map(|s| s.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join(" ")
    );

    let t_start = std::time::Instant::now();
    let rc = unsafe {
        entry(
            argv_strings.len() as c_int,
            argv_ptrs.as_ptr() as *const *const c_char,
        )
    };
    let elapsed = t_start.elapsed();

    println!("\n=== IN-PROCESS REF RUN ===");
    println!("exit code: {rc}");
    println!("wall time (load + generate): {:.3}s", elapsed.as_secs_f64());

    if rc != 0 {
        return Err(anyhow!("dflash_run_main_c returned {rc}"));
    }

    // Read the output token file the library wrote.
    let bytes = std::fs::read(&args.out_bin).with_context(|| format!("read {:?}", args.out_bin))?;
    let n = bytes.len() / 4;
    let mut first16 = Vec::with_capacity(16);
    for i in 0..n.min(16) {
        let mut b = [0u8; 4];
        b.copy_from_slice(&bytes[i * 4..i * 4 + 4]);
        first16.push(i32::from_le_bytes(b));
    }
    println!("output tokens: {n} total, first 16: {first16:?}");

    Ok(())
}
