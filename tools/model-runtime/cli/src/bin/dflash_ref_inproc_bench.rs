//! In-process reference bench: dlopen the reference's `libdflash_run_lib.so`
//! and invoke its `dflash_run_main_c(argc, argv)` entry directly — no
//! subprocess, no file I/O for prompt beyond what `test_dflash` itself
//! needs (prompt.bin + out.bin).
//!
//! This is the concrete answer to "why not just use the reference kernels
//! in-process": one dlopen, one extern "C" call.
//!
//! Two invocation styles:
//!
//!   1. Single-shot at one context size (default):
//!        dflash-ref-inproc-bench --synth-prompt-len 32768 --n-tokens 256 --ddtree
//!
//!   2. Sweep across sizes — prints a table at the end:
//!        dflash-ref-inproc-bench --sweep --ddtree
//!
//!      Sweep default set: 1024, 4096, 16384, 32768, 65536, 120000.
//!      Override with --sweep-lens=1024,8192,65536.
//!
//! Purpose of the sweep: prove the FFI path holds its tok/s all the way
//! out to ~128K context (reference's --max-ctx ceiling), so we know the
//! binding is a legitimate production intermediate, not just a toy that
//! works at HumanEval-sized prompts.

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::path::{Path, PathBuf};

/// HumanEval sample prompt — the 9 tokens we use as the base repeating
/// pattern when synthesizing long prompts. Matches the ids used in
/// `dflash-ref-bench` (`--prompt-ids` default).
const BASE_PROMPT_IDS: &[i32] = &[7734, 264, 6185, 36974, 883, 13094, 6326, 61369, 25];

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

    /// Prompt token ids file (little-endian i32). Ignored when
    /// --synth-prompt-len or --sweep is set — those synthesize the
    /// file into /tmp.
    #[arg(long, default_value = "/tmp/humeval_prompt.bin")]
    prompt_bin: PathBuf,

    /// Output token ids file (little-endian i32) — written by library.
    #[arg(long, default_value = "/tmp/ref_inproc_out.bin")]
    out_bin: PathBuf,

    /// How many new tokens to generate per run.
    #[arg(long, default_value_t = 256)]
    n_tokens: usize,

    /// Enable DDTree tree verify.
    #[arg(long)]
    ddtree: bool,

    /// DDTree budget.
    #[arg(long, default_value_t = 22)]
    ddtree_budget: usize,

    /// Max KV-cache context (passed as --max-ctx=N to the library).
    /// Automatically raised to prompt_len + n_tokens + 256 in sweep /
    /// synth mode.
    #[arg(long, default_value_t = 4096)]
    max_ctx: usize,

    /// Single-shot: synthesize a prompt.bin of this many tokens by
    /// repeating BASE_PROMPT_IDS, write to /tmp/synth_prompt_<len>.bin,
    /// and point the library at it.
    #[arg(long)]
    synth_prompt_len: Option<usize>,

    /// Sweep mode: run at each prompt length in --sweep-lens and print
    /// a summary table. Overrides --synth-prompt-len.
    #[arg(long)]
    sweep: bool,

    /// Comma-separated prompt lengths for --sweep. Default:
    /// "1024,4096,16384,32768,65536,120000".
    #[arg(long, default_value = "1024,4096,16384,32768,65536,120000")]
    sweep_lens: String,
}

/// Synthesize a prompt of exactly `len` tokens by repeating
/// BASE_PROMPT_IDS, write it as little-endian i32 to `path`. The output
/// is semantically garbage (the LLM will produce nonsense completions)
/// but is valid BPE and exercises the full KV-cache footprint, which
/// is all we need for a throughput+scaling bench.
fn synth_prompt(path: &Path, len: usize) -> Result<()> {
    let mut bytes: Vec<u8> = Vec::with_capacity(len * 4);
    for i in 0..len {
        let tok = BASE_PROMPT_IDS[i % BASE_PROMPT_IDS.len()];
        bytes.extend_from_slice(&tok.to_le_bytes());
    }
    std::fs::write(path, &bytes).with_context(|| format!("write synth prompt to {:?}", path))?;
    Ok(())
}

/// One FFI invocation. Returns (exit_code, wall_seconds, out_token_count).
fn invoke_once(
    entry: &libloading::Symbol<
        unsafe extern "C" fn(c_int, *const *const c_char) -> c_int,
    >,
    target_gguf: &Path,
    draft_st: &Path,
    prompt_bin: &Path,
    out_bin: &Path,
    n_tokens: usize,
    ddtree: bool,
    ddtree_budget: usize,
    max_ctx: usize,
) -> Result<(i32, f64, usize)> {
    let mut argv_strings: Vec<CString> = Vec::new();
    argv_strings.push(CString::new("dflash-ref-inproc")?);
    argv_strings.push(CString::new(target_gguf.to_str().ok_or(anyhow!("target path not utf8"))?)?);
    argv_strings.push(CString::new(draft_st.to_str().ok_or(anyhow!("draft path not utf8"))?)?);
    argv_strings.push(CString::new(prompt_bin.to_str().ok_or(anyhow!("prompt path not utf8"))?)?);
    argv_strings.push(CString::new(format!("{}", n_tokens))?);
    argv_strings.push(CString::new(out_bin.to_str().ok_or(anyhow!("out path not utf8"))?)?);
    if ddtree {
        argv_strings.push(CString::new("--ddtree")?);
        argv_strings.push(CString::new(format!("--ddtree-budget={}", ddtree_budget))?);
    }
    argv_strings.push(CString::new(format!("--max-ctx={}", max_ctx))?);

    let mut argv_ptrs: Vec<*const c_char> =
        argv_strings.iter().map(|s| s.as_ptr()).collect();
    argv_ptrs.push(std::ptr::null());

    eprintln!(
        "invoking dflash_run_main_c: {}",
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
    let elapsed = t_start.elapsed().as_secs_f64();

    let out_count = match std::fs::read(out_bin) {
        Ok(b) => b.len() / 4,
        Err(_) => 0,
    };

    Ok((rc, elapsed, out_count))
}

#[derive(Debug, Clone)]
struct SweepRow {
    prompt_len: usize,
    max_ctx: usize,
    rc: i32,
    wall_s: f64,
    out_tokens: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let lib = unsafe { libloading::Library::new(&args.lib) }
        .with_context(|| format!("dlopen {:?}", args.lib))?;

    let entry: libloading::Symbol<
        unsafe extern "C" fn(c_int, *const *const c_char) -> c_int,
    > = unsafe { lib.get(b"dflash_run_main_c\0") }
        .context("dlsym dflash_run_main_c")?;

    // Decide mode.
    let sweep_lens: Vec<usize> = if args.sweep {
        args.sweep_lens
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<usize>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("parse --sweep-lens")?
    } else if let Some(n) = args.synth_prompt_len {
        vec![n]
    } else {
        // Plain single-shot against whatever --prompt-bin already is.
        let (rc, wall, out_n) = invoke_once(
            &entry,
            &args.target_gguf,
            &args.draft_st,
            &args.prompt_bin,
            &args.out_bin,
            args.n_tokens,
            args.ddtree,
            args.ddtree_budget,
            args.max_ctx,
        )?;
        println!("\n=== IN-PROCESS REF RUN ===");
        println!("exit code: {rc}");
        println!("wall time (load + generate): {:.3}s", wall);
        if rc != 0 {
            return Err(anyhow!("dflash_run_main_c returned {rc}"));
        }
        let bytes = std::fs::read(&args.out_bin)
            .with_context(|| format!("read {:?}", args.out_bin))?;
        let n = bytes.len() / 4;
        let mut first16 = Vec::with_capacity(16);
        for i in 0..n.min(16) {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[i * 4..i * 4 + 4]);
            first16.push(i32::from_le_bytes(b));
        }
        println!("output tokens: {n} total, first 16: {first16:?}  (out-tokens read = {})", out_n);
        return Ok(());
    };

    // Synth-or-sweep path: for each len, synthesize prompt, run, collect.
    let mut rows: Vec<SweepRow> = Vec::with_capacity(sweep_lens.len());
    for &plen in &sweep_lens {
        let prompt_path: PathBuf = format!("/tmp/synth_prompt_{}.bin", plen).into();
        synth_prompt(&prompt_path, plen)?;
        // Max-ctx must fit prompt + generated; add headroom for DDTree
        // tree nodes (budget=22 per step × ~n_tokens/2 steps ≈ a few k).
        let tree_headroom = if args.ddtree { args.ddtree_budget * 512 } else { 256 };
        let max_ctx = plen + args.n_tokens + tree_headroom;
        eprintln!(
            "\n--- sweep: prompt_len={} max_ctx={} n_tokens={} ddtree={} ---",
            plen, max_ctx, args.n_tokens, args.ddtree
        );
        let (rc, wall, out_n) = invoke_once(
            &entry,
            &args.target_gguf,
            &args.draft_st,
            &prompt_path,
            &args.out_bin,
            args.n_tokens,
            args.ddtree,
            args.ddtree_budget,
            max_ctx,
        )?;
        rows.push(SweepRow {
            prompt_len: plen,
            max_ctx,
            rc,
            wall_s: wall,
            out_tokens: out_n,
        });
        if rc != 0 {
            eprintln!(
                "run FAILED at prompt_len={}: rc={}  (continuing with remaining sizes)",
                plen, rc
            );
        }
    }

    // Summary table.
    println!("\n=== DFLASH-REF-INPROC SWEEP ===");
    println!(
        "ddtree={}  n_tokens={}  lib={}",
        args.ddtree,
        args.n_tokens,
        args.lib.display()
    );
    println!(
        "{:>10}  {:>10}  {:>8}  {:>10}  {:>10}  {:>10}",
        "prompt_len", "max_ctx", "rc", "wall_s", "out_toks", "wall_tok/s"
    );
    println!("{}", "-".repeat(70));
    for r in &rows {
        // "wall_tok/s" is (out_tokens / wall_s) — includes model-load
        // cost, so it's pessimistic vs the reference's reported "decode
        // tok/s". Still useful as a scaling shape indicator; the
        // reference's own stderr line [dflash] generated N tokens in T s
        // gives the pure decode rate in each run's log above.
        let tps = if r.wall_s > 0.0 {
            r.out_tokens as f64 / r.wall_s
        } else {
            0.0
        };
        println!(
            "{:>10}  {:>10}  {:>8}  {:>10.2}  {:>10}  {:>10.2}",
            r.prompt_len, r.max_ctx, r.rc, r.wall_s, r.out_tokens, tps
        );
    }
    println!(
        "\nNote: wall_tok/s includes model-load time (seconds to tens of seconds)."
    );
    println!(
        "      Pure decode tok/s is in each run's [dflash] stderr line."
    );
    Ok(())
}
