//! Persistent-runtime DFlash bench.
//!
//! Exercises the new `dflash_ctx_init / dflash_ctx_generate / dflash_ctx_free`
//! API via the safe `ctox-dflash-ffi` wrapper: one model load, N
//! generate calls on different prompts, per-call tok/s measured by the
//! library (excludes model load).
//!
//! This is the concrete answer to "does init/step/free actually give us
//! what we want": the argv bench paid 10-30 s per invocation for a
//! fresh model load; here the load is amortized over all generate calls
//! and the per-call tok/s should match the second invocation onwards.
//!
//! Build:
//!   cargo build --release -p ctox-engine-cli --bin dflash-persistent-bench
//!
//! Run:
//!   target/release/dflash-persistent-bench \
//!       --lib /home/metricspace/dflash-ref/dflash/build/libdflash_run_lib.so \
//!       --target-gguf .../Qwen3.5-27B-Q4_K_M.gguf \
//!       --draft-st    .../draft/model.safetensors \
//!       --n-tokens 256 \
//!       --ddtree \
//!       --prompt-lens 1024,4096,16384
//!
//! Prints a per-call table with `decode_tok_s`, `n_accepted`, and wall time
//! so you can see the load amortization.

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

use ctox_dflash_ffi::{DflashOpts, DflashRuntime};

/// HumanEval sample prompt — base pattern for synthesized long prompts.
/// Same 9 tokens used by `dflash-ref-inproc-bench`.
const BASE_PROMPT_IDS: &[i32] = &[7734, 264, 6185, 36974, 883, 13094, 6326, 61369, 25];

#[derive(Parser, Debug)]
struct Args {
    /// Path to the built libdflash_run_lib.so.
    #[arg(long, default_value = "/home/metricspace/dflash-ref/dflash/build/libdflash_run_lib.so")]
    lib: PathBuf,

    #[arg(long, default_value = "/home/metricspace/dflash-ref/dflash/models/Qwen3.5-27B-Q4_K_M.gguf")]
    target_gguf: PathBuf,

    #[arg(long, default_value = "/home/metricspace/dflash-ref/dflash/models/draft/model.safetensors")]
    draft_st: PathBuf,

    /// Comma-separated prompt lengths to generate against. Each length is
    /// exercised once with its own synthesized prompt. The runtime is
    /// initialized ONCE — all generate calls reuse it.
    #[arg(long, default_value = "1024,4096,16384,32768,65536,100000")]
    prompt_lens: String,

    /// Tokens to generate per call.
    #[arg(long, default_value_t = 256)]
    n_tokens: usize,

    /// Enable DDTree tree verify (budget controlled by --ddtree-budget).
    #[arg(long)]
    ddtree: bool,

    #[arg(long, default_value_t = 22)]
    ddtree_budget: u32,

    /// Pre-allocate KV for the largest expected prompt + decode tail.
    /// If 0, computed as max(prompt_lens) + n_tokens + 4096 headroom.
    #[arg(long, default_value_t = 0)]
    max_ctx: u32,

    /// CUDA device ordinal.
    #[arg(long, default_value_t = 0)]
    cuda_device: u32,
}

fn synth_prompt(len: usize) -> Vec<i32> {
    (0..len)
        .map(|i| BASE_PROMPT_IDS[i % BASE_PROMPT_IDS.len()])
        .collect()
}

fn parse_prompt_lens(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|p| {
            p.parse::<usize>()
                .with_context(|| format!("prompt_lens entry {:?}", p))
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Basic subscriber so the crate's `tracing::info!` from init lands
    // somewhere visible.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .try_init();

    let prompt_lens = parse_prompt_lens(&args.prompt_lens)?;
    if prompt_lens.is_empty() {
        return Err(anyhow!("--prompt-lens is empty"));
    }
    let max_prompt = *prompt_lens.iter().max().unwrap();
    let max_ctx = if args.max_ctx > 0 {
        args.max_ctx
    } else {
        // Headroom for DDTree tree nodes + safety.
        (max_prompt + args.n_tokens + 4096) as u32
    };

    let opts = DflashOpts {
        target_gguf: args.target_gguf.clone(),
        draft_safetensors: args.draft_st.clone(),
        max_ctx,
        ddtree_mode: args.ddtree,
        ddtree_budget: args.ddtree_budget,
        ddtree_temp: 1.0,
        ddtree_chain_seed: true,
        fast_rollback: false,
        seq_verify: false,
        cuda_device: args.cuda_device,
        tbq_kv: false,
    };

    eprintln!(
        "loading runtime: lib={} target={} draft={} max_ctx={} ddtree={}",
        args.lib.display(),
        args.target_gguf.display(),
        args.draft_st.display(),
        max_ctx,
        args.ddtree,
    );
    let t_init0 = Instant::now();
    let mut rt = DflashRuntime::new(&args.lib, &opts)
        .with_context(|| format!("init runtime via {}", args.lib.display()))?;
    let init_wall = t_init0.elapsed().as_secs_f64();
    eprintln!("runtime ready after {:.2}s (model load)", init_wall);

    let mut rows: Vec<(usize, f64, i32, i32, i32, f64)> = Vec::new();
    for (i, &plen) in prompt_lens.iter().enumerate() {
        eprintln!(
            "\n--- call {} of {}: prompt_len={} n_tokens={} ---",
            i + 1,
            prompt_lens.len(),
            plen,
            args.n_tokens
        );
        let prompt = synth_prompt(plen);
        let t_call0 = Instant::now();
        let (out, stats) = rt
            .generate(&prompt, args.n_tokens)
            .with_context(|| format!("generate prompt_len={}", plen))?;
        let call_wall = t_call0.elapsed().as_secs_f64();
        eprintln!(
            "  → {} tokens out (prompt {} + {} new); decode_tok_s={:.2}, wall={:.2}s",
            out.len(),
            plen,
            stats.n_generated,
            stats.decode_tok_s,
            call_wall,
        );
        let tail: Vec<i32> = out.iter().rev().take(10).cloned().collect::<Vec<_>>();
        let tail_fwd: Vec<i32> = tail.into_iter().rev().collect();
        eprintln!("  tail: {:?}", tail_fwd);
        rows.push((
            plen,
            stats.decode_tok_s,
            stats.n_generated,
            stats.n_draft_steps,
            stats.n_accepted,
            call_wall,
        ));
    }

    // Summary table.
    println!("\n=== DFLASH PERSISTENT BENCH ===");
    println!(
        "lib={} target={} draft={}",
        args.lib.display(),
        args.target_gguf.display(),
        args.draft_st.display()
    );
    println!(
        "ddtree={} budget={} n_tokens={} max_ctx={} model_load_s={:.2}",
        args.ddtree, args.ddtree_budget, args.n_tokens, max_ctx, init_wall
    );
    println!(
        "\n{:>10}  {:>12}  {:>10}  {:>7}  {:>10}  {:>8}",
        "prompt_len", "decode_tok_s", "n_generated", "n_steps", "n_accepted", "wall_s"
    );
    println!("{}", "-".repeat(70));
    for (plen, tps, gen, steps, acc, wall) in &rows {
        println!(
            "{:>10}  {:>12.2}  {:>10}  {:>7}  {:>10}  {:>8.2}",
            plen, tps, gen, steps, acc, wall
        );
    }
    println!(
        "\nNote: model loaded ONCE in {:.2}s; per-call walls are pure generate cost.",
        init_wall
    );
    println!("     decode_tok_s comes from the library's [dflash] output line.");
    Ok(())
}
