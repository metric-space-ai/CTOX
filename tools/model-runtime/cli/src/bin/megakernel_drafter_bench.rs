//! Standalone smoke-bench for the Qwen3.5-0.8B megakernel drafter.
//!
//! Loads a Qwen3.5-0.8B BF16 safetensors checkpoint, runs prefill
//! over a small prompt, then decodes `N` tokens via `step()` and
//! prints per-token wall-time + aggregate tok/s. Primarily used to
//! confirm the FFI links + the kernel runs end-to-end on the
//! target GPU before wiring the drafter into a speculative
//! pipeline.
//!
//! Usage:
//!   megakernel-drafter-bench \
//!       --model /path/to/Qwen3.5-0.8B \
//!       --prompt "Hello world" \
//!       --n-tokens 128
//!
//! Build:
//!   cargo build --release -p ctox-engine-cli \
//!       --features cuda --bin megakernel-drafter-bench
//!
//! Expected reference performance on RTX 3090 sm_86:
//!   decode ~413 tok/s, prefill ~37800 tok/s (per megakernel
//!   RESULTS.md). A6000 (sm_86, +20% SMs) should be similar or
//!   slightly faster.

#![cfg(feature = "cuda")]

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use clap::Parser;
use engine_core::megakernel_drafter::{load_megakernel_weights, MegakernelDrafter};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Megakernel drafter smoke bench")]
struct Args {
    /// Path to the directory containing the BF16 safetensors shards.
    /// The loader scans for `*.safetensors` under this path.
    #[arg(long)]
    model: PathBuf,

    /// Prompt to feed through prefill. Encoded as a single
    /// hard-coded BOS token by default — proper tokenisation
    /// requires the target-side tokenizer which this bench skips.
    #[arg(long, default_value = "1")]
    prompt_ids: String,

    /// Number of decode tokens to generate via `step()`.
    #[arg(long, default_value_t = 128)]
    n_tokens: usize,

    /// Max prompt tokens (sizes prefill scratch). Default 128 is
    /// enough for the smoke path; bump for larger prompts.
    #[arg(long, default_value_t = 128)]
    max_prefill: usize,

    /// CUDA device ordinal.
    #[arg(long, default_value_t = 0)]
    device: usize,

    /// Simulate spec-decode rollback cost: after warming up N tokens
    /// of pure decode, measure drafter under the spec pattern
    /// `snapshot → step × K → restore → step × accepted` per round
    /// for `spec_rounds` rounds, then report the per-round wall time.
    /// `accepted` simulates a target acceptance length of K * accept_ratio.
    #[arg(long, default_value_t = 0)]
    spec_rounds: usize,

    /// K — draft tokens per spec round.
    #[arg(long, default_value_t = 6)]
    spec_k: usize,

    /// Acceptance ratio (fraction of K that the target would accept).
    /// 0.5 means half-reject; 1.0 means full accept.
    #[arg(long, default_value_t = 0.5)]
    spec_accept: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = Device::new_cuda(args.device)
        .with_context(|| format!("open CUDA device {}", args.device))?;

    // Scan safetensors shards under --model.
    let mut shards: Vec<PathBuf> = std::fs::read_dir(&args.model)
        .with_context(|| format!("read model dir {:?}", args.model))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() {
        return Err(anyhow!(
            "no *.safetensors files found under {:?}",
            args.model
        ));
    }
    eprintln!("loading {} safetensors shard(s) from {:?}", shards.len(), args.model);

    // Build a BF16 CUDA VarBuilder.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&shards, DType::BF16, &device)
            .context("open safetensors via VarBuilder")?
    };
    // Qwen3.5 multimodal safetensors nest everything under
    // `model.language_model.` — this is where the 24-layer hybrid
    // transformer lives. The `model.vision_tower.*` branch plus a
    // separate `mtp.*` multi-token-prediction head share the same
    // checkpoint but aren't used by the drafter.
    eprintln!("loading megakernel weights …");
    let t0 = Instant::now();
    let weights = load_megakernel_weights(vb.pp("model").pp("language_model"), device.clone())
        .context("load_megakernel_weights")?;
    eprintln!("  weights loaded in {:.3}s", t0.elapsed().as_secs_f64());

    let mut drafter = MegakernelDrafter::new(weights, args.max_prefill)
        .context("MegakernelDrafter::new")?;
    eprintln!("drafter buffers allocated");

    // Parse comma-separated prompt ids.
    let prompt_ids: Vec<i32> = args
        .prompt_ids
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<i32>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("parse --prompt-ids as comma-separated ints")?;
    if prompt_ids.is_empty() {
        return Err(anyhow!("--prompt-ids must be non-empty"));
    }

    // ── Prefill.
    eprintln!("prefill ({} tokens) …", prompt_ids.len());
    let t_prefill = Instant::now();
    let first = drafter
        .prefill(&prompt_ids)
        .context("MegakernelDrafter::prefill")?;
    let dt_prefill = t_prefill.elapsed();
    let prefill_tps = prompt_ids.len() as f64 / dt_prefill.as_secs_f64();
    eprintln!(
        "  prefill: first_tok={first}  {:.2} ms  {:.0} tok/s",
        dt_prefill.as_secs_f64() * 1000.0,
        prefill_tps
    );

    // ── Decode loop.
    eprintln!("decode {} tokens …", args.n_tokens);
    let mut tok = first;
    let t_decode = Instant::now();
    let mut generated: Vec<i32> = Vec::with_capacity(args.n_tokens);
    for _ in 0..args.n_tokens {
        tok = drafter.step(tok).context("MegakernelDrafter::step")?;
        generated.push(tok);
    }
    let dt_decode = t_decode.elapsed();
    let decode_tps = args.n_tokens as f64 / dt_decode.as_secs_f64();
    eprintln!(
        "  decode:  {:.2} ms  {:.0} tok/s  (avg {:.2} ms/token)",
        dt_decode.as_secs_f64() * 1000.0,
        decode_tps,
        dt_decode.as_secs_f64() * 1000.0 / args.n_tokens as f64
    );

    eprintln!("\ngenerated token ids:");
    for (i, tid) in generated.iter().enumerate() {
        eprint!("{tid}");
        if i + 1 < generated.len() {
            eprint!(",");
        }
    }
    eprintln!();

    // ── Optional: simulate the spec-decode drafter pattern to
    //    measure snapshot/restore + replay overhead isolated from
    //    the target forward.
    if args.spec_rounds > 0 {
        let k = args.spec_k.max(1);
        let accepted = ((k as f32) * args.spec_accept).round() as usize;
        let accepted = accepted.clamp(0, k);
        eprintln!(
            "\nspec-decode simulation: {} rounds of K={} drafter steps, accept={} (ratio {:.2})",
            args.spec_rounds, k, accepted, args.spec_accept
        );
        let mut spec_tok = *generated.last().unwrap_or(&1);
        let mut total_accepted: usize = 0;
        let t_spec = Instant::now();
        for _ in 0..args.spec_rounds {
            // snapshot
            let snap = drafter.snapshot_state().context("snapshot")?;
            // draft K tokens
            let mut draft_ids: Vec<i32> = Vec::with_capacity(k);
            let mut last = spec_tok;
            for _ in 0..k {
                last = drafter.step(last).context("spec step (draft)")?;
                draft_ids.push(last);
            }
            // restore to pre-draft, replay only accepted prefix
            drafter.restore_state(&snap).context("restore")?;
            let mut replay_last = spec_tok;
            for i in 0..accepted {
                replay_last = drafter.step(draft_ids[i]).context("spec step (replay)")?;
            }
            // advance to the "post-accepted" spec token for the next round
            spec_tok = if accepted > 0 {
                replay_last
            } else {
                spec_tok
            };
            total_accepted += accepted;
        }
        let dt_spec = t_spec.elapsed();
        let spec_ms_per_round = dt_spec.as_secs_f64() * 1000.0 / args.spec_rounds as f64;
        let effective_tps = total_accepted as f64 / dt_spec.as_secs_f64();
        eprintln!(
            "  spec loop: {:.2} ms / round  →  effective {:.0} accepted tok/s (drafter-only, no target cost)",
            spec_ms_per_round, effective_tps
        );
        eprintln!(
            "  (each round = {} drafter steps + {} replay steps + 1 snapshot + 1 restore; target verify cost not included)",
            k, accepted
        );
    }

    Ok(())
}
