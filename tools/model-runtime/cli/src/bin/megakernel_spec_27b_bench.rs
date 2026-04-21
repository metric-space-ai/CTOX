//! End-to-end speculative-decode bench: Qwen3.5-0.8B megakernel
//! drafter + Qwen3.5-27B Q4_K_M target via the existing engine
//! vision loader.
//!
//! Pipeline:
//!   1. Target — `VisionLoaderBuilder` constructs a
//!      `Box<dyn Loader>` for `Qwen/Qwen3.5-27B` (uses the HF cache
//!      at `HF_HOME`; we relocated it to the SanDisk earlier in
//!      this session). `load_model_from_hf` with
//!      `in_situ_quant=Some(IsqType::Q4K)` applies GGUF-equivalent
//!      Q4_K_M ISQ on load — ~15GB target weight footprint on
//!      A6000's 48GB, leaves plenty for drafter + KV @ 128K.
//!   2. Drafter — loaded standalone from the 0.8B safetensors via
//!      `load_megakernel_weights` → `MegakernelDrafter::new`.
//!   3. Generation — `run_greedy_megakernel` orchestrates target
//!      prefill + drafter prefill + spec-round loop.
//!
//! Reports: generated tokens, decode tok/s, acceptance length.
//!
//! Build:
//!   cargo build --release -p ctox-engine-cli --features cuda \
//!       --bin megakernel-spec-27b-bench
//!
//! Run:
//!   target/release/megakernel-spec-27b-bench \
//!       --target-model-id Qwen/Qwen3.5-27B \
//!       --drafter-path /mnt/hfcache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/<hash> \
//!       --prompt-ids 1,2,3,4,5 --n-tokens 256 --spec-k 6

#![cfg(feature = "cuda")]

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use engine_core::{
    megakernel_drafter::{load_megakernel_weights, MegakernelDrafter, MegakernelSpecOpts},
    DeviceMapSetting, TokenSource,
};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "End-to-end megakernel drafter + Qwen3.5-27B Q4_K_M spec-decode bench")]
struct Args {
    /// HF model id for the target (uses HF_HOME cache for shard resolution).
    #[arg(long, default_value = "Qwen/Qwen3.5-27B")]
    target_model_id: String,

    /// Local snapshot dir for the Qwen3.5-0.8B drafter safetensors.
    #[arg(long)]
    drafter_path: PathBuf,

    /// Prompt token ids (comma-separated). Uses int ids directly so
    /// we don't pull in a tokenizer for this bench.
    #[arg(long, default_value = "1,2,3,4,5")]
    prompt_ids: String,

    /// How many new tokens to generate (including the first token
    /// from target prefill).
    #[arg(long, default_value_t = 128)]
    n_tokens: usize,

    /// Drafter proposals per spec round.
    #[arg(long, default_value_t = 6)]
    spec_k: usize,

    /// Max prompt tokens (sizes drafter prefill scratch).
    #[arg(long, default_value_t = 512)]
    max_prefill: usize,

    /// CUDA device ordinal.
    #[arg(long, default_value_t = 0)]
    device: usize,

    /// Whether to apply Q4_K_M ISQ on the target. Default on;
    /// disable for BF16 measurements (won't fit 27B on 48GB A6000
    /// anyway — will OOM).
    #[arg(long, default_value_t = true)]
    q4k: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialise tracing so the engine's info/warn messages surface.
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .compact()
        .init();

    let args = Args::parse();
    let device = Device::new_cuda(args.device)
        .with_context(|| format!("open CUDA device {}", args.device))?;
    eprintln!("opened CUDA:{}", args.device);

    // ── Target: Qwen3.5 vision loader via builder, then
    //    load_model_from_hf with Q4K ISQ.
    eprintln!(
        "building target loader for {} (ISQ Q4_K_M = {})",
        args.target_model_id, args.q4k
    );
    let target_loader = engine_core::VisionLoaderBuilder::new(
        engine_core::VisionSpecificConfig::default(),
        None,
        None,
        Some(args.target_model_id.clone()),
        None,
    )
    .build(Some(engine_core::VisionLoaderType::Qwen3_5));

    let isq = if args.q4k {
        Some(engine_core::IsqType::Q4K)
    } else {
        None
    };
    let t_target_load = Instant::now();
    let pipeline = target_loader
        .load_model_from_hf(
            None,
            TokenSource::None,
            &candle_core::DType::BF16,
            &device,
            false,
            DeviceMapSetting::dummy(),
            isq,
            None, // paged_attn_config — we use the vision model's eager SDPA path
        )
        .map_err(|e| anyhow!("load target: {e}"))?;
    eprintln!(
        "target loaded in {:.2}s",
        t_target_load.elapsed().as_secs_f64()
    );

    // ── Drafter: Qwen3.5-0.8B megakernel.
    eprintln!("scanning drafter safetensors under {:?}", args.drafter_path);
    let mut shards: Vec<PathBuf> = std::fs::read_dir(&args.drafter_path)
        .with_context(|| format!("read drafter dir {:?}", args.drafter_path))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|s| s.to_str()) == Some("safetensors")
        })
        .collect();
    shards.sort();
    if shards.is_empty() {
        return Err(anyhow!(
            "no *.safetensors found under {:?}",
            args.drafter_path
        ));
    }
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&shards, DType::BF16, &device)
            .context("mmap drafter safetensors")?
    };
    eprintln!("loading megakernel weights …");
    let t_draft_load = Instant::now();
    let weights = load_megakernel_weights(vb.pp("model").pp("language_model"), device.clone())
        .context("load_megakernel_weights")?;
    let mut drafter = MegakernelDrafter::new(weights, args.max_prefill)
        .context("MegakernelDrafter::new")?;
    eprintln!(
        "drafter loaded in {:.2}s",
        t_draft_load.elapsed().as_secs_f64()
    );

    // ── Parse prompt ids + build tensor.
    let prompt_ids: Vec<u32> = args
        .prompt_ids
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u32>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("parse --prompt-ids")?;
    if prompt_ids.is_empty() {
        return Err(anyhow!("--prompt-ids must be non-empty"));
    }
    let prompt_len = prompt_ids.len();
    let prompt_tensor = Tensor::from_vec(prompt_ids, (1, prompt_len), &device)?;

    // ── Acquire target text-model reference + run.
    let pipeline_guard = pipeline.lock().await;
    let text_model = pipeline_guard
        .dflash_text_model()
        .ok_or_else(|| anyhow!("target is not a Qwen3.5 vision pipeline"))?;

    let opts = MegakernelSpecOpts {
        k: args.spec_k,
        target_capture_layers: Vec::new(),
    };
    // Try to read EOS from the target's metadata. Empty vec = no
    // early stop.
    let eos_ids: Vec<u32> = pipeline_guard.get_metadata().eos_tok.clone();

    eprintln!(
        "prompt_len={} spec_k={} n_tokens={} eos_ids={:?}",
        prompt_len, args.spec_k, args.n_tokens, eos_ids
    );
    let t_gen = Instant::now();
    let outcome = engine_core::run_greedy_megakernel(
        text_model,
        &mut drafter,
        &prompt_tensor,
        args.n_tokens,
        &eos_ids,
        &opts,
    )
    .context("run_greedy_megakernel")?;
    let elapsed = t_gen.elapsed();

    let n_new = outcome.generated_tokens.len().saturating_sub(1);
    let al = if outcome.draft_steps > 0 {
        outcome.draft_accepted_total as f64 / outcome.draft_steps as f64
    } else {
        0.0
    };
    eprintln!("\n=== RESULT ===");
    eprintln!("total wall: {:.2}s", elapsed.as_secs_f64());
    eprintln!("new tokens: {n_new}");
    eprintln!("decode tok/s (engine-reported): {:.1}", outcome.decode_tok_per_s);
    eprintln!(
        "decode tok/s (wall):            {:.1}",
        n_new as f64 / elapsed.as_secs_f64()
    );
    eprintln!("spec rounds: {}", outcome.draft_steps);
    eprintln!(
        "acceptance length (AL): {:.2}  ({} accepted / {} rounds)",
        al, outcome.draft_accepted_total, outcome.draft_steps
    );
    eprintln!("\ngenerated ids:");
    for (i, tid) in outcome.generated_tokens.iter().enumerate() {
        eprint!("{tid}");
        if i + 1 < outcome.generated_tokens.len() {
            eprint!(",");
        }
    }
    eprintln!();

    Ok(())
}
