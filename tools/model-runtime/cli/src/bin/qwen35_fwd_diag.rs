//! Diagnostic forward for the Qwen3.5-27B target.
//!
//! Loads the 27B target and runs a 128-token PREFILL in single-token
//! DECODE mode (n_tokens=1 per step, KV cache grows), matching the
//! reference `dflash-ref/test/test_generate.cpp` execution shape. At the
//! final step (pos=prompt_len-1) the `forward_diag` variant is used,
//! which dumps per-layer L2/absmax of the hidden-state last row to
//! stderr with the `DIAG L2[NN]` tag. After that, top-10 logits and
//! argmax are printed.
//!
//! Expected next token after the reference HumanEval prompt is 6185.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use ctox_cuda_primitives::device::DeviceContext;
use ctox_cuda_primitives::kv_cache::KvCache;
use ctox_cuda_primitives::tensor::CudaTensor;
use ctox_qwen35_27b::gguf_loader::{parse_qwen35_metadata, LoaderConfig};
use ctox_qwen35_27b::{Qwen35Config, Qwen35Target};
use half::{bf16, f16};

#[derive(Parser, Debug)]
#[command(
    name = "qwen35-fwd-diag",
    about = "Per-token decode diagnostic with per-layer L2 dump at the final step."
)]
struct Args {
    #[arg(long, default_value = "/home/metricspace/dflash-ref/dflash/models/Qwen3.5-27B-Q4_K_M.gguf")]
    target_gguf: PathBuf,
    /// Number of prompt tokens to feed (single-token decode per token).
    #[arg(long, default_value_t = 128)]
    prompt_len: usize,
    /// Capacity of the KV cache.
    #[arg(long, default_value_t = 512)]
    max_ctx: usize,
    #[arg(long, default_value_t = 0)]
    cuda_device: u32,
    /// Print per-layer L2 (default on). Pass `--no-dump-l2` to suppress.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    dump_l2: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")))
        .with(tracing_subscriber::fmt::layer())
        .init();
    let args = Args::parse();

    let device = Arc::new(
        DeviceContext::new(args.cuda_device as usize)
            .with_context(|| format!("DeviceContext::new({})", args.cuda_device))?,
    );

    eprintln!("[diag] parse metadata...");
    let meta = parse_qwen35_metadata(&args.target_gguf).with_context(|| "parse_qwen35_metadata")?;
    let config = Qwen35Config::from_metadata(&meta, 128);

    eprintln!("[diag] load target 27B (keep_packed=true)...");
    let t0 = Instant::now();
    let target = Qwen35Target::load_from_gguf_with_config(
        device.clone(),
        config.clone(),
        &args.target_gguf,
        LoaderConfig { keep_packed: true },
    )
    .with_context(|| "load target")?;
    eprintln!(
        "[diag] target loaded in {:.2}s (vocab={} n_fa={} n_gdn={})",
        t0.elapsed().as_secs_f64(),
        target.vocab_size,
        target.n_full_attn,
        target.n_gdn,
    );

    // HumanEval pattern repeated — same prompt as reference test_generate.
    let base: [i32; 9] = [7734, 264, 6185, 36974, 883, 13094, 6326, 61369, 25];
    let prompt: Vec<i32> = (0..args.prompt_len).map(|i| base[i % 9]).collect();
    eprintln!(
        "[diag] prompt_len={} first9={:?}",
        prompt.len(),
        &prompt[..prompt.len().min(9)]
    );

    // Fresh KV cache + fresh GDN states.
    let mut kv_cache = KvCache::new(
        device.clone(),
        target.n_full_attn,
        args.max_ctx,
        target.config.n_kv_heads,
        target.config.head_dim,
    )
    .with_context(|| "alloc KvCache")?;

    let cfg = target.config;
    let s_v = cfg.gdn_ssm_dim;
    let h_v = cfg.gdn_num_v_heads;
    let qkv_proj_dim = cfg.gdn_qkv_proj_dim();
    let conv_state_rows = 3usize;

    let mut gdn_states: Vec<CudaTensor<f32>> = Vec::with_capacity(target.n_gdn);
    let mut gdn_inter: Vec<CudaTensor<f16>> = Vec::with_capacity(target.n_gdn);
    let mut gdn_conv_states: Vec<CudaTensor<f32>> = Vec::with_capacity(target.n_gdn);
    for _ in 0..target.n_gdn {
        gdn_states.push(CudaTensor::<f32>::zeros(
            device.clone(),
            vec![s_v, s_v, h_v, 1],
        )?);
        // decode mode: n_tokens=1, so gdn_inter holds 1 step's worth.
        gdn_inter.push(CudaTensor::<f16>::zeros(
            device.clone(),
            vec![s_v, s_v, h_v, 1],
        )?);
        gdn_conv_states.push(CudaTensor::<f32>::zeros(
            device.clone(),
            vec![conv_state_rows, qkv_proj_dim],
        )?);
    }

    eprintln!(
        "[diag] running PER-TOKEN decode over {} positions (dump_l2 at final step = {}) ...",
        prompt.len(),
        args.dump_l2
    );
    let t0 = Instant::now();
    let mut last_logits: Option<CudaTensor<f32>> = None;
    for i in 0..prompt.len() {
        // Single token.
        let tok_vec = vec![prompt[i]];
        let tokens = CudaTensor::<i32>::from_host(device.clone(), vec![1], &tok_vec)
            .with_context(|| "upload tokens")?;
        // MRoPE 4D positions: first 3 axes = absolute position, axis 3 = 0.
        let p = i as i32;
        let pos_host = vec![p, p, p, 0i32];
        let positions =
            CudaTensor::<i32>::from_host(device.clone(), vec![4, 1], &pos_host)
                .with_context(|| "upload positions")?;

        let is_last = i + 1 == prompt.len();
        let logits = if is_last && args.dump_l2 {
            target.forward_diag(
                &tokens,
                &positions,
                &mut kv_cache,
                &mut gdn_states,
                &mut gdn_inter,
                &mut gdn_conv_states,
            )?
        } else {
            target.forward(
                &tokens,
                &positions,
                &mut kv_cache,
                &mut gdn_states,
                &mut gdn_inter,
                &mut gdn_conv_states,
            )?
        };
        if is_last {
            last_logits = Some(logits);
        }
    }
    eprintln!("[diag] decode-loop done in {:.2}s", t0.elapsed().as_secs_f64());

    let logits = last_logits.expect("last_logits set on final iter");

    // Extract the LAST-position row of the logits ([1, vocab]).
    let vocab = target.vocab_size;
    let shape = logits.shape().to_vec();
    assert_eq!(shape, vec![1, vocab]);
    let host = logits.to_host().with_context(|| "download logits")?;
    let last_row = &host[..vocab];

    // Argmax + top-10.
    let mut idx: Vec<usize> = (0..vocab).collect();
    idx.sort_by(|&a, &b| last_row[b].partial_cmp(&last_row[a]).unwrap());
    let top10 = &idx[..10];
    let argmax = top10[0] as i32;
    eprintln!("DIAG final_argmax = {}", argmax);
    eprintln!("DIAG expected     = 6185");
    eprintln!("DIAG top10 (id:logit):");
    for &i in top10 {
        eprintln!("  {:>6} : {:.4}", i, last_row[i]);
    }

    let want = 6185usize;
    let rank = idx.iter().position(|&i| i == want).unwrap_or(vocab);
    eprintln!("DIAG rank_of_6185 = {} logit_6185 = {:.4}", rank, last_row[want]);

    let _: Option<bf16> = None;
    Ok(())
}
