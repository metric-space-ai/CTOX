//! Diagnostic forward for the Qwen3.5-27B target.
//!
//! Loads the 27B target, runs a SINGLE forward (no chunking,
//! no speculative decoding) on a 128-token HumanEval-pattern prompt,
//! prints per-layer L2/absmax of the hidden state's last row, and then
//! prints the top-10 final-position logits with their token ids.
//!
//! Used to localize which layer first diverges from the reference.
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
    about = "Single-chunk diagnostic forward with per-layer L2 dump."
)]
struct Args {
    #[arg(long, default_value = "/home/metricspace/dflash-ref/dflash/models/Qwen3.5-27B-Q4_K_M.gguf")]
    target_gguf: PathBuf,
    /// Number of prompt tokens to feed in a single forward call.
    #[arg(long, default_value_t = 128)]
    prompt_len: usize,
    /// Capacity of the KV cache and feature buffers.
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

    // HumanEval pattern repeated — same prompt as qwen35-spec-bench.
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
        gdn_inter.push(CudaTensor::<f16>::zeros(
            device.clone(),
            vec![s_v, s_v, h_v, args.prompt_len],
        )?);
        gdn_conv_states.push(CudaTensor::<f32>::zeros(
            device.clone(),
            vec![conv_state_rows, qkv_proj_dim],
        )?);
    }

    // Token tensor + 4-axis MRoPE positions for a SINGLE forward.
    let tokens = CudaTensor::<i32>::from_host(device.clone(), vec![prompt.len()], &prompt)
        .with_context(|| "upload tokens")?;
    let mut pos_host = vec![0i32; 4 * prompt.len()];
    for i in 0..prompt.len() {
        let p = i as i32;
        pos_host[i] = p;
        pos_host[prompt.len() + i] = p;
        pos_host[2 * prompt.len() + i] = p;
    }
    let positions =
        CudaTensor::<i32>::from_host(device.clone(), vec![4, prompt.len()], &pos_host)
            .with_context(|| "upload positions")?;

    eprintln!(
        "[diag] running SINGLE forward (n_tokens={}, dump_l2={}) ...",
        prompt.len(),
        args.dump_l2
    );
    let t0 = Instant::now();
    let logits = if args.dump_l2 {
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
    eprintln!("[diag] forward done in {:.2}s", t0.elapsed().as_secs_f64());

    // Extract the LAST-position row of the logits.
    let vocab = target.vocab_size;
    let shape = logits.shape().to_vec();
    assert_eq!(shape, vec![prompt.len(), vocab]);
    let host = logits.to_host().with_context(|| "download logits")?;
    let last_row_start = (prompt.len() - 1) * vocab;
    let last_row = &host[last_row_start..last_row_start + vocab];

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

    // Report rank of expected token 6185.
    let want = 6185usize;
    let rank = idx.iter().position(|&i| i == want).unwrap_or(vocab);
    eprintln!("DIAG rank_of_6185 = {} logit_6185 = {:.4}", rank, last_row[want]);

    // Silence the unused-import warning for `bf16` — the target API
    // threads bf16 internally; we re-export the type here for clarity.
    let _: Option<bf16> = None;
    Ok(())
}
