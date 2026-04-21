//! Thin wrapper around the reference `test_dflash` binary. Establishes
//! the ceiling — the exact tok/s + AL the engine would achieve if our
//! target-forward path matched the reference's ggml hand-tuned kernels.
//!
//! This is a pragmatic intermediate: skips the C++/Rust FFI + ggml-
//! backend shared-library linkage complexity and just spawns the
//! already-built `test_dflash` as a subprocess, reading its stdout for
//! tokens + stderr for stats.
//!
//! Build:
//!   cargo build --release -p ctox-engine-cli --bin dflash-ref-bench
//!
//! Run:
//!   target/release/dflash-ref-bench \
//!       --test-dflash /home/metricspace/dflash-ref/dflash/build/test_dflash \
//!       --ggml-libs  /home/metricspace/dflash-ref/dflash/build/deps/llama.cpp/ggml/src \
//!       --target-gguf /home/metricspace/dflash-ref/dflash/models/Qwen3.5-27B-Q4_K_M.gguf \
//!       --draft-st    /home/metricspace/dflash-ref/dflash/models/draft/model.safetensors \
//!       --prompt-ids 7734,264,6185,36974,883,13094,6326,61369,25 \
//!       --n-tokens 128 \
//!       [--ddtree [--ddtree-budget 22]]

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::process::{Command, Stdio};

#[derive(Parser, Debug)]
struct Args {
    /// Path to the reference `test_dflash` binary.
    #[arg(long, default_value = "/home/metricspace/dflash-ref/dflash/build/test_dflash")]
    test_dflash: PathBuf,

    /// Directory containing libggml*.so that `test_dflash` needs via
    /// LD_LIBRARY_PATH (two dirs joined via colon internally).
    #[arg(
        long,
        default_value = "/home/metricspace/dflash-ref/dflash/build/deps/llama.cpp/ggml/src:/home/metricspace/dflash-ref/dflash/build/deps/llama.cpp/ggml/src/ggml-cuda"
    )]
    ggml_libs: String,

    /// Path to the Q4_K_M GGUF target.
    #[arg(long)]
    target_gguf: PathBuf,

    /// Path to the DFlash draft safetensors.
    #[arg(long)]
    draft_st: PathBuf,

    /// Prompt token ids (comma-separated).
    #[arg(long, default_value = "7734,264,6185,36974,883,13094,6326,61369,25")]
    prompt_ids: String,

    /// Number of new tokens to generate.
    #[arg(long, default_value_t = 128)]
    n_tokens: usize,

    /// Enable DDTree tree verify (reference's --ddtree flag).
    #[arg(long)]
    ddtree: bool,

    /// DDTree budget (reference's --ddtree-budget=N). Only used with --ddtree.
    #[arg(long, default_value_t = 22)]
    ddtree_budget: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let prompt_ids: Vec<i32> = args
        .prompt_ids
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<i32>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("parse --prompt-ids")?;

    // Write prompt.bin the way test_dflash expects (little-endian i32).
    let prompt_path = std::env::temp_dir().join("dflash_ref_prompt.bin");
    let out_path = std::env::temp_dir().join("dflash_ref_out.bin");
    {
        let mut bytes: Vec<u8> = Vec::with_capacity(prompt_ids.len() * 4);
        for id in &prompt_ids {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        std::fs::write(&prompt_path, &bytes).with_context(|| format!("write {:?}", prompt_path))?;
    }

    let mut cmd = Command::new(&args.test_dflash);
    cmd.env("LD_LIBRARY_PATH", &args.ggml_libs)
        .arg(&args.target_gguf)
        .arg(&args.draft_st)
        .arg(&prompt_path)
        .arg(format!("{}", args.n_tokens))
        .arg(&out_path);
    if args.ddtree {
        cmd.arg("--ddtree");
        cmd.arg(format!("--ddtree-budget={}", args.ddtree_budget));
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    eprintln!(
        "spawning: {} {} {} {} {} {}{}",
        args.test_dflash.display(),
        args.target_gguf.display(),
        args.draft_st.display(),
        prompt_path.display(),
        args.n_tokens,
        out_path.display(),
        if args.ddtree {
            format!(" --ddtree --ddtree-budget={}", args.ddtree_budget)
        } else {
            String::new()
        }
    );

    let t_start = std::time::Instant::now();
    let output = cmd.output().context("spawn test_dflash")?;
    let elapsed = t_start.elapsed();

    if !output.status.success() {
        return Err(anyhow!(
            "test_dflash exit={:?}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    // Parse stdout for the stats line(s) test_dflash prints.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Key lines we care about (see test_dflash.cpp tail):
    //   [dflash] generated N tokens in T s  →  X tok/s
    //   [dflash] M draft steps, accepted=A/B (P% per step), avg commit/step=C
    //   [timing] per-step averages ...
    let mut tok_per_s: Option<f64> = None;
    let mut tokens: Option<i32> = None;
    let mut draft_steps: Option<i32> = None;
    let mut accepted: Option<(i32, i32)> = None;
    let mut avg_commit: Option<f32> = None;
    for line in stdout.lines() {
        let l = line.trim();
        if l.starts_with("[dflash] generated ") {
            // [dflash] generated 128 tokens in 2.608 s  →  49.08 tok/s
            if let Some(idx) = l.find("→") {
                let tail = &l[idx + "→".len()..].trim();
                if let Some(val) = tail.split_whitespace().next() {
                    tok_per_s = val.parse::<f64>().ok();
                }
            }
            if let Some(rest) = l.strip_prefix("[dflash] generated ") {
                if let Some(val) = rest.split_whitespace().next() {
                    tokens = val.parse::<i32>().ok();
                }
            }
        } else if l.starts_with("[dflash] ") && l.contains("draft steps") {
            // [dflash] 22 draft steps, accepted=107/352 (30.4% per step), avg commit/step=5.82
            let parts: Vec<&str> = l.split_whitespace().collect();
            if let Some(d) = parts.get(1) {
                draft_steps = d.parse::<i32>().ok();
            }
            for p in &parts {
                if let Some(rest) = p.strip_prefix("accepted=") {
                    let rest = rest.trim_end_matches(',');
                    if let Some((a, b)) = rest.split_once('/') {
                        if let (Ok(ai), Ok(bi)) = (a.parse::<i32>(), b.parse::<i32>()) {
                            accepted = Some((ai, bi));
                        }
                    }
                }
                if let Some(rest) = p.strip_prefix("commit/step=") {
                    avg_commit = rest.parse::<f32>().ok();
                }
            }
        }
    }

    println!("\n=== REFERENCE test_dflash RESULT ===");
    println!(
        "wall time incl. model load: {:.2}s (model load dominates!)",
        elapsed.as_secs_f64()
    );
    if let (Some(t), Some(n)) = (tok_per_s, tokens) {
        println!("decode tok/s: {:.2}  ({} new tokens)", t, n);
    }
    if let Some(ds) = draft_steps {
        println!("draft steps: {}", ds);
    }
    if let Some((a, b)) = accepted {
        let al = a as f64 / draft_steps.unwrap_or(1).max(1) as f64;
        println!("accepted: {}/{}  → AL per step: {:.2}", a, b, al);
    }
    if let Some(ac) = avg_commit {
        println!("avg commit/step: {:.2}", ac);
    }
    // Show last few stderr lines too — they contain the timing breakdown.
    let stderr_lines: Vec<&str> = stderr.lines().collect();
    if !stderr_lines.is_empty() {
        println!("\n--- last 10 stderr lines ---");
        for line in stderr_lines.iter().rev().take(10).rev() {
            println!("{line}");
        }
    }

    // Try to read the output token ids file for sanity.
    match std::fs::read(&out_path) {
        Ok(bytes) => {
            let n = bytes.len() / 4;
            let mut first: Vec<i32> = Vec::new();
            for i in 0..n.min(16) {
                let mut b = [0u8; 4];
                b.copy_from_slice(&bytes[i * 4..i * 4 + 4]);
                first.push(i32::from_le_bytes(b));
            }
            println!("\nout_ids.bin: {} tokens, first 16: {:?}", n, first);
        }
        Err(e) => {
            println!("\ncould not read out_ids.bin: {e}");
        }
    }

    Ok(())
}
