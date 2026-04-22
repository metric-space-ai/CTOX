//! DFlash FFI backend — CTOX's intermediate production path for
//! Qwen3.5-27B.
//!
//! Wraps [`ctox_dflash_ffi::DflashRuntime`] (itself a `libloading`
//! handle on the vendored `libdflash_run_lib.so`) together with a
//! HuggingFace `tokenizers::Tokenizer` so callers get a
//! [`GenerativeModel`] they can drive with plain text.
//!
//! ## Why this backend exists
//!
//! The bare-metal native port (`ctox-qwen35-27b`) is still catching
//! up on end-to-end tok/s (see `AGENTS.md` for the audit). Until it
//! reaches reference parity the CTOX serving stack routes Qwen3.5-27B
//! requests through this FFI wrapper, which delegates to the exact
//! same CUDA kernels the reference uses (`ggml-cuda` + DFlash
//! speculative decoding) and hits 77–118 tok/s on an A6000.
//!
//! ## Statefulness
//!
//! The C API (`dflash_ctx_generate`) resets the underlying KV /
//! SSM state on every call, so each [`generate`] starts cold — there
//! is no prefix reuse across calls. That's the contract the reference
//! harness always had; prefix sharing needs a new C API layer before
//! this wrapper can expose it.
//!
//! ## Thread-safety
//!
//! `DflashBackend` is neither `Send` nor `Sync` because
//! [`DflashRuntime`] isn't — the underlying `ggml-cuda` state is
//! process-global and the reference library is not internally
//! synchronized. One backend instance per process, driven from one
//! thread. Callers that need per-request concurrency need to funnel
//! through a single-threaded actor on top of this type.
//!
//! [`generate`]: GenerativeModel::generate

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use ctox_dflash_ffi::{DflashOpts, DflashRuntime};
use tokenizers::Tokenizer;

use crate::model::{GenerateStats, GenerativeModel};

/// Configuration for loading a [`DflashBackend`].
///
/// Paths are validated at load time — an invalid GGUF, draft, or
/// tokenizer file surfaces as an error from [`DflashBackend::load`]
/// rather than panicking later inside the `.so`.
#[derive(Debug, Clone)]
pub struct DflashBackendConfig {
    /// Path to `libdflash_run_lib.so`. Usually emitted by the
    /// reference CMake build into `tools/model-runtime/dflash-ffi/build/`.
    pub lib_path: PathBuf,
    /// Path to the Qwen3.5-27B GGUF target weights.
    pub target_gguf: PathBuf,
    /// Path to the DFlash draft safetensors (5-layer block-diffusion
    /// draft model; shipped alongside the reference).
    pub draft_safetensors: PathBuf,
    /// Path to a HuggingFace `tokenizer.json` matching the target.
    pub tokenizer_json: PathBuf,
    /// Forwarded to the reference — see `ctox_dflash_ffi::DflashOpts`.
    /// Default is reasonable for a 4K-context chat workload.
    pub runtime: DflashOpts,
    /// Human-readable id for this backend. Mostly used in logs and
    /// serving-layer dispatch (`"qwen35-27b-dflash"` is a good
    /// starting value).
    pub id: String,
}

impl DflashBackendConfig {
    /// Convenience constructor for the canonical Qwen3.5-27B layout.
    ///
    /// Assumes the reference `.so`, target GGUF, draft safetensors,
    /// and tokenizer all live under a single root directory with the
    /// expected filenames. Any layout deviation should use the
    /// struct-literal form instead.
    pub fn qwen35_27b(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref();
        Self {
            lib_path: root.join("libdflash_run_lib.so"),
            target_gguf: root.join("qwen35-27b.gguf"),
            draft_safetensors: root.join("draft.safetensors"),
            tokenizer_json: root.join("tokenizer.json"),
            runtime: DflashOpts {
                max_ctx: 4096,
                ddtree_mode: true,
                ..DflashOpts::default()
            },
            id: "qwen35-27b-dflash".to_string(),
        }
    }
}

/// Concrete `GenerativeModel` backed by the DFlash FFI.
///
/// Construct with [`DflashBackend::load`]. Drop frees the FFI context
/// via `dflash_ctx_free` before unmapping the `.so`.
pub struct DflashBackend {
    id: String,
    tokenizer: Tokenizer,
    runtime: DflashRuntime,
    vocab_size: usize,
}

// SAFETY: DflashRuntime holds a `*mut DflashCtxRaw` that makes it
// !Send by default. We assert `Send` here — but **not** `Sync` —
// because the serving layer needs to hand the backend to a dedicated
// worker task (Tokio blocking thread) after construction. The
// invariant the caller must uphold is the same one documented on
// DflashRuntime: one backend instance per process, driven from one
// thread at a time. Moving it to another thread between calls is
// fine; using it from multiple threads concurrently is undefined
// behavior in the reference library regardless of what Rust's type
// system would let us do.
unsafe impl Send for DflashBackend {}

impl DflashBackend {
    /// Load the `.so`, initialize the DFlash context, and mmap the
    /// tokenizer. Any failure surfaces as `anyhow::Error` with context
    /// identifying which path / step failed.
    ///
    /// The reference library uses process-global CUDA state; calling
    /// this twice in one process is supported by the Rust type system
    /// but **not** by the C++ side. Callers that want a clean second
    /// run should drop the first backend first.
    pub fn load(cfg: DflashBackendConfig) -> Result<Self> {
        // Tokenizer first — a cheap failure mode, and it validates
        // the JSON before we spin up CUDA.
        let tokenizer = Tokenizer::from_file(&cfg.tokenizer_json).map_err(|e| {
            anyhow!(
                "failed to load tokenizer from {}: {e}",
                cfg.tokenizer_json.display()
            )
        })?;
        let vocab_size = tokenizer.get_vocab_size(true);

        // Validate the GGUF + draft + so paths exist on disk so the
        // dlopen failure (if any) is about symbol resolution, not
        // ENOENT masquerading as a cryptic libloading error.
        for (label, path) in [
            ("lib_path", &cfg.lib_path),
            ("target_gguf", &cfg.target_gguf),
            ("draft_safetensors", &cfg.draft_safetensors),
        ] {
            if !path.exists() {
                return Err(anyhow!(
                    "DflashBackendConfig.{label} does not exist on disk: {}",
                    path.display()
                ));
            }
        }

        // Runtime opts need the paths as owned PathBuf; copy from the
        // config into a fresh DflashOpts so we don't mutate the caller's
        // template values.
        let opts = DflashOpts {
            target_gguf: cfg.target_gguf.clone(),
            draft_safetensors: cfg.draft_safetensors.clone(),
            ..cfg.runtime.clone()
        };

        let runtime = DflashRuntime::new(&cfg.lib_path, &opts)
            .with_context(|| format!("initializing dflash runtime from {}", cfg.lib_path.display()))?;

        tracing::info!(
            id = %cfg.id,
            lib = %cfg.lib_path.display(),
            target = %cfg.target_gguf.display(),
            draft = %cfg.draft_safetensors.display(),
            tokenizer = %cfg.tokenizer_json.display(),
            vocab_size = vocab_size,
            "dflash backend loaded"
        );

        Ok(Self {
            id: cfg.id,
            tokenizer,
            runtime,
            vocab_size,
        })
    }
}

impl GenerativeModel for DflashBackend {
    fn id(&self) -> &str {
        &self.id
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn encode(&self, text: &str) -> Result<Vec<i32>> {
        // Special tokens enabled to match the reference seed — dflash
        // was fed prompts via HF encode(text, true), so anything less
        // would skew chat-template alignment.
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("tokenizer encode failed: {e}"))?;
        // Vocab ids are u32; the FFI boundary uses i32. Qwen3.5
        // vocabs (~151k / ~248k) fit comfortably in i32 so this cast
        // is lossless.
        Ok(encoding.get_ids().iter().map(|&id| id as i32).collect())
    }

    fn decode(&self, ids: &[i32]) -> Result<String> {
        let u32_ids: Vec<u32> = ids
            .iter()
            .map(|&id| {
                if id < 0 {
                    Err(anyhow!("negative token id {id} passed to decode"))
                } else {
                    Ok(id as u32)
                }
            })
            .collect::<Result<_>>()
            .context("converting i32 ids to u32 for tokenizer decode")?;
        self.tokenizer
            .decode(&u32_ids, false)
            .map_err(|e| anyhow!("tokenizer decode failed: {e}"))
    }

    fn generate(
        &mut self,
        prompt_ids: &[i32],
        n_new: usize,
    ) -> Result<(Vec<i32>, GenerateStats)> {
        let (tokens, raw) = self
            .runtime
            .generate(prompt_ids, n_new)
            .map_err(|e| anyhow!("dflash generate failed: {e}"))?;
        let stats = GenerateStats {
            n_generated: raw.n_generated.max(0) as usize,
            n_draft_steps: raw.n_draft_steps.max(0) as usize,
            n_accepted: raw.n_accepted.max(0) as usize,
            n_proposed: raw.n_proposed.max(0) as usize,
            wall_s: raw.wall_s,
            decode_tok_s: raw.decode_tok_s,
            last_tok: raw.last_tok,
        };
        Ok((tokens, stats))
    }
}

#[cfg(test)]
mod tests {
    //! Integration tests for `DflashBackend` require the compiled
    //! reference library + Qwen3.5-27B weights on disk. The tests
    //! below only run when `CTOX_DFLASH_LIB`, `CTOX_QWEN35_27B_GGUF`,
    //! `CTOX_QWEN35_DRAFT_SAFETENSORS`, and `CTOX_QWEN_TOKENIZER_JSON`
    //! are all set; otherwise they skip cleanly so cargo test on a
    //! plain laptop still passes. CI runners that have the fixtures
    //! mounted pick them up automatically.
    //!
    //! Run with:
    //!
    //! ```bash
    //! CTOX_DFLASH_LIB=/path/to/libdflash_run_lib.so \
    //! CTOX_QWEN35_27B_GGUF=/path/to/qwen35-27b.gguf \
    //! CTOX_QWEN35_DRAFT_SAFETENSORS=/path/to/draft.safetensors \
    //! CTOX_QWEN_TOKENIZER_JSON=/path/to/tokenizer.json \
    //! cargo test -p ctox-engine-runtime --features dflash-backend -- \
    //!     --ignored --nocapture dflash_smoke
    //! ```

    use super::*;

    fn cfg_from_env() -> Option<DflashBackendConfig> {
        let lib = std::env::var_os("CTOX_DFLASH_LIB")?;
        let gguf = std::env::var_os("CTOX_QWEN35_27B_GGUF")?;
        let draft = std::env::var_os("CTOX_QWEN35_DRAFT_SAFETENSORS")?;
        let tok = std::env::var_os("CTOX_QWEN_TOKENIZER_JSON")?;
        Some(DflashBackendConfig {
            lib_path: PathBuf::from(lib),
            target_gguf: PathBuf::from(gguf),
            draft_safetensors: PathBuf::from(draft),
            tokenizer_json: PathBuf::from(tok),
            runtime: DflashOpts {
                max_ctx: 4096,
                ddtree_mode: false,
                ..DflashOpts::default()
            },
            id: "qwen35-27b-dflash-test".to_string(),
        })
    }

    #[test]
    #[ignore]
    fn dflash_smoke() {
        let Some(cfg) = cfg_from_env() else {
            eprintln!("skipping: set CTOX_DFLASH_LIB / CTOX_QWEN35_27B_GGUF / CTOX_QWEN35_DRAFT_SAFETENSORS / CTOX_QWEN_TOKENIZER_JSON to run this test");
            return;
        };

        let mut backend = DflashBackend::load(cfg).expect("load dflash backend");
        let prompt_ids = backend
            .encode("def fibonacci(n: int) -> int:\n    ")
            .expect("tokenize prompt");
        let (tokens, stats) = backend
            .generate(&prompt_ids, 32)
            .expect("generate 32 tokens");

        eprintln!(
            "dflash: generated {} tokens in {:.2}s → {:.1} tok/s (last_tok={})",
            stats.n_generated, stats.wall_s, stats.decode_tok_s, stats.last_tok
        );
        assert!(
            tokens.len() >= prompt_ids.len(),
            "output ({}) should at least contain the prompt ({})",
            tokens.len(),
            prompt_ids.len()
        );
        let text = backend.decode(&tokens).expect("decode output");
        eprintln!("--- decoded ---\n{text}\n---");
    }
}
