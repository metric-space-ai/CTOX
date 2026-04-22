//! DFlash chain-verify speculative decoding for Qwen3.5-27B.
//!
//! # What this is
//!
//! A Rust port of the reference chain-verify loop in
//! `dflash-ref/dflash/test/test_dflash_lib.cpp::run_dflash_gen_loop`
//! (the non-ddtree branch). Binds a full [`crate::target::Qwen35Target`]
//! and a 5-layer [`crate::draft::DraftModel`] into a single sequence-
//! level generator that:
//!
//!   1. Prefills the target over the prompt, capturing the per-layer
//!      hidden states that the draft cross-attends over.
//!   2. Per step:
//!        a. Builds a block-diffusion noise embedding
//!           `[last_commit, MASK × (block_size - 1)]`.
//!        b. Runs the draft forward → `block_size` logits.
//!        c. Argmaxes → `block_size` draft candidate tokens. Pins
//!           `draft[0] = last_commit_tok` (the draft is free to
//!           "denoise" position 0 and may drift — the reference does
//!           the same pin).
//!        d. Runs a batched target verify over the candidates with
//!           capture on → `block_size` posterior logits.
//!        e. Longest-prefix greedy accept: `draft[i+1] ==
//!           argmax(verify[i])` for increasing `i`. First mismatch at
//!           `k` → accept `k+1` tokens, plus one bonus token
//!           (`target_argmax[k]`) = the target's correction. `commit_n
//!           = accept_n + 1` (legacy replay semantics; matches the
//!           reference's `!fast_rollback` path).
//!        f. Rewinds KV + GDN state by `block_size - commit_n`. The
//!           replay-style rewind works because the verify wrote K/V
//!           and advanced GDN for ALL block_size tokens; we only
//!           keep the first `commit_n`.
//!        g. Updates `last_commit_tok ← accepted tokens[-1]`.
//!
//! # Why separate from the bare-metal `Qwen35Target`
//!
//! The [`crate::engine::GenerativeModel`] trait represents sequence-
//! level generation; `Qwen35Target::forward` is step-level. Wrapping
//! them lets CTOX's serving layer swap a `SpeculativeDecoder` in for
//! the single-token decode path without touching the step-level
//! API, matching the same adapter pattern CTOX uses for the DFlash
//! FFI backend.
//!
//! # Perf caveats — day-one port
//!
//! * GDN state snapshot/restore is a full clone (memcpy-dtod of each
//!   of the 48 `[S_v, S_v, H_v, 1]` f32 tensors per step). Good for
//!   correctness, wasteful on wall-clock. The reference uses a
//!   persistent shadow buffer filled in-place by the GDN kernel
//!   (`cache.ssm_intermediate`) and avoids any snapshot when
//!   `fast_rollback` is on. See `TODO(perf)` in
//!   [`crate::target::Qwen35Target::snapshot_gdn_states`].
//! * Capture-buffer writes use one `memcpy_dtod` per (capture_layer,
//!   token) pair — 5 × n_tokens small launches per step. A single
//!   strided-copy kernel would collapse this to 5 launches.
//! * `target_feat_buf` has no ring wrap — sized linearly for the
//!   full context. The reference wraps at `target_feat_cap = 16384`
//!   to cap resident VRAM. Fine until we exercise > 64K context.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use anyhow::{anyhow, Result};
use half::{bf16, f16};

use ctox_cuda_primitives::device::DeviceContext;
use ctox_cuda_primitives::kv_cache::KvCache;
use ctox_cuda_primitives::tensor::CudaTensor;

use ctox_engine_runtime::model::{GenerateStats, GenerativeModel};

use crate::draft::DraftModel;
use crate::kernels::launch_embedding_bf16;
use crate::target::{Qwen35Target, CAPTURE_LAYERS};
use crate::tokenizer::Qwen35Tokenizer;

/// Qwen3.5-27B "MASK" token id used by the block-diffusion draft's
/// noise-embed construction. Matches
/// `DFLASH27B_DRAFT_MASK_TOKEN_ID` in
/// `dflash-ref/dflash/include/dflash27b.h`. Confirmed in that header
/// and in `dflash-ref/dflash/scripts/convert_dflash_to_gguf.py`
/// (`MASK_TOKEN_ID = 248070`). NOT the tokenizer's own `<mask>` slot —
/// this is a DFlash-specific slot the target's training set up as the
/// "unknown future token" seed for the draft.
pub const MASK_TOKEN_ID: i32 = 248070;

/// Per-step and aggregate stats for one `generate()` call. Mirrors
/// the reference's `RunStats` accounting — useful for tok/s reports
/// and for tuning the chain `block_size`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpecDecodeStats {
    /// Total speculative steps executed.
    pub n_steps: usize,
    /// Sum of `commit_n` across all steps (total committed tokens).
    pub n_accepted_total: usize,
    /// Sum of `block_size` across all steps (total proposed tokens).
    pub n_proposed_total: usize,
    /// `n_accepted_total as f64 / n_proposed_total as f64` — how
    /// aggressively the draft is landing. Unset (0.0) if no step has
    /// been run.
    pub acceptance_rate: f64,
}

impl SpecDecodeStats {
    fn record_step(&mut self, commit_n: usize, proposed: usize) {
        self.n_steps += 1;
        self.n_accepted_total += commit_n;
        self.n_proposed_total += proposed;
        self.acceptance_rate = if self.n_proposed_total == 0 {
            0.0
        } else {
            self.n_accepted_total as f64 / self.n_proposed_total as f64
        };
    }
}

/// Maximum slice of target context the draft cross-attends over. The
/// reference uses 2048 (`DRAFT_CTX_MAX` in `test_dflash_lib.cpp`); we
/// inherit it unchanged. Older target context stays in the target's
/// KV cache but is invisible to the draft.
const DRAFT_CTX_MAX: usize = 2048;

/// Wraps [`Qwen35Target`] + [`DraftModel`] + [`Qwen35Tokenizer`] in a
/// single [`GenerativeModel`]-shaped handle. See the module docstring
/// for the algorithm.
pub struct SpeculativeDecoder {
    target: Qwen35Target,
    draft: DraftModel,
    tokenizer: Arc<Qwen35Tokenizer>,
    device: Arc<DeviceContext>,
    /// Persisted across generate() calls so callers can inspect the
    /// last run's acceptance rate without re-running.
    last_stats: SpecDecodeStats,
    /// Upper-bound KV-cache / capture-buffer size the spec decoder
    /// is willing to allocate. Used to size the per-call KV cache
    /// and target_feat_buf.
    max_ctx: usize,
}

impl SpeculativeDecoder {
    /// Construct a new speculative decoder. Allocates the target's
    /// feature-capture buffer up front (sized for `max_ctx`). Caller
    /// retains ownership of the models via move; `tokenizer` is
    /// shared via `Arc`.
    pub fn new(
        mut target: Qwen35Target,
        draft: DraftModel,
        tokenizer: Arc<Qwen35Tokenizer>,
        max_ctx: usize,
    ) -> Result<Self> {
        if target.config.hidden_dim != draft.config().hidden {
            return Err(anyhow!(
                "SpeculativeDecoder: target.hidden_dim={} != draft.hidden={}",
                target.config.hidden_dim,
                draft.config().hidden
            ));
        }
        if draft.config().block_size == 0 {
            return Err(anyhow!(
                "SpeculativeDecoder: draft.block_size must be > 0"
            ));
        }
        let device = target.device.clone();
        target.ensure_capture_buf(max_ctx)?;
        Ok(Self {
            target,
            draft,
            tokenizer,
            device,
            last_stats: SpecDecodeStats::default(),
            max_ctx,
        })
    }

    /// Immutable access to the target — e.g. for configuration
    /// inspection from outside the engine crate.
    pub fn target(&self) -> &Qwen35Target {
        &self.target
    }

    /// Stats for the most recent `generate()` call.
    pub fn last_stats(&self) -> SpecDecodeStats {
        self.last_stats
    }

    /// Allocate a fresh KV cache matching this target's FA-layer count
    /// and head dims, sized for `self.max_ctx` tokens.
    fn alloc_kv_cache(&self) -> Result<KvCache> {
        KvCache::new(
            self.device.clone(),
            self.target.n_full_attn,
            self.max_ctx,
            self.target.config.n_kv_heads,
            self.target.config.head_dim,
        )
    }

    /// Allocate the vector of per-GDN recurrent states and the per-
    /// GDN intermediate buffer sized for `max_verify_tokens` (the
    /// largest single-forward token count we'll feed the target in
    /// this generate call — prefill ubatch OR block_size, whichever
    /// is larger).
    fn alloc_gdn_state(
        &self,
        max_verify_tokens: usize,
    ) -> Result<(Vec<CudaTensor<f32>>, Vec<CudaTensor<f16>>)> {
        let cfg = self.target.config;
        let s_v = cfg.gdn_ssm_dim;
        let h_v = cfg.gdn_num_v_heads;
        let mut states: Vec<CudaTensor<f32>> = Vec::with_capacity(self.target.n_gdn);
        let mut inter: Vec<CudaTensor<f16>> = Vec::with_capacity(self.target.n_gdn);
        for _ in 0..self.target.n_gdn {
            states.push(CudaTensor::<f32>::zeros(
                self.device.clone(),
                vec![s_v, s_v, h_v, 1],
            )?);
            inter.push(CudaTensor::<f16>::zeros(
                self.device.clone(),
                vec![s_v, s_v, h_v, max_verify_tokens],
            )?);
        }
        Ok((states, inter))
    }

    /// Host-side argmax over a `[rows, vocab]` f32 contiguous slab.
    /// Returns `rows` ids.
    fn argmax_rows(logits: &[f32], rows: usize, vocab: usize) -> Vec<i32> {
        let mut out = Vec::with_capacity(rows);
        for r in 0..rows {
            let row = &logits[r * vocab..(r + 1) * vocab];
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = i;
                }
            }
            out.push(best_idx as i32);
        }
        out
    }

    /// Build 4-axis MRoPE positions `[4, n_tokens]` where the first
    /// three axes hold the absolute text position `start + i` and the
    /// fourth axis is zero (no vision/audio modality). Matches
    /// [`Qwen35Target::forward`]'s expected layout.
    fn build_positions_4axis(
        device: &Arc<DeviceContext>,
        start: usize,
        n_tokens: usize,
    ) -> Result<CudaTensor<i32>> {
        let mut host = vec![0i32; 4 * n_tokens];
        for i in 0..n_tokens {
            let p = (start + i) as i32;
            host[i] = p;
            host[n_tokens + i] = p;
            host[2 * n_tokens + i] = p;
            // axis 3: zeros.
        }
        CudaTensor::<i32>::from_host(device.clone(), vec![4, n_tokens], &host)
    }

    /// Build the noise embedding for one chain-verify step.
    ///
    /// Produces `[block_size, hidden]` bf16 where row 0 is the target
    /// embedding of `last_commit_tok` and rows 1..block_size are the
    /// target embedding of [`MASK_TOKEN_ID`]. Uses the target's own
    /// `token_embd.weight` via `launch_embedding_bf16` — no separate
    /// draft embedding table, matching the reference.
    fn build_noise_embed(
        &self,
        last_commit_tok: i32,
        block_size: usize,
    ) -> Result<CudaTensor<bf16>> {
        let mut ids_host = vec![MASK_TOKEN_ID; block_size];
        ids_host[0] = last_commit_tok;
        let ids = CudaTensor::<i32>::from_host(
            self.device.clone(),
            vec![block_size],
            &ids_host,
        )?;
        let hidden_dim = self.target.config.hidden_dim;
        let mut out =
            CudaTensor::<bf16>::zeros(self.device.clone(), vec![block_size, hidden_dim])?;
        launch_embedding_bf16(&self.device, &self.target.embed, &ids, &mut out)?;
        Ok(out)
    }

    /// Build the draft's `target_hidden_cat` by slicing the target's
    /// capture buffer on the range `[draft_start, committed)` and
    /// reshaping — no wrap (the capture buffer is sized for the full
    /// context). Returns a fresh bf16 tensor of shape
    /// `[draft_ctx, 5 * hidden]`.
    fn build_target_hidden_cat(
        &self,
        draft_start: usize,
        draft_ctx: usize,
    ) -> Result<CudaTensor<bf16>> {
        let hidden_dim = self.target.config.hidden_dim;
        let feat_dim = CAPTURE_LAYERS.len() * hidden_dim;
        let stream = self.device.raw().default_stream();
        let Some(feat_buf) = self.target.capture_buf() else {
            return Err(anyhow!(
                "SpeculativeDecoder: target capture buffer not allocated"
            ));
        };
        // The capture buffer is row-major `[max_ctx, 5*hidden]`, so the
        // slice we want — rows `[draft_start..draft_start+draft_ctx)` —
        // is already contiguous. One memcpy into a fresh tensor.
        let mut dst = CudaTensor::<bf16>::zeros(
            self.device.clone(),
            vec![draft_ctx, feat_dim],
        )?;
        let src_start = draft_start * feat_dim;
        let src_end = src_start + draft_ctx * feat_dim;
        let src_view = feat_buf.buf().slice(src_start..src_end);
        let dst_view = dst.buf_mut();
        stream.memcpy_dtod(&src_view, dst_view).map_err(|e| {
            anyhow!(
                "SpeculativeDecoder::build_target_hidden_cat: memcpy_dtod: {:?}",
                e
            )
        })?;
        Ok(dst)
    }

    /// Internal state-carrying generate loop. Returns the full token
    /// sequence (prompt + continuation) plus stats.
    fn generate_inner(
        &mut self,
        prompt_ids: &[i32],
        n_new: usize,
    ) -> Result<(Vec<i32>, SpecDecodeStats, i32)> {
        // ── 0. Shape + config sanity ────────────────────────────────
        if prompt_ids.is_empty() {
            return Err(anyhow!(
                "SpeculativeDecoder.generate: prompt_ids must be non-empty"
            ));
        }
        let block_size = self.draft.config().block_size;
        // The target's single-forward path needs gdn_inter sized for
        // the largest chunk we'll run. Prefill uses
        // [`Qwen35Target::prefill_ubatch_for`], verify uses block_size.
        let prefill_ubatch = Qwen35Target::prefill_ubatch_for(prompt_ids.len());
        let max_verify_tokens = prefill_ubatch.max(block_size);
        if prompt_ids.len() + n_new > self.max_ctx {
            return Err(anyhow!(
                "SpeculativeDecoder.generate: prompt_len={} + n_new={} > max_ctx={}",
                prompt_ids.len(),
                n_new,
                self.max_ctx
            ));
        }

        // ── 1. State: KV cache + GDN state ─────────────────────────
        let mut kv_cache = self.alloc_kv_cache()?;
        let (mut gdn_states, mut gdn_inter) = self.alloc_gdn_state(max_verify_tokens)?;

        // ── 2. Prefill — chunked through forward_with_capture so the
        //      target_feat_buf is populated for every prompt position.
        //      We can't use Qwen35Target::prefill directly because it
        //      calls the non-capturing forward. Port the chunking
        //      inline.
        let prompt_host: Vec<i32> = prompt_ids.to_vec();
        let mut last_logits: Option<CudaTensor<f32>> = None;
        let mut pf_start: usize = 0;
        while pf_start < prompt_host.len() {
            let chunk_n = std::cmp::min(prefill_ubatch, prompt_host.len() - pf_start);
            let tk = CudaTensor::<i32>::from_host(
                self.device.clone(),
                vec![chunk_n],
                &prompt_host[pf_start..pf_start + chunk_n],
            )?;
            let pos = Self::build_positions_4axis(&self.device, pf_start, chunk_n)?;
            let logits = self.target.forward_with_capture(
                &tk,
                &pos,
                &mut kv_cache,
                &mut gdn_states,
                &mut gdn_inter,
                pf_start,
            )?;
            last_logits = Some(logits);
            pf_start += chunk_n;
        }
        let prefill_logits = last_logits
            .ok_or_else(|| anyhow!("SpeculativeDecoder.generate: prefill produced no logits"))?;
        // Download just the last row and argmax — the prefill chunk's
        // final-position logits are our seed `last_commit_tok`.
        let vocab = self.target.vocab_size;
        let pf_shape = prefill_logits.shape().to_vec();
        if pf_shape.len() != 2 || pf_shape[1] != vocab {
            return Err(anyhow!(
                "SpeculativeDecoder.generate: prefill logits shape {:?} unexpected",
                pf_shape
            ));
        }
        let pf_host = prefill_logits.to_host()?;
        let last_row_start = (pf_shape[0] - 1) * vocab;
        let last_row = &pf_host[last_row_start..last_row_start + vocab];
        let seed_ids = Self::argmax_rows(last_row, 1, vocab);
        let mut last_commit_tok: i32 = seed_ids[0];

        tracing::info!(
            prompt_len = prompt_host.len(),
            seed_tok = last_commit_tok,
            block_size,
            n_new,
            "spec_decode: prefill done"
        );

        // ── 3. Chain-verify loop ───────────────────────────────────
        //
        // `out_all` holds the full emitted token stream, prompt-
        // prefixed. `n_generated` counts only the tokens beyond the
        // prompt (matches reference semantics). `committed` = KV-cache
        // n_filled = position where the next verify batch writes.
        let mut out_all: Vec<i32> = prompt_host.clone();
        let mut stats = SpecDecodeStats::default();
        let mut n_generated: usize = 0;

        while n_generated < n_new {
            let need_commit_budget = n_new - n_generated;
            let committed = kv_cache.n_filled();

            // ── 3a. Build noise_embed = [last_commit_tok, MASK × 15]
            let noise_embed = self.build_noise_embed(last_commit_tok, block_size)?;

            // ── 3b. Slice target_feat over the draft-window, build
            //       positions_q / positions_k, run draft forward.
            let draft_ctx = std::cmp::min(committed, DRAFT_CTX_MAX);
            let draft_start = committed - draft_ctx;
            if draft_ctx == 0 {
                return Err(anyhow!(
                    "SpeculativeDecoder.generate: draft_ctx=0 — committed={} at step {}",
                    committed,
                    stats.n_steps
                ));
            }
            let target_hidden_cat =
                self.build_target_hidden_cat(draft_start, draft_ctx)?;

            let pos_q_host: Vec<i32> =
                (0..block_size).map(|i| (draft_ctx + i) as i32).collect();
            let positions_q =
                CudaTensor::<i32>::from_host(self.device.clone(), vec![block_size], &pos_q_host)?;
            let pos_k_host: Vec<i32> =
                (0..(draft_ctx + block_size)).map(|i| i as i32).collect();
            let positions_k = CudaTensor::<i32>::from_host(
                self.device.clone(),
                vec![draft_ctx + block_size],
                &pos_k_host,
            )?;

            let draft_logits = self.draft.forward(
                &noise_embed,
                &target_hidden_cat,
                &positions_q,
                &positions_k,
                &self.target.lm_head,
            )?;

            // ── 3c. Argmax draft logits, pin slot 0.
            let draft_logits_host = draft_logits.to_host()?;
            let mut draft_tok = Self::argmax_rows(&draft_logits_host, block_size, vocab);
            draft_tok[0] = last_commit_tok;

            // ── 3d. Snapshot GDN state — correctness-first clone.
            let gdn_snapshot = self.target.snapshot_gdn_states(&gdn_states)?;

            // ── 3e. Batched target verify at positions
            //       [committed..committed + block_size].
            let verify_tokens = CudaTensor::<i32>::from_host(
                self.device.clone(),
                vec![block_size],
                &draft_tok,
            )?;
            let verify_positions =
                Self::build_positions_4axis(&self.device, committed, block_size)?;
            let verify_logits = self.target.forward_with_capture(
                &verify_tokens,
                &verify_positions,
                &mut kv_cache,
                &mut gdn_states,
                &mut gdn_inter,
                committed,
            )?;
            // kv_cache has now advanced by block_size; target_feat_buf
            // has new rows at [committed..committed + block_size].

            let verify_logits_host = verify_logits.to_host()?;
            let target_tok =
                Self::argmax_rows(&verify_logits_host, block_size, vocab);

            // ── 3f. Chain-accept: find longest prefix such that
            //       draft_tok[i+1] == target_tok[i]. draft_tok[0] is
            //       the pinned last_commit_tok so it's accepted
            //       unconditionally (accept_n starts at 1).
            let mut accept_n = 1usize;
            for i in 0..(block_size - 1) {
                if draft_tok[i + 1] == target_tok[i] {
                    accept_n += 1;
                } else {
                    break;
                }
            }

            // Legacy replay path (matches reference's `!fast_rollback`):
            //   - accept_n < block_size → bonus = target_tok[accept_n-1]
            //     (target's correction for the mismatched draft token),
            //     commit_n = accept_n + 1.
            //   - accept_n == block_size → full acceptance, no bonus
            //     (no mismatch to correct), commit_n = block_size.
            let (bonus_tok, pre_trim_commit_n) = if accept_n < block_size {
                (Some(target_tok[accept_n - 1]), accept_n + 1)
            } else {
                (None, accept_n)
            };
            let mut commit_n = pre_trim_commit_n;
            if commit_n > need_commit_budget {
                commit_n = need_commit_budget;
            }
            if commit_n == 0 {
                // n_new budget exhausted mid-step.
                break;
            }

            tracing::info!(
                step = stats.n_steps,
                committed,
                accept_n,
                commit_n,
                bonus_tok = bonus_tok.unwrap_or(-1),
                "spec_decode: step accepted"
            );

            // ── 3g. Rewind KV + restore GDN.
            //
            // Verify advanced KV and GDN by block_size. We commit
            // commit_n tokens' worth of state; roll back the rest.
            // The KV rewind just moves n_filled; the slab bytes at
            // positions [committed + commit_n, committed + block_size)
            // are now "unfilled" and will be overwritten by the next
            // iteration's verify. GDN restore: clone the pre-verify
            // snapshot back, then re-apply the accepted prefix.
            //
            // Correctness-first shortcut: the reference's legacy path
            // restores the snapshot AND replays accepted tokens
            // through the target to re-advance GDN by commit_n. We
            // take the same approach but without the replay — instead
            // we roll back GDN fully and re-run a target forward on
            // just the accepted tokens via the next iteration's
            // verify prefix. That is exactly what happens naturally
            // because the next iteration's verify starts at
            // `committed + commit_n`, reads GDN state at the
            // pre-verify snapshot, and advances by block_size
            // including the "already accepted" tail — producing the
            // correct state.
            //
            // But that double-counts tokens in the KV cache. So:
            //   - KV: rewind by (block_size - commit_n). The first
            //     `commit_n` slots stay valid (KV was written by the
            //     verify in position order, so slots
            //     [committed..committed+commit_n) hold the correct
            //     K/V for the accepted tokens). The remaining
            //     `block_size - commit_n` slots are "uncommitted" and
            //     will be rewritten next iter.
            //   - GDN: restore the pre-verify snapshot, then replay
            //     the accepted tokens with a single forward so GDN
            //     state advances by exactly `commit_n`. Without the
            //     replay, GDN state would be too far ahead.
            //
            // Replay uses forward_with_capture on the accepted token
            // slice so the target_feat_buf rows at
            // [committed..committed+commit_n) are refreshed with the
            // "clean" accepted-only state — matches the reference's
            // legacy replay semantics.
            // Build the replay_tok array the reference uses:
            //   replay_tok[0..accept_n)          = draft_tok[0..accept_n)
            //   replay_tok[accept_n..commit_n)   = bonus_tok (if any)
            //
            // Slot 0 is last_commit_tok (pinned on draft_tok[0]). The
            // full array is `commit_n` tokens long. This is what gets
            // emitted to out_all AND, on partial-accept, what gets
            // replayed through the target.
            let mut commit_ids: Vec<i32> = Vec::with_capacity(commit_n);
            for i in 0..commit_n {
                let id = if i < accept_n {
                    draft_tok[i]
                } else {
                    bonus_tok.ok_or_else(|| {
                        anyhow!(
                            "spec_decode: commit_ids needs bonus_tok but it's None \
                             (accept_n={} commit_n={})",
                            accept_n,
                            commit_n
                        )
                    })?
                };
                commit_ids.push(id);
            }

            // ── 3g. Rewind KV + restore GDN on partial accept, then
            //       run the replay forward so both caches advance by
            //       exactly `commit_n`. On full accept the verify's
            //       state is already correct.
            //
            // We always want the "next last_tok" to be the argmax of
            // the logits at position `committed + commit_n - 1`. That
            // comes from either the verify's final row (full accept)
            // or the replay's final row (partial accept — the verify
            // row at `accept_n - 1` / `commit_n - 1` predicted the
            // bonus already, but we want the prediction AT the bonus,
            // i.e. the replay's last row).
            let next_last_tok: i32;
            if commit_n >= block_size {
                // Fully accepted the block — KV/GDN are already at
                // `committed + block_size`. Use the verify's final-row
                // argmax as the next last_commit_tok.
                next_last_tok = target_tok[commit_n - 1];
            } else {
                // Partial accept: rewind fully, restore GDN, replay the
                // committed prefix with capture so target_feat is
                // refreshed over the clean accepted-only state.
                kv_cache.rewind(block_size);
                self.target.restore_gdn_states(&mut gdn_states, &gdn_snapshot)?;

                let replay_tokens = CudaTensor::<i32>::from_host(
                    self.device.clone(),
                    vec![commit_n],
                    &commit_ids,
                )?;
                let replay_positions =
                    Self::build_positions_4axis(&self.device, committed, commit_n)?;
                let replay_logits = self.target.forward_with_capture(
                    &replay_tokens,
                    &replay_positions,
                    &mut kv_cache,
                    &mut gdn_states,
                    &mut gdn_inter,
                    committed,
                )?;
                // Argmax of the final replay row → next last_commit_tok.
                let replay_shape = replay_logits.shape().to_vec();
                if replay_shape.len() != 2 || replay_shape[1] != vocab {
                    return Err(anyhow!(
                        "spec_decode: replay logits shape {:?} unexpected",
                        replay_shape
                    ));
                }
                let replay_host = replay_logits.to_host()?;
                let last_row_start = (commit_n - 1) * vocab;
                let last_row = &replay_host[last_row_start..last_row_start + vocab];
                next_last_tok = Self::argmax_rows(last_row, 1, vocab)[0];
            }
            drop(gdn_snapshot);

            // ── 3h. Emit exactly `commit_n` tokens — the pending
            //       last_commit_tok at slot 0 plus each subsequent
            //       accepted/bonus token. Respects the n_new budget.
            for &tok in &commit_ids {
                out_all.push(tok);
                n_generated += 1;
                if n_generated >= n_new {
                    break;
                }
            }

            last_commit_tok = next_last_tok;
            stats.record_step(commit_n, block_size);

            if n_generated >= n_new {
                break;
            }
        }

        tracing::info!(
            n_generated,
            n_steps = stats.n_steps,
            n_accepted_total = stats.n_accepted_total,
            n_proposed_total = stats.n_proposed_total,
            acceptance_rate = stats.acceptance_rate,
            "spec_decode: generate done"
        );

        Ok((out_all, stats, last_commit_tok))
    }
}

impl GenerativeModel for SpeculativeDecoder {
    fn id(&self) -> &str {
        "qwen35-27b-specdecode"
    }

    fn vocab_size(&self) -> usize {
        self.target.vocab_size
    }

    fn encode(&self, text: &str) -> Result<Vec<i32>> {
        self.tokenizer.encode(text)
    }

    fn decode(&self, ids: &[i32]) -> Result<String> {
        self.tokenizer.decode(ids)
    }

    fn generate(
        &mut self,
        prompt_ids: &[i32],
        n_new: usize,
    ) -> Result<(Vec<i32>, GenerateStats)> {
        let t0 = std::time::Instant::now();
        let (tokens, spec_stats, last_tok) = self.generate_inner(prompt_ids, n_new)?;
        let wall_s = t0.elapsed().as_secs_f64();
        self.last_stats = spec_stats;

        let n_generated = tokens.len().saturating_sub(prompt_ids.len());
        let decode_tok_s = if wall_s > 0.0 {
            n_generated as f64 / wall_s
        } else {
            0.0
        };
        let stats = GenerateStats {
            n_generated,
            n_draft_steps: spec_stats.n_steps,
            n_accepted: spec_stats.n_accepted_total,
            n_proposed: spec_stats.n_proposed_total,
            wall_s,
            decode_tok_s,
            last_tok,
        };
        Ok((tokens, stats))
    }
}
