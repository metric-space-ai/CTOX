//! End-to-end DFlash speculative step with chain verify.
//!
//! Stitches the four previous pieces together:
//!
//!   - [`DFlashDraftRunner`] (commit 4) — produces `block_size`
//!     candidate tokens from the ring's current feature window.
//!   - [`DFlashTargetForward`] (this commit) — abstracts the target
//!     forward + capture.
//!   - [`FeatureCapture`] (commit 2) — carries the captured per-layer
//!     hidden states out of the target.
//!   - [`TargetFeatureRing`] (commit 3) — stores those captures for
//!     the NEXT step to condition on.
//!
//! Verify is **chain-style** in this commit: accept the longest
//! prefix of candidates for which `argmax(target_logits[i]) ==
//! candidates[i]`, plus one extra "target's own" token at the
//! boundary (which can differ from the draft's rejected candidate).
//! DDTree tree-verify lands in a later commit; chain is the natural
//! baseline and matches what `test_dflash --seq-verify` does in the
//! reference, giving us an apples-to-apples measurement before we
//! add the tree complexity.

use std::sync::Mutex;

use candle_core::{DType, IndexOp, Result, Tensor, D};

use super::capture::FeatureCapture;
use super::config::DFlashDraftConfig;
use super::model::DFlashDraftModel;
use super::ring::TargetFeatureRing;
use super::runner::{DFlashDraftRunner, DraftStepOpts};
use super::target::DFlashTargetForward;

/// Per-step options, tuning knobs for the verify loop.
#[derive(Debug, Clone)]
pub struct StepperOpts {
    /// How many committed-token rows to use as draft cross-attention
    /// context. Clamped down to `ring.len()` per call.
    pub ctx_len: usize,

    /// Top-K for the draft output. Unused by chain verify; reserved
    /// for DDTree. A safe default keeps the runner happy.
    pub draft_top_k: usize,
}

impl Default for StepperOpts {
    fn default() -> Self {
        Self {
            ctx_len: 64,
            draft_top_k: 8,
        }
    }
}

/// Per-step wall-time breakdown in milliseconds. Separates kernel
/// cost of the verify forward from the commit-replay forward so
/// telemetry can show how much of a step is the rollback overhead.
#[derive(Debug, Clone, Copy, Default)]
pub struct StepTimings {
    /// Verify forward (B+1 tokens) + batched argmax.
    pub verify_ms: f64,
    /// Commit replay forward (draft_accepted+1 tokens). Zero on the
    /// full-accept fast path where the verify cache state is reused.
    pub commit_ms: f64,
}

/// One step's result — how many tokens were committed, which ids,
/// and whether the target produced its own token at the boundary.
#[derive(Debug, Clone)]
pub struct StepOutcome {
    /// Tokens to append to the output sequence. Length is at least 1
    /// (the target always commits its boundary sample) and at most
    /// `block_size`.
    pub accepted: Vec<u32>,

    /// How many of the draft's candidates were accepted (before the
    /// target's boundary sample). Equals `accepted.len() - 1` when
    /// the boundary came from the target disagreeing, equals
    /// `accepted.len()` when the draft ran out of candidates to
    /// propose (i.e. full block accepted).
    pub draft_accepted: usize,
}

/// Chain-verify stepper. Owns the ring (mutably, behind a mutex so
/// the pipeline loop can share the stepper across concurrent inflight
/// requests once commit 6 wires it up); everything else lives by
/// borrow.
pub struct DFlashChainStepper {
    ring: Mutex<TargetFeatureRing>,
    cfg: DFlashDraftConfig,
}

impl DFlashChainStepper {
    pub fn new(ring: TargetFeatureRing, cfg: DFlashDraftConfig) -> Self {
        Self {
            ring: Mutex::new(ring),
            cfg,
        }
    }

    pub fn ring(&self) -> &Mutex<TargetFeatureRing> {
        &self.ring
    }

    pub fn config(&self) -> &DFlashDraftConfig {
        &self.cfg
    }

    /// Seed the ring with the features captured during prefill.
    ///
    /// The pipeline calls this once after the initial target prefill,
    /// with the captured features for each prompt token packed into a
    /// single `[ctx_len, target_layer_ids.len() * hidden]` tensor (the
    /// caller is responsible for the feature-dim concatenation — see
    /// [`fuse_captured_features`]).
    pub fn seed_ring(&self, features: &Tensor) -> Result<()> {
        let mut ring = self.ring.lock().unwrap();
        ring.append(features)
    }

    /// Run one chain-verify step. Returns the committed token(s).
    ///
    /// `last_committed_token` is the id the target produced at the
    /// end of the previous step (or the last prompt token for the
    /// very first step). `past_kv_len` is the current KV length of
    /// the target (position of the first fed token in this step).
    pub fn step<T: DFlashTargetForward>(
        &self,
        target: &T,
        draft: &DFlashDraftModel,
        last_committed_token: u32,
        past_kv_len: usize,
        opts: &StepperOpts,
    ) -> Result<StepOutcome> {
        self.step_with_diag(target, draft, last_committed_token, past_kv_len, opts)
            .map(|(outcome, _)| outcome)
    }

    /// Same as [`Self::step`] but also returns the `(verify_ms, commit_ms)`
    /// timings. The pipeline's telemetry log uses this to separate kernel
    /// cost from accept-loop cost without breaking the public step API.
    pub fn step_with_diag<T: DFlashTargetForward>(
        &self,
        target: &T,
        draft: &DFlashDraftModel,
        last_committed_token: u32,
        past_kv_len: usize,
        opts: &StepperOpts,
    ) -> Result<(StepOutcome, StepTimings)> {
        // ── 1. Ask the draft for candidates. `apply_lm_head` is a
        //    closure over the target's `apply_lm_head` method so the
        //    runner doesn't need to know whether the projection is a
        //    plain Linear, a QuantMethod, or some future tiled variant.
        let draft_out = {
            let ring = self.ring.lock().unwrap();
            let runner = DFlashDraftRunner::new(draft, target.embed_tokens());
            // Chain verify reads only `candidates`; forcing top_k=1
            // here hits the batched-argmax fast path in the runner
            // (one D→H of `block_size` u32 indices vs 16 per-row
            // full-vocab log-prob transfers). DDTree stepper will
            // need opts.draft_top_k honoured; that's a separate path.
            let ropts = DraftStepOpts {
                top_k: 1,
                ctx_len: opts.ctx_len,
            };
            let _ = opts.draft_top_k; // preserved for the DDTree stepper
            runner.step(last_committed_token, &ring, &ropts, |h| target.apply_lm_head(h))?
        };

        // ── 2. Build verify input: [last_tok, cand_0, cand_1, ..., cand_{B-1}]
        //
        //    B+1 tokens in, B+1 logit rows out. target_choices[i] is
        //    the target's greedy pick for the token right after
        //    feed[i]:
        //      target_choices[0]      ←→  candidates[0]
        //      ...
        //      target_choices[B-1]    ←→  candidates[B-1]
        //      target_choices[B]      = "free bonus" if all B candidates matched.
        let block = self.cfg.block_size;
        let candidates = &draft_out.candidates;
        let mut feed_verify = Vec::with_capacity(block + 1);
        feed_verify.push(last_committed_token);
        for c in candidates.iter() {
            feed_verify.push(*c);
        }
        let feed_len = feed_verify.len();
        let device = target.embed_tokens().embeddings().device();
        let verify_ids = Tensor::from_vec(feed_verify.clone(), (1, feed_len), device)?;

        // ── 3. Snapshot recurrent state + run verify forward.
        //    The rollback below needs the pre-verify Gated-DeltaNet
        //    state to recover after discarding the verify-forward's
        //    advance across B+1 feed positions (only draft_accepted+1
        //    of which end up committed). Attention cache is rewound
        //    via `truncate_attention_to`; recurrent state needs
        //    snapshot/restore because GDN advance isn't invertible
        //    (gating < 1 destroys information).
        //
        //    See the reference's qwen3_dflash_graph for the same
        //    pattern — `// Restore SSM state. Replay the accepted
        //    tokens through target.`
        let t_verify_start = std::time::Instant::now();
        let recurrent_snapshot = target.snapshot_recurrent_state()?;
        let mut verify_capture =
            FeatureCapture::new(self.cfg.dflash.target_layer_ids.clone());
        let verify_logits =
            target.forward_with_capture(&verify_ids, past_kv_len, &mut verify_capture)?;
        verify_capture.validate().map_err(candle_core::Error::msg)?;

        // ── 4. Chain verify on the GPU: one batched argmax + one tiny
        //    D→H copy instead of B+1 per-row `to_vec1` calls.
        let target_choices: Vec<u32> = {
            let row_argmax = verify_logits.i(0)?.argmax(D::Minus1)?;
            match row_argmax.dtype() {
                DType::U32 => row_argmax.to_vec1()?,
                _ => row_argmax
                    .to_dtype(DType::U32)?
                    .to_vec1::<u32>()?,
            }
        };
        let verify_ms = t_verify_start.elapsed().as_secs_f64() * 1000.0;
        if target_choices.is_empty() {
            candle_core::bail!(
                "DFlashChainStepper::step: target produced no logits — \
                 forward returned empty output?"
            );
        }

        let mut accepted: Vec<u32> = Vec::with_capacity(block + 1);
        let mut draft_accepted = 0usize;
        let n_compare = candidates.len().min(target_choices.len().saturating_sub(1));
        for i in 0..n_compare {
            let t = target_choices[i];
            accepted.push(t);
            if t == candidates[i] {
                draft_accepted += 1;
            } else {
                break;
            }
        }
        if draft_accepted == candidates.len() {
            if let Some(bonus) = target_choices.get(candidates.len()) {
                accepted.push(*bonus);
            }
        }

        // ── 5. Rollback + commit replay.
        //
        //    full-accept fast path (draft_accepted == block): every
        //    verify-feed position was also a committed-output
        //    predecessor, so the verify forward already produced the
        //    correct KV/recurrent state we need for the next step.
        //    Skip the replay, reuse the verify capture — saves one
        //    target forward per full-accept step.
        //
        //    partial-accept: rollback (truncate attention + restore
        //    recurrent) then replay just the accepted-feed prefix
        //    [last_tok, target_choices[0..draft_accepted-1]]. Its
        //    length is `draft_accepted + 1` — the number of tokens
        //    whose KVs we want in the cache going forward. The
        //    committed boundary `target_choices[draft_accepted]` has
        //    no KV yet; it becomes the next step's `last_tok` and
        //    picks up its KV via feed[0] of that step's verify.
        let t_commit_start = std::time::Instant::now();
        let commit_len = draft_accepted + 1;
        let full_accept = draft_accepted == candidates.len();
        let (commit_capture, commit_ms) = if full_accept {
            // Verify cache state == committed state; no replay needed.
            // We reuse the verify capture for the ring append below.
            (verify_capture, 0.0)
        } else {
            target.truncate_attention_to(past_kv_len)?;
            target.restore_recurrent_state(&recurrent_snapshot)?;
            let mut commit_ids = Vec::with_capacity(commit_len);
            commit_ids.push(last_committed_token);
            for i in 0..draft_accepted {
                commit_ids.push(candidates[i]); // == target_choices[i] since matched
            }
            let commit_input = Tensor::from_vec(commit_ids, (1, commit_len), device)?;
            let mut cap = FeatureCapture::new(self.cfg.dflash.target_layer_ids.clone());
            let _ = target.forward_with_capture(&commit_input, past_kv_len, &mut cap)?;
            cap.validate().map_err(candle_core::Error::msg)?;
            (cap, t_commit_start.elapsed().as_secs_f64() * 1000.0)
        };

        // ── 6. Append feature rows for the commit replay to the ring.
        //
        //    Commit capture rows map to feed positions:
        //      row 0                  — last_tok
        //      row i (1..=K)          — target_choices[i-1]   (K = draft_accepted)
        //
        //    We append ALL K+1 rows. The subtle point: `last_tok` is
        //    the PREVIOUS step's committed boundary/bonus, which never
        //    got a feature during its own step (no feed position for
        //    a boundary in that step's commit). This step's commit
        //    replay IS the first forward where that token's features
        //    are captured, so we need to ring it here. Without this
        //    append, on runs dominated by `draft_accepted=0` steps the
        //    ring never grows past prefill → draft never sees fresh
        //    context → every new step has a stale ring → draft
        //    predictions stay random → chain verify collapses to
        //    pure autoregressive (K=0 every step). Feedback loop.
        //
        //    The only committed token whose feature we never write is
        //    the run's FINAL boundary — one row shy of the committed
        //    sequence length at any moment. The ring is always exactly
        //    one token behind the committed tail; the draft conditions
        //    on "everything but the anchor", which is the correct
        //    semantics for the cross-attention (see the draft's
        //    runner — it feeds `last_tok` separately and uses the
        //    ring for the rest of the history).
        let n_accepted = accepted.len();
        let n_ring_rows = draft_accepted + 1; // rows 0..=draft_accepted of commit_capture
        if n_ring_rows > 0 {
            let captured = fuse_captured_features(&commit_capture, 0, n_ring_rows)?;
            let mut ring = self.ring.lock().unwrap();
            ring.append(&captured)?;
        }

        let _ = n_accepted; // retained for readability of the accept block above

        Ok((
            StepOutcome {
                accepted,
                draft_accepted,
            },
            StepTimings {
                verify_ms,
                commit_ms,
            },
        ))
    }
}

/// Narrow each captured tensor to `seq[start..start + n_rows]` and
/// concatenate along the feature dim. Returns `[n_rows, fused_dim]`
/// ready for [`TargetFeatureRing::append`].
///
/// The DFlash chain-verify feeds `last_tok` at position 0 of the
/// target input, then draft candidates for positions 1..=B. Only
/// rows 1..=accepted map to NEW committed tokens — callers pass
/// `start=1, n_rows=accepted.len()` to pull exactly those.
///
/// Exposed for tests that want to assemble features without running a
/// target forward.
pub fn fuse_captured_features(
    capture: &FeatureCapture,
    start: usize,
    n_rows: usize,
) -> Result<Tensor> {
    capture.validate().map_err(candle_core::Error::msg)?;
    if capture.captured.is_empty() {
        candle_core::bail!("fuse_captured_features: capture is empty");
    }
    if n_rows == 0 {
        // No rows to append — return an empty tensor with the right
        // shape so callers can skip a conditional.
        let first = &capture.captured[0];
        let (_, _, hidden) = first.dims3()?;
        let fused_dim = hidden * capture.captured.len();
        return Tensor::zeros((0, fused_dim), first.dtype(), first.device());
    }
    let mut narrowed: Vec<Tensor> = Vec::with_capacity(capture.captured.len());
    for t in &capture.captured {
        let (b, seq, _hidden) = t.dims3()?;
        if b != 1 {
            candle_core::bail!(
                "fuse_captured_features: captured tensor has batch={}, expected 1",
                b
            );
        }
        if start + n_rows > seq {
            candle_core::bail!(
                "fuse_captured_features: asked for rows [{start}..{}) but capture has only {seq}",
                start + n_rows
            );
        }
        let cropped = t.narrow(1, start, n_rows)?; // [1, n_rows, hidden]
        narrowed.push(cropped);
    }
    let fused_batched = Tensor::cat(&narrowed, D::Minus1)?;
    fused_batched.i(0)
}

/// Pure helper: argmax over the last dim of a 1-D tensor (vocab row).
/// Historically called once per verify position; the hot path now
/// batches argmax on GPU (one kernel + one D→H copy for the whole
/// row block), so this stays only for tests that need a scalar
/// argmax without a trip through candle's kernel dispatch.
#[cfg(test)]
fn argmax_last_dim(row: &Tensor) -> Result<u32> {
    let host: Vec<f32> = row.to_dtype(DType::F32)?.to_vec1()?;
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in host.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    Ok(best_i as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn fuse_captured_features_happy_path() {
        let dev = Device::Cpu;
        // 2 layers, each capturing 4 tokens × 3 hidden dims.
        let t0 = Tensor::arange(0f32, 12.0, &dev)
            .unwrap()
            .reshape((1, 4, 3))
            .unwrap();
        let t1 = Tensor::arange(100f32, 112.0, &dev)
            .unwrap()
            .reshape((1, 4, 3))
            .unwrap();
        let cap = FeatureCapture {
            layer_ids: vec![7, 15],
            captured: vec![t0, t1],
        };
        // Start at row 0, take 2 rows.
        let fused = fuse_captured_features(&cap, 0, 2).unwrap();
        // Expect [2, 6] — 2 rows, 2 layers × 3 hidden = 6 features.
        assert_eq!(fused.dims(), &[2, 6]);
        let got: Vec<f32> = fused.flatten_all().unwrap().to_vec1().unwrap();
        // Row 0: t0[0..3] = [0,1,2] then t1[0..3] = [100,101,102]
        // Row 1: t0[3..6] = [3,4,5] then t1[3..6] = [103,104,105]
        assert_eq!(got, vec![0.0, 1.0, 2.0, 100.0, 101.0, 102.0, 3.0, 4.0, 5.0, 103.0, 104.0, 105.0]);
    }

    #[test]
    fn fuse_captured_features_offset_skips_last_tok_row() {
        // Emulates the chain-verify case: feed was [last, c0, c1, c2, c3],
        // we accept 3 tokens, so we want rows 1..=3 (the 3 NEW positions).
        let dev = Device::Cpu;
        let t = Tensor::arange(0f32, 15.0, &dev)
            .unwrap()
            .reshape((1, 5, 3))
            .unwrap();
        let cap = FeatureCapture {
            layer_ids: vec![0],
            captured: vec![t],
        };
        let fused = fuse_captured_features(&cap, 1, 3).unwrap();
        assert_eq!(fused.dims(), &[3, 3]);
        let got: Vec<f32> = fused.flatten_all().unwrap().to_vec1().unwrap();
        // Rows 1, 2, 3 of [0..15] reshaped (1,5,3): each row has 3 vals.
        // Row 1 = [3,4,5], Row 2 = [6,7,8], Row 3 = [9,10,11]
        assert_eq!(got, vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn fuse_captured_features_rejects_oversize_range() {
        let dev = Device::Cpu;
        let t = Tensor::zeros((1, 3, 4), DType::F32, &dev).unwrap();
        let cap = FeatureCapture {
            layer_ids: vec![0],
            captured: vec![t],
        };
        assert!(fuse_captured_features(&cap, 0, 10).is_err());
        assert!(fuse_captured_features(&cap, 2, 3).is_err());
    }

    #[test]
    fn fuse_captured_features_zero_rows_returns_empty() {
        let dev = Device::Cpu;
        let t = Tensor::zeros((1, 3, 4), DType::F32, &dev).unwrap();
        let cap = FeatureCapture {
            layer_ids: vec![0],
            captured: vec![t],
        };
        let fused = fuse_captured_features(&cap, 0, 0).unwrap();
        assert_eq!(fused.dims(), &[0, 4]);
    }

    #[test]
    fn argmax_picks_highest() {
        let dev = Device::Cpu;
        let row = Tensor::from_vec(vec![0.1f32, 3.2, -1.0, 2.9, 7.1, 0.0], (6,), &dev).unwrap();
        assert_eq!(argmax_last_dim(&row).unwrap(), 4);
    }
}
