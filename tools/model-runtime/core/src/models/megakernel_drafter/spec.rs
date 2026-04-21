//! End-to-end speculative decoding stepper: Qwen3.5-0.8B
//! megakernel drafter × K autoregressive steps + target chain-verify
//! forward on a larger Qwen3.5 target (27B Q4_K_M is the canonical
//! pairing), with drafter state rollback on reject.
//!
//! Shape of one spec round:
//!
//! 1. Snapshot drafter state (DN + conv + position). ~5ms on A6000
//!    (18 MB dn_states + 1.7 MB conv D→D copy).
//! 2. Step drafter K times: each `step()` feeds the previous token,
//!    returns the next greedy-argmax. Collect K candidate tokens.
//! 3. Build verify feed `[last_committed, cand_0, …, cand_{K-1}]`
//!    (K+1 tokens).
//! 4. Run target forward with causal mask — returns logits for each
//!    position; argmax gives K+1 target choices.
//! 5. Chain-accept: walk candidates; accept while
//!    `target_choice[i] == cand_i`. Let A = accepted count (0..=K).
//!    If A == K, also commit `target_choice[K]` as the bonus.
//! 6. Rollback drafter iff A < K: restore snapshot, replay the A
//!    accepted tokens so the DN recurrence is on the accepted
//!    trajectory rather than the full K. (Full-accept skips this.)
//! 7. Commit `accepted + (bonus ? 1 : 0)` tokens to the output, advance
//!    target past_kv_len by `accepted + 1` (the replayed accepted
//!    tokens land in the target's KV via the verify forward; the
//!    bonus token is the next round's `last_committed`).
//!
//! Why no tree verify here: chain is enough for the first pass.
//! DDTree needs per-drafter-step top-K logits, which the megakernel
//! doesn't emit natively. A follow-up either extends `launch_decode`
//! to write out the K best logits or runs a host-side top-K on the
//! LM-head output. Once that lands, the stepper grows a `step_tree`
//! sibling method paralleling `DFlashChainStepper::step_tree`.

#![cfg(feature = "cuda")]

use candle_core::{DType, IndexOp, Result, Tensor, D};

use super::driver::MegakernelDrafter;
use crate::models::dflash_draft::target::DFlashTargetForward;
use crate::models::dflash_draft::capture::FeatureCapture;

/// Tuning knobs for one spec round.
#[derive(Debug, Clone)]
pub struct MegakernelSpecOpts {
    /// Number of drafter tokens to propose per round.
    pub k: usize,
    /// Optional layer ids to capture from the target verify forward
    /// — mirrors `DFlashChainStepper`'s `FeatureCapture` contract.
    /// Not used by the megakernel spec loop (we don't feed target
    /// features back to the drafter), so the default is empty.
    pub target_capture_layers: Vec<usize>,
}

impl Default for MegakernelSpecOpts {
    fn default() -> Self {
        Self {
            k: 6,
            target_capture_layers: Vec::new(),
        }
    }
}

/// One spec round's outcome.
#[derive(Debug, Clone)]
pub struct SpecOutcome {
    /// Tokens committed to the output (accepted drafter tokens plus
    /// optional bonus). Length ≥ 1 (we always commit the target's
    /// disagreement pick), ≤ K + 1.
    pub accepted: Vec<u32>,
    /// How many drafter candidates matched the target's argmax —
    /// zero on immediate reject, K on full accept. Separate from
    /// `accepted.len()` because of the bonus token on full accept.
    pub draft_accepted: usize,
}

/// Per-round wall-time breakdown in milliseconds. Separated so the
/// bench can show exactly where time goes (drafter vs target vs
/// snapshot/restore vs replay).
#[derive(Debug, Clone, Copy, Default)]
pub struct SpecTimings {
    pub snapshot_ms: f64,
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub replay_ms: f64,
}

/// Run ONE spec round against a target + megakernel drafter.
///
/// Caller owns:
///   * the target, behind a `DFlashTargetForward` handle (the same
///     trait the DFlash chain/tree stepper uses, so this integrates
///     with the already-loaded Qwen3.5-27B target infrastructure).
///   * the megakernel drafter, mutably — we snapshot + step + maybe
///     restore.
///
/// Caller must have already:
///   * Called `drafter.prefill(prompt_tokens)` so the drafter's
///     position matches the target's past_kv_len (= prompt_len).
///   * Run the target prefill that produced `last_committed_token`
///     (the token the target picked after the prompt).
///
/// `past_kv_len` is the target's current KV-cache length BEFORE this
/// round's verify forward — i.e. the number of tokens already
/// committed. The verify forward feeds K+1 tokens at
/// `[past_kv_len .. past_kv_len + K + 1]` positions.
#[allow(clippy::too_many_arguments)]
pub fn spec_round<T: DFlashTargetForward>(
    target: &T,
    drafter: &mut MegakernelDrafter,
    last_committed_token: u32,
    past_kv_len: usize,
    opts: &MegakernelSpecOpts,
) -> Result<(SpecOutcome, SpecTimings)> {
    let k = opts.k.max(1);
    let mut timings = SpecTimings::default();

    // ── 1. Snapshot drafter state.
    let t_snap = std::time::Instant::now();
    let snap = drafter.snapshot_state()?;
    timings.snapshot_ms = t_snap.elapsed().as_secs_f64() * 1000.0;

    // ── 2. Drafter: K autoregressive steps.
    let t_draft = std::time::Instant::now();
    let mut draft_ids: Vec<i32> = Vec::with_capacity(k);
    let mut last = last_committed_token as i32;
    for _ in 0..k {
        last = drafter.step(last)?;
        draft_ids.push(last);
    }
    timings.draft_ms = t_draft.elapsed().as_secs_f64() * 1000.0;

    // ── 3. Build verify feed `[last_committed, cand_0, …]`.
    let mut feed: Vec<u32> = Vec::with_capacity(k + 1);
    feed.push(last_committed_token);
    for &tid in &draft_ids {
        feed.push(tid as u32);
    }

    // ── 4. Target forward.
    let device = target.embed_tokens().embeddings().device();
    let feed_len = feed.len();
    let verify_ids = Tensor::from_vec(feed.clone(), (1, feed_len), device)?;
    let t_verify = std::time::Instant::now();
    // Empty FeatureCapture — megakernel spec doesn't need target
    // features (the drafter is autoregressive over raw tokens, not
    // conditioned on target hidden states).
    let mut cap = FeatureCapture::new(opts.target_capture_layers.clone());
    let verify_logits = target.forward_with_capture(&verify_ids, past_kv_len, &mut cap)?;
    timings.verify_ms = t_verify.elapsed().as_secs_f64() * 1000.0;

    // Argmax along vocab dim for every position. Single D→H of
    // (K+1) u32 ids.
    let target_choices: Vec<u32> = {
        let am = verify_logits.i(0)?.argmax(D::Minus1)?;
        match am.dtype() {
            DType::U32 => am.to_vec1()?,
            _ => am.to_dtype(DType::U32)?.to_vec1::<u32>()?,
        }
    };
    if target_choices.is_empty() {
        candle_core::bail!(
            "megakernel spec_round: target produced 0 logits — forward returned empty output?"
        );
    }

    // ── 5. Chain-accept walk.
    let mut accepted: Vec<u32> = Vec::with_capacity(k + 1);
    let mut draft_accepted = 0usize;
    let n_compare = draft_ids
        .len()
        .min(target_choices.len().saturating_sub(1));
    for i in 0..n_compare {
        let t = target_choices[i];
        accepted.push(t);
        if t as i32 == draft_ids[i] {
            draft_accepted += 1;
        } else {
            break;
        }
    }
    if draft_accepted == draft_ids.len() {
        if let Some(bonus) = target_choices.get(draft_ids.len()) {
            accepted.push(*bonus);
        }
    }

    // ── 6. Rollback drafter if we rejected anything.
    let full_accept = draft_accepted == draft_ids.len();
    let t_replay = std::time::Instant::now();
    if !full_accept {
        drafter.restore_state(&snap)?;
        // Replay the accepted candidates so the drafter DN state
        // reflects only what was committed. `draft_accepted` steps
        // because `accepted[draft_accepted]` is the target's
        // disagreement pick — its KV goes into the target but the
        // drafter's next round will feed it as `last_committed`,
        // triggering that token's drafter step at that time.
        let mut replay_last = last_committed_token as i32;
        for i in 0..draft_accepted {
            replay_last = drafter.step(draft_ids[i])?;
        }
        let _ = replay_last;
    }
    timings.replay_ms = t_replay.elapsed().as_secs_f64() * 1000.0;

    Ok((
        SpecOutcome {
            accepted,
            draft_accepted,
        },
        timings,
    ))
}
