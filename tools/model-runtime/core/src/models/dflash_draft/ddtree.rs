//! DDTree — tree-structured verify for DFlash speculative decoding.
//!
//! Direct Rust port of the reference `dflash/test/test_dflash.cpp`
//! DDTree section (lines 117–427). Ported verbatim at the algorithm
//! level: same online-logsumexp + best-first heap tree build + tree
//! walk + ancestor-visibility mask.
//!
//! Why: the current CTOX DFlash pipeline runs chain verify (accept up
//! to K tokens along a single greedy path from the draft). Chain
//! acceptance length (AL) on Qwen3.5-27B Q4_K_M HumanEval is ~3; DDTree
//! budget=22 in the reference recovers AL ~8.3, which is where the
//! 130+ tok/s headline number comes from. The algorithm itself is the
//! cheap part — the expensive part is the custom tree attention mask
//! on the target's attention kernel, which is wired in separately.
//!
//! Contract (from the reference):
//!  * Inputs are draft logits `[n_positions, vocab]` laid out row-major
//!    on CPU after detach from GPU.
//!  * `extract_draft_topk` returns per-position top-K log-probabilities
//!    sorted descending (rank 0 = argmax).
//!  * `build_ddtree` expands at most `budget` nodes via a best-first
//!    heap over log-probability sums; `chain_seed=true` pre-seeds the
//!    full top-1 chain so AL never regresses below chain mode even
//!    when Q4_K_M flattens the draft softmax.
//!  * `follow_verified_tree` walks from the root, at each node picking
//!    the child whose token_id matches the target's posterior argmax
//!    at that node; stops when no child matches and returns the
//!    accepted flat indices plus the "bonus" next-token argmax.
//!  * `build_tree_mask` writes an f16 attention mask with −inf for
//!    non-ancestor positions and 0 for ancestors + the past KV block.

use std::collections::{BinaryHeap, HashMap};

/// Hard cap on the tree size. The reference uses 64 (default) and
/// recommends budget=22 for Qwen3.5-27B Q4_K_M (where the draft
/// softmax is too flat for larger budgets to pay off).
pub const DEFAULT_DDTREE_BUDGET: usize = 22;

/// A flat DFS-ordered tree built from the draft's top-K softmax
/// distributions. Slot 0 is the root (the bonus token inherited from
/// the previous spec-decode step). Slots `1..=n_nodes` are the DFS-
/// ordered tree nodes. All vectors except `parents`/`child_maps`/
/// `visibility` are sized `n_nodes`; those three include the root and
/// are sized `n_nodes + 1`.
#[derive(Debug, Clone, Default)]
pub struct DDTree {
    /// Number of non-root nodes.
    pub n_nodes: usize,
    /// Token id for each non-root node, indexed 0..n_nodes.
    pub token_ids: Vec<i32>,
    /// Absolute tree depth (1..=L) of each non-root node.
    pub depths: Vec<i32>,
    /// Parent flat-index for each slot including root (root's parent = -1).
    pub parents: Vec<i32>,
    /// Per-slot map from child-token-id to child flat-index.
    pub child_maps: Vec<HashMap<i32, i32>>,
    /// Ancestor-only visibility mask, row-major `(1+n_nodes) x (1+n_nodes)`.
    /// `visibility[i * N + j] == true` iff `j` is an ancestor of `i`
    /// (including `j == i`).
    pub visibility: Vec<bool>,
}

impl DDTree {
    /// `1 + n_nodes` — the side length of the visibility matrix.
    #[inline]
    pub fn side_len(&self) -> usize {
        self.n_nodes + 1
    }
}

/// Extract per-position top-K log-probabilities and token ids from
/// draft logits.
///
/// Runs a single pass per position that simultaneously (a) maintains
/// a running max + running sum of `exp(x - max)` (online logsumexp, so
/// we never allocate a full softmax buffer) and (b) keeps a min-heap
/// of size `k` holding the current top-K logits. At the end, the
/// top-K logits are normalised by the true `log Z` and emitted in
/// descending order (rank 0 = argmax).
///
/// `temperature < 1.0` sharpens the softmax (widens the top-1 gap)
/// and is recommended for Q4_K_M draft where quantization flattens
/// the distribution; the reference uses 1.0 for the default.
///
/// Input layout: `logits[i * vocab + j]` for position `i`, token `j`.
/// Output layout: `out_log_probs[i * k + r]` for position `i`, rank `r`.
pub fn extract_draft_topk(
    logits: &[f32],
    n_positions: usize,
    vocab: usize,
    k: usize,
    temperature: f32,
) -> (Vec<f32>, Vec<i32>) {
    assert!(k > 0, "DDTree top-K must be > 0");
    assert_eq!(
        logits.len(),
        n_positions * vocab,
        "extract_draft_topk: logits len mismatch"
    );
    let inv_t = 1.0 / temperature.max(1e-3);
    let mut out_log_probs = vec![0.0_f32; n_positions * k];
    let mut out_token_ids = vec![0_i32; n_positions * k];

    for i in 0..n_positions {
        let row = &logits[i * vocab..i * vocab + vocab];

        // Min-heap keyed on the scaled logit so popping the smallest
        // when we overflow K entries leaves the K largest behind.
        // `std::collections::BinaryHeap` is a max-heap, so we wrap
        // the key in `Reverse` — but we also want tie-break on id,
        // so we encode the key as an `OrdEntry` struct below.
        let mut heap: BinaryHeap<MinHeapEntry> = BinaryHeap::with_capacity(k);

        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum_exp = 0.0_f32;

        for j in 0..vocab {
            let l = row[j] * inv_t;

            // Online logsumexp with rescaling when a new max appears.
            if l > running_max {
                if running_max > f32::NEG_INFINITY {
                    running_sum_exp *= (running_max - l).exp();
                }
                running_sum_exp += 1.0;
                running_max = l;
            } else {
                running_sum_exp += (l - running_max).exp();
            }

            // Top-K min-heap: push until full, then replace the
            // smallest if the new logit beats it.
            if heap.len() < k {
                heap.push(MinHeapEntry {
                    logit: l,
                    id: j as i32,
                });
            } else if let Some(top) = heap.peek() {
                if l > top.logit {
                    heap.pop();
                    heap.push(MinHeapEntry {
                        logit: l,
                        id: j as i32,
                    });
                }
            }
        }

        let log_z = running_max + running_sum_exp.ln();

        // Pour the heap into a vector and sort DESCENDING by logit.
        let mut entries: Vec<MinHeapEntry> = heap.into_vec();
        entries.sort_by(|a, b| {
            // `MinHeapEntry::cmp` inverts for the heap; we want
            // natural descending order here.
            b.logit.partial_cmp(&a.logit).unwrap_or(std::cmp::Ordering::Equal)
        });
        for (r, entry) in entries.iter().enumerate() {
            out_log_probs[i * k + r] = entry.logit - log_z;
            out_token_ids[i * k + r] = entry.id;
        }
    }

    (out_log_probs, out_token_ids)
}

/// Min-heap entry for top-K — smallest logit bubbles to `.peek()`.
#[derive(Clone, Copy)]
struct MinHeapEntry {
    logit: f32,
    id: i32,
}
impl PartialEq for MinHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.logit == other.logit && self.id == other.id
    }
}
impl Eq for MinHeapEntry {}
impl PartialOrd for MinHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MinHeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Invert so that the smallest-logit entry is the heap's max.
        other
            .logit
            .partial_cmp(&self.logit)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.id.cmp(&self.id))
    }
}

/// Best-first heap entry for the DDTree construction.
#[derive(Clone)]
struct BuildHeapEntry {
    /// Negated log-probability sum — max-heap semantics give us the
    /// lowest neg_logw (= highest logw = most probable prefix) first.
    neg_logw: f32,
    parent_index: i32,
    depth: i32,
    rank: i32,
    logw: f32,
}
impl PartialEq for BuildHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.neg_logw == other.neg_logw
    }
}
impl Eq for BuildHeapEntry {}
impl PartialOrd for BuildHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for BuildHeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a max-heap; we want smallest neg_logw at top.
        other
            .neg_logw
            .partial_cmp(&self.neg_logw)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.depth.cmp(&self.depth))
            .then_with(|| other.rank.cmp(&self.rank))
    }
}

/// Build a DFS-ordered DDTree from the draft's per-position top-K
/// distributions. Runs a best-first heap expanding at most `budget`
/// non-root nodes.
///
/// `top_log_probs`/`top_token_ids`: `[l * k + r]` for depth-1 index
/// `l` (0..L), rank `r`. Emitted by [`extract_draft_topk`].
/// `l_max`: maximum tree depth (equals draft `block_size - 1`).
/// `k`: top-K per position used when extracting the above tensors.
/// `budget`: maximum non-root node count.
/// `chain_seed`: if true, pre-seed the full top-1 chain so AL never
///   drops below chain mode. Otherwise pure best-first (paper mode).
pub fn build_ddtree(
    top_log_probs: &[f32],
    top_token_ids: &[i32],
    l_max: usize,
    k: usize,
    budget: usize,
    chain_seed: bool,
) -> DDTree {
    let mut tree = DDTree::default();
    tree.parents.push(-1);
    tree.child_maps.push(HashMap::new());

    if budget == 0 || l_max == 0 {
        tree.visibility.push(true);
        return tree;
    }

    assert_eq!(top_log_probs.len(), l_max * k, "top_log_probs shape");
    assert_eq!(top_token_ids.len(), l_max * k, "top_token_ids shape");

    let mut heap: BinaryHeap<BuildHeapEntry> = BinaryHeap::new();

    tree.token_ids.reserve(budget);
    tree.depths.reserve(budget);
    tree.parents.reserve(budget + 1);

    if chain_seed {
        let chain_depth = l_max.min(budget);
        let mut cum_logw = 0.0_f32;
        let mut prev_idx = 0_i32;
        for d in 1..=chain_depth {
            let tok_id = top_token_ids[(d - 1) * k];
            cum_logw += top_log_probs[(d - 1) * k];

            let cur_idx = (tree.n_nodes as i32) + 1;
            tree.token_ids.push(tok_id);
            tree.depths.push(d as i32);
            tree.parents.push(prev_idx);
            tree.child_maps.push(HashMap::new());
            tree.child_maps[prev_idx as usize].insert(tok_id, cur_idx);
            tree.n_nodes += 1;

            if k > 1 {
                let sibling_logw = cum_logw - top_log_probs[(d - 1) * k]
                    + top_log_probs[(d - 1) * k + 1];
                heap.push(BuildHeapEntry {
                    neg_logw: -sibling_logw,
                    parent_index: prev_idx,
                    depth: d as i32,
                    rank: 1,
                    logw: sibling_logw,
                });
            }
            prev_idx = cur_idx;
        }
    } else {
        let root_logw = top_log_probs[0];
        heap.push(BuildHeapEntry {
            neg_logw: -root_logw,
            parent_index: 0,
            depth: 1,
            rank: 0,
            logw: root_logw,
        });
    }

    while let Some(top) = heap.pop() {
        if tree.n_nodes >= budget {
            break;
        }

        let depth_minus_1 = (top.depth - 1) as usize;
        let rank = top.rank as usize;
        let token_id = top_token_ids[depth_minus_1 * k + rank];

        let current_index = (tree.n_nodes as i32) + 1;
        tree.token_ids.push(token_id);
        tree.depths.push(top.depth);
        tree.parents.push(top.parent_index);
        tree.child_maps.push(HashMap::new());
        tree.child_maps[top.parent_index as usize].insert(token_id, current_index);
        tree.n_nodes += 1;

        // Next sibling at the same depth, next-best rank.
        if rank + 1 < k {
            let sibling_logw = top.logw - top_log_probs[depth_minus_1 * k + rank]
                + top_log_probs[depth_minus_1 * k + rank + 1];
            heap.push(BuildHeapEntry {
                neg_logw: -sibling_logw,
                parent_index: top.parent_index,
                depth: top.depth,
                rank: (rank + 1) as i32,
                logw: sibling_logw,
            });
        }

        // First child at depth+1, top-1 rank under this node.
        if (top.depth as usize) < l_max {
            let child_logw = top.logw + top_log_probs[top.depth as usize * k];
            heap.push(BuildHeapEntry {
                neg_logw: -child_logw,
                parent_index: current_index,
                depth: top.depth + 1,
                rank: 0,
                logw: child_logw,
            });
        }
    }

    // Ancestor-only visibility mask: row i inherits row parents[i]
    // up to column i-1, then marks self at column i.
    let n = tree.side_len();
    tree.visibility = vec![false; n * n];
    tree.visibility[0] = true;
    for i in 1..n {
        let p = tree.parents[i] as usize;
        for j in 0..i {
            tree.visibility[i * n + j] = tree.visibility[p * n + j];
        }
        tree.visibility[i * n + i] = true;
    }

    tree
}

/// Walk the verified tree greedily: start at the root, at each node
/// pick the child whose token id matches the target's argmax at that
/// node. Returns (accepted_flat_indices, bonus_next_token), where the
/// bonus token is the target's argmax at the last accepted node —
/// either because no child matched it, or because we reached a leaf.
///
/// `posterior` is size `1 + tree.n_nodes` (argmax per flat slot).
pub fn follow_verified_tree(tree: &DDTree, posterior: &[i32]) -> (Vec<i32>, i32) {
    assert_eq!(posterior.len(), tree.side_len(), "posterior shape");
    let mut accepted = Vec::with_capacity(tree.n_nodes + 1);
    accepted.push(0);

    let mut current_index = 0_usize;
    let mut next_token = posterior[current_index];
    loop {
        let children = &tree.child_maps[current_index];
        match children.get(&next_token) {
            Some(&cur) => {
                current_index = cur as usize;
                accepted.push(cur);
                next_token = posterior[current_index];
            }
            None => break,
        }
    }
    (accepted, next_token)
}

/// Build an attention mask for tree verify. Returns a `Vec<f16>` in
/// `[kv_len, q_len]` row-major (kv fastest) layout, with
/// `mask[q, k] = 0` for allowed attention and `-inf` otherwise.
///
/// `past_length` is the number of tokens already committed to the
/// target's KV cache (all visible). The tree region sits at KV
/// positions `[past_length, past_length + side_len)` and follows the
/// tree's ancestor-only mask.
///
/// This function does not pad to any alignment — the caller is
/// responsible for applying model-specific padding (e.g. `KQ_MASK_PAD`
/// for flash-attention) on top of the returned shape.
pub fn build_tree_mask(tree: &DDTree, past_length: usize) -> (Vec<half::f16>, usize, usize) {
    let n = tree.side_len();
    let kv_len = past_length + n;
    let q_len = n;
    let neg_inf = half::f16::NEG_INFINITY;
    let zero = half::f16::ZERO;
    let mut mask = vec![neg_inf; kv_len * q_len];

    for q in 0..q_len {
        // Past KV — always visible.
        for k in 0..past_length {
            mask[q * kv_len + k] = zero;
        }
        // Tree region — ancestors-only.
        for j in 0..n {
            if tree.visibility[q * n + j] {
                mask[q * kv_len + (past_length + j)] = zero;
            }
        }
    }

    (mask, q_len, kv_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_argmax_matches_argmax() {
        // One position, 8 tokens, easy-to-eyeball logits.
        let logits = vec![0.1_f32, 5.0, 0.3, -2.0, 4.0, 0.5, -1.0, 3.0];
        let (lp, ids) = extract_draft_topk(&logits, 1, 8, 3, 1.0);
        assert_eq!(ids[0], 1, "rank-0 must be argmax (logit 5.0)");
        assert_eq!(ids[1], 4, "rank-1 must be logit 4.0");
        assert_eq!(ids[2], 7, "rank-2 must be logit 3.0");
        // Softmax over [5, 4, 3, 0.5, 0.3, 0.1, -1, -2] has a
        // dominant mass on rank 0; rank-0 log-prob must be highest.
        assert!(lp[0] > lp[1] && lp[1] > lp[2]);
    }

    #[test]
    fn build_ddtree_chain_seed_produces_exact_chain_when_budget_eq_l() {
        // L=4, K=2, budget=4: chain-seed should fill the chain exactly.
        let top_log_probs = vec![
            -0.1, -2.0, // d=1
            -0.2, -2.1, // d=2
            -0.3, -2.2, // d=3
            -0.4, -2.3, // d=4
        ];
        let top_tokens = vec![10, 11, 20, 21, 30, 31, 40, 41];
        let tree = build_ddtree(&top_log_probs, &top_tokens, 4, 2, 4, true);
        assert_eq!(tree.n_nodes, 4);
        assert_eq!(tree.token_ids, vec![10, 20, 30, 40]);
        // parents: root=0, chain each other
        assert_eq!(tree.parents, vec![-1, 0, 1, 2, 3]);
        // depths 1..=4
        assert_eq!(tree.depths, vec![1, 2, 3, 4]);
    }

    #[test]
    fn follow_verified_tree_walks_to_leaf() {
        // Build a trivial 2-node chain: root -> [t=42] -> [t=99].
        let mut tree = DDTree::default();
        tree.parents = vec![-1, 0, 1];
        tree.token_ids = vec![42, 99];
        tree.depths = vec![1, 2];
        tree.n_nodes = 2;
        tree.child_maps = vec![
            [(42_i32, 1_i32)].into_iter().collect(),
            [(99_i32, 2_i32)].into_iter().collect(),
            HashMap::new(),
        ];
        // Posterior: root predicts 42 (matches first child), node-1
        // predicts 99 (matches second child), leaf predicts 123
        // (no child → stop, 123 is the bonus).
        let posterior = vec![42_i32, 99, 123];
        let (accepted, bonus) = follow_verified_tree(&tree, &posterior);
        assert_eq!(accepted, vec![0, 1, 2]);
        assert_eq!(bonus, 123);
    }

    #[test]
    fn tree_mask_shape_and_ancestor_semantics() {
        // Same 2-node chain. past_length=3, so kv_len = 3 + 3 = 6, q=3.
        let mut tree = DDTree::default();
        tree.parents = vec![-1, 0, 1];
        tree.n_nodes = 2;
        tree.child_maps = vec![HashMap::new(), HashMap::new(), HashMap::new()];
        // Build the visibility matrix for this chain.
        let n = tree.side_len();
        tree.visibility = vec![false; n * n];
        tree.visibility[0] = true; // root sees root
        tree.visibility[n + 0] = true; // node1 sees root
        tree.visibility[n + 1] = true; // node1 sees itself
        tree.visibility[2 * n + 0] = true; // node2 sees root
        tree.visibility[2 * n + 1] = true; // node2 sees node1
        tree.visibility[2 * n + 2] = true; // node2 sees itself

        let (mask, q_len, kv_len) = build_tree_mask(&tree, 3);
        assert_eq!(q_len, 3);
        assert_eq!(kv_len, 6);
        assert_eq!(mask.len(), 18);
        // Past is all zero for every q.
        for q in 0..3 {
            for k in 0..3 {
                assert_eq!(mask[q * 6 + k], half::f16::ZERO, "past kv q={q} k={k}");
            }
        }
        // Tree region: q=0 (root) sees only slot 0 (past+0) → zero,
        // slots 1..3 (past+1, past+2) remain -inf.
        assert_eq!(mask[0 * 6 + 3], half::f16::ZERO);
        assert_eq!(mask[0 * 6 + 4], half::f16::NEG_INFINITY);
        assert_eq!(mask[0 * 6 + 5], half::f16::NEG_INFINITY);
        // q=1 (node1) sees slot 0 and slot 1.
        assert_eq!(mask[1 * 6 + 3], half::f16::ZERO);
        assert_eq!(mask[1 * 6 + 4], half::f16::ZERO);
        assert_eq!(mask[1 * 6 + 5], half::f16::NEG_INFINITY);
        // q=2 (node2) sees all three.
        assert_eq!(mask[2 * 6 + 3], half::f16::ZERO);
        assert_eq!(mask[2 * 6 + 4], half::f16::ZERO);
        assert_eq!(mask[2 * 6 + 5], half::f16::ZERO);
    }
}
