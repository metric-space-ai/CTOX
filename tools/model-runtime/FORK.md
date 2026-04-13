# CTOX Fork Record: candle-fork

Canonical path: `tools/model-runtime`

Origin:

- Canonical CTOX classification: Candle-derived serving hard fork
- Historical lineage: older imports and naming passed through `mistral.rs` / `engine.rs`
- Integration mode: `hard_fork`

Fork policy:

- This tree is integrated directly into CTOX and is not treated as a package dependency.
- This tree is integrated directly into CTOX as part of the model-serving execution layer.
- Local modifications inside this subtree belong to the CTOX fork state unless explicitly documented otherwise.
- CTOX must not auto-clone, auto-fetch, or auto-update this subtree from upstream.

Attribution rule:

- Treat the checked-in source tree as authoritative CTOX fork state.
- Do not attribute local changes to a precise upstream Candle snapshot unless the provenance is recorded explicitly for that file or change.
