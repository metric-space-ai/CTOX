# Pinned static model catalog

`models.json` is the source snapshot of
`internal/registry/models/models.json` from CLIProxyAPI commit
`ffdb9c9fbc78a6235d59c9ccbdc4243ba35ecdcd` (MIT).

The upstream file has SHA-256
`483f7fb1b0f159bcda08c01ea91e21162b8f50ad34e83b7d7884e6a5384525c7`.
The checked-in text ends with the repository-standard newline; the Rust guard
hashes `trim_end()` so it verifies the upstream bytes rather than that local
newline normalization.

When advancing the upstream pin, replace this asset from the new pin, update
the hash constant in `model_definitions.rs`, then run
`scripts/run_management_models_differential.sh`. The differential probe calls
the pinned Go registry and catches both asset drift and changes to code-defined
Codex/xAI built-ins.
