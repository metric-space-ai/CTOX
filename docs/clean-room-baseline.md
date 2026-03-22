# Clean-Room Baseline

The first clean-room `CTOX` version starts from two vendored upstream dependencies and nothing more:

1. `references/openai-codex`
   The canonical `codex-cli` / `codex-rs` execution baseline.
2. `references/mistral.rs`
   The canonical local runtime and OpenAI-compatible serving layer for local models.

## Bootstrap

Run:

```sh
ctox clean-room-bootstrap-deps
```

This Rust entrypoint clones or updates:

- `https://github.com/openai/codex.git` -> `references/openai-codex`
- `https://github.com/EricLBuehler/mistral.rs.git` -> `references/mistral.rs`

## Runtime Families

The first clean-room runtime bridge supports two local model families:

- `GPT-OSS`
  - baseline model example: `openai/gpt-oss-20b`
  - served by `mistralrs serve ... -a gpt_oss`
  - codex-cli stays on the `responses` API
  - a narrow proxy may rewrite tool schemas into the exact `mistral.rs` shape

- `Qwen3.5`
  - baseline model examples: `Qwen/Qwen3.5-27B`, `Qwen/Qwen3.5-35B-A3B`
  - served by `mistralrs serve vision ...`
  - follows the `mistral.rs` Qwen3.5 path for image URL, local path, or base64 inputs
  - stays separate from the codex-cli baseline and is attached through custom execution

## Compatibility Tests

The first hard compatibility gate lives in:

```sh
cargo test execution_baseline
```

These Rust unit tests assert that codex-cli-style tool requests are rewritten into a `mistral.rs`-compatible `responses` shape:

- function tools are nested under `function`
- unsupported tool types are removed
- known `mistral.rs` breakers can be filtered
- structured `input` is flattened
- `parallel_tool_calls=false` and `max_tool_calls` are normalized away

This is not yet the full `CTOX` loop. It is the dependency and runtime baseline that the clean-room runtime must stand on before any higher wrapper logic is allowed back in.
