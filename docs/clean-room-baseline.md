# Clean-Room Baseline

The first clean-room `CTOX` version starts from two vendored upstream dependencies and nothing more:

1. `references/openai-codex`
   The canonical `codex-cli` / `codex-rs` execution baseline.
2. `ctox-vllm-serve`
   The canonical local runtime and OpenAI-compatible serving layer for local models.

## Bootstrap

Installations and local builds now expect the references to already be present inside the CTOX project tree:

- `references/openai-codex`
- `ctox-vllm-serve`

Run:

```sh
ctox clean-room-bootstrap-deps
```

This Rust entrypoint can still clone or update:

- `https://github.com/openai/codex.git` -> `references/openai-codex`
- `https://github.com/EricLBuehler/mistral.rs.git` -> `ctox-vllm-serve`

Current short-term freeze:

- `openai-codex` is pinned to `c6ab4ee537e5b118a20e9e0d3e0c0023cae2d982`
- bootstrap refuses to update dirty vendor checkouts, so local CTOX patches are not overwritten during `clean-room-bootstrap-deps`
- the main install script no longer uses bootstrap as its source of truth; it builds from the references bundled in the CTOX checkout

## Runtime Families

The first clean-room runtime bridge supports two local model families:

- `GPT-OSS`
  - baseline model example: `openai/gpt-oss-20b`
  - served by the local `vllm-serve` engine with the GPT-OSS startup profile
  - codex-cli stays on the `responses` API
  - a narrow proxy may rewrite tool schemas into the exact `vllm-serve` shape

- `Qwen3.5`
  - baseline model examples: `Qwen/Qwen3.5-27B`
  - served by the local `vllm-serve` vision startup profile
  - follows the `vllm-serve` Qwen3.5 path for image URL, local path, or base64 inputs
  - stays separate from the codex-cli baseline and is attached through custom execution

## Compatibility Tests

The first hard compatibility gate lives in:

```sh
cargo test execution_baseline
```

These Rust unit tests assert that codex-cli-style tool requests are rewritten into a `vllm-serve`-compatible `responses` shape:

- function tools are nested under `function`
- unsupported tool types are removed
- known `vllm-serve` breakers can be filtered
- structured `input` is flattened
- `parallel_tool_calls=false` and `max_tool_calls` are normalized away

This is not yet the full `CTOX` loop. It is the dependency and runtime baseline that the clean-room runtime must stand on before any higher wrapper logic is allowed back in.
