# CTOX

`CTOX` is the clean-room restart of the earlier CTO-Agent effort.

The active project is intentionally minimal. Its first responsibility is only this:

- vendor `codex-cli` / `codex-rs`
- own the forked local engine in `ctox-vllm-serve/`
- define the local execution baseline for `GPT-OSS` and `Qwen3.5`
- keep the official Codex `responses` path compatible with the local CTOX engine runtime

Everything from the old project that is not part of that minimal baseline was moved out of the active compile path into:

- `old-legacy-for-transplation-only/`

That directory is ignored and exists only as a later transplant source.

## Active Files

The active Rust code currently lives in:

- `src/main.rs`
- `src/execution_baseline.rs`
- `src/channels.rs`

## Install

Local install:

```sh
./scripts/install_ctox.sh
```

The installer now does five baseline steps:

- builds `ctox`
- uses the project-bundled `openai-codex` sources under `references/` and the forked engine under `ctox-vllm-serve/`
- syncs repo-managed system skills into the vendored `codex` build before compilation
- builds the forked `vllm-serve` engine binary and vendored `codex-exec`
- writes a minimal runtime file at `runtime/vllm_serve.env`
- installs bundled skills into `CODEX_HOME/skills` and `CODEX_HOME/skills/.system`

Short-term vendor safety:

- `openai-codex` is currently pinned to the exact upstream commit `c6ab4ee537e5b118a20e9e0d3e0c0023cae2d982`
- install expects `references/openai-codex` and `ctox-vllm-serve` to already exist in the project checkout
- `ctox clean-room-bootstrap-deps` now refuses to update a dirty vendor checkout, so local CTOX customizations in `references/openai-codex` are not silently overwritten

Active local support is intentionally narrow:

- `openai/gpt-oss-20b`
- `Qwen/Qwen3.5-27B`

Other model families or larger Qwen3.5 variants remain out of the active support matrix until they are validated end-to-end on target hosts.

On Linux CUDA hosts, the installer also carries over the current CTOX `vllm-serve` runtime assumptions:

- detect `cuda flash-attn` features automatically
- add `nccl` when NCCL is present
- add `cudnn` when `libcudnn` is present
- install missing CUDA build prerequisites via `apt` when needed
- install `libnccl2` / `libnccl-dev` on multi-GPU hosts when the runtime is still missing

One-liner remote install:

```sh
bash -lc "$(curl -fsSL https://raw.githubusercontent.com/metric-space-ai/CTOX/main/scripts/install_ctox_remote.sh)"
```

Default target directory:

```text
~/ctox
```

## Commands

Bootstrap vendored dependencies:

```sh
ctox clean-room-bootstrap-deps
```

Print the GPT-OSS baseline plan:

```sh
ctox clean-room-baseline-plan gpt_oss "Reply with CTOX_OK and nothing else."
```

Print the Qwen3.5 baseline plan:

```sh
ctox clean-room-baseline-plan qwen3_5 "Describe the attached image."
```

Rewrite a captured Codex `responses` payload into the narrower `vllm-serve`-compatible form:

```sh
ctox clean-room-rewrite-responses /path/to/request.json
```

Run the local `vllm-serve` backend through the active CTOX launcher:

```sh
./scripts/run_vllm_serve_backend.sh
```

The launcher reads `runtime/vllm_serve.env` and applies the active CUDA/NCCL runtime environment before starting the local `vllm-serve` engine process.
If no explicit GPU override is set, the launcher uses all visible NVIDIA GPUs by default and derives the local NCCL world size from them for tensor-parallel families.

Initialize and inspect the shared channel store:

```sh
ctox channel init
ctox channel list --limit 20
```

Sync inbound messages from adapters into the shared SQLite substrate:

```sh
ctox channel sync --channel email --email you@example.com
ctox channel sync --channel jami --account-id jami:youraccount
```

Lease, acknowledge, and send multi-channel messages:

```sh
ctox channel take --limit 5 --lease-owner codex
ctox channel ack email-import::example
ctox channel send --channel tui --account-key tui:local --thread-key local/test --body "hello"
```

## Tests

Run the active Rust tests with:

```sh
cargo test execution_baseline
```

These tests currently cover:

- dependency discovery
- `vllm-serve` startup plans for `GPT-OSS` and `Qwen3.5`
- `codex-exec` baseline planning for the `responses` path
- Rust-side rewrite of Codex `responses` payloads into the shape currently needed for `vllm-serve`

## Current Scope

The active project still does not claim that `codex-exec + vllm-serve + GPT-OSS 20B` is fully validated on every host. The current goal is narrower:

- make the local install path honest
- keep the `responses` bridge in Rust
- make GPU-host assumptions explicit and reproducible
- use legacy code only as a transplant source, not as an active dependency
