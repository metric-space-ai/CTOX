# CTOX

`CTOX` is the clean-room restart of the earlier CTO-Agent effort.

The active project is intentionally minimal. Its first responsibility is only this:

- vendor `codex-cli` / `codex-rs`
- vendor `mistral.rs`
- define the local execution baseline for `GPT-OSS` and `Qwen3.5`
- keep the official Codex `responses` path compatible with the local `mistral.rs` runtime

Everything from the old project that is not part of that minimal baseline was moved out of the active compile path into:

- `old-legacy-for-transplation-only/`

That directory is ignored and exists only as a later transplant source.

## Active Files

The active Rust code currently lives in:

- `src/main.rs`
- `src/execution_baseline.rs`

## Install

Local install:

```sh
./scripts/install_ctox.sh
```

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

Rewrite a captured Codex `responses` payload into the narrower `mistral.rs`-compatible form:

```sh
ctox clean-room-rewrite-responses /path/to/request.json
```

## Tests

Run the active Rust tests with:

```sh
cargo test execution_baseline
```

These tests currently cover:

- dependency discovery
- `mistralrs` startup plans for `GPT-OSS` and `Qwen3.5`
- `codex-exec` baseline planning for the `responses` path
- Rust-side rewrite of Codex `responses` payloads into the shape currently needed for `mistral.rs`
