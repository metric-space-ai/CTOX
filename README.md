# CTOX

CTOX brings autonomy to servers.

It is an always-on operations agent that you install on a server or into your own network so it can look after the systems entrusted to it 24/7.

CTOX is deliberately not a private agent for everything. It is not primarily a lifestyle assistant, not a general-purpose personal chatbot, and not just another development helper. Its center of gravity is operations: persistent execution, follow-through, checks, queues, scheduling, communication, and responsibility for servers and services.

At its core, CTOX combines:

- an extended always-on execution loop based on `codex-cli` / `codex-rs`
- a local proxy and inference engine in `ctox-vllm-serve/`
- a focused operations layer for autonomy, continuity, scheduling, communication, and host-side control

## Operating Model

CTOX is designed to run as a persistent local control plane on a host you operate.

In the current implementation, that means:

- a long-running CTOX service loop that can stay active in the background
- a local OpenAI-compatible `responses` proxy for Codex
- persistent local sidecars for chat, embeddings, STT, and TTS
- an explicit routing substrate for queue items, channels, plans, schedules, and follow-up work
- a local continuity and compaction layer so long-running work can keep moving without turning into an unbounded chat transcript

CTOX is meant to help operate servers and services continuously, not just answer one prompt at a time.

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

The installer currently does these baseline steps:

- builds `ctox`
- uses the project-bundled `openai-codex` sources under `references/` and the forked engine under `ctox-vllm-serve/`
- syncs repo-managed system skills into the vendored `codex` build before compilation
- builds the forked `vllm-serve` engine binary and vendored `codex-exec`
- writes a minimal runtime file at `runtime/vllm_serve.env`
- installs bundled skills into `CODEX_HOME/skills` and `CODEX_HOME/skills/.system`
- on Linux, installs and enables a persistent `systemd --user` `ctox.service` background loop
- on supported Linux distributions, installs the Jami daemon runtime when needed

Short-term vendor safety:

- `openai-codex` is currently pinned to the exact upstream commit `c6ab4ee537e5b118a20e9e0d3e0c0023cae2d982`
- install expects `references/openai-codex` and `ctox-vllm-serve` to already exist in the project checkout
- `ctox clean-room-bootstrap-deps` now refuses to update a dirty vendor checkout, so local CTOX customizations in `references/openai-codex` are not silently overwritten

## Core Commands

Manage the persistent CTOX loop:

```sh
ctox status
ctox start
ctox stop
ctox
```

On Linux installs with `systemd --user`, `ctox start` enables and starts `ctox.service`, and `ctox stop` stops and disables it again.

Bootstrap vendored dependencies:

```sh
ctox clean-room-bootstrap-deps
```

Inspect clean-room runtime plans:

```sh
ctox clean-room-baseline-plan gpt_oss "Reply with CTOX_OK and nothing else."
ctox clean-room-baseline-plan qwen3_5 "Describe the attached image."
```

Rewrite a captured Codex `responses` payload into the narrower `vllm-serve`-compatible form:

```sh
ctox clean-room-rewrite-responses /path/to/request.json
```

Apply local chat runtime presets:

```sh
ctox chat-runtime-apply openai/gpt-oss-20b quality
ctox chat-runtime-apply Qwen/Qwen3.5-35B-A3B performance
ctox chat-runtime-apply-legacy openai/gpt-oss-20b
```

Run the local backend through the active CTOX launcher:

```sh
./scripts/run_vllm_serve_backend.sh
```

The launcher reads `runtime/vllm_serve.env` and applies the active CUDA/NCCL runtime environment before starting the local `vllm-serve` process.

If no explicit GPU override is set, the launcher uses all visible NVIDIA GPUs by default and derives the local NCCL world size from them for tensor-parallel families.

If `CTOX_AUXILIARY_CUDA_VISIBLE_DEVICES` is set, the persistent embedding, STT, and TTS sidecars bind to those GPUs and the main chat backend prefers the remaining visible devices.

The persistent CTOX service keeps these local HTTP sidecars warm in parallel:

- chat backend on `CTOX_VLLM_SERVE_PORT`
- responses proxy on `CTOX_PROXY_PORT`
- embeddings on `CTOX_EMBEDDING_PORT`
- STT on `CTOX_STT_PORT`
- TTS on `CTOX_TTS_PORT`

Auxiliary CPU fallbacks:

- embeddings can now stay on `Qwen/Qwen3-Embedding-0.6B` but run in a CPU-only mode
- STT can switch from Voxtral on GPU to a `faster-whisper` CPU path
- TTS now standardizes its CPU fallback on Speaches-backed Piper voices, so the proxy and Rust control plane only need one CPU speech-serving path while still covering languages such as `de_DE`, `fr_FR`, and `en_US`

The TUI exposes these as explicit `[GPU]` and `[CPU]` variants on the auxiliary model rows so hosts without CUDA can keep embeddings, transcription, and speech generation available.

## Ops Substrate

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

Create and inspect persistent multi-step plans:

```sh
ctox plan init
ctox plan draft --title "remote rollout" --prompt "inspect host, patch deploy script, run smoke check"
ctox plan ingest --title "remote rollout" --prompt "inspect host, patch deploy script, run smoke check"
ctox plan list
ctox plan show --goal-id <goal-id>
ctox plan emit-next --goal-id <goal-id>
ctox plan block-step --step-id <step-id> --reason "waiting for owner approval"
ctox plan unblock-step --step-id <step-id>
```

Inspect and manage the explicit execution queue:

```sh
ctox queue list
ctox queue add --title "run smoke check" --prompt "Run the remote smoke check on host X and summarize failures." --skill "follow-up-orchestrator" --priority high
ctox queue show --message-key <message-key>
ctox queue reprioritize --message-key <message-key> --priority urgent
ctox queue block --message-key <message-key> --reason "waiting for owner approval"
ctox queue release --message-key <message-key>
ctox queue complete --message-key <message-key> --note "smoke check passed"
```

Evaluate whether completed work should end, replan, block, or emit follow-up work:

```sh
ctox follow-up evaluate \
  --goal "stabilize remote rollout" \
  --result "deploy script patched; smoke test still pending" \
  --step-title "patch deploy script" \
  --open-item "run remote smoke test"
```

CTOX carries these orchestration pieces so autonomous server work can continue, be resumed, be scheduled, and be routed. They are part of the operations model, not an attempt to become a general consumer agent.

## Bundled Admin Skills

CTOX now also carries bundled system skills for core admin work:

- `discovery-graph`
- `reliability-ops`
- `change-lifecycle`
- `security-posture`
- `recovery-assurance`
- `incident-response`
- `automation-engineering`
- `ops-insight`

These skills sit on top of concrete host and service tooling such as `htop`, `btop`, `top`, `ps`, `vmstat`, `iostat`, `ss`, `journalctl`, `systemctl`, `df`, `du`, `curl`, and `nvidia-smi`, plus the existing CTOX `queue`, `plan`, `schedule`, and `follow-up` substrate.

They do not introduce a second execution loop. They are additional skill-level operating knowledge that rides the existing CTOX routing and Codex execution path.

## Continuity And Retrieval

Inspect and operate the local conversation memory and continuity substrate:

```sh
ctox lcm-init runtime/ctox_lcm.db
ctox lcm-dump runtime/ctox_lcm.db 1
ctox lcm-compact runtime/ctox_lcm.db 1
ctox lcm-show-continuity runtime/ctox_lcm.db 1
ctox context-retrieve --db runtime/ctox_lcm.db --conversation-id 1 --mode current
```

The current LCM and continuity layer exists so long-running operational work can preserve narrative state, anchors, focus, retrieval, and compaction locally.

## Supported Models

Current local chat support:

- `openai/gpt-oss-20b`
- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen3.5-27B`
- `Qwen/Qwen3.5-35B-A3B`
- `zai-org/GLM-4.7-Flash`

Current API chat passthrough support:

- `gpt-5.4`
- `gpt-5.4-mini`
- `gpt-5.4-nano`

Current auxiliary local support:

- embeddings: `Qwen/Qwen3-Embedding-0.6B`
- STT: `mistralai/Voxtral-Mini-4B-Realtime-2602`
- TTS: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- TTS: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`

The support matrix is intentionally selective. CTOX only carries model families that have a concrete local runtime plan and `responses` bridge path in the active code.

On Linux CUDA hosts, CTOX currently assumes:

- detect `cuda flash-attn` features automatically
- add `nccl` when NCCL is present
- add `cudnn` when `libcudnn` is present
- install missing CUDA build prerequisites via `apt` when needed
- install `libnccl2` / `libnccl-dev` on multi-GPU hosts when the runtime is still missing
- keep aux backends and chat visible on the same GPU set by default, but automatically reduce chat layer placement on GPUs that also host embeddings, STT, or TTS
- prefer a live per-GPU reservation map from the running embedding/STT/TTS processes when those sidecars are already loaded, so the first GPU is not overcommitted just because TTS or STT spill asymmetrically across the device set
- fall back from even NCCL tensor parallelism to weighted `--device-layers` when shared aux GPUs require asymmetric layer placement

## Current Scope

The active project still does not claim that every supported model or host combination is fully validated. The current goal is narrower:

- make the local GPU-server install path honest
- keep the `responses` bridge in Rust
- make GPU-host assumptions explicit and reproducible
- keep local skills and local tools easy to carry into vendored Codex
- provide an always-on autonomy layer for server operations
- use legacy code only as a transplant source, not as an active dependency

## Non-Goals

CTOX is not trying to become:

- a private agent for everything
- a lifestyle assistant
- a consumer chat product
- a browser-first general-purpose agent
- a broad multi-platform personal-agent framework
- an IDE helper whose main job is interactive development convenience

If a feature does not make server-side autonomy, operational continuity, or host-side responsibility stronger, it is probably outside the center of gravity for CTOX.

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
