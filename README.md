# CTOX

CTOX brings autonomy to servers.

CTOX is deliberately not a private agent for everything. It is not primarily a lifestyle assistant, not a general-purpose personal chatbot, and not just another development helper. Its center of gravity is operations: persistent execution, follow-through, checks, queues, scheduling, communication, and responsibility for servers and services.

At its core, CTOX combines:

- a Codex-based execution engine used as one component inside a larger control plane
- a local inference gateway in Rust
- a local Candle-based inference engine in `engine/candle/`
- a focused operations layer for autonomy, continuity, scheduling, communication, and host-side control

## If You Already Know Codex

If you already know standalone Codex, the important distinction is this:

- Codex in CTOX is the execution engine, not the whole product
- CTOX adds the persistent control plane around that engine
- most of the product-specific behavior lives in CTOX, not in the vendored Codex tree

The main CTOX-specific layers on top of Codex are:

- a long-running service loop instead of one prompt at a time
- durable queue, plan, schedule, and follow-up orchestration
- channel routing for inbound and outbound work across TUI, email, Jami, cron, and queue tasks
- long-context memory with continuity documents, compaction, and mission state
- mission watchdogs, queue-pressure guards, timeout continuations, and other background repair mechanisms
- completion review, verification runs, and persistent mission claims
- a local model gateway that normalizes different backends behind one stable API surface
- explicit web-path routing across `WebSearch`, `WebRead`, `interactive-browser`, and `WebScrape`

This matters for feature placement:

- if a feature changes persistence, routing, scheduling, continuity, governance, review, communication, proxying, or web-path policy, it belongs in CTOX
- if a feature changes the internals of `codex-exec` itself, such as tool protocol, event streaming, agent runtime semantics, or patch execution behavior, it belongs in Codex

CTOX started as a wrapper around Codex, but the current architecture is already materially broader than that description.

## Operating Model

CTOX is designed to run as a persistent local control plane on your servers.

In the current implementation, that means:

- a long-running CTOX service loop that can stay active in the background
- a local OpenAI-compatible `responses` gateway for Codex
- persistent local sidecars for chat, embeddings, STT, and TTS
- an explicit routing substrate for queue items, channels, plans, schedules, and follow-up work
- a local continuity and compaction layer so long-running work can keep moving without turning into an unbounded chat transcript

CTOX is meant to help operate servers and services continuously, not just answer one prompt at a time.

## Proxy Server

The CTOX proxy is not just a thin pass-through. It is the compatibility and runtime mediation layer that lets one stable client surface drive multiple model families and sidecars.

Today the proxy provides:

- one stable OpenAI-style front door on `CTOX_PROXY_PORT`
- `POST /v1/responses` as the main chat entrypoint for local and API-backed models
- automatic request rewriting from `responses` into the narrower backend-specific forms required by GPT-OSS, Qwen, Nemotron, GLM, and the local engine bridge
- routing for auxiliary model traffic to `POST /v1/embeddings`, `POST /v1/audio/transcriptions`, `POST /v1/audio/speech`, and `POST /v1/audio/voices`
- runtime telemetry on `GET /ctox/telemetry`
- live model switching on `POST /ctox/switch`
- boost-lease support so heavier temporary models can be activated without permanently changing the base runtime policy
- automatic backend readiness checks and startup/recovery logic for chat and auxiliary sidecars
- optional web-search augmentation on the `responses` path so the model-facing request stays compact while the proxy injects reviewed retrieval results

In practice, this means Codex and other clients can talk to one local endpoint while CTOX handles model-family differences, sidecar routing, operational telemetry, and host-specific recovery behavior.

## Install

Local install:

```sh
./scripts/install_ctox.sh
```

One-liner remote install:

```sh
bash -lc "$(curl -fsSL https://raw.githubusercontent.com/metric-space-ai/CTOX/main/scripts/install_ctox_remote.sh)"
```

By default the remote one-liner now wipes an existing `~/ctox` install before cloning fresh again, so rerunning the same command behaves like a clean reinstall instead of layering onto an old checkout. Set `CTOX_REMOTE_WIPE_EXISTING=0` only if you explicitly want an in-place refresh.

Default target directory:

```text
~/ctox
```

The installer currently does these baseline steps:

- builds `ctox`
- uses the project-bundled `openai-codex` sources under `references/` and the CTOX Candle engine tree under `engine/candle/`
- syncs repo-managed system skills into the vendored `codex` build before compilation
- builds the local `ctox-engine` entrypoint and vendored `codex-exec`
- writes a minimal runtime file at `runtime/engine.env`
- installs bundled skills into `CODEX_HOME/skills` and `CODEX_HOME/skills/.system`
- on Linux, installs and enables a persistent `systemd --user` `ctox.service` background loop
- on supported Linux distributions, installs the Jami daemon runtime when needed
- on repeat installs, stops existing CTOX user services, wipes prior runtime state, refreshes repo-managed skills, and recreates the browser reference workspace

Short-term vendor safety:

- `openai-codex` is currently pinned to the exact upstream commit `c6ab4ee537e5b118a20e9e0d3e0c0023cae2d982`
- install expects `references/openai-codex` and `engine/candle` to already exist in the project checkout
- `ctox clean-room-bootstrap-deps` now refuses to update a dirty checkout, so local CTOX customizations in `references/openai-codex` or `engine/candle` are not silently overwritten

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

Rewrite a captured Codex `responses` payload into the narrower `ctox-engine`-compatible form:

```sh
ctox clean-room-rewrite-responses /path/to/request.json
```

Apply local chat runtime presets:

```sh
ctox chat-runtime-apply openai/gpt-oss-20b quality
ctox chat-runtime-apply Qwen/Qwen3.5-35B-A3B performance
ctox chat-runtime-apply nvidia/Nemotron-Cascade-2-30B-A3B quality
```

Run the local backend through the active CTOX launcher:

```sh
./scripts/engine/run_engine.sh
```

The launcher reads `runtime/engine.env` and applies the active CUDA/NCCL runtime environment before starting the local `ctox-engine` process.

If no explicit GPU override is set, the launcher uses all visible NVIDIA GPUs by default and derives the local NCCL world size from them for tensor-parallel families.

If `CTOX_AUXILIARY_CUDA_VISIBLE_DEVICES` is set, the persistent embedding, STT, and TTS sidecars bind to those GPUs and the main chat backend prefers the remaining visible devices.

The persistent CTOX service keeps these local HTTP sidecars warm in parallel:

- chat backend on `CTOX_ENGINE_PORT`
- responses gateway on `CTOX_PROXY_PORT`
- embeddings on `CTOX_EMBEDDING_PORT`
- STT on `CTOX_STT_PORT`
- TTS on `CTOX_TTS_PORT`

Auxiliary CPU fallbacks:

- embeddings can now stay on `Qwen/Qwen3-Embedding-0.6B` but run in a CPU-only mode
- STT can switch from Voxtral on GPU to a `faster-whisper` CPU path
- TTS now standardizes its CPU fallback on Speaches-backed Piper voices, so the gateway and Rust control plane only need one CPU speech-serving path while still covering languages such as `de_DE`, `fr_FR`, and `en_US`

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
- `universal-scraping`
- `interactive-browser`

These skills sit on top of concrete host and service tooling such as `htop`, `btop`, `top`, `ps`, `vmstat`, `iostat`, `ss`, `journalctl`, `systemctl`, `df`, `du`, `curl`, and `nvidia-smi`, plus the existing CTOX `queue`, `plan`, `schedule`, and `follow-up` substrate.

They do not introduce a second execution loop. They are additional skill-level operating knowledge that rides the existing CTOX routing and Codex execution path.

`universal-scraping` uses the same explicit CTOX model: repeat scrape work should be registered, versioned, and scheduled through runtime state and `ctox schedule`, not left as one-off ad hoc scripts in chat history.

CTOX now also carries `interactive-browser` as the reviewed real-browser path for work that truly needs live DOM state, client-side JavaScript execution, auth/session behavior, or screenshots through `js_repl` plus Playwright.

## Web Capability Paths

CTOX deliberately separates four web paths instead of flattening everything into one "browser" concept:

- `WebSearch`
  - current discovery and recent-information lookup
- `WebRead`
  - concrete source reading through the local source-reading runtime, including `open_page`, `find_in_page`, PDF evidence, and source-specific readers such as GitHub/docs/news adapters
- `interactive-browser`
  - real browser interaction when JavaScript, session state, or visual evidence is the source of truth
- `WebScrape`
  - durable repeated extraction through `ctox scrape`

`WebScrape` is not just an ad hoc browser helper. It is an operational path with a concrete service surface. Each target can expose four stable HTTP read paths from the CTOX service:

- `/ctox/scrape/targets/{target_key}/api`
- `/ctox/scrape/targets/{target_key}/records`
- `/ctox/scrape/targets/{target_key}/semantic`
- `/ctox/scrape/targets/{target_key}/latest`

The routing rule is simple:

- use `WebSearch` when the task is "find out what is true now"
- use `WebRead` when the task is "read this concrete source well"
- use `interactive-browser` when the task is "the page behavior itself matters"
- use `WebScrape` when the task is "make this extraction repeatable and operational"

Interactive browser work is explicitly not a license to make CTOX a browser-first general-purpose agent. It is a reviewed specialist path for the cases where the cheaper web paths are not enough.

Prepare the local Playwright reference workspace with:

```sh
ctox browser install-reference
ctox browser doctor
```

If the host still needs a browser binary for Playwright:

```sh
ctox browser install-reference --install-browser
```

CTOX chat sessions now pass `features.js_repl=true` into Codex and automatically add `runtime/browser/interactive-reference/node_modules` to the `js_repl` module search path when that reference workspace exists, so the interactive browser path can be used without ad hoc per-turn setup.

Example native scrape flow:

```sh
ctox scrape init
ctox scrape upsert-target --input /path/to/target.json
ctox scrape register-script --target-key acme-jobs --script-file /path/to/extractor.js --change-reason initial_import
ctox scrape register-source-module --target-key acme-jobs --source-key board-a --module-file /path/to/board-a.js --change-reason initial_source_import
ctox scrape execute --target-key acme-jobs --allow-heal
ctox scrape show-latest --target-key acme-jobs
ctox scrape show-api --target-key acme-jobs
ctox scrape query-records --target-key acme-jobs --where classification.category=job --limit 20
ctox scrape semantic-search --target-key acme-jobs --query "remote rust jobs"
```

Targets can now define first-class `config.sources[]` entries so one scraper API can aggregate multiple websites, feeds, or endpoints behind one `target_key`. Each source is materialized under `runtime/scraping/targets/<target_key>/sources/<source_key>/`, can carry its own extraction module, and those modules can be revisioned separately with `ctox scrape register-source-module`.

`ctox scrape execute` records runs in `runtime/ctox_scraping.db`, stores concrete run artifacts under `runtime/scraping/targets/<target_key>/runs/`, applies the target-local `api/llm_enrichment_template.json` when enabled, materializes the enriched canonical record set under `runtime/scraping/targets/<target_key>/state/`, writes a default target API scaffold under `runtime/scraping/targets/<target_key>/api/`, includes configured source and source-module context in the run manifest, and only creates automatic CTOX repair follow-up when the failure looks like real portal drift or partial extraction instead of temporary downtime.

The default scrape API then exposes:

- exact-match record filters over the canonical latest state
- semantic retrieval backed by the configured embedding service
- editable per-target enrichment templates for classifications, structured extraction, and summaries
- one shared API surface for multi-source aggregation instead of one endpoint per ad hoc script

Edited semantic and enrichment templates are preserved across future target/run updates; CTOX only creates the default template files when they do not already exist.

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

## Code Footprint

For the current first-party Rust runtime under `src/**/*.rs`, a simple non-blank, non-`//` LOC count shows that CTOX is no longer just a light wrapper around Codex.

Strict Codex-facing count:

- `src/inference/chat.rs` plus `src/browser.rs`: `1,482` LOC
- this is about `3.0%` of the active first-party Rust runtime
- the remaining CTOX-native runtime logic is `47,874` LOC, or about `97.0%`

Broad Codex-facing count:

- if `src/inference/engine.rs` is also counted as part of the Codex-facing integration layer, the Codex-facing total becomes `5,892` LOC
- that is still only about `11.94%` of the active first-party Rust runtime
- the remaining CTOX-native runtime logic is `43,464` LOC, or about `88.06%`

Method and boundary:

- the strict count treats only the direct `codex-exec` invocation path and Codex browser-config glue as the Codex-facing layer
- the broad count also includes the larger runtime-planning and vendoring helper layer in `src/inference/engine.rs`
- these percentages are for CTOX's active first-party Rust runtime, not for the vendored upstream dependency trees under `references/openai-codex/`

Vendored dependency context:

- pinned upstream `references/openai-codex/codex-rs` currently contributes about `596,559` Rust LOC
- local `engine/candle/` currently contributes about `252,480` Rust LOC

Those vendored trees are important dependencies, but they are not the same thing as CTOX's own orchestration logic. The active product behavior described above lives primarily in CTOX's own runtime and control-plane code.

## Supported Models

Current local chat support:

- `openai/gpt-oss-20b`
- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen3.5-27B`
- `Qwen/Qwen3.5-35B-A3B`
- `nvidia/Nemotron-Cascade-2-30B-A3B`
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
cargo test engine
```

These tests currently cover:

- dependency discovery
- `ctox-engine` startup plans for `GPT-OSS` and `Qwen3.5`
- `codex-exec` baseline planning for the `responses` path
- Rust-side rewrite of Codex `responses` payloads into the shape currently needed for `ctox-engine`
