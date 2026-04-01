# CTOX

CTOX brings autonomy to servers.

CTOX is an AI agent system for autonomous work on hosts and services. It is built for long-running build, operations, and infrastructure missions.

Install CTOX on a server and it acts as a persistent technical control layer around that host: planning work, continuing interrupted tasks, managing communication, supervising context, and driving execution until the mission is actually closed.


## What CTOX Is

CTOX combines four layers:

- a CTOX orchestration layer optimized for autonomous server and DevOps work
- Codex as the current execution engine
- a CTOX proxy server endpoint for OpenAI-compatible model access and sidecar routing
- a standalone local inference engine for on-host AI models


## The CTOX Orchestration Layer

CTOX adds system behavior that standalone execution CLIs do not provide:

- persistent service and control-plane behavior
- external communication routing across TUI, email, Jami, cron, and queue-backed work
- durable queue, plan, schedule, and follow-up execution
- long-run mission state with explicit blocker, next slice, and done gate
- long-context memory with continuity tracking, memory retrieval, long-run context optimization and self heal mechanisms
- mission watchdogs, timeout continuations, queue-pressure guards, and other background repair mechanisms
- completion review, verification runs, persistent claims, and mission assurance
- native integration for LLM, embeddings, STT, TTS
- a `WebSearch`, `WebRead`, `WebScrape`, and browser-automation stack
- built-in DevOps skills and tools
- built-in email and chat-app integration


## Execution Engine

CTOX currently uses Codex as its execution engine.

That means CTOX delegates the bounded execution slice to `codex-exec`, but wraps that slice in:

- persistent mission context
- durable routing and scheduling
- review and assurance
- model and backend control
- host-side operational policy

This is the intended split between orchestration and execution:

- CTOX owns persistence, orchestration, governance, communication, verification, and runtime control
- Codex owns execution semantics inside the bounded agent run

## Proxy Server

The CTOX proxy serves local models for LLM, embeddings, STT, and TTS workloads. It converts model-specific request formats and chat templates into an OpenAI-compatible `responses` surface.

It provides:

- one OpenAI-style endpoint surface on `CTOX_PROXY_PORT`
- `POST /v1/responses` as the primary chat path
- request rewriting from `responses` into backend-specific forms for GPT-OSS, Qwen, Nemotron, GLM, and the local engine bridge
- routing for `POST /v1/embeddings`, `POST /v1/audio/transcriptions`, `POST /v1/audio/speech`, and `POST /v1/audio/voices`
- runtime telemetry on `GET /ctox/telemetry`
- live model switching on `POST /ctox/switch`
- backend readiness checks, startup, and recovery behavior
- optional web-search augmentation on the `responses` path

The proxy lets clients talk to one stable local API while CTOX handles model-family differences and host-side runtime behavior.

## Web Capability Model

CTOX uses four distinct web paths:

- `WebSearch` for current discovery and recent-information lookup
- `WebRead` for reading concrete sources well
- `interactive-browser` for real browser interaction when the page behavior itself matters
- `WebScrape` for durable, repeatable extraction

## Dependencies

CTOX currently depends on two major local or vendored codebases:

- `references/openai-codex/codex-rs`
- `engine/candle/`

Those are important implementation dependencies, but the persistent orchestration system, service loop, queueing, continuity, governance, verification, and runtime mediation live in CTOX itself.


## Quick Start

Install directly on a server with the one-liner:

```sh
bash -lc "$(curl -fsSL https://raw.githubusercontent.com/metric-space-ai/CTOX/main/scripts/install_ctox_remote.sh)"
```

Local install from a checked-out repository:

```sh
./scripts/install_ctox.sh
```

Start the persistent loop:

```sh
ctox start
ctox status
```

Open the visual command-line interface:

```sh
ctox
```

Stop the persistent loop:

```sh
ctox stop
```

The full CLI reference lives in the docs, not in this README.

## Documentation

- [Docs Index](docs/index.md)
- [Architecture](docs/architecture.md)
- [CLI Reference](docs/cli.md)
- [Web Paths](docs/web-paths.md)
- [Clean-Room Baseline](docs/clean-room-baseline.md)

## Project Site

This repository can publish `docs/` as a GitHub Pages project site.

The basic shape is already compatible with that setup:

- `README.md` as the repo landing page
- `docs/index.md` as the docs landing page
- topic pages under `docs/`

The remaining step is repository-side Pages activation in GitHub settings.
