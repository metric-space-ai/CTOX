# CTOX

CTOX is an agentic daemon for long-running technical work.

It is built for people who already use coding agents and notice where a single
agent session stops being enough: real work spreads across tickets, servers,
communication, approvals, waiting states, failed checks, follow-ups, and context
that must still be correct tomorrow.

CTOX does not try to be "another better coding agent". It owns the layer around
agent execution: durable work state, context assembly, queue and ticket
management, verification, process evidence, communication, and continuation.

## Links

- Project page: <https://metric-space-ai.github.io/ctox/>
- Technical documentation: <https://metric-space-ai.github.io/ctox/docs.html>
- CLI reference: <https://metric-space-ai.github.io/ctox/cli.html>
- Releases and binaries: <https://github.com/metric-space-ai/ctox/releases>

## Quick Introduction

Coding agents are good at bounded sessions. CTOX is for the work around those
sessions.

A CTOX instance runs on a workstation, server, or remote host. You give it work
through the TUI, `ctox chat`, mail, tickets, schedules, or other configured
channels. The daemon records that work in durable state, builds the next worker
context from the runtime database, lets an agent perform a bounded run, records
the result, and decides whether the work is done, blocked, waiting, scheduled,
or needs another continuation.

The important unit is not a chat transcript. The important unit is the runtime
state:

- current work, queue items, plans, schedules, and follow-ups
- tickets, cases, approvals, writebacks, and audit history
- Focus, Anchors, Narrative, knowledge, claims, and recent communication
- verification records and process-mining events
- a core state model with transition checks and process evidence

That is the practical difference: CTOX uses agents, but CTOX itself is the
daemon that keeps technical work organized over time.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/metric-space-ai/ctox/main/install.sh | bash
```

The installer creates a managed layout by default:

- install root: `~/.local/lib/ctox`
- state root: `~/.local/state/ctox`
- cache root: `~/.cache/ctox`
- binary symlink directory: `~/.local/bin`

Installer flags:

```sh
curl -fsSL https://raw.githubusercontent.com/metric-space-ai/ctox/main/install.sh \
  | bash -s -- --backend=metal --model=google/gemma-4-E4B-it
```

Common options:

| Flag | Environment variable | Purpose |
| --- | --- | --- |
| `--backend=<cuda\|metal\|cpu>` | `CTOX_BACKEND` | Select local inference backend. |
| `--model=<model>` | `CTOX_MODEL` | Seed the default local model profile. |
| `--install-root=<path>` | `CTOX_INSTALL_ROOT` | Override install root. |
| `--state-root=<path>` | `CTOX_STATE_ROOT` | Override runtime state root. |
| `--cache-root=<path>` | `CTOX_CACHE_ROOT` | Override cache root. |
| `--bin-dir=<path>` | `CTOX_BIN_DIR` | Override binary symlink directory. |
| `--repo=<url>` | `CTOX_REPO` | Install from a custom repository. |
| `--branch=<branch>` | `CTOX_BRANCH` | Install from a custom branch. |

## First Run

```sh
ctox doctor
ctox
ctox start
ctox status
ctox chat "Take ownership of OPS-204. Wait for approval, deploy when allowed, verify the gates, and update the ticket."
```

What these commands do:

- `ctox doctor` checks the installation and runtime environment.
- `ctox` opens the TUI for configuration and operation.
- `ctox start` starts the persistent daemon.
- `ctox status` shows the current service state.
- `ctox chat <instruction>` submits work to the daemon.

Most users should start in the TUI, configure the model backend and credentials
there, then submit work through the TUI or `ctox chat`.

## How CTOX Runs Work

The daemon loop is roughly:

```text
intake
  -> durable queue item, ticket case, schedule, or plan step
  -> leased worker run
  -> context build from runtime state
  -> bounded agent execution
  -> verification, writeback, knowledge, and process events
  -> complete, blocked, waiting, scheduled, requeued, or continued
```

CTOX workers can call CTOX commands themselves. This is intentional: internal
tools such as `ctox ticket`, `ctox queue`, `ctox verification`, and
`ctox process-mining` make the daemon inspect and update its own runtime state
through an auditable command surface instead of relying only on prompt memory.

The command surface is documented in the
[CLI reference](https://metric-space-ai.github.io/ctox/cli.html). Many commands
are daemon tools first and human commands second; normal operation should happen
through the TUI, `ctox chat`, configured channels, tickets, and schedules.

## Model Backends

CTOX can run with API-backed models or with the integrated local inference path,
depending on the configured runtime.

Typical configuration is done in the TUI. Important settings include:

- `CTOX_CHAT_SOURCE=api|local`
- `CTOX_API_PROVIDER=openai|anthropic|openrouter|minimax`
- provider API keys such as `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
- `CTOX_LOCAL_RUNTIME`
- `CTOX_CHAT_MODEL`
- `CTOX_CHAT_MODEL_MAX_CONTEXT`
- `CTOX_AUTONOMY_LEVEL`

See the technical documentation for the current model/runtime details:
<https://metric-space-ai.github.io/ctox/docs.html#configuration>

## Desktop App

The CTOX Desktop app is an optional management surface. It is useful when you
want to install CTOX locally, connect to remote CTOX instances, or manage
multiple instances from one place.

The core runtime is still the daemon. The Desktop app manages instances; the
daemon owns the work.

## Update

```sh
ctox update status
ctox update check
ctox upgrade --stable
ctox update apply --version <tag>
ctox update rollback
```

## Repository Layout

- `src/` - CTOX daemon, runtime, mission systems, TUI, model control, and tools.
- `src/harness/` - integrated in-process agent harness.
- `src/inference/` - local inference work.
- `skills/` - system skills used by CTOX workers.
- `tools/` - supporting tool packages.
- `site/` - GitHub Pages project site and documentation.
- `.github/workflows/` - CI, release, and Pages workflows.

## Development

```sh
cargo fmt --check
cargo check
cargo test
```

The repository contains platform-specific code paths for macOS, Linux, Windows,
and optional local inference backends. Use the release workflow for production
binaries.

## License

[Apache License 2.0](LICENSE)

See [NOTICE](NOTICE) for attribution of integrated source trees.
