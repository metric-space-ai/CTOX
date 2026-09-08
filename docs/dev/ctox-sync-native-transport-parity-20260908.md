# Native transport validation parity — 2026-09-08

Status: dependency correction under validation; no production acceptance or tenant deployment.

Durability: branch `codex/native-transport-parity`, source commit
`a0e8e59d8f3844ec8f3bfe94b974eec1b8b5093a`, pushed and verified in draft
[PR #69](https://github.com/metric-space-ai/ctox/pull/69).
The existing durable checkout remains in use; no worktree or build cleanup has
been performed while validation is running.

## Finding

At main `8e6f132a82549be9235dd433fc9bd28d5ae868e6`, the three independent Cargo
roots resolved different transports:

| Cargo root | WebRTC/RTC | ICE implementation |
| --- | --- | --- |
| CTOX binary | 0.20.0-alpha.1 | repository patch |
| standalone RxDB tests | 0.20.0-alpha.1 | crates.io, without repository patch |
| standalone Sync tests | 0.20.5 | crates.io |

The RxDB manifest used compatible version requirements. Cargo also ignores
patch declarations in dependencies: the root's ICE patch was absent in both
standalone test manifests. Therefore the historical 74/74 Sync result does not
prove the shipped native transport works. The four-process CLI acceptance
remains red after signaling; parity alone does not establish its root cause.

## Correction

Pin the two direct transport dependencies exactly to the existing production
version and declare the same local ICE patch in each standalone Cargo root.
Seed the standalone lockfiles from the production lockfile and let Cargo
resolve/prune their dependency closures, retaining production versions where
possible. This also avoids a mixture of alpha and stable transitive RTC crates.
The production root lockfile and transport implementation are unchanged.

`native-transport-parity-smoke.mjs`, discovered by the browser runtime suite,
compares versions, registry provenance/checksums and local ICE patch paths for
all 17 WebRTC/RTC packages across the three locks. It also requires exact direct
transport pins. This is a build-input guard, not a networking or feature-parity
acceptance test.

## Evidence so far

- Before correction, the guard exits 1: standalone RxDB resolves registry ICE
  instead of the production local patch.
- After correction, it exits 0 for all 17 packages.
- Cargo metadata resolves offline for aarch64-apple-darwin with the native
  WebRTC feature enabled. Unfiltered offline metadata first failed because an
  uncached Android-only package was requested; no download was attempted.
- The production-graph Sync suite is being rebuilt and rerun. Its result and
  subsequent real-process diagnosis must be recorded before acceptance.

No customer data, runtime configuration, shell slot or tenant release was changed.

## Continuous verification

The existing workflows did not invoke the standalone Sync crate. Add a focused
`Native Sync` workflow on Linux and macOS: compare the 17 transport packages,
check formatting, run the full Sync suite with WebRTC, run native RxDB with the
production ICE patch, and lint all Sync targets. It supplements the existing
app/CLI gates; it does not replace the separate four-process host acceptance.
All runs use locked dependencies and keep the existing test deadlines.

For PR #69 at `3432b3488`, the existing Desktop macOS job stops at npm audit
(`@xmldom/xmldom`, `fast-uri`). The x86_64 Linux CLI job stops before compilation
at the platform-freeze guard (`modules/explorer/index.js` contextmenu handler).
Job IDs: `101935097745`, `101935097780`, run `34186266711`. These unchanged app
paths are outside this dependency correction; no guard is bypassed.
