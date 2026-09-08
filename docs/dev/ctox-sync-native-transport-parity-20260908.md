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
- Browser-runtime suite: 120 passed, zero failed, zero skipped. This includes
  the new parity guard. The existing wire-daemon fixture was available; this
  run is not a newly built full CTOX host or product-performance acceptance.
- Local production-graph build completed in 50m04s under heavy machine load.
  Unit tests: 19 passed. Authority cluster: 1 passed, 10 failed, mostly on
  confirmation deadlines; recorded RPC delays reached over one second. This is
  a red result, not a production-performance acceptance. Cargo stopped before
  the WebRTC test executable.
- Clean Linux/macOS CI run 34186877185 compiled the production graph. On each
  platform 19 unit tests and 10 of 11 authority-cluster tests passed. The actual
  Workjet IPC consumer test failed because its required sibling checkout was
  absent. No WebRTC result can be inferred from these early exits.

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

The workflow now prepares both repositories as siblings, including the actual
Workjet consumer at immutable revision `ca7dd885f3615ff31b18eb6ffb6c7c58c45ceaf3`.
Node 24.13.1 and pnpm 11.10.0 match that revision's declared engines and package
manager. Dependencies are installed with the frozen Workjet lockfile and no
lifecycle scripts. The existing IPC assertions remain unchanged. Cargo uses
`--no-fail-fast` to collect all test-executable failures instead of hiding the
WebRTC result behind the first failing test executable.

## Obsolete remote compiler output removed

On 2026-09-08 at 04:41 UTC, removed only the old isolated test directory
`/home/ctox/.cache/ctox/file-preservation-fbeca32b7/cargo-target` on Welsch.
Allocated size: 28,723,452 KiB. Free filesystem space afterwards: 37,873,111,040
bytes. A privileged /proc audit immediately before removal found no compiler,
open file, working directory, executable or mapped-file reference to that
exact target; no process was unreadable in that audit. The earlier unprivileged
audit was insufficient and did not authorize deletion.

All source snapshots, fixture databases, reports and release-target outside
cargo-target were retained. The removed build was disposable and its correction
and evidence are already in main via PR #65. No service was stopped, restarted
or upgraded; the Office worker owns the Welsch cutover. The remote receipt is
`/home/ctox/.cache/ctox/file-preservation-fbeca32b7/cleanup-native-parity-20260908.json`.
paths are outside this dependency correction; no guard is bypassed.
