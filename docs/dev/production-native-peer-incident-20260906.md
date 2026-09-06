# Production incident: native peer recovery, 2026-09-06

Status at 11:14 UTC: partial recovery; not a production-readiness acceptance.

## Routing mitigation already active

The production alias `ctox.dev` and wildcard `*.ctox.dev` resolve to Vercel
deployment `dpl_7Cvc8DeqVjkokApXfEYTzbkjpfP2`
(`ctox-nlzd986z9-metric-spaces-projects.vercel.app`, READY).
The premature strict generation-conflict rollout remains reverted by ctox.dev
main commit `63eb4444577f88842011033c1c66924a849a3dd6`.
Do not remove the compatibility fallback again before the complete native and
client dependency-generation contract passes real upgrade/browser tests.

## A subsequent independent native upgrade failed after reporting success

Welsch's pre-existing systemd job
`ctox-office-native-upgrade-20260906.service` began at 10:29:22 UTC.
It compiled and activated `branch-main-20260906T102922Z`, then reported
`updated: true` and completion at 10:57:00 UTC after 27m37s.

At 10:57:03 the service panicked inside `schema_from_json`:

> invalid type: sequence, expected a string

The panic was at `src/core/business_os/rxdb_peer.rs:8933`.
The process and systemd unit remained alive, while its native RxDB peer was
dead: running=false, replicationUp=false, heartbeatFresh=false, errorTotal=1.
The browser consequently retained maintenance write protection. This disproves
the updater's process-running-only success criterion.

The relevant Rust fragment models JsonSchema.type as Option<String>, while
JSON Schema permits type unions represented as arrays. The failing packaged
field was subsequently isolated as `ctox_crew_members.active_task_id`, whose
contract is `["string", "null"]`. An independent regression using every packaged
Business OS schema reproduced the panic before the fix. Do not normalize away
the union or weaken validation merely to suppress the panic.

Main commit `01fe390c7` adds native union representation and validation. Its
crate-level success did not establish daemon compatibility: Welsch's subsequent
`ctox-office-native-union-upgrade-20260906.service` build failed with three E0599
errors at the remaining `Option<JsonSchemaType>::as_deref` consumers in the
Business OS projection layer. The build failure prevented activation of that
candidate. Its recovery script restored the prior service executable without
restoring database snapshots. Projection normalization and repair/tombstone
defaults must explicitly handle unions, and the entire CTOX binary must pass
compilation before another activation. Native-crate tests alone miss this boundary.

## Data-preserving emergency service fallback

The standard `ctox update rollback` was deliberately not run: its implementation
restores the complete update backup, which would rewind data to before the
27-minute build.

Instead, only Welsch's service executable was temporarily switched back to:
`/home/ctox/.local/lib/ctox/releases/branch-main-20260905T193754Z/bin/ctox-real`.

The explicit systemd drop-in is:
`/home/ctox/.config/systemd/user/ctox.service.d/90-incident-native-recovery.conf`.
It clears ExecStart and starts that binary with `service --foreground`.
The current symlink, runtime data, active shell slot, and install manifest were
not rewound. The manifest therefore names the newly installed release while
the service executable is temporarily older. This is an incident mitigation,
not the final runtime compatibility contract. Preserve the previous release
until this drop-in is removed after a verified forward repair.

Backup location on Welsch:
`/home/ctox/.local/state/ctox-incident-20260906T1105/`.

- `current-state.tar`: state archive excluding the existing backups directory
  and sockets. First attempt returned tar exit 1 because SQLite sidecars
  disappeared and the root directory changed. It must not be represented as a
  self-contained valid SQLite snapshot.
- `sqlite-snapshots/`: 42 databases copied with SQLite's backup API, each
  passing PRAGMA quick_check; manifest lists the databases.
- Restore planning must use those SQLite snapshots in place of database files
  and WAL/SHM sidecars from the tar. No restoration was performed.
- `current-state.tar.sha256` and `ctox.service.before.txt` retain archive
  integrity and the original service definition.

The original service was restarted automatically when the first backup attempt
failed. Only after the database snapshots passed did the executable change.
Service start: 11:06:58 UTC, PID 2429984.
Native replication startup log: 11:07:05 UTC, 198 collections registered.
Subsequent projected health: running=true, replicationUp=true,
heartbeatFresh=true, errorTotal=0.

The temporary fallback also logged a schema-hash conflict for
`workjet_computers`. It is not acceptable as a permanent mixed-version runtime.

## Browser and fleet observations

All actions used existing operator access; no other Codex task was messaged.

- Welsch fresh page: shell 0.1.46-beta.8, signed document source
  87daa431b1604bfcca364a1eb1a851e90da1874d. Desktop boots. After peer recovery
  the maintenance banner disappeared and the CTOX harness container appears.
  A later screenshot also verified the painted Harness flow, task list,
  progress and timeline. This does not certify execution of those tasks.
- Documents still reported document_versions peer-reopen timeout.
- Spreadsheets still reported Office RPC editor.open timeout.
- SKF: public page reaches the desktop and research data; native replication
  becomes active while its browser is connected. Shell is explicitly Recovery.
- Thesen: public page reaches the desktop and app surfaces; shell is Recovery.
- Miltonticket: public route reaches its login page. No credentials were
  entered; authenticated E2E was not performed.
- All four native peers had fresh heartbeats and zero health errors in the
  final fleet observation. A disconnected browser can leave replicationUp
  false; this flag is not equivalent to a dead daemon.

## Measurement limits and next acceptance

The native peer registered replication 7 seconds after service start.
This is not a critical-collection browser-boot p95, warm command p50, or a
full user-workflow latency measurement. No performance acceptance is claimed.
No production-readiness claim follows from these incident checks.

Required forward repair: preserve type-union schema semantics, certify every
actual schema through the native parser and real persisted collections,
correct the updater to reject a dead mandatory peer, resolve the
workjet_computers schema conflict through the approved migration path, remove
the temporary service override, and test actual Documents/Spreadsheets/Harness
flows with persistence and timings. The original shell-generation work remains
unfinished and must not be deployed during this recovery.

## Follow-up at 12:18 UTC

The daemon union integration is on main in `3fffd9f0a`; merge `3a4d45bf7`
preserves the independent full-property-contract regression and additional
projection coverage. The unfinished shell-generation draft is not included.

The first union forward build failed on three remaining daemon consumers.
The subsequent `ctox-office-native-projection-upgrade-20260906.service` is
still running (PID 2451497). Welsch remains on the recovered service
(PID 2449549); its peer reports running=true, replicationUp=true,
heartbeatFresh=true and errorTotal=0. This is not evidence that the corrected
release has been activated. Browser update maintenance still blocks app writes
during the build. Harness task list, painted flow, progress and timeline were
verified again; Documents and Spreadsheets are not accepted.

All four public entrypoints returned HTTP 200 in three requests each.
[Raw entrypoint timings](beweise/raw/production-entrypoint-timings-20260906.json)
record TTFB from 0.266132 to 2.943577 seconds. These unauthenticated HTTP
measurements exclude WebRTC bootstrap, critical-collection readiness and
authenticated command/app latency. They are not a p50/p95 acceptance.

Local full-daemon checking, the expanded native schema regression and the JS
suite were still running at this checkpoint. No new pass result or production
readiness is claimed from these pending commands.
