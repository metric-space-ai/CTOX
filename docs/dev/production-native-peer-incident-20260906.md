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

## Forward repair and independent browser check after 13:00 UTC

The corrected native release `branch-main-20260906T120734Z` is active.
The emergency executable override was removed after the forward activation.
A subsequent read-only check observed PID 2460544, running=true,
replicationUp=true, heartbeatFresh=true and errorTotal=0. SKF, Thesen and
Miltonticket also reported running peers with fresh heartbeats and no health
errors; their replicationUp=false without connected browsers is not an outage.

A fresh authenticated Welsch page now identifies shell 0.1.46-beta.9.
The original startup failure did not recur. An existing Word acceptance
document painted its persisted text, and the CTOX task list, flow, progress
and timeline rendered. No new task execution or document-save acceptance
is implied by these read-only checks.

Spreadsheets remains a failing user story: both the beta9 CSV acceptance
record and an existing XLSX display "no saved version". The browser log
instead records repeated WebRTC peer-reopen timeouts for spreadsheet_versions.
Read-only SQLite inspection confirms that the CSV record's current_version_id
points to an existing, non-deleted version. The UI must not turn a transport
failure into an assertion that the user's saved file is missing.

Two additional shell sync defects were reproduced independently:

- The local tab coordinator used APP_BUILD alone. beta8 and beta9 share v340
  despite different canonical DB bundles, allowing a corrected tab to follow
  an older runtime. The coordinator key now includes RXDB_BUNDLE_URL from the
  sole canonical loader, as well as the shell epoch.
- In a follower tab, an ordinary startCollection overwrote an already acquired
  direct bridge with a follower stub. The next forceDirect acquisition could
  replace its still-active registration. Cached live bridges are now reused
  before the follower branch. Actual role transitions still own demotion.

Both regressions failed before the corresponding correction and pass after
it. The acquisition test runs the real multi-tab coordinator and shell runtime
with a simulated native transport; it is not production WebRTC E2E. The
37-app static shell contract and all 37 geometry cases also passed.

The coordinated source build identifier is now
`20260906-shell-v2-sync-bridge-v341` in HTML, APP_BUILD, DB and sync loader
imports. An actual public sync.js response at 13:18 UTC carried
`cache-control: max-age=14400, must-revalidate`; reusing v340 would leave
previously cached code eligible for four hours. This source fix is not an
activation or production-readiness claim.

Browser automation observed a warm Harness switch plus accessibility
inspection in 23,012 ms and a Tables switch plus inspection in 7,951 ms.
These wall-clock observations include automation/inspection overhead and
must not be reported as app interaction latency, command p50 or boot p95.
The performance acceptance remains open.

A separate read-only data audit found three non-deleted document chunk rows
containing omission markers (324,210 / 324,326 / 324,318 original encoded
bytes). Each has a document reference but no live document-version reference.
Their originals were not found in the two examined native record/chunk
stores. This audit checked canonical blob_id references only and was
incomplete: editor_blob_id and staged_editor_blob_id must also be checked.
The later restart investigation below confirms damage to a referenced editor
cache. The earlier lack of canonical references is not evidence of safety.

## beta10 activation and referenced file corruption

Commit 87abdd514 is on main. Its signed shell release 0.1.46-beta.10 passed
GitHub Actions run 34036560364 and was activated on Welsch at 13:38:26 UTC.
The service was restarted by this task (PID 2479926); the browser loaded
v341 and beta10. The native executable stayed branch-main-20260906T120734Z.
No database backup was restored.

The native peer reported running=true, replicationUp=true, heartbeatFresh=true
and errorTotal=0, but real Office workflows still failed. At 13:41:36 UTC the
browser reported ctox_webrtc_incoming_transfer_stalled across collections;
Spreadsheet editor.open subsequently timed out. Word reported an atob error.
These health flags therefore do not establish app readiness.

The restart logged that it trimmed 201 oversized projected documents.
The startup clamp scans all physical Business OS collection tables, including
file chunks. Browser Office writes chunks up to 256,000 bytes; Base64 encoding
can exceed the 262,144-byte projection budget. The generic clamp replaces
data with an omission marker and increments its revision. Direct native
projection writes have the same lossy clamp.

For document doc_0d4889d0-fe96-453f-85b7-9f01a2f62eae, current version
doc_0d4889d0-fe96-453f-85b7-9f01a2f62eae_office_v2_-z9gChy2PR references
editor blob office_document_66f73256-6e61-49eb-bebc-6a7e6a172488.
Its first chunk contains an omission marker for 324,442 encoded bytes.
The canonical DOCX remains intact: 1,163 decoded bytes, matching source_sha256,
valid ZIP CRC, and word/document.xml containing the previously observed
Office-Abnahme text. This verifies this canonical file, not every file or
every formatting property.

Recovery command cmd_incident_20260906_reprepare_doc_0d4889d0 was dispatched
through the daemon-owned office.document.prepare command. It failed with
role_or_scope_denied (data.write, documents), so no regenerated cache is
claimed from that rejected attempt. No direct SQL repair or permission bypass
was performed.

At 14:33 UTC, SQLite backup API snapshots of business-os.sqlite3 (129,073,152
bytes) and business-os-rxdb.sqlite3 (217,681,920 bytes) both passed quick_check:
`/home/ctox/.local/state/ctox-incident-file-recovery-20260906T143338Z/`.
The existing codex-shell-rollout maintenance identity then authenticated via
the supported local issue-capability command, without creating users or
changing roles/grants. The token stayed in the subprocess pipe and the
daemon's normal policy gate accepted the authenticated prepare command.
`cmd_incident_20260906_reprepare_doc_0d4889d0_authorized` completed.
The canonical source hash remained
214c1da6cdbbb67a7d8e6ffb2b8c68b356deb3a26828d934f4f3ee3560aeb811.
The new editor payload passed native protocol validation; reopening the
document in the real beta10 browser painted its saved Office-Abnahme text.
This is a successful read/recovery check, not edit/save or performance acceptance.

The same audit found two more current document editor caches with omission
markers and verified canonical hashes: doc_0d1190f0-d6e1-482a-844c-754331fcf528
and doc_25e4549b-c56a-4660-a829-8dab4805cf76. Their authenticated prepare
commands also completed. Their browser rendering is not yet certified.
Historical staged_editor_blob_id references still retain the damaged artifacts;
regenerating a current editor cache does not recover every historical artifact.
Three older spreadsheets had no canonical chunk rows in the inspected RxDB
table. Other stores/backups must be checked before classifying them as lost.

The pending native correction excludes built-in and runtime-declared
demand-chunks storage collections from both write-time and startup projection
clamping, using the canonical demand-file registry. It does not exempt
desktop_files metadata or weaken unrelated projection budgets. Verification,
native activation, remaining damaged-artifact recovery and full Office E2E
remain open. The local build machine was severely overloaded during verification
(load average 120.01, 0.49% CPU idle, 13 GiB compressed memory at 14:43 UTC);
browser automation wall times on this machine cannot serve as isolated
production-performance acceptance.

Native initial replication measurements after the restart included:
desktop_file_index 39,324 ms, business_records 47,397 ms and knowledge_tables
50,677 ms. SQLite recorded 12,519 statements, 155.417 s cumulative time and
1.028 s maximum; maximum writer-lock wait was 0.955 ms. These are individual
native observations, not browser boot p95 or command p50. Current performance
acceptance is failing/open.
