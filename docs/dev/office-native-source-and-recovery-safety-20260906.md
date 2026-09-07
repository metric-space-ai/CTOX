# Office native source and recovery safety

Target: Welsch. This repair does not constitute the complete Office release gate.

Two production findings were reproduced during Office acceptance:

1. A native Pi turn with a read-only review prompt returned the complete source
   snapshot. The owner applied all files, including an unchanged, stale
   Documents source projection. The active signed beta.13 shell slot was not
   changed, but the native release's Documents source lost newer close-guard
   and coalesced-refresh code. Read-only model intent alone is not write safety.
2. Automatic executable recovery after upgrade activation failure restored a
   database backup taken before the build. The old daemon could have accepted
   newer writes while the candidate compiled. Executable recovery must not
   silently rewind that live state.

The repair filters Pi snapshots against their original input and writes only
changed files. Before any file is applied, its original content is checked
against the current authoritative module source. A conflicting or deleted
source fails with an explicit reload instruction. The writer rechecks each
file before writing. This is optimistic conflict detection, not an atomic
multi-file transaction. Existing policy gates and source-path/symlink checks
remain in place. CLI results now include assistant text, excluding thinking
and tool-result blocks, instead of silently dropping a review's answer.

Automatic release recovery now stops safely, switches to the previous
executable, refreshes its service unit and propagates activation failures.
It has no backup argument and does not restore database snapshots. Explicit
operator-requested rollback/data restoration remains a separate operation.
Backups remain available for an intentional recovery decision.

Regression coverage includes unchanged stale snapshots, stale edits rejected
before the first proposed write, a successful changed-file round trip,
assistant-text filtering, and real SQLite WAL writes accepted after the
pre-build backup surviving executable recovery and database reopening.
Service-stop and activation failures are tested using injected callbacks;
the tests do not stop or restart the operator's real service.

Deployment requirement: the old running updater still contains the unsafe
automatic database restoration. Build and verify the repaired main revision
first, then invoke that repaired updater through the managed update workflow.
Do not start the old updater and assume a newer candidate fixes its recovery
logic. No direct SQL repair or unchecked database rollback is authorized here.

The Documents layout regression also now asserts the actual Shell V2
`--shell-col-left` contract, left resizing, desktop editor width, narrow left
overlay behavior, and strict absence of any right actions column or handle.

Verified on 2026-09-06: the corrected native candidate passed the combined
release-mode regression run (2 recovery, 16 Pi, 1 demand-file preservation;
19 passed, none failed or ignored). The first isolated overlay omitted the
already-landed demand-file writer exemption; its preservation regression
failed and prevented publication. The exemption was restored before this
successful combined run. The strict JS DB suite passed all119 tests with
`--require-wire-daemon` and no skips. Documents source validation passed
55 tests, including the updated layout browser regression. Full deployed
Office, authentication and security acceptance remains a separate open gate.
