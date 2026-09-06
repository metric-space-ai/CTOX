# Spreadsheet version recovery — 2026-09-07

Target: Welsch only, https://welsch.ctox.dev/.
This is a repair checkpoint, **not a production-readiness approval**.

## Finding and bounded repair

The deployed main490/beta13 loader collapses WebRTC peer-reopen errors into
`null`. The editor then claims that no saved version exists, while the header
claims Saved. Exact native metadata and the canonical XLSX were present.

The repaired loader recognizes the observed peer-reopen timeout, retains
terminal permission/schema/integrity failures, and permits at most two
recovery rounds. Failed bridge bring-up consumes that same budget. Existing
4500 ms local-read and 60000 ms recovered-read limits are retained.
Supersession/disposal checks surround asynchronous boundaries; concurrent
editing, saving, selection changes and newer versions remain protected.

Per-selection states distinguish loading, error, missing and ready. Failed
reads show the actual escaped error and a localized Retry button, never a
missing-version message or Saved badge. Retry is explicit after exhaustion;
background refresh does not turn a terminal failure into an endless loop.
No data paths, permission grants, schema changes or blank replacements were
added. Current version pointers are not rewritten to fallback metadata.

## Evidence

- Native embedded Pi coding review completed, one gateway request HTTP200,
  `ok: true`, `applied_files: []`, message_count2. Proposal retained at
  ctox-dev/output/welsch-office-pi-proposal-1788731666165.json. It used supplied
  authorized repository excerpts because the native source projection was stale.
- Earlier normal `ctox.source.load` commands for both apps were denied with
  `apps.source.view` / `role_or_scope_denied`. Those denials were preserved;
  no source projection writes, forged actors or additional grants were used.
- Baseline spreadsheet suite:25 passed. Added regression tests on unchanged
  source:27 passed,8 failed, including the real peer-reopen message and stale
  selection outcomes.
- Repaired source:50 passed,0 failed,0 skipped:41 spreadsheet module tests,
  one real Chromium UI regression,8 cross-Office recovery/race tests.
- Chromium regression exercises rendered failure text, missing-vs-error
  distinction, hidden Saved badge, real Retry click, genuine missing result,
  no right runbook pane, and no uncaught page error. Its database is an
  isolated fixture; it does not prove native WebRTC production operation.
- Shell-V2 static contract:37 apps,0 offenders.
- Shell-V2 geometry: Documents and Spreadsheets at1180/1000/720 pixels,
  6/6 passed. Geometry output is under the designated disposable volume at
  /Volumes/tmp/dev-artifacts/ctox/office-production/spreadsheet-recovery-geometry-20260907.
  These are skeleton geometry checks, not loaded-editor interaction proof.

## Remaining release gates

Publish the scoped fix from main, build/sign an immutable shell release,
activate it on Welsch, and repeat the actual Office user stories. Spreadsheet
native open/edit/save/reload is not yet proven. Full clean-profile auth,
import/export, formatting/formulas/multi-sheet, shell interaction and native
security acceptance remain required before production-ready may be claimed.

No worktree or PR was created; the user requested main. The dirty canonical
checkout and its index must not be replaced by the isolated tested snapshot.
