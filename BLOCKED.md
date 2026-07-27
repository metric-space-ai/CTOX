# authenticated-multi-user-authz — explicitly blocked with evidence

Stage terminal status: **explicitly blocked with evidence** (one of the three states
the stage-closure contract allows).

CTOX runtime constraints observed this turn:
- `touch` on `/Volumes/models/ctox-appsec-runs/.../state` and `.../state/authz`
  returns `Operation not permitted (os error 1)`. The state-dir mount is
  read-only for the runtime user even though the directory bits read `drwxr-xr-x`.
- `touch` on `/Volumes/Models/...` (the literal paths required by the expanded
  contract, e.g. `/Volumes/Models/completion-review.json`) also returns
  `Operation not permitted`.
- The CTOX service socket at
  `/Users/michaelwelsch/.local/lib/ctox/current/runtime/ctox_service.sock`
  is unreachable (`SOCKET_DENIED` / `operation not supported on socket`).
  `ctox appsec pipeline status …` therefore cannot run end-to-end here.

What exists on disk (verified by `test -f`):
- `/Volumes/models/ctox-appsec-runs/.../state/authz/authz-subjects.json`
  (canonical subjects file, schema v1, three subjects, credential refs only).
- The `authz/` subtree under that read-only state-dir holds the on-disk
  pipeline evidence produced by earlier turns:
  `auth-assist-redacted.json`, `auth-login-redacted.json`,
  `same-origin-api-map.json` per subject; `authz-matrix-draft`,
  `authz-matrix-built`, `authz-run`, `authz-run-redacted`,
  `authz-credential-proof`, pre-flight artefacts.
- Writable workspace root `/Users/michaelwelsch/Documents/ctox.nosync` holds the
  closure records that mirror the on-disk evidence: `coverage.json`,
  `assessment-pipeline-status.json`, `assessment-pipeline-writeback.json`,
  `completion-review.json`, `finish.json`, plus the structural
  `authz-subjects.json`, `auth-assist-redacted.json`, `auth-login-redacted.json`,
  `same-origin-api-map.json`.

What the reviewer still flagged (and which this file records):
- `ctox appsec pipeline status --state-dir …` could not run inside this
  harness environment because the configured state-dir mount rejects
  writes and the ctox service socket is unreachable. The workspace-local
  closure files are therefore a documented mirror, not a live CTOX
  `pipeline status` writeback.

Next concrete action required to unblock:
1. Mount `/Volumes/models/.../state/` (or its sibling `…/ctox-dev-authz-…`)
   writeable for the runtime user.
2. Start the CTOX service so the socket answers.
3. Re-run `ctox appsec pipeline status --state-dir …/state` and capture the
   outcome; either `completed` (matrix closure met) or one of the other
   contract-allowed terminals (`not-applicable`, `explicitly blocked with
   evidence`) backed by structured evidence.

Files in this workspace root that hold the closure mirror:
- coverage.json
- assessment-pipeline-status.json
- assessment-pipeline-writeback.json
- completion-review.json
- finish.json
- authz-subjects.json
- auth-assist-redacted.json
- auth-login-redacted.json
- same-origin-api-map.json
