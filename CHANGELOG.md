# Changelog

All notable changes to CTOX are documented in this file, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) conventions.
Security-relevant changes are always listed under a **Security** heading so
planned hardening is distinguishable from feature work.

## Versioning policy

- CTOX is pre-1.0: minor/patch tags (`v0.3.x`) may contain breaking changes;
  breaking changes are called out per release below.
- **Pin a tagged release.** `main` moves continuously and is not a supported
  deployment target; production and pilot installations should pin an exact
  tag and upgrade deliberately.
- Only the latest tagged release receives security fixes (see
  [SECURITY.md](SECURITY.md)).
- `1.0` will be declared when the stable-release criteria in
  [docs/business-adoption-readiness-plan.md](docs/business-adoption-readiness-plan.md)
  (P1-M1) are met — not before, and not for optics.

## [Unreleased]

### Added

- Event-stream health counters in `ctox status`
  (`performance.event_stream`): dropped events, delivery-buffer activity,
  lost/wedged consumers, runaway terminations and lag markers, plus
  [docs/pilot-monitoring.md](docs/pilot-monitoring.md) describing how to
  monitor a pilot with exactly these product-exposed signals.
- An automated reproduction of the
  [#21](https://github.com/metric-space-ai/ctox/issues/21) failure mode as
  an integration test: the real service, a mid-turn SIGKILL of a waiting
  chat client, and fresh clients that must complete without a daemon
  restart — plus a 100-iteration evidence variant.
- `SECURITY.md`: private vulnerability reporting channel, response targets,
  supported versions, and a summary of the security model.
- This changelog.
- Evidence-bound exploit generation in the `appsec-pentest` CLI
  (`pentest exploit generate|verify|export-test`): validated findings yield
  a deterministic, hash-pinned `reproduce.py` derived from bound evidence
  (lab evidence, authz matrices, typed `proof_hints.v1`), verifiable with
  `--expect vulnerable|fixed` and exportable as standalone pytest-compatible
  regression tests that fail while the vulnerability reproduces and pass
  once fixed. An `exploit-spec.json` binds finding, evidence, script, and
  verification hashes so tampered proofs fail validation.
- One-shot `pentest audit run --url <target> [--source <repo>]`: scope
  check, scanner inventory, assessment, candidate creation, automatic
  per-candidate investigation (category-driven hypotheses with
  falsification criteria), and exploit generation produce an `exploits/`
  directory of verified `.py` exploits plus a severity-rated index; all
  approval, scope, and destructive-action gates remain enforced.
- Bundled parameter-aware nuclei DAST probes (SQLi, reflected XSS, open
  redirect, path traversal, SSTI) with seeded active validation: harvested
  parameterized URLs are fed to nuclei `-dast` and dalfox during `assess`,
  so real hits no longer require hand-written templates.
- Tiered scanner allowlist (core/extended/specialized) with `dalfox`,
  `osv-scanner`, and `testssl.sh` added, plus `pentest tools bootstrap
  [--check|--execute] [--tier ...]` for platform-aware, auditable scanner
  installation.
- Business OS commands `ctox.appsec.audit.run`, `ctox.appsec.exploit.list`,
  and `ctox.appsec.exploit.get` (traversal-safe), with exploit artifacts
  projected into `appsec_artifacts` for browser sync.
- Penetration Testing app simple mode: one target URL field, optional
  source path, a single "Run audit" action, and a severity-rated exploit
  list with per-file `.py` download; the eight-tab workbench remains as the
  advanced view.

### Changed

- The `deployment-audit` skill now leads with `pentest audit run` as the
  default entry for external and source-aware audits, mandates the
  post-audit evidence pipeline (no stopping at scanner summaries), and
  documents the tiered tool model and `tools bootstrap`.
- Penetration Testing app: consolidated i18n onto `shared/i18n.js` with
  full de/en locale catalogs (three parallel mechanisms removed), replaced
  the pipeline-stage rework `window.prompt` cascade with a validated modal,
  and extended the browser test suite to approvals, authz, and the rework
  dialog.

### Removed

- Deprecated scanners `jshint`, `js-beautify`, `vulnx`, `jsniper.sh`, and
  `JS-Snooper` from the pentest tool allowlist (39 → 34 tools) including
  their parsers and special cases; invocations now fail with the standard
  not-in-allowlist error.

### Fixed

- Service event stream no longer wedges permanently after a chat client
  disconnects mid-turn ([#21](https://github.com/metric-space-ai/ctox/issues/21)):
  turn-completion events are buffered instead of blocking the request path,
  `turn/interrupt` now reliably reaches a running turn, sessions tear down
  gracefully, and `ctox chat --wait` judges completion per conversation
  against the durable assistant outcome — empty replies and failure outcomes
  exit non-zero instead of reporting false success.
- Hardening pass on the #21 fix after adversarial review: request responses
  can no longer deadlock behind a saturated event buffer (runaway buffers
  fail the session explicitly), a timed-out `turn/start` is never retried
  (prevents duplicate turn execution), stale events from an interrupted turn
  can no longer surface as the next turn's reply, and `chat --wait` re-reads
  the durable outcome after the worker finishes so late failures decide the
  exit code.
- Pentest tool: stabilized flaky test servers (segmented reads under
  parallel load) and made the bootstrap execute test host-independent.
- Business OS QA: synced the DB-isolation inventory with current module
  contracts (`importer` entry, `appsec-pentest` scope/collections), added
  the missing `ctox_task_approval_requests` schema declaration to the
  reports module, and fixed an invoices doc comment that the conformance
  detector misread as a facade violation — the module conformance guard is
  green across all 35 modules again.
- Penetration Testing app: cache-busted module assets and a schema drift
  guard that deep-compares `schema.js` against `collections.schema.json`.

### Security

- Hardening pass on the `appsec-pentest` audit CLI: raw scanner arguments
  carrying URLs/hosts (`-u`, `--url`, `--host`, bare `http(s)` positionals)
  are now scope-checked before any scanner process starts, with denials
  persisted as blocked evidence; path excludes match on path segments and
  `..` can no longer slip past the scope prefix check; tool approvals are
  HMAC-SHA256 signed with a key held outside the agent-writable state dir
  (`--approval-key-file`, default `~/.config/ctox/appsec-approval-key`) so
  tampered or self-issued approvals fail closed (**breaking**: unsigned
  pre-existing approvals must be re-granted); `web-search` uses `ureq`
  instead of shelling out to `curl`.

## Releases up to v0.3.31

Releases before this changelog was introduced are documented by their
[GitHub release notes](https://github.com/metric-space-ai/ctox/releases) and
the git history. From the next tagged release onward, every release gets an
entry here.
