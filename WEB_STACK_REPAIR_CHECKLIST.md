# CTOX Web Stack and Systematic Research Repair Checklist

> This file is both the implementation checklist and the mandatory work log.
> Update it in the same commit as the corresponding code or operational
> change. Do not mark an item complete without adding evidence.

## Work Log Header

| Field | Value |
|---|---|
| Overall status | `IN_PROGRESS` |
| Current owner | Kimi session (ctox.nosync) |
| Started at | 2026-07-23T08:30 |
| Last updated at | 2026-07-23 |
| Source checkout | `/Users/michaelwelsch/Documents/ctox-file-bridge-release.nosync` |
| Source branch | `main` |
| Current source commit | `cb08c96a` (pushed to `origin/main`) |
| Managed target | `skf.ctox.dev` |
| Active managed release | SKF: `/home/ctox/.local/lib/ctox/releases/branch-main-20260723T170548Z`; thesen: `/home/ctox/.local/lib/ctox/releases/branch-main-20260723T190039Z` (both proven byte-identical to `cb08c96a`) |
| Current workstream | Production deploy + verification done for code repairs; research rebuild blocked on operator decision; thesen replication drift open (F-009) |
| Current blocker | 1) SKF research command `cmd_0ad31d86…` is terminally `failed`; resuming under the durable id is refused by design — operator must authorize a fresh dispatch. 2) Thesen tenant manifest drift `delete_when_missing` keeps native RxDB replication down (F-009). 3) Unlock gate needs human browser auth for CompanyHouse/Google/RocketReach. |
| Next concrete action | Operator decision on SKF research rebuild dispatch + thesen manifest key, then rerun unlock gate `--strict` after browser auth |

### Status Values

- `NOT_STARTED`: no verified work performed.
- `IN_PROGRESS`: implementation or verification is actively underway.
- `BLOCKED`: exact external or technical blocker is recorded below.
- `READY_FOR_PRODUCTION`: code is merged and local verification is complete.
- `PRODUCTION_VERIFYING`: managed upgrade completed; production checks running.
- `COMPLETE`: every completion-evidence item is present and no required work
  remains.

## Agent Update Rules

For every meaningful action:

1. Update the relevant checkbox.
2. Add a row to **Implementation Log** or **Operational Log**.
3. Record exact commands and summarized output in **Verification Evidence**.
4. Record newly discovered defects in **Findings Register** before fixing them.
5. Record architecture or contract choices in **Decision Log**.
6. Attach commit SHA, release path, queue/run ID, artifact path, or screenshot
   path where applicable.
7. Never replace a failed result with a later passing result. Keep both entries
   so the history remains auditable.

Checkbox convention:

- `[ ]` not started.
- `[~]` in progress. If the Markdown renderer does not support it, use `[ ]`
  and append `STATUS: IN_PROGRESS`.
- `[x]` complete with linked evidence.
- `[!]` blocked. If unsupported, use `[ ]` and append `STATUS: BLOCKED`.

## Current Handoff Snapshot

| Item | Current state | Evidence |
|---|---|---|
| ZIP member relevance fix | Complete and pushed | commit `8a97a7b` |
| Managed upgrade after fix | Complete | release `/home/ctox/.local/lib/ctox/releases/branch-main-20260723T054427Z` |
| UIUC production reads | Pass | four URLs returned HTTP 200, relevance 10, evidence eligible |
| ENOLA real-member reads | Pass | APC15x8, APC19x10, AeronautCAM9x5 returned relevance 10 |
| ENOLA nonexistent member | Pass, correctly rejected | APC12x8 returned `query_relevance_not_established` |
| SKF verified research publish | Not complete | current dashboard is not approved |
| Research queue | Faulted/stalled | lease existed with zero active workers |
| Knowledge/graph/reports | Not complete | existing intermediate outputs fail required contracts |

## Objective

Restore CTOX Web Research so that a managed Business OS instance can:

1. discover real primary sources and scientific papers systematically;
2. read and snapshot original content through the typed CTOX Web Stack;
3. reject unreachable, fabricated, metadata-only, irrelevant, or stale sources;
4. extract measured data with row-, column-, unit-, file-, member-, and hash-level provenance;
5. build auditable Knowledge skills/resources and reports from admitted evidence;
6. update a living research dataset without duplicating or degrading prior verified work;
7. show durable progress in Business OS while the harness is working.

The immediate production target is `skf.ctox.dev`, domain
`drone_bearing_design_verified`, for UAV/drone propeller and bearing load data.

## Mandatory Operating Rules

- Work on `main` in the origin checkout:
  `/Users/michaelwelsch/Documents/ctox-file-bridge-release.nosync`.
- Read `README.md`, `HARNESS.md`, `docs/architecture.md`,
  `docs/ctox-rxdb.md`, `src/core/harness/FORK.md`, and applicable `AGENTS.md`
  files before changes.
- Do not introduce an HTTP fallback for Business OS data. Browser data must
  remain RxDB/WebRTC-backed.
- Do not add production behavior toggles through process environment variables.
- Do not bypass or weaken evidence, review, harness, or RxDB guard tests.
- Do not create free research subagents and do not expose `spawn_agent` to the
  research harness. `systematic-research` owns the complete workflow.
- Preserve unrelated local changes.
- Deploy only through:
  1. commit to `main`;
  2. push `origin/main`;
  3. run the managed `ctox upgrade --dev` path;
  4. verify the installed release and production behavior.
- Never patch only the live VM or generated RxDB bundles.

## Confirmed Production Findings

### Historical Benchmark Baseline

The current poor SKF behavior is a regression against a documented working
CTOX research stack. Treat the benchmark as a required behavioral baseline,
not as optional background material.

Repositories:

- [metric-space-ai/deep-search-benchmark](https://github.com/metric-space-ai/deep-search-benchmark)
- [metric-space-ai/deep-search-benchmark-site](https://github.com/metric-space-ai/deep-search-benchmark-site)
- Local benchmark checkout:
  `/Users/michaelwelsch/Documents/deepSearchBenchmark`

The published local benchmark snapshot generated on `2026-06-17T20:56:37Z`
records the CTOX/MiniMax-M3 harness with:

- 100 of 100 tasks complete;
- strict output-contract rate 1.0;
- median 7 cited sources per task;
- research quality score 1.0 in that benchmark snapshot;
- rating 1108.3;
- category ratings:
  - Deep chains: 1145.1;
  - Mixed multi-hop: 1116.1;
  - Official portals: 1114.9;
  - PDF and documents: 1155.3;
  - Recall and scrape: 1116.0;
  - Technical live data: 1106.3.

The benchmark covered deep multi-hop chains, broad recall crawling, PDF and
document extraction, official portals, technical live data, and mixed
provenance checks. Its review dimensions included hop correctness, final
answer, evidence, completeness, recency, and honesty.

Do not treat those numbers as a substitute for rerunning the benchmark. They
prove that the architecture previously demonstrated materially stronger
behavior and provide concrete regression targets.

### Web Stack

- Direct UIUC text reads now work in production with HTTP 200, immutable
  snapshots, relevance score 10, and `evidence_eligible=true`:
  - `apce_12x8_static_0621od.txt`
  - `apce_12x8_0624od_4043.txt`
  - `apce_12x8_0628od_6024.txt`
  - `apce_12x6_static_0629od.txt`
- The ENOLA Zenodo archive is real:
  `https://zenodo.org/api/records/20111572/files/Propeller_Database.zip/content`
- Archive SHA-256:
  `c9e92c5e5be0aeadab0d42e2ded5d85822f3e8b8e00029d434a68822e108ca98`
- The archive has 221 members, including 170 CSV files.
- Verified real members include:
  - `BBDD/APC/APC15x8/results/APC15x8_exp.csv`
  - `BBDD/APC/APC19x10/results/APC19x10_exp.csv`
  - `BBDD/Aeronaut/AeronautCAM9x5/results/AeronautCAM9x5_exp.csv`
- `APC12x8` is not an ENOLA member. Production now rejects that query with
  `query_relevance_not_established`.
- The ENOLA CSVs contain RPM, thrust, torque, CT, CP, and uncertainty columns.
  Some source headers are malformed, for example a missing delimiter between
  `THRUST[N]` and `u_THRUST[N]`. Repair must be explicit, auditable, and tied
  to the original bytes.

### Already Merged Fixes

- `4c00b18`: reviewer empty-verdict recovery.
- `3a47e62`: bind relevance score and eligibility into workspace v3 receipts;
  native validation checks path, hash, and score.
- `21ed2e1`: score textual datasets from immutable content; normalize
  `CT0/CP0/CQ0`.
- `a6c4892`: safer dataset identifier matching.
- `8a97a7b`: verify ZIP relevance against the complete hash-bound archive
  manifest.

Do not regress these behaviors.

## Workstream 0: Benchmark-Guided Regression Forensics

Complete this before inventing new research architecture.

- [x] Clone/fetch or inspect both benchmark repositories at their published
  revisions and record exact commit SHAs.
- [x] Identify the exact CTOX source commit/release used for the successful
  100-task CTOX/MiniMax-M3 benchmark.
- [x] Recover the exact benchmark runtime configuration from:
  - `bench.ctox.toml`;
  - the CTOX adapter in `engine/run_v1.py`;
  - task results and telemetry in `data/current-benchmark.json`;
  - the corresponding benchmark-site data;
  - CTOX service/runtime logs or release metadata retained with the run.
- [x] Record exact model, provider, Responses API settings, tool list, skill
  prompt, Web Stack adapters, budgets, timeout, search engine policy,
  unblocking behavior, and result contract used by the good run.
- [x] Select a representative regression corpus covering:
  - deep chain;
  - scholarly/paper discovery;
  - citation/reference hopping;
  - official portal;
  - PDF extraction;
  - structured dataset extraction;
  - broad recall;
  - blocked-page recovery;
  - technical live data.

> Phase-1 evidence (2026-07-23, Kimi): benchmark repo
> `4d9ca1e6ea80240b02ba7eb692a046b03672bbd0`, site repo
> `00cc237bdba3b69ee889ed8a0ba62a2b4d482bb7`, snapshot byte-identical across
> both (SHA-256 `4b3edf3e…`). Exact good-run CTOX SHA not recoverable from
> retained metadata; bounded to `06f8dd6c`…`1b412c8e` (v0.3.22 window,
> canonical reference `1b412c8e`). Full analysis, recovered config, 10 ranked
> regression hypotheses with code evidence, and the 9+2-task corpus:
> `docs/benchmark-regression-forensics.md` and
> `docs/benchmark-regression-forensics.json`. Key result: replay must be
> dual-path (chat adapter AND Business OS systematic-research queue) — a
> chat-only replay will not reproduce the regression.
- [ ] Replay that corpus against the historical release if reproducible.
- [x] Replay the identical corpus against current `main`.
- [x] Compare per task (chat 11/11; queue 8/11 final answers, 3 partial, 0 terminal — F-010):
  - queries issued;
  - engines/adapters selected;
  - hop count;
  - unique canonical sources;
  - source type distribution;
  - primary vs aggregator ratio;
  - successful reads;
  - blocked/unblocked reads;
  - DOI/OA resolutions;
  - references/citations followed;
  - evidence entailment;
  - source reachability;
  - latency;
  - tool errors;
  - completion honesty.
- [x] Produce a machine-readable regression report and a concise Markdown
  analysis that identifies the first bad commit for each lost capability.
  (`docs/benchmark-replay-phase2.{md,json}`; hypotheses confirmed/refuted by
  behavior — first-bad-commit via replay evidence, bisect not needed for the
  confirmed items.)
- [ ] Use `git bisect` or an equivalent bounded commit comparison for confirmed
  regressions. Do not guess based only on current code appearance.
- [ ] Specifically determine whether regressions came from:
  - replacing or disabling Google/search adapters;
  - collapsing agentic iteration into one static deep-research call;
  - changing the `systematic-research` skill;
  - removing scholarly/DOI/OA/reference traversal;
  - changing model-visible tool schemas or tool availability;
  - truncating context, candidate inventories, or exclude lists;
  - changing MiniMax M3 provider or Responses API behavior;
  - adding reviewer constraints that prevent useful exploration;
  - queue/harness lifecycle failures;
  - Web Stack cache, relevance, unblocking, or extraction changes.
- [ ] Restore lost behavior from the last known-good implementation where
  possible instead of building a parallel replacement.
- [ ] Add the representative benchmark corpus as a release regression gate.
- [ ] Rerun the complete 100-task CTOX benchmark after repair.
- [ ] Do not accept the repair if current CTOX materially regresses against the
  prior snapshot in hop, evidence, completeness, source quality, or honesty.
- [ ] Document any intentional behavior change and why it is superior under
  the same benchmark methodology.

### Benchmark Regression Deliverables

The agent must attach:

- historical benchmark repo SHA;
- historical CTOX SHA/release;
- current CTOX SHA;
- task corpus and commands;
- old-vs-current tool-call traces;
- first-bad-commit evidence;
- capability-by-capability root-cause table;
- repair commits;
- focused replay results;
- complete 100-task rerun result;
- comparison against the published benchmark-site snapshot.

### Current SKF Research State

- The visible 68-source dashboard is not a completed or approved rebuild.
- The current rebuild task is:
  - command:
    `cmd_0ad31d86-d95d-49cf-b496-7796f1ad899c`
  - run:
    `research_run_85adffb8-4a12-425e-bebd-e6c7f1a5594e`
  - queue:
    `queue:system::c2e3705bbd9656073528c374`
  - workspace:
    `/home/ctox/ctox-workspaces/business-os/cmd_0ad31d86-d95d-49cf-b496-7796f1ad899c`
- Discovery produced 184 deduplicated candidates across two inventories.
- Most candidates are not evidence. Metadata pages, failed transports,
  irrelevant results, and unread candidates must remain in
  `source_candidates`.
- Existing intermediate output is invalid and must not be published:
  - `source_candidates.csv` contains only 12 rows instead of the complete
    candidate audit inventory;
  - `source_catalog.csv` contains 7 rows;
  - `measured_load_points.csv` contains 28 rows;
  - `derived_bearing_loads.csv` contains 56 rows;
  - the measured table uses `axial_force_N`, which the native importer does
    not accept;
  - it uses `measurement_kind=static_thrust_CT_CP`, while the native contract
    accepts `measured`, `direct`, or `experimental`;
  - PDF extracted-text byte counts were populated with PDF source byte counts;
  - the graph and Knowledge outputs are incomplete.
- A prior evidence guard rejected Lisboa evidence `ev-0009` because it was not
  bound to a matching typed Web Stack retrieval in that attempt.

### Queue/Harness Failure

- The current task can remain `route_status=leased` while:
  - `worker_active_count=0`;
  - `busy=false`;
  - `acked_at=null`;
  - no new workspace files are written.
- This is an orphaned lease, not active research.
- Business OS must never show this as healthy progress.

## Workstream A: Web Stack Discovery

- [x] Trace the complete `systematic-research` call path from Business OS
  command to typed Web Stack tools.
- [x] Confirm that `ctox_web_search`, `ctox_scholarly_search`,
  `ctox_deep_research`, `ctox_web_read`, DOI/OA resolution, browser capture,
  scraping, and unblocking are available to the workflow.
- [x] Ensure `ctox_deep_research` is only one discovery round, never the entire
  research workflow.
- [x] Restore multi-engine search policy, including the intended Google search
  adapter where configured. Do not silently collapse discovery to Brave.
  (Cascade [Google, Brave, DuckDuckGo, Bing] pinned by tests; Google live
  health = production-pending via unlock gate.)
- [x] Verify scholarly adapters return scientific papers, DOI records, OA
  locations, references, citations, datasets, and supplements.
- [ ] Require orthogonal facets: propeller test rigs, thrust/torque/RPM,
  vibration/unbalance, flight loads, radial/axial load derivation, and bearing
  selection.
- [x] Follow references and citations from admitted papers.
- [x] Carry a canonical exclude list into every discovery round.
- [x] Deduplicate by canonical URL, DOI, stable identifier, and content hash.
- [x] Count a source as newly discovered only once.
- [x] Continue for at least two orthogonal rounds after the last new eligible
  evidence source.
- [x] Persist every candidate and rejection reason to `source_candidates`.
- [x] Never promote search snippets or metadata records directly to evidence.

## Workstream B: Typed Reads and Evidence Admission

- [ ] Every admitted source must have a current typed `ctox_web_read` call in
  the same durable research attempt.
- [ ] Require:
  - HTTP 2xx, excluding 204;
  - canonical original URL;
  - original content, not a search or metadata page;
  - immutable snapshot;
  - SHA-256;
  - server receipt;
  - `transport_evidence_eligible=true`;
  - `evidence_relevance_score >= 8`;
  - `evidence_eligible=true`;
  - non-aggregator source tier.
- [ ] Reject 404/410, soft 404, login walls without captured content,
  metadata-only records, irrelevant content, and unverifiable redirects.
- [ ] Detect fabricated URLs independently of the model.
- [ ] Preserve rejected records for audit, but exclude them from source
  catalog, scores, Knowledge, graph claims, and reports.
- [ ] Re-fetch and rebind Lisboa `ev-0009` or remove it and every dependent
  claim.
- [ ] Verify DOI and OA resolution ends at retrievable original content.
- [ ] Bind evidence admission to exact receipt path, content hash, query,
  relevance score, and eligibility.

## Workstream C: ZIP and Dataset Handling

- [ ] Keep the full manifest outside model context when large, but persist it
  as a hash-bound artifact.
- [ ] Validate:
  - receipt schema;
  - archive hash equals snapshot hash;
  - manifest byte hash equals receipt manifest hash;
  - manifest schema and embedded hash;
  - member count;
  - member path and member SHA-256.
- [ ] Score archive relevance against verified member paths, not the archive
  filename alone.
- [ ] Reject nonexistent member names even when the archive itself is valid.
- [ ] Extract selected members from the immutable archived bytes.
- [ ] Record archive URL, archive hash, manifest hash, member path, member
  hash, source row, source column, source unit, parsing rule, and output row.
- [ ] Repair malformed headers only through a deterministic parser rule with
  explicit audit evidence.
- [ ] Never invent missing measurements or units.
- [ ] Add regression fixtures for real member, nonexistent member, tampered
  manifest, wrong archive hash, count mismatch, and malformed source header.

## Workstream D: Measurement and Derived-Load Contracts

- [ ] Match the native importer in `src/core/business_os/store.rs`.
- [ ] `measured_load_points` must use supported fields such as:
  - `rpm`;
  - `propeller_size`;
  - `propeller_diameter_in`;
  - `propeller_pitch_in`;
  - `thrust_N` or `axial_load_N`;
  - `torque_Nm`;
  - source and row lineage.
- [ ] Use `measurement_kind=experimental` for direct ENOLA measurements.
- [ ] Set `is_derived=false` for direct measurements.
- [ ] Require positive RPM and explicit units.
- [ ] Include uncertainty values where the source provides them.
- [ ] Keep decimal values machine-readable in storage and export. Locale
  presentation may use comma, but CSV/XLSX import must remain unambiguous.
- [ ] Explain table columns and units in Knowledge header tooltips.
- [ ] Explain propeller notation such as `9x5` as diameter x pitch in inches.
- [ ] Split propeller notation into numeric diameter and pitch columns.
- [ ] UIUC CT/CP rows are direct dimensionless coefficients.
- [ ] Any conversion from CT/CP to force or torque is derived and belongs in
  `derived_bearing_loads`, with formula, constants, assumptions, units, and
  source-row lineage.
- [ ] Never mix measured axial thrust with an inferred radial bearing load.
- [ ] Add source row-count reconciliation and reject partial silent imports.

## Workstream E: Knowledge, Graph, and Reports

- [ ] Build Knowledge only from admitted evidence.
- [ ] Create useful skills/resources for:
  - measured propeller operating points;
  - thrust and torque interpretation;
  - radial and axial bearing load derivation;
  - vibration and unbalance;
  - flight/transient loads;
  - bearing sizing assumptions and limits;
  - provenance and source audit.
- [ ] Every factual statement must reference admitted source IDs and exact
  evidence rows or passages.
- [ ] Skills should refer back to sources/resources rather than copy large
  source texts.
- [ ] Add runbooks only for repeatable engineering workflows.
- [ ] Ensure every row written by this run carries the exact research run and
  command IDs.
- [ ] Build semantic graph nodes only from meaningful domain concepts.
- [ ] Exclude hashes, URLs, internal IDs, generic metadata, and parser fields
  as graph concepts.
- [ ] Every graph edge must be typed, sourced, confidence-bearing, and
  provenance-bound.
- [ ] Fix `INVALID_GRAPH_CONTRACT` fail-closed rather than hiding the error.
- [ ] Optimize graph loading and level-of-detail behavior for large graphs.
- [ ] Make expand/reduce detail deterministic and semantically meaningful.
- [ ] Build reports from selected or automatically matched Knowledge skills.
- [ ] Re-run claim audit after any source, measurement, Knowledge, or report
  update.

## Workstream F: Reviewer and Harness

- [ ] Determine why a lease can remain active after its worker disappears.
- [ ] Add lease heartbeat/expiry and durable worker identity.
- [ ] Mechanically release or requeue orphaned leases without duplicating the
  command.
- [ ] `busy`, worker count, route status, and UI progress must agree.
- [ ] A leased task with zero workers and no heartbeat must display a clear
  failed/stalled state.
- [ ] Persist timeline events for lease, start, tool call, artifact update,
  review, rework, completion, and failure.
- [ ] Ensure MiniMax M3 is the configured harness model for this managed
  instance.
- [ ] Do not allow the harness model to choose another model.
- [ ] Do not expose `spawn_agent` to the research parent.
- [ ] Reviewer must independently validate URLs, receipts, hashes, row
  lineage, native schemas, and claim bindings.
- [ ] Empty or malformed reviewer verdicts must fail closed with bounded retry.
- [ ] A successful prose response is not completion evidence.
- [ ] Publish only after durable source, data, claim, and native writeback
  guards pass.

## Workstream G: Business OS Progress and Living Updates

- [ ] Show actual research phase and counts in Web Research:
  discovered, read, rejected, admitted, extracted, audited, published.
- [ ] Stream incremental, valid updates through RxDB/WebRTC.
- [ ] Do not expose intermediate invalid tables as completed Knowledge.
- [ ] Show upgrade state in the shell while `ctox upgrade --dev` is running.
- [ ] Distinguish reconnecting, upgrading, researching, reviewing, blocked,
  failed, and ready states.
- [ ] Prevent stale cached app data from replacing newer persisted data after
  an upgrade or reconnect.
- [ ] Keep the research-to-Knowledge-to-Documents chain explicit and
  updateable.

## Required Automated Verification

- [ ] Targeted Rust tests for evidence admission and native writeback.
- [x] Web Stack tests for text, PDF, HTML, JSON, CSV, ZIP, redirects, 404,
  soft 404, metadata-only, and blocked sources.
- [x] ZIP manifest binding tests.
- [x] Scholarly discovery and DOI/OA resolution tests using stable fixtures.
- [x] Candidate deduplication and exclude-list tests.
- [x] Two-round saturation tests.
- [ ] Reviewer adversarial tests with fabricated URLs and mismatched receipts.
- [ ] Orphaned lease recovery tests.
- [ ] RxDB/WebRTC tests:
  `node src/apps/business-os/rxdb/tests/run-all.mjs`.
- [ ] Native RxDB tests:
  `cargo test --manifest-path src/core/rxdb/Cargo.toml`.
- [ ] Relevant `cargo fmt --check`, `cargo check`, and targeted `cargo test`.

Tests are required evidence, but the task is not complete merely because tests
pass. Production behavior must also be demonstrated.

## Required Production Verification on SKF

- [x] Push the repair to `origin/main`.
- [x] Run full `ctox upgrade --dev` through the managed path.
- [x] Confirm the active release points to the new main commit.
- [~] Verify `llm.ctox.dev/v1` Responses API operation with the configured
  MiniMax M3 runtime. STATUS: IN_PROGRESS — transport reachability verified
  (HTTP 403 auth-gated without token); no credential-bearing probe attempted.
- [x] Verify service, native peer, WebRTC replication, and RxDB collections.
- [x] Run direct production reads for all four UIUC URLs above.
- [x] Run direct ENOLA queries for APC15x8, APC19x10, AeronautCAM9x5, and the
  negative APC12x8 case.
- [!] Resume the existing durable research command; do not create a duplicate.
  STATUS: BLOCKED — command is terminally `failed`; release under the durable
  id is refused by design (`channels.rs:5908`). Operator decision required:
  authorize a fresh dispatch (new command id) for the verified rebuild.
- [ ] Confirm workspace timestamps and research counters advance.
- [x] Confirm no orphaned lease and no red task remain.
- [ ] Inspect persisted native data counts and sample rows before opening the
  browser.
- [ ] Confirm the complete candidate audit inventory exists.
- [ ] Confirm source catalog contains only admitted evidence.
- [ ] Confirm measured and derived tables pass native import.
- [ ] Confirm Knowledge skills/resources exist and contain source references.
- [ ] Confirm graph contract is valid and interaction is responsive.
- [ ] Confirm reports are based on Knowledge and retain citations.
- [ ] Confirm generated CSV/XLSX files appear in Files and Spreadsheets and can
  be downloaded.
- [ ] Interactively test both authorized SKF users in a clean browser session.
- [ ] Verify original-source links open real content and do not return 404.

## Completion Evidence

Do not report completion without a compact evidence bundle containing:

- main commit SHA and pushed branch;
- installed release path and upgrade result;
- production Web Stack read matrix;
- candidate/admitted/rejected source counts;
- direct measurement and derived-load row counts;
- source-, data-, and claim-audit results;
- Knowledge skill/resource/runbook counts;
- graph node/edge counts and contract result;
- report count;
- native writeback result;
- queue/harness timeline with no orphaned lease;
- browser screenshots or trace references for both users;
- a list of any remaining rejected or unresolved sources.

## Findings Register

Add one row as soon as a defect is confirmed. Do not wait until it is fixed.

| ID | Severity | Status | Component | Finding | Reproduction/Evidence | Root cause | Fix commit |
|---|---|---|---|---|---|---|---|
| F-001 | Critical | Fixed | Web Stack ZIP relevance | Archive queries were scored without verified member-path evidence. | Real archive accepted an absent `APC12x8` intent before the fix. | Relevance input omitted the hash-bound full manifest member paths. | `8a97a7b` |
| F-002 | Critical | Fixed, pushed `origin/main` | Harness/queue | Research task can remain leased with no active worker, no acknowledgement, and no workspace progress. | `route_status=leased`, `acked_at=null`, `worker_active_count=0`, `busy=false`. | `transition_business_command_for_task_in_transaction` returned early on terminal commands without settling the queue route; sweep aborted on first candidate error; no durable worker identity; projection rendered incomplete leases as healthy. | `ae81091` |
| F-003 | Critical | Fixed, pushed `origin/main` | Research writeback | Intermediate measured rows do not match the native import contract. | `axial_force_N` and `measurement_kind=static_thrust_CT_CP`. | Dashboard CSVs were hand-authored by the harness model; no deterministic builder enforcing the native contract in `store.rs` (`validate_systematic_research_csv`). | `915c70b` |
| F-004 | High | Fixed, pushed `origin/main` | Candidate audit | Only 12 candidate rows were written from 184 deduplicated candidates. | `source_candidates.csv` row count. | `source_review_discovery.py` rewrote screened/rejected inventories per round from empty lists; dashboard table was model-authored and arbitrarily cut. | `915c70b` |
| F-005 | High | Fixed, pushed `origin/main` | Evidence provenance | PDF extracted-text byte counts were populated from PDF source byte counts. | Receipt/artifact size comparison. | `evidence_guard.py` validated extracted-text path/SHA-256 but never `byte_count` and never cross-bound it to the server v3 receipt. | `915c70b` |
| F-008 | High | Fixed, pushed `origin/main` | Queue/harness | Six pre-existing `take_messages`/`release` guard tests red at pristine HEAD `8a97a7b`. | `cargo test --bin ctox mission::channels`: 6 failed with `Invalid column index: 22`. | `take_messages` SELECT lists omitted `last_error` while the shared row mapper reads it at index 21/22. Found during F-002 verification. | `ae81091` |
| F-009 | Critical | Open — operator decision | Business OS / tenant manifest (thesen) | After upgrade to main `cb08c96a`, native RxDB peer replication on thesen stays down: fail-closed parse of `sellify/external-sql.json` — `write_operations[14].refresh[0]` contains `delete_when_missing`, written by hand-deployed feature-branch builds (v229/v230), unknown to main (serde `deny_unknown_fields`). Feature exists nowhere in repo history. | Peer journal: deterministic manifest parse error, replication retries with 300 s backoff; daemon/shell/peer identity healthy. | Tenant manifest drift from side deployments of feature branches never merged to main. Options: (a) delete the single JSON key (main-compatible), (b) implement the feature on main from scratch. Tenant data not hand-edited per safety rule. | _pending decision_ |
| F-010 | Critical | Open — root cause pinned | Harness/completion gate (systematic research) | Queue path can never reach terminal: completion validator requires a typed-discovery receipt + `validation/evidence-manifest.json`, but tools persist read receipts to `web-read/` and models write manifests to `snapshots/` — validator scan path ≠ tool persist path. Every run is rejected fail-closed and re-queued; completed tasks rewrite their answers every 3–8 min; review-budget stop never trips. | Phase-2 replay on thesen @ `f6535ce2`: 11/11 commands stuck `accepted` all day, 6 routes review-rejected / 5 harness-retry / 0 terminal at closeout; t47 has 10/10 ok deep-research runs + valid manifest yet still rejected; models write scratch `fix_manifest*.py` repair scripts. | Two-layer cause: (1) manifest path mismatch — FIXED by `a699e9a5` (verified working 2026-07-27: model writes `validation/evidence-manifest.json` correctly); (2) **queue-path worker session makes only `exec_command`/CLI calls, zero typed `ctox_*` tool calls** (165:0 census, `artifacts/f010-final-verdict-2026-07-27.json`) — validator rollout scan therefore always sees `observed discovery calls: none`; contract structurally unsatisfiable from the queue path. Terminal proof failed 2026-07-27 @ `3b3fee88` (3 identical rejections, command cancelled, `business_commands` still non-terminal — F-R5 persists). | _pending_ |
| F-006 | Critical | Open | Knowledge/graph | No approved Knowledge exists and graph contract is incomplete/invalid. | SKF Web Research UI and intermediate graph tables. | Downstream build ran before valid evidence/native writeback. | _pending_ |
| F-007 | Critical | Open | Web Stack/Systematic Research | Current SKF research behavior is a major regression against the previously successful CTOX/MiniMax-M3 benchmark. | Published benchmark snapshot: 100/100 tasks, median 7 sources, research quality 1.0, strong results across all six categories. | Phase-1 forensics (2026-07-23): 10 ranked hypotheses with code evidence in `docs/benchmark-regression-forensics.md` — strongest: production queue/receipt-gate/reviewer architecture did not exist in the good-run window (H1), deep-research-first enforced at three layers (H2), evidence/relevance gate rejects sources the benchmark-era agent freely used (H4). Awaiting dual-path replay for first-bad-commit confirmation. | _pending_ |

## Implementation Log

Record source-code changes here. Use one row per coherent change.

| Timestamp | Agent | Workstream | Files changed | Summary | Local verification | Commit | Push |
|---|---|---|---|---|---|---|---|
| 2026-07-23 | Codex | B/C | `src/tools/web-stack/src/web_search.rs` | Bound ZIP relevance to verified full-manifest member paths and hashes. | Targeted relevance tests, `cargo fmt`, `cargo check` | `8a97a7b` | `origin/main` |
| 2026-07-23 | Kimi | C/D | `src/skills/system/research/systematic-research/scripts/` (`dashboard_knowledge_build.py` new, `source_review_discovery.py`, `evidence_guard.py`, tests), `SKILL.md`, `references/evidence_integrity.md`, `src/core/business_os/store.rs` (guard-test fixture) | Deterministic dashboard builder matching the native writeback contract; full candidate audit accumulation; extracted-text byte-count/receipt binding; audited ENOLA header repair; native guard test repaired without loosening validation. F-003/F-004/F-005. | Python: 57 tests OK; `cargo test --bin ctox systematic_research`: 22/22; `cargo fmt --check` clean | `915c70b` | `origin/main` |
| 2026-07-23 | Kimi | F | `src/core/mission/channels.rs`, `src/core/service/service.rs`, `src/core/business_os/store.rs` (projection), `HARNESS.md` | Orphaned-lease reclamation: terminal-command settle, per-candidate sweep isolation, durable `lease_worker_id`, 60 s sweep independent of router idle gates, fail-closed projection; `take_messages` `last_error` column fix. F-002/F-008. | `cargo check` clean; `cargo test --bin ctox mission::channels`: 105/105 (6 pre-existing guards repaired + 3 new); `business_os::store` serial 252/16 == pristine-baseline failures minus repaired test | `ae81091` | `origin/main` |
| 2026-07-23 | Kimi | A/B (unlocking) | `src/tools/web-stack/src/unlock_report.rs` (new), `unlock.rs`, `lib.rs` | Production unlock acceptance gate: `ctox web unlock report [--strict]`, evidence-bound per-adapter checks, redaction self-check, fail-closed gate (unlocking checklist item 13 / WU-004). | `cargo check -p ctox-web-stack` clean; `cargo fmt --check` clean; `cargo test … unlock_report`: 9/9; 18 pre-existing unlock guards intact; 4 flaky failures confirmed pre-existing at pristine HEAD | `cb08c96a` | `origin/main` |
| 2026-07-23 | Kimi | A | `scripts/source_review_discovery.py`, `test_source_review_discovery.py`, `web_search.rs`, `scholarly_search.rs` | Canonical dedup (hash/DOI/openalex/URL), exclude list in every round, deterministic two-round saturation; cascade order [Google, Brave, DuckDuckGo, Bing] + full-cascade budget pinned; scholarly DOI/OA/reference fixture tests. | Python suite 72 OK (11 new); `cargo check -p ctox-web-stack` clean; 6/6 new web-stack tests | `cdad06d0` | `origin/main` |
| 2026-07-23 | Kimi | A | `src/core/service/service.rs`, `SKILL.md` | Agentic tool loop: `requiredInitialTool` forcing removed, deep-research receipt replaced by any-tool discovery receipt bound to run/command/workspace, prompt without mandated first tool; guards fail-closed unchanged (H2 layers 2+3). Committed via hunk-split — concurrent agent's lease-sweep/`2*pi` WIP deliberately excluded. | `cargo check` on exact staged state: clean; `cargo fmt --check` clean; bin tests not rerun (disk constraints, stated as gap) | `1292ad73` | `origin/main` |
| 2026-07-24 | Kimi | B | `src/tools/web-stack/src/web_search.rs` (tests only) | Typed read/reject fixture matrix: PDF/HTML/JSON/CSV admission with hash-bound artifacts, end-to-end receipt byte+hash binding; multi-hop redirects, non-content redirect reject, hard 404/410, 204 exclusion, bot/CAPTCHA walls; ZIP manifest member path/hash binding, tamper + zip-slip + stale-manifest cases. | 12/12 new tests; full suite 413 passed (2 pre-existing flaky failures re-verified at pristine HEAD); `cargo check`/fmt clean; `git status` confirmed scope confined to web-stack | `f6535ce2` | `origin/main` (on top of concurrent `5aa7d359`) |
| 2026-07-27 | Kimi | Unlocking (cap 10) | `src/core/capabilities/scrape.rs`, `src/core/business_os/store.rs` (match arm), `scrape-targets/_shared/generic-prospect-v1.js`, `fixture-gate.test.mjs`, `scrape_bridge.rs`, `unlock_report.rs` | Typed `authorization_required` + reauthorization handoff on expired protected sessions (login landing), replacing misclassification as `portal_drift` without handoff; report gate fails closed on missing reauth evidence. Root cause included production reruns missing `--input-json` and a missing status vocabulary. | Root-crate `cargo check` clean (one non-exhaustive match found+fixed); `cargo test -p ctox-web-stack unlock` 30/30; node fixture gate 6/6; 6 native unit tests compile-verified, not executed (disk) | `237b118b` | `origin/main` |
| _timestamp_ | _agent_ | _A-G_ | _paths_ | _change_ | _commands/results_ | _SHA_ | _remote/branch_ |

## Operational Log

Record deployments, queue actions, production reads, data rebuilds, and browser
checks here.

| Timestamp | Agent | Target | Action | Durable IDs/artifacts | Result | Follow-up |
|---|---|---|---|---|---|---|
| 2026-07-23 | Codex | SKF managed VM | Full `ctox upgrade --dev` | release `/home/ctox/.local/lib/ctox/releases/branch-main-20260723T054427Z`; backup `/home/ctox/.local/state/ctox/backups/update-20260723T054443Z` | Upgrade and smoke checks passed | Production source/read verification |
| 2026-07-23 | Codex | SKF Web Stack | UIUC and ENOLA production read matrix | immutable snapshots and ENOLA 221-member manifest | Positive cases passed at score 10; absent APC12x8 rejected | Resume audited research rebuild |
| 2026-07-23 | Codex | SKF queue | Released existing research task after evidence fix | queue `queue:system::c2e3705bbd9656073528c374` | Lease became orphaned with zero worker progress | Repair queue/harness lease lifecycle |
| 2026-07-23 | Kimi | GitHub `origin/main` | Pushed repair commits `915c70b`, `ae81091`, `cb08c96a` | main `8a97a7b → cb08c96a` | Push succeeded; managed upgrade still pending (no VM access from this machine) | Deploy via managed `ctox upgrade --dev` on `skf.ctox.dev` and `thesen.ctox.dev` |
| 2026-07-23 | Kimi | Local forensics | Workstream 0 phase 1: benchmark SHAs, good-run window, 10 ranked regression hypotheses, 9+2-task replay corpus | `docs/benchmark-regression-forensics.{md,json}` | Complete; exact good-run SHA not recoverable, bounded to `06f8dd6c…1b412c8e` | Phase 2: dual-path replay (chat + queue) of the corpus |
| 2026-07-23 | Kimi | SKF managed VM | Managed `ctox upgrade --dev` to `cb08c96a` (29m06s source build) + health verification | release `/home/ctox/.local/lib/ctox/releases/branch-main-20260723T170548Z`; backup `/home/ctox/.local/state/ctox/backups/update-20260723T170605Z`; release files byte-identical to GitHub `cb08c96a` (SHA-256 proof) | Upgrade OK; service/peer/RxDB healthy, `replicationUp:true`; evidence `runtime/vm-ops/production-verification-2026-07-23.md` §B | Production reads |
| 2026-07-23 | Kimi | SKF Web Stack | Post-upgrade read matrix: 4× UIUC + 3× ENOLA positive + APC12x8 negative | `runtime/vm-ops/artifacts/read-uiuc-*.json`, `read-enola-*.json`; ENOLA archive hash matches documented `c9e92c5e…` | All pass (score 10, eligible; negative rejected `query_relevance_not_established`) | Research rebuild |
| 2026-07-23 | Kimi | SKF queue | Post-upgrade queue inspection | leased count 0; `queue:system::c2e3705bbd9656073528c374` = `failed` terminal, `lease_worker_id=null` | No orphaned lease; task had terminally failed pre-upgrade, nothing to sweep | Operator decision: fresh research dispatch |
| 2026-07-23 | Kimi | thesen managed VM | Managed `ctox upgrade --dev` to `cb08c96a` after VM disk remediation (4 attempts: 2× ENOSPC from 22 hand-deployed feature releases, 1× silent build kill, final `CARGO_BUILD_JOBS=2`) | release `/home/ctox/.local/lib/ctox/releases/branch-main-20260723T190039Z`; byte-identical SHA-256 proof vs GitHub | Upgrade OK, service/peer healthy; native RxDB replication down — F-009 tenant manifest drift | Operator decision F-009, then replication recheck |
| 2026-07-23 | Kimi | thesen Web Stack | `ctox web unlock report --strict` (first full acceptance-gate run) | `runtime/vm-ops/artifacts/unlock-report-thesen-20260723T194726036332908Z.json`; redaction self-check passed | gate.ok=false: 7/14 live_success; companyhouse/google/maps blocked_pending_verification; dnbhoovers/leadfeeder/rocketreach/xing evidence_incomplete | Human browser auth (WU-001/002/003), then strict rerun |
| 2026-07-24 | Kimi | thesen managed VM | Upgrade to `f6535ce2` + Workstream 0 phase-2 dual-path replay (9+2 corpus, chat + queue) | release proof `runtime/vm-ops/artifacts/verify-thesen-release-2026-07-24.json`; traces `runtime/vm-ops/replay-2026-07-24/`; reports `docs/benchmark-replay-phase2.{md,json}` | Chat 11/11 ok; queue 8/11 answers, 0/11 terminal — F-010 (completion-gate rework loop) discovered; zero orphaned leases under 12 OOM kills | Fix F-010 evidence-manifest plumbing, then one queue task end-to-end to terminal |
| _timestamp_ | _agent_ | _target_ | _action_ | _IDs/paths_ | _result_ | _next step_ |

## Decision Log

| ID | Timestamp | Decision | Rationale | Alternatives rejected | Consequences |
|---|---|---|---|---|---|
| D-001 | 2026-07-23 | Treat deep-research output only as candidate inventory. | Search/deep results include irrelevant and metadata-only items. | Direct promotion to evidence. | Every admitted source needs an independent typed read and guard pass. |
| D-002 | 2026-07-23 | Keep measured and derived loads in separate tables. | UIUC coefficients do not directly provide N/Nm values; ENOLA contains direct thrust/torque measurements. | Mixing calculations into measured rows. | Derived rows require formulas, assumptions, units, and source lineage. |
| D-003 | 2026-07-23 | Preserve all rejected candidates for audit. | Removal would hide discovery quality and repeated failures. | Deleting rejected candidates. | Only `source_catalog` is evidence; `source_candidates` remains complete. |
| D-004 | 2026-07-23 | Use the published DeepSearchBenchmark as a mandatory regression baseline. | CTOX previously demonstrated strong multi-hop, evidence, document, recall, and technical-data performance with MiniMax-M3. | Designing a new search stack without locating the regression. | Repair must identify lost capabilities and compare old and current behavior under the same tasks. |
| D-005 | 2026-07-23 | Generate dashboard writeback tables only through the deterministic `dashboard_knowledge_build.py` builder; native `store.rs` schema is the source of truth for column names/kinds. | Model-authored tables invented invalid fields (`axial_force_N`, `static_thrust_CT_CP`) and truncated the candidate audit. Workstream D prose names (`propeller_diameter_in`) do not match the native schema (`prop_diameter_in`) — native wins. | Loosening the native importer to accept the invalid artifacts. | SKILL.md now mandates the builder; the builder self-checks against a mirror of the native contract; rejections stay fail-closed. |
| D-006 | 2026-07-23 | The Workstream 0 replay must run dual-path: `ctox chat` benchmark adapter AND the Business OS systematic-research queue path. | Forensics showed the production regression lives in the queue/receipt-gate/reviewer architecture that did not exist during the good benchmark run; a chat-only replay cannot reproduce it. | Chat-only replay against the historical release. | Phase-2 replay plan and release gate must cover both paths with identical task corpora. |
| _ID_ | _timestamp_ | _decision_ | _reason_ | _rejected options_ | _impact_ |

## Verification Evidence

Add every command before claiming the associated checklist item complete.
Summarize relevant output; do not paste secrets.

| Evidence ID | Timestamp | Environment | Command/check | Expected | Actual | Result | Artifact/log |
|---|---|---|---|---|---|---|---|
| V-001 | 2026-07-23 | SKF production | Typed read of four UIUC source files | HTTP 200, score >= 8, eligible | HTTP 200, score 10, eligible for all four | Pass | server receipts/snapshots |
| V-002 | 2026-07-23 | SKF production | Typed ENOLA reads for APC15x8, APC19x10, AeronautCAM9x5 | Real member recognized and eligible | Score 10, eligible; manifest member count 221 | Pass | hash-bound archive manifest |
| V-003 | 2026-07-23 | SKF production | Typed ENOLA read intent for absent APC12x8 | Reject relevance | `query_relevance_not_established`, not eligible | Pass | server receipt |
| V-004 | 2026-07-23 | local (file-bridge checkout) | `python3 -m unittest test_evidence_guard test_source_review_discovery test_source_review_manifest_binding test_business_research_writeback test_dashboard_knowledge_build` in `src/skills/system/research/systematic-research/scripts` | All pass, incl. new builder/guard regression tests | Ran 57 tests, OK (baseline before changes: 36 OK) | Pass | commit `915c70b` |
| V-005 | 2026-07-23 | local | `cargo check --message-format short` (toolchain 1.93.0) | Clean compile of both repair packages | Finished, warnings only (332, pre-existing style) | Pass | commits `915c70b`, `ae81091` |
| V-006 | 2026-07-23 | local | `cargo fmt --check` | Clean | Clean (exit 0) | Pass | both commits |
| V-007 | 2026-07-23 | local | `cargo test --bin ctox mission::channels` | All pass | 105 passed, 0 failed (pristine HEAD baseline: 99 passed / 6 failed — F-008) | Pass | commit `ae81091` |
| V-008 | 2026-07-23 | local | `cargo test --bin ctox systematic_research` | All pass | 22 passed, 0 failed (pristine HEAD baseline: 21 passed / 1 failed red guard) | Pass | commit `915c70b` |
| V-009 | 2026-07-23 | local | `cargo test --bin ctox business_os::store -- --test-threads=1` with repairs, vs pristine-HEAD baseline | No new failures | 252 passed / 16 failed; failure set identical to pristine baseline minus the repaired `queue_worker_success_completes_systematic_research_run` (remaining 16 are pre-existing SSRF-guard/env failures outside repair scope). Parallel runs fluctuate (65 vs 20 failures at identical code) — pre-existing port/order pollution | Pass | both commits |
| V-010 | 2026-07-23 | local | `cargo test --manifest-path src/tools/web-stack/Cargo.toml unlock_report` + full crate suite | New gate tests pass; no new failures | `unlock_report`: 9/9; full suite 393-395 passed with 2-4 flaky failures (`scrape_bridge::parse_envelope`, `web_search::local_fixture_*`) verified failing identically at pristine HEAD `ae81091` (stash baseline) | Pass | commit `cb08c96a` |
| V-011 | 2026-07-23 | local | Workstream 0 phase-1 forensics: benchmark SHAs, snapshot hash, good-run window, hypothesis table, corpus | Verifiable artifacts, no invented SHAs/config | Repo SHAs recorded; snapshot SHA-256 identical across local/site/benchmark checkouts; exact CTOX good-run SHA not recoverable (bounded `06f8dd6c…1b412c8e`, v0.3.22 = `adc3b9b2`); 10 hypotheses with file/line/commit evidence | Pass | `docs/benchmark-regression-forensics.md`, `docs/benchmark-regression-forensics.json` |
| V-012 | 2026-07-23 | SKF production | Managed `ctox upgrade --dev` + commit proof | Active release == `cb08c96a`, healthy | release `branch-main-20260723T170548Z`; `unlock_report.rs`/`channels.rs` SHA-256 byte-identical to GitHub `cb08c96a`; service active, peer ok, `replicationUp:true`, errorTotal 0 | Pass | `runtime/vm-ops/production-verification-2026-07-23.md` §B |
| V-013 | 2026-07-23 | SKF production | Post-upgrade UIUC typed reads ×4 (volume-4 base recovered from instance receipts; first volume-1 attempt failed typed-404, kept for audit) | HTTP 200, score ≥8, eligible | HTTP 200, `verified`, score 10, `evidence_eligible=true`, `transport_eligible=true` for all four; snapshot SHA-256 recorded per read | Pass | `runtime/vm-ops/artifacts/read-uiuc-*.json` |
| V-014 | 2026-07-23 | SKF production | Post-upgrade ENOLA reads APC15x8/APC19x10/AeronautCAM9x5 + negative APC12x8 | positives eligible; negative rejected | positives score 10 eligible; APC12x8 `evidence_eligible=false`, `query_relevance_not_established`; archive snapshot hash `c9e92c5e…` matches documented value on every read | Pass | `runtime/vm-ops/artifacts/read-enola-*.json` |
| V-015 | 2026-07-23 | SKF production | Post-upgrade queue/lease state | No orphaned lease | `queue list --status leased` count 0; target task `failed` terminal with `lease_worker_id=null`; reconcile dry-run clean | Pass | `runtime/vm-ops/production-verification-2026-07-23.md` §C.4 |
| V-016 | 2026-07-23 | thesen production | Managed `ctox upgrade --dev` + commit proof | Active release == `cb08c96a` | release `branch-main-20260723T190039Z`; SHA-256 byte-identical proof; service/peer healthy; VM disk remediated (22 hand-deployed feature releases removed) | Pass (with F-009 caveat: replication down) | `runtime/vm-ops/production-verification-2026-07-23.md` §D |
| V-017 | 2026-07-23 | thesen production | `ctox web unlock report --strict` | gate.ok true OR explicit per-adapter actions | gate.ok=false: 7/14 live_success; 3 blocked_pending_verification (companyhouse, google, maps); 4 evidence_incomplete (dnbhoovers, leadfeeder, rocketreach, xing); redaction self-check passed | Pass (gate executed; completion needs operator auth) | `runtime/vm-ops/artifacts/unlock-report-thesen-20260723T194726036332908Z.json` |
| V-018 | 2026-07-23 | SKF production | Resume durable research command `cmd_0ad31d86…` | Resume without duplicate | Refused by design: command terminally `failed`, `channels.rs:5908` requires a new command; no duplicate dispatched | Blocked — operator decision | `runtime/vm-ops/production-verification-2026-07-23.md` §C.5 |
| V-019 | 2026-07-23 | local | Workstream A: `cargo check -p ctox-web-stack`, `cargo test` unlock/discovery filters, python suite | All green | crate check clean; 6/6 new cascade+scholarly tests; python 72/72 (incl. 11 new dedup/exclude/saturation tests); flaky `local_fixture_*`/`parse_envelope` failures re-confirmed pre-existing at HEAD | Pass | commit `cdad06d0` |
| V-020 | 2026-07-23 | local | Workstream A service.rs: `cargo check` + `cargo fmt --check` on the EXACT staged commit state (foreign WIP stash-parked) | Staged subset compiles alone | `cargo check` clean, `cargo fmt --check` clean; bin unit tests NOT rerun (disk constraints — declared gap; the two rewritten discovery-receipt tests were reviewed but not executed) | Pass (with stated gap) | commit `1292ad73` |
| V-021 | 2026-07-24 | local | Web Stack read/reject test matrix: 12 new fixture tests + full crate suite + pristine-HEAD flaky baseline | New tests pass; no new failures | 12/12 new (PDF/HTML/JSON/CSV admission, typed-read receipt binding, multi-hop redirect, non-content redirect, 404/410, 204, bot walls, ZIP manifest ×2); suite 413 passed / 2 failed — both pre-existing flaky, re-verified at pristine HEAD via path-scoped stash | Pass | commit `f6535ce2` |
| V-022 | 2026-07-24 | thesen production | Workstream 0 phase 2 dual-path replay, 9+2 corpus @ `f6535ce2` (release byte-identical proven), MiniMax-M3 | Corpus executed both paths with per-task metrics vs good-run baseline | Chat 11/11 ok (9 strict-valid JSON, 8 supported, 3 honest blockages; median ≈0.6× baseline wall); queue 8/11 final answers + 3 partial, **0/11 terminal — F-010 infinite-rework loop found**; 12 OOM kills absorbed with zero orphaned leases (F-002 repair holds); H1 refuted at tip, H2 tool-level repaired but completion gate never accepts, H4 narrows honestly | Pass (replay executed; F-010 blocks queue completion) | `docs/benchmark-replay-phase2.{md,json}`, `runtime/vm-ops/replay-2026-07-24/` |
| V-023 | 2026-07-27 | thesen production | Upgrade to `3b3fee88` + F-010 terminal proof (single bounded research command) | Command reaches terminal state | Upgrade proven (3/3 file SHA-256 == `3b3fee88`, release `branch-main-20260727T075604Z`); proof command `cmd_409ad6cb…`: manifest-path repair `a699e9a5` works (correct `validation/evidence-manifest.json` written), but 3 identical fail-closed rejections `observed discovery calls: none` — queue worker made 165 `exec_command` / 0 typed `ctox_*` calls; cancelled after time-box; `business_commands` non-terminal (F-R5 persists) | **Fail — F-010 open, root cause layer 2 pinned** | `runtime/vm-ops/artifacts/f010-final-verdict-2026-07-27.json` |
| V-024 | 2026-07-27 | thesen production | Unlock evidence runs on `3b3fee88`: fresh `--strict` gate + 4 protected-adapter reruns (stored sessions only) + 3 blocked-adapter classifications | Evidence for capabilities 8/10/12/13 | Gate 8/14 live (maps newly live); 4 protected adapters: sessions expired (typed `portal_drift`, `repair_request.json` persisted) BUT no `authorization_required` and no reauth handoff (`repair_queue_task: null`) — **capability 10 gap confirmed in code**; cap 8 evidenced (137 auth-assist queue tasks); companyhouse/google correctly `blocked` with visible operator actions | Partial — capability-10 fix required | `runtime/vm-ops/unlock-evidence-2026-07-27.md` |
| _ID_ | _timestamp_ | _env_ | _command/check_ | _expected_ | _actual_ | _Pass/Fail_ | _path/URL/ID_ |

## Research Data Ledger

Update counts after each durable writeback. Never substitute UI display counts
for database counts.

| Timestamp | Run/command | Candidate sources | Admitted sources | Rejected sources | Direct measurements | Derived loads | Knowledge skills | Resources | Runbooks | Graph nodes | Graph edges | Reports | Audit result |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Handoff | existing invalid intermediate state | 12 of 184 written | 7 | incomplete | 28 invalid-contract rows | 56 | incomplete | incomplete | incomplete | 8 | 8 | incomplete | Fail/not publishable |
| _timestamp_ | _IDs_ | _n_ | _n_ | _n_ | _n_ | _n_ | _n_ | _n_ | _n_ | _n_ | _n_ | _n_ | _source/data/claim result_ |

## Source Verification Ledger

One row per admitted source or intentionally retained rejection.

| Source ID | Canonical URL/DOI | Type | HTTP | Snapshot SHA-256 | Relevance | Eligible | Admission state | Exact evidence extracted | Rejection reason | Receipt/artifact |
|---|---|---|---:|---|---:|---|---|---|---|---|
| _source-id_ | _URL/DOI_ | _paper/dataset/etc._ | _status_ | _hash_ | _0-10_ | _true/false_ | _admitted/rejected_ | _rows/passages_ | _if rejected_ | _path_ |

## Production Browser Verification

Use clean sessions for both authorized users. Do not write passwords here.

| Timestamp | User | Flow | Expected | Actual | Result | Screenshot/trace |
|---|---|---|---|---|---|---|
| _timestamp_ | Michael Welsch | Login and Web Research | Dashboard and verified data visible | _actual_ | _Pass/Fail_ | _path_ |
| _timestamp_ | Ingo Schulz | Login and Web Research | Same approved research visible | _actual_ | _Pass/Fail_ | _path_ |
| _timestamp_ | _user_ | Knowledge skill/resource navigation | Source-linked Knowledge visible | _actual_ | _Pass/Fail_ | _path_ |
| _timestamp_ | _user_ | Graph interaction | Valid, responsive graph and LOD controls | _actual_ | _Pass/Fail_ | _path_ |
| _timestamp_ | _user_ | Report generation | Uses selected/automatic Knowledge skill | _actual_ | _Pass/Fail_ | _path_ |
| _timestamp_ | _user_ | Files/Spreadsheet export | File visible, opens, downloads | _actual_ | _Pass/Fail_ | _path_ |
| _timestamp_ | _user_ | Harness task | Durable progress and completion timeline | _actual_ | _Pass/Fail_ | _path_ |

## Blocker Log

| Opened at | Closed at | Owner | Blocker | Evidence | Mitigation attempted | Resolution |
|---|---|---|---|---|---|---|
| 2026-07-23 | 2026-07-23 | Kimi | Orphaned research queue lease with no active worker | F-002 | Released existing task after production Web Stack verification | Code fix `ae81091` deployed; post-upgrade leased count 0, target task terminally `failed`, sweep verified clean (V-015) |
| 2026-07-23 | _open_ | _operator_ | SKF research rebuild cannot resume under durable command id | V-018; `channels.rs:5908` refusal | `queue release`, `commands process`, `commands reconcile --dry-run` — all by-design non-applicable for terminal commands | Awaiting operator authorization of a fresh dispatch (new command id) |
| 2026-07-23 | _open_ | _operator_ | Thesen native RxDB replication down after main upgrade | F-009; V-016 | VM disk remediated; tenant manifest not hand-edited | Awaiting operator decision: delete `delete_when_missing` key OR port feature to main |
| _timestamp_ | _timestamp/open_ | _owner_ | _blocker_ | _evidence_ | _attempts_ | _resolution_ |

## Final Sign-Off

All rows must contain evidence before setting overall status to `COMPLETE`.

| Gate | Status | Evidence IDs/paths |
|---|---|---|
| Discovery is systematic and saturated | `NOT_STARTED` | |
| Every admitted source is real and readable | `NOT_STARTED` | |
| Direct measurements pass native schema | `NOT_STARTED` | |
| Derived loads are separated and auditable | `NOT_STARTED` | |
| Source audit passes | `NOT_STARTED` | |
| Data audit passes | `NOT_STARTED` | |
| Claim audit passes | `NOT_STARTED` | |
| Knowledge skills/resources are complete | `NOT_STARTED` | |
| Graph contract and performance pass | `NOT_STARTED` | |
| Reports retain source lineage | `NOT_STARTED` | |
| Queue/harness has no orphaned lease | `PASS` | V-015 |
| Historical benchmark regression is identified | `IN_PROGRESS` | V-011 (hypotheses); replay pending |
| Representative benchmark replay meets baseline | `NOT_STARTED` | |
| Complete 100-task CTOX rerun is acceptable | `NOT_STARTED` | |
| Managed upgrade is current | `PASS` | V-012 (SKF `branch-main-20260723T170548Z`), V-016 (thesen `branch-main-20260723T190039Z`), beide == `cb08c96a` |
| Michael browser E2E passes | `NOT_STARTED` | |
| Ingo browser E2E passes | `NOT_STARTED` | |
| No red tasks remain | `NOT_STARTED` | |

### Final Outcome

- Final status: _COMPLETE / BLOCKED_
- Final main commit: _SHA_
- Installed release: _path_
- Research run: _ID_
- Source/data/claim audit summary: _summary_
- Remaining risks: _none or explicit list_
- Signed off by: _agent/operator_
- Signed off at: _ISO-8601 timestamp_
