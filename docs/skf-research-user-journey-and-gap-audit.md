# SKF Research User Journey and Gap Audit

## Product Promise

The SKF workspace must expose one coherent, living value chain:

1. Web Research discovers and verifies original sources.
2. Research evidence becomes substantive Knowledge Skills and reusable Runbooks.
3. Documents use an explicitly selected or automatically matched Skill.
4. Reports and exports remain linked to their sources and are available in Files.
5. Updated evidence deterministically refreshes or invalidates downstream content.

Internal prompts, storage paths, queue metadata, and implementation contracts are not
primary user content.

## Two-Second Visual Gate

A release fails when a user can identify any of these conditions without interacting:

- clipped titles, tabs, panes, controls, or cards;
- overlapping graph controls, legends, insights, or search;
- an endless graph start animation;
- duplicate Research entries for one domain lineage;
- a Knowledge view that shows only a registry sentence instead of the selected Skillbook;
- a verified-source ranking led by lower audited grades while higher grades exist;
- empty Reports when a current verified report exists;
- a right pane dominated by an internal prompt or implementation instructions.

## Expected Journey

### Workspace Start

- The shell reaches a usable state or shows a bounded, actionable error.
- Web Research, Knowledge, Documents, Files, and Spreadsheets open consistently as
  windowed applications.
- Upgrade or synchronization activity is visible and cannot masquerade as empty data.

### Web Research

- One card represents the current Research lineage for a Knowledge domain.
- Historical runs remain accessible as history, not duplicate primary cards.
- Counts distinguish candidates, verified sources, measurements, Knowledge, and reports.
- Starting or extending research creates a visible task and progress state.
- Existing verified sources are passed as an exclude list for subsequent discovery.

### Sources and Ranking

- Only sources with a successful original read, canonical URL, snapshot, receipt, and
  hash are ranked as verified evidence.
- The visible grade comes from the audited evidence tier.
- Sorting is grade-first and score-second.
- Original links open a real source, not a discovery URL or metadata-only page.
- Rejected candidates remain auditable but never appear as verified evidence.

### Research Graph

- The graph starts stationary and fitted to the available viewport.
- Search, detail level, 2D/3D mode, fit, layers, and insights have distinct,
  non-overlapping locations.
- Opening insights reduces or reflows the graph canvas.
- Overview, standard, and deep detail produce meaningful bounded changes.
- Nodes represent technical concepts and verified evidence relations, not arbitrary
  prompt vocabulary.
- Invalid graph provenance fails closed with an actionable explanation.

### Knowledge

- Selecting a domain exposes every available Skillbook through a secondary switcher.
- Selecting a Skillbook displays its full Markdown, mission, policy, answer contract,
  rules, workflow, and linked Runbooks.
- A generic source-registry Skill cannot silently replace the selected Skillbook.
- Runbooks contain repeatable engineering procedures, not one-line summaries.
- Tables expose units and explanatory header help.
- Skills and claims retain source and evidence identifiers.

### Documents and Reports

- Research exposes current verified reports linked to the active domain lineage.
- Creating a document selects or recommends a relevant Skill.
- Documents retain citations through editing and export.
- Stale or quarantined reports do not appear as current deliverables.

### Files and Spreadsheets

- Generated artifacts appear in Files with creation and modification sorting.
- CSV files open in Spreadsheets; document formats open in Documents.
- Upload, download, drag-in, and drag-out are available where the host permits them.
- The UI reports the final file location and offers a direct open/download action.

## Pane Responsibilities

### Left Pane

Domain lineages, current status, and evidence ranking. It must remain scannable and
must not duplicate historical runs as separate active domains.

### Center Pane

The current work product: graph, source registry, measurements, Knowledge, or report.
Its controls must remain inside the pane at every supported viewport.

### Right Pane

Concise context and next actions:

- domain description;
- verified counts and current run status;
- build/update Knowledge;
- create/open a report;
- selected source facts and original-source action.

It must not display a long system prompt, local filesystem paths, hidden workflow
policy, or a wall of scoring internals by default.

## Responsive Contract

- Every grid child uses `min-width: 0`.
- Titles wrap to a bounded number of lines and never push controls out of view.
- Compact layouts collapse columns deliberately instead of clipping them.
- Graph overlays reserve canvas space or reflow below it.
- Touch layouts keep controls at usable sizes without hiding content.
- No horizontal overflow is accepted at supported desktop, compact, and touch widths.

## Current Repair Work Log

The statuses below are deliberately split into code, deployment, and live acceptance. A
unit test or source-code change is not evidence that the production UI works.

| Area | Failure | Required Result | Code | Live acceptance |
| --- | --- | --- | --- | --- |
| Knowledge projection | Summary only was replicated | Full Skill, Skillbook, and Runbook Markdown | Implemented; module tests pass | Pending managed upgrade and browser verification |
| Knowledge selection | Generic registry Skill replaced selected Skillbook | Explicit Skillbook switcher and selected Skillbook content | Implemented | Pending |
| Knowledge sources | Verified receipts were not visible in Knowledge | Counted source tab with receipt, canonical URL, and snapshot hash | Implemented | Pending |
| Research lineage | Two cards for one domain | One current lineage with historical task IDs | Implemented; module tests pass | Pending |
| Ranking | Computed score overrode audited tier | Audited grade-first ranking | Implemented; module tests pass | Pending |
| Graph motion | Graph restarted or animated on hover | Stationary graph that preserves simulation state | Implemented | Pending pointer and resize tests |
| Graph layout | Controls, insights, labels, and search overlapped | Reserved responsive regions with no clipping | Implemented | Pending viewport matrix |
| Right pane | Internal prompt text wall | Concise context, facts, status, and actions | Implemented | Pending |
| Reports | Current report not linked across lineage | Explicit task/domain lineage and current verified report | Implemented | Pending open/export test |
| Knowledge depth | Native records appeared shallow | Substantive verified Skill and executable Runbooks | Six typed engineering Runbooks imported | Pending full-content and execution test |

## Complete Defect And Acceptance Matrix

### Web Research layout and interaction

- [ ] The application remains a windowed module and never renders an empty body.
- [ ] Header, module tabs, pane titles, counts, filters, and actions are not clipped at
  1920, 1440, 1280, 1024, 768, and mobile widths.
- [ ] The three-pane layout collapses only when the available module width requires it;
  it does not switch prematurely because of the outer browser width.
- [ ] Source cards do not clip titles, grades, metadata, or actions.
- [ ] Source-table columns have stable widths and never overlap.
- [ ] Source and measurement tables scroll horizontally when required.
- [ ] Independent pane scroll positions survive selection, refresh, tab changes, and
  background replication updates.
- [ ] Selecting a ranked source on the left selects and reveals the same source in the
  center and renders it in the right pane.
- [ ] Selecting a source in the center updates the left ranking and right details.
- [ ] Tag filters produce a visibly filtered result set and can be reset with `Alle`.
- [ ] Source cards contain a factual summary, concrete contribution, limitations, and
  provenance; generic audit boilerplate is not accepted as content.

### Graph

- [ ] The graph initializes once, fits once, and does not restart on hover.
- [ ] Pointer movement, node hover, selection, resize, detail changes, and pane
  navigation preserve the graph instance unless the graph data changes.
- [ ] The graph becomes stationary after bounded layout stabilization.
- [ ] Search, layers, fit, detail, 2D/3D, zoom, insights, and legend never overlap.
- [ ] Overview, standard, and deep modes produce meaningful, bounded graph changes.
- [ ] Node labels, topics, and relations are domain concepts derived from evidence.
- [ ] Invalid graph provenance fails closed instead of rendering invented relations.

### Research data and evidence

- [ ] The current lineage shows 1,643 candidates, 138 admitted sources, 4,177 direct
  measurements, 3,925 separate derivations, 8 Knowledge entries, and 1 current report,
  unless a newer audited run intentionally changes those counts.
- [ ] Direct measurements expose the original reported quantities and source row.
- [ ] UIUC coefficient rows expose `CT` and `CP` as direct, dimensionless measurements.
- [ ] Derived thrust, torque, and power use a separate view and state the assumed air
  density and formula.
- [ ] Propeller diameter and pitch are separate metric columns with explanatory headers.
- [ ] Numerical exports remain machine-readable; German display formatting does not
  corrupt stored numeric values.
- [ ] Every admitted source opens a valid canonical original or captured primary
  artifact and exposes receipt and SHA-256 provenance.
- [ ] Rejected or quarantined candidates never appear as verified evidence.
- [ ] Ranking is audited-grade first and score second; A/B evidence is visible before
  lower tiers.

### Task and Harness flow

- [ ] `Research fortsetzen` creates exactly one continuation task.
- [ ] The task immediately becomes visible in Chat/CTOX and receives focus.
- [ ] Repeated clicks cannot create duplicate active runs.
- [ ] The Harness leases, runs, reviews, and completes the task with MiniMax M3.
- [ ] The Responses proxy streams long-running work without truncation.
- [ ] No red, blocked, orphaned, or stale task remains after completion.
- [ ] Research progress and upgrade/synchronization states cannot look like empty data.

### Knowledge

- [ ] The active domain displays a substantive UAV motor-bearing design Skill, not a
  source-registry sentence.
- [ ] The Skill defines purpose, required inputs, engineering method, assumptions,
  evidence policy, decision gates, outputs, limitations, and citations.
- [ ] Every Runbook is an executable specialization with required inputs, ordered
  actions, validation gates, artifacts, failure conditions, and an execution command.
- [ ] Runbook execution creates a visible CTOX task and a traceable output.
- [ ] The six required procedures cover load points, bearing reactions, life spectrum,
  clearance/preload, lubrication/sealing, and validation.
- [ ] Skill, Runbook, Sources, and Tables switchers are labelled, horizontally
  scrollable where needed, and keyboard accessible.
- [ ] Physical replication chunks are merged into logical tables; users do not see
  hundreds of cryptic one-letter chunk tabs.
- [ ] Incomplete chunk sets show recovery/retry status and do not silently masquerade
  as complete Knowledge.
- [ ] Table headers expose units and concise hover explanations.

### Documents, reports, files, and spreadsheets

- [ ] The current verified report is visible and opens from Research.
- [ ] Document creation automatically recommends or explicitly accepts the UAV-bearing
  Skill and a relevant Runbook.
- [ ] Generated documents retain evidence citations through edit and export.
- [ ] Generated CSV and document files appear in Files with direct open and download.
- [ ] CSV opens in the actual Spreadsheets application; documents open in Documents.
- [ ] Upload, download, drag-in, and drag-out work in browser and desktop hosts where
  the platform permits them.

### Account and session acceptance

- [ ] A fresh Michael session logs in, reloads, uses all required modules, and logs out.
- [ ] A fresh Ingo session does the same with the Owner role.
- [ ] Exactly those two active users exist.
- [ ] Settings render completely for both authorized roles.
- [ ] No anonymous `Local CTOX` identity or local-runtime fallback appears on the
  managed tenant.

## Release Gate

Deployment is allowed only after:

- JavaScript and Rust checks pass;
- browser screenshots pass desktop, compact, and touch overlap checks;
- the live tenant shows one Research lineage;
- full Knowledge content is visible;
- audited A/B sources lead the ranking;
- a current verified report is visible;
- no red task remains;
- both authorized users pass a fresh-session end-to-end walkthrough.
