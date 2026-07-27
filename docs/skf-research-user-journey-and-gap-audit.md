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

| Area | Failure | Required Result | Status |
| --- | --- | --- | --- |
| Knowledge projection | Summary only was replicated | Full Skill, Skillbook, and Runbook Markdown | Implemented; 22/22 module tests pass |
| Knowledge selection | Generic registry Skill replaced selected Skillbook | Explicit Skillbook switcher and selected Skillbook content | Implemented; 22/22 module tests pass |
| Knowledge sources | Verified receipts were not visible in Knowledge | Counted source tab with receipt, canonical URL, and snapshot hash | Implemented; 138-resource import prepared |
| Research lineage | Two cards for one domain | One current lineage with historical task IDs | Implemented; 42/42 module tests pass |
| Ranking | Computed score overrode audited tier | Audited grade-first ranking | Implemented; 42/42 module tests pass |
| Graph motion | Continuous directional particle animation | Stationary graph after fit | Implemented; browser QA passes |
| Graph layout | Controls and insights overlapped | Reserved, responsive overlay regions | Implemented; desktop, compact, and touch QA pass |
| Right pane | Internal prompt text wall | Concise context and actions | Implemented; browser QA passes |
| Reports | Current report not linked across lineage | Explicit task/domain lineage and current verified report | Verified report live; two inadmissible legacy reports quarantined |
| Knowledge depth | Several native records appeared shallow | Substantive verified Skills, Runbooks, resources, and tables | Nine full Books and thirteen Runbooks verified; resource promotion pending deployment |

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
