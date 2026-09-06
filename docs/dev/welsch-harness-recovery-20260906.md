# Welsch Harness rendering recovery — 2026-09-06

## Observed production failure

The authenticated Welsch browser displayed CTOX tasks, but no `.ctox-flow-diagram`
or `.ctox-flow-node-g` elements. Repeated `ReferenceError: standortNodeId is not
defined` exceptions came from `flowSvg`. Commit `5c3287695` declared that value in
`renderMain` and referenced it from the separate `flowSvg` function. The active
0.1.46-beta.4 release preserves this source; its Office fix is retained here.

Move the computation into its consumer, preserving current-task highlighting.
Expose the actual SVG renderer to the existing module harness and exercise it
both without a selection and with a running task. The new check failed before
the fix and passes afterwards. Make this module suite a shell release gate.

The existing crew test still expected all inactive tasks to appear together,
contradicting the intentional selection behavior documented in `5c3287695`.
It now checks all three selections separately, rejects unrelated inactive tasks,
and retains the node, telemetry, progress and status assertions.

## Supporting checks (not end-to-end acceptance)

- CTOX module suite: 29 passing checks; existing duplicate `unterhaltung` key warning.
- Shell V2 static contract: 37/37 apps.
- CTOX geometry: widths 1180, 1000 and 720, 3/3 passing.
- Artifact, inventory and signature checks: 15/15 passing.

Live deployment, authenticated reload, task selection and performance measurements
are pending. Test execution durations are not application performance evidence.
The observed Web Research `invalid_graph_contract` and spreadsheet
`PEER_UNAVAILABLE` are separate unresolved findings. This repair is not a
production-readiness claim for the Sync architecture or other apps.
