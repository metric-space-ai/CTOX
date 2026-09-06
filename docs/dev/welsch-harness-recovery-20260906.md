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

## Production verification

Commit `4b31732237492df9219070a3aaa7bf2fd89c92bf` is on main. Release run
`34018635068` passed, including the new rendering gate. Signed candidate
`0.1.46-beta.5` was staged and activated on Welsch at 07:18:01 UTC, then the
regular service was restarted. The instance reports current/healthy and no
recovery shell. This shell release preserves the preceding Office change.

The authenticated in-app browser displays the candidate version, 16 harness
nodes, five real task records, timeline and connection status. All 20 measured
task selections selected the requested record; each details drawer was closed
before the next selection. Location highlighting pointed to model-failed for
these failed historical tasks. A first invalid measurement attempted the next
card while the preceding details drawer overlaid it; it is excluded from the
20 completed interactions and retained as a test-procedure finding.

The same profile was reloaded and again showed 16 nodes and five tasks. No new
standortNodeId exception was observed. The original user tab also recovered.
This is not a clean-profile authentication or worker-execution acceptance.

### Observed performance

Clock: automation-host wall time from before the browser click to the awaited
DOM assertion. Includes browser-control round trips; it is not isolated render
time, INP, or native command latency. Twenty verified selections, nearest-rank
percentiles: p50 529 ms, p95 1562 ms, maximum 2847 ms.

Samples in milliseconds:
`375,475,517,765,1562,2847,916,769,427,428,403,414,414,486,529,742,711,738,987,653`.

One authenticated reload took 26051 ms until the candidate version, 16 nodes
and five tasks were observed. This includes a Page.getFrameTree observation
timeout. It is an observed upper bound with instrumentation interference,
not a statistically valid boot p95 or a measurement of critical collection
readiness. The initial post-deployment reload also temporarily stopped answering
DOM queries. These failures remain part of the acceptance evidence.

### Open failures

The spreadsheet still reports that its selected version cannot be loaded;
Web Research previously reported invalid_graph_contract. The overall reload
latency remains unresolved. Neither the local warm-command p50 target nor the
critical-collection boot p95 target has been verified by this UI measurement.
Native four-process admission, SSH/QR onboarding, real harness failover and the
remaining architecture migration requirements remain open. There is no
production-readiness claim for the Sync architecture or other apps.
