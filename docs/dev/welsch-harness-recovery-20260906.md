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

## Subsequent release change during acceptance

At 07:29:17 UTC, a separate deployment activated 0.1.46-beta.6 and restarted
CTOX (PID 2362225). Its source f823420f1 contains the Harness fix plus an Office
source-synchronization change; release run 34019072184 succeeded. We did not
activate that release. A follow-up browser reload did not prove beta.6 loaded:
after the 40-second observation window the UI subsequently rendered 16 nodes
but still displayed v0.1.46-beta.5. Do not transfer beta.5 measurements or its
UI acceptance to beta.6. The active-versus-browser version discrepancy remains
unresolved. Direct unauthenticated manifest requests return 403; no browser
authentication state was extracted to bypass that boundary. Local 8080/8081
manifest guesses returned 404 and do not identify the Business OS asset server.
The existing `requestTenantBusinessOsAsset` control-plane function subsequently
returned the native `ctox.business-os-shell.v1` manifest with version
0.1.46-beta.6. Thus the instance asset server and its supported SSH forwarding
path expose beta.6; the remaining discrepancy is on the hosted/browser path.

## Further diagnosis: document identity and narrow layout

The native manifest response also carries
`public, max-age=300, stale-while-revalidate=86400`. The inspected ctox.dev
proxy in the older local checkout forwards successful native bodies and cache
policy but masks generation conflicts as 502. The subsequent comparison with
ctox.dev-main ccaec625edd6204cee9056303e196b3fd0ffe2dd found an additional
generation-stripping retry after typed 409 responses: the proxy requests the
active slot without the original query. This violates instance release
authority and must be removed. These findings do not
prove which hosted/browser cache caused the observed stale version label.
New artifacts stamp version and source commit into the signed HTML inventory.
The browser reads this document identity without a second manifest request.
The native manifest policy is changed to `no-store`; a shell-only deployment
does not deploy that native correction. Complete dependency-generation binding
remains open as described in `docs/business-os-shell-releases.md`.

The original narrow browser tab had 16 SVG nodes but a canvas of zero visible
height. Therefore the earlier observation that the original tab recovered
proved DOM recovery only, not usable rendering at that width. The main grid
omitted an explicit progress row and its stacked narrow layout clipped content.
The correction gives progress, canvas, timeline and footer explicit rows and
allows the stacked view to scroll.

The new component browser gate renders the real module and Shell V2 CSS with
synthetic task data. Before the CSS correction it reproduced a 418-by-0 canvas
at a viewport width of 430. Afterwards all five widths passed: 430, 630 and
768 produced canvas heights of 279 px; 1000 and 1280 produced 311.984375 px.
After integrating main e2f952320, all five widths passed again; the two wider
canvases measured 322.21875 px with the updated shared Kit tokens.
Every page contained 16 nodes, no page exceptions, a non-clipped node verified
by hit testing, the stamped document identity and no manifest request.
These checks are now a shell release gate. Artifact tests passed 16/16 and
status tests 5/5. These are supporting checks; they do not replace production
authentication, WebRTC replication, reload or performance acceptance.

At this point these further corrections have not been deployed or accepted
on Welsch. The full native Cargo check is still running. No live success or
performance result from beta.5 is assigned to this new source.
