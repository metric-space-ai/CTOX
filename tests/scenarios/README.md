# Scenario Tests

This directory is reserved for heavier multi-slice flows that are still regression tests, not open-ended benchmarks.

Examples:

- mission recovery after repeated blockers
- schedule plus queue plus follow-up interactions
- continuity repair after compaction

Keep these deterministic and bounded so they can still run in normal CI.
