Main meta-skill: `ops-debug-terminal`

Use inside CTOX:

- Read the debug goal from `Focus`.
- Use `Verified evidence`, `Anchors`, and `Governance` as the runtime context for the investigation.
- This phase stays read-only until the evidence is narrow enough to justify re-classifying into a write phase.

- Stay read-only until the likely cause is narrow enough.
- Capture symptom, time window, and affected target first.
- Compare healthy and failing state when possible.
- Narrow one layer at a time: app, config, dependency, database, network, deploy, or resources.
- Keep evidence for each finding.
- Stop before write actions and re-classify if a write is needed.
