Main meta-skill: `rollback-recovery`

Use inside CTOX:

- Read the recovery target and safety boundary from `Focus` and `Anchors`.
- Use `Verified evidence` for the actual last known good state.
- Keep post-rollback open work durable in `Workflow state`.

- Identify the last known good state first.
- Use the existing rollback path.
- Roll back one target at a time.
- Verify health, version, and smoke behavior after each rollback step.
- Do not mix rollback with new fixes.
- Stop if the rollback target or data compatibility is unclear.
