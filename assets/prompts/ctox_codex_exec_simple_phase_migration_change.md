Main meta-skill: `migration-change`

Use inside CTOX:

- Read the schema goal and done gate from `Focus`.
- Respect `Anchors`, `Verified evidence`, and rollback boundaries.
- Keep migration work separate from unrelated open work in `Workflow state`.

- Keep one schema change per step if possible.
- Keep schema work separate from backfill work.
- Use the repo migration path, not a guessed command.
- Run the smallest migration check you can.
- Then update the nearest app code and app checks.
- Stop if rollback, data handling, or destructive impact is unclear.
