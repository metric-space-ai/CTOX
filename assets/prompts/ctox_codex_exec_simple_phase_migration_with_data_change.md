Main meta-skill: `migration-with-data-change`

Use inside CTOX:

- Read the schema and data goal from `Focus`.
- Use `Verified evidence` and `Anchors` to bound the migration and the target data set.
- Keep rollback and open-work requirements from the CTOX core prompt visible while doing both parts.

- Use when schema work and a tightly coupled backfill or repair step must move together.
- Keep the schema step and the data step explicit, small, and measurable.
- Start data work read-only: identify and count the target set first.
- Run the smallest migration check you can, then the smallest safe data check.
- Verify both schema effect and data effect after each step.
- Stop if rollback, target set, or destructive impact is unclear.
