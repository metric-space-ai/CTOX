Main meta-skill: `data-change-safe`

Use inside CTOX:

- Read the target change and done gate from `Focus`.
- Use `Verified evidence` and `Anchors` to define the exact safe target set.
- Keep runtime open-work rules from `Workflow state` intact while changing data.

- Start read-only.
- Define the exact target set before any write.
- Count before write, then dry-run if possible.
- Prefer idempotent scripts and small batches.
- Verify counts and sample rows after the change.
- Stop if the write scope or target set is unclear.
