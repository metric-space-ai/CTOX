Meta-skill: `task-router`

Use inside CTOX:

- Read the current task and done gate from `Focus`.
- Read checked facts from `Verified evidence`.
- Read durable constraints from `Anchors`.
- This router chooses the working method only. It does not change CTOX policy or runtime state rules.

- Choose one main phase only.
- If the task is unclear, use `read-trace-first`.
- If the phase changes, stop and classify again.
- Do not mix two main phases in one step.
- If a dedicated mixed phase fits, use that one mixed phase.
- Otherwise split mixed work into separate phases.
- If the current phase seems only partly right, one small read-only inspect step or one small check is allowed before re-classifying.
- Re-classify before risky, broad, or destructive actions.
