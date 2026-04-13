Main meta-skill: `infra-config-change`

Use inside CTOX:

- Read the intended config behavior from `Focus`.
- Respect `Anchors` and CTOX core safety rules for environment and blast radius.
- If the change leaves open work, keep it in CTOX runtime state.

- Change one config behavior at a time.
- Keep environment and target explicit.
- Prefer validate, lint, diff, template, plan, or dry-run before apply.
- Separate infra change from app code change.
- Keep secrets out of the diff.
- Stop if the environment, blast radius, or validation path is unclear.
