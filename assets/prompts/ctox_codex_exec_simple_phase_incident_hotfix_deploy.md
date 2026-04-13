Main meta-skill: `incident-hotfix-deploy`

Use inside CTOX:

- Read the restore target and required outcome from `Focus`.
- Respect `Anchors`, `Governance`, and the CTOX core owner/safety rules while deploying.
- Keep follow-up work durable in `Workflow state` after the restore.

- Use when service restore needs one small code or config hotfix and an immediate deploy.
- Isolate one symptom and choose the smallest restore path.
- Make one small hotfix only.
- Deploy only that hotfix to the exact target environment.
- Verify symptom, health, version, and recent errors after deploy.
- Keep follow-up work separate from the restore step.
