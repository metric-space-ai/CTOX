Main meta-skill: `deploy-release`

Use inside CTOX:

- Read the target environment and required outcome from `Focus`.
- Respect `Anchors`, `Governance`, and owner-channel rules from the CTOX core prompt.
- Deployment status alone does not replace CTOX runtime-state requirements for open work.

- Confirm environment first.
- Confirm the exact version, artifact, image, or commit first.
- Use the existing deploy path, not a guessed command.
- Run pre-deploy checks, then deploy one target at a time.
- Verify health, version, and smoke behavior after deploy.
- Stop if environment, version, or deploy output is unclear.
