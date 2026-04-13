Main meta-skill: `dependency-update`

Use inside CTOX:

- Read the requested version change and boundaries from `Focus`.
- Use `Anchors` and `Verified evidence` to keep the update scoped.
- If follow-up work remains, it still must live in CTOX runtime state, not only in prose.

- Update one dependency or one tight version group at a time.
- Change manifest and lockfile only as needed.
- Run the nearest install, build, or test checks.
- Fix only failures caused by this update.
- Keep version changes separate from feature, refactor, schema, or data work.
