Main meta-skill: `bug-fix`

Use inside CTOX:

- Read the current bug-fix target and done gate from `Focus`.
- Use `Verified evidence` for the actual failure, not guesswork.
- Respect `Anchors` and existing `Workflow state` while fixing.

- Reproduce with the smallest failing check if possible.
- Change the closest code to the failure.
- Make the smallest fix that matches the evidence.
- Re-run the same failing check first.
- Then run the nearest related checks.
- If the fix turns into API, schema, or security work, stop and re-classify.
