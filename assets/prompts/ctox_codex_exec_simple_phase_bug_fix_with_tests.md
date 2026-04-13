Main meta-skill: `bug-fix-with-tests`

Use inside CTOX:

- Read the current fix target and done gate from `Focus`.
- Use `Verified evidence` for the real failure and `Workflow state` for any open follow-up.
- This phase allows one tight test update around the fix, not a broader rewrite.

- Use when a real bug fix and a small test update belong to the same fix step.
- Reproduce with the smallest failing test or command first.
- Change the closest production code first, then the nearest test only if needed.
- Re-run the same failing check first.
- Then run the nearest related tests.
- Do not turn this into a broader refactor or feature change.
