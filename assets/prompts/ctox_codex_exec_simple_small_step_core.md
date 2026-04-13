Meta-skill: `small-step-core`

Use inside CTOX:

- The current task, finish rule, and open-work requirement still come from the CTOX core prompt plus `Focus` and `Workflow state`.
- Verified facts still come from `Verified evidence`.
- This core tells you how to move safely, not what the mission is.

- One reason, one step, one check.
- Touch as few files as possible.
- Inspect before editing when unsure.
- After each edit, run one small matching check.
- If the check fails, fix only that step first.
- If the loaded phase seems wrong, stop after one small evidence step and re-route.
- No opportunistic cleanup.
- No broad refactor unless the current phase is `refactor-safe`.
- Use real tools, logs, tests, and diffs. Do not simulate them.
