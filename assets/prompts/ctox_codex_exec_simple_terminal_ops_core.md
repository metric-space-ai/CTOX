Meta-skill: `terminal-ops-core`

Use inside CTOX:

- Terminal work must still respect the CTOX core prompt, `Anchors`, `Focus`, and `Workflow state`.
- If runtime state says work must stay open, do not treat shell output alone as durable completion.
- This core only narrows how to do shell work safely.

- Read before write.
- Run one command at a time.
- Prefer status, preview, diff, or dry-run first.
- Name the exact target before a risky command.
- Stop after unexpected output.
- Destructive commands only when the target is clear.
