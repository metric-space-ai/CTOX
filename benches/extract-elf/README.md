# Bench: `extract-elf`

Terminal-Bench-2 task. Implement `/app/extract.js` in JavaScript that
parses an ELF binary at `/app/a.out` and outputs a JSON map of memory
addresses → integer values from loaded segments.

- Image: `alexgshaw/extract-elf:20251031`
- Task timeout: `900s` (run with `--agent-timeout-multiplier 3`)
- Verifier: runs `node extract.js /app/a.out`, compares output JSON to
  expected key/value pairs

## Results

| Model | Reward | Turns | tok_in/out | Final reply | Status |
|---|---|---|---|---|---|
| gpt-5.4-mini | **1.0** | 1 | 170 / 49  | "Done — extract.js is in place and extracts loaded 4-byte memory words…" | ✅ legit pass |
| gpt-5.4-nano | 0.0 | 1 | 170 / 43  | "Implemented /app/extract.js. Running… outputs JSON object mapping addresses…" | model fail (output format) |
| MiniMax-M2.7 | 0.0 | 1 | 170 / 208 | Self-verified: "addr 0 = 1179403647 (ELF magic 0x464c457f), addr 4 = 65794 — both match. The file /app/extract.js is complete and ready." | model fail (output mismatch) |

## CTOX integration assessment

**No CTOX bugs found on this task.**

- All 3 trials ran cleanly with `errors=0`
- All 3 produced valid trajectories
- All 3 emitted completion-signaled replies (no mid-work flag fired,
  correctly)
- All 3 completed in 1 turn (no continuation loop needed)

## Why nano + M2.7 failed

Both nano and M2.7 confidently report the file is in place and the
script "works". M2.7 even shows it self-verified two values (addr 0
matches ELF magic 0x464c457f; addr 4 matches expected). But the
verifier disagreed with both.

Possible causes (all model-side):
- Wrong endianness or word size assumption
- Missing or extra entries in the output JSON
- Strings vs integers (M2.7 explicitly notes "not strings, as required"
  — so it knew the requirement, but the implementation might still be
  off in some edge case)
- Output format detail (trailing whitespace, key sorting, etc.)

These are **legitimate model fails** — the agent did the work, the
solution simply isn't what the verifier expects.

## Verdict per user's bench-acceptance rules

- gpt-5.4-mini: passed (no need for gpt-5.4-quality reference)
- gpt-5.4-nano: legitimate model fail
- MiniMax-M2.7: legitimate model fail (extra credit for self-
  verification effort, but the verifier is unmoved)

No CTOX changes from this task.

## Mid-work fix not needed here

Unlike count-dataset-tokens / configure-git-webserver, all three
models managed this task in 1 turn with completion-signaled replies.
The mid-work heuristic correctly stayed silent.
