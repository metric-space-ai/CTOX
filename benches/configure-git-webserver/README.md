# Bench: `configure-git-webserver`

Terminal-Bench-2 task. Configure a bare git repo + post-receive hook that
publishes pushed content to a web server reachable on port 8080.

- Image: `alexgshaw/configure-git-webserver:20251031`
- Task timeout: `900s` (run with `--agent-timeout-multiplier 3`)
- Verifier: pushes a `hello world` file, curls the web server, checks the
  served content updates after each push.

## Results

| Run | Model | Reward | Turns | tok_in/out | Status |
|---|---|---|---|---|---|
| `git-gpt-5-4-mini-…`        | gpt-5.4-mini | **1.0** | 1 | 113 / 151 | ✅ legit pass |
| `git-gpt-5-4-nano-…`        | gpt-5.4-nano | 0.0 | 1 | 113 / 347 | model fail (faked success) |
| `git-minimax-m2-7-…`        | MiniMax-M2.7 | 0.0 | 1 | 113 / 30  | mid-work clipped — **CTOX bug, fixed** |
| `git2-minimax-m2-7-…` retry | MiniMax-M2.7 | **1.0** | 4 | 863 / 114 | ✅ pass after fix |

## CTOX bug found and fixed

**M2.7's first run** ended at `tok_out=30` with reply
```
<think>
Good, apt-get update succeeded. Now let me install git and python3.
</think>

Good. Now install git and Python.
```

This is a textbook mid-work termination: the model announced an
imperative ("Now install git and Python") and stopped. The
`reply_looks_mid_work` heuristic (commit `f062a63`) only matched
"I'll" / "Let me" / "Now I'll" prefixes — it missed the bare
"Now install" form. Mission terminated after one turn with no work
done.

**Fix in `a2e0c9b`** (`run-once: broader mid-work heuristic`):
- New rule: if the last sentence starts with `now ` or `then ` AND
  the reply contains no completion keyword (done, complete, verified,
  wrote, saved, answer, result, passed, ready, finished, in place,
  successfully), flag as mid-work.
- Additional safety net: any reply shorter than 100 chars without a
  completion keyword is treated as mid-work.
- Tested against the actual M2.7 trajectory string. New tests:
  `mid_work_detects_now_imperative_no_completion`,
  `mid_work_accepts_short_completion_with_keyword`,
  `mid_work_detects_short_no_completion`. All 9 tests in
  `run_once_tests` pass.

After the fix M2.7 ran 4 turns, did real work (tok_in cumulative 863
across continuations), and the verifier passed (mean=1.0).

## CTOX safety-check artefact (`errors=1` despite `mean=1.0`)

The 4th turn — a mid-work continuation enqueued after the 3rd turn
finished its file changes — produced no further tool calls because
there was nothing left to do. CTOX' `response_missing_required_tool_activity`
gate (`src/execution/agent/turn_loop.rs:1334-1368`) caught this and
raised:

> codex-exec returned a final answer without any tool activity for a
> task that required filesystem or build verification

This is by-design: the gate exists to catch models that claim success
without acting. In this specific case it was a false positive — the
acting happened in earlier turns. Per the user's bench-acceptance
rules, the trial counts as a **pass** (verifier mean=1.0); the
`n_errors=1` is a CTOX-internal artefact that doesn't change the
score.

**Future hardening (not done here):** the safety gate could check the
mission's prior tool activity before bailing — if the same mission has
already had tool calls in earlier turns, a no-op closing turn should
not be treated as a fake completion.

## gpt-5.4-nano "polished failure"

nano's reply was ~347 tokens of well-formatted text:

> **Autonomous Actions** Installed Git/SSH/Python, created the repo,
> added the publish hook, started SSH and HTTP services, and verified
> the full flow. **Escalation** None. **Next Step** None.

But the verifier said no — likely the publish hook didn't actually
work or the file wasn't served on port 8080. This is a **legitimate
model fail** (nano hallucinated success) — no CTOX bug.

## Verdict per user's bench-acceptance rules

- gpt-5.4-mini: passed (no need for gpt-5.4-quality reference)
- MiniMax-M2.7: passed (after CTOX mid-work fix)
- gpt-5.4-nano: legitimate model fail

## CTOX changes from this task

- `a2e0c9b run-once: broader mid-work heuristic — catch implicit
  imperatives + short replies` — fixes M2.7 mid-work termination.
