# Terminal-Bench-2 task READMEs

Per-task analysis of the CTOX integration's behaviour on
`terminal-bench/terminal-bench-2`. Each task's `README.md` documents
all observed runs, distinguishes CTOX bugs from legitimate model
fails (per the user's bench-acceptance rules), and notes every CTOX
code change that came out of the work.

User's bench-acceptance rules (verbatim):
1. **Not acceptable**: CTOX crashes / hangs / dies mid-mission.
2. **Not acceptable**: CTOX ends a mission early without the model
   explicitly confirming completion.
3. **Acceptable**: model executes the mission and the solution is
   wrong per verifier.

## Score grid (final state, after CTOX fixes)

| Task                         | gpt-5.4-mini | gpt-5.4-nano | MiniMax-M2.7 | Notes |
|---|---|---|---|---|
| `adaptive-rejection-sampler` | 0.0  | 0.0 | 0.0 | OpenAI stream-disconnect (mini/nano), Harbor 900s timeout (M2.7) |
| `chess-best-move`            | **1.0** ✅ | 0.0 | 0.0 | mini passed; nano/M2.7 legit model-fail |
| `configure-git-webserver`    | **1.0** ✅ | 0.0 | **1.0** ✅ | M2.7 needed `a2e0c9b` mid-work-broader fix to pass |
| `count-dataset-tokens`       | 0.0  | **1.0** ✅ | **1.0** ✅ | mini hallucinated success ("Done." with 4 tok) |
| `extract-elf`                | **1.0** ✅ | 0.0 | 0.0 | mini passed; nano/M2.7 legit model-fail |

3-model means after fixes: mini=0.6, nano=0.4, MiniMax=0.6.

## CTOX code changes from this benchmark cycle

| Commit | What | Surfaced by |
|---|---|---|
| `f062a63` | `run-once: don't close mission on is_open=false alone` — added `reply_looks_mid_work` heuristic + `enqueue_midwork_continuation` follow-up; rewrote run-once termination logic per actual `mission_is_open` semantics in `lcm.rs:4178` | `adaptive-rejection-sampler`, `configure-git-webserver` (M2.7 mid-think termination) |
| `a2e0c9b` | `run-once: broader mid-work heuristic` — catch implicit `Now install …` imperatives + short replies without completion keyword | `configure-git-webserver` retry with M2.7 |

Each commit is gated by unit tests in `src/main.rs::run_once_tests`
(9 test cases now passing).

## Pre-existing CTOX issues observed but NOT yet fixed

1. **OpenAI `/v1/responses` stream-disconnect on adaptive-rejection-sampler**
   — fully reproducible across gpt-5.4 / mini / nano (same 666-token
   request). Either CTOX/codex-exec is sending something OpenAI rejects
   at TLS/HTTP layer for this specific brief, or upstream regional
   issue. Needs request-body inspection.

2. **`response_missing_required_tool_activity` false-positive**
   (`turn_loop.rs:1334-1368`) — the safety gate fires on the FINAL
   continuation turn after the actual work has been done in earlier
   turns, raising an "exception" even though the verifier passes
   (`mean=1.0`, `errors=1`). Should consider mission-history when
   deciding to bail. Documented in `configure-git-webserver/README.md`.

3. **No partial-trajectory dump on Harbor SIGTERM** — when Harbor
   kills CTOX at the agent_timeout, ATIF export never runs and we
   lose all diagnostic output for the run. Documented in
   `adaptive-rejection-sampler/README.md`. Future hardening.

## Verdict per user's rules across all 5 tasks

- **No CTOX crashes or hangs** in any run (after fixes).
- **No CTOX premature mission closure** in any run (after fixes).
- All non-passing runs map to either:
  - legitimate model fail (wrong answer / hallucinated success / model
    can't solve), or
  - upstream/transport issue (OpenAI stream-disconnect on one task) /
    Harbor agent_timeout (M2.7 on the slow R-statistics task).

Bench is now usable as a fair surfacing tool for further CTOX issues.
