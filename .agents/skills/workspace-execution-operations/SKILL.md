---
name: workspace-execution-operations
description: Use when the CTO-Agent must keep multi-step repo or workspace work coherent across bounded turns instead of replaying the same shell edit or build step.
---

# Workspace Execution Operations

Use this skill for repo-local owner work, repair work, or other code tasks that span more than one shell interaction.

## Required sources

Read these first:

1. The active task context package, especially `contextDistillation.activeFocus`
2. The latest task checkpoints for the same task
3. Any visible exec session entries for the same task

## Runtime contract

- Prefer `execSessionAction` over repeated one-shot `execCommand` when the work will take more than one shell interaction.
- If the next machine step would require multi-line shell content, a heredoc, or an embedded script body, treat it as exec-session work; keep one-shot `execCommand` JSON strictly single-line and JSON-safe.
- Once verified workspace anchor evidence already exists for the same substantive repo task and more than one shell interaction is still expected, session continuity becomes required unless one exact anchored command is enough to finish the next bounded step.
- If an exec session already exists for the task, reuse it before starting a new one.
- After a real bounded workspace step on the same substantive task, keep the task in `nextMode=execute_task` unless there is a real boundary such as done, blocked, deliberate parking, or delegation.
- If the latest checkpoints describe the same bounded step again, inspect current workspace state before issuing another write, edit, or build command.
- If the task context includes verified machine evidence from the previous bounded step, treat that as the active workspace anchor; do not claim the context is missing and do not restart with another broad repo scan unless the anchor itself is insufficient.
- If the anchor is insufficient, name the exact missing fact first. A generic repo scan, generic history reload, or generic re-anchoring step is not valid continuation once the anchor already names the repo path, target files, or build target.
- A sentence saying that an exec session is now open does not create session continuity. Session continuity only exists after the turn actually returns `execSessionAction=start` and the machine path succeeds.
- Use narrow repo-state checks such as `git status --short`, a targeted `git diff`, or file inspection before repeating an edit.
- Prefer `rg` for repo search when it is available, but if `rg` is missing on the current host, fall back to `find`, `grep`, or direct file inspection in the same bounded turn instead of stopping on the missing binary.
- After a real edit, run the smallest relevant verification step and record what changed.
- Keep the checkpoint tied to the actual delta: touched files, verification result, and any session reused.

## Minimal loop

1. Read the active focus and the latest checkpoint.
2. Reuse or start a task-bound exec session if the work is multi-step. After a verified workspace anchor exists, this is the default path.
3. If verified machine evidence already exists, continue from that anchor instead of reopening broad history or repeating the same scan.
4. Make one bounded code or verification step.
5. Checkpoint the new evidence, not a restatement of the plan.

## Never do

- Never replay the same edit command just because the previous turn ended in `continue`.
- Never say the workspace or repo anchor is missing when the last bounded machine step already identified the workdir, target files, or other verified anchors.
- Never ignore an existing task-bound exec session and start over blindly.
- Never claim that an exec session exists or was opened unless the same turn actually returned a matching `execSessionAction`.
- Never spend a bounded turn on another broad repo scan or broad history reload after a verified workspace anchor already exists, unless the exact missing fact is named first.
- Never rewrite governance contracts or mode policies just to escape a repo task.
- Never discard user changes or reset the repo as a shortcut.
