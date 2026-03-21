---
name: codex-command-exec-operations
description: Use when the CTO-Agent must drive the real Codex `command_exec` surface correctly, especially for exec-session lifecycle, buffered reads, and bounded one-shot shell work.
---

# Codex Command Exec Operations

Use this skill when a task depends on the real Codex CLI execution engine rather than on free-form shell narration.

## Required sources

Read these first:

1. `src/command_exec.rs`
2. `src/supervisor.rs`
3. The active task context package, especially `execSessions`, `contextDistillation.activeFocus`, and `current_task_machine_evidence`

## Canonical surface

- `execCommand` and `execSessionAction` are two paths on the same `command_exec` engine.
- `execCommand` is the one-shot bounded path. Use it for one non-interactive shell step.
- `execSessionAction=start` opens a persistent session.
- `execSessionAction=write` writes bytes into that session's stdin.
- `execSessionAction=read` reads the buffered snapshot for that session.
- `execSessionAction=terminate` closes the session.
- Every control-payload string must stay valid JSON. Do not place raw literal newlines inside JSON string fields.
- `execCommand` array items must stay single-line and JSON-safe. If the shell step would need a heredoc, embedded multi-line Python, or other literal newlines, do not force it into `execCommand`.
- For multi-line shell or script content, prefer `execSessionAction=start` plus `execSessionAction=write`, or split the work into smaller bounded one-line steps.

## Real session semantics

- A session only exists after the turn actually returns `execSessionAction` and the machine path succeeds.
- Do not claim that a session exists just because the reply says so.
- Reuse a visible task-bound session before starting another one.
- If no session is visible, either start one or omit `execSessionId` and let the kernel assign the task-bound default.
- For shell sessions, a `write` step must send a real command line, usually ending with a newline.
- A `read` step does not run a command. It only returns the buffered stdout/stderr snapshot that already exists.
- Repeating `read` without a prior `write`, new process output, or another real state change is not progress.
- Streaming sessions still need readable snapshots later; treat `read` as the recovery surface for prior shell output, not as a new action.
- If a command would require multiple shell lines, a heredoc, or a script body, treat that as session work by default instead of a one-shot JSON command.

## Practical loop

1. If the task is multi-step, start or reuse an exec session.
2. Write one concrete shell step into that session.
3. Read the resulting buffered output only after the shell had a chance to produce it.
4. Use the resulting snapshot as the next turn's verified anchor.
5. If the step was really one-shot, use `execCommand` instead of opening a session.

## Failure boundaries

- If `read` returns the same empty snapshot repeatedly, do not loop on `read`.
- Diagnose whether the previous `write` actually sent a runnable command, whether the shell is waiting for a newline, or whether no output was produced yet.
- If a session id is not found, do not invent continuity. Start a new session or use the visible one.
- If a session is active but stale, inspect the snapshot before sending another write.
- If there is no machine path in the turn, keep the checkpoint at planning/inspection only.

## Never do

- Never describe an opened session unless the same turn actually returned `execSessionAction=start`.
- Never treat `execSessionAction=read` as if it had executed a command.
- Never reread the same empty session output blindly across turns.
- Never replace visible session ids with vanity ids like `cxx-01` unless that exact id is already real in context.
- Never claim build, test, edit, or runtime results without the matching `command_exec` evidence.
- Never put heredocs or other raw multi-line script bodies directly into `execCommand` JSON strings.
