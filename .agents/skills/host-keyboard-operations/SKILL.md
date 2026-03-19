---
name: host-keyboard-operations
description: Use when the CTO-Agent must inspect or change the host keyboard layout for a trusted owner request and must follow the reviewed keyboard capability contract instead of improvising shell commands.
---

# Host Keyboard Operations

Use this skill when a trusted owner interrupt asks for a real keyboard-layout change on the host and the task must be executed and verified on the live machine.

## Required sources

Read these first:

1. `contracts/system/host-keyboard-capability-policy.json`
2. `src/tooling.rs`
3. `src/supervisor.rs`
4. `src/runtime_db.rs`

## Commands

- `printf '%s\n' "$XDG_SESSION_TYPE" "$DISPLAY" "$XDG_SESSION_ID"`
- `localectl status`
- `setxkbmap -query`
- `loginctl session-status "$XDG_SESSION_ID"`

## Runtime contract

- Tool path: native terminal commands through the codex-backed `execCommand` and `execSessionAction` engine
- Runtime storage: `runtime/cto_agent.db`, task checkpoints, turn history, active exec sessions
- Inputs: trusted owner interrupt, requested layout or variant, active desktop/session evidence, reviewed keyboard capability contract
- Outputs: bounded diagnosis, one real mutation step, concrete verification evidence, exact checkpoint text with method and result
- Side effects: live session layout change, possible persistent system keyboard change, possible need for root-capable follow-up if only a persistent path can satisfy the request

## Failure handling

- Inspect the current session and keyboard state before choosing a mutator.
- If the contract path is missing from context, read `contracts/system/host-keyboard-capability-policy.json` before acting.
- If session scope and persistent scope differ, say which one you changed and which one still remains.
- If the available mechanism requires privileges or a desktop session that is not actually present, checkpoint the missing precondition instead of pretending success.
- Verify with the contract's verification commands and record the observed layout in the checkpoint detail.

## Never do

- Never mark the task done from intent, plan text or remembered history alone.
- Never assume X11, Wayland, console or desktop settings without inspecting the current host.
- Never repeat the same failed keyboard command without new host evidence.
