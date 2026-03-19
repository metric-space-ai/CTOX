---
name: desktop-session-operations
description: Use when the CTO-Agent needs to inspect, target or repair the active desktop session for host-side browser or GUI automation without improvising session environment wiring.
---

# Desktop Session Operations

Use this skill when a host task depends on the real active desktop session instead of a plain TTY or systemd service shell.

## Required sources

Read these first:

1. `contracts/system/desktop-session-capability-policy.json`
2. `src/desktop_session.rs`
3. `src/tooling.rs`
4. `src/browser_engine.rs`
5. `src/supervisor.rs`

## Commands

- `loginctl list-sessions --no-legend`
- `loginctl show-session "$SESSION_ID" --property=User --property=State --property=Remote --property=Display --property=Type --property=Desktop --property=Leader`
- `ps eww -u "$(id -un)"`
- `printf '%s\n' "$DISPLAY" "$WAYLAND_DISPLAY" "$XAUTHORITY" "$DBUS_SESSION_BUS_ADDRESS" "$XDG_RUNTIME_DIR"`

## Runtime contract

- Tool path: `src/desktop_session.rs` plus codex-backed `execCommand` and `execSessionAction`
- Runtime storage: active exec sessions, task checkpoints, `runtime/cto_agent.db`
- Inputs: owner or internal host task, active graphical session evidence, reviewed desktop-session capability contract
- Outputs: detected desktop-session env, bounded command/session launched against the live desktop context, verification output tied to the actual session
- Side effects: browser launch or host GUI mutation happens against the active desktop session instead of a detached service shell

## Failure handling

- Inspect the active session first; do not assume the service process env is the right target.
- If the contract is missing from context, read `contracts/system/desktop-session-capability-policy.json` before acting.
- If the host exposes only a TTY and no real desktop session, checkpoint that missing precondition instead of fabricating GUI success.
- Verify that the launched command saw the expected desktop env before marking the task done.

## Never do

- Never invent `DISPLAY`, `XAUTHORITY` or DBus values without host evidence.
- Never treat a persistent system setting as proof that the live desktop session changed.
- Never install or switch desktop environments as a hidden side effect of browser bootstrap.
