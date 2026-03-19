# Interrupt Bridge

Inbound communication should not write task rows directly.
It should enter the same interrupt path the runtime already uses for homepage, BIOS and terminal input.

## Required flow

1. Persist the incoming message in the communication store.
2. Trigger a loop interrupt for the channel.
3. Convert that interrupt into a queued task.
4. Let supervisor reprioritization pull it into the next safe turn boundary.

## Rust path

The runtime already has the canonical pieces:

- `src/runtime_db.rs`
  - `enqueue_loop_interrupt(...)`
  - `queue_loop_interrupt_as_task(...)`
  - `ingest_pending_loop_interrupts(...)`
  - `compute_priority_score(...)`
  - `create_task_from_interrupt(...)`
- `src/supervisor.rs`
  - runs interrupt ingest and reprioritization before selecting the next queued task
- `src/bootstrap.rs`
  - `queue_channel_interrupt(...)`
- `src/main.rs`
  - `cto-agent channel-interrupt <source_channel> <speaker> <message>`

## Semantics

- The interrupt does not hard-kill the current bounded step.
- It gets recorded immediately.
- The running turn may receive a turn signal.
- The interrupt becomes a queued task with channel-specific trust and priority.
- The task is then considered in the next reprioritization cycle.

## For email tools

For a mail client, the minimal safe pattern is:

1. detect a new inbound message
2. write it to `communication_messages`
3. call:

```text
cto-agent channel-interrupt email "<sender>" "<subject and short summary>"
```

Good interrupt payloads include:

- sender identity
- subject
- short preview
- urgency hints if they are visible in the mail

Do not dump the full raw message into the interrupt text.
Store the full message in SQLite and pass a short actionable summary into the interrupt.
