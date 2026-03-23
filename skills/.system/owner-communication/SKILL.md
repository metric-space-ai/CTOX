---
name: owner-communication
description: Use when CTOX needs to communicate with the owner through TUI, email, or Jami, choose the correct communication path, continue an existing owner thread, or decide how proactive outbound owner contact should work.
metadata:
  short-description: Route owner communication across TUI, email, and Jami
---

# Owner Communication

Use this skill whenever CTOX needs to interpret, continue, or initiate communication with the owner.

## Scope

- Channels are limited to `tui`, `email`, and `jami`.
- Treat `tui` as the local, direct CTOX session.
- Treat `email` as topic-threaded and archival.
- Treat `jami` as continuous conversation flow, similar to chat.

## Channel Selection

1. If the owner contacted CTOX on a specific channel, prefer replying on that same channel.
2. If CTOX initiates contact and `CTOX_OWNER_PREFERRED_CHANNEL` is set in TUI settings, prefer that channel.
3. If CTOX initiates contact and no preferred channel is set, choose the lowest-friction configured channel that fits the urgency and persistence needs.
4. Do not invent a channel that is not configured in the prompt context.

## Channel Semantics

### TUI

- Use for direct local interaction with the owner.
- If the owner enters email credentials or Jami account details in TUI settings, treat TUI as the setup surface, not the long-term remote reply target.
- TUI is continuous within the local session rather than topic-threaded.

### Email

- Email is topic-threaded.
- Before sending a new outbound email, first look for an existing relevant owner thread in the communication store.
- Reuse the existing email thread when the topic matches.
- Start a new thread only when the subject materially changes.
- Use email for durable summaries, approvals, decisions, handoffs, and anything the owner may need to revisit later.

### Jami

- Treat Jami as an ongoing conversation stream rather than a subject-threaded mailbox.
- Prefer continuing the existing conversation tied to the owner account and conversation id.
- Use Jami for short operational updates, lightweight follow-ups, and rapid clarification when TUI is unavailable.

## Operational Rules

- Keep replies short and stateful.
- Match the owner's current thread or conversation context before opening a new one.
- When responding to inbound owner communication, continue the established path unless there is a clear reason to escalate to a more durable channel.
- When escalating from `jami` or `tui` to `email`, explicitly say that the detailed follow-up is moving to email.
- Verify the transport state after proactive outbound communication instead of assuming delivery.
- Treat email `accepted` as weaker than email `confirmed`.
- Treat Jami `queued` as not yet delivered.
- Do not leak secrets, passwords, root auth material, or BIOS-protected state into outbound channels unless the owner explicitly requests it and the channel choice is justified.

## Communication Shapes

- `tui`: direct answer, immediate clarification, local setup guidance
- `jami`: concise update, quick question, acknowledgement, short coordination
- `email`: durable summary, structured proposal, longer decision memo, explicit approval request

## Setup And Health

- Before relying on a configured remote channel, prefer running `ctox channel test --channel email` or `ctox channel test --channel jami`.
- If the test fails, keep setup and troubleshooting in `tui` until the remote path is healthy.
- If the owner entered communication credentials in TUI settings, treat that as configuration input, not automatic proof that the transport works.

## References

- For routing rules and examples, read `references/channel-routing.md`.
