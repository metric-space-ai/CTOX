---
name: email-operations
description: Use when the CTO-Agent needs to operate, inspect or patch the current local email client, inbox sync state or email interrupt path in this repo.
---

# Email Operations

Use this skill only after the CTO-Agent has a real working mail client and wants to make that capability persistent for later turns.

This file is an example template resource, not the bound live skill yet.
To bind it for later turns, copy and specialize it into `.agents/skills/email-operations/SKILL.md`.

## Required sources

Read these first:

1. `<mail-client-path>`
2. `../communication-client-bootstrap/references/communication-store-v1.md`
3. `../communication-client-bootstrap/references/interrupt-bridge.md`
4. `<mail-runtime-db-or-config-path>`

## Commands

- `node <mail-client-path> sync --account-id <account-id>`
- `node <mail-client-path> list --account-id <account-id> --mailbox INBOX --limit 20`
- `node <mail-client-path> send --account-id <account-id> --to <recipient> --subject "<subject>" --text "<body>"`

## Runtime contract

- Tool path: `<mail-client-path>`
- Runtime storage: `<mail-runtime-db-or-config-path>`
- Shared communication store tables: `communication_accounts`, `communication_threads`, `communication_messages`, `communication_sync_runs`
- Inputs: mailbox credentials, recipient, subject, body, sync options
- Outputs: stored communication rows, outbound send result, optional loop interrupt for new inbound mail
- Side effects: may write SQLite rows and may emit `cto-agent channel-interrupt email "<sender>" "<summary>"`

## Failure handling

- If sync fails, inspect provider auth, IMAP/SMTP host settings and the latest rows in `communication_sync_runs`
- If messages are stored but no task appears, inspect the interrupt bridge and verify the `channel-interrupt` path
- If provider behavior changed, patch the mail client code first; do not try to hide transport breakage in prompts
- Verify with one bounded live check: one inbox sync and, if safe, one test send

## Never do

- Never treat this template as already bound capability until you actually write `.agents/skills/email-operations/SKILL.md`
- Never claim the client is operational without a real read or send test
- Never hardcode live credentials into the skill text
