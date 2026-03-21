---
name: email-operations
description: Use when the CTO-Agent needs to operate, inspect or patch the current local IMAP/SMTP mail client, inbox sync state or email interrupt path in this repo.
---

# Email Operations

Use this skill when the local mail path already exists and later turns need to send, sync, inspect, or repair email without rebuilding the client from scratch.

## Required sources

Read these first:

1. `../../../scripts/communication_mail_cli.mjs`
2. `../communication-client-bootstrap/references/communication-store-v1.md`
3. `../communication-client-bootstrap/references/interrupt-bridge.md`
4. `../../../runtime/kleinhirn.env` when runtime credentials or account wiring need to be verified

## Commands

- `node scripts/communication_mail_cli.mjs list --db runtime/cto_agent.db --limit 20`
- `CTO_EMAIL_ADDRESS=<account> CTO_EMAIL_PASSWORD=<password> node scripts/communication_mail_cli.mjs sync --db runtime/cto_agent.db --imap-host imap.one.com --folder INBOX --limit 20 --emit-interrupts true`
- `CTO_EMAIL_ADDRESS=<account> CTO_EMAIL_PASSWORD=<password> node scripts/communication_mail_cli.mjs send --db runtime/cto_agent.db --smtp-host send.one.com --to <recipient> --subject "<subject>" --body "<body>"`

## Runtime contract

- Tool path: `scripts/communication_mail_cli.mjs`
- Runtime storage: `runtime/cto_agent.db` plus `runtime/kleinhirn.env`
- Mail account defaults: `CTO_EMAIL_ADDRESS`, `CTO_EMAIL_PASSWORD`, `imap.one.com`, `send.one.com`
- Outputs: cached outbound mail rows, sync rows, JSON success/error payloads
- Side effects: may send real outbound mail and may write `communication_accounts`, `communication_messages`, and `communication_sync_runs`

## Failure handling

- If send or sync fails, inspect the JSON error first; then verify `CTO_EMAIL_ADDRESS` / `CTO_EMAIL_PASSWORD` and the one.com IMAP/SMTP hosts.
- If the owner supplied bounded credentials directly in the task context, use explicit env wiring for that bounded command instead of waiting for runtime env wiring.
- If the script works manually but the agent cannot use it, inspect the active task context for this skill and the matching workspace execution guidance before changing the mail client.
- If inbound mail is synced successfully, `--emit-interrupts true` should bridge new messages into `channel-interrupt email ...`; if follow-up work still does not appear, inspect the interrupt bridge separately.
- Do not claim the mail path is ready for real outbound work unless the same JS adapter can also run inbound sync with interrupt emission.
- Verify with one bounded live check before claiming the mail path is operational.

## Never do

- Never hardcode live credentials into this skill or committed code.
- Never claim a mail was sent without the JSON success result and `message_id`.
- Never patch BIOS or trust policy just to force a low-trust email path to carry more authority than intended.
- Never send real mail through raw `python -c smtplib`, ad-hoc SMTP shell one-liners, or any path outside `scripts/communication_mail_cli.mjs`.
- If a bounded owner-visible mail step succeeded once with JSON `ok=true` plus `message_id`, stop and close that bounded mail task instead of resending the same message.
