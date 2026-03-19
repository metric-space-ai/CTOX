# Communication Store V1

This reference defines the default SQLite persistence shape for communication tooling built by the CTO-Agent.

The schema is channel-neutral on purpose.
Email is only the first adapter.

## Design goals

- one SQLite substrate for mail, chat, webhooks and later channels
- stable identifiers for accounts, threads, messages and sync runs
- adapter-specific protocol details stored in `metadata_json`
- trust stays explicit and does not get inferred silently from transport alone

## Required tables

### `communication_accounts`

One row per local account or transport identity.

Required fields:

- `account_key`: stable primary key, for example `email:cto1@metric-space.ai`
- `channel`: `email`, `chat`, `webhook`, `whatsapp`, ...
- `address`: account address or endpoint identity
- `provider`: provider label such as `one.com`
- `profile_json`: adapter config snapshot that is safe to persist
- `last_inbound_ok_at`
- `last_outbound_ok_at`

### `communication_threads`

Conversation grouping that survives adapter differences.

Required fields:

- `thread_key`: stable conversation key
- `channel`
- `account_key`
- `subject`
- `participant_keys_json`
- `last_message_key`
- `last_message_at`
- `message_count`
- `unread_count`
- `metadata_json`

### `communication_messages`

Canonical event log for inbound and outbound communication.

Required fields:

- `message_key`: stable primary key
- `channel`
- `account_key`
- `thread_key`
- `remote_id`: provider-side id if available
- `direction`: `inbound`, `outbound`, `internal`
- `folder_hint`: `INBOX`, `sent`, `drafts`, queue name, webhook bucket, ...
- `sender_display`
- `sender_address`
- `recipient_addresses_json`
- `cc_addresses_json`
- `bcc_addresses_json`
- `subject`
- `preview`
- `body_text`
- `body_html`
- `raw_payload_ref`: path or external blob ref, not necessarily inline raw payload
- `trust_level`: default low for email unless separately validated
- `status`: `received`, `sent`, `queued`, `draft`, `failed`, ...
- `seen`
- `has_attachments`
- `external_created_at`
- `observed_at`
- `metadata_json`

### `communication_sync_runs`

Audit trail for sync or polling jobs.

Required fields:

- `run_key`
- `channel`
- `account_key`
- `folder_hint`
- `started_at`
- `finished_at`
- `ok`
- `fetched_count`
- `stored_count`
- `error_text`
- `metadata_json`

## Mapping rules

- Channel-specific ids belong in `remote_id` and `metadata_json`, not in new custom tables unless there is a strong reason.
- Sender verification against organigram is a separate trust step. Transport receipt alone does not raise `trust_level`.
- If a provider has no real thread id, derive `thread_key` from stable headers or local conversation heuristics and document that logic in `metadata_json`.
- Keep bodies normalized, but allow raw payloads to live outside SQLite when they are too large.

## Email starter mapping

- `channel`: `email`
- `account_key`: `email:<mailbox>`
- `thread_key`: prefer `References` root, then `In-Reply-To`, then `Message-ID`, then a local fallback
- `remote_id`: UID or provider message id
- `folder_hint`: `INBOX`, `sent`, `drafts`, ...
- `trust_level`: `low` by default under current BIOS policy

## Why this exists

The CTO-Agent should be able to build its own communication tools.
That only scales if the storage contract is shared across channels and adapters.
