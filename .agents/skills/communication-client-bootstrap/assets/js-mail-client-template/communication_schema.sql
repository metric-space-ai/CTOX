PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;

CREATE TABLE IF NOT EXISTS communication_accounts (
    account_key TEXT PRIMARY KEY,
    channel TEXT NOT NULL,
    address TEXT NOT NULL,
    provider TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_inbound_ok_at TEXT,
    last_outbound_ok_at TEXT
);

CREATE TABLE IF NOT EXISTS communication_threads (
    thread_key TEXT PRIMARY KEY,
    channel TEXT NOT NULL,
    account_key TEXT NOT NULL,
    subject TEXT NOT NULL,
    participant_keys_json TEXT NOT NULL,
    last_message_key TEXT NOT NULL,
    last_message_at TEXT NOT NULL,
    message_count INTEGER NOT NULL,
    unread_count INTEGER NOT NULL,
    metadata_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS communication_messages (
    message_key TEXT PRIMARY KEY,
    channel TEXT NOT NULL,
    account_key TEXT NOT NULL,
    thread_key TEXT NOT NULL,
    remote_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    folder_hint TEXT NOT NULL,
    sender_display TEXT NOT NULL,
    sender_address TEXT NOT NULL,
    recipient_addresses_json TEXT NOT NULL,
    cc_addresses_json TEXT NOT NULL,
    bcc_addresses_json TEXT NOT NULL,
    subject TEXT NOT NULL,
    preview TEXT NOT NULL,
    body_text TEXT NOT NULL,
    body_html TEXT NOT NULL,
    raw_payload_ref TEXT NOT NULL,
    trust_level TEXT NOT NULL,
    status TEXT NOT NULL,
    seen INTEGER NOT NULL,
    has_attachments INTEGER NOT NULL,
    external_created_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_communication_messages_account_time
    ON communication_messages(account_key, external_created_at DESC);

CREATE INDEX IF NOT EXISTS idx_communication_messages_thread
    ON communication_messages(thread_key, external_created_at DESC);

CREATE INDEX IF NOT EXISTS idx_communication_messages_channel_remote
    ON communication_messages(channel, account_key, remote_id);

CREATE TABLE IF NOT EXISTS communication_sync_runs (
    run_key TEXT PRIMARY KEY,
    channel TEXT NOT NULL,
    account_key TEXT NOT NULL,
    folder_hint TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    ok INTEGER NOT NULL,
    fetched_count INTEGER NOT NULL,
    stored_count INTEGER NOT NULL,
    error_text TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);
