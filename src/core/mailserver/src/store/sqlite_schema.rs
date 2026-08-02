// ref: stalwart/src/store/sqlite/schema.rs:1-50
// ref: ctox-mailserver new code for unified campaign/collaboration sqlite schema

pub const SQLITE_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS stalwart_domains (
    domain_name TEXT PRIMARY KEY,
    dkim_selector TEXT NOT NULL,
    dkim_private_key TEXT NOT NULL,
    spf_record TEXT,
    dmarc_record TEXT
);

CREATE TABLE IF NOT EXISTS stalwart_smtp_queue (
    id TEXT PRIMARY KEY,
    from_addr TEXT NOT NULL,
    to_addr TEXT NOT NULL,
    msg_body TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0,
    next_attempt_at INTEGER NOT NULL,
    status TEXT NOT NULL
);

-- Audit history retained for the fallback reconciler. Delivery to Business OS
-- is driven by the durable outbox below, not by polling this log.
CREATE TABLE IF NOT EXISTS stalwart_smtp_delivery_log (
    id TEXT NOT NULL,
    from_addr TEXT NOT NULL,
    to_addr TEXT NOT NULL,
    outcome TEXT NOT NULL,
    error_text TEXT,
    completed_at INTEGER NOT NULL,
    PRIMARY KEY (id, completed_at)
);
CREATE INDEX IF NOT EXISTS stalwart_smtp_delivery_log_id_idx
    ON stalwart_smtp_delivery_log (id);

-- Durable handoff from the SMTP runner to the Business OS store. The runner
-- acks a row only after the targeted Business OS update succeeds; failures stay
-- pending with bounded exponential retry. provider_message_id is unique because
-- an SMTP queue item has exactly one terminal outcome.
CREATE TABLE IF NOT EXISTS stalwart_smtp_delivery_outbox (
    outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_message_id TEXT NOT NULL UNIQUE,
    from_addr TEXT NOT NULL,
    to_addr TEXT NOT NULL,
    outcome TEXT NOT NULL,
    error_text TEXT,
    completed_at INTEGER NOT NULL,
    delivery_attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at INTEGER NOT NULL,
    last_error TEXT,
    acked_at INTEGER
);
CREATE INDEX IF NOT EXISTS stalwart_smtp_delivery_outbox_pending_idx
    ON stalwart_smtp_delivery_outbox (acked_at, next_attempt_at, outbox_id);

CREATE TABLE IF NOT EXISTS stalwart_caldav_calendars (
    id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    display_name TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS stalwart_caldav_events (
    id TEXT PRIMARY KEY,
    calendar_id TEXT NOT NULL,
    uid TEXT NOT NULL,
    ical_data TEXT NOT NULL,
    last_modified INTEGER NOT NULL,
    FOREIGN KEY(calendar_id) REFERENCES stalwart_caldav_calendars(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS stalwart_carddav_addressbooks (
    id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    display_name TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS stalwart_carddav_contacts (
    id TEXT PRIMARY KEY,
    addressbook_id TEXT NOT NULL,
    uid TEXT NOT NULL,
    vcard_data TEXT NOT NULL,
    last_modified INTEGER NOT NULL,
    FOREIGN KEY(addressbook_id) REFERENCES stalwart_carddav_addressbooks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS stalwart_users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS stalwart_mailboxes (
    id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    name TEXT NOT NULL,
    uid_validity INTEGER NOT NULL DEFAULT 1,
    uid_next INTEGER NOT NULL DEFAULT 1,
    UNIQUE(owner, name)
);

CREATE TABLE IF NOT EXISTS stalwart_messages (
    id TEXT PRIMARY KEY,
    mailbox_id TEXT NOT NULL,
    from_addr TEXT NOT NULL,
    to_addr TEXT NOT NULL,
    subject TEXT,
    body TEXT NOT NULL,
    headers TEXT,
    is_read INTEGER NOT NULL DEFAULT 0,
    received_at INTEGER NOT NULL,
    uid INTEGER,
    is_deleted INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(mailbox_id) REFERENCES stalwart_mailboxes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_stalwart_messages_mailbox_received
    ON stalwart_messages(mailbox_id, received_at DESC, id);

CREATE TABLE IF NOT EXISTS stalwart_greylist (
    ip TEXT,
    sender TEXT,
    recipient TEXT,
    first_seen_at INTEGER NOT NULL,
    PRIMARY KEY(ip, sender, recipient)
);
"#;
