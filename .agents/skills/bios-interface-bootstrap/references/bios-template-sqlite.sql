-- BIOS start template SQLite reference queries
-- Runtime DB path: runtime/cto_agent.db
-- These queries back the operational side of the BIOS start template.

-- Overview / trust
SELECT
    owner_name,
    committed_owner_name,
    owner_contact_established,
    bios_primary_channel_confirmed,
    superpassword_set,
    owner_commitment_score,
    last_owner_dialogue_at,
    calibration_notes,
    brain_access_mode
FROM owner_trust
WHERE singleton = 1;

-- Runtime focus
SELECT
    mode,
    active_task_id,
    active_task_title,
    queue_depth,
    last_reprioritized_at,
    last_task_completed_at,
    note
FROM focus_state
WHERE singleton = 1;

-- Open tasks
SELECT
    id,
    created_at,
    updated_at,
    task_kind,
    title,
    detail,
    trust_level,
    priority_score,
    status,
    run_count,
    last_checkpoint_summary,
    last_checkpoint_at
FROM tasks
WHERE status IN ('queued', 'running', 'blocked')
ORDER BY priority_score DESC, updated_at DESC
LIMIT 12;

-- Recent BIOS dialogue
SELECT
    created_at,
    speaker,
    channel,
    message,
    used_grosshirn
FROM bios_dialogue
ORDER BY id DESC
LIMIT 12;

-- Memory items
SELECT
    created_at,
    kind,
    summary,
    detail,
    source,
    important
FROM memory_items
ORDER BY id DESC
LIMIT 12;

-- Memory summaries
SELECT
    scope,
    updated_at,
    summary
FROM memory_summaries
WHERE scope IN (
    'owner_calibration',
    'learning_working_set',
    'learning_operational',
    'learning_general',
    'learning_negative'
)
ORDER BY scope ASC;

-- Active learning entries
SELECT
    id,
    updated_at,
    learning_class,
    status,
    summary,
    detail,
    evidence,
    applicability,
    confidence,
    salience,
    recall_count,
    last_recalled_at
FROM learning_entries
WHERE status = 'active'
ORDER BY salience DESC, updated_at DESC
LIMIT 16;

-- Resources for runtime panel
SELECT
    category,
    name,
    observed_at,
    status,
    detail
FROM resources
ORDER BY category ASC, name ASC
LIMIT 32;

-- Installed skills
SELECT
    name,
    path,
    status,
    notes,
    last_seen_at
FROM skills
ORDER BY name ASC;

-- Homepage revision trail
SELECT
    created_at,
    source_channel,
    title,
    headline,
    owner_branding_applied,
    notes
FROM homepage_revisions
ORDER BY id DESC
LIMIT 8;

-- BIOS chat uploads
SELECT
    created_at,
    speaker,
    source_channel,
    note,
    file_name,
    public_path,
    mime_type
FROM bios_uploads
ORDER BY id DESC
LIMIT 8;
