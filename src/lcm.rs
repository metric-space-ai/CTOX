use anyhow::Context;
use anyhow::Result;
use regex::Regex;
use rusqlite::params;
use rusqlite::Connection;
use rusqlite::OptionalExtension;
use serde::Serialize;
use sha2::Digest;
use sha2::Sha256;
use std::path::Path;
#[cfg(test)]
use std::sync::atomic::AtomicU64;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

const DEFAULT_CONTEXT_THRESHOLD: f64 = 0.75;
const DEFAULT_FRESH_TAIL_COUNT: usize = 8;
const DEFAULT_LEAF_CHUNK_TOKENS: i64 = 20_000;
const DEFAULT_LEAF_TARGET_TOKENS: usize = 600;
const DEFAULT_CONDENSED_TARGET_TOKENS: usize = 900;
const DEFAULT_LEAF_MIN_FANOUT: usize = 4;
const DEFAULT_CONDENSED_MIN_FANOUT: usize = 3;
const DEFAULT_MAX_ROUNDS: usize = 6;
const FALLBACK_MAX_CHARS: usize = 512 * 4;
const CONDENSED_MIN_INPUT_RATIO: f64 = 0.1;
#[cfg(test)]
static TEMP_DB_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SummaryKind {
    Leaf,
    Condensed,
}

impl SummaryKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Leaf => "leaf",
            Self::Condensed => "condensed",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContextItemType {
    Message,
    Summary,
}

impl ContextItemType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Message => "message",
            Self::Summary => "summary",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageRecord {
    pub message_id: i64,
    pub conversation_id: i64,
    pub seq: i64,
    pub role: String,
    pub content: String,
    pub token_count: i64,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SummaryRecord {
    pub summary_id: String,
    pub conversation_id: i64,
    pub kind: SummaryKind,
    pub depth: i64,
    pub content: String,
    pub token_count: i64,
    pub descendant_count: i64,
    pub descendant_token_count: i64,
    pub source_message_token_count: i64,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SummarySubtreeNode {
    pub summary_id: String,
    pub parent_summary_id: Option<String>,
    pub depth_from_root: i64,
    pub kind: SummaryKind,
    pub depth: i64,
    pub token_count: i64,
    pub descendant_count: i64,
    pub descendant_token_count: i64,
    pub source_message_token_count: i64,
    pub child_count: i64,
    pub path: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DescribeSummary {
    pub summary: SummaryRecord,
    pub parent_ids: Vec<String>,
    pub child_ids: Vec<String>,
    pub message_ids: Vec<i64>,
    pub subtree: Vec<SummarySubtreeNode>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DescribeResult {
    Summary(DescribeSummary),
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageSearchResult {
    pub message_id: i64,
    pub conversation_id: i64,
    pub role: String,
    pub snippet: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SummarySearchResult {
    pub summary_id: String,
    pub conversation_id: i64,
    pub kind: SummaryKind,
    pub snippet: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct GrepResult {
    pub messages: Vec<MessageSearchResult>,
    pub summaries: Vec<SummarySearchResult>,
    pub total_matches: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExpandChild {
    pub summary_id: String,
    pub kind: SummaryKind,
    pub content: String,
    pub token_count: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExpandMessage {
    pub message_id: i64,
    pub role: String,
    pub content: String,
    pub token_count: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExpandResult {
    pub children: Vec<ExpandChild>,
    pub messages: Vec<ExpandMessage>,
    pub estimated_tokens: i64,
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContextItemSnapshot {
    pub ordinal: i64,
    pub item_type: ContextItemType,
    pub message_id: Option<i64>,
    pub summary_id: Option<String>,
    pub seq: i64,
    pub depth: i64,
    pub token_count: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LcmSnapshot {
    pub conversation_id: i64,
    pub messages: Vec<MessageRecord>,
    pub summaries: Vec<SummaryRecord>,
    pub context_items: Vec<ContextItemSnapshot>,
    pub summary_edges: Vec<(String, String)>,
    pub summary_messages: Vec<(String, i64)>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContinuityRevision {
    pub revision_id: String,
    pub conversation_id: i64,
    pub narrative: String,
    pub anchors: String,
    pub focus: String,
    pub source_summary_ids: Vec<String>,
    pub source_message_ids: Vec<i64>,
    pub created_at: String,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContinuityKind {
    Narrative,
    Anchors,
    Focus,
}

impl ContinuityKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Narrative => "narrative",
            Self::Anchors => "anchors",
            Self::Focus => "focus",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "narrative" => Ok(Self::Narrative),
            "anchors" => Ok(Self::Anchors),
            "focus" => Ok(Self::Focus),
            other => anyhow::bail!("unsupported continuity kind: {other}"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ContinuityDocumentState {
    pub conversation_id: i64,
    pub kind: ContinuityKind,
    pub head_commit_id: String,
    pub content: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContinuityCommitRecord {
    pub commit_id: String,
    pub conversation_id: i64,
    pub kind: ContinuityKind,
    pub parent_commit_id: Option<String>,
    pub diff_text: String,
    pub rendered_text: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContinuityForgottenEntry {
    pub commit_id: String,
    pub conversation_id: i64,
    pub kind: ContinuityKind,
    pub line: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContinuityShowAll {
    pub conversation_id: i64,
    pub narrative: ContinuityDocumentState,
    pub anchors: ContinuityDocumentState,
    pub focus: ContinuityDocumentState,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContinuityPromptPayload {
    pub conversation_id: i64,
    pub kind: ContinuityKind,
    pub current_document: String,
    pub recent_messages: Vec<String>,
    pub recent_summaries: Vec<String>,
    pub forgotten_lines: Vec<String>,
    pub prompt: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompactionDecision {
    pub should_compact: bool,
    pub reason: String,
    pub current_tokens: i64,
    pub threshold: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompactionResult {
    pub action_taken: bool,
    pub tokens_before: i64,
    pub tokens_after: i64,
    pub created_summary_ids: Vec<String>,
    pub rounds: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct FixtureMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct FixtureGrep {
    pub scope: String,
    pub mode: String,
    pub query: String,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct FixtureExpand {
    pub summary_id: Option<String>,
    pub depth: Option<usize>,
    pub include_messages: Option<bool>,
    pub token_cap: Option<i64>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LcmFixture {
    pub conversation_id: i64,
    pub token_budget: i64,
    pub force_compact: Option<bool>,
    pub config: Option<LcmFixtureConfig>,
    pub messages: Vec<FixtureMessage>,
    pub grep_queries: Option<Vec<FixtureGrep>>,
    pub expand_queries: Option<Vec<FixtureExpand>>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LcmFixtureConfig {
    pub context_threshold: Option<f64>,
    pub fresh_tail_count: Option<usize>,
    pub leaf_chunk_tokens: Option<i64>,
    pub leaf_target_tokens: Option<usize>,
    pub condensed_target_tokens: Option<usize>,
    pub leaf_min_fanout: Option<usize>,
    pub condensed_min_fanout: Option<usize>,
    pub max_rounds: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FixtureRunOutput {
    pub compaction: CompactionResult,
    pub snapshot: LcmSnapshot,
    pub grep_results: Vec<GrepResult>,
    pub expand_results: Vec<ExpandResult>,
}

#[derive(Debug, Clone)]
pub struct LcmConfig {
    pub context_threshold: f64,
    pub fresh_tail_count: usize,
    pub leaf_chunk_tokens: i64,
    pub leaf_target_tokens: usize,
    pub condensed_target_tokens: usize,
    pub leaf_min_fanout: usize,
    pub condensed_min_fanout: usize,
    pub max_rounds: usize,
}

impl Default for LcmConfig {
    fn default() -> Self {
        Self {
            context_threshold: DEFAULT_CONTEXT_THRESHOLD,
            fresh_tail_count: DEFAULT_FRESH_TAIL_COUNT,
            leaf_chunk_tokens: DEFAULT_LEAF_CHUNK_TOKENS,
            leaf_target_tokens: DEFAULT_LEAF_TARGET_TOKENS,
            condensed_target_tokens: DEFAULT_CONDENSED_TARGET_TOKENS,
            leaf_min_fanout: DEFAULT_LEAF_MIN_FANOUT,
            condensed_min_fanout: DEFAULT_CONDENSED_MIN_FANOUT,
            max_rounds: DEFAULT_MAX_ROUNDS,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GrepMode {
    Regex,
    FullText,
}

impl GrepMode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "regex" => Ok(Self::Regex),
            "full_text" | "full-text" | "fts" => Ok(Self::FullText),
            other => anyhow::bail!("unsupported grep mode: {other}"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GrepScope {
    Messages,
    Summaries,
    Both,
}

impl GrepScope {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "messages" => Ok(Self::Messages),
            "summaries" => Ok(Self::Summaries),
            "both" => Ok(Self::Both),
            other => anyhow::bail!("unsupported grep scope: {other}"),
        }
    }
}

#[derive(Debug, Clone)]
struct ContextEntry {
    ordinal: i64,
    item_type: ContextItemType,
    message_id: Option<i64>,
    summary_id: Option<String>,
    seq: i64,
    depth: i64,
    token_count: i64,
}

pub trait Summarizer {
    fn summarize(&self, kind: SummaryKind, depth: i64, lines: &[String], target_tokens: usize) -> Result<String>;
}

struct EscalatedSummary {
    content: String,
}

pub struct HeuristicSummarizer;

impl Summarizer for HeuristicSummarizer {
    fn summarize(&self, kind: SummaryKind, depth: i64, lines: &[String], target_tokens: usize) -> Result<String> {
        let mut header = match kind {
            SummaryKind::Leaf => format!("LCM leaf summary at depth {depth}:"),
            SummaryKind::Condensed => format!("LCM condensed summary at depth {depth}:"),
        };
        let mut output = Vec::new();
        let max_chars = target_tokens.saturating_mul(4);
        let mut current_len = header.len();
        output.push(header.clone());
        for line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let bullet = format!("- {}", collapse_whitespace(trimmed));
            if current_len + bullet.len() + 1 > max_chars {
                break;
            }
            current_len += bullet.len() + 1;
            output.push(bullet);
        }
        if output.len() == 1 {
            header.push_str(" no significant content captured.");
            output[0] = header;
        }
        Ok(output.join("\n"))
    }
}

pub struct LcmEngine {
    conn: Connection,
    config: LcmConfig,
}

impl LcmEngine {
    pub fn open(path: &Path, config: LcmConfig) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("failed to open SQLite database {}", path.display()))?;
        let engine = Self { conn, config };
        engine.init_schema()?;
        Ok(engine)
    }

    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            r#"
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_conversation_seq
                ON messages(conversation_id, seq);

            CREATE TABLE IF NOT EXISTS summaries (
                summary_id TEXT PRIMARY KEY,
                conversation_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                depth INTEGER NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                descendant_count INTEGER NOT NULL DEFAULT 0,
                descendant_token_count INTEGER NOT NULL DEFAULT 0,
                source_message_token_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS summary_edges (
                parent_summary_id TEXT NOT NULL,
                child_summary_id TEXT NOT NULL,
                PRIMARY KEY(parent_summary_id, child_summary_id)
            );

            CREATE TABLE IF NOT EXISTS summary_messages (
                summary_id TEXT NOT NULL,
                message_id INTEGER NOT NULL,
                PRIMARY KEY(summary_id, message_id)
            );

            CREATE TABLE IF NOT EXISTS context_items (
                conversation_id INTEGER NOT NULL,
                ordinal INTEGER NOT NULL,
                item_type TEXT NOT NULL,
                message_id INTEGER,
                summary_id TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY(conversation_id, ordinal)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                content,
                content='',
                tokenize='unicode61'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(
                summary_id UNINDEXED,
                content,
                content='',
                tokenize='unicode61'
            );

            CREATE TABLE IF NOT EXISTS continuity_revisions (
                revision_id TEXT PRIMARY KEY,
                conversation_id INTEGER NOT NULL,
                narrative TEXT NOT NULL,
                anchors TEXT NOT NULL,
                focus TEXT NOT NULL,
                source_summary_ids TEXT NOT NULL,
                source_message_ids TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS continuity_documents (
                document_id TEXT PRIMARY KEY,
                conversation_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                head_commit_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(conversation_id, kind)
            );

            CREATE TABLE IF NOT EXISTS continuity_commits (
                commit_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                parent_commit_id TEXT,
                diff_text TEXT NOT NULL,
                rendered_text TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            "#,
        )?;
        Ok(())
    }

    pub fn add_message(&self, conversation_id: i64, role: &str, content: &str) -> Result<MessageRecord> {
        let _ = self.continuity_init_documents(conversation_id)?;
        let now = iso_now();
        let seq = self
            .conn
            .query_row(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE conversation_id = ?1",
                [conversation_id],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(1);
        let token_count = estimate_tokens(content) as i64;
        self.conn.execute(
            "INSERT INTO messages (conversation_id, seq, role, content, token_count, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![conversation_id, seq, role, content, token_count, now],
        )?;
        let message_id = self.conn.last_insert_rowid();
        self.conn.execute(
            "INSERT INTO messages_fts (rowid, content) VALUES (?1, ?2)",
            params![message_id, normalize_for_fts(content)],
        )?;
        let ordinal = self.next_context_ordinal(conversation_id)?;
        self.conn.execute(
            "INSERT INTO context_items (conversation_id, ordinal, item_type, message_id, summary_id, created_at)
             VALUES (?1, ?2, ?3, ?4, NULL, ?5)",
            params![conversation_id, ordinal, ContextItemType::Message.as_str(), message_id, iso_now()],
        )?;
        Ok(MessageRecord {
            message_id,
            conversation_id,
            seq,
            role: role.to_string(),
            content: content.to_string(),
            token_count,
            created_at: now,
        })
    }

    pub fn evaluate_compaction(&self, conversation_id: i64, token_budget: i64) -> Result<CompactionDecision> {
        let current_tokens = self.context_token_count(conversation_id)?;
        let threshold = ((token_budget as f64) * self.config.context_threshold).floor() as i64;
        Ok(CompactionDecision {
            should_compact: current_tokens > threshold,
            reason: if current_tokens > threshold {
                "threshold".to_string()
            } else {
                "none".to_string()
            },
            current_tokens,
            threshold,
        })
    }

    pub fn compact<S: Summarizer>(
        &self,
        conversation_id: i64,
        token_budget: i64,
        summarizer: &S,
        force: bool,
    ) -> Result<CompactionResult> {
        let tokens_before = self.context_token_count(conversation_id)?;
        let threshold = ((token_budget as f64) * self.config.context_threshold).floor() as i64;
        if !force && tokens_before <= threshold {
            return Ok(CompactionResult {
                action_taken: false,
                tokens_before,
                tokens_after: tokens_before,
                created_summary_ids: Vec::new(),
                rounds: 0,
            });
        }

        let mut created = Vec::new();
        let mut rounds = 0usize;
        let mut previous_tokens = tokens_before;

        while rounds < self.config.max_rounds {
            rounds += 1;
            let Some(summary_id) = self.compact_leaf_pass(conversation_id, summarizer, force)? else {
                break;
            };
            created.push(summary_id);

            let current = self.context_token_count(conversation_id)?;
            if !force && current <= threshold {
                break;
            }
            if current >= previous_tokens {
                break;
            }
            previous_tokens = current;
        }

        while rounds < self.config.max_rounds && (force || previous_tokens > threshold) {
            rounds += 1;
            let Some(summary_id) = self.compact_condensed_pass(conversation_id, summarizer)? else {
                break;
            };
            created.push(summary_id);

            let current = self.context_token_count(conversation_id)?;
            if !force && current <= threshold {
                break;
            }
            if current >= previous_tokens {
                break;
            }
            previous_tokens = current;
        }

        let tokens_after = self.context_token_count(conversation_id)?;
        Ok(CompactionResult {
            action_taken: !created.is_empty(),
            tokens_before,
            tokens_after,
            created_summary_ids: created,
            rounds,
        })
    }

    pub fn grep(
        &self,
        conversation_id: Option<i64>,
        scope: GrepScope,
        mode: GrepMode,
        query: &str,
        limit: usize,
    ) -> Result<GrepResult> {
        let messages = match scope {
            GrepScope::Messages | GrepScope::Both => self.search_messages(conversation_id, mode, query, limit)?,
            GrepScope::Summaries => Vec::new(),
        };
        let summaries = match scope {
            GrepScope::Summaries | GrepScope::Both => self.search_summaries(conversation_id, mode, query, limit)?,
            GrepScope::Messages => Vec::new(),
        };
        Ok(GrepResult {
            total_matches: messages.len() + summaries.len(),
            messages,
            summaries,
        })
    }

    pub fn describe(&self, id: &str) -> Result<Option<DescribeResult>> {
        let summary = self.get_summary(id)?;
        let Some(summary) = summary else {
            return Ok(None);
        };
        let parent_ids = self.summary_parent_ids(id)?;
        let child_ids = self.summary_child_ids(id)?;
        let message_ids = self.summary_message_ids(id)?;
        let subtree = self.summary_subtree(id)?;
        Ok(Some(DescribeResult::Summary(DescribeSummary {
            summary,
            parent_ids,
            child_ids,
            message_ids,
            subtree,
        })))
    }

    pub fn expand(
        &self,
        summary_id: &str,
        depth: usize,
        include_messages: bool,
        token_cap: i64,
    ) -> Result<ExpandResult> {
        let mut estimated = 0i64;
        let mut truncated = false;
        let mut children = Vec::new();
        let mut messages = Vec::new();
        let mut queue = vec![(summary_id.to_string(), 0usize)];

        while let Some((current, current_depth)) = queue.pop() {
            if current_depth >= depth {
                continue;
            }
            for child in self.child_summaries(&current)? {
                if estimated + child.token_count > token_cap {
                    truncated = true;
                    break;
                }
                estimated += child.token_count;
                children.push(ExpandChild {
                    summary_id: child.summary_id.clone(),
                    kind: child.kind,
                    content: child.content.clone(),
                    token_count: child.token_count,
                });
                queue.push((child.summary_id, current_depth + 1));
            }
            if truncated {
                break;
            }
        }

        if include_messages && !truncated {
            for message in self.messages_for_summary(summary_id)? {
                if estimated + message.token_count > token_cap {
                    truncated = true;
                    break;
                }
                estimated += message.token_count;
                messages.push(ExpandMessage {
                    message_id: message.message_id,
                    role: message.role,
                    content: message.content,
                    token_count: message.token_count,
                });
            }
        }

        Ok(ExpandResult {
            children,
            messages,
            estimated_tokens: estimated,
            truncated,
        })
    }

    pub fn snapshot(&self, conversation_id: i64) -> Result<LcmSnapshot> {
        let messages = self.messages_for_conversation(conversation_id)?;
        let summaries = self.summaries_for_conversation(conversation_id)?;
        let context_items = self
            .context_entries(conversation_id)?
            .into_iter()
            .map(|entry| ContextItemSnapshot {
                ordinal: entry.ordinal,
                item_type: entry.item_type,
                message_id: entry.message_id,
                summary_id: entry.summary_id,
                seq: entry.seq,
                depth: entry.depth,
                token_count: entry.token_count,
            })
            .collect();
        let summary_edges = self.summary_edges_for_conversation(conversation_id)?;
        let summary_messages = self.summary_message_links_for_conversation(conversation_id)?;
        Ok(LcmSnapshot {
            conversation_id,
            messages,
            summaries,
            context_items,
            summary_edges,
            summary_messages,
        })
    }

    pub fn refresh_continuity(&self, conversation_id: i64) -> Result<ContinuityRevision> {
        let _ = self.continuity_init_documents(conversation_id)?;
        self.latest_continuity(conversation_id)?
            .context("continuity documents missing after init")
    }

    pub fn latest_continuity(&self, conversation_id: i64) -> Result<Option<ContinuityRevision>> {
        let show_all = self.continuity_show_all(conversation_id)?;
        let snapshot = self.snapshot(conversation_id)?;
        let revision_id = continuity_heads_revision_id(
            conversation_id,
            &show_all.narrative.head_commit_id,
            &show_all.anchors.head_commit_id,
            &show_all.focus.head_commit_id,
        );
        let created_at = std::cmp::max(
            show_all.narrative.updated_at.clone(),
            std::cmp::max(show_all.anchors.updated_at.clone(), show_all.focus.updated_at.clone()),
        );
        Ok(Some(ContinuityRevision {
            revision_id,
            conversation_id,
            narrative: show_all.narrative.content,
            anchors: show_all.anchors.content,
            focus: show_all.focus.content,
            source_summary_ids: snapshot.summaries.iter().map(|summary| summary.summary_id.clone()).collect(),
            source_message_ids: snapshot.messages.iter().map(|message| message.message_id).collect(),
            created_at,
        }))
    }

    pub fn continuity_init_documents(&self, conversation_id: i64) -> Result<ContinuityShowAll> {
        let narrative = self.ensure_continuity_document(conversation_id, ContinuityKind::Narrative)?;
        let anchors = self.ensure_continuity_document(conversation_id, ContinuityKind::Anchors)?;
        let focus = self.ensure_continuity_document(conversation_id, ContinuityKind::Focus)?;
        Ok(ContinuityShowAll {
            conversation_id,
            narrative,
            anchors,
            focus,
        })
    }

    pub fn continuity_show(&self, conversation_id: i64, kind: ContinuityKind) -> Result<ContinuityDocumentState> {
        self.ensure_continuity_document(conversation_id, kind)
    }

    pub fn continuity_show_all(&self, conversation_id: i64) -> Result<ContinuityShowAll> {
        self.continuity_init_documents(conversation_id)
    }

    pub fn continuity_log(
        &self,
        conversation_id: i64,
        kind: Option<ContinuityKind>,
    ) -> Result<Vec<ContinuityCommitRecord>> {
        let mut out = Vec::new();
        let kinds = if let Some(kind) = kind {
            vec![kind]
        } else {
            vec![ContinuityKind::Narrative, ContinuityKind::Anchors, ContinuityKind::Focus]
        };
        for kind in kinds {
            let document = self.ensure_continuity_document(conversation_id, kind)?;
            let mut commits = self.continuity_commits_for_document(&document.head_commit_id, conversation_id, kind)?;
            out.append(&mut commits);
        }
        out.sort_by(|left, right| left.created_at.cmp(&right.created_at));
        Ok(out)
    }

    pub fn continuity_apply_diff(
        &self,
        conversation_id: i64,
        kind: ContinuityKind,
        diff_text: &str,
    ) -> Result<ContinuityDocumentState> {
        let document = self.ensure_continuity_document(conversation_id, kind)?;
        let rendered = apply_continuity_diff(&document.content, diff_text)?;
        let created_at = iso_now();
        let commit_id = continuity_commit_id(conversation_id, kind, diff_text, &rendered, &created_at);
        let document_id = continuity_document_id(conversation_id, kind);
        self.conn.execute(
            "INSERT INTO continuity_commits (commit_id, document_id, parent_commit_id, diff_text, rendered_text, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                commit_id,
                document_id,
                document.head_commit_id,
                diff_text,
                rendered,
                created_at
            ],
        )?;
        self.conn.execute(
            "UPDATE continuity_documents SET head_commit_id = ?1, updated_at = ?2 WHERE document_id = ?3",
            params![commit_id, created_at, document_id],
        )?;
        self.continuity_show(conversation_id, kind)
    }

    pub fn continuity_rebuild(
        &self,
        conversation_id: i64,
        kind: ContinuityKind,
    ) -> Result<ContinuityDocumentState> {
        let document_id = continuity_document_id(conversation_id, kind);
        let commits = self.continuity_commits_for_document_id(&document_id, conversation_id, kind)?;
        let base = continuity_template(kind).to_string();
        let rebuilt = commits.iter().skip(1).try_fold(base, |current, commit| {
            apply_continuity_diff(&current, &commit.diff_text)
        })?;
        let head_commit_id = commits
            .last()
            .map(|commit| commit.commit_id.clone())
            .unwrap_or_else(|| continuity_base_commit_id(conversation_id, kind));
        let updated_at = commits
            .last()
            .map(|commit| commit.created_at.clone())
            .unwrap_or_else(iso_now);
        Ok(ContinuityDocumentState {
            conversation_id,
            kind,
            head_commit_id,
            content: rebuilt,
            created_at: commits
                .first()
                .map(|commit| commit.created_at.clone())
                .unwrap_or_else(iso_now),
            updated_at,
        })
    }

    pub fn continuity_forgotten(
        &self,
        conversation_id: i64,
        kind: Option<ContinuityKind>,
        query: Option<&str>,
    ) -> Result<Vec<ContinuityForgottenEntry>> {
        let query_lower = query.map(|value| value.to_lowercase());
        let commits = self.continuity_log(conversation_id, kind)?;
        let mut out = Vec::new();
        for commit in commits {
            for line in removed_lines_from_diff(&commit.diff_text) {
                if query_lower
                    .as_ref()
                    .map(|needle| line.to_lowercase().contains(needle))
                    .unwrap_or(true)
                {
                    out.push(ContinuityForgottenEntry {
                        commit_id: commit.commit_id.clone(),
                        conversation_id,
                        kind: commit.kind,
                        line,
                        created_at: commit.created_at.clone(),
                    });
                }
            }
        }
        Ok(out)
    }

    pub fn continuity_build_prompt(
        &self,
        conversation_id: i64,
        kind: ContinuityKind,
    ) -> Result<ContinuityPromptPayload> {
        let document = self.ensure_continuity_document(conversation_id, kind)?;
        let snapshot = self.snapshot(conversation_id)?;
        let forgotten = self
            .continuity_forgotten(conversation_id, Some(kind), None)?
            .into_iter()
            .rev()
            .take(8)
            .map(|entry| entry.line)
            .collect::<Vec<_>>();
        let recent_messages = snapshot
            .messages
            .iter()
            .rev()
            .take(8)
            .map(|message| format!("[{} #{}] {}", message.role, message.seq, sentence_fragment(&message.content, 220)))
            .collect::<Vec<_>>();
        let recent_summaries = snapshot
            .summaries
            .iter()
            .rev()
            .take(4)
            .map(|summary| format!("[{} depth={}] {}", summary.kind.as_str(), summary.depth, sentence_fragment(&summary.content, 240)))
            .collect::<Vec<_>>();
        let prompt = build_continuity_prompt_text(
            conversation_id,
            kind,
            &document.content,
            &recent_messages,
            &recent_summaries,
            &forgotten,
        );

        Ok(ContinuityPromptPayload {
            conversation_id,
            kind,
            current_document: document.content,
            recent_messages,
            recent_summaries,
            forgotten_lines: forgotten,
            prompt,
        })
    }

    fn compact_leaf_pass<S: Summarizer>(
        &self,
        conversation_id: i64,
        summarizer: &S,
        _force: bool,
    ) -> Result<Option<String>> {
        let entries = self.context_entries(conversation_id)?;
        let message_entries: Vec<_> = entries
            .iter()
            .filter(|entry| entry.item_type == ContextItemType::Message)
            .cloned()
            .collect();
        if message_entries.len() <= self.config.fresh_tail_count {
            return Ok(None);
        }

        let tail_start_ordinal = if self.config.fresh_tail_count == 0 {
            i64::MAX
        } else {
            message_entries[message_entries.len() - self.config.fresh_tail_count].ordinal
        };
        let mut selected = Vec::new();
        let mut selected_tokens = 0i64;
        let mut started = false;
        for entry in entries {
            if entry.ordinal >= tail_start_ordinal {
                break;
            }
            if !started {
                if entry.item_type != ContextItemType::Message || entry.message_id.is_none() {
                    continue;
                }
                started = true;
            } else if entry.item_type != ContextItemType::Message || entry.message_id.is_none() {
                break;
            }

            if selected_tokens > 0 && selected_tokens + entry.token_count > self.config.leaf_chunk_tokens {
                break;
            }
            selected_tokens += entry.token_count;
            selected.push(entry.clone());
            if selected_tokens >= self.config.leaf_chunk_tokens {
                break;
            }
        }
        if selected.is_empty() {
            return Ok(None);
        }

        let first_ordinal = selected[0].ordinal;
        let source_text = self.leaf_source_text(&selected)?;
        let content = self
            .summarize_with_escalation(
                SummaryKind::Leaf,
                0,
                &source_text,
                self.config.leaf_target_tokens,
                summarizer,
            )?
            .content;
        let source_message_token_count = selected
            .iter()
            .filter_map(|entry| entry.message_id)
            .map(|message_id| self.get_message(message_id).map(|message| message.token_count))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .sum();
        let summary_id = self.insert_summary(
            conversation_id,
            SummaryKind::Leaf,
            0,
            &content,
            0,
            0,
            source_message_token_count,
            &[],
            selected.iter().filter_map(|entry| entry.message_id).collect(),
            first_ordinal,
            selected.iter().map(|entry| entry.ordinal).collect(),
        )?;
        Ok(Some(summary_id))
    }

    fn compact_condensed_pass<S: Summarizer>(&self, conversation_id: i64, summarizer: &S) -> Result<Option<String>> {
        let entries = self.context_entries(conversation_id)?;
        let message_entries: Vec<_> = entries
            .iter()
            .filter(|entry| entry.item_type == ContextItemType::Message)
            .cloned()
            .collect();
        let tail_start_ordinal = if self.config.fresh_tail_count == 0 {
            None
        } else if message_entries.len() > self.config.fresh_tail_count {
            Some(message_entries[message_entries.len() - self.config.fresh_tail_count].ordinal)
        } else {
            None
        };
        let eligible_entries: Vec<_> = entries
            .into_iter()
            .take_while(|entry| tail_start_ordinal.map(|ordinal| entry.ordinal < ordinal).unwrap_or(true))
            .collect();
        let min_chunk_tokens = self.resolve_condensed_min_chunk_tokens();

        for depth in self.distinct_summary_depths(&eligible_entries)? {
            let same_depth = self.select_oldest_summary_chunk_at_depth(&eligible_entries, depth)?;
            if same_depth.len() < self.config.condensed_min_fanout {
                continue;
            }

            let token_count: i64 = same_depth.iter().map(|entry| entry.token_count).sum();
            if token_count < min_chunk_tokens {
                continue;
            }

            let first_ordinal = same_depth[0].ordinal;
            let child_ids: Vec<String> = same_depth
                .iter()
                .filter_map(|entry| entry.summary_id.clone())
                .collect();
            let source_text = self.condensed_source_text(&child_ids)?;
            let source_message_token_count = child_ids
                .iter()
                .map(|id| self.summary_source_message_token_count(id))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .sum();
            let descendant_count = child_ids
                .iter()
                .map(|id| Ok(self.summary_descendant_count(id)? + 1))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .sum();
            let descendant_tokens = child_ids
                .iter()
                .map(|id| Ok(self.summary_token_count(id)? + self.summary_descendant_token_count(id)?))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .sum();
            let content = self
                .summarize_with_escalation(
                    SummaryKind::Condensed,
                    depth + 1,
                    &source_text,
                    self.config.condensed_target_tokens,
                    summarizer,
                )?
                .content;
            let summary_id = self.insert_summary(
                conversation_id,
                SummaryKind::Condensed,
                depth + 1,
                &content,
                descendant_count,
                descendant_tokens,
                source_message_token_count,
                &child_ids,
                Vec::new(),
                first_ordinal,
                same_depth.iter().map(|entry| entry.ordinal).collect(),
            )?;
            return Ok(Some(summary_id));
        }

        Ok(None)
    }

    fn distinct_summary_depths(&self, entries: &[ContextEntry]) -> Result<Vec<i64>> {
        let mut depths = Vec::new();
        for entry in entries {
            if entry.item_type != ContextItemType::Summary {
                continue;
            }
            let Some(summary_id) = entry.summary_id.as_deref() else {
                continue;
            };
            let Some(summary) = self.get_summary(summary_id)? else {
                continue;
            };
            if !depths.contains(&summary.depth) {
                depths.push(summary.depth);
            }
        }
        depths.sort_unstable();
        Ok(depths)
    }

    fn select_oldest_summary_chunk_at_depth(
        &self,
        entries: &[ContextEntry],
        target_depth: i64,
    ) -> Result<Vec<ContextEntry>> {
        let mut chunk = Vec::new();
        let mut token_count = 0i64;
        for entry in entries {
            if entry.item_type != ContextItemType::Summary {
                if !chunk.is_empty() {
                    break;
                }
                continue;
            }
            let Some(summary_id) = entry.summary_id.as_deref() else {
                if !chunk.is_empty() {
                    break;
                }
                continue;
            };
            let Some(summary) = self.get_summary(summary_id)? else {
                if !chunk.is_empty() {
                    break;
                }
                continue;
            };
            if summary.depth != target_depth {
                if !chunk.is_empty() {
                    break;
                }
                continue;
            }
            if token_count > 0 && token_count + summary.token_count > self.config.leaf_chunk_tokens {
                break;
            }
            token_count += summary.token_count;
            chunk.push(entry.clone());
            if token_count >= self.config.leaf_chunk_tokens {
                break;
            }
        }
        Ok(chunk)
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_summary(
        &self,
        conversation_id: i64,
        kind: SummaryKind,
        depth: i64,
        content: &str,
        descendant_count: i64,
        descendant_token_count: i64,
        source_message_token_count: i64,
        child_summary_ids: &[String],
        message_ids: Vec<i64>,
        ordinal: i64,
        replaced_ordinals: Vec<i64>,
    ) -> Result<String> {
        let created_at = iso_now();
        let summary_id = summary_id_for(conversation_id, content, depth);
        let token_count = estimate_tokens(content) as i64;
        self.conn.execute(
            "INSERT INTO summaries (
                summary_id, conversation_id, kind, depth, content, token_count,
                descendant_count, descendant_token_count, source_message_token_count, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                summary_id,
                conversation_id,
                kind.as_str(),
                depth,
                content,
                token_count,
                descendant_count,
                descendant_token_count,
                source_message_token_count,
                created_at
            ],
        )?;
        self.conn.execute(
            "INSERT INTO summaries_fts (rowid, summary_id, content)
             VALUES ((SELECT rowid FROM summaries WHERE summary_id = ?1), ?1, ?2)",
            params![summary_id, normalize_for_fts(content)],
        )?;

        for child_id in child_summary_ids {
            self.conn.execute(
                "INSERT OR IGNORE INTO summary_edges (parent_summary_id, child_summary_id)
                 VALUES (?1, ?2)",
                params![summary_id, child_id],
            )?;
        }
        for message_id in message_ids {
            self.conn.execute(
                "INSERT OR IGNORE INTO summary_messages (summary_id, message_id)
                 VALUES (?1, ?2)",
                params![summary_id, message_id],
            )?;
        }

        for old_ordinal in replaced_ordinals {
            self.conn.execute(
                "DELETE FROM context_items WHERE conversation_id = ?1 AND ordinal = ?2",
                params![conversation_id, old_ordinal],
            )?;
        }
        self.conn.execute(
            "INSERT INTO context_items (conversation_id, ordinal, item_type, message_id, summary_id, created_at)
             VALUES (?1, ?2, ?3, NULL, ?4, ?5)",
            params![conversation_id, ordinal, ContextItemType::Summary.as_str(), summary_id, iso_now()],
        )?;
        self.resequence_context_items(conversation_id)?;
        Ok(summary_id)
    }

    fn resequence_context_items(&self, conversation_id: i64) -> Result<()> {
        let ordinals = {
            let mut stmt = self.conn.prepare(
                "SELECT ordinal FROM context_items WHERE conversation_id = ?1 ORDER BY ordinal ASC",
            )?;
            let rows = stmt.query_map([conversation_id], |row| row.get::<_, i64>(0))?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };
        for (new_ordinal, old_ordinal) in ordinals.into_iter().enumerate() {
            self.conn.execute(
                "UPDATE context_items SET ordinal = -?1 - 1 WHERE conversation_id = ?2 AND ordinal = ?3",
                params![new_ordinal as i64, conversation_id, old_ordinal],
            )?;
        }
        self.conn.execute(
            "UPDATE context_items SET ordinal = (-ordinal) - 1 WHERE conversation_id = ?1",
            [conversation_id],
        )?;
        Ok(())
    }

    fn context_entries(&self, conversation_id: i64) -> Result<Vec<ContextEntry>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT
                ci.ordinal,
                ci.item_type,
                ci.message_id,
                ci.summary_id,
                COALESCE(m.seq, ci.ordinal) AS seq,
                COALESCE(s.depth, 0) AS depth,
                COALESCE(m.token_count, s.token_count, 0) AS token_count
            FROM context_items ci
            LEFT JOIN messages m ON m.message_id = ci.message_id
            LEFT JOIN summaries s ON s.summary_id = ci.summary_id
            WHERE ci.conversation_id = ?1
            ORDER BY ci.ordinal ASC
            "#,
        )?;
        let rows = stmt.query_map([conversation_id], |row| {
            Ok(ContextEntry {
                ordinal: row.get(0)?,
                item_type: match row.get::<_, String>(1)?.as_str() {
                    "message" => ContextItemType::Message,
                    _ => ContextItemType::Summary,
                },
                message_id: row.get(2)?,
                summary_id: row.get(3)?,
                seq: row.get(4)?,
                depth: row.get(5)?,
                token_count: row.get(6)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn context_token_count(&self, conversation_id: i64) -> Result<i64> {
        Ok(self
            .conn
            .query_row(
                r#"
                SELECT COALESCE(SUM(COALESCE(m.token_count, s.token_count, 0)), 0)
                FROM context_items ci
                LEFT JOIN messages m ON m.message_id = ci.message_id
                LEFT JOIN summaries s ON s.summary_id = ci.summary_id
                WHERE ci.conversation_id = ?1
                "#,
                [conversation_id],
                |row| row.get(0),
            )
            .unwrap_or(0))
    }

    fn next_context_ordinal(&self, conversation_id: i64) -> Result<i64> {
        Ok(self
            .conn
            .query_row(
                "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM context_items WHERE conversation_id = ?1",
                [conversation_id],
                |row| row.get(0),
            )
            .unwrap_or(1))
    }

    fn leaf_source_text(&self, entries: &[ContextEntry]) -> Result<String> {
        let mut chunks = Vec::new();
        for entry in entries {
            let Some(message_id) = entry.message_id else {
                continue;
            };
            let message = self.get_message(message_id)?;
            chunks.push(format!(
                "[{}]\n{}",
                format_summary_timestamp(&message.created_at),
                message.content
            ));
        }
        Ok(chunks.join("\n\n"))
    }

    fn condensed_source_text(&self, summary_ids: &[String]) -> Result<String> {
        let mut chunks = Vec::new();
        for summary_id in summary_ids {
            let Some(summary) = self.get_summary(summary_id)? else {
                continue;
            };
            let timestamp = format_summary_timestamp(&summary.created_at);
            chunks.push(format!("[{timestamp} - {timestamp}]\n{}", summary.content));
        }
        Ok(chunks.join("\n\n"))
    }

    fn summarize_with_escalation<S: Summarizer>(
        &self,
        kind: SummaryKind,
        depth: i64,
        source_text: &str,
        target_tokens: usize,
        summarizer: &S,
    ) -> Result<EscalatedSummary> {
        let trimmed = source_text.trim();
        if trimmed.is_empty() {
            return Ok(EscalatedSummary {
                content: "[Truncated from 0 tokens]".to_string(),
            });
        }

        let input_tokens = estimate_tokens(trimmed) as i64;
        let lines: Vec<String> = trimmed.lines().map(str::to_string).collect();
        let summary = summarizer.summarize(kind, depth, &lines, target_tokens)?;
        let content = if summary.trim().is_empty() || estimate_tokens(&summary) as i64 >= input_tokens {
            build_deterministic_fallback(trimmed, input_tokens)
        } else {
            summary.trim().to_string()
        };
        Ok(EscalatedSummary { content })
    }

    fn get_message(&self, message_id: i64) -> Result<MessageRecord> {
        self.conn.query_row(
            "SELECT message_id, conversation_id, seq, role, content, token_count, created_at
             FROM messages WHERE message_id = ?1",
            [message_id],
            |row| {
                Ok(MessageRecord {
                    message_id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    seq: row.get(2)?,
                    role: row.get(3)?,
                    content: row.get(4)?,
                    token_count: row.get(5)?,
                    created_at: row.get(6)?,
                })
            },
        ).context("message not found")
    }

    fn messages_for_conversation(&self, conversation_id: i64) -> Result<Vec<MessageRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT message_id, conversation_id, seq, role, content, token_count, created_at
             FROM messages WHERE conversation_id = ?1 ORDER BY seq ASC",
        )?;
        let rows = stmt.query_map([conversation_id], |row| {
            Ok(MessageRecord {
                message_id: row.get(0)?,
                conversation_id: row.get(1)?,
                seq: row.get(2)?,
                role: row.get(3)?,
                content: row.get(4)?,
                token_count: row.get(5)?,
                created_at: row.get(6)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn get_summary(&self, summary_id: &str) -> Result<Option<SummaryRecord>> {
        self.conn.query_row(
            "SELECT summary_id, conversation_id, kind, depth, content, token_count,
                    descendant_count, descendant_token_count, source_message_token_count, created_at
             FROM summaries WHERE summary_id = ?1",
            [summary_id],
            |row| {
                Ok(SummaryRecord {
                    summary_id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    kind: parse_summary_kind(&row.get::<_, String>(2)?),
                    depth: row.get(3)?,
                    content: row.get(4)?,
                    token_count: row.get(5)?,
                    descendant_count: row.get(6)?,
                    descendant_token_count: row.get(7)?,
                    source_message_token_count: row.get(8)?,
                    created_at: row.get(9)?,
                })
            },
        ).optional().map_err(Into::into)
    }

    fn summaries_for_conversation(&self, conversation_id: i64) -> Result<Vec<SummaryRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT summary_id, conversation_id, kind, depth, content, token_count,
                    descendant_count, descendant_token_count, source_message_token_count, created_at
             FROM summaries WHERE conversation_id = ?1 ORDER BY depth ASC, created_at ASC",
        )?;
        let rows = stmt.query_map([conversation_id], |row| {
            Ok(SummaryRecord {
                summary_id: row.get(0)?,
                conversation_id: row.get(1)?,
                kind: parse_summary_kind(&row.get::<_, String>(2)?),
                depth: row.get(3)?,
                content: row.get(4)?,
                token_count: row.get(5)?,
                descendant_count: row.get(6)?,
                descendant_token_count: row.get(7)?,
                source_message_token_count: row.get(8)?,
                created_at: row.get(9)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn summary_edges_for_conversation(&self, conversation_id: i64) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT e.parent_summary_id, e.child_summary_id
            FROM summary_edges e
            JOIN summaries parent ON parent.summary_id = e.parent_summary_id
            JOIN summaries child ON child.summary_id = e.child_summary_id
            WHERE parent.conversation_id = ?1 AND child.conversation_id = ?1
            ORDER BY e.parent_summary_id ASC, e.child_summary_id ASC
            "#,
        )?;
        let rows = stmt.query_map([conversation_id], |row| Ok((row.get(0)?, row.get(1)?)))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn summary_message_links_for_conversation(&self, conversation_id: i64) -> Result<Vec<(String, i64)>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT sm.summary_id, sm.message_id
            FROM summary_messages sm
            JOIN summaries s ON s.summary_id = sm.summary_id
            WHERE s.conversation_id = ?1
            ORDER BY sm.summary_id ASC, sm.message_id ASC
            "#,
        )?;
        let rows = stmt.query_map([conversation_id], |row| Ok((row.get(0)?, row.get(1)?)))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn summary_parent_ids(&self, summary_id: &str) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT parent_summary_id FROM summary_edges WHERE child_summary_id = ?1 ORDER BY parent_summary_id",
        )?;
        let rows = stmt.query_map([summary_id], |row| row.get(0))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn summary_child_ids(&self, summary_id: &str) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT child_summary_id FROM summary_edges WHERE parent_summary_id = ?1 ORDER BY child_summary_id",
        )?;
        let rows = stmt.query_map([summary_id], |row| row.get(0))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn summary_message_ids(&self, summary_id: &str) -> Result<Vec<i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT message_id FROM summary_messages WHERE summary_id = ?1 ORDER BY message_id",
        )?;
        let rows = stmt.query_map([summary_id], |row| row.get(0))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn child_summaries(&self, summary_id: &str) -> Result<Vec<SummaryRecord>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT s.summary_id, s.conversation_id, s.kind, s.depth, s.content, s.token_count,
                   s.descendant_count, s.descendant_token_count, s.source_message_token_count, s.created_at
            FROM summary_edges e
            JOIN summaries s ON s.summary_id = e.child_summary_id
            WHERE e.parent_summary_id = ?1
            ORDER BY s.depth ASC, s.created_at ASC
            "#,
        )?;
        let rows = stmt.query_map([summary_id], |row| {
            Ok(SummaryRecord {
                summary_id: row.get(0)?,
                conversation_id: row.get(1)?,
                kind: parse_summary_kind(&row.get::<_, String>(2)?),
                depth: row.get(3)?,
                content: row.get(4)?,
                token_count: row.get(5)?,
                descendant_count: row.get(6)?,
                descendant_token_count: row.get(7)?,
                source_message_token_count: row.get(8)?,
                created_at: row.get(9)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn messages_for_summary(&self, summary_id: &str) -> Result<Vec<MessageRecord>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT m.message_id, m.conversation_id, m.seq, m.role, m.content, m.token_count, m.created_at
            FROM summary_messages sm
            JOIN messages m ON m.message_id = sm.message_id
            WHERE sm.summary_id = ?1
            ORDER BY m.seq ASC
            "#,
        )?;
        let rows = stmt.query_map([summary_id], |row| {
            Ok(MessageRecord {
                message_id: row.get(0)?,
                conversation_id: row.get(1)?,
                seq: row.get(2)?,
                role: row.get(3)?,
                content: row.get(4)?,
                token_count: row.get(5)?,
                created_at: row.get(6)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn summary_descendant_count(&self, summary_id: &str) -> Result<i64> {
        Ok(self
            .conn
            .query_row(
                "SELECT descendant_count FROM summaries WHERE summary_id = ?1",
                [summary_id],
                |row| row.get(0),
            )
            .unwrap_or(0))
    }

    fn summary_token_count(&self, summary_id: &str) -> Result<i64> {
        Ok(self
            .conn
            .query_row(
                "SELECT token_count FROM summaries WHERE summary_id = ?1",
                [summary_id],
                |row| row.get(0),
            )
            .unwrap_or(0))
    }

    fn summary_descendant_token_count(&self, summary_id: &str) -> Result<i64> {
        Ok(self
            .conn
            .query_row(
                "SELECT descendant_token_count FROM summaries WHERE summary_id = ?1",
                [summary_id],
                |row| row.get(0),
            )
            .unwrap_or(0))
    }

    fn resolve_condensed_min_chunk_tokens(&self) -> i64 {
        let ratio_floor = ((self.config.leaf_chunk_tokens as f64) * CONDENSED_MIN_INPUT_RATIO).floor() as i64;
        std::cmp::max(self.config.condensed_target_tokens as i64, ratio_floor)
    }

    fn ensure_continuity_document(
        &self,
        conversation_id: i64,
        kind: ContinuityKind,
    ) -> Result<ContinuityDocumentState> {
        if let Some(state) = self.fetch_continuity_document(conversation_id, kind)? {
            return Ok(state);
        }

        let document_id = continuity_document_id(conversation_id, kind);
        let created_at = iso_now();
        let template = continuity_template(kind).to_string();
        let base_commit_id = continuity_base_commit_id(conversation_id, kind);
        self.conn.execute(
            "INSERT INTO continuity_commits (commit_id, document_id, parent_commit_id, diff_text, rendered_text, created_at)
             VALUES (?1, ?2, NULL, ?3, ?4, ?5)",
            params![base_commit_id, document_id, "", template, created_at],
        )?;
        self.conn.execute(
            "INSERT INTO continuity_documents (document_id, conversation_id, kind, head_commit_id, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
            params![document_id, conversation_id, kind.as_str(), base_commit_id, created_at],
        )?;

        self.fetch_continuity_document(conversation_id, kind)?
            .context("continuity document missing after init")
    }

    fn fetch_continuity_document(
        &self,
        conversation_id: i64,
        kind: ContinuityKind,
    ) -> Result<Option<ContinuityDocumentState>> {
        self.conn
            .query_row(
                "SELECT d.head_commit_id, c.rendered_text, d.created_at, d.updated_at
                 FROM continuity_documents d
                 JOIN continuity_commits c ON c.commit_id = d.head_commit_id
                 WHERE d.conversation_id = ?1 AND d.kind = ?2",
                params![conversation_id, kind.as_str()],
                |row| {
                    Ok(ContinuityDocumentState {
                        conversation_id,
                        kind,
                        head_commit_id: row.get(0)?,
                        content: row.get(1)?,
                        created_at: row.get(2)?,
                        updated_at: row.get(3)?,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    fn continuity_commits_for_document(
        &self,
        _head_commit_id: &str,
        conversation_id: i64,
        kind: ContinuityKind,
    ) -> Result<Vec<ContinuityCommitRecord>> {
        let document_id = continuity_document_id(conversation_id, kind);
        self.continuity_commits_for_document_id(&document_id, conversation_id, kind)
    }

    fn continuity_commits_for_document_id(
        &self,
        document_id: &str,
        conversation_id: i64,
        kind: ContinuityKind,
    ) -> Result<Vec<ContinuityCommitRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT commit_id, parent_commit_id, diff_text, rendered_text, created_at
             FROM continuity_commits
             WHERE document_id = ?1
             ORDER BY created_at ASC",
        )?;
        let rows = stmt.query_map([document_id], |row| {
            Ok(ContinuityCommitRecord {
                commit_id: row.get(0)?,
                conversation_id,
                kind,
                parent_commit_id: row.get(1)?,
                diff_text: row.get(2)?,
                rendered_text: row.get(3)?,
                created_at: row.get(4)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn summary_source_message_token_count(&self, summary_id: &str) -> Result<i64> {
        Ok(self
            .conn
            .query_row(
                "SELECT source_message_token_count FROM summaries WHERE summary_id = ?1",
                [summary_id],
                |row| row.get(0),
            )
            .unwrap_or(0))
    }

    fn summary_subtree(&self, summary_id: &str) -> Result<Vec<SummarySubtreeNode>> {
        let mut stmt = self.conn.prepare(
            r#"
            WITH RECURSIVE subtree(summary_id, parent_summary_id, depth_from_root, path) AS (
                SELECT s.summary_id, NULL, 0, s.summary_id
                FROM summaries s
                WHERE s.summary_id = ?1
                UNION ALL
                SELECT child.summary_id, edge.parent_summary_id, subtree.depth_from_root + 1,
                       subtree.path || '>' || child.summary_id
                FROM subtree
                JOIN summary_edges edge ON edge.parent_summary_id = subtree.summary_id
                JOIN summaries child ON child.summary_id = edge.child_summary_id
            )
            SELECT
                subtree.summary_id,
                subtree.parent_summary_id,
                subtree.depth_from_root,
                s.kind,
                s.depth,
                s.token_count,
                s.descendant_count,
                s.descendant_token_count,
                s.source_message_token_count,
                (
                    SELECT COUNT(*)
                    FROM summary_edges edge2
                    WHERE edge2.parent_summary_id = subtree.summary_id
                ) AS child_count,
                subtree.path,
                s.created_at
            FROM subtree
            JOIN summaries s ON s.summary_id = subtree.summary_id
            ORDER BY subtree.depth_from_root ASC, subtree.path ASC
            "#,
        )?;
        let rows = stmt.query_map([summary_id], |row| {
            Ok(SummarySubtreeNode {
                summary_id: row.get(0)?,
                parent_summary_id: row.get(1)?,
                depth_from_root: row.get(2)?,
                kind: parse_summary_kind(&row.get::<_, String>(3)?),
                depth: row.get(4)?,
                token_count: row.get(5)?,
                descendant_count: row.get(6)?,
                descendant_token_count: row.get(7)?,
                source_message_token_count: row.get(8)?,
                child_count: row.get(9)?,
                path: row.get(10)?,
                created_at: row.get(11)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn search_messages(
        &self,
        conversation_id: Option<i64>,
        mode: GrepMode,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MessageSearchResult>> {
        match mode {
            GrepMode::FullText => self.search_messages_fts(conversation_id, query, limit),
            GrepMode::Regex => self.search_messages_regex(conversation_id, query, limit),
        }
    }

    fn search_summaries(
        &self,
        conversation_id: Option<i64>,
        mode: GrepMode,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SummarySearchResult>> {
        match mode {
            GrepMode::FullText => self.search_summaries_fts(conversation_id, query, limit),
            GrepMode::Regex => self.search_summaries_regex(conversation_id, query, limit),
        }
    }

    fn search_messages_fts(&self, conversation_id: Option<i64>, query: &str, limit: usize) -> Result<Vec<MessageSearchResult>> {
        let sql = if conversation_id.is_some() {
            r#"
            SELECT m.message_id, m.conversation_id, m.role, m.content, m.created_at
            FROM messages_fts f
            JOIN messages m ON m.rowid = f.rowid
            WHERE messages_fts MATCH ?1 AND m.conversation_id = ?2
            ORDER BY m.created_at DESC
            LIMIT ?3
            "#
        } else {
            r#"
            SELECT m.message_id, m.conversation_id, m.role, m.content, m.created_at
            FROM messages_fts f
            JOIN messages m ON m.rowid = f.rowid
            WHERE messages_fts MATCH ?1
            ORDER BY m.created_at DESC
            LIMIT ?2
            "#
        };
        let mut stmt = self.conn.prepare(sql)?;
        let rows = if let Some(conversation_id) = conversation_id {
            stmt.query_map(params![sanitize_fts_query(query), conversation_id, limit as i64], |row| {
                Ok(MessageSearchResult {
                    message_id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    role: row.get(2)?,
                    snippet: snippet(&row.get::<_, String>(3)?, query),
                    created_at: row.get(4)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?
        } else {
            stmt.query_map(params![sanitize_fts_query(query), limit as i64], |row| {
                Ok(MessageSearchResult {
                    message_id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    role: row.get(2)?,
                    snippet: snippet(&row.get::<_, String>(3)?, query),
                    created_at: row.get(4)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?
        };
        Ok(rows)
    }

    fn search_messages_regex(&self, conversation_id: Option<i64>, query: &str, limit: usize) -> Result<Vec<MessageSearchResult>> {
        let regex = Regex::new(query).with_context(|| format!("invalid regex: {query}"))?;
        let mut stmt = if conversation_id.is_some() {
            self.conn.prepare(
                "SELECT message_id, conversation_id, role, content, created_at
                 FROM messages WHERE conversation_id = ?1 ORDER BY created_at DESC",
            )?
        } else {
            self.conn.prepare(
                "SELECT message_id, conversation_id, role, content, created_at
                 FROM messages ORDER BY created_at DESC",
            )?
        };
        let mut out = Vec::new();
        if let Some(conversation_id) = conversation_id {
            let rows = stmt.query_map([conversation_id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                ))
            })?;
            for row in rows {
                let (message_id, conversation_id, role, content, created_at) = row?;
                if regex.is_match(&content) {
                    out.push(MessageSearchResult {
                        message_id,
                        conversation_id,
                        role,
                        snippet: snippet(&content, query),
                        created_at,
                    });
                    if out.len() >= limit {
                        break;
                    }
                }
            }
        } else {
            let rows = stmt.query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                ))
            })?;
            for row in rows {
                let (message_id, conversation_id, role, content, created_at) = row?;
                if regex.is_match(&content) {
                    out.push(MessageSearchResult {
                        message_id,
                        conversation_id,
                        role,
                        snippet: snippet(&content, query),
                        created_at,
                    });
                    if out.len() >= limit {
                        break;
                    }
                }
            }
        }
        Ok(out)
    }

    fn search_summaries_fts(&self, conversation_id: Option<i64>, query: &str, limit: usize) -> Result<Vec<SummarySearchResult>> {
        let sql = if conversation_id.is_some() {
            r#"
            SELECT s.summary_id, s.conversation_id, s.kind, s.content, s.created_at
            FROM summaries_fts f
            JOIN summaries s ON s.rowid = f.rowid
            WHERE summaries_fts MATCH ?1 AND s.conversation_id = ?2
            ORDER BY s.created_at DESC
            LIMIT ?3
            "#
        } else {
            r#"
            SELECT s.summary_id, s.conversation_id, s.kind, s.content, s.created_at
            FROM summaries_fts f
            JOIN summaries s ON s.rowid = f.rowid
            WHERE summaries_fts MATCH ?1
            ORDER BY s.created_at DESC
            LIMIT ?2
            "#
        };
        let mut stmt = self.conn.prepare(sql)?;
        let rows = if let Some(conversation_id) = conversation_id {
            stmt.query_map(params![sanitize_fts_query(query), conversation_id, limit as i64], |row| {
                Ok(SummarySearchResult {
                    summary_id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    kind: parse_summary_kind(&row.get::<_, String>(2)?),
                    snippet: snippet(&row.get::<_, String>(3)?, query),
                    created_at: row.get(4)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?
        } else {
            stmt.query_map(params![sanitize_fts_query(query), limit as i64], |row| {
                Ok(SummarySearchResult {
                    summary_id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    kind: parse_summary_kind(&row.get::<_, String>(2)?),
                    snippet: snippet(&row.get::<_, String>(3)?, query),
                    created_at: row.get(4)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?
        };
        Ok(rows)
    }

    fn search_summaries_regex(&self, conversation_id: Option<i64>, query: &str, limit: usize) -> Result<Vec<SummarySearchResult>> {
        let regex = Regex::new(query).with_context(|| format!("invalid regex: {query}"))?;
        let mut stmt = if conversation_id.is_some() {
            self.conn.prepare(
                "SELECT summary_id, conversation_id, kind, content, created_at
                 FROM summaries WHERE conversation_id = ?1 ORDER BY created_at DESC",
            )?
        } else {
            self.conn.prepare(
                "SELECT summary_id, conversation_id, kind, content, created_at
                 FROM summaries ORDER BY created_at DESC",
            )?
        };
        let mut out = Vec::new();
        if let Some(conversation_id) = conversation_id {
            let rows = stmt.query_map([conversation_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                ))
            })?;
            for row in rows {
                let (summary_id, conversation_id, kind, content, created_at) = row?;
                if regex.is_match(&content) {
                    out.push(SummarySearchResult {
                        summary_id,
                        conversation_id,
                        kind: parse_summary_kind(&kind),
                        snippet: snippet(&content, query),
                        created_at,
                    });
                    if out.len() >= limit {
                        break;
                    }
                }
            }
        } else {
            let rows = stmt.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                ))
            })?;
            for row in rows {
                let (summary_id, conversation_id, kind, content, created_at) = row?;
                if regex.is_match(&content) {
                    out.push(SummarySearchResult {
                        summary_id,
                        conversation_id,
                        kind: parse_summary_kind(&kind),
                        snippet: snippet(&content, query),
                        created_at,
                    });
                    if out.len() >= limit {
                        break;
                    }
                }
            }
        }
        Ok(out)
    }
}

pub fn run_init(db_path: &Path) -> Result<()> {
    let _ = LcmEngine::open(db_path, LcmConfig::default())?;
    Ok(())
}

pub fn run_add_message(db_path: &Path, conversation_id: i64, role: &str, content: &str) -> Result<MessageRecord> {
    let engine = LcmEngine::open(db_path, LcmConfig::default())?;
    engine.add_message(conversation_id, role, content)
}

pub fn run_compact(db_path: &Path, conversation_id: i64, token_budget: i64, force: bool) -> Result<CompactionResult> {
    let engine = LcmEngine::open(db_path, LcmConfig::default())?;
    engine.compact(conversation_id, token_budget, &HeuristicSummarizer, force)
}

pub fn run_grep(
    db_path: &Path,
    conversation_id: Option<i64>,
    scope: &str,
    mode: &str,
    query: &str,
    limit: usize,
) -> Result<GrepResult> {
    let engine = LcmEngine::open(db_path, LcmConfig::default())?;
    engine.grep(conversation_id, GrepScope::parse(scope)?, GrepMode::parse(mode)?, query, limit)
}

pub fn run_describe(db_path: &Path, id: &str) -> Result<Option<DescribeResult>> {
    let engine = LcmEngine::open(db_path, LcmConfig::default())?;
    engine.describe(id)
}

pub fn run_expand(
    db_path: &Path,
    summary_id: &str,
    depth: usize,
    include_messages: bool,
    token_cap: i64,
) -> Result<ExpandResult> {
    let engine = LcmEngine::open(db_path, LcmConfig::default())?;
    engine.expand(summary_id, depth, include_messages, token_cap)
}

pub fn run_dump(db_path: &Path, conversation_id: i64) -> Result<LcmSnapshot> {
    let engine = LcmEngine::open(db_path, LcmConfig::default())?;
    engine.snapshot(conversation_id)
}

pub fn run_refresh_continuity(db_path: &Path, conversation_id: i64) -> Result<ContinuityRevision> {
    let engine = LcmEngine::open(db_path, LcmConfig::default())?;
    engine.refresh_continuity(conversation_id)
}

pub fn run_show_continuity(db_path: &Path, conversation_id: i64) -> Result<Option<ContinuityRevision>> {
    let engine = LcmEngine::open(db_path, LcmConfig::default())?;
    engine.latest_continuity(conversation_id)
}

pub fn run_continuity_init(db_path: &Path, conversation_id: i64) -> Result<ContinuityShowAll> {
    let engine = LcmEngine::open(db_path, LcmConfig::default())?;
    engine.continuity_init_documents(conversation_id)
}

pub fn run_continuity_show(
    db_path: &Path,
    conversation_id: i64,
    kind: Option<&str>,
) -> Result<serde_json::Value> {
    let engine = LcmEngine::open(db_path, LcmConfig::default())?;
    if let Some(kind) = kind {
        Ok(serde_json::to_value(engine.continuity_show(conversation_id, ContinuityKind::parse(kind)?)?)?)
    } else {
        Ok(serde_json::to_value(engine.continuity_show_all(conversation_id)?)?)
    }
}

pub fn run_continuity_apply(
    db_path: &Path,
    conversation_id: i64,
    kind: &str,
    diff_path: &Path,
) -> Result<ContinuityDocumentState> {
    let engine = LcmEngine::open(db_path, LcmConfig::default())?;
    let diff_text = std::fs::read_to_string(diff_path)
        .with_context(|| format!("failed to read continuity diff {}", diff_path.display()))?;
    engine.continuity_apply_diff(conversation_id, ContinuityKind::parse(kind)?, &diff_text)
}

pub fn run_continuity_log(
    db_path: &Path,
    conversation_id: i64,
    kind: Option<&str>,
) -> Result<Vec<ContinuityCommitRecord>> {
    let engine = LcmEngine::open(db_path, LcmConfig::default())?;
    engine.continuity_log(conversation_id, kind.map(ContinuityKind::parse).transpose()?)
}

pub fn run_continuity_rebuild(
    db_path: &Path,
    conversation_id: i64,
    kind: &str,
) -> Result<ContinuityDocumentState> {
    let engine = LcmEngine::open(db_path, LcmConfig::default())?;
    engine.continuity_rebuild(conversation_id, ContinuityKind::parse(kind)?)
}

pub fn run_continuity_forgotten(
    db_path: &Path,
    conversation_id: i64,
    kind: Option<&str>,
    query: Option<&str>,
) -> Result<Vec<ContinuityForgottenEntry>> {
    let engine = LcmEngine::open(db_path, LcmConfig::default())?;
    engine.continuity_forgotten(
        conversation_id,
        kind.map(ContinuityKind::parse).transpose()?,
        query,
    )
}

pub fn run_continuity_build_prompt(
    db_path: &Path,
    conversation_id: i64,
    kind: &str,
) -> Result<ContinuityPromptPayload> {
    let engine = LcmEngine::open(db_path, LcmConfig::default())?;
    engine.continuity_build_prompt(conversation_id, ContinuityKind::parse(kind)?)
}

pub fn run_context_retrieve(
    db_path: &Path,
    conversation_id: i64,
    mode: &str,
    query: Option<&str>,
    continuity_kind: Option<&str>,
    summary_id: Option<&str>,
    limit: usize,
    depth: usize,
    include_messages: bool,
    token_cap: i64,
) -> Result<serde_json::Value> {
    let engine = LcmEngine::open(db_path, LcmConfig::default())?;
    match mode {
        "current" => {
            let snapshot = engine.snapshot(conversation_id)?;
            let continuity = engine.continuity_show_all(conversation_id)?;
            Ok(serde_json::json!({
                "mode": "current",
                "conversation_id": conversation_id,
                "continuity": continuity,
                "context_items": snapshot.context_items,
                "messages": snapshot.messages,
                "summaries": snapshot.summaries,
            }))
        }
        "continuity" => {
            if let Some(kind) = continuity_kind {
                Ok(serde_json::to_value(engine.continuity_show(
                    conversation_id,
                    ContinuityKind::parse(kind)?,
                )?)?)
            } else {
                Ok(serde_json::to_value(engine.continuity_show_all(conversation_id)?)?)
            }
        }
        "forgotten" => Ok(serde_json::to_value(engine.continuity_forgotten(
            conversation_id,
            continuity_kind.map(ContinuityKind::parse).transpose()?,
            query,
        )?)?),
        "search" => {
            let query = query.context("context_retrieve mode=search requires query")?;
            Ok(serde_json::to_value(engine.grep(
                Some(conversation_id),
                GrepScope::Both,
                GrepMode::FullText,
                query,
                limit,
            )?)?)
        }
        "describe" => {
            let summary_id = summary_id.context("context_retrieve mode=describe requires summary_id")?;
            Ok(serde_json::to_value(engine.describe(summary_id)?)?)
        }
        "expand" => {
            let summary_id = summary_id.context("context_retrieve mode=expand requires summary_id")?;
            Ok(serde_json::to_value(engine.expand(
                summary_id,
                depth,
                include_messages,
                token_cap,
            )?)?)
        }
        other => anyhow::bail!(
            "unsupported context_retrieve mode: {other}; expected one of current, continuity, forgotten, search, describe, expand"
        ),
    }
}

pub fn run_fixture(db_path: &Path, fixture_path: &Path) -> Result<FixtureRunOutput> {
    let fixture_bytes = std::fs::read(fixture_path)
        .with_context(|| format!("failed to read fixture {}", fixture_path.display()))?;
    let fixture: LcmFixture = serde_json::from_slice(&fixture_bytes)
        .with_context(|| format!("failed to parse fixture {}", fixture_path.display()))?;
    let config = merge_fixture_config(fixture.config.clone());
    let engine = LcmEngine::open(db_path, config)?;
    let _ = engine.continuity_init_documents(fixture.conversation_id)?;
    for message in &fixture.messages {
        engine.add_message(fixture.conversation_id, &message.role, &message.content)?;
    }
    let compaction = engine.compact(
        fixture.conversation_id,
        fixture.token_budget,
        &HeuristicSummarizer,
        fixture.force_compact.unwrap_or(false),
    )?;
    let snapshot = engine.snapshot(fixture.conversation_id)?;
    let grep_results = fixture
        .grep_queries
        .unwrap_or_default()
        .into_iter()
        .map(|query| {
            engine.grep(
                Some(fixture.conversation_id),
                GrepScope::parse(&query.scope)?,
                GrepMode::parse(&query.mode)?,
                &query.query,
                query.limit.unwrap_or(20),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let fallback_summary_id = compaction.created_summary_ids.first().cloned();
    let mut expand_results = Vec::new();
    for query in fixture.expand_queries.unwrap_or_default() {
        if let Some(summary_id) = query.summary_id.or_else(|| fallback_summary_id.clone()) {
            expand_results.push(engine.expand(
                &summary_id,
                query.depth.unwrap_or(1),
                query.include_messages.unwrap_or(false),
                query.token_cap.unwrap_or(8_000),
            )?);
        }
    }
    Ok(FixtureRunOutput {
        compaction,
        snapshot,
        grep_results,
        expand_results,
    })
}

fn merge_fixture_config(config: Option<LcmFixtureConfig>) -> LcmConfig {
    let mut merged = LcmConfig::default();
    if let Some(config) = config {
        if let Some(value) = config.context_threshold {
            merged.context_threshold = value;
        }
        if let Some(value) = config.fresh_tail_count {
            merged.fresh_tail_count = value;
        }
        if let Some(value) = config.leaf_chunk_tokens {
            merged.leaf_chunk_tokens = value;
        }
        if let Some(value) = config.leaf_target_tokens {
            merged.leaf_target_tokens = value;
        }
        if let Some(value) = config.condensed_target_tokens {
            merged.condensed_target_tokens = value;
        }
        if let Some(value) = config.leaf_min_fanout {
            merged.leaf_min_fanout = value;
        }
        if let Some(value) = config.condensed_min_fanout {
            merged.condensed_min_fanout = value;
        }
        if let Some(value) = config.max_rounds {
            merged.max_rounds = value;
        }
    }
    merged
}

fn continuity_template(kind: ContinuityKind) -> &'static str {
    match kind {
        ContinuityKind::Narrative => {
            "# CONTINUITY NARRATIVE\n\n## Ausgangslage\n\n## Problem\n\n## Ursache\n\n## Wendepunkte\n\n## Dauerhafte Entscheidungen\n\n## Aktueller Stand\n\n## Offene Spannung\n"
        }
        ContinuityKind::Anchors => {
            "# CONTINUITY ANCHORS\n\n## Artefakte\n\n## Hosts / Ports\n\n## Skripte / Commands\n\n## Invarianten / Verbote\n\n## Gates / Pruefpfade\n"
        }
        ContinuityKind::Focus => {
            "# ACTIVE FOCUS\n\n## Status\n\n## Blocker\n\n## Next\n\n## Done / Gate\n"
        }
    }
}

fn continuity_document_id(conversation_id: i64, kind: ContinuityKind) -> String {
    format!("contdoc_{}_{}", conversation_id, kind.as_str())
}

fn continuity_base_commit_id(conversation_id: i64, kind: ContinuityKind) -> String {
    format!("contbase_{}_{}", conversation_id, kind.as_str())
}

fn continuity_commit_id(
    conversation_id: i64,
    kind: ContinuityKind,
    diff_text: &str,
    rendered_text: &str,
    created_at: &str,
) -> String {
    let mut hash = Sha256::new();
    hash.update(conversation_id.to_string().as_bytes());
    hash.update(kind.as_str().as_bytes());
    hash.update(diff_text.as_bytes());
    hash.update(rendered_text.as_bytes());
    hash.update(created_at.as_bytes());
    let digest = hash.finalize();
    let prefix = digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("contc_{prefix}")
}

fn continuity_heads_revision_id(
    conversation_id: i64,
    narrative_head: &str,
    anchors_head: &str,
    focus_head: &str,
) -> String {
    let mut hash = Sha256::new();
    hash.update(conversation_id.to_string().as_bytes());
    hash.update(narrative_head.as_bytes());
    hash.update(anchors_head.as_bytes());
    hash.update(focus_head.as_bytes());
    let digest = hash.finalize();
    let prefix = digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("contrev_{prefix}")
}

fn apply_continuity_diff(base: &str, diff_text: &str) -> Result<String> {
    let mut sections = parse_continuity_sections(base)?;
    let mut current_section: Option<String> = None;
    for raw_line in diff_text.lines() {
        let line = raw_line.trim_end();
        if line.trim().is_empty() {
            continue;
        }
        if line.starts_with("## ") {
            let section = line.trim().to_string();
            if !sections.contains_key(&section) {
                anyhow::bail!("unknown continuity section in diff: {section}");
            }
            current_section = Some(section);
            continue;
        }
        let section = current_section
            .as_ref()
            .context("continuity diff requires a section header before +/- lines")?;
        let entry = sections.get_mut(section).context("diff section missing in document")?;
        if let Some(added) = line.strip_prefix('+') {
            let value = collapse_whitespace(added);
            if !value.is_empty() && !entry.contains(&value) {
                entry.push(value);
            }
        } else if let Some(removed) = line.strip_prefix('-') {
            let value = collapse_whitespace(removed);
            entry.retain(|existing| existing != &value);
        } else {
            anyhow::bail!("unsupported continuity diff line: {line}");
        }
    }
    render_continuity_sections(base, &sections)
}

fn parse_continuity_sections(base: &str) -> Result<std::collections::BTreeMap<String, Vec<String>>> {
    let mut sections = std::collections::BTreeMap::new();
    let mut current_section: Option<String> = None;
    for raw_line in base.lines() {
        let line = raw_line.trim_end();
        if line.starts_with("## ") {
            current_section = Some(line.to_string());
            sections.entry(line.to_string()).or_insert_with(Vec::new);
            continue;
        }
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        if let Some(section) = current_section.as_ref() {
            sections
                .entry(section.clone())
                .or_insert_with(Vec::new)
                .push(collapse_whitespace(line.trim_start_matches("- ").trim()));
        }
    }
    Ok(sections)
}

fn render_continuity_sections(
    template: &str,
    sections: &std::collections::BTreeMap<String, Vec<String>>,
) -> Result<String> {
    let mut out = Vec::new();
    for raw_line in template.lines() {
        let line = raw_line.trim_end();
        if line.starts_with("## ") {
            out.push(line.to_string());
            if let Some(items) = sections.get(line) {
                if !items.is_empty() {
                    for item in items {
                        out.push(format!("- {item}"));
                    }
                }
            }
            out.push(String::new());
        } else if line.starts_with("# ") {
            out.push(line.to_string());
            out.push(String::new());
        }
    }
    Ok(out.join("\n").trim_end().to_string() + "\n")
}

fn removed_lines_from_diff(diff_text: &str) -> Vec<String> {
    diff_text
        .lines()
        .filter_map(|line| line.strip_prefix('-'))
        .map(collapse_whitespace)
        .filter(|line| !line.is_empty())
        .collect()
}

fn build_continuity_prompt_text(
    conversation_id: i64,
    kind: ContinuityKind,
    current_document: &str,
    recent_messages: &[String],
    recent_summaries: &[String],
    forgotten_lines: &[String],
) -> String {
    let kind_label = match kind {
        ContinuityKind::Narrative => "CONTINUITY NARRATIVE",
        ContinuityKind::Anchors => "CONTINUITY ANCHORS",
        ContinuityKind::Focus => "ACTIVE FOCUS",
    };
    [
        format!("You are updating the CTOX continuity document for conversation {}.", conversation_id),
        format!("Document kind: {}.", kind.as_str()),
        "Output only a strict section-based diff.".to_string(),
        "Rules:".to_string(),
        "1. Use only existing `##` section headers from the current template.".to_string(),
        "2. Inside a section, emit only lines starting with `+` or `-`.".to_string(),
        "3. `+` means keep/add to continuity. `-` means remove as no longer relevant.".to_string(),
        "4. Do not rewrite unchanged lines. Do not output prose, explanations, markdown fences, or summaries.".to_string(),
        "5. Prefer minimal diffs. If nothing changes, output an empty string.".to_string(),
        "6. Never invent facts not supported by recent messages or summaries.".to_string(),
        String::new(),
        format!("<DOCUMENT_KIND>\n{}\n</DOCUMENT_KIND>", kind_label),
        String::new(),
        format!("<CURRENT_DOCUMENT>\n{}\n</CURRENT_DOCUMENT>", current_document.trim_end()),
        String::new(),
        format!(
            "<RECENT_MESSAGES>\n{}\n</RECENT_MESSAGES>",
            if recent_messages.is_empty() {
                "(none)".to_string()
            } else {
                recent_messages.join("\n")
            }
        ),
        String::new(),
        format!(
            "<RECENT_SUMMARIES>\n{}\n</RECENT_SUMMARIES>",
            if recent_summaries.is_empty() {
                "(none)".to_string()
            } else {
                recent_summaries.join("\n")
            }
        ),
        String::new(),
        format!(
            "<PREVIOUSLY_FORGOTTEN_LINES>\n{}\n</PREVIOUSLY_FORGOTTEN_LINES>",
            if forgotten_lines.is_empty() {
                "(none)".to_string()
            } else {
                forgotten_lines.join("\n")
            }
        ),
        String::new(),
        "<OUTPUT_FORMAT_EXAMPLE>\n## Ursache\n- Old line that should be removed.\n+ New line that should be kept.\n</OUTPUT_FORMAT_EXAMPLE>".to_string(),
    ]
    .join("\n")
}

fn parse_summary_kind(value: &str) -> SummaryKind {
    match value {
        "condensed" => SummaryKind::Condensed,
        _ => SummaryKind::Leaf,
    }
}

fn estimate_tokens(content: &str) -> usize {
    let chars = content.chars().count();
    chars.div_ceil(4).max(1)
}

fn collapse_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalize_for_fts(value: &str) -> String {
    collapse_whitespace(value)
}

fn sanitize_fts_query(value: &str) -> String {
    let sanitized = value
        .chars()
        .filter(|ch| ch.is_alphanumeric() || ch.is_whitespace() || *ch == '_')
        .collect::<String>();
    if sanitized.trim().is_empty() {
        "match".to_string()
    } else {
        sanitized
    }
}

fn snippet(content: &str, query: &str) -> String {
    let content_lower = content.to_lowercase();
    let query_lower = query.to_lowercase();
    if let Some(pos) = content_lower.find(&query_lower) {
        let start = pos.saturating_sub(40);
        let end = (pos + query.len() + 80).min(content.len());
        return content[start..end].to_string();
    }
    content.chars().take(140).collect()
}

fn build_deterministic_fallback(source_text: &str, input_tokens: i64) -> String {
    let truncated = if source_text.len() > FALLBACK_MAX_CHARS {
        &source_text[..FALLBACK_MAX_CHARS]
    } else {
        source_text
    };
    format!(
        "{} [Truncated from {input_tokens} tokens]",
        collapse_whitespace(truncated)
    )
}

fn format_summary_timestamp(value: &str) -> String {
    let millis = value.parse::<u128>().unwrap_or(0);
    let secs = (millis / 1000) as i64;
    if let Some(dt) = chrono::DateTime::<chrono::Utc>::from_timestamp(secs, 0) {
        dt.format("%Y-%m-%d %H:%M UTC").to_string()
    } else {
        "1970-01-01 00:00 UTC".to_string()
    }
}

fn sentence_fragment(content: &str, max_chars: usize) -> String {
    let collapsed = collapse_whitespace(content);
    if collapsed.len() <= max_chars {
        return collapsed;
    }
    let clipped = collapsed[..max_chars].trim_end();
    format!("{clipped}...")
}

fn summary_id_for(conversation_id: i64, content: &str, depth: i64) -> String {
    let mut hash = Sha256::new();
    hash.update(conversation_id.to_string().as_bytes());
    hash.update(depth.to_string().as_bytes());
    hash.update(content.as_bytes());
    hash.update(iso_now().as_bytes());
    let digest = hash.finalize();
    let prefix = digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("sum_{prefix}")
}

fn iso_now() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_millis())
        .unwrap_or(0);
    millis.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        let counter = TEMP_DB_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        path.push(format!("ctox-lcm-{nanos}-{counter}.sqlite"));
        path
    }

    #[test]
    fn compacts_messages_and_supports_retrieval() -> Result<()> {
        let db_path = temp_db();
        let engine = LcmEngine::open(&db_path, LcmConfig {
            context_threshold: 0.4,
            fresh_tail_count: 2,
            leaf_chunk_tokens: 20,
            leaf_target_tokens: 120,
            condensed_target_tokens: 120,
            leaf_min_fanout: 3,
            condensed_min_fanout: 2,
            max_rounds: 4,
        })?;

        for idx in 0..8 {
            engine.add_message(
                1,
                if idx % 2 == 0 { "user" } else { "assistant" },
                &format!("message {idx} about postgres migration planning and rollout details"),
            )?;
        }

        let result = engine.compact(1, 40, &HeuristicSummarizer, false)?;
        assert!(result.action_taken);
        assert!(!result.created_summary_ids.is_empty());

        let grep = engine.grep(Some(1), GrepScope::Both, GrepMode::FullText, "postgres", 10)?;
        assert!(grep.total_matches > 0);

        let described = engine.describe(&result.created_summary_ids[0])?;
        assert!(described.is_some());

        let expanded = engine.expand(&result.created_summary_ids[0], 1, true, 10_000)?;
        assert!(!expanded.messages.is_empty());

        let _ = std::fs::remove_file(db_path);
        Ok(())
    }

    #[test]
    fn creates_condensed_summary_from_leaf_summaries() -> Result<()> {
        let db_path = temp_db();
        let engine = LcmEngine::open(&db_path, LcmConfig {
            context_threshold: 0.2,
            fresh_tail_count: 0,
            leaf_chunk_tokens: 60,
            leaf_target_tokens: 10,
            condensed_target_tokens: 10,
            leaf_min_fanout: 2,
            condensed_min_fanout: 2,
            max_rounds: 6,
        })?;

        let leaf_a = engine.insert_summary(
            7,
            SummaryKind::Leaf,
            0,
            "leaf summary A with rollout evidence and retrieval details",
            0,
            0,
            24,
            &[],
            Vec::new(),
            0,
            Vec::new(),
        )?;
        let leaf_b = engine.insert_summary(
            7,
            SummaryKind::Leaf,
            0,
            "leaf summary B with fallback notes and verification details",
            0,
            0,
            26,
            &[],
            Vec::new(),
            1,
            Vec::new(),
        )?;

        let condensed_id = engine
            .compact_condensed_pass(7, &HeuristicSummarizer)?
            .context("expected condensed summary")?;
        let condensed = engine.get_summary(&condensed_id)?.context("missing condensed summary")?;

        assert_eq!(condensed.kind, SummaryKind::Condensed);
        assert_eq!(condensed.depth, 1);
        assert_eq!(condensed.source_message_token_count, 50);
        assert_eq!(condensed.descendant_count, 2);
        assert_eq!(
            engine.summary_parent_ids(&leaf_a)?,
            vec![condensed_id.clone()]
        );
        assert_eq!(engine.summary_parent_ids(&leaf_b)?, vec![condensed_id]);

        let _ = std::fs::remove_file(db_path);
        Ok(())
    }

    #[test]
    fn new_session_starts_with_raw_continuity_templates() -> Result<()> {
        let db_path = temp_db();
        let engine = LcmEngine::open(&db_path, LcmConfig::default())?;

        engine.add_message(9, "user", "First session message.")?;

        let current = engine.latest_continuity(9)?.context("expected continuity state")?;
        assert!(current.narrative.contains("# CONTINUITY NARRATIVE"));
        assert!(current.narrative.contains("## Ausgangslage"));
        assert!(current.anchors.contains("# CONTINUITY ANCHORS"));
        assert!(current.focus.contains("# ACTIVE FOCUS"));
        assert!(!current.narrative.contains("- "));

        let _ = std::fs::remove_file(db_path);
        Ok(())
    }

    #[test]
    fn continuity_diff_documents_apply_and_track_forgotten_lines() -> Result<()> {
        let db_path = temp_db();
        let engine = LcmEngine::open(&db_path, LcmConfig::default())?;

        let docs = engine.continuity_init_documents(11)?;
        assert!(docs.narrative.content.contains("## Ausgangslage"));

        let updated = engine.continuity_apply_diff(
            11,
            ContinuityKind::Narrative,
            "## Ausgangslage\n+ Service started with a fragile migration plan.\n## Ursache\n+ Cache warmer timing caused the breakage.\n",
        )?;
        assert!(updated.content.contains("Service started with a fragile migration plan."));
        assert!(updated.content.contains("Cache warmer timing caused the breakage."));

        let updated_again = engine.continuity_apply_diff(
            11,
            ContinuityKind::Narrative,
            "## Ursache\n- Cache warmer timing caused the breakage.\n+ Cache warmer timing after verification caused the breakage.\n",
        )?;
        assert!(updated_again
            .content
            .contains("Cache warmer timing after verification caused the breakage."));
        assert!(!updated_again
            .content
            .contains("Cache warmer timing caused the breakage."));

        let forgotten = engine.continuity_forgotten(11, Some(ContinuityKind::Narrative), Some("Cache warmer"))?;
        assert_eq!(forgotten.len(), 1);
        assert!(forgotten[0].line.contains("Cache warmer timing caused the breakage."));

        let rebuilt = engine.continuity_rebuild(11, ContinuityKind::Narrative)?;
        assert_eq!(rebuilt.content, updated_again.content);

        let _ = std::fs::remove_file(db_path);
        Ok(())
    }

    #[test]
    fn continuity_prompt_contains_document_and_diff_rules() -> Result<()> {
        let db_path = temp_db();
        let engine = LcmEngine::open(&db_path, LcmConfig::default())?;
        let _ = engine.continuity_init_documents(12)?;
        engine.add_message(12, "user", "Keep the rollout gate active until validation passes on db-prod.internal.")?;

        let payload = engine.continuity_build_prompt(12, ContinuityKind::Narrative)?;
        assert!(payload.prompt.contains("Output only a strict section-based diff."));
        assert!(payload.prompt.contains("<CURRENT_DOCUMENT>"));
        assert!(payload.prompt.contains("<RECENT_MESSAGES>"));
        assert!(payload.prompt.contains("## Ausgangslage"));

        let _ = std::fs::remove_file(db_path);
        Ok(())
    }
}
