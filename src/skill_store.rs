use anyhow::Context;
use anyhow::Result;
use include_dir::Dir;
use rusqlite::params;
use rusqlite::Connection;
use rusqlite::OptionalExtension;
use sha2::Digest;
use sha2::Sha256;
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::inference::runtime_env;
use crate::persistence;

const SKILL_BUNDLES_TABLE: &str = "ctox_skill_bundles";
const SKILL_FILES_TABLE: &str = "ctox_skill_files";

/// System skills are embedded into the ctox binary at compile time and
/// imported into SQLite at service start. The repo-root `skills/system/`
/// directory uses cluster subfolders (e.g. `host_ops/`, `mission_orchestration/`)
/// to organize the 39 skills; those subfolders become the `cluster` field on
/// each bundle. This is the canonical store for system skills — `.system/`
/// on disk is materialized from here for the Codex skill manager to consume.
const EMBEDDED_SYSTEM_SKILLS: Dir<'static> =
    include_dir::include_dir!("$CARGO_MANIFEST_DIR/skills/system");

#[derive(Debug, Clone)]
pub struct SkillBundleView {
    pub skill_id: String,
    pub skill_name: String,
    pub class: String,
    pub state: String,
    pub description: String,
    pub source_path: Option<String>,
    pub cluster: String,
}

#[derive(Debug, Clone)]
pub struct SkillFileView {
    pub relative_path: String,
    pub content: Vec<u8>,
}

pub fn bootstrap_from_roots(root: &Path) -> Result<()> {
    ensure_schema(root)?;
    for base in skill_roots(root) {
        import_skill_tree(root, &base)?;
    }
    Ok(())
}

/// Import the binary-embedded `skills/system/` tree (clustered by subfolder)
/// into the SQLite skill catalog. Idempotent — safe to call on every service
/// start. The cluster is taken from the `cluster:` frontmatter field; if
/// missing, it falls back to the first path segment under `skills/system/`.
pub fn bootstrap_embedded_system_skills(root: &Path) -> Result<()> {
    ensure_schema(root)?;
    walk_embedded_for_skills(root, &EMBEDDED_SYSTEM_SKILLS, "")?;
    Ok(())
}

pub fn list_skill_bundles(root: &Path) -> Result<Vec<SkillBundleView>> {
    ensure_schema(root)?;
    let conn = open_db(root)?;
    let mut statement = conn.prepare(&format!(
        "SELECT skill_id, skill_name, class, state, description, source_path, cluster, updated_at
         FROM {SKILL_BUNDLES_TABLE}
         ORDER BY cluster ASC, class ASC, skill_name ASC, skill_id ASC"
    ))?;
    let rows = statement.query_map([], |row| {
        Ok(SkillBundleView {
            skill_id: row.get(0)?,
            skill_name: row.get(1)?,
            class: row.get(2)?,
            state: row.get(3)?,
            description: row.get(4)?,
            source_path: row.get(5)?,
            cluster: row.get(6)?,
        })
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(anyhow::Error::from)
}

pub fn list_skill_files(root: &Path, skill_id: &str) -> Result<Vec<SkillFileView>> {
    ensure_schema(root)?;
    let conn = open_db(root)?;
    let mut statement = conn.prepare(&format!(
        "SELECT relative_path, content_blob
         FROM {SKILL_FILES_TABLE}
         WHERE skill_id = ?1
         ORDER BY relative_path ASC"
    ))?;
    let rows = statement.query_map(params![skill_id], |row| {
        Ok(SkillFileView {
            relative_path: row.get(0)?,
            content: row.get(1)?,
        })
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(anyhow::Error::from)
}

pub fn upsert_skill_bundle_from_dir(root: &Path, dir: &Path) -> Result<Option<SkillBundleView>> {
    ensure_schema(root)?;
    if !dir.exists() || !dir.is_dir() {
        return Ok(None);
    }
    let skill_md = dir.join("SKILL.md");
    if !skill_md.is_file() {
        return Ok(None);
    }
    let files = collect_bundle_files(dir)?;
    let skill_name = dir
        .file_name()
        .map(|value| value.to_string_lossy().to_string())
        .context("skill dir has no name")?;
    let skill_body = files
        .get("SKILL.md")
        .map(|bytes| String::from_utf8_lossy(bytes).to_string())
        .unwrap_or_default();
    let metadata = parse_skill_catalog_metadata(&skill_body);
    let source_path = Some(dir.display().to_string());
    let (class, state) = classify_skill(root, dir, &skill_md, &metadata);
    let description = parse_skill_description(&skill_body);
    let cluster = metadata
        .cluster
        .clone()
        .unwrap_or_else(|| infer_cluster_from_disk_path(root, dir));

    upsert_bundle_records(root, &skill_name, &class, &state, &description, source_path.as_deref(), &cluster, &files)
}

/// Shared upsert path used by both filesystem and embedded importers.
fn upsert_bundle_records(
    root: &Path,
    skill_name: &str,
    class: &str,
    state: &str,
    description: &str,
    source_path: Option<&str>,
    cluster: &str,
    files: &BTreeMap<String, Vec<u8>>,
) -> Result<Option<SkillBundleView>> {
    let skill_id = stable_skill_id(skill_name);
    let updated_at = now_epoch_string();

    let mut conn = open_db(root)?;
    let tx = conn
        .transaction()
        .context("failed to open skill transaction")?;
    tx.execute(
        &format!(
            "INSERT INTO {SKILL_BUNDLES_TABLE}
             (skill_id, skill_name, class, state, description, source_path, cluster, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(skill_id) DO UPDATE SET
               skill_name = excluded.skill_name,
               class = excluded.class,
               state = excluded.state,
               description = excluded.description,
               source_path = excluded.source_path,
               cluster = excluded.cluster,
               updated_at = excluded.updated_at"
        ),
        params![
            skill_id,
            skill_name,
            class,
            state,
            description,
            source_path,
            cluster,
            updated_at,
        ],
    )?;
    tx.execute(
        &format!("DELETE FROM {SKILL_FILES_TABLE} WHERE skill_id = ?1"),
        params![skill_id],
    )?;
    for (relative_path, content) in files {
        tx.execute(
            &format!(
                "INSERT INTO {SKILL_FILES_TABLE} (skill_id, relative_path, content_blob)
                 VALUES (?1, ?2, ?3)"
            ),
            params![skill_id, relative_path, content],
        )?;
    }
    tx.commit().context("failed to commit skill transaction")?;

    Ok(Some(SkillBundleView {
        skill_id,
        skill_name: skill_name.to_string(),
        class: class.to_string(),
        state: state.to_string(),
        description: description.to_string(),
        source_path: source_path.map(str::to_string),
        cluster: cluster.to_string(),
    }))
}

/// Walk the embedded skills/system tree (clustered by subfolder) and import
/// each SKILL.md-bearing directory into SQLite.
fn walk_embedded_for_skills(root: &Path, dir: &Dir<'_>, path_prefix: &str) -> Result<()> {
    let has_skill_md = dir.files().any(|f| {
        f.path()
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n == "SKILL.md")
            .unwrap_or(false)
    });
    if has_skill_md {
        upsert_embedded_skill(root, dir, path_prefix)?;
        return Ok(());
    }
    for entry in dir.entries() {
        if let include_dir::DirEntry::Dir(subdir) = entry {
            let segment = subdir
                .path()
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let new_prefix = if path_prefix.is_empty() {
                segment
            } else {
                format!("{path_prefix}/{segment}")
            };
            walk_embedded_for_skills(root, subdir, &new_prefix)?;
        }
    }
    Ok(())
}

fn upsert_embedded_skill(root: &Path, skill_dir: &Dir<'_>, path_in_system: &str) -> Result<()> {
    let mut files: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let dir_path_len = skill_dir.path().to_string_lossy().len();
    collect_embedded_files(skill_dir, dir_path_len, &mut files);

    let skill_name = skill_dir
        .path()
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    if skill_name.is_empty() {
        return Ok(());
    }
    let skill_body = files
        .get("SKILL.md")
        .map(|bytes| String::from_utf8_lossy(bytes).to_string())
        .unwrap_or_default();
    let metadata = parse_skill_catalog_metadata(&skill_body);
    let description = parse_skill_description(&skill_body);
    let class = metadata
        .class
        .clone()
        .unwrap_or_else(|| "ctox_core".to_string());
    let state = metadata
        .state
        .clone()
        .unwrap_or_else(|| "stable".to_string());
    // Cluster: prefer frontmatter, fall back to the first path segment under skills/system/
    let cluster = metadata.cluster.clone().unwrap_or_else(|| {
        path_in_system
            .split('/')
            .next()
            .filter(|s| !s.is_empty() && *s != skill_name)
            .map(str::to_string)
            .unwrap_or_default()
    });
    let source_path = Some(format!("embedded:skills/system/{path_in_system}"));

    upsert_bundle_records(
        root,
        &skill_name,
        &class,
        &state,
        &description,
        source_path.as_deref(),
        &cluster,
        &files,
    )?;
    Ok(())
}

fn collect_embedded_files(dir: &Dir<'_>, root_path_len: usize, out: &mut BTreeMap<String, Vec<u8>>) {
    for entry in dir.entries() {
        match entry {
            include_dir::DirEntry::File(file) => {
                let full = file.path().to_string_lossy();
                let relative = full[root_path_len..].trim_start_matches('/').to_string();
                out.insert(relative, file.contents().to_vec());
            }
            include_dir::DirEntry::Dir(subdir) => {
                collect_embedded_files(subdir, root_path_len, out);
            }
        }
    }
}

fn infer_cluster_from_disk_path(root: &Path, dir: &Path) -> String {
    let system_root = root.join("skills").join("system");
    if let Ok(rel) = dir.strip_prefix(&system_root) {
        let mut comps = rel.components();
        if let Some(first) = comps.next() {
            // If there's a deeper component, the first is a cluster name
            if comps.next().is_some() {
                return first.as_os_str().to_string_lossy().to_string();
            }
        }
    }
    String::new()
}

pub fn materialize_skill_bundle(root: &Path, skill_name: &str) -> Result<Option<PathBuf>> {
    ensure_schema(root)?;
    let bundle = load_skill_bundle_by_name(root, skill_name)?;
    let Some(bundle) = bundle else {
        return Ok(None);
    };
    let files = list_skill_files(root, &bundle.skill_id)?;
    let base_dir = managed_materialized_skills_root(root).join(&bundle.skill_name);
    if base_dir.exists() {
        fs::remove_dir_all(&base_dir).with_context(|| {
            format!("failed to reset materialized skill {}", base_dir.display())
        })?;
    }
    fs::create_dir_all(&base_dir)
        .with_context(|| format!("failed to create materialized skill {}", base_dir.display()))?;
    for file in files {
        let path = base_dir.join(&file.relative_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create skill file dir {}", parent.display()))?;
        }
        fs::write(&path, &file.content).with_context(|| {
            format!("failed to write materialized skill file {}", path.display())
        })?;
    }
    Ok(Some(base_dir))
}

pub fn resolve_materialized_skill_dir(root: &Path, skill_name: &str) -> Result<Option<PathBuf>> {
    if let Some(dir) = materialize_skill_bundle(root, skill_name)? {
        return Ok(Some(dir));
    }
    Ok(None)
}

fn load_skill_bundle_by_name(root: &Path, skill_name: &str) -> Result<Option<SkillBundleView>> {
    let conn = open_db(root)?;
    conn.query_row(
        &format!(
            "SELECT skill_id, skill_name, class, state, description, source_path, cluster, updated_at
             FROM {SKILL_BUNDLES_TABLE}
             WHERE skill_name = ?1
             LIMIT 1"
        ),
        params![skill_name],
        |row| {
            Ok(SkillBundleView {
                skill_id: row.get(0)?,
                skill_name: row.get(1)?,
                class: row.get(2)?,
                state: row.get(3)?,
                description: row.get(4)?,
                source_path: row.get(5)?,
                cluster: row.get(6)?,
            })
        },
    )
    .optional()
    .map_err(anyhow::Error::from)
}

fn skill_roots(root: &Path) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    let mut seen = HashSet::new();
    for candidate in [
        root.join("skills"),
        configured_skill_root(root),
        configured_generated_skill_root(root),
    ] {
        if seen.insert(candidate.clone()) {
            roots.push(candidate);
        }
    }
    roots
}

fn import_skill_tree(root: &Path, base: &Path) -> Result<()> {
    if !base.exists() {
        return Ok(());
    }
    let Ok(base_canon) = fs::canonicalize(base) else {
        return Ok(());
    };
    let mut queue = vec![base_canon];
    while let Some(dir) = queue.pop() {
        let skill_md = dir.join("SKILL.md");
        if skill_md.is_file() {
            let _ = upsert_skill_bundle_from_dir(root, &dir)?;
            continue;
        }
        let Ok(read_dir) = fs::read_dir(&dir) else {
            continue;
        };
        for child in read_dir.flatten() {
            if child
                .file_type()
                .map(|value| value.is_dir())
                .unwrap_or(false)
            {
                if child
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".materialized")
                {
                    continue;
                }
                queue.push(child.path());
            }
        }
    }
    Ok(())
}

fn collect_bundle_files(dir: &Path) -> Result<BTreeMap<String, Vec<u8>>> {
    let mut files = BTreeMap::new();
    let mut queue = vec![dir.to_path_buf()];
    while let Some(current) = queue.pop() {
        let read_dir = fs::read_dir(&current)
            .with_context(|| format!("failed to read skill dir {}", current.display()))?;
        for child in read_dir.flatten() {
            let path = child.path();
            let file_type = child.file_type().ok();
            if file_type.map(|value| value.is_dir()).unwrap_or(false) {
                queue.push(path);
                continue;
            }
            if !file_type.map(|value| value.is_file()).unwrap_or(false) {
                continue;
            }
            let relative = path
                .strip_prefix(dir)
                .context("failed to relativize skill path")?
                .to_string_lossy()
                .replace('\\', "/");
            let content = fs::read(&path)
                .with_context(|| format!("failed to read skill file {}", path.display()))?;
            files.insert(relative, content);
        }
    }
    Ok(files)
}

fn open_db(root: &Path) -> Result<Connection> {
    let path = persistence::sqlite_path(root);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create skill db dir {}", parent.display()))?;
    }
    let conn = Connection::open(&path)
        .with_context(|| format!("failed to open skill db {}", path.display()))?;
    conn.busy_timeout(persistence::sqlite_busy_timeout_duration())
        .context("failed to configure SQLite busy_timeout for skills")?;
    ensure_schema_on_conn(&conn)?;
    Ok(conn)
}

fn ensure_schema(root: &Path) -> Result<()> {
    let conn = open_db(root)?;
    ensure_schema_on_conn(&conn)
}

fn ensure_schema_on_conn(conn: &Connection) -> Result<()> {
    let busy_timeout_ms = persistence::sqlite_busy_timeout_millis();
    conn.execute_batch(&format!(
        "PRAGMA journal_mode=WAL;
         PRAGMA busy_timeout={busy_timeout_ms};
         CREATE TABLE IF NOT EXISTS {SKILL_BUNDLES_TABLE} (
             skill_id TEXT PRIMARY KEY,
             skill_name TEXT NOT NULL,
             class TEXT NOT NULL,
             state TEXT NOT NULL,
             description TEXT NOT NULL,
             source_path TEXT,
             cluster TEXT NOT NULL DEFAULT '',
             updated_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS {SKILL_FILES_TABLE} (
             skill_id TEXT NOT NULL,
             relative_path TEXT NOT NULL,
             content_blob BLOB NOT NULL,
             PRIMARY KEY (skill_id, relative_path),
             FOREIGN KEY (skill_id) REFERENCES {SKILL_BUNDLES_TABLE}(skill_id) ON DELETE CASCADE
         );"
    ))
    .context("failed to initialize skill store schema")?;
    // Migration for installs that pre-date the cluster column.
    let has_cluster: bool = conn
        .prepare(&format!("PRAGMA table_info({SKILL_BUNDLES_TABLE})"))?
        .query_map([], |row| row.get::<_, String>(1))?
        .filter_map(|r| r.ok())
        .any(|name| name == "cluster");
    if !has_cluster {
        conn.execute(
            &format!(
                "ALTER TABLE {SKILL_BUNDLES_TABLE} ADD COLUMN cluster TEXT NOT NULL DEFAULT ''"
            ),
            [],
        )
        .context("failed to add cluster column to skill bundles")?;
    }
    Ok(())
}

fn stable_skill_id(skill_name: &str) -> String {
    let digest = Sha256::digest(skill_name.trim().as_bytes());
    format!("skill-{:x}", digest)[..18].to_string()
}

fn parse_skill_description(body: &str) -> String {
    let body = if let Some(frontmatter) = extract_frontmatter(body) {
        body.strip_prefix("---\n")
            .and_then(|rest| rest.strip_prefix(frontmatter))
            .and_then(|rest| rest.strip_prefix("\n---"))
            .map(str::trim_start)
            .unwrap_or(body)
    } else {
        body
    };
    let mut in_code_fence = false;
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            in_code_fence = !in_code_fence;
            continue;
        }
        if in_code_fence
            || trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with('-')
            || trimmed.starts_with('*')
            || trimmed.starts_with('<')
        {
            continue;
        }
        return trimmed.to_string();
    }
    "No inline summary available.".to_string()
}

#[derive(Debug, Clone, Default)]
struct SkillCatalogMetadata {
    class: Option<String>,
    state: Option<String>,
    cluster: Option<String>,
}

fn parse_skill_catalog_metadata(body: &str) -> SkillCatalogMetadata {
    let Some(frontmatter) = extract_frontmatter(body) else {
        return SkillCatalogMetadata::default();
    };
    let mut metadata = SkillCatalogMetadata::default();
    for raw_line in frontmatter.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, raw_value)) = line.split_once(':') else {
            continue;
        };
        let value = raw_value.trim().trim_matches('"').trim_matches('\'').trim();
        match key.trim() {
            "class" => metadata.class = Some(normalize_class(value)),
            "state" => metadata.state = Some(normalize_state(value)),
            "cluster" if !value.is_empty() => metadata.cluster = Some(value.to_string()),
            _ => {}
        }
    }
    metadata
}

fn extract_frontmatter(body: &str) -> Option<&str> {
    let body = body.strip_prefix("---\n")?;
    let end = body.find("\n---")?;
    body.get(..end)
}

fn classify_skill(
    root: &Path,
    base_dir: &Path,
    skill_md: &Path,
    metadata: &SkillCatalogMetadata,
) -> (String, String) {
    let class = metadata
        .class
        .clone()
        .unwrap_or_else(|| infer_skill_class(root, base_dir, skill_md));
    let mut state = metadata
        .state
        .clone()
        .unwrap_or_else(|| infer_skill_state(root, skill_md, &class));
    if class != "personal" && matches!(state.as_str(), "authored" | "generated") {
        state = "stable".to_string();
    }
    (class, state)
}

fn infer_skill_class(root: &Path, base_dir: &Path, skill_md: &Path) -> String {
    let repo_skills_root = root.join("skills");
    if path_matches_prefix(skill_md, &repo_skills_root.join(".curated")) {
        return "installed_packs".to_string();
    }
    if path_matches_prefix(skill_md, &repo_skills_root.join(".system"))
        || path_matches_prefix(skill_md, &repo_skills_root)
    {
        return "ctox_core".to_string();
    }
    if path_matches_prefix(skill_md, &configured_generated_skill_root(root))
        || path_matches_prefix(skill_md, &configured_skill_root(root))
    {
        return "personal".to_string();
    }
    let base_text = base_dir.to_string_lossy();
    if base_text.contains(".codex/skills") || base_text.contains(".agents/skills") {
        return "codex_core".to_string();
    }
    "ctox_core".to_string()
}

fn infer_skill_state(root: &Path, skill_md: &Path, class: &str) -> String {
    if class == "personal" {
        if path_matches_prefix(skill_md, &configured_generated_skill_root(root)) {
            return "generated".to_string();
        }
        return "authored".to_string();
    }
    if path_matches_prefix(skill_md, &root.join("skills/drafts")) {
        return "draft".to_string();
    }
    "stable".to_string()
}

fn normalize_class(value: &str) -> String {
    let value = value.trim().to_ascii_lowercase();
    match value.as_str() {
        "ctox-core" | "ctox_core" | "ctox" => "ctox_core".to_string(),
        "codex-core" | "codex_core" | "codex" => "codex_core".to_string(),
        "installed-packs" | "installed_pack" | "pack" | "packs" | "curated" => {
            "installed_packs".to_string()
        }
        "personal" => "personal".to_string(),
        _ => "ctox_core".to_string(),
    }
}

fn normalize_state(value: &str) -> String {
    let value = value.trim().to_ascii_lowercase();
    match value.as_str() {
        "authored" => "authored".to_string(),
        "generated" => "generated".to_string(),
        "draft" => "draft".to_string(),
        _ => "stable".to_string(),
    }
}

fn configured_skill_root(root: &Path) -> PathBuf {
    runtime_env::env_or_config(root, "CTOX_SKILLS_ROOT")
        .map(PathBuf::from)
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| root.join("runtime/skills"))
}

fn configured_generated_skill_root(root: &Path) -> PathBuf {
    runtime_env::env_or_config(root, "CTOX_GENERATED_SKILLS_ROOT")
        .map(PathBuf::from)
        .filter(|value| !value.as_os_str().is_empty())
        .or_else(|| {
            runtime_env::env_or_config(root, "CTOX_STATE_ROOT")
                .map(PathBuf::from)
                .filter(|value| !value.as_os_str().is_empty())
                .map(|state_root| state_root.join("generated-skills"))
        })
        .unwrap_or_else(|| root.join("runtime").join("generated-skills"))
}

fn managed_materialized_skills_root(root: &Path) -> PathBuf {
    configured_generated_skill_root(root).join(".materialized")
}

fn path_matches_prefix(path: &Path, prefix: &Path) -> bool {
    if path.starts_with(prefix) {
        return true;
    }
    match (fs::canonicalize(path), fs::canonicalize(prefix)) {
        (Ok(path), Ok(prefix)) => path.starts_with(prefix),
        _ => false,
    }
}

fn now_epoch_string() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string()
}
