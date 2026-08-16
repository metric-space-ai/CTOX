fn business_records_projection_clock_stamp(
    conn: &Connection,
    collections: &[String],
    mut hasher: Sha256,
) -> anyhow::Result<BusinessRecordsProjectionStamp> {
    let mut row_count = 0usize;
    let mut latest_updated_at_ms = 0i64;
    if let Some(sql) = business_records_projection_clock_stamp_query(collections.len()) {
        let mut statement = conn
            .prepare(&sql)
            .context("prepare business records projection clock stamp query")?;
        let mut rows = statement
            .query(params_from_iter(collections.iter().map(String::as_str)))
            .context("query business records projection clock stamp")?;
        while let Some(row) = rows
            .next()
            .context("read business records projection clock stamp row")?
        {
            let collection: String = row.get(0)?;
            let version: i64 = row.get(1)?;
            let collection_row_count: i64 = row.get(2)?;
            let deleted_count: i64 = row.get(3)?;
            let collection_latest_updated_at_ms: i64 = row.get(4)?;
            hasher.update(collection.len().to_le_bytes());
            hasher.update(collection.as_bytes());
            hasher.update(version.to_le_bytes());
            hasher.update(collection_row_count.to_le_bytes());
            hasher.update(deleted_count.to_le_bytes());
            hasher.update(collection_latest_updated_at_ms.to_le_bytes());
            row_count = row_count.saturating_add(collection_row_count.max(0) as usize);
            latest_updated_at_ms = latest_updated_at_ms.max(collection_latest_updated_at_ms.max(0));
        }
    }

    Ok(BusinessRecordsProjectionStamp {
        row_count,
        latest_updated_at_ms,
        content_hash: format!("{:x}", hasher.finalize()),
    })
}

fn business_records_projection_metadata_stamp(
    conn: &Connection,
    collections: &[String],
    mut hasher: Sha256,
) -> anyhow::Result<BusinessRecordsProjectionStamp> {
    let mut row_count = 0usize;
    let mut latest_updated_at_ms = 0i64;
    if let Some(sql) = business_records_projection_metadata_stamp_query(collections.len()) {
        let mut statement = conn
            .prepare(&sql)
            .context("prepare business records projection metadata stamp query")?;
        let mut rows = statement
            .query(params_from_iter(collections.iter().map(String::as_str)))
            .context("query business records projection metadata stamp")?;
        while let Some(row) = rows
            .next()
            .context("read business records projection metadata stamp row")?
        {
            row_count += 1;
            let collection: String = row.get(0)?;
            let record_id: String = row.get(1)?;
            let rev: String = row.get(2)?;
            let deleted: i64 = row.get(3)?;
            let updated_at_ms: i64 = row.get(4)?;
            for value in [&collection, &record_id, &rev] {
                hasher.update(value.len().to_le_bytes());
                hasher.update(value.as_bytes());
            }
            hasher.update(deleted.to_le_bytes());
            hasher.update(updated_at_ms.to_le_bytes());
            latest_updated_at_ms = latest_updated_at_ms.max(updated_at_ms);
        }
    }

    Ok(BusinessRecordsProjectionStamp {
        row_count,
        latest_updated_at_ms,
        content_hash: format!("{:x}", hasher.finalize()),
    })
}

fn business_records_projection_clock_stamp_query(collection_count: usize) -> Option<String> {
    if collection_count == 0 {
        return None;
    }
    let placeholders = std::iter::repeat("?")
        .take(collection_count)
        .collect::<Vec<_>>()
        .join(", ");
    Some(format!(
        "SELECT collection, version, row_count, deleted_count, latest_updated_at_ms
         FROM business_records_projection_clock
         WHERE collection IN ({placeholders})
         ORDER BY collection ASC"
    ))
}

fn business_records_projection_metadata_stamp_query(collection_count: usize) -> Option<String> {
    if collection_count == 0 {
        return None;
    }
    let placeholders = std::iter::repeat("?")
        .take(collection_count)
        .collect::<Vec<_>>()
        .join(", ");
    Some(format!(
        "SELECT collection, record_id, rev, deleted, updated_at_ms
         FROM business_records
         WHERE collection IN ({placeholders})
         ORDER BY collection ASC, record_id ASC"
    ))
}

fn empty_business_records_projection_stamp() -> BusinessRecordsProjectionStamp {
    BusinessRecordsProjectionStamp {
        row_count: 0,
        latest_updated_at_ms: 0,
        content_hash: String::new(),
    }
}

pub(super) fn configured_business_users_projection_hash() -> String {
    let mut users = configured_business_users();
    users.sort_by(|left, right| left.id.cmp(&right.id));
    let mut hasher = Sha256::new();
    for user in users {
        let id = user.id.trim();
        let role = normalize_business_role(&user.role);
        for value in [id, role.as_str()] {
            hasher.update(value.len().to_le_bytes());
            hasher.update(value.as_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

const WORKSPACE_BRANDING_ID: &str = "workspace-branding";
const WORKSPACE_BRANDING_TOKENS: &[&str] = &[
    "bg",
    "surface",
    "surface_2",
    "line",
    "text",
    "text_strong",
    "muted",
    "accent",
    "accent_soft",
    "accent_foreground",
    "danger",
    "warning",
    "success",
    "focus_ring",
];

pub fn workspace_branding_for_rxdb(root: &Path) -> anyhow::Result<Value> {
    let conn = open_store(root)?;
    let row = conn
        .query_row(
            "SELECT name, light_json, dark_json, module_accents_json, updated_at_ms
             FROM business_workspace_branding
             WHERE id = ?1",
            params![WORKSPACE_BRANDING_ID],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, i64>(4)?,
                ))
            },
        )
        .optional()?;
    let Some((name, light_json, dark_json, module_accents_json, updated_at_ms)) = row else {
        return Ok(default_workspace_branding_projection());
    };
    Ok(serde_json::json!({
        "id": WORKSPACE_BRANDING_ID,
        "ok": true,
        "custom": true,
        "name": name,
        "light": serde_json::from_str::<Value>(&light_json).unwrap_or_else(|_| serde_json::json!({})),
        "dark": serde_json::from_str::<Value>(&dark_json).unwrap_or_else(|_| serde_json::json!({})),
        "module_accents": serde_json::from_str::<Value>(&module_accents_json).unwrap_or_else(|_| serde_json::json!({})),
        "updated_at_ms": updated_at_ms,
        "is_deleted": false
    }))
}

pub(crate) fn workspace_branding_projection_stamp(root: &Path) -> WorkspaceBrandingProjectionStamp {
    let (row_count, latest_updated_at_ms, content_hash) =
        workspace_branding_content_stamp(root).unwrap_or_else(|_| (0, 0, String::new()));
    WorkspaceBrandingProjectionStamp {
        store: business_os_sqlite_store_stamp(&root.join("runtime").join(STORE_FILE)),
        row_count,
        latest_updated_at_ms,
        content_hash,
    }
}

fn workspace_branding_content_stamp(root: &Path) -> anyhow::Result<(usize, i64, String)> {
    let conn = open_store(root)?;
    let mut statement = conn.prepare(
        "SELECT name, light_json, dark_json, module_accents_json, updated_at_ms
         FROM business_workspace_branding
         ORDER BY id ASC",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, i64>(4)?,
        ))
    })?;
    let mut count = 0usize;
    let mut latest_updated_at_ms = 0i64;
    let mut hasher = Sha256::new();
    for row in rows {
        let (name, light, dark, module_accents, updated_at_ms) = row?;
        count = count.saturating_add(1);
        latest_updated_at_ms = latest_updated_at_ms.max(updated_at_ms);
        hasher.update(name.as_bytes());
        hasher.update([0]);
        hasher.update(light.as_bytes());
        hasher.update([0]);
        hasher.update(dark.as_bytes());
        hasher.update([0]);
        hasher.update(module_accents.as_bytes());
        hasher.update([0]);
        hasher.update(updated_at_ms.to_le_bytes());
    }
    Ok((
        count,
        latest_updated_at_ms,
        format!("{:x}", hasher.finalize()),
    ))
}

pub fn save_workspace_branding_command(
    root: &Path,
    session: &BusinessOsSession,
    request: WorkspaceBrandingUpdateRequest,
) -> anyhow::Result<Value> {
    let conn = open_store(root)?;
    let observed_at_ms = now_ms() as i64;
    if request.reset {
        conn.execute(
            "DELETE FROM business_workspace_branding WHERE id = ?1",
            params![WORKSPACE_BRANDING_ID],
        )?;
        record_workspace_branding_change_event(
            &conn,
            session,
            serde_json::json!({
                "action": "reset",
                "updated_at_ms": observed_at_ms
            }),
            observed_at_ms,
        )?;
        return Ok(serde_json::json!({
            "ok": true,
            "kind": "workspace_branding",
            "action": "reset",
            "branding": default_workspace_branding_projection()
        }));
    }

    let normalized = normalize_workspace_branding_request(request)?;
    let name = normalized
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("Workspace Branding")
        .to_owned();
    let light = normalized
        .get("light")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let dark = normalized
        .get("dark")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let module_accents = normalized
        .get("module_accents")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    conn.execute(
        "INSERT INTO business_workspace_branding
             (id, name, light_json, dark_json, module_accents_json, updated_by, updated_at_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
         ON CONFLICT(id) DO UPDATE SET
             name = excluded.name,
             light_json = excluded.light_json,
             dark_json = excluded.dark_json,
             module_accents_json = excluded.module_accents_json,
             updated_by = excluded.updated_by,
             updated_at_ms = excluded.updated_at_ms",
        params![
            WORKSPACE_BRANDING_ID,
            name,
            serde_json::to_string(&light)?,
            serde_json::to_string(&dark)?,
            serde_json::to_string(&module_accents)?,
            session
                .user
                .as_ref()
                .map(|user| user.id.as_str())
                .unwrap_or(""),
            observed_at_ms,
        ],
    )?;
    record_workspace_branding_change_event(&conn, session, normalized, observed_at_ms)?;
    Ok(serde_json::json!({
        "ok": true,
        "kind": "workspace_branding",
        "action": "updated",
        "branding": workspace_branding_for_rxdb(root)?
    }))
}

fn default_workspace_branding_projection() -> Value {
    serde_json::json!({
        "id": WORKSPACE_BRANDING_ID,
        "ok": true,
        "custom": false,
        "name": "CTOX Default",
        "light": {},
        "dark": {},
        "module_accents": {},
        "updated_at_ms": 0,
        "is_deleted": false
    })
}

fn normalize_workspace_branding_request(
    request: WorkspaceBrandingUpdateRequest,
) -> anyhow::Result<Value> {
    anyhow::ensure!(
        request.light.is_object(),
        "branding light object is required"
    );
    anyhow::ensure!(request.dark.is_object(), "branding dark object is required");
    let name = clean_workspace_branding_name(&request.name);
    let light = normalize_workspace_branding_tokens("light", &request.light)?;
    let dark = normalize_workspace_branding_tokens("dark", &request.dark)?;
    let module_accents = normalize_workspace_branding_module_accents(&request.module_accents)?;
    validate_workspace_branding_contrast("light", &light)?;
    validate_workspace_branding_contrast("dark", &dark)?;
    Ok(serde_json::json!({
        "name": if name.is_empty() { "Workspace Branding" } else { name.as_str() },
        "light": light,
        "dark": dark,
        "module_accents": module_accents
    }))
}

fn normalize_workspace_branding_tokens(theme: &str, value: &Value) -> anyhow::Result<Value> {
    let object = value
        .as_object()
        .with_context(|| format!("branding {theme} tokens must be an object"))?;
    let mut out = serde_json::Map::new();
    for (key, raw) in object {
        anyhow::ensure!(
            WORKSPACE_BRANDING_TOKENS.contains(&key.as_str()),
            "unsupported branding token `{key}`"
        );
        let Some(text) = raw
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            anyhow::bail!("branding token `{key}` must be a non-empty string");
        };
        anyhow::ensure!(
            is_safe_workspace_branding_color(text),
            "branding token `{key}` contains an unsupported color value"
        );
        out.insert(key.clone(), Value::String(text.to_owned()));
    }
    Ok(Value::Object(out))
}

fn normalize_workspace_branding_module_accents(value: &Value) -> anyhow::Result<Value> {
    if value.is_null() {
        return Ok(serde_json::json!({}));
    }
    let object = value
        .as_object()
        .context("branding module_accents must be an object")?;
    let mut out = serde_json::Map::new();
    for (module_id, raw) in object {
        let id = safe_workspace_branding_module_id(module_id);
        anyhow::ensure!(!id.is_empty(), "branding module accent id is invalid");
        let Some(text) = raw
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            anyhow::bail!("module accent `{module_id}` must be a non-empty string");
        };
        anyhow::ensure!(
            is_safe_workspace_branding_color(text),
            "module accent `{module_id}` contains an unsupported color value"
        );
        out.insert(id, Value::String(text.to_owned()));
    }
    Ok(Value::Object(out))
}

fn validate_workspace_branding_contrast(theme: &str, tokens: &Value) -> anyhow::Result<()> {
    let merged = workspace_branding_tokens_with_defaults(theme, tokens);
    let surface = workspace_branding_color(&merged, "surface")?;
    for (token, min_ratio) in [
        ("text", 4.5),
        ("text_strong", 4.5),
        ("muted", 3.0),
        ("accent", 3.0),
        ("danger", 3.0),
        ("warning", 2.4),
        ("success", 3.0),
    ] {
        let color = workspace_branding_color(&merged, token)?;
        let ratio = contrast_ratio(color, surface);
        anyhow::ensure!(
            ratio >= min_ratio,
            "branding {theme} token `{token}` has insufficient contrast against `surface` ({ratio:.2}:1)"
        );
    }
    if let Some(accent_foreground) = merged.get("accent_foreground").and_then(Value::as_str) {
        let foreground = parse_workspace_branding_color(accent_foreground)
            .with_context(|| "branding accent_foreground is not parseable for contrast")?;
        let accent = workspace_branding_color(&merged, "accent")?;
        let ratio = contrast_ratio(foreground, accent);
        anyhow::ensure!(
            ratio >= 3.0,
            "branding {theme} token `accent_foreground` has insufficient contrast against `accent` ({ratio:.2}:1)"
        );
    }
    Ok(())
}

fn workspace_branding_tokens_with_defaults(
    theme: &str,
    tokens: &Value,
) -> serde_json::Map<String, Value> {
    let mut merged = serde_json::Map::new();
    let defaults: &[(&str, &str)] = if theme == "dark" {
        &[
            ("surface", "#11161b"),
            ("text", "#e7ecf2"),
            ("text_strong", "#ffffff"),
            ("muted", "#a3afbd"),
            ("accent", "#6cb8aa"),
            ("danger", "#e06b60"),
            ("warning", "#f59e0b"),
            ("success", "#10b981"),
        ]
    } else {
        &[
            ("surface", "oklch(0.987 0.005 235)"),
            ("text", "oklch(0.22 0.015 235)"),
            ("text_strong", "oklch(0.14 0.018 235)"),
            ("muted", "oklch(0.48 0.015 235)"),
            ("accent", "oklch(0.48 0.115 215)"),
            ("danger", "oklch(0.56 0.13 28)"),
            ("warning", "#b45309"),
            ("success", "#047857"),
        ]
    };
    for (key, value) in defaults {
        merged.insert((*key).to_owned(), Value::String((*value).to_owned()));
    }
    if let Some(object) = tokens.as_object() {
        for (key, value) in object {
            merged.insert(key.clone(), value.clone());
        }
    }
    merged
}

fn workspace_branding_color(
    tokens: &serde_json::Map<String, Value>,
    token: &str,
) -> anyhow::Result<(f64, f64, f64)> {
    let raw = tokens
        .get(token)
        .and_then(Value::as_str)
        .with_context(|| format!("branding token `{token}` is missing"))?;
    parse_workspace_branding_color(raw)
        .with_context(|| format!("branding token `{token}` is not parseable for contrast"))
}

fn is_safe_workspace_branding_color(value: &str) -> bool {
    let text = value.trim();
    if text.is_empty() || text.len() > 96 {
        return false;
    }
    let lower = text.to_ascii_lowercase();
    if lower.contains("url(")
        || lower.contains("var(")
        || lower.contains("attr(")
        || lower.contains("calc(")
        || lower.contains("@import")
        || text
            .chars()
            .any(|ch| matches!(ch, ';' | '{' | '}' | '<' | '>'))
    {
        return false;
    }
    text.starts_with('#')
        || lower.starts_with("rgb(")
        || lower.starts_with("rgba(")
        || lower.starts_with("hsl(")
        || lower.starts_with("hsla(")
        || lower.starts_with("oklch(")
        || lower.starts_with("oklab(")
}

fn parse_workspace_branding_color(value: &str) -> anyhow::Result<(f64, f64, f64)> {
    let text = value.trim();
    if let Some(hex) = text.strip_prefix('#') {
        return parse_hex_color(hex);
    }
    let lower = text.to_ascii_lowercase();
    if lower.starts_with("rgb(") || lower.starts_with("rgba(") {
        return parse_rgb_color(text);
    }
    if lower.starts_with("hsl(") || lower.starts_with("hsla(") {
        return parse_hsl_color(text);
    }
    if lower.starts_with("oklch(") {
        return parse_oklch_color(text);
    }
    if lower.starts_with("oklab(") {
        return parse_oklab_color(text);
    }
    anyhow::bail!("unsupported color format")
}

fn parse_hex_color(hex: &str) -> anyhow::Result<(f64, f64, f64)> {
    let expand = |ch: char| -> String { format!("{ch}{ch}") };
    let (r, g, b) = match hex.len() {
        3 | 4 => (
            expand(hex.chars().nth(0).unwrap()),
            expand(hex.chars().nth(1).unwrap()),
            expand(hex.chars().nth(2).unwrap()),
        ),
        6 | 8 => (
            hex[0..2].to_owned(),
            hex[2..4].to_owned(),
            hex[4..6].to_owned(),
        ),
        _ => anyhow::bail!("invalid hex color length"),
    };
    Ok((
        u8::from_str_radix(&r, 16)? as f64 / 255.0,
        u8::from_str_radix(&g, 16)? as f64 / 255.0,
        u8::from_str_radix(&b, 16)? as f64 / 255.0,
    ))
}

fn parse_rgb_color(value: &str) -> anyhow::Result<(f64, f64, f64)> {
    let inner = color_function_inner(value)?;
    let parts = color_components(inner);
    anyhow::ensure!(parts.len() >= 3, "rgb color requires three components");
    Ok((
        parse_rgb_component(&parts[0])?,
        parse_rgb_component(&parts[1])?,
        parse_rgb_component(&parts[2])?,
    ))
}

fn parse_hsl_color(value: &str) -> anyhow::Result<(f64, f64, f64)> {
    let inner = color_function_inner(value)?;
    let parts = color_components(inner);
    anyhow::ensure!(parts.len() >= 3, "hsl color requires three components");
    let hue = parse_hue_degrees(&parts[0])?.rem_euclid(360.0);
    let saturation = parse_percent_component(&parts[1])?;
    let lightness = parse_percent_component(&parts[2])?;
    Ok(hsl_to_srgb(hue, saturation, lightness))
}

fn parse_oklch_color(value: &str) -> anyhow::Result<(f64, f64, f64)> {
    let inner = color_function_inner(value)?;
    let parts = color_components(inner);
    anyhow::ensure!(parts.len() >= 3, "oklch color requires three components");
    let l = parse_unit_or_percent(&parts[0])?;
    let c = parse_plain_float(&parts[1])?;
    let h = parse_hue_degrees(&parts[2])?.to_radians();
    let a = c * h.cos();
    let b = c * h.sin();
    oklab_to_srgb(l, a, b)
}

fn parse_oklab_color(value: &str) -> anyhow::Result<(f64, f64, f64)> {
    let inner = color_function_inner(value)?;
    let parts = color_components(inner);
    anyhow::ensure!(parts.len() >= 3, "oklab color requires three components");
    oklab_to_srgb(
        parse_unit_or_percent(&parts[0])?,
        parse_plain_float(&parts[1])?,
        parse_plain_float(&parts[2])?,
    )
}

fn color_function_inner(value: &str) -> anyhow::Result<&str> {
    let start = value
        .find('(')
        .context("color function missing open paren")?;
    let end = value
        .rfind(')')
        .context("color function missing close paren")?;
    anyhow::ensure!(end > start, "invalid color function");
    Ok(&value[start + 1..end])
}

fn color_components(inner: &str) -> Vec<String> {
    inner
        .replace(',', " ")
        .split('/')
        .next()
        .unwrap_or("")
        .split_whitespace()
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect()
}

fn parse_rgb_component(value: &str) -> anyhow::Result<f64> {
    let trimmed = value.trim();
    if let Some(percent) = trimmed.strip_suffix('%') {
        return Ok((percent.trim().parse::<f64>()? / 100.0).clamp(0.0, 1.0));
    }
    Ok((trimmed.parse::<f64>()? / 255.0).clamp(0.0, 1.0))
}

fn parse_unit_or_percent(value: &str) -> anyhow::Result<f64> {
    let trimmed = value.trim();
    if let Some(percent) = trimmed.strip_suffix('%') {
        return Ok((percent.trim().parse::<f64>()? / 100.0).clamp(0.0, 1.0));
    }
    Ok(trimmed.parse::<f64>()?.clamp(0.0, 1.0))
}

fn parse_plain_float(value: &str) -> anyhow::Result<f64> {
    Ok(value.trim().parse::<f64>()?)
}

fn parse_percent_component(value: &str) -> anyhow::Result<f64> {
    let percent = value
        .trim()
        .strip_suffix('%')
        .context("hsl saturation/lightness must be percentages")?;
    Ok((percent.trim().parse::<f64>()? / 100.0).clamp(0.0, 1.0))
}

fn parse_hue_degrees(value: &str) -> anyhow::Result<f64> {
    Ok(value.trim().trim_end_matches("deg").trim().parse::<f64>()?)
}

fn hsl_to_srgb(hue: f64, saturation: f64, lightness: f64) -> (f64, f64, f64) {
    let chroma = (1.0 - (2.0 * lightness - 1.0).abs()) * saturation;
    let hue_prime = hue / 60.0;
    let x = chroma * (1.0 - (hue_prime.rem_euclid(2.0) - 1.0).abs());
    let (r1, g1, b1) = match hue_prime.floor() as i32 {
        0 => (chroma, x, 0.0),
        1 => (x, chroma, 0.0),
        2 => (0.0, chroma, x),
        3 => (0.0, x, chroma),
        4 => (x, 0.0, chroma),
        _ => (chroma, 0.0, x),
    };
    let m = lightness - chroma / 2.0;
    (r1 + m, g1 + m, b1 + m)
}

fn oklab_to_srgb(l: f64, a: f64, b: f64) -> anyhow::Result<(f64, f64, f64)> {
    let l_ = l + 0.396_337_777_4 * a + 0.215_803_757_3 * b;
    let m_ = l - 0.105_561_345_8 * a - 0.063_854_172_8 * b;
    let s_ = l - 0.089_484_177_5 * a - 1.291_485_548_0 * b;
    let l3 = l_ * l_ * l_;
    let m3 = m_ * m_ * m_;
    let s3 = s_ * s_ * s_;
    let r = 4.076_741_662_1 * l3 - 3.307_711_591_3 * m3 + 0.230_969_929_2 * s3;
    let g = -1.268_438_004_6 * l3 + 2.609_757_401_1 * m3 - 0.341_319_396_5 * s3;
    let b = -0.004_196_086_3 * l3 - 0.703_418_614_7 * m3 + 1.707_614_701_0 * s3;
    Ok((linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(b)))
}

fn linear_to_srgb(value: f64) -> f64 {
    let value = value.clamp(0.0, 1.0);
    if value <= 0.003_130_8 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn contrast_ratio(left: (f64, f64, f64), right: (f64, f64, f64)) -> f64 {
    let a = relative_luminance(left);
    let b = relative_luminance(right);
    let (lighter, darker) = if a >= b { (a, b) } else { (b, a) };
    (lighter + 0.05) / (darker + 0.05)
}

fn relative_luminance((r, g, b): (f64, f64, f64)) -> f64 {
    0.2126 * luminance_channel(r) + 0.7152 * luminance_channel(g) + 0.0722 * luminance_channel(b)
}

fn luminance_channel(value: f64) -> f64 {
    if value <= 0.03928 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn clean_workspace_branding_name(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(80)
        .collect()
}

fn safe_workspace_branding_module_id(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_owned()
}

fn record_workspace_branding_change_event(
    conn: &Connection,
    session: &BusinessOsSession,
    current: Value,
    observed_at_ms: i64,
) -> anyhow::Result<()> {
    insert_business_event(
        conn,
        "business_workspace_branding",
        WORKSPACE_BRANDING_ID,
        "business_os.workspace_branding.changed",
        serde_json::json!({
            "event_type": "business_os.workspace_branding.changed",
            "actor": session_audit_actor_context(session),
            "current": current,
            "observed_at_ms": observed_at_ms
        }),
        observed_at_ms,
    )
}
