use crate::browser_agent_bridge::BrowserAgentBridgeState;
use crate::contracts::AgentState;
use crate::contracts::Bios;
use crate::contracts::BrowserEnginePolicy;
use crate::contracts::BrowserEngineState;
use crate::contracts::BrowserSubworkerPolicy;
use crate::contracts::BootEntry;
use crate::contracts::Genome;
use crate::contracts::HomepagePolicy;
use crate::contracts::ModelPolicy;
use crate::contracts::Organigram;
use crate::contracts::RootAuthState;
use crate::contracts::SystemCensus;
use crate::contracts::describe_browser_vision_kleinhirn_selection;
use crate::contracts::describe_kleinhirn_selection;
use crate::contracts::recommended_kleinhirn;
use crate::runtime_db::BiosDialogueEntry;
use crate::runtime_db::BiosUploadRecord;
use crate::runtime_db::HomepageRevisionRecord;
use crate::runtime_db::LearningEntryRecord;
use crate::runtime_db::MemoryItemRecord;
use crate::runtime_db::OwnerTrustSnapshot;
use crate::runtime_db::PersonNoteRecord;
use crate::runtime_db::PersonProfileRecord;
use crate::runtime_db::ProactiveContactCandidateRecord;
use crate::runtime_db::ResourceRecord;
use crate::runtime_db::SkillRecord;
use crate::runtime_db::WorkerJobRecord;
use serde_json::json;

fn esc(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
    .replace('"', "&quot;")
    .replace('\'', "&#39;")
}

fn esc_attr(input: &str) -> String {
    esc(input)
}

fn layout(title: &str, body: &str) -> String {
    format!(
        r#"<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f6f4ee;
      --card: #fffdfa;
      --ink: #1e1b18;
      --muted: #6d655d;
      --line: #d8d1c8;
      --accent: #8a4b17;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background: linear-gradient(180deg, #faf7f1 0%, var(--bg) 100%);
    }}
    header {{
      padding: 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.65);
      backdrop-filter: blur(6px);
      position: sticky;
      top: 0;
    }}
    nav a {{
      margin-right: 14px;
      color: var(--accent);
      text-decoration: none;
      font-weight: bold;
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      gap: 18px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 6px 20px rgba(0,0,0,0.04);
    }}
    .hero {{
      background: linear-gradient(135deg, #fffdfa 0%, #f4e6d8 100%);
    }}
    .muted {{ color: var(--muted); }}
    .warn {{ color: #965200; }}
    .grid {{
      display: grid;
      gap: 18px;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    }}
    label {{ display: block; font-weight: bold; margin-bottom: 6px; }}
    input, textarea {{
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }}
    textarea {{ min-height: 120px; resize: vertical; }}
    button {{
      margin-top: 12px;
      padding: 10px 14px;
      border: 0;
      border-radius: 999px;
      background: var(--accent);
      color: #fff;
      font: inherit;
      cursor: pointer;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #fbfaf7;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
      overflow: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    td, th {{
      border-bottom: 1px solid var(--line);
      padding: 8px 0;
      vertical-align: top;
      text-align: left;
    }}
  </style>
</head>
<body>
  <header>
    <nav>
      <a href="/">Home</a>
      <a href="/chat">Bootstrap Chat</a>
      <a href="/bios">BIOS</a>
      <a href="/org">Organigramm</a>
      <a href="/root-auth">Root Auth</a>
      <a href="/history">History</a>
      <a href="/browser">Browser</a>
      <a href="/models">Models</a>
      <a href="/census">Census</a>
    </nav>
  </header>
  <main>{body}</main>
</body>
</html>"#,
        title = esc(title),
        body = body,
    )
}

fn bios_vanilla_layout(title: &str, body: &str, script: &str) -> String {
    format!(
        r#"<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #001a8c;
      --panel: #0024ad;
      --deep: #00145f;
      --line: #7d95ea;
      --text: #eef3ff;
      --muted: #b8c7ff;
      --head: #d7d7d7;
      --head-text: #111;
      --accent: #ffd95e;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ height: 100%; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.4 "Courier New", monospace;
    }}
    a {{ color: inherit; }}
    .bios {{
      min-height: 100%;
      display: grid;
      grid-template-rows: auto auto auto 1fr auto;
    }}
    .bar, .foot {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
      padding: 6px 10px;
    }}
    .bar {{
      background: var(--head);
      color: var(--head-text);
      border-bottom: 2px solid #000;
    }}
    .foot {{
      border-top: 1px solid var(--line);
      color: var(--muted);
    }}
    .top-links a {{
      color: var(--head-text);
      text-decoration: none;
      margin-left: 12px;
    }}
    .flash {{
      margin: 8px;
      border: 1px solid var(--line);
      background: #00104d;
      color: var(--accent);
      padding: 10px;
    }}
    #tabs {{
      display: flex;
      gap: 4px;
      flex-wrap: wrap;
      padding: 8px 8px 0;
    }}
    .tab {{
      border: 1px solid var(--line);
      border-bottom: 0;
      background: #1730a5;
      color: var(--muted);
      padding: 6px 10px;
      cursor: pointer;
      font: inherit;
    }}
    .tab.on {{
      background: var(--panel);
      color: var(--text);
    }}
    .body {{
      display: grid;
      grid-template-columns: 340px 1fr;
      min-height: calc(100vh - 120px);
      padding: 0 8px 8px;
    }}
    .left, .right {{
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 10px;
      overflow: auto;
    }}
    .left {{
      border-right: 0;
    }}
    .pane {{
      display: none;
    }}
    .pane.on {{
      display: block;
    }}
    fieldset {{
      margin: 0 0 10px;
      border: 1px solid var(--line);
      padding: 10px;
    }}
    legend {{
      padding: 0 6px;
      color: var(--accent);
    }}
    label {{
      display: block;
      margin: 0 0 8px;
      color: var(--muted);
    }}
    input, textarea, select, button {{
      width: 100%;
      font: inherit;
    }}
    input, textarea, select {{
      margin-top: 4px;
      padding: 6px;
      color: var(--text);
      background: var(--deep);
      border: 1px solid var(--line);
    }}
    textarea {{
      min-height: 82px;
      resize: vertical;
    }}
    input[readonly], textarea[readonly] {{
      opacity: 0.9;
    }}
    button {{
      padding: 6px 8px;
      border: 1px solid var(--line);
      background: #1738cb;
      color: var(--text);
      cursor: pointer;
    }}
    .panel {{
      background: var(--deep);
      border: 1px solid #5b74d8;
      padding: 10px;
      margin-bottom: 10px;
    }}
    .prompt {{
      color: var(--accent);
      margin-bottom: 8px;
    }}
    .muted {{
      color: var(--muted);
      font-size: 12px;
    }}
    .chatlog {{
      background: #00104d;
      border: 1px solid var(--line);
      padding: 8px;
      min-height: 220px;
      max-height: 340px;
      overflow: auto;
      margin-bottom: 8px;
    }}
    .msg {{
      padding: 6px 0;
      border-bottom: 1px dotted #5e79e2;
    }}
    .msg:last-child {{
      border-bottom: 0;
    }}
    .meta {{
      color: var(--muted);
      font-size: 12px;
    }}
    .tree, .code {{
      background: #00104d;
      border: 1px solid var(--line);
      padding: 10px;
      white-space: pre-wrap;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      border: 1px solid #5f7ce6;
      padding: 6px;
      vertical-align: top;
      text-align: left;
    }}
    th {{
      font-weight: normal;
      color: var(--muted);
    }}
    img {{
      max-width: 100%;
      border: 1px solid var(--line);
    }}
    @media (max-width: 900px) {{
      .body, .grid {{
        grid-template-columns: 1fr;
      }}
      .left {{
        border-right: 1px solid var(--line);
      }}
    }}
  </style>
</head>
<body>
{body}
<script>{script}</script>
</body>
</html>"#,
        title = esc(title),
        body = body,
        script = script,
    )
}

fn banner(message: Option<&str>) -> String {
    match message {
        Some(msg) if !msg.is_empty() => {
            format!(r#"<div class="card"><strong>{}</strong></div>"#, esc(msg))
        }
        _ => String::new(),
    }
}

pub fn home_page(
    message: Option<&str>,
    genome: &Genome,
    homepage: &HomepagePolicy,
    bios: &Bios,
    root_auth: &RootAuthState,
    organigram: &Organigram,
    model_policy: &ModelPolicy,
    census: &SystemCensus,
    trust: &OwnerTrustSnapshot,
    dialogue: &[BiosDialogueEntry],
    homepage_revisions: &[HomepageRevisionRecord],
    uploads: &[BiosUploadRecord],
    state: &AgentState,
    browser_state: &BrowserEngineState,
    testimony_count: usize,
) -> String {
    let boot_status = if bios.frozen { "BIOS_FROZEN" } else { "BOOTSTRAP" };
    let branded_title = if homepage.owner_branding_applied && !trust.committed_owner_name.is_empty()
    {
        format!("{} for {}", homepage.current_title, trust.committed_owner_name)
    } else {
        homepage.current_title.clone()
    };
    let homepage_chat_speaker = if !trust.committed_owner_name.is_empty() {
        trust.committed_owner_name.as_str()
    } else if !organigram.owner.name.is_empty() {
        organigram.owner.name.as_str()
    } else {
        "Michael Welsch"
    };
    let dialogue_log = if dialogue.is_empty() {
        r#"<div class="msg"><div class="meta">system</div><div>Noch kein BIOS-Dialog vorhanden.</div></div>"#
            .to_string()
    } else {
        dialogue
            .iter()
            .map(|entry| {
                format!(
                    r#"<div class="msg"><div class="meta">{created} · {speaker} · Grosshirn {grosshirn}</div><div>{message}</div></div>"#,
                    created = esc(&entry.created_at),
                    speaker = esc(&entry.speaker),
                    grosshirn = if entry.used_grosshirn { "ja" } else { "nein" },
                    message = esc(&entry.message),
                )
            })
            .collect()
    };
    let revisions_rows = if homepage_revisions.is_empty() {
        r#"<tr><td colspan="5" class="muted">Noch keine Homepage-Revisionen vorhanden.</td></tr>"#
            .to_string()
    } else {
        homepage_revisions
            .iter()
            .map(|revision| {
                format!(
                    "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr><tr><td></td><td colspan=\"4\" class=\"muted\">{}</td></tr>",
                    esc(&revision.created_at),
                    esc(&revision.source_channel),
                    esc(&revision.title),
                    if revision.owner_branding_applied { "ja" } else { "nein" },
                    esc(&revision.notes),
                    esc(&revision.headline),
                )
            })
            .collect()
    };
    let action_rows = bios
        .communication_policy
        .action_rules
        .iter()
        .map(|rule| {
            format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                esc(&rule.action_class),
                esc(&rule.allowed_channels.join(", ")),
                rule.requires_root_verification,
                rule.requires_bios_primary,
                rule.terminal_only,
            )
        })
        .collect::<String>();
    let upload_cards = if uploads.is_empty() {
        r#"<div class="panel"><div class="muted">Noch keine Bilder im 1:1 Homepage-/BIOS-Chat hochgeladen.</div></div>"#
            .to_string()
    } else {
        uploads
            .iter()
            .map(|upload| {
                format!(
                    r#"<div class="panel"><p><strong>{speaker}</strong> via {channel} am {created}</p><p class="muted">{note}</p><p class="muted">Typ: {mime}</p><p><a href="{src}">{file}</a></p><img src="{src}" alt="{alt}"></div>"#,
                    speaker = esc(&upload.speaker),
                    channel = esc(&upload.source_channel),
                    created = esc(&upload.created_at),
                    note = esc(if upload.note.trim().is_empty() { "ohne Notiz" } else { &upload.note }),
                    mime = esc(&upload.mime_type),
                    src = esc_attr(&upload.public_path),
                    alt = esc_attr(&upload.file_name),
                    file = esc(&upload.file_name),
                )
            })
            .collect()
    };
    let organigram_tree = format!(
        "{}\n└─ Reports To: {}\n{}\n{}\n{}",
        if organigram.owner.name.trim().is_empty() {
            "Owner unbekannt".to_string()
        } else {
            organigram.owner.name.clone()
        },
        if organigram.reports_to.trim().is_empty() {
            "noch offen".to_string()
        } else {
            organigram.reports_to.clone()
        },
        if organigram.board.is_empty() {
            "   • Board: noch leer".to_string()
        } else {
            organigram
                .board
                .iter()
                .map(|entry| format!("   • Board: {entry}"))
                .collect::<Vec<_>>()
                .join("\n")
        },
        "   └─ CTO-Agent".to_string(),
        if organigram.subordinates.agents.is_empty() {
            "      └─ noch keine untergeordneten Agents".to_string()
        } else {
            organigram
                .subordinates
                .agents
                .iter()
                .map(|entry| format!("      └─ {entry}"))
                .collect::<Vec<_>>()
                .join("\n")
        },
    );
    let footer_right = format!(
        "Template {} · Phase {} · BIOS sichtbar {} · Loop {}",
        homepage.template_name,
        homepage.stage,
        homepage.bios_visible,
        state.loop_health
    );
    let page_body = format!(
        r#"<div class="bios">
  <div class="bar">
    <div>{title}</div>
    <div class="top-links">
      <a href="/">Home</a>
      <a href="/bios">BIOS</a>
      <a href="/org">Organigramm</a>
      <a href="/root-auth">Root Auth</a>
      <a href="/browser">Browser</a>
    </div>
  </div>
  {banner}
  <div id="tabs">
    <button class="tab on" data-tab="main" type="button">Main</button>
    <button class="tab" data-tab="organigram" type="button">Organigram</button>
    <button class="tab" data-tab="runtime" type="button">Runtime</button>
    <button class="tab" data-tab="revisions" type="button">Revisions</button>
    <button class="tab" data-tab="uploads" type="button">Uploads</button>
    <button class="tab" data-tab="root" type="button">Root</button>
  </div>
  <div class="body">
    <aside class="left">
      <div id="left-main" class="pane on">
        <form method="post" action="/homepage/update">
          <fieldset>
            <legend>Homepage Template</legend>
            <input type="hidden" name="source_channel" value="terminal">
            <label for="title">Title<input id="title" name="title" value="{raw_title}"></label>
            <label for="headline">Headline<textarea id="headline" name="headline">{raw_headline}</textarea></label>
            <label for="intro">Intro<textarea id="intro" name="intro">{raw_intro}</textarea></label>
            <label for="communication_note">Communication<textarea id="communication_note" name="communication_note">{communication_note_raw}</textarea></label>
            <label for="terminal_fallback_note">Terminal Fallback<textarea id="terminal_fallback_note" name="terminal_fallback_note">{terminal_fallback_note_raw}</textarea></label>
            <button type="submit">Homepage-Template speichern</button>
          </fieldset>
        </form>
        <fieldset>
          <legend>Status</legend>
          <label>Homepage State<input value="{homepage_stage}" readonly></label>
          <label>Boot Status<input value="{boot_status}" readonly></label>
          <label>Brain Access<input value="{brain_access}" readonly></label>
          <label>Loop Health<input value="{loop_health}" readonly></label>
          <label>Heartbeat<input value="{heartbeat}" readonly></label>
        </fieldset>
      </div>
      <div id="left-organigram" class="pane">
        <fieldset>
          <legend>Organigram</legend>
          <label>Owner<input value="{owner_name}" readonly></label>
          <label>Reports To<input value="{reports_to}" readonly></label>
          <label>Board<textarea readonly>{board}</textarea></label>
          <label>Subagents<textarea readonly>{sub_agents}</textarea></label>
        </fieldset>
      </div>
      <div id="left-runtime" class="pane">
        <fieldset>
          <legend>Runtime</legend>
          <label>Kleinhirn<input value="{kleinhirn}" readonly></label>
          <label>Browser Vision<input value="{browser_vision}" readonly></label>
          <label>Browser Engine<input value="{browser_status}" readonly></label>
          <label>Chrome<input value="{chrome_binary}" readonly></label>
          <label>Supervisor<input value="{supervisor_status}" readonly></label>
        </fieldset>
        <fieldset>
          <legend>Policy</legend>
          <label>Template<input value="{template_name}" readonly></label>
          <label>Homepage Ready<input value="{homepage_ready}" readonly></label>
          <label>Redesign via Terminal<input value="{redesign_terminal}" readonly></label>
          <label>Redesign via BIOS<input value="{redesign_bios}" readonly></label>
        </fieldset>
      </div>
      <div id="left-revisions" class="pane">
        <fieldset>
          <legend>Current Contract</legend>
          <label>Homepage Title<input value="{raw_title}" readonly></label>
          <label>Headline<textarea readonly>{raw_headline}</textarea></label>
          <label>Intro<textarea readonly>{raw_intro}</textarea></label>
        </fieldset>
      </div>
      <div id="left-uploads" class="pane">
        <form method="post" action="/bios/upload" enctype="multipart/form-data">
          <fieldset>
            <legend>Upload into 1:1 Chat</legend>
            <input type="hidden" name="source_channel" value="homepage">
            <input type="hidden" name="redirect_to" value="/">
            <label for="homepage_upload_speaker">Speaker<input id="homepage_upload_speaker" name="speaker" value="{homepage_chat_speaker}"></label>
            <label for="homepage_upload_note">Note<textarea id="homepage_upload_note" name="note" placeholder="Warum ist dieses Bild wichtig und was sollen wir daran besprechen?"></textarea></label>
            <label for="homepage_upload_image">Bild<input id="homepage_upload_image" name="image" type="file" accept="image/*"></label>
            <button type="submit">Bild in den 1:1 Chat laden</button>
          </fieldset>
        </form>
      </div>
      <div id="left-root" class="pane">
        <form method="post" action="/homepage/branding-lock">
          <fieldset>
            <legend>Root / Branding</legend>
            <label>Superpassword<input value="{root_configured}" readonly></label>
            <label>BIOS Primary<input value="{bios_primary}" readonly></label>
            <label>Branding Locked<input value="{branding_locked}" readonly></label>
            <label for="branding_password">Superpassword zur Root-Verifikation<input id="branding_password" name="password" type="password"></label>
            <button type="submit">Owner-Branding sperren</button>
          </fieldset>
        </form>
      </div>
    </aside>
    <main class="right">
      <div id="right-main" class="pane on">
        <div class="panel"><div class="prompt">START HERE: CHAT WITH CTO</div><div class="muted">{communication_note}</div></div>
        <div class="panel">
          <table>
            <tr><th>Homepage</th><td>{title}</td></tr>
            <tr><th>Owner</th><td>{owner_display}</td></tr>
            <tr><th>Terminal Fallback</th><td>{terminal_fallback_note}</td></tr>
            <tr><th>Boot Testimony</th><td>{testimony_count}</td></tr>
            <tr><th>Desktop / Headless</th><td>{browser_desktop} / {browser_headless}</td></tr>
          </table>
        </div>
        <div class="panel">
          <form method="post" action="/bios/chat">
            <label for="homepage_bios_speaker">Sprecher<input id="homepage_bios_speaker" name="speaker" value="{homepage_chat_speaker}"></label>
            <label for="homepage_bios_message">Nachricht<textarea id="homepage_bios_message" name="message" placeholder="Lass uns das bitte im 1:1 Chat auf der Homepage klaeren."></textarea></label>
            <button type="submit">In den 1:1 BIOS-Chat wechseln</button>
          </form>
        </div>
        <div class="chatlog">{dialogue_log}</div>
      </div>
      <div id="right-organigram" class="pane">
        <div class="panel">
          <div class="prompt">ORGANIGRAM PREVIEW</div>
          <div class="tree">{organigram_tree}</div>
        </div>
      </div>
      <div id="right-runtime" class="pane">
        <div class="grid">
          <div class="panel">
            <div class="prompt">RUNTIME SUMMARY</div>
            <table>
              <tr><th>Boot</th><td>{boot_status}</td></tr>
              <tr><th>Supervisor</th><td>{supervisor_status}</td></tr>
              <tr><th>Loop Health</th><td>{loop_health}</td></tr>
              <tr><th>Browser</th><td>{browser_status}</td></tr>
              <tr><th>Heartbeat</th><td>{heartbeat}</td></tr>
            </table>
          </div>
          <div class="panel">
            <div class="prompt">GENOME / POLICY</div>
            <div class="code">Immutable genes ({immutable_gene_count}): {immutable_gene_sample}

Adaptive surfaces ({adaptive_surface_count}): {adaptive_surface_sample}</div>
          </div>
        </div>
      </div>
      <div id="right-revisions" class="pane">
        <div class="panel">
          <div class="prompt">HOMEPAGE REVISIONS</div>
          <table>
            <tr><th>Zeit</th><th>Quelle</th><th>Titel</th><th>Branding</th><th>Notiz</th></tr>
            {revisions_rows}
          </table>
        </div>
      </div>
      <div id="right-uploads" class="pane">
        <div class="panel"><div class="prompt">UPLOADS IM 1:1 CHAT</div></div>
        <div class="grid">{upload_cards}</div>
      </div>
      <div id="right-root" class="pane">
        <div class="panel">
          <div class="prompt">TRUST GATES</div>
          <table>
            <tr><th>Aktion</th><th>Kanaele</th><th>Root</th><th>BIOS primary</th><th>Terminal only</th></tr>
            {action_rows}
          </table>
        </div>
      </div>
    </main>
  </div>
  <div class="foot">
    <div>Terminal bleibt immer Fallback. BIOS bleibt sichtbar.</div>
    <div>{footer_right}</div>
  </div>
</div>"#,
        title = esc(&branded_title),
        banner = banner(message),
        raw_title = esc(&homepage.current_title),
        raw_headline = esc(&homepage.current_headline),
        raw_intro = esc(&homepage.current_intro),
        communication_note_raw = esc(&homepage.communication_note),
        terminal_fallback_note_raw = esc(&homepage.terminal_fallback_note),
        homepage_stage = esc(&homepage.stage),
        boot_status = esc(boot_status),
        brain_access = esc(&trust.brain_access_mode),
        loop_health = esc(&state.loop_health),
        heartbeat = esc(state.last_heartbeat_at.as_deref().unwrap_or("noch keiner")),
        owner_name = esc(if organigram.owner.name.trim().is_empty() { "unbekannt" } else { &organigram.owner.name }),
        reports_to = esc(if organigram.reports_to.trim().is_empty() { "noch offen" } else { &organigram.reports_to }),
        board = esc(&organigram.board.join("\n")),
        sub_agents = esc(&organigram.subordinates.agents.join("\n")),
        kleinhirn = esc(&describe_kleinhirn_selection(model_policy, census)),
        browser_vision = esc(&describe_browser_vision_kleinhirn_selection(model_policy, census)),
        browser_status = esc(&browser_state.status),
        chrome_binary = esc(browser_state.chrome_binary.as_deref().unwrap_or("nicht gefunden")),
        supervisor_status = esc(&state.supervisor_status),
        template_name = esc(&homepage.template_name),
        homepage_ready = homepage.homepage_ready,
        redesign_terminal = homepage.redesign_allowed_via_terminal,
        redesign_bios = homepage.redesign_allowed_via_bios_chat,
        root_configured = root_auth.configured,
        bios_primary = trust.bios_primary_channel_confirmed,
        branding_locked = homepage.owner_branding_locked,
        communication_note = esc(&homepage.communication_note),
        terminal_fallback_note = esc(&homepage.terminal_fallback_note),
        testimony_count = testimony_count,
        browser_desktop = browser_state.desktop_available,
        browser_headless = browser_state.headless_ready,
        homepage_chat_speaker = esc(homepage_chat_speaker),
        dialogue_log = dialogue_log,
        organigram_tree = esc(&organigram_tree),
        immutable_gene_count = genome.immutable_genes.len(),
        immutable_gene_sample = esc(&genome.immutable_genes.join(", ")),
        adaptive_surface_count = genome.adaptive_surfaces.len(),
        adaptive_surface_sample = esc(&genome.adaptive_surfaces.join(", ")),
        revisions_rows = revisions_rows,
        upload_cards = upload_cards,
        action_rows = action_rows,
        footer_right = esc(&footer_right),
        owner_display = esc(if !trust.committed_owner_name.is_empty() {
            &trust.committed_owner_name
        } else if !organigram.owner.name.is_empty() {
            &organigram.owner.name
        } else {
            "noch ungebunden"
        }),
    );
    bios_vanilla_layout(
        &branded_title,
        &page_body,
        r#"document.addEventListener("click", function (ev) {
  const tab = ev.target.closest("[data-tab]");
  if (!tab) return;
  const name = tab.getAttribute("data-tab");
  document.querySelectorAll(".tab").forEach((item) => {
    item.classList.toggle("on", item.getAttribute("data-tab") === name);
  });
  document.querySelectorAll(".left .pane").forEach((item) => {
    item.classList.toggle("on", item.id === "left-" + name);
  });
  document.querySelectorAll(".right .pane").forEach((item) => {
    item.classList.toggle("on", item.id === "right-" + name);
  });
});"#,
    )
}

pub fn chat_page(message: Option<&str>, entries: &[BootEntry]) -> String {
    let rows = if entries.is_empty() {
        r#"<tr><td colspan="3" class="muted">Noch keine Boot-Zeugnisse vorhanden.</td></tr>"#
            .to_string()
    } else {
        entries
            .iter()
            .rev()
            .take(25)
            .map(|entry| {
                format!(
                    "<tr><td>{}</td><td>{}</td><td>{}</td></tr>",
                    esc(&entry.timestamp),
                    esc(&entry.speaker),
                    esc(&entry.message),
                )
            })
            .collect()
    };

    layout(
        "Bootstrap Chat",
        &format!(
            r#"{banner}
<section class="card hero">
  <h1>Bootstrap-Chat</h1>
  <p>Hier sammelt der Agent das verbale Boot-Zeugnis seines Initiators.</p>
</section>
<section class="grid">
  <form class="card" method="post" action="/chat">
    <h2>Neues Boot-Zeugnis</h2>
    <label for="speaker">Sprecher</label>
    <input id="speaker" name="speaker" value="initiator">
    <label for="message">Nachricht</label>
    <textarea id="message" name="message" placeholder="Beschreibe Owner, CEO, Board, Mission, Systeme, Kanaele oder Regeln."></textarea>
    <button type="submit">Zeugnis speichern</button>
  </form>
  <section class="card">
    <h2>Letzte Zeugnisse</h2>
    <table>
      <tr><th>Zeit</th><th>Sprecher</th><th>Inhalt</th></tr>
      {rows}
    </table>
  </section>
</section>"#,
            banner = banner(message),
            rows = rows,
        ),
    )
}

pub fn org_page(message: Option<&str>, organigram: &Organigram, bios: &Bios) -> String {
    let frozen_note = if bios.frozen {
        r#"<p class="warn">BIOS ist eingefroren. Dieser Editor ist gesperrt.</p>"#
    } else {
        ""
    };
    layout(
        "Organigramm",
        &format!(
            r#"{banner}
<section class="card hero">
  <h1>Organigramm</h1>
  <p>Dieser Contract beschreibt die Machtstruktur rund um den CTO-Agenten.</p>
  {frozen_note}
</section>
<form class="card" method="post" action="/org">
  <label for="owner_name">Owner</label>
  <input id="owner_name" name="owner_name" value="{owner_name}">
  <label for="owner_email">Owner E-Mail</label>
  <input id="owner_email" name="owner_email" value="{owner_email}">
  <label for="ceo">CEO</label>
  <input id="ceo" name="ceo" value="{ceo}">
  <label for="reports_to">CTO-Agent berichtet an</label>
  <input id="reports_to" name="reports_to" value="{reports_to}">
  <label for="board">Board, eine Zeile pro Person</label>
  <textarea id="board" name="board">{board}</textarea>
  <label for="peer_cxos">Peer-CxO-Rollen, eine Zeile pro Rolle oder Person</label>
  <textarea id="peer_cxos" name="peer_cxos">{peer_cxos}</textarea>
  <label for="sub_people">Untergebene Menschen, eine Zeile pro Person</label>
  <textarea id="sub_people" name="sub_people">{sub_people}</textarea>
  <label for="sub_agents">Untergebene Agents, eine Zeile pro Agent</label>
  <textarea id="sub_agents" name="sub_agents">{sub_agents}</textarea>
  <label for="sub_vendors">Dienstleister, eine Zeile pro Eintrag</label>
  <textarea id="sub_vendors" name="sub_vendors">{sub_vendors}</textarea>
  <button type="submit" {disabled}>Organigramm speichern</button>
</form>"#,
            banner = banner(message),
            frozen_note = frozen_note,
            owner_name = esc(&organigram.owner.name),
            owner_email = esc(&organigram.owner.email),
            ceo = esc(&organigram.ceo),
            reports_to = esc(&organigram.reports_to),
            board = esc(&organigram.board.join("\n")),
            peer_cxos = esc(&organigram.peer_cxos.join("\n")),
            sub_people = esc(&organigram.subordinates.people.join("\n")),
            sub_agents = esc(&organigram.subordinates.agents.join("\n")),
            sub_vendors = esc(&organigram.subordinates.vendors.join("\n")),
            disabled = if bios.frozen { "disabled" } else { "" },
        ),
    )
}

pub fn root_auth_page(message: Option<&str>, root_auth: &RootAuthState) -> String {
    let status = if root_auth.configured {
        "gesetzt"
    } else {
        "noch nicht gesetzt"
    };
    layout(
        "Root Auth",
        &format!(
            r#"{banner}
<section class="card hero">
  <h1>Root Auth</h1>
  <p>Hier setzt der Root-Owner das Superpassword. Es darf nie ueber den normalen Chat laufen.</p>
</section>
<section class="grid">
  <div class="card">
    <h2>Status</h2>
    <p>Superpassword ist {status}.</p>
    <p class="muted">Es dient nur fuer Root-Verifikation, BIOS-Aenderungen und Notfaelle.</p>
  </div>
  <form class="card" method="post" action="/root-auth/set">
    <h2>Superpassword setzen</h2>
    <label for="password">Superpassword</label>
    <input id="password" name="password" type="password">
    <label for="confirm">Wiederholen</label>
    <input id="confirm" name="confirm" type="password">
    <button type="submit">Superpassword speichern</button>
  </form>
</section>"#,
            banner = banner(message),
            status = esc(status),
        ),
    )
}

pub fn bios_page(
    message: Option<&str>,
    genome: &Genome,
    bios: &Bios,
    trust: &OwnerTrustSnapshot,
    model_policy: &ModelPolicy,
    census: &SystemCensus,
    memory_summary: Option<&str>,
    learning_working_set: Option<&str>,
    learning_operational: Option<&str>,
    learning_general: Option<&str>,
    learning_negative: Option<&str>,
    people_working_set: Option<&str>,
    dialogue: &[BiosDialogueEntry],
    memory_items: &[MemoryItemRecord],
    learning_entries: &[LearningEntryRecord],
    person_profiles: &[PersonProfileRecord],
    person_notes: &[PersonNoteRecord],
    proactive_contacts: &[ProactiveContactCandidateRecord],
    resources: &[ResourceRecord],
    skills: &[SkillRecord],
    uploads: &[BiosUploadRecord],
) -> String {
    let bios_blob = json!({
        "agentIdentity": bios.agent_identity,
        "mission": bios.mission,
        "owner": bios.owner,
        "rootAuthorities": bios.root_authorities,
        "reportsTo": bios.reports_to,
        "board": bios.board,
        "peerCxos": bios.peer_cxos,
        "subordinates": bios.subordinates,
        "communicationPolicy": bios.communication_policy,
        "rootAuth": bios.root_auth,
        "changeRules": bios.change_rules,
        "frozen": bios.frozen,
        "frozenAt": bios.frozen_at,
        "websitePath": bios.website_path,
    });
    let selected_kleinhirn = recommended_kleinhirn(model_policy, census);
    let communication_rows = bios
        .communication_policy
        .channel_rules
        .iter()
        .map(|rule| {
            format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                esc(&rule.channel),
                esc(&rule.layer),
                esc(&rule.trust_level),
                esc(&rule.interpretation),
                esc(&rule.binding_power),
                esc(&rule.sensitive_topic_policy),
            )
        })
        .collect::<String>();
    let action_rows = bios
        .communication_policy
        .action_rules
        .iter()
        .map(|rule| {
            format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                esc(&rule.action_class),
                esc(&rule.description),
                esc(&rule.allowed_channels.join(", ")),
                rule.requires_root_verification,
                rule.requires_bios_primary,
                rule.terminal_only,
            )
        })
        .collect::<String>();
    let pretty = serde_json::to_string_pretty(&bios_blob).unwrap_or_else(|_| "{}".to_string());
    let status = if bios.frozen {
        "eingefroren"
    } else {
        "noch nicht eingefroren"
    };
    let dialogue_rows = if dialogue.is_empty() {
        r#"<tr><td colspan="4" class="muted">Noch kein BIOS-Dialog vorhanden.</td></tr>"#.to_string()
    } else {
        dialogue
            .iter()
            .map(|entry| {
                format!(
                    "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                    esc(&entry.created_at),
                    esc(&entry.speaker),
                    esc(&entry.message),
                    if entry.used_grosshirn { "ja" } else { "nein" }
                )
            })
            .collect()
    };
    let memory_rows = if memory_items.is_empty() {
        r#"<tr><td colspan="5" class="muted">Noch keine Memory-Eintraege vorhanden.</td></tr>"#.to_string()
    } else {
        memory_items
            .iter()
            .map(|item| {
                format!(
                    "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                    esc(&item.created_at),
                    esc(&item.kind),
                    esc(&item.summary),
                    esc(&item.source),
                    esc(&format!(
                        "{}{}",
                        if item.important { "[wichtig] " } else { "" },
                        item.detail
                    )),
                )
            })
            .collect()
    };
    let learning_rows = if learning_entries.is_empty() {
        r#"<tr><td colspan="8" class="muted">Noch keine aktiven Learnings im Lernpfad vorhanden.</td></tr>"#
            .to_string()
    } else {
        learning_entries
            .iter()
            .map(|entry| {
                format!(
                    "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.2}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                    esc(&entry.updated_at),
                    esc(&entry.learning_class),
                    esc(&entry.summary),
                    entry.confidence,
                    entry.salience,
                    esc(&entry.applicability),
                    esc(&entry.evidence),
                    esc(&format!(
                        "{}{}",
                        if entry.recall_count > 0 {
                            format!(
                                "Recall {}x, zuletzt {}. ",
                                entry.recall_count,
                                entry.last_recalled_at
                                    .as_deref()
                                    .unwrap_or("unbekannt")
                            )
                        } else {
                            String::new()
                        },
                        entry.detail
                    )),
                )
            })
            .collect()
    };
    let person_profile_rows = if person_profiles.is_empty() {
        r#"<tr><td colspan="8" class="muted">Noch keine Personenprofile vorhanden.</td></tr>"#.to_string()
    } else {
        person_profiles
            .iter()
            .map(|profile| {
                format!(
                    "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                    esc(&profile.display_name),
                    esc(if profile.primary_email.trim().is_empty() {
                        "unbekannt"
                    } else {
                        &profile.primary_email
                    }),
                    esc(&profile.relationship_kind),
                    esc(&profile.trust_level),
                    esc(&profile.last_interaction_at.clone().unwrap_or_else(|| "noch keine".to_string())),
                    profile.interaction_count,
                    esc(&profile.conversation_memory_summary),
                    esc(&profile.notebook_summary),
                )
            })
            .collect()
    };
    let person_note_rows = if person_notes.is_empty() {
        r#"<tr><td colspan="6" class="muted">Noch keine Personen-Notizen vorhanden.</td></tr>"#.to_string()
    } else {
        person_notes
            .iter()
            .map(|note| {
                format!(
                    "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                    esc(&note.updated_at),
                    esc(&note.person_display_name),
                    esc(&note.note_kind),
                    esc(&note.source_channel),
                    esc(&note.summary),
                    esc(&format!(
                        "{}{}",
                        if note.important { "[wichtig] " } else { "" },
                        note.detail
                    )),
                )
            })
            .collect()
    };
    let proactive_contact_rows = if proactive_contacts.is_empty() {
        r#"<tr><td colspan="10" class="muted">Noch keine proaktiven Kontaktentwuerfe vorhanden.</td></tr>"#
            .to_string()
    } else {
        proactive_contacts
            .iter()
            .map(|candidate| {
                format!(
                    "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                    esc(&candidate.updated_at),
                    esc(&candidate.person_display_name),
                    esc(&candidate.status),
                    esc(&candidate.channel),
                    esc(if candidate.dispatch_channel.trim().is_empty() {
                        "noch keiner"
                    } else {
                        &candidate.dispatch_channel
                    }),
                    esc(&candidate.subject),
                    esc(&candidate.rationale),
                    esc(&candidate.conflict_check),
                    esc(&candidate.dispatched_at.clone().unwrap_or_else(|| "noch nicht".to_string())),
                    esc(&format!(
                        "{} {} {}",
                        candidate.validation_decision,
                        candidate.validation_note,
                        format!(
                            "{}{}",
                            if candidate.outbound_message_id.trim().is_empty() {
                                String::new()
                            } else {
                                format!("[{}] ", candidate.outbound_message_id)
                            },
                            candidate.dispatch_note
                        )
                    )),
                )
            })
            .collect()
    };
    let resource_rows = if resources.is_empty() {
        r#"<tr><td colspan="5" class="muted">Noch keine Ressourcen gespiegelt.</td></tr>"#.to_string()
    } else {
        resources
            .iter()
            .map(|resource| {
                format!(
                    "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                    esc(&resource.category),
                    esc(&resource.name),
                    esc(&resource.status),
                    esc(&resource.detail),
                    esc(&resource.observed_at),
                )
            })
            .collect()
    };
    let skill_rows = if skills.is_empty() {
        r#"<tr><td colspan="4" class="muted">Noch keine Skills registriert.</td></tr>"#.to_string()
    } else {
        skills
            .iter()
            .map(|skill| {
                format!(
                    "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                    esc(&skill.name),
                    esc(&skill.status),
                    esc(&skill.path),
                    esc(&skill.notes),
                )
            })
            .collect()
    };
    let upload_rows = if uploads.is_empty() {
        r#"<tr><td colspan="6" class="muted">Noch keine Bild-Uploads im 1:1 Chat vorhanden.</td></tr>"#
            .to_string()
    } else {
        uploads
            .iter()
            .map(|upload| {
                format!(
                    r#"<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td><a href="{}">{}</a><br><span class="muted">{}</span></td><td><img src="{}" alt="{}" style="max-width:180px; border-radius:10px; border:1px solid #d8d1c8;"></td></tr>"#,
                    esc(&upload.created_at),
                    esc(&upload.speaker),
                    esc(&upload.source_channel),
                    esc(if upload.note.trim().is_empty() { "ohne Notiz" } else { &upload.note }),
                    esc_attr(&upload.public_path),
                    esc(&upload.file_name),
                    esc(&upload.mime_type),
                    esc_attr(&upload.public_path),
                    esc_attr(&upload.file_name),
                )
            })
            .collect()
    };
    layout(
        "BIOS",
        &format!(
            r#"{banner}
<section class="card hero">
  <h1>BIOS</h1>
  <p>Diese Seite ist die Vertrauens-, Kalibrierungs- und Steuerflaeche des CTO-Agenten.</p>
  <p><strong>Status:</strong> {status}</p>
</section>
<section class="grid">
  <div class="card">
    <h2>Genome</h2>
    <p><strong>Unveraenderliche Gene</strong></p>
    <pre>{immutable_genes}</pre>
    <p><strong>Adaptive Flaechen</strong></p>
    <pre>{adaptive_surfaces}</pre>
  </div>
  <div class="card">
    <h2>Owner Trust</h2>
    <table>
      <tr><th>Owner aus Contract</th><td>{owner}</td></tr>
      <tr><th>Commitment</th><td>{committed_owner}</td></tr>
      <tr><th>Owner-Kontakt</th><td>{owner_contact}</td></tr>
      <tr><th>BIOS als Primaerkanal</th><td>{bios_channel}</td></tr>
      <tr><th>Superpassword gesetzt</th><td>{superpassword}</td></tr>
      <tr><th>Commitment-Score</th><td>{commitment_score}/100</td></tr>
      <tr><th>Letzter Owner-Dialog</th><td>{last_owner_dialogue}</td></tr>
      <tr><th>Brain Access</th><td>{brain_access}</td></tr>
      <tr><th>Kleinhirn aktiv</th><td>{kleinhirn}</td></tr>
    </table>
    <p class="muted">{calibration_notes}</p>
  </div>
  <form class="card" method="post" action="/bios/update">
    <h2>BIOS-Entwurf</h2>
    <label for="agent_name">Agentenname</label>
    <input id="agent_name" name="agent_name" value="{agent_name}">
    <label for="mission">Mission</label>
    <textarea id="mission" name="mission" placeholder="Was ist die Mission dieses CTO-Agenten?">{mission}</textarea>
    <button type="submit" {disabled}>BIOS-Entwurf speichern</button>
    <p class="muted">Owner und Organigramm werden aus dem Organigramm-Contract gezogen.</p>
  </form>
  <div class="card">
    <h2>BIOS auf der Webseite</h2>
    <pre>{pretty}</pre>
    <p class="muted">Basis-Owner: {owner}</p>
  </div>
</section>
<section class="card">
  <h2>Kommunikations- und Vertrauensmatrix</h2>
  <table>
    <tr><th>Kanal</th><th>Ebene</th><th>Trust</th><th>Interpretation</th><th>Binding</th><th>Sensitive Topics</th></tr>
    {communication_rows}
  </table>
  <p class="muted">
    Sender muessen gegen das Organigramm geprueft werden: {organigram_check}. Unbekannte Sender fallen auf low trust: {unknown_sender_low_trust}.
    Sensitive Themen duerfen von low-trust Kanaelen auf {redirect_target} umgeleitet werden: {may_refuse}.
  </p>
</section>
<section class="card">
  <h2>Aenderungsklassen und erlaubte Ebenen</h2>
  <table>
    <tr><th>Aktion</th><th>Beschreibung</th><th>Kanaele</th><th>Root</th><th>BIOS primary</th><th>Terminal only</th></tr>
    {action_rows}
  </table>
</section>
<section class="grid">
  <form class="card" method="post" action="/bios/chat">
    <h2>BIOS-Chat</h2>
    <p>Hier soll der Besitzer fortan mit dem CTO-Agenten sprechen, Vertrauen aufbauen und ihn kalibrieren.</p>
    <p class="muted">Wenn hier Kritik an Homepage, BIOS-Sichtbarkeit oder Kommunikationspfad geaeussert wird, darf der Agent die Homepage ebenfalls ueber seinen Homepage-Skill anpassen.</p>
    <label for="bios_speaker">Sprecher</label>
    <input id="bios_speaker" name="speaker" value="{speaker}">
    <label for="bios_message">Nachricht</label>
    <textarea id="bios_message" name="message" placeholder="Kalibriere den Agenten ueber Zweck, Grenzen, Prioritaeten und Vertrauen."></textarea>
    <button type="submit">Im BIOS chatten</button>
  </form>
  <form class="card" method="post" action="/bios/brain-access">
    <h2>Brain Access</h2>
    <p>Hier legt der Owner fest, ob der Agent nur Kleinhirn oder auch Grosshirn nutzen darf.</p>
    <label for="brain_mode">Modus</label>
    <select id="brain_mode" name="mode">
      <option value="kleinhirn_only" {kleinhirn_only_selected}>Nur Kleinhirn</option>
      <option value="kleinhirn_plus_grosshirn" {kleinhirn_plus_grosshirn_selected}>Kleinhirn + Grosshirn</option>
    </select>
    <label for="brain_password">Superpassword zur Root-Verifikation</label>
    <input id="brain_password" name="password" type="password">
    <button type="submit">Brain Access speichern</button>
  </form>
  <form class="card" method="post" action="/root-auth/set">
    <h2>Superpassword im BIOS setzen</h2>
    <p>Der Root-Trust soll direkt aus dem BIOS heraus aufgebaut werden.</p>
    <label for="bios_password">Superpassword</label>
    <input id="bios_password" name="password" type="password">
    <label for="bios_confirm">Wiederholen</label>
    <input id="bios_confirm" name="confirm" type="password">
    <button type="submit">Superpassword speichern</button>
  </form>
  <form class="card" method="post" action="/bios/upload" enctype="multipart/form-data">
    <h2>Bild in den BIOS-Chat laden</h2>
    <p>Hier entsteht die erste komfortable Vertrauensstufe ueber dem Terminal. Bilder koennen direkt im 1:1 Chat geteilt werden.</p>
    <input type="hidden" name="source_channel" value="bios">
    <input type="hidden" name="redirect_to" value="/bios">
    <label for="bios_upload_speaker">Sprecher</label>
    <input id="bios_upload_speaker" name="speaker" value="{speaker}">
    <label for="bios_upload_note">Notiz</label>
    <textarea id="bios_upload_note" name="note" placeholder="Warum ist dieses Bild wichtig und was sollen wir daran besprechen?"></textarea>
    <label for="bios_upload_image">Bild</label>
    <input id="bios_upload_image" name="image" type="file" accept="image/*">
    <button type="submit">Bild in den BIOS-Chat laden</button>
  </form>
</section>
<section class="grid">
  <div class="card">
    <h2>Memory</h2>
    <p class="muted">{memory_summary}</p>
    <table>
      <tr><th>Zeit</th><th>Art</th><th>Summary</th><th>Quelle</th><th>Detail</th></tr>
      {memory_rows}
    </table>
  </div>
  <div class="card">
    <h2>Ressourcen</h2>
    <table>
      <tr><th>Kategorie</th><th>Name</th><th>Status</th><th>Detail</th><th>Observed</th></tr>
      {resource_rows}
    </table>
  </div>
  <div class="card">
    <h2>Skills</h2>
    <table>
      <tr><th>Name</th><th>Status</th><th>Pfad</th><th>Notiz</th></tr>
      {skill_rows}
    </table>
  </div>
</section>
<section class="card">
  <h2>Lernpfad</h2>
  <p class="muted">{learning_working_set}</p>
  <table>
    <tr><th>Klasse</th><th>Verdichtete Referenz</th></tr>
    <tr><td>operational</td><td>{learning_operational}</td></tr>
    <tr><td>general</td><td>{learning_general}</td></tr>
    <tr><td>negative</td><td>{learning_negative}</td></tr>
  </table>
  <p class="muted">Operational bleibt im taeglichen Arbeitsgedaechtnis, general sammelt breitere Erkenntnisse und negative konserviert Fehlschlaege, Sackgassen und Anti-Patterns.</p>
  <table>
    <tr><th>Updated</th><th>Klasse</th><th>Summary</th><th>Confidence</th><th>Salience</th><th>Applicability</th><th>Evidence</th><th>Detail</th></tr>
    {learning_rows}
  </table>
</section>
<section class="card">
  <h2>Personenpfade</h2>
  <p class="muted">{people_working_set}</p>
  <p class="muted">Jede Person bekommt ein eigenes Profil mit Gespraechsspur, Notebook-Referenzen und einer Guard-Notiz fuer proaktive Kontakte.</p>
  <table>
    <tr><th>Name</th><th>E-Mail</th><th>Beziehung</th><th>Trust</th><th>Letzte Interaktion</th><th>Interaktionen</th><th>Gespaechsspur</th><th>Notebook</th></tr>
    {person_profile_rows}
  </table>
</section>
<section class="card">
  <h2>Personen-Notizbuch</h2>
  <table>
    <tr><th>Updated</th><th>Person</th><th>Art</th><th>Quelle</th><th>Summary</th><th>Detail</th></tr>
    {person_note_rows}
  </table>
</section>
<section class="card">
  <h2>Proaktive Kontaktentwuerfe</h2>
  <p class="muted">Diese Eintraege entstehen modellgetrieben, werden vor Aussendung validiert und koennen danach autonom ueber den vorhandenen Versandpfad rausgehen. Die Tabelle zeigt daher auch den realen Dispatch-Zustand.</p>
  <table>
    <tr><th>Updated</th><th>Person</th><th>Status</th><th>Intent</th><th>Dispatch</th><th>Subject</th><th>Rationale</th><th>Konfliktcheck</th><th>Versendet</th><th>Validierung / Dispatch</th></tr>
    {proactive_contact_rows}
  </table>
</section>
<section class="card">
  <h2>Bild-Uploads im 1:1 Chat</h2>
  <table>
    <tr><th>Zeit</th><th>Sprecher</th><th>Quelle</th><th>Notiz</th><th>Datei</th><th>Vorschau</th></tr>
    {upload_rows}
  </table>
</section>
<section class="card">
  <h2>BIOS-Dialogverlauf</h2>
  <table>
    <tr><th>Zeit</th><th>Sprecher</th><th>Inhalt</th><th>Grosshirn</th></tr>
    {dialogue_rows}
  </table>
</section>
<section class="card">
  <h2>BIOS einfrieren</h2>
  <p>Zum Freeze braucht der Agent ein Organigramm, ein gesetztes Superpassword und die Root-Verifikation.</p>
  <form method="post" action="/bios/freeze">
    <label for="freeze_password">Superpassword fuer Root-Verifikation</label>
    <input id="freeze_password" name="password" type="password">
    <button type="submit" {disabled}>BIOS einfrieren</button>
  </form>
</section>"#,
            banner = banner(message),
            status = esc(status),
            immutable_genes = esc(&genome.immutable_genes.join("\n")),
            adaptive_surfaces = esc(&genome.adaptive_surfaces.join("\n")),
            agent_name = esc(&bios.agent_identity.agent_name),
            mission = esc(&bios.mission),
            pretty = esc(&pretty),
            owner = esc(if bios.owner.name.is_empty() {
                "noch nicht gesetzt"
            } else {
                &bios.owner.name
            }),
            committed_owner = esc(if trust.committed_owner_name.is_empty() {
                "noch nicht fest kalibriert"
            } else {
                &trust.committed_owner_name
            }),
            owner_contact = trust.owner_contact_established,
            bios_channel = trust.bios_primary_channel_confirmed,
            superpassword = trust.superpassword_set,
            commitment_score = trust.owner_commitment_score,
            last_owner_dialogue = esc(
                trust
                    .last_owner_dialogue_at
                    .as_deref()
                    .unwrap_or("noch keiner")
            ),
            brain_access = esc(&trust.brain_access_mode),
            kleinhirn = esc(&format!(
                "{} ({})",
                selected_kleinhirn.official_label, selected_kleinhirn.model_id
            )),
            communication_rows = communication_rows,
            action_rows = action_rows,
            organigram_check = bios.communication_policy.escalation_policy.sender_identity_checked_against_organigram,
            unknown_sender_low_trust = bios.communication_policy.escalation_policy.unknown_sender_defaults_to_low_trust,
            redirect_target = esc(&bios.communication_policy.escalation_policy.sensitive_topics_redirect_to),
            may_refuse = bios.communication_policy.escalation_policy.may_refuse_sensitive_topics_on_low_trust_channels,
            calibration_notes = esc(if trust.calibration_notes.is_empty() {
                "Noch keine Kalibrierungsnotizen vorhanden."
            } else {
                &trust.calibration_notes
            }),
            speaker = esc(if !bios.owner.name.is_empty() {
                &bios.owner.name
            } else {
                "Michael Welsch"
            }),
            kleinhirn_only_selected = if trust.brain_access_mode == "kleinhirn_only" {
                "selected"
            } else {
                ""
            },
            kleinhirn_plus_grosshirn_selected =
                if trust.brain_access_mode == "kleinhirn_plus_grosshirn" {
                    "selected"
                } else {
                    ""
                },
            memory_summary = esc(memory_summary.unwrap_or(
                "Noch keine verdichtete Memory-Zusammenfassung vorhanden."
            )),
            learning_working_set = esc(learning_working_set.unwrap_or(
                "Noch kein dauerhaft verdichtetes Learning-Working-Set vorhanden."
            )),
            learning_operational = esc(learning_operational.unwrap_or(
                "Noch keine operativen Kernlearnings vorhanden."
            )),
            learning_general = esc(learning_general.unwrap_or(
                "Noch keine allgemeinen Learnings vorhanden."
            )),
            learning_negative = esc(learning_negative.unwrap_or(
                "Noch keine negativen Learnings oder Anti-Patterns vorhanden."
            )),
            people_working_set = esc(people_working_set.unwrap_or(
                "Noch kein verdichtetes Personen-Working-Set vorhanden."
            )),
            memory_rows = memory_rows,
            learning_rows = learning_rows,
            person_profile_rows = person_profile_rows,
            person_note_rows = person_note_rows,
            proactive_contact_rows = proactive_contact_rows,
            resource_rows = resource_rows,
            skill_rows = skill_rows,
            upload_rows = upload_rows,
            dialogue_rows = dialogue_rows,
            disabled = if bios.frozen { "disabled" } else { "" },
        ),
    )
}

pub fn census_page(message: Option<&str>, bios: &Bios, census: &SystemCensus) -> String {
    let pretty = serde_json::to_string_pretty(census).unwrap_or_else(|_| "{}".to_string());
    layout(
        "Census",
        &format!(
            r#"{banner}
<section class="card hero">
  <h1>System Census</h1>
  <p>Nach dem BIOS-Freeze darf der Agent die erste read-only Erfassung seiner Umgebung machen.</p>
</section>
<section class="grid">
  <form class="card" method="post" action="/census/run">
    <h2>Read-only Census</h2>
    <p>Der Census erfasst Host, GPUs, VRAM und kann via <code>mistralrs tune</code> pruefen, welche lokalen Kleinhirn-Modelle dieser Host real tragen kann.</p>
    <button type="submit" {disabled}>Census ausfuehren</button>
  </form>
  <div class="card">
    <h2>Letzter Census</h2>
    <pre>{pretty}</pre>
  </div>
</section>"#,
            banner = banner(message),
            disabled = if bios.frozen { "" } else { "disabled" },
            pretty = esc(&pretty),
        ),
    )
}

pub fn history_page(message: Option<&str>, origin_story: &str, creation_ledger: &str) -> String {
    layout(
        "History",
        &format!(
            r#"{banner}
<section class="card hero">
  <h1>Entstehungsgeschichte</h1>
  <p>Diese Seite ist das Historiker-Gedaechtnis des CTO-Agenten. Sie haelt Herkunft, Gruendungszweck und wichtige Entwicklungsschritte fest.</p>
</section>
<section class="grid">
  <div class="card">
    <h2>Ursprung</h2>
    <pre>{origin_story}</pre>
  </div>
  <div class="card">
    <h2>Chronik</h2>
    <pre>{creation_ledger}</pre>
  </div>
</section>"#,
            banner = banner(message),
            origin_story = esc(origin_story),
            creation_ledger = esc(creation_ledger),
        ),
    )
}

pub fn models_page(message: Option<&str>, model_policy: &ModelPolicy, census: &SystemCensus) -> String {
    let selected = recommended_kleinhirn(model_policy, census);
    let browser_vision_selected = describe_browser_vision_kleinhirn_selection(model_policy, census);
    let pretty = serde_json::to_string_pretty(model_policy).unwrap_or_else(|_| "{}".to_string());
    let gpu_inventory = census
        .gpus
        .as_ref()
        .filter(|gpus| !gpus.is_empty())
        .map(|gpus| {
            gpus.iter()
                .map(|gpu| format!("#{} {} ({} MiB)", gpu.index, gpu.name, gpu.memory_total_mb))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_else(|| "keine GPU-Daten vorhanden".to_string());
    let tune_rows = census
        .model_tune_candidates
        .as_ref()
        .filter(|candidates| !candidates.is_empty())
        .map(|candidates| {
            candidates
                .iter()
                .map(|candidate| {
                    format!(
                        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr><tr><td></td><td colspan=\"4\" class=\"muted\">{}</td></tr>",
                        esc(&candidate.official_label),
                        esc(&candidate.status),
                        esc(candidate.recommended_isq.as_deref().unwrap_or("n/a")),
                        esc(candidate.device_layers_cli.as_deref().unwrap_or("n/a")),
                        esc(
                            &candidate
                                .max_context_tokens
                                .map(|value| value.to_string())
                                .unwrap_or_else(|| "n/a".to_string())
                        ),
                        esc(candidate.note.as_deref().unwrap_or("")),
                    )
                })
                .collect::<String>()
        })
        .unwrap_or_else(|| {
            r#"<tr><td colspan="5" class="muted">Noch keine mistralrs-tune-Ergebnisse vorhanden. Fuehre zuerst den Census aus.</td></tr>"#
                .to_string()
        });
    layout(
        "Models",
        &format!(
            r#"{banner}
<section class="card hero">
  <h1>Modelle</h1>
  <p>Diese Seite haelt die kanonische Kleinhirn- und Grosshirn-Modellpolitik des CTO-Agenten fest.</p>
</section>
<section class="grid">
  <div class="card">
    <h2>Kleinhirn</h2>
    <table>
      <tr><th>Basis</th><td>{label}</td></tr>
      <tr><th>Basis-ID</th><td>{model_id}</td></tr>
      <tr><th>Empfohlen jetzt</th><td>{selected_label}</td></tr>
      <tr><th>Browser/Vision</th><td>{browser_vision_selected}</td></tr>
      <tr><th>Upgrade erlaubt</th><td>{upgrade_allowed}</td></tr>
      <tr><th>Unabhaengig von Grosshirn</th><td>{independent}</td></tr>
      <tr><th>Provider</th><td>{provider}</td></tr>
      <tr><th>Reasoning</th><td>{reasoning}</td></tr>
      <tr><th>Deployment</th><td>{deployment}</td></tr>
      <tr><th>Lokale Threads</th><td>{cpu_threads}</td></tr>
      <tr><th>Lokaler RAM</th><td>{memory_gb}</td></tr>
      <tr><th>GPUs</th><td>{gpu_count}</td></tr>
      <tr><th>Gesamt-VRAM</th><td>{total_gpu_memory_gb}</td></tr>
      <tr><th>Groesste GPU</th><td>{max_single_gpu_memory_gb}</td></tr>
    </table>
    <p class="muted">{purpose}</p>
  </div>
  <div class="card">
    <h2>Kleinhirn-Upgrades</h2>
    <pre>{upgrades}</pre>
  </div>
  <div class="card">
    <h2>Grosshirn-Kandidaten</h2>
    <pre>{grosshirn}</pre>
  </div>
  <div class="card">
    <h2>GPU-Inventar</h2>
    <pre>{gpu_inventory}</pre>
  </div>
  <div class="card">
    <h2>mistralrs tune</h2>
    <table>
      <tr><th>Modell</th><th>Status</th><th>ISQ</th><th>Device-Layers</th><th>Max Context</th></tr>
      {tune_rows}
    </table>
  </div>
  <div class="card">
    <h2>Model Policy</h2>
    <pre>{pretty}</pre>
  </div>
</section>"#,
            banner = banner(message),
            label = esc(&model_policy.kleinhirn.official_label),
            model_id = esc(&model_policy.kleinhirn.model_id),
            selected_label = esc(&format!(
                "{} ({})",
                selected.official_label, selected.model_id
            )),
            browser_vision_selected = esc(&browser_vision_selected),
            upgrade_allowed = model_policy.kleinhirn_upgrade_allowed,
            independent = model_policy.kleinhirn_upgrade_independent_from_grosshirn,
            provider = esc(&model_policy.kleinhirn.provider),
            reasoning = esc(&model_policy.kleinhirn.reasoning_effort),
            deployment = esc(&model_policy.kleinhirn.deployment_mode),
            cpu_threads = esc(
                &census
                    .cpu_threads
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "unbekannt".to_string())
            ),
            memory_gb = esc(
                &census
                    .total_memory_gb
                    .map(|value| format!("{value} GiB"))
                    .unwrap_or_else(|| "unbekannt".to_string())
            ),
            gpu_count = esc(
                &census
                    .gpu_count
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "unbekannt".to_string())
            ),
            total_gpu_memory_gb = esc(
                &census
                    .total_gpu_memory_gb
                    .map(|value| format!("{value} GiB"))
                    .unwrap_or_else(|| "unbekannt".to_string())
            ),
            max_single_gpu_memory_gb = esc(
                &census
                    .max_single_gpu_memory_gb
                    .map(|value| format!("{value} GiB"))
                    .unwrap_or_else(|| "unbekannt".to_string())
            ),
            purpose = esc(&model_policy.kleinhirn.purpose),
            gpu_inventory = esc(&gpu_inventory),
            tune_rows = tune_rows,
            upgrades = esc(
                &serde_json::to_string_pretty(&model_policy.kleinhirn_upgrade_candidates)
                    .unwrap_or_else(|_| "[]".to_string())
            ),
            grosshirn = esc(
                &serde_json::to_string_pretty(&model_policy.grosshirn_candidates)
                    .unwrap_or_else(|_| "[]".to_string())
            ),
            pretty = esc(&pretty),
        ),
    )
}

pub fn browser_page(
    message: Option<&str>,
    policy: &BrowserEnginePolicy,
    subworker_policy: &BrowserSubworkerPolicy,
    state: &BrowserEngineState,
    bridge_state: &BrowserAgentBridgeState,
    census: &SystemCensus,
    worker_jobs: &[WorkerJobRecord],
) -> String {
    let policy_pretty = serde_json::to_string_pretty(policy).unwrap_or_else(|_| "{}".to_string());
    let subworker_policy_pretty =
        serde_json::to_string_pretty(subworker_policy).unwrap_or_else(|_| "{}".to_string());
    let state_pretty = serde_json::to_string_pretty(state).unwrap_or_else(|_| "{}".to_string());
    let bridge_pretty =
        serde_json::to_string_pretty(bridge_state).unwrap_or_else(|_| "{}".to_string());
    let worker_rows = if worker_jobs.is_empty() {
        r#"<tr><td colspan="6" class="muted">Noch keine Browser-/Repair-Worker-Jobs vorhanden.</td></tr>"#
            .to_string()
    } else {
        worker_jobs
            .iter()
            .map(|job| {
                format!(
                    "<tr><td>#{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td class=\"muted\">{}</td></tr>",
                    job.id,
                    esc(&job.worker_kind),
                    esc(&job.status),
                    esc(&job.contract_title),
                    esc(job.completed_at.as_deref().unwrap_or("-")),
                    esc(job.result_summary.as_deref().unwrap_or(job.request_note.as_str())),
                )
            })
            .collect::<String>()
    };
    layout(
        "Browser Engine",
        &format!(
            r#"{banner}
<section class="card hero">
  <h1>Browser Engine</h1>
  <p>Diese Seite beschreibt die explizite zweite Haupt-Engine neben der CLI-Engine und die ersten Browser-Subworker darunter.</p>
</section>
<section class="grid">
  <div class="card">
    <h2>Status</h2>
    <table>
      <tr><th>Status</th><td>{status}</td></tr>
      <tr><th>Chrome Binary</th><td>{chrome_binary}</td></tr>
      <tr><th>Chrome Version</th><td>{chrome_version}</td></tr>
      <tr><th>Desktop verfuegbar</th><td>{desktop_available}</td></tr>
      <tr><th>Headless bereit</th><td>{headless_ready}</td></tr>
      <tr><th>Interaktiv bereit</th><td>{interactive_ready}</td></tr>
      <tr><th>Artifacts</th><td>{artifacts_dir}</td></tr>
      <tr><th>Install Script</th><td>{install_script}</td></tr>
    </table>
    <p class="muted">Census-Sicht: {census_status}</p>
  </div>
  <div class="card">
    <h2>Tool Surface</h2>
    <pre>{actions}</pre>
  </div>
  <div class="card">
    <h2>Runtime Policy</h2>
    <pre>{policy_pretty}</pre>
  </div>
  <div class="card">
    <h2>Extension Bridge</h2>
    <table>
      <tr><th>Base URL</th><td>{bridge_base_url}</td></tr>
      <tr><th>Bridge Port</th><td>{bridge_port}</td></tr>
      <tr><th>Workspace</th><td>{extension_workspace}</td></tr>
      <tr><th>Manifest</th><td>{manifest_path}</td></tr>
      <tr><th>Queued Jobs</th><td>{queued_jobs}</td></tr>
      <tr><th>Leased Jobs</th><td>{leased_jobs}</td></tr>
      <tr><th>Workers</th><td>{active_workers}</td></tr>
    </table>
    <p class="muted">Der Browser-Agent lebt als entkoppelte Chrome-Extension und pollt diese Bridge selbststaendig.</p>
  </div>
  <div class="card">
    <h2>Subworker Policy</h2>
    <p class="muted">Browser-Agent fuer echte Browserarbeit, CTO-Repair-Pfad fuer Codeprobleme, Specialist-Fabrik fuer wiederkehrende Flows.</p>
    <pre>{subworker_policy_pretty}</pre>
  </div>
  <div class="card">
    <h2>Runtime State</h2>
    <pre>{state_pretty}</pre>
  </div>
  <div class="card">
    <h2>Bridge State</h2>
    <pre>{bridge_pretty}</pre>
  </div>
</section>
<section class="card">
  <h2>Recent Worker Jobs</h2>
  <table>
    <tr><th>ID</th><th>Kind</th><th>Status</th><th>Vertrag</th><th>Completed</th><th>Summary</th></tr>
    {worker_rows}
  </table>
</section>"#,
            banner = banner(message),
            status = esc(&state.status),
            chrome_binary = esc(state.chrome_binary.as_deref().unwrap_or("none")),
            chrome_version = esc(state.chrome_version.as_deref().unwrap_or("unknown")),
            desktop_available = state.desktop_available,
            headless_ready = state.headless_ready,
            interactive_ready = state.interactive_ready,
            artifacts_dir = esc(&state.artifacts_dir),
            install_script = esc(&state.install_script),
            census_status = esc(census.browser_engine_status.as_deref().unwrap_or("unknown")),
            bridge_base_url = esc(&bridge_state.base_url),
            bridge_port = bridge_state.bridge_port,
            extension_workspace = esc(&bridge_state.extension_workspace),
            manifest_path = esc(&bridge_state.manifest_path),
            queued_jobs = bridge_state.queued_jobs,
            leased_jobs = bridge_state.leased_jobs,
            active_workers = bridge_state.active_workers.len(),
            actions = esc(
                &serde_json::to_string_pretty(&policy.tool_surface.directive_actions)
                    .unwrap_or_else(|_| "[]".to_string())
            ),
            policy_pretty = esc(&policy_pretty),
            subworker_policy_pretty = esc(&subworker_policy_pretty),
            state_pretty = esc(&state_pretty),
            bridge_pretty = esc(&bridge_pretty),
            worker_rows = worker_rows,
        ),
    )
}
