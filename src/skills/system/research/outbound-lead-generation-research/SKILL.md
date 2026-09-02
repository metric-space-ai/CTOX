---
name: outbound-lead-generation-research
description: Research a company and its contacts for the Business OS "Outbound Lead Generation" app with the CTOX web stack — browser, search, source adapters and the scraping pipeline — and return the result to the lead through the command bus. Trigger on "Starte eine Outbound Nachrecherche für <Firma> [<lead-id>] (Auftrag <command-id>)" or "Starte eine Outbound Neurecherche für …".
cluster: research
---

# Outbound Lead Generation · Recherche über den CTOX Web-Stack

## CTOX Runtime Contract

- Task spawning is allowed only for real bounded work steps that add mission progress, external waiting, recovery, or explicit decomposition. Do not spawn work merely because review feedback exists.
- The Review Gate is a quality checkpoint, not a control loop. After review feedback, continue the same main work item whenever possible and incorporate the feedback there.
- Everything you do goes through the `ctox` CLI. There is no other data path: the CLI runs inside the daemon and writes to the CTOX SQLite stores; the Business OS UI receives results through replication of the collection the command bus writes.

## 1. Wie die App, der Harness und der Web-Stack zusammenspielen

- The app button only formulates the assignment (one sentence) and puts the data you need into the command payload. It does not research anything.
- You are the research. You get this skill plus the assignment, read the payload, use the web stack, and send exactly one writeback command per lead at the end.
- The web stack has two working modes, use both:
  - **Browser** — open, read, click, submit, keep a persistent session per human owner. When a site needs a login or blocks you, the browser streams live to the human (auth assist); the human signs in or solves the challenge, and you continue with the same session.
  - **Scraping** — for sources you will need again, write an extraction script once, register it as a revision on a scrape target in SQLite, run it through the pipeline, and query the records. Drift is repaired, not re-researched.

## 2. Read your assignment

```bash
ctox business-os commands inspect <command-id>
```

The command id stands in the assignment sentence. The payload carries:

| key | meaning |
| --- | --- |
| `lead_id`, `company`, `country` | the lead (record id in `outbound_lead_generation_leads`), company name, `DE`/`AT`/`CH` |
| `mode` | `new_record` (first research) or `update_firm` (Nachrecherche) |
| `fields` | the requested field keys (default: all 32) |
| `lead_snapshot` | the lead's current `data.*` values and `contacts[]` — what is already known |
| `known_person_records` | Sellify persons (name, function, e-mail) — keep, complete, never re-guess |
| `research_instructions` | the owner's "Rechercheablauf" — overrides the source order in §5, never the evidence or writeback rules |
| `person_priorities` | contact categories in the required order |
| `include_private` | login sources the owner allows (`linkedin.com`, `xing.com`, `dnbhoovers.com`, `leadfeeder.com`, `rocketreach.com`) |
| `writeback_contract` | `record_ids`, `min_independent_sources` (2) |

## 3. The 32 fields

Company (21): `firma_name`, `firma_fruehere_namen`, `firma_aktivitaetsstatus`, `firma_anschrift`, `firma_besucheranschrift`, `firma_postanschrift`, `firma_postfach`, `firma_plz`, `firma_ort`, `firma_land`, `firma_email`, `firma_domain`, `firma_telefon`, `firma_fax`, `firma_geschaeftstaetigkeit`, `firma_homepage_fact_sheet`, `firma_geschaeftsfuehrung`, `firma_prokura`, `wz_code`, `umsatz`, `mitarbeiter`.

Person (11, per contact, keyed by a stable `person_key`): `person_geschlecht`, `person_titel`, `person_vorname`, `person_nachname`, `person_funktion`, `person_position`, `person_email`, `person_email_validation`, `person_telefon`, `person_linkedin`, `person_xing`.

Contacts: at least one per category, in this order — Geschäftsführung/Gesamtverantwortung, Prokura, Leitung Finanzen, Einkauf, Supply Chain Management, Operations, Technik, Entwicklung. Sellify contacts are kept under their `person_key` and completed.

## 4. The CLI you work with

### Search, read, sources, adapter batch

```bash
ctox web sources list [--country <DE|AT|CH>] [--tier <P|S|C>]... [--field <field-key>]
ctox web sources info --id <source-id>
ctox web search --query <text> [--domain <host>]... [--source <id>]... [--country <DE|AT|CH>] [--context-size <low|medium|high>] [--cached] [--include-sources]
ctox web read --url <url> [--query <text>] [--find <text>]... [--workspace <path>] [--country <DE|AT|CH>]
ctox web person-research --company <name> --country <DE|AT|CH> --mode <new_record|update_firm|update_person|update_inventory_general|have_data> [--field <field-key>]... [--include-private <source-id>]... [--workspace <path>] [--no-workspace]
```

`person-research` is the adapter batch: it plans sources per (mode, country, field), runs search+read per source and returns an envelope `fields{field:{value, confidence, source_id, source_url, candidates}}`, `plan`, `search_runs`, `read_runs`, `scrape_runs`. Run it once at the start with the requested fields; take every `high`/`medium` value with its source; treat the rest as open. It is a helper, not the research.

Search engines rate-limit. When a search answers "rate limit" or "low relevance", switch the engine (`--source html.duckduckgo.com`, `--source bing`, `--domain <host>` pinning) and go on; never close a field because one engine was tired.

### Browser

```bash
ctox web browser-capture --url <url> [--dir <path>] [--out-dir <path>] [--timeout-ms <n>]
ctox web browser-automation [--dir <path>] [--timeout-ms <n>] [--script-file <path>] < script.js
ctox web unlock <list-probes|list-vectors|baseline|history|add-vector|set-vector-status> [...]
```

`browser-capture` renders a page in the real browser (JavaScript sites, screenshots, DOM text). `browser-automation` runs a Playwright script you write (navigate, click, fill, extract) in the owner's persistent browser. `unlock` is the stealth registry when bot detection blocks you — see the `web-unlock` skill before touching it.

### Login sources and the human in the loop

```bash
ctox business-os web-stack auth-assist-request --source-id <id> [--target-url <url>] [--credential-ref <ctox-secret://scope/name>] [--login-hint <hint>] [--task-id <id>]
ctox business-os web-stack auth-assist-status --session-id <id>
ctox business-os web-stack context-capture --session-id <id> [--source-id <id>] [--task-id <id>] [--no-handoff]
ctox business-os web-stack context-extract --session-id <id> [--source-id <id>] [--capture-script <id>] [--task-id <id>]
ctox business-os web-stack source-capture --source-id <dnbhoovers.com|leadfeeder.com|rocketreach.com|xing.com> --company <name> [--country <DE|AT|CH>] [--session-id <id>] [--credential-ref <ctox-secret://scope/name>] [--timeout-ms <n>]
ctox business-os web-stack authenticated-automation --source-id <id> --target-url <url> --credential-ref <ctox-secret://scope/name> [--login-hint <hint>] [--task-id <id>] [--timeout-ms <n>]
```

Unblocking with continuation, in this order:

1. `auth-assist-request --source-id <id> --task-id <your command id>` — opens the owner's streamed browser on that source and returns the browser `session_id`; the human signs in or solves the challenge in the stream.
2. `auth-assist-status --session-id <id>` — poll until the session reports authenticated; do not proceed on a pending session.
3. Continue **in the same session**: `ctox web browser-automation --session-id <id> --script-file <path>` for your own navigation and extraction, `source-capture --source-id <id> --session-id <id> --company <name>` for the built-in extractors of dnbhoovers.com, leadfeeder.com, rocketreach.com and xing.com, `context-capture --session-id <id>` / `context-extract --session-id <id>` for a page the human positioned for you.

Never type credentials yourself; never guess what a login source would have said. If the human does not complete the login within the turn, the field ends `action_required` with the `session_id` and your command id as reference.

### Scraping pipeline (scripts and records live in SQLite)

```bash
ctox scrape list-targets
ctox scrape show-target --target-key <key>
ctox scrape show-api --target-key <key>
ctox scrape show-latest --target-key <key> [--limit <n>]
ctox scrape query-records --target-key <key> [--where field=value]... [--limit <n>]
ctox scrape semantic-search --target-key <key> --query <text> [--limit <n>]
ctox scrape upsert-target --input <json-path>
ctox scrape register-script --target-key <key> --script-file <path> [--language <lang>] [--change-reason <text>] [--notes <text>]
ctox scrape register-source-module --target-key <key> --source-key <key> --module-file <path> [--language <lang>] [--change-reason <text>] [--notes <text>]
ctox scrape execute --target-key <key> [--trigger-kind <manual|scheduled|repair>] [--timeout-seconds <n>] [--allow-heal] [--thread-key <key>] [--queue-priority <urgent|high|normal|low>]
ctox scrape record-template-example --target-key <key> --template-key <template> --script-file <path> [--language <lang>] [--result-count <n>] [--challenge-score <n>] [--reason <text>]
ctox scrape promote-template --template-key <template> --script-file <path> [--language <lang>] --reason <text>
ctox web scrape --target-key <key> --mode <latest|semantic> [--query <text>] [--limit <n>]
```

Where things are: `ctox.sqlite3` holds `scrape_target` (key, start URL, `target_kind`, config, output schema), `scrape_script_revision` (revision number, script body, sha256, change reason), `scrape_source_revision` (per-source extractor modules), `scrape_run` (status, classification, timing), `scrape_record_latest` (the extracted records). Working files (inputs, outputs, artifacts) live under `~/.local/state/ctox/scraping/targets/<target-key>/`. Registered targets today: `handelsregister-de`, `northdata-de`, `bundesanzeiger-de`, `companyhouse-de` (`target_kind = prospect-research`).

When to write a script: a source you will hit again for many leads (register lists, company directories) or one whose page needs structured extraction. Look at `show-api`/`show-target` first; if the target exists, `execute --allow-heal`; if the run classifies `portal_drift`, the repair task is already queued — record it and move on, do not retry the same source in this run. If no target exists and the source will recur, write the script (`universal-scraping` skill explains authoring, fixtures and `upsert-target`), register it, run it. For a one-off page, just read or capture it.

## 5. Source order (DE default; `research_instructions` overrides the order)

1. Identity and register: Handelsregister, Northdata, Bundesanzeiger, CompanyHouse (AT: Firmenbuch/JustizOnline, FirmenABC; CH: Zefix, SHAB, Moneyhouse). Fields: `firma_name`, `firma_fruehere_namen`, `firma_aktivitaetsstatus`, `firma_geschaeftsfuehrung`, `firma_prokura`.
2. Website, address, communication: find `firma_domain` first, then the Impressum for `firma_anschrift`, `firma_besucheranschrift`, `firma_postanschrift`, `firma_postfach`, `firma_plz`, `firma_ort`, `firma_land`, `firma_email`, `firma_telefon`, `firma_fax`. Fallback FirmenABC (AT), Zefix (CH), D&B Hoovers (DE/CH), Northdata last.
3. Figures: `wz_code`, `umsatz`, `mitarbeiter` from D&B Hoovers or Leadfeeder (DE also Bundesanzeiger); `firma_geschaeftstaetigkeit` and `firma_homepage_fact_sheet` from the homepage.
4. Persons: register and Impressum first, homepage team pages, then LinkedIn/XING **by name search** (never from clicked search hits), RocketReach; derive the e-mail pattern from known addresses and validate it (MailTester) → `person_email_validation`.

## 6. Evidence rules

- A field is `verified` only with a value and **two independent sources on different hosts**; each source has `source_id`, `url` and a verbatim `quote`. Two pages of one host are one source. Sellify alone proves nothing, but counts as one source.
- `no_match` only after at least one documented search and two documented page reads for that field.
- `action_required` only for a login or approval you could not get: reference the auth-assist (`source_id` and your command id) or a source with `requires_credential=true`. Also for conflicts: keep both candidates with their sources, leave the value empty, reason `conflict`.
- `unsupported` only when the field cannot be researched under this contract (e.g. AT-only field on a DE lead).
- A blocked or temporarily unreachable source proves nothing, neither the value nor its absence.
- Keep a running checkpoint in your workspace: `gap_closure/field_status.json` after every attempt and one file per attempt under `gap_closure/attempts/<field>/<n>.json` (`kind`, `query_or_url`, `result`, `artifact_path`, `at`). A follow-up turn resumes from it.

## 7. Writeback — the only way results reach the lead

One command per lead, dispatched through the command bus; the native handler validates it, writes the lead collection, and the UI updates through replication. Never edit collections or SQLite directly, never report results as chat text only.

```bash
ctox business-os commands dispatch --json '{
  "id": "research-writeback-<lead-id>-<n>",
  "command_id": "research-writeback-<lead-id>-<n>",
  "module": "outbound-lead-generation",
  "command_type": "outbound.lead.research_writeback",
  "record_id": "<lead-id>",
  "status": "pending_sync",
  "payload": {
    "record_id": "<lead-id>",
    "module": "outbound-lead-generation",
    "research_command_id": "<your command id>",
    "gap_task_id": "",
    "field_status": {
      "<field>": {
        "status": "verified|no_match|unsupported|action_required",
        "value": "...",
        "reason": "...",
        "sources": [{"source_id": "...", "url": "...", "quote": "...", "person_key": null, "requires_credential": false, "task_id": "", "command_id": ""}],
        "attempts": [{"kind": "web_search|web_read|browser_capture|scrape", "query_or_url": "...", "result": "...", "artifact_path": "...", "at": "<iso>"}]
      }
    },
    "result": {
      "fields": { "<field>": {"value": "...", "sources": [ ... ]} },
      "person_records": [ {"person_key": "...", "person_vorname": "...", "person_nachname": "...", "person_funktion": "...", "person_position": "...", "person_email": "...", "sources": [ ... ]} ],
      "evidence": [ {"field_key": "<field>", "source_id": "...", "url": "...", "quote": "...", "person_key": null} ]
    }
  }
}'
```

- `field_status` covers **every requested field** with a terminal status; a missing field means the work is not finished.
- Person fields carry a `person_key`; `result.fields` holds structured objects only, never free text.
- `research_command_id` is your own command id; `gap_task_id` stays empty for a chat assignment (it is only set when the daemon handed you a queue task "Lückenschluss: …" — then copy it from that task).
- Done means the dispatch answered `ok: true` with status `accepted` or `completed`. Report the counts (verified / no_match / action_required / unsupported) and the persons found in one short chat message.

## 8. Stop conditions

- One turn is bounded; keep `field_status.json` current so a follow-up continues instead of restarting.
- Never fabricate a value, a person, an e-mail address or a source. An empty verified field is better than a plausible one.
- If the CLI itself fails (sandbox, relay, daemon), report the exact command and error; do not work around it with curl, raw HTTP or a private browser.
