# I-051 — Runde 1 (nur Messung)

## was_geaendert

- Keine Repo-Datei geaendert, nichts committed.
- Nur dieser Messbericht wurde unter `/tmp/i-051-report.md` geschrieben.
- Der Checkout war bereits stark veraendert und wurde waehrend der Messung durch einen anderen Prozess weiter veraendert (u. a. `src/core/service/service.rs` und `src/core/mission/channels/mod.rs`). Alle Zeilenangaben unten beziehen sich auf den am Ende der Messung sichtbaren Arbeitsbaum.

## ursache_belegt

### 1. Wer setzt `leased`, und wie wird normal freigegeben?

- Der kanonische Schreiber ist `channels::lease_queue_task` in `src/core/mission/channels/mod.rs:3374-3445`. Er CAS-t nur `pending -> leased` (`:3411-3437`) und schreibt atomar `lease_owner`, `leased_at`, `lease_expires_at` und anfangs `lease_worker_id=NULL` (`:3412-3422`).
- Der Service waehlt einen pending Task und ruft diesen Schreiber in `src/core/service/service.rs:18312-18353`, konkret `:18336-18337`; der Task wird danach als `QueuedPrompt` an den Worker uebergeben (`:18339-18353`).
- Beim wirklichen Workerstart wird eine pro Slice eindeutige Worker-ID gestempelt und ein Heartbeat gestartet: `PromptWorkerActivity::start`, `src/core/service/service.rs:5416-5485` (Heartbeat erneuert alle 60 s, `:5430-5452`; Worker-ID, `:5454-5474`).
- Normaler Abschluss geht durch `ack_leased_messages*`/`hold_leased_messages`; `ack_messages_in_transaction` wechselt den Route-Status weg von `leased` und leert `lease_owner`, `leased_at` und `lease_worker_id` in `src/core/mission/channels/mod.rs:5426-5513`, besonders `:5485-5501`. Holds leeren Ablauf/Worker-ID in `:2475-2623` (heutige Datei, gemessener Bereich).
- Im Worker werden die normalen terminalen/retry/rework Acks in `src/core/service/service.rs:7302-7440` und die Fehler-Acks in `:7788-7957` ausgefuehrt.

### TTL und owner-agnostischer Reclaim: heute JA

Die Behauptung aus `docs/ctox-harness-review-2026-07-10.md` ist fuer den heutigen Code falsch:

- TTL: 15 Minuten beim Lease (`src/core/mission/channels/mod.rs:3394-3395`) und bei jeder Erneuerung (`:3657-3679`).
- Migration alter Leases: fehlendes `lease_expires_at` wird fuer bereits geleaste Zeilen auf `leased_at + 15 minutes` gesetzt (`src/core/mission/channels/mod.rs:4688-4737`).
- Owner-agnostisch: `release_stale_queue_task_leases` ignoriert den uebergebenen Owner (`_lease_owner`) und selektiert jede `leased`-Zeile mit fehlendem Owner/Zeitstempel/Ablauf oder abgelaufenem TTL (`src/core/mission/channels/mod.rs:3494-3528`). Die Schreibseite prueft die komplette alte Lease-Identitaet per CAS und leert alle Lease-Felder (`:3555-3644`).
- Laufender Sweep: alle 60 s, unabhaengig von Router-Idle-Gates, `src/core/service/service.rs:16679-16684` und `:16788-16882`.
- Boot-Reclaim: `src/core/service/service.rs:1869-1934` ruft denselben typed Reclaim mit leerer Active-Menge auf.
- Terminale Business-Command/Queue-Drift ist ebenfalls repariert: Wenn der Command bereits terminal ist, aber die Route noch `leased/running`, setzt `src/core/mission/channels/command_saga.rs:1106-1147` die Queue auf `handled/cancelled/failed`. Der Kommentar `:1111-1116` dokumentiert den frueheren Defekt: der alte Pfad kehrte zurueck, ohne die Route zu bereinigen, sodass Recovery scheinbar erfolgreich war, die Zeile aber dauerhaft `leased` blieb.

### 2. Wann kann eine Lease heute stehen bleiben?

#### Prozess-Crash: ja, aber begrenzt

- Bei Prozessabbruch wird weder Worker-`Drop` noch ein Ack garantiert erreicht. Der Heartbeat aus `src/core/service/service.rs:5430-5452` stoppt mit dem Prozess; die zuletzt erneuerte 15-Minuten-Lease bleibt bis Boot-Reclaim bzw. TTL-Sweep bestehen.
- Nach Neustart laeuft Boot-Recovery (`src/core/service/service.rs:1869-1934`); abgelaufene/unvollstaendige Leases werden owner-agnostisch freigegeben. Fuer App-Tasks laeuft davor weiterhin die spezielle App-Recovery (`:1873-1886`).

#### Geordneter Stop: normaler App-Stop nein; Force/direct ja

- Der normale bewachte Stop verweigert das Stoppen, wenn ein App-Create/Modify-Task geleast ist: `src/core/service/service.rs:2441-2457`.
- Der direkte IPC-Stop beendet den Prozess jedoch nach 50 ms mit `std::process::exit(0)` und wartet nicht auf Worker/Ack: `src/core/service/service.rs:3303-3318`. Gleiches gilt fuer den HTTP-Stop `:3600-3611`. `--force` umgeht den Guard ueber `stop_background_guarded` (`:2453-2457`). In diesen Pfaden bleibt eine aktive Lease wie beim Crash bis Boot/TTL stehen.
- Fuer nicht als App erkannten Queue-Task gibt es im allgemeinen Stop-Pfad keinen entsprechenden Drain-Guard.

#### Validierungs-Abbruch: semantisch nein; Persistenzfehler beim Abschluss ja

- Ein rotes App-Validation-Ergebnis wird zu `review_rework` oder nach Budgetende zu `failed`; der Worker acked diese Status in `src/core/service/service.rs:6853-7038` und `:7364-7440`. Nach Workerfehler gilt dasselbe in `:7676-7957`.
- Ein Validator-Fehler (`Err`) wird ebenfalls in Rework/Failure ueberfuehrt (`:6988-7037`); er springt nicht einfach aus dem Worker heraus.
- Stehen bleibt die Lease, wenn der anschliessende durable Ack selbst fehlschlaegt: Der Service entfernt die In-Memory-Ownership bereits vor den Ack-Schreibversuchen (`src/core/service/service.rs:6770-6784`). `record_queue_ack_and_refresh_business_os_projections_locked` protokolliert einen Ack-Fehler danach nur im 24 Eintraege grossen In-Memory-Feed und kehrt zurueck (`:24644-24696`). Damit existiert kein aktiver Owner mehr, die DB-Zeile kann aber weiter `leased` sein; erst TTL/Sweep repariert sie.
- Derselbe Defekt existiert beim Lease-Handoff: Wenn zwischen Lease und Enqueue Working-Hours/Busy greift, versucht `enqueue_prompt` die Lease auf `pending` zurueckzusetzen. Schlaegt dieser Ack fehl, wird nur ein Event geschrieben und der Prompt nicht gestartet (`src/core/service/service.rs:19431-19563`, besonders `:19553-19561`). Das ist ein konkreter heutiger Schreibpfad, der eine ownerlose frische Lease erzeugen kann.
- `PromptWorkerActivity::Drop` hat zusaetzliche Cleanup-/Fail-Ack-Netze (`src/core/service/service.rs:5504-5660`), aber auch dort werden Fehler letztlich nur gemeldet; ein Prozessabbruch umgeht `Drop` komplett.

#### Neustart des Daemons: nicht Ursache, sondern Reclaim

- Ein Neustart erzeugt keine neue Lease. Er findet die vom vorherigen Prozess hinterlassene Lease und fuehrt App-Recovery plus generischen Lease-Reclaim aus (`src/core/service/service.rs:1869-1934`).
- Eine frische, noch nicht abgelaufene App-Lease wird durch die spezielle `recover_abandoned_*`-Logik ohne Altersgrenze verarbeitet (`src/core/service/service.rs:18579-18629`). Eine frische Nicht-App-Lease wird am Boot zusaetzlich durch den alten service-owner Boot-Pfad freigegeben, sofern sie nicht fuer die spezielle App-Recovery reserviert wird (`src/core/service/service.rs:1983-2061`).

### 3. Persistenzmessung

#### `runtime/ctox.sqlite3`, `ctox_harness_flow_events`

- Gesamtzeilen bei letzter Messung: **1.859**.
- Automatische Business-OS-App-Recovery-Ereignisse (`Business OS app` plus `recover|abandon|stale` ueber `event_kind/title/body_text/metadata_json`): **0**.
- Das ist eine Instrumentierungsluecke, nicht der Beweis, dass der Pfad nie lief: Die Recovery ruft nur `push_event`; `push_event_locked` haelt maximal 24 Eintraege im Speicher (`src/core/service/service.rs:24638-24648`) und schreibt nicht in `ctox_harness_flow_events`.
- Sieben `queue.cleanup_scope`-Events enthalten stale/recovery/aborted-App-Creator-Begruendungen und Selektoren, die `leased` einschlossen. Ihre `matched_count` summiert sich auf **28**, aber das Event speichert keine Aufteilung nach Route-Status; daraus kann die Zahl wirklich geleaster Rows nicht seriös abgeleitet werden. Diese sieben Events stammen vom 18.–22.06.2026.

#### Snapshots unter `/Volumes/tmp/ctox-state-backups/`

Drei grosse konsolidierte Snapshots enthalten die relevante Tabelle:

- `update-20260718T073042Z/ctox.sqlite3`: **1.521** Flow-Events, **0** automatische App-Recovery-Events, **0** aktuell geleaste Rows.
- `update-20260718T101507Z/ctox.sqlite3`: **1.535** Flow-Events, **0** automatische App-Recovery-Events, **1** geleaste Row.
- `update-20260718T115230Z/ctox.sqlite3`: **1.537** Flow-Events, **0** automatische App-Recovery-Events, **0** geleaste Rows.

Die eine Lease um 10:15 war keine verlassene App-Lease: `queue:system::89adaf070df3b415054c84dd` war seit 10:12:52Z geleast (also rund zwei Minuten), hatte laufende Work-Outcome-Ereignisse und war im 11:52-Snapshot terminal `failed` (Update 11:09:11Z).

#### Durable Nebenbelege

- In `communication_messages.metadata_json.status_note` gibt es in Live-DB und allen drei Snapshots genau **1** exakten Marker `...during idle recovery...`: Task `queue:system::23c1c6b93803d36884a8748e`, Update `2026-06-23T20:21:26Z`. Damit ist die App-Recovery historisch **mindestens einmal real gelaufen**. Der Marker liegt vor TTL (eingefuehrt 10.07.) und dem vollstaendigen orphaned-lease Fix (23.07.).
- `governance_events`: **38** alte `boot_lease_reclaim`-Events mit Summe **40** freigegebener Service-Leases in der Live-DB; der letzte liegt am 18.07.2026. Diese alte Boot-Recovery reserviert App-Leases fuer die Speziallogik und ist deshalb kein App-spezifischer Zaehler.
- Fuer die seit 23.07. vorhandenen Mechanismen `boot_queue_lease_reclaim` und `orphaned_queue_lease_sweep`: jeweils **0** Events in Live-DB und Snapshots.
- Aktueller Live-Zustand: **0** Rows mit `route_status='leased'`.

### Schlussfolgerung

- Historisch war die App-Recovery nicht komplett tot: mindestens **1** exakter Idle-Recovery-Seiteneffekt und sieben manuelle stale/aborted Cleanup-Ereignisse sind persistent.
- Der Defekt, der Leases unbegrenzt liegen liess, ist im heutigen Code bereits an der Queue-/Command-Grenze repariert (`command_saga.rs:1106-1147`) und durch 15-Minuten-TTL, 60-s Heartbeat, owner-agnostischen Reclaim und 60-s Sweep abgesichert.
- Seit Einfuehrung des vollstaendigen generischen Reclaims gibt es **0** persistierte automatische Reclaim-Ereignisse und aktuell **0** Leases. Damit ist das grosse App-spezifische Recovery-Netz fuer die Lease-Liveness heute nicht mehr belegt.
- Verbleibende Ursache fuer kurzzeitig verwaiste Leases sind nicht App-Artefakte, sondern nicht-atomare Handoffs/fehlgeschlagene Acks sowie Force/direct Stop. Diese Leases sind heute durch TTL/Sweep begrenzt, aber sie koennen weiterhin bis zu 15 Minuten als `leased` sichtbar sein.

## kompensationen_geloescht

- Keine; reine Messung.

## verblieben

- Vollstaendig verblieben: `start_business_os_app_recovery_loop`, Recovery-Stempel/Idle-Gates/Preflight und beide `recover_*`-Einstiege in `src/core/service/service.rs`.
- Der generische TTL/Heartbeat/owner-agnostische Reclaim ist **keine** App-Kompensation, sondern die heute notwendige Queue-Lease-Semantik und muss bleiben.
- Ebenfalls verblieben und ursachennah: swallowed Ack-/Handoff-Fehler (`src/core/service/service.rs:19553-19561`, `:24656-24696`) und Force/direct `process::exit` ohne Worker-Drain (`:3303-3318`, `:3600-3611`).

## tests

- `CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline-targets/I-051 cargo fmt --check`
  - **FAIL** wegen 3 Format-Diffs in bereits parallel veraendertem `src/core/service/service.rs` (um Zeilen 6146, 10697, 10973).
  - Keine `test result`-Zeile: `cargo fmt --check` ist kein Test-Runner.
- `CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline-targets/I-051 cargo check --bin ctox`
  - **PASS**, `Finished dev profile`, **404 warnings**.
  - Keine `test result`-Zeile: `cargo check` ist kein Test-Runner.
- Zieltest (erlaubter enger Filter): `CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline-targets/I-051 cargo test --bin ctox stale_queue_task_lease_recovery_does_not_clobber_concurrent_renewal -- --nocapture`
  - Nach mehr als 20 Minuten noch im Compile unter hoher paralleler Last; von mir beendet, Exit 144. **0 Tests ausgefuehrt**, daher keine `test result`-Zeile. Der Filter war spezifisch und nicht einer der verbotenen Filter.
- Keine weiteren Tests gestartet, um die bereits stark ausgelastete Maschine nicht weiter zu belasten.

## gegenprobe

- Entfaellt laut Auftrag, da kein Diff erzeugt wurde.
- Keine temporaere Repo-Aenderung war zurueckzubauen.
- Abschliessendes `git diff --stat`: **33 Dateien, 1730 Insertions, 1533 Deletions**; diese Aenderungen waren vorbestehend bzw. entstanden parallel durch andere Prozesse. Der Bericht selbst liegt ausserhalb des Repos unter `/tmp`.

## offene_bedenken

- Recovery-Erfolge werden nicht in `ctox_harness_flow_events` persistiert. Deshalb ist der verlangte Zaehler dort 0, obwohl ein exakter durable Status-Marker eine reale App-Idle-Recovery belegt. Eine Aussage „0 Flow-Events = nie gelaufen“ waere sachlich falsch.
- Ein lebender, aber festgefahrener Worker erneuert die Lease ueber einen separaten Heartbeat weiter. Solange sein Key in der In-Memory-Active-Menge steht, ueberspringt der generische Sweep ihn (`src/core/mission/channels/mod.rs:3546-3548`); die App-Recovery ueberspringt ihn ebenfalls (`src/core/service/service.rs:18596-18599`). Dieser Fall wird nur als stuck evidence eskaliert, nicht automatisch reclaimed.
- Der Checkout wurde waehrend der Messung parallel geaendert. Runde 2 muss vor dem Schneiden der Welle den dann aktuellen Diff erneut pruefen.

## pfade

1. `src/core/service/service.rs:19431-19563, 24656-24696, 3303-3318, 3600-3611, 16511-16521, 18358-18405, 18565-18629`
   - Ursache/Handoff und Stop sauber machen; danach das App-spezifische Recovery-Netz entfernen.
2. `src/core/mission/channels/command_saga.rs:1106-1147`
   - Keine neue Reparatur erwartet; dies ist der bereits vorhandene, load-bearing Root-Fix, gegen den Runde 2 die Loeschung des App-Netzes absichern muss.
3. `src/core/mission/channels/mod.rs:3374-3742, 4688-4737, 5426-5513`
   - Keine neue App-Sonderlogik; TTL, Heartbeat-Reclaim, CAS und Lease-Clearing als generische Queue-Invarianten beibehalten und in Runde 2 gezielt verifizieren.
