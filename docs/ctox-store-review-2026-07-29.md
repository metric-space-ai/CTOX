# P7 — store.rs Review (6 Fokus-Pakete, Kimi, 2026-07-29)

Reviewt wurde der HEAD-Snapshot (stabile Zeilennummern als snapshot:<zeile>).
store.rs: 68.359 Zeilen (44.662 Produktion), 1.052 Produktions-Funktionen.
Noten: P7a Command-Routing **C−** · P7b Queue/Projektionen **C−** · P7c Modul-Lifecycle **C+** · P7d Policy/Capabilities **B−** · P7e Backup/Restore **C−** · P7f Querschnitt **D+**

---

# P7a — Command-Routing & Acceptance

## grade
**C-** — Das Command-Routing ist funktional und auf den sicherheitskritischen Pfaden (Origin-Trennung, Idempotenz, Recoverable-Replays) ernsthaft getestet, aber es ist kein Zustandsautomat, sondern ein akkretierter Sonderfall-Stapel: `accept_rxdb_business_command_with_origin` ist eine ~2.650-Zeilen-Funktion (snapshot:21481-24133) mit ~90 Inline-Match-Armen, 62 handgerollten Session-/Policy-Aufrufen und mindestens sieben verschiedenen Autorisierungsstapeln. Darunter liegt mit `mission::channels` (Aggregates/Effects/Transitions/Outbox, Idempotency-Conflict per payload_hash) ein echter Zustandsautomat — die store.rs-Schicht konsumiert ihn aber über stringly Dispositions, drei inkompatible „already accepted"-Antwortformen und eine Recoverable-Allowlist, die den generischen blockierten Todespfad für genau fünf Typen zurücknimmt. Gut ist: die Claim-Ledger-Integration mit echter Idempotenz-Durchsetzung, die Origin/Capability-Trennung mit Regressionstests und der ehrlich dokumentierte Fallthrough-Chokepoint. Dominantes Verfallsmuster: Sonderfall-auf-Sonderfall plus Text-Dispatch über Modulgrenzen, flankiert von vier parallelen Listen für dieselbe Recoverable-Familie.

## findings
`[HIGH] snapshot:21636-21672 — Uncertain-Claim-Sackgasse für alle nicht-recoverable Control-Commands`
Fällt ein Control-Command zwischen durablem Claim und Outcome-Write aus (Crash, SQLite-Contention), liefert der Replay `ok:false, task_status:"blocked", error_code:"dependency_missing", retryable:false` — terminal unlösbar, obwohl die Ursache keinerlei fehlende Dependency ist. Nur die fünf Typen aus `is_recoverable_background_control_command_type` (snapshot:21409-21418) entkommen per Re-Execution; alle anderen ~120 Control-Typen nicht. Im Feld: Commands kleben für immer auf „accepted/blocked", der irreführende error_code schickt die Diagnose in die falsche Richtung — exakt die Diagnose-Modi 6-8 aus `ctox-command-bus-diagnose-2026-07-28.md`, hier die store.rs-Seite davon.

`[HIGH] snapshot:22580-22592 — Fehlerpfad schluckt den Outcome-Write (8 Stellen)`
Das Muster `let _ = write_rxdb_control_command_outcome(..., "failed", ...); return Err(error);` wiederholt sich in den Guard-Armen (customers, external_sql, iot, invoices, appsec, support, threads; 8× im File). Schlägt der Outcome-Write unter SQLite-Contention fehl, geht der terminale Fehlerzustand verloren, der Claim bleibt non-terminal — und produziert beim nächsten Replay exakt die Sackgasse aus Finding 1. Ein transienter Lock verwandelt so einen retrybaren Handler-Fehler in einen permanent gestrandeten Command.

`[MED] snapshot:21379-21467, 24135-24191, 41726-41759 — Recoverable-/Klassifikations-Familie als vier parallele Hand-Listen`
Dieselbe Typmenge wird vierfach gepflegt: `is_rxdb_control_command_type` (21379), `is_recoverable_background_control_command_type` (21409), `recoverable_background_control_permission` (21452) und die modul-eigenen Listen in store.rs (`is_outbound_active_command` mit 40 Typen, `is_iot_active_command`, `appsec_..._requires_data_write`). Ein neuer recoverable Typ braucht vier synchronisierte Edits; Drift bedeutet entweder Verlust des Recovery-Pfads (→ Finding 1) oder unbeabsichtigt weites Replay. Dazu dupliziert store.rs Listenwissen, das den Modulen outbound/iot/appsec gehört.

`[MED] snapshot:21382 — Zirkuläre generierte Wahrheit: Klassifizierer konsumiert einen Regex-Scrape seiner selbst`
`is_rxdb_control_command_type` lädt `exact_control_types` per `include_str!("business_command_inventory.json")` — einer Datei, die `build_business_command_inventory.mjs` per Regex aus genau diesem Match scrapt. Ein Arm, der vom Regex-Schema abweicht (mehrzeilig, Guard-Syntax, Makro), verschwindet lautlos aus der Laufzeit-Klassifikation und landet in der Queue. Der `--check`-Drift-Test pinnt Datei-Frische, nicht die semantische Übereinstimmung Arm ↔ Klassifizierer — ein Guard-Arm ohne Klassifizierer-Eintrag läuft ohne durablem Claim.

`[MED] snapshot:21561-21596 — Drei inkompatible „already accepted"-Antwortformen`
Je nachdem, welcher Store den Replay zuerst beantwortet, kommt (a) die gemergte Lifecycle-Projektion mit Feld-Whitelist-Patching (`chat_id`, `outbound_text`, `response`, `answer`, `summary`), (b) das rohe Stored-Outcome oder (c) der Stub `{"status": "already_accepted"}` — ein Wert, der gar kein Status ist, ohne `ok`-Flag und ohne `task_id`. Consumer müssen alle drei Formen verstehen; Variante (c) ist eine Quelle des Diagnose-Workarounds #2 („accepted-Receipt trägt task_id nicht zuverlässig").

`[MED] snapshot:30452-30478 — Recoverable-Re-Execution mit eingefrorener Autorisierung bei gelöschtem User`
`authorize_recoverable_background_control_command` prüft die Session nur, wenn die Auflösung *gelingt*; bei `Err` (User gelöscht/deaktiviert, Token-System down) fällt es auf den persistierten Receipt mit `allowed:true` vom Zeitpunkt des ersten Accept zurück. Ein entzogener oder gelöschter Account stoppt also die Re-Ausführung seiner recoverable Commands (person_research, external_sql, outbound-Adapter) nicht. Als Durable-Authorization-Design nachvollziehbar, aber semantisch „Autorisierung eingefroren bei Accept" — und der Fall „User existiert nicht mehr" ist im Testmodul nicht gepinnt (65768 pinnt nur den fehlenden Receipt).

`[MED] snapshot:30762-30900 — ~140 Zeilen toter Code hinter `return` mit `#[allow(unreachable_code)]``
`record_business_command_intake_failure` returnt bei 30762 das channels-Ergebnis; die komplette alte Implementierung (SQL-Inserts, failure_document-Bau) steht unerreichbar dahinter. „Falsch schreiben → später reparieren"-Überrest der Migration: Leser können nicht unterscheiden, welche Buchführung die wahre ist, und der tote Code referenziert Contract-Felder (`contract_version == 2`, `replication_phase`), die bei Contract-Änderungen still veralten.

`[MED] snapshot:30622-30665 — Outbox-„business-os"-Destination schreibt Parallelwahrheit mit Regressions-Default`
Die Destination schreibt die Projektion erneut in `business_commands`/business_records — denselben Zustand, den `write_rxdb_control_command_state` (30316-30377) bereits dual geschrieben hat; ein Command-Status existiert so in fünf Repräsentationen (Ledger, Outbox, SQL-Tabelle, business_records, RxDB-Collection). Fehlt `status` in der Projektion, defaulted der Insert auf `"accepted"` (30625) — ein möglicher Status-Regress von terminal auf accepted, ohne dass ein Test die Destination abdeckt.

`[MED] snapshot:21616-21674 — Stringly Disposition-Match mit offenem Wildcard-Arm über Modulgrenze`
`claim.disposition` (`&'static str` aus mission::channels) wird per Stringliteral `"new"`/`"terminal"` gematcht; der `_`-Wildcard schluckt jede künftige vierte Disposition — für recoverable Typen stünde die dann still in der Re-Execution (21635), für alle anderen in der Blocked-Sackgasse. Text-Dispatch auf dem zentralen Lifecycle-Vertrag statt eines geteilten Enums.

`[LOW] snapshot:30270-30284, 30296-30300 — Status-/Feldableitung per String-Probing`
`target_task_id`/`target_record_id` werden über `command_type.starts_with("ctox.task.")` bestimmt; die Terminal-Abbildung matcht `"completed"|"cancelled"` und faltet alles andere auf `"failed"`; `task_status` ist pro Arm Freitext (`"updated"` bei task.update 22037, `"cancelled"` bei task.delete 22061). Keine Status-Enumeration, Vokabular driftet pro Arm.

`[LOW] snapshot:14628-14639 — Text-Dispatch auf Operation-Namen im Failed-Outcome`
`write_rxdb_failed_control_command_outcome` formt Felder über `operation == "release"` und `operation.starts_with("rollback")` — Fehlertexte/Feldformen hängen an frei erfundenen Operations-Strings der ~60 Aufrufstellen.

`[LOW] snapshot:13253, 30760 — Best-Effort-Outbox-Flush mit geschlucktem Fehler, Outbox ganz ohne Test`
`let _ = deliver_business_command_outbox(root, 10)` an zwei Stellen; `deliver_business_command_outbox` hat keinen einzigen direkten Test im Modul (kein Aufruf nach 44621), obwohl es Retry-Budget, Failed-Markierung und beide Destinationen enthält.

## healthiest_aspects
- **Echte Idempotenz-Durchsetzung**: payload_hash über den kanonischen Intent (snapshot:21320-21340) wird im Claim-Ledger mit `idempotency_conflict` bei geändertem Intent durchgesetzt — inkl. Verhaltenstest `replayed_control_command_rejects_changed_intent_before_stored_outcome_shortcut` (snapshot:54564-54613). Keine Parallelwahrheit: der Hash wird konsumiert, nicht nur dekorativ mitgeschleppt.
- **Origin-Trennung mit Zähnen**: TrustedLocal vs. ReplicatedPeer ist an der Accept-Grenze dokumentiert (snapshot:21469-21480) und durch den Regressionstest `capability_token_enforcement_gates_privileged_commands` (snapshot:47655-47720) gegen stille Aufweichung gepinnt.
- **Der Fallthrough-Chokepoint** (snapshot:24104-24130) ist als solcher benannt, begründet und auf ReplicatedPeer begrenzt — eine der wenigen Stellen, wo das Design schriftlich im Code steht.
- **Recoverable-Familie ist verhaltens-getestet** (snapshot:65610-66100): In-Flight-Schutz, Receipt-Recovery, Re-Auth beim Peer — die Tests laufen gegen tempdir-SQLite, also echte Verhaltens-Pins statt Mocks.

## coupling
Der Fokusbereich ist ein Nadelöhr mit breiter Fächerung: `mission::channels` (Claim-Ledger, Projektion, Outbox — die eigentliche State-Machine), `app_runtime` (admit/snapshot), `external_sql_sync`, `support`, `threads`, `invoices`, `iot`, `ats`-Handler, `coding_agents`, `person_research_command`, `install`, `secrets`. Die Grenze ist formal sauber (`is_*_command`-Prädikate + `handle_*`-Funktionen pro Modul), aber die Prädikat-Listen leben in store.rs und duplizieren Modulwissen (outbound: 40 Typen, iot: 8, appsec: 27). Zur Schwesterdatei `rxdb_peer.rs` ist `is_recoverable_background_control_command_type` als `pub(crate)` geteilt — gut; dass rxdb_peer laut Diagnose zusätzlich eine eigene 6-Typen-Liste im Kandidaten-SQL pflegt, ist die Parallelwahrheit über die Dateigrenze hinweg. Der generierte Inventory-Vertrag (`business_command_inventory.json`, Smoke-Tests) wird vom Fokus konsumiert — aber nur für exakte Typen; Prädikate bleiben handgeschrieben, und die Datei wird zirkulär aus dem eigenen Match generiert.

## test_coverage
Das Testmodul (ab snapshot:44621, ~314 Tests) deckt den Fokus auf den kritischen Pfaden gut ab: Idempotenz (49318, 54422, 54564, 54616), Origin/Capability-Gating (47650 ff., 60893), Recoverable-Familie (65610-66100), In-Flight-Guard (54616). Es sind überwiegend echte Verhaltens-Tests mit tempdir-Roots und SQLite — belastbare Pins. Zwei Einschränkungen: einige Tests bauen den Vorzustand per direktem SQL-Insert (z. B. 54584-54595), was implizit das Tabellen-Layout pinnt (Impl-Pin-Anteil); und drei Lücken sind belegbar: `deliver_business_command_outbox` hat null direkte Tests, die Blocked-Sackgasse (`dependency_missing`, snapshot:21636-21672) ist ungetestet, und der Receipt-Fallback bei gelöschtem User (30452-30478) ist nicht gepinnt. Ausgerechnet die zwei Pfade mit der größten Feld-Konsequenz (Sackgasse, Outbox-Regress) haben die dünnste Decke.

---

# P7b — Queue & Projektionen


## grade
C- — Der Queue-/Chat-Projektionskern funktioniert und ist auf den Terminalpfaden ernsthaft verhaltensgetestet (rxdb-Ebene, adversariale Trigger), aber die dominante Verfallsform ist exakt die aus rxdb_peer.rs bekannte Schuldklasse: sechs-plus parallele Schreiber mit divergierenden Dokumentformen für dieselben `business_commands`-/`ctox_queue_tasks`-Records, ein kanonischer Zustand in `channels` neben Kompatibilitätsprojektionen in `business_records`+RxDB, Drift wird nicht verhindert, sondern per `repair_queue_projections` nachträglich repariert — inklusive Text-Sniffing auf Status-Notizen, um terminale Zustände zu erraten. Dazu eine ~140-zeilige tote Schattenimplementierung und ein komplett ungetesteter `waiting_dependencies`-Pfad. Gut ist die kanonische Outbox mit Retry-Budget und Dead-Letters, das strukturelle Validierungs-Gate vor dem Writeback und die ehrliche Rennbedingungs-Dokumentation im Code.

## findings
`[HIGH] snapshot:39655-39674 + 39504-39516 — Terminal-Erkennung per Substring-Matching auf status_note`
`queue_status_note_is_terminal_success`/`_failure` raten terminale Zustände aus Freitext ("completed." + "changed ", "input exceeds the maximum length"). Das läuft nicht nur im Repair, sondern live in `effective_queue_projection_route_status` beim Schreiben jeder Queue-Projektion. Ein Worker, der deutsch notiert ("Aufgabe abgeschlossen, 3 Dateien geändert") oder anders formuliert, bleibt projektionsseitig `leased`/"running", obwohl der Task terminal ist — im Feld: Chat-Tasks hängen sichtbar auf "läuft", bis die Orphan-Reparatur sie irgendwann als failed stempelt. Der Test 58784 zementiert das Sniffing als Verhalten.

`[HIGH] snapshot:39478-39502 + 33081 — queue_task_payload fälscht command_type/inbound_channel pauschal zu "business_os.chat.task"`
Die generische Queue-Task-Projektion hardcodiert `"inbound_channel": "business_os.llm.chat"` und `"command_type": "business_os.chat.task"`, egal welches Kommando dahintersteht. `update_ctox_task` (33044) schreibt dieses Payload für beliebige Queue-Tasks und liefert es an Clients zurück. Im Feld: Nach jeder Titel-/Prioritäts-Editierung eines Nicht-Chat-Tasks (z. B. Research, Outbound) erscheint er in Queue-Views und Filtern als Chat-Task — eine selbstgebaute Falschwahrheit in einer angeblich kanonisch gespeisten Projektion.

`[MED] snapshot:30763-30903 — Tote Schattenimplementierung der Intake-Failure-Aufzeichnung`
`record_business_command_intake_failure` returned bei 30762 unbedingt; danach folgen ~140 Zeilen unerreichbarer Code (eigene Transaktion, eigenes failure_document-Shape, eigener Upsert) unter `#[allow(unreachable_code)]`. Der Test 46188 assertiert genau die Felder dieser toten Form (`attempt`, `exhausted`, `failure_document`), die aktuell nur zufällig von `channels::` in gleicher Gestalt geliefert werden. Im Feld: Wer einen Intake-Failure-Bug fixt, editiert mit hoher Wahrscheinlichkeit die tote Kopie und wundert sich über wirkungslose Änderungen.

`[MED] snapshot:13290-13311 vs. 13479-13492 — record_command ist replay-inkonsistent und nicht idempotent`
Der Hauptpfad macht `INSERT ... ON CONFLICT DO UPDATE SET status='accepted'` — ein Replay derselben command_id setzt ein terminales Kommando kompromisslos auf `accepted` zurück und überschreibt die Projektion; der Markdown-Schnellpfad dagegen macht schlichtes `INSERT` und scheitert beim Replay hart am UNIQUE-Constraint. Nur die Accept-Schicht (21556-21559) fängt Duplikate vorher ab; die primitive `record_command` selbst ist unsicher und verhält sich je nach Sonderpfad anders. Feld-Folge latent: doppelte Submits (UI-Retry, Replikations-Redrive) können terminale Commands sichtbar "zurückspulen".

`[MED] snapshot:13242-13262 + 21558 — waiting_dependencies: kein interner Wiederanlauf, null Tests`
Fehlende Dependencies führen zu Claim + Status `waiting_dependencies` und Outbox-Zustellung — danach endet jede Eigeninitiative des Stores; Wiederanlauf passiert nur, wenn der Client dasselbe Kommando erneut pusht (der Accept-Pfad lässt genau diesen Status durch, 21558). Trifft die Dependency per Replikation ein, evaluiert nichts neu. Im Testmodul (ab ~44.663) existiert kein einziger Test für Claim, Evidence-Shape, generation/content_hash-Pinning oder Resume. Im Feld: hängende Commands, wenn der Client nicht erneut sendet.

`[MED] snapshot:13451-13501 — Sonderfall-Synchronschluss für documents/Markdown mitten in der generischen Intake`
Ein kompletter alternativer Completion-Pfad (Modul==documents, Heuristik auf deutsche Anweisungstexte, eigener INSERT, eigene Reply, direkter `process_business_chat_reply`-Aufruf) sitzt vor der Queue-Einbettung. Das ist Sonderfall-auf-Sonderfall statt Zustandsautomat: eigene Idempotenzlücke (s. o.), eigener Stempel, eigener Reply-Text, und es umgeht die Queue-Projektion, die alle anderen Chat-Commands bekommen.

`[MED] snapshot:41537-41553 + 30583-30586 — Stempel-Disziplin: unbedingtes Überschreiben mit caller-beliebigem updated_at_ms`
`upsert_business_record` überschreibt rev/updated_at_ms/payload ohne LWW-Guard; `persist_business_command_lifecycle_projection` übernimmt den Stempel ungeprüft aus dem Dokument. Eine stale replayte Lifecycle-Projektion (rxdb_peer mergt Browser-Felder hinein und ruft genau diesen Pfad) schreibt den Record mit altem Stempel zurück — Checkpoint-basierte Pulls sehen die Regression nicht, Peers behalten Falschstand. Dieselbe Terminierung stempelt dreimal unterschiedlich: `process_business_chat_reply` (18938), `persist_terminal_...` (18544), `refresh_...` (20928).

`[MED] snapshot:41482-41492 + 39628-39646 + 38623-38636 — Hartkodierte Statusmaschinen parallel zum generierten Contract`
Drei separate Mappings (queue→command, command→queue, chat-tracking inkl. `"erledigt"`-Token) duplizieren, was `command-lifecycle.generated.js` bereits definiert — exakt der in der Command-Bus-Diagnose als G8 benannte Befund. `business_command_inventory.json` wird per `include_str!` konsumiert (21382), die Lifecycle-Wahrheit dagegen nicht; jede Statusänderung muss an vier Stellen zweisprachig nachgezogen werden.

`[MED] snapshot:18521-18532 — Post-Terminal-Refresh-Fehler nur als eprintln`
Wenn nach dem kanonischen Terminal-Übergang die Kompatibilitätsprojektion oder Outbox-Zustellung scheitert, wird nur geloggt ("remains queued for reconciliation"). Der Command ist kanonisch completed, UI-seitig ggf. stale, bis der Repair-/Reconcile-Pfad greift — die Reparaturschleife ist also nicht Fallback, sondern fester Bestandteil des Normalbetriebs.

`[LOW] snapshot:38463-38498 — Chat-Projektion: lastTrackingId-Clobbering und camelCase/snake_case-Mischung`
`business_chat_payload` schreibt `lastTrackingId` unbedingt, im task-losen Pfad als leeren String, und überschreibt damit einen zuvor gültigen Tracking-Zeiger; dasselbe Dokument mischt `lastTrackingId`, `replyFor`, `createdAt` mit `updated_at_ms`, `tracking_*`. Zusätzlich sind die abgeleiteten `tracking_*`-Felder im selben, per `push_collection_records` (20761-20811) von Clients beschreibbaren Dokument persistiert — abgeleiteter Zustand, den Clients fälschen oder veralten lassen können.

`[LOW] snapshot:19463-19491 — Repair verändert bei apply den kanonischen Queue-Task aus der Projektionsschicht heraus`
`repair_queue_projections` schreibt im apply-Modus über `channels::update_queue_task`/`ack_leased_messages` in den kanonischen Store zurück — ausgelöst durch das obige Notiz-Sniffing. Eine Reparaturfunktion, die aufgrund geratener Textmuster kanonische Zustände mutiert, invertiert die deklarierte Besitzrichtung.

## healthiest_aspects
- snapshot:18594-18610 + 19017-19023 — Strukturelles Gate `ensure_business_command_terminal_gate_ready` (Writeback nur nach persistiertem Ergebnis + Review/Validation) und ein Kommentar, der die `validating -> leased`-Race benennt und die Commit-Reihenfolge begründet — echte Zustandsautomaten-Disziplin statt nachträglicher Reparatur.
- snapshot:30600-30710 — Outbox mit Retry-Budget (MAX_ATTEMPTS=8), Dead-Letter-Pfad und transaktionalem Business-OS-Delivery (30630-30665): kanonischer Fortschritt hängt nie an der Verfügbarkeit der Kompatibilitätsstores.
- snapshot:38287-38317 + 38499-38513 — Chat-Materialisierung ist replay-idempotent (Dedup über message-id bzw. `replyFor`), in-place-Update der Status-Nachricht statt Duplikat.
- snapshot:41533-41536 + 13412-13446 — Secret-Redaktion (capability_token) zentral an der Projektionsgrenze; Dependency-Evidence mit generation_id/content_hash-Pinning statt bloßer Existenzprüfung.

## coupling
Die kanonische Wahrheit liegt in `channels` (ctox.sqlite3); store.rs delegiert Claims/Transitionen sauber dorthin — aber durchbricht die Grenze zweimal: Notiz-Sniffing auf `channels`-Textfeldern (leaky String-Vertrag) und kanonische Rückschreiben aus der Reparatur. `rxdb_peer.rs` co-besitzt die Projektionsdokument-Form (Merge in rxdb_peer.rs:5455-5493, Persistierung via 30559) und pumpt die Outbox (rxdb_peer.rs:4530) — Besitz ist effektiv dreigeteilt ohne Shape-Owner. Gegenüber den generierten Quellen: `business_command_inventory.json` wird konsumiert (gut), `command-lifecycle.generated.js` nicht (G8-Befund bestätigt). `push_collection_records` routet `business_commands` korrekt durch den Accept-Chokepoint, lässt aber Clients beliebige andere Collections (inkl. `business_chats` mit Derived-Tracking-Feldern) direkt schreiben.

## test_coverage
Das Testmodul deckt den Fokus auf den Terminalpfaden gut und verhaltensbasiert ab: `complete_business_command_from_queue_reply` mit rxdb-seitigen Assertions und einem adversarialen Trigger gegen präterminale Projektionen (57844-57953, Research-Variante 58354), `refresh_business_command_queue_task_projection` für blocked/active/retry_wait (57454, 57556, 57641), `repair_queue_projections` in vier Tests inkl. Dry-Run/Apply und Note-Sniffing (58583-59067), Chat-Persistenz über `record_command` (49374), Replay-Schutz `already_accepted` (54422, 54674), Intake-Failure-Budget (46188). Lücken: `waiting_dependencies` komplett ungetestet (kein Test referenziert den Status), Doppel-Completion-Idempotenz von `complete_business_command_from_queue_reply` nicht direkt gepinnt, Markdown-Schnellpfad-Replay ungetestet, Sprachrobustheit des Note-Sniffings nicht getestet — im Gegenteil: 58784 pinnt das Text-Matching als Sollpin, was die Schulde verfestigt.

---

# P7c — Modul-Lifecycle


## grade

C+ — Der Modul-Lifecycle ist an seinen neuesten Schreibpfaden diszipliniert: staged-swap-Aktivierung mit Backup-Restore, transaktionale Release-Aufzeichnung mit Manifest-Rollback bei DB-Fehler, Orphan-Guards direkt an den beiden Quellpfaden (User-Deaktivierung, Founder-Entzug), und ein dichtes Verhaltens-Testnetz, das diese Invarianten pinnt. Das dominante Verfallsmuster ist aber genau die aus der rxdb_peer-Kampagne bekannte Schuldklasse: Reparieren statt korrekt schreiben — hier sogar in drei übereinanderliegenden Schichten (Startup-Reconcile, Backfills im **Lesepfad**, manueller Repair-Command), während der Delete/Uninstall-Pfad seinen DB-Müll (aktive Grants, ACL, Release-Zeilen mit Status `released`) einfach liegen lässt und so genau die Defekte erzeugt, die später niemand automatisch repariert. Dazu Parallelwahrheiten (Manifest-Lifecycle-Triple vs. Release-Tabelle vs. Projektion), vier divergente „hat sich das Modul geändert"-Hash-Definitionen und hartkodierte `ctox-system`-Admin-Sessions als Policy-Bypass.

## findings

`[HIGH] snapshot:10854-10874 + 12670-12712 — Delete/Uninstall räumt keinen Lifecycle-DB-Zustand ab; Repair ist nur manuell`
`delete_installed_module` und `uninstall_app_module` löschen nur Verzeichnis + Layout-Eintrag. `business_module_releases` (inkl. Status `released`), `business_module_acl` und aktive `business_permission_grants` mit `scope_type='module'` bleiben vollständig stehen. Die dafür vorgesehene Reparatur (`repair_stale_module_permission_grants`, 9130) läuft ausschließlich über den manuellen Command `ctox.module.repair_lifecycle_projection` (einziger Aufruf: 22475) — kein automatischer Trigger. Feldfolge: Nach Reinstall eines Moduls unter derselben ID werden die alten Grants still wieder wirksam (Zugriffs-Resurrection für frühere Subjekte), und `projected_module_lifecycle` (2991-2996) findet die alte `released`-Zeile → die neu installierten Bytes erscheinen sofort als team-sichtbar mit dem **alten** data_access_review als Evidenz, ohne je ein Release-Kommando gesehen zu haben.

`[HIGH] snapshot:6448-6454 + 3915-3981 — Der Katalog-Lesepfad schreibt: permanente Backfill-Migration statt Release-Gate`
`module_catalog_for_rxdb` — die Projektions-**Lese**funktion, die auch von `business_os_why_diagnostics` (6504) und dem rxdb_peer-Polling aufgerufen wird — führt bei jedem Aufruf `backfill_semver_public_release_records` und `backfill_manifest_preview_audience_grants` aus. Die Backfill-Bedingung ist nur `runtime_installed && semver_major >= 1 && keine released-Zeile`: Wer über `ctox.module.save` (AppsModify) die `version` in module.json auf `1.0.0` setzt, bekommt beim nächsten Katalog-Read automatisch eine `released`-Zeile mit Channel `team` und synthetischem Review (`reviewed_by: ctox.release-record-migration`) — das data-access-review-Gate von `record_module_release` (8693) ist damit umgangen. Eine Migration ohne Migrations-Flag, die dauerhaft im Lesepfad lebt und Audit-Zeilen erzeugt, die behaupten, ein Release habe stattgefunden (test-gepinnt in 66515).

`[MED] snapshot:8757 + 8853 — Release-Status `rolled_back` konflatiert „abgelöst" und „zurückgerollt"`
`record_module_release` setzt bei **jedem** neuen Release alle bisherigen `released`-Zeilen auf `rolled_back`; `rollback_module_release` setzt per CASE alle außer der Ziel-Version ebenfalls auf `rolled_back` (inkl. älterer, längst historischer Releases). Es gibt keinen `superseded`-Status. `rollback_target` in der Projektion (4095-4099) ist damit schlicht „die vorige Release" — nach einem normalen v1→v2-Release wird v1 als Rollback-Ziel angezeigt, obwohl nie etwas zurückgerollt wurde. Die Historie kann einen echten Rollback nicht von normaler Ablösung unterscheiden; das ist durch Tests (67662-67667) zementiert.

`[MED] snapshot:8844-8848 + 8709-8735 — rollback_module_release überschreibt module.json asymmetrisch und ohne Snapshot`
`record_module_release` schreibt das Manifest nur bei `runtime_installed`; `rollback_module_release` schreibt die gespeicherte `manifest_json` **unconditional** in den Modulpfad — auch bei Katalog-Source-Modulen, deren Manifest beim Release gar nicht angefasst wurde. Zwischen Release und Rollback am Working-Tree geänderte Manifest-Inhalte werden ohne vorherigen Versionssnapshot (anders als `rollback_module_to_version`, 12228-12231) verworfen. Feldfolge: stiller Datenverlust un-gesicherter Edits beim Rollback.

`[MED] snapshot:8983-9046 — repair_invalid_module_release_version_refs repariert einen Defekt, den kein Produktiv-Schreibpfad erzeugt`
`record_module_release` validiert beide Refs vor dem Schreiben (`ensure_module_version_ref_exists`, 8667-8673), kein Produktivcode löscht Zeilen aus `business_module_versions` (das einzige DELETE steht im Test, 55097). Die Repair-Funktion heilt also nur Legacy-DBs — während die tatsächlich produzierte Defektklasse (Stale Grants nach Uninstall, s.o.) ohne Aufruf-Automatik bleibt. Das Repair-Portfolio ist an den Defektquellen vorbei sortiert; zudem läuft `repair_module_lifecycle_projections` (9210) ohne Transaktion — ein Fehler mittendrin hinterlässt halb applizierte Repairs, obwohl das Resultat „completed" meldet.

`[MED] snapshot:7616-7725 — Startup-Reconcile: gefälschte Admin-Session, unbeaufsichtigter Discard, Store vor HTTP-Bind`
`reconcile_release_managed_module_shadows` baut eine hartkodierte `ctox-system`-Admin-Session (7638-7650, wortgleich dupliziert in 6458-6470) und ruft `update_module_to_catalog` mit `mode: "discard"` und leerem Baseline-Precondition — lokal angepasste Shadows werden beim Serverstart ohne Rückfrage überschrieben (immerhin mit Recovery-Version). Außerdem läuft der Reconcile in `serve_business_os` **vor** dem HTTP-Bind und öffnet dabei den SQLite-Store — exakt die Reihenfolge, die der Kommentar wenige Zeilen später in server.rs („Claim the HTTP surface before opening the store") vermeiden will.

`[LOW] snapshot:6249-6310 + 6177-6195 + 7776-7812 + 11430-11462 — Vier divergente Änderungs-Hash-Definitionen im selben Domänenrand`
File-Tree-Stamp (Name+Größe+Mtime, skippt Dotfiles), Asset-Revision (Inhalt, skippt Dotfiles), Payload-SHA des Reconcile (Inhalt, alles außer module.json, keine Pfadfilter) und Bundle-SHA (nur `is_allowed_source_path`, <1MB, skippt Dotfiles). Reconcile-Skip-Test (7689-7695) und `already_current`-Test des Updaters (7532) messen also Verschiedenes: Module mit Dotfiles, >1MB-Assets oder nur-Manifest-Drift fallen zwischen die Definitionen — im Feld Dauer-Reconcile ohne Update oder unentdeckte Drift; zudem triggert eine inhaltliche Änderung bei gleicher Größe+Mtime keinen Katalog-Resync.

`[LOW] snapshot:8719-8731 + 4124-4187 + 7025-7055 — Lifecycle als Parallelwahrheit: Triple-Write ins Manifest, Recompute in der Projektion, SemVer als Zustandssignal`
Release schreibt `visibility_state`/`audience`/`release_channel` mit identischem Wert ins Manifest; die Projektion rechnet alles neu und überschreibt das Geschriebene wieder. Die „private App"-Invariante hängt an `parse_business_app_semver_major >= 1` — einem frei editierbaren String. Crash zwischen Manifest-Write (8733) und DB-Commit hinterlässt ein Manifest mit Version 1.x + Lifecycle `team` ohne Release-Zeile: `module_requires_active_responsibility` liefert dann `false` und der Orphan-Guard greift nicht mehr — kein Repair deckt genau diesen Zustand ab.

`[LOW] snapshot:12578-12585 + 14967/15093 — Kleinigkeiten: Temp-Dir-Leak im Install-Fehlerpfad; Activity-Feed als rekonstruierte Parallelwahrheit`
Schlägt `find_module_json_dir_for_install` fehl, bleibt das entpackte ZIP in `std::env::temp_dir()` liegen (Cleanup erst bei 12645). Und der Activity-Feed liest echte `business_events` **plus** aus `business_commands`-Projektionen rekonstruierte Lifecycle-Events mit Dedup (15093+) — zwei Wahrheiten für dasselbe Geschehen; die Event-Mapping-Tabelle (14744) kennt nur release/rollback, Install/Delete/Update tauchen im Feed nicht auf.

## healthiest_aspects

- `activate_staged_module_directory` (snapshot:11049-11104): Stage→Backup→Rename mit Restore-Pfad — kein halbgeschriebenes Modulverzeichnis; sauber test-gepinnt (49526, 49554).
- `record_module_release` (snapshot:8787-8802): Manifest-Restore bei DB-Fehler, Ref-Validierung **vor** dem Manifest-Write (8667-8673), Versionszähler+Status-Umschlag in einer Transaktion — inkl. Failure-Injection-Tests (55407, 55486, 55567) und Replay-Idempotenz (54422).
- Orphan-Verhütung an der Quelle statt nachträglich: `upsert_user` (7250-7260) und `assign_module_founder` (7318-7331) erzwingen Recovery-Responsibility auditiert (`upsert_module_founder_assignment_record` schreibt Audit-Event + Projektion, 7125-7173), bevor die letzte Verantwortung entfällt.
- Der Katalog-Change-Stamp (6127-6142) kapselt die Resync-Entscheidung vollständig; der rxdb_peer konsumiert ihn als schmale Grenze statt eigener Listen.

## coupling

Nach außen sauber genutzt: server.rs (Startup-Reconcile + Katalog-Refresh), rxdb_peer (pollt `module_catalog_projection_stamp`, schreibt `module_catalog_for_rxdb` in die RxDB — Grenze ist der Stamp, das ist sauber), channels-Saga für `set_visible` (23526-23600, dort sogar mit Kompensation). Weniger sauber: Der Repair-Command ist in der generierten `business_command_inventory.json` als First-Class-Command verankert und wird vom handgeschriebenen Dispatch-Match bedient (22463) — die generierte Wahrheit wird konsumiert, nicht dupliziert, gut; aber die gefälschte `ctox-system`-Session ist an zwei Stellen wortgleich dupliziert, und der Startup-Reconcile unterläuft die Bind-vor-Store-Ordnung des Servers. Der Activity-Feed (Nachbardomäne) rekonstruiert Lifecycle-Events parallel — Kopplung durch doppelte Wahrheit, nicht durch Schnittstelle.

## test_coverage

Das Testmodul deckt den Fokus ungewöhnlich gut ab und pinnt **Verhalten**, nicht Implementierung: Repair dry-run/apply inkl. Projektions-Sanitisierung (54687-55312, inkl. Secret-Nicht-Weitergabe 55167), Ref-Validierung vor Manifest-Write (55407), Manifest-Restore bei DB-Fehlern (55486, 55567), Orphan-Guards in beide Richtungen (51564-51646, 51712), Release-Audit-Events auch für Fehlschläge (55661-55960), Major-Line-Guard (67747), Rollback-Target-Semantik (67582), Preview-Grant- und Release-Backfill (66515, 50687). Lücken: Der volle Reconcile-Sweep hat nur Prädikats-Unit-Tests (50749, 50768), keinen End-to-End-Lauf; Uninstall→Reinstall mit Grant-/Release-Erbe ist ungetestet (genau die HIGH-Lücke); der SemVer-Bump-Bypass des Backfills ist als gewollte Migration gepinnt, nicht als Gate verifiziert.

---

# P7d — Policy, Permissions & Capabilities


## grade
B- — Der Policy-Kern von store.rs ist erstaunlich gesund für eine 68k-Zeilen-Datei: es gibt genau EINE Bewertungsstelle (`policy::evaluate` in policy.rs plus exakt ein Grant-Overlay in `evaluate_policy_with_explicit_grants`, snapshot:2708-2752), durch die alle ~60 Entscheidungs-Call-Sites trichtern; die Netzgrenze ist fail-closed (ReplicatedPeer ohne gültiges Capability-Token → harter Bail, snapshot:30125-30127); Token-Verifikation bindet Rolle UND Epoch gegen die DB (29698-29726); Revocation ist als DB-Trigger-Invariante implementiert, nicht als App-Disziplin (44468-44509). Das dominante Verfallsmuster liegt nicht in der Bewertung, sondern drumherum: die Capability-Token-Ausstellung ist mit stiller User-Provisionierung/Reaktivierung verbacken (ung audited), die Eskalations-Guards special-casen nur "chef" und vergessen "admin", und das „legacy"-Grant-Materialisierungsprogramm ist kein Migrationsskript mit Endzustand, sondern ein permanentes Allow-All-Fundament der Datenebene. Enforcement ist zudem ein ~60-mal copy-pastetes Drei-Zeilen-Idiom — die Bewertung ist zentral, die Durchsetzung verschmiert.

## findings
[HIGH] snapshot:29947-29981 — Token-Ausstellung mutiert still User-Stammdaten (managed path)
`issue_business_os_capability_token_for_managed_user` nimmt id+rolle vom Wire (SSH-Control-Plane/ctox.dev) und macht `INSERT … ON CONFLICT DO UPDATE SET role=excluded.role, active=1` — ohne Prüfung des Vorzustands und ohne `insert_business_event` (Kontrast: Grant-Änderungen werden auditiert, 21918-21939; User-Upserts auditiert, 7288). Ein in Business OS deaktivierter User, der im ctox.dev-Tenant noch existiert, wird beim nächsten Token-Request silently reaktiviert und mit der assertierten Rolle (inkl. chef) neu signiert; der einzige Effekt ist der Epoch-Bump des Triggers, im Audit-Trail steht nichts. Der Session-Pfad (29905-29939) ist durch `session_with_persisted_user`-Revalidierung (12854-12868) weitgehend gedeckt, der Managed-Pfad nicht.

[HIGH] snapshot:18254-18264 — users.manage-Grant kann admin-User minten
`user_upsert_policy_decision` eskaliert den Check auf WorkspaceManage nur bei `target_role == "chef"`; für "admin" reicht die UsersManage-Entscheidung — die auch ein expliziter Workspace-Grant erfüllt (2718-2720). Ein User mit nur-users.manage-Grant legt einen admin-User an; admin hat nativ RolesManage/UsersManage/SecretsManage/AppsInstall (policy.rs-Matrix) → faktisch Vollübernahme bis auf workspace.manage. Test 49194 pinnt exakt den chef-Block (`users_manage_grant_cannot_assign_owner_role`), ein analoger admin-Pin existiert nicht — die Lücke ist also weder Zufall noch abgesichert.

[MED] snapshot:29766-29841 — „legacy" Grants sind permanentes Allow-All, kein Migrations-Endzustand
`ensure_legacy_collection_grants` materialisiert role-weite data.read+data.write für founder+user auf JEDE nicht-admin Collection und läuft bei Token-Ausstellung (29873), rxdb_peer-Startup und threads mit. Da die Rollenmatrix für user/founder bei DataRead/DataWrite deny ist (policy.rs), ist diese „Migration" der einzige Grund, warum die Datenebene für Normaluser funktioniert — sie ist load-bearing und hat keinen Abschaltpfad; per-Collection-Granularität existiert, wird aber defaultmäßig unterhöhlt. Dazu Sonderfall-Magie: `ctox_queue_tasks` wird als String-Literal neben `policy::ADMIN_ONLY_COLLECTIONS` geskippt (29782) statt in der einen Deny-Set-Quelle.

[MED] snapshot:18217-18231, 21676-22176 — Enforcement ist Copy-Paste-Idiom, kein Zwang
Jeder Arm im Control-Dispatcher wiederholt manuell `session → policy_decision → reject_command_if_policy_denied`. Nichts im Typsystem oder in einer Tabelle zwingt einen neuen Command-Type durch ein Gate; das Meta-Test `active_app_command_families_require_native_policy_gates` (48038) deckt nur ausgewählte App-Familien ab. Ein vergessener Arm ist ein ungegates privilegiertes Kommando — genau die Fehlerklasse, die bei ~60 Stellen statistisch irgendwann zuschlägt.

[MED] snapshot:21452-21467, 14357-14374, 14561-14572 — Command-Type→Permission-Mapping als Handlisten, parallel zur generierten Inventory
Mindestens drei handgepflegte String-Matcher (`recoverable_background_control_permission`, `queue_command_policy_target`, `app_build_command_policy_target`) bilden Command-Types auf Permissions ab; `business_command_inventory.json`/`command-lifecycle.generated.js` werden von diesem Bereich nie konsultiert. Neue Command-Types driftet das Mapping still auseinander — bei `recoverable_background_control_permission` bedeutet `None`, dass der Recovery-Pfad ohne Authorization-Receipt läuft.

[LOW] snapshot:44480-44505 — Epoch-Trigger invalidiert flächendeckend
Jede INSERT/UPDATE/DELETE auf `business_permission_grants` mit subject_type='role' bumped die capability_epoch ALLER User dieser Rolle. Ein einzelnes `ctox.app.access.grant` an Rolle 'user' killt jeden User-Token der Instanz (12h-TTL, 29654) — fail-safe-Richtung, aber spürbare Re-Issuance-Flut auf lebenden Systemen.

[LOW] snapshot:2852-2868 — `deny_supported: false` als dokumentierte Granularitäts-Decke
Es gibt keine Deny-Grants: einen einzelnen User aus einer Role-weiten (Legacy-)Grant auszunehmen ist unmöglich, außer man deaktiviert die Grant für alle. Im Feld heißt das: Austritte/Sonderfreigaben werden über User-Deaktivierung statt über gezielte Verweigerung gelöst.

[LOW] snapshot:29698-29726 — Token-Verifikation macht pro Aufruf einen SQLite-Lookup
Korrektheit vor Performance ist richtig, aber auf heißen Command-Pfaden ist das ein ungepufferter DB-Roundtrip pro Verifikation; kein Claim-Cache mit Epoch-Invalidierung.

## healthiest_aspects
- snapshot:30068-30151 — `rxdb_session_from_command` dokumentiert die Trust-Boundary explizit (SECURITY-Kommentar) und setzt sie um: ReplicatedPeer ohne Token bails hart; auch TrustedLocal liest die Rolle aus der DB (`trusted_rxdb_command_user`), nur die ID ist geclaimt.
- snapshot:2708-2752 — `evaluate_policy_with_explicit_grants`: eine Overlay-Stelle (Rollenmatrix → exakte aktive Grants), kein zweiter Bewertungspfad in Produktion (die ~60 Call-Sites trichtern hier durch).
- snapshot:44468-44509 — Capability-Revocation als SQLite-Trigger (role/active-Änderung + Grant-DML), also DB-Invariante statt App-Disziplin; Test 47485 pinnt das Verhalten.
- snapshot:17992-18081 + 20198-20203 — Audit-Kontext wird per Key-Allowlist + Truncation sanitisiert, und `capability_token` wird post-merge aus jedem replizierten Dokument geschrubbt; der Token lebt nur in der nativen `client_context_json`-Spalte.

## coupling
policy.rs (Matrix, `ADMIN_ONLY_COLLECTIONS`) und capability.rs (HMAC sign/verify) sind saubere Import-Grenzen; der HMAC-Key liegt korrekt im Secret-Store (29650-29685). Nach außen: rxdb_peer.rs konsumiert `ensure_legacy_collection_grants`/`collection_authz_enabled`/`capability_allows_collection_permission` (Grenze sauber), threads.rs ebenso plus Server-Owned-Collection-Wissen (29786 — leichte Doppelverwahrung des „server-owned"-Konzepts über zwei Module). Unsauber ist die Grenze nach server.rs/mcp_channel.rs/service: `issue_business_os_capability_token_for_managed_user` wird von fünf Stellen mit Wire-assertierter Rolle gefüttert (server.rs:3815, mcp_channel.rs:6979, rxdb_peer.rs:15382+, service/business_os.rs:502) — die Rolle-als-Wahrheit-Verschmelzung (Finding 1) liegt genau auf dieser Naht. `session_with_persisted_user`-Revalidierung in server.rs mildert den Session-Pfad korrekt ab.

## test_coverage
Das Testmodul deckt den Fokus ungewöhnlich gut und auf Verhaltensebene (End-to-End über `accept_rxdb_business_command*`, nicht Impl-Pins): Token-Bindung/Epoch-Revocation (46799, 47455, 47485, 47519), exakte Legacy-Grants fail-closed inkl. Admin-Only- und Server-Owned-Ausnahmen (47541, 47594), Enforcement am ReplicatedPeer (47656), Token-Redaction (47732, 47842), Meta-Gate für App-Command-Familien (48038), Grant-Semantik (48559, 48681, 48755, 49464, 49580), Eskalations-Guard für chef (49194, 48597), strukturierte Denials + Audit-Events (50418, 50853, 51097, 51215), Retention-Policy-Gating (52278, 52372), Release/Rollback-Permission-Sonderwege (53477-54326), Stale-Grant-Repair (54909), Why-Erklärbarkeit (66887). Lücken spiegeln exakt die HIGH-Findings: kein Test, dass ein deaktivierter User durch Managed-Token-Reissuance deaktiviert BLEIBT; kein Test, dass users.manage-Grant kein admin minten kann; kein Meta-Test, der einen neuen Control-Arm ohne Policy-Gate als Fehler markiert.

---

# P7e — Backup, Restore & Audit


## grade
**C-** — Der Backup/Restore-Bereich ist handwerklich weit über dem rxdb_peer-Schuldniveau: chunkweises AES-256-GCM mit per-Chunk-AAD, Verify-after-Encrypt-Roundtrip, VACUUM-INTO-Snapshots, PRAGMA integrity_check auf den Restore-Kopien, Dry-Run-Prune und ein erzwungener Redaktions-Scanner vor jedem Artifact-Write sind echte, getestete Substanz. Das dominante Verfallsmuster ist aber „belegte Behauptung statt durchgesetzter Kontrolle": Manifest-Felder wie `key_must_not_travel_with_artifact: true`, die 14-Tage-Raw-Retention und die HMAC-Manifest-Signatur behaupten Kontrollen, die der Code strukturell unterläuft (Schlüssel liegt im Klartext-Snapshot neben dem Manifest; Retention hat keinen Scheduler; Signaturschlüssel reist im Backup mit). Die Leitfrage — ist Restore je end-to-end verifiziert? — ist mit **Nein** zu beantworten: Der Drill validiert eine isolierte Kopie, der tatsächliche Active-Root-Restore existiert nur als manuelles JSON-Runbook, nie ausgeführt, nie getestet. Die Versionskompatibilitäts-Matrix ist echt erzwungen, aber degeneriert (min == max == 1, nur Same-Version).

## findings
`[HIGH] snapshot:16433-16467, 16082-16166 — Portable-Encryption-Key reist ab dem zweiten Drill-Lauf im eigenen Artefakt mit`
`run_business_os_backup_restore_drill` snapshotet `ctox-secrets.sqlite3` (16453-16456) in den Klartext-Snapshot und in die ZIP; der Portable-Key wird erst beim ersten Lauf danach erzeugt (16098). Ab Lauf zwei liegt der AES-256-GCM-Schlüssel sowohl im unverschlüsselten `runtime/backup/business-os-drill-*/snapshot/`-Verzeichnis als auch innerhalb des mit sich selbst verschlüsselten Zips. Das Manifest behauptet gleichzeitig `key_must_not_travel_with_backup_artifact: true` (16162). Im Feld: Die Eskrow-Trennung ist nur nominal — wer das Drill-Verzeichnis liest, hat Schlüssel und Chiffrat beisammen; ein Auditor, der die ZIP-Inhalte prüft, findet den Schlüssel, der angeblich nie im Artefakt ist.

`[HIGH] snapshot:15990-16011, 16806-16850 — HMAC-Manifest-Signatur ist zirkulär und im Disaster-Fall unbrauchbar`
Der Signaturschlüssel liegt in `ctox-secrets.sqlite3`, die im selben Drill-Verzeichnis im Klartext-Snapshot mitgesichert wird: Wer das Backup manipulieren kann, kann das Manifest neu signieren — Tamper-Evidence nur gegen Angreifer ohne Dateizugriff, also gegen niemanden. Umgekehrt schlägt `inspect_business_os_backup_manifest` auf einer neuen Maschine (echter Disaster-Restore) immer fehl, weil der Signing-Key nur aus dem aktuellen Root gelesen wird (16827) und `signature_valid=false` zu `ok=false` führt (16730-16742). Das Eskrow-Runbook deckt nur den Portable-Key ab, nicht den Signing-Key. Die Kontrolle versagt in beide Richtungen: offen gegen die Bedrohung, geschlossen gegen den legitimen Fall.

`[HIGH] snapshot:16637-16710, 95 — 14-Tage-Raw-Retention wird deklariert, aber nie automatisch vollzogen`
`BUSINESS_OS_BACKUP_RAW_RETENTION_DAYS = 14` erzeugt ein `expires_at_ms` im Manifest, aber `prune_business_os_backup_restore_drills` wird ausschließlich vom manuellen CLI-Pfad `ctox business-os backup prune-drills` aufgerufen (einziger externer Caller: `src/core/service/business_os.rs:359`) — kein Daemon, kein Scheduler, kein Start-Hook. Im Feld bleiben Klartext-Snapshots inklusive komplettem Secret-Store unbefristet auf Platte, während Manifest und Runbook eine 14-Tage-Löschung behaupten. DSGVO-relevant: Die Retention-Policy ist ein Etikett ohne Vollzug.

`[MED] snapshot:16992-17142 — Active-Root-Restore existiert nur als JSON-Runbook; Drill ≠ Realfall`
Der Drill validiert eine isolierte Kopie unter `restore-root/`, aber der destruktive Rücksicherungspfad ist nirgends Code — nur ein Runbook aus zehn manuellen Operator-Gates (Quiesce, Copy-back, Restart). Copy-back laufender WAL-Datenbanken, Löschen von -wal/-shm, Reihenfolge der Store-Ersetzung: alles ungetestet und unautomatisierbar. `remaining_boundaries` (16628-16633) ist ehrlich, aber die Konsequenz bleibt: Der einzige je verifizierte „Restore" ist der, der im Ernstfall nicht stattfindet.

`[MED] snapshot:15571, 17160-17191, 1024-1029 — Blocking-Check native_store_tables ist selbsterfüllend`
Der Readiness-Export auf dem restore_root öffnet den Store über `open_store`, das `ensure_store_schema_once` → `migrate()` ausführt: Fehlende Kern-Tabellen werden beim Prüfen angelegt. Der blocking-Check kann eine leere oder driftende Restore-Kopie nicht erkennen (nur komplett fehlende Dateien fängt der Integrity-Pfad ab), und der Drill mutiert nebenbei den Restore-Snapshot — Schema-Drift, den die Kompatibilitätsschicht offiziell blockt, heilt der Prüfer still.

`[MED] snapshot:15345-15359, 22230-22251 — Typisierte Audit-Retention-Policy ist per Request aushebelbar`
`business_os_effective_audit_retention_days` lässt `request.retention_days` die persistierte Policy schlagen; der `ctox.business_os.audit.retention`-Command (Gate nur `UsersManage`) kann mit `prune: true` und `retention_days: 1` das Audit-Log faktisch löschen, ohne die Policy zu ändern. Der Drill-Check `typed_audit_retention_policy_state` (15669-15675) erzählt eine stärkere Geschichte als der Code durchsetzt. Zusätzlich haben die Export-Artefakte in `runtime/business-os/audit-exports` selbst keine Retention — sie wachsen unbegrenzt und werden in jedes Backup mitkopiert.

`[LOW] snapshot:16013-16028, 16860-16901 — Versions-„Matrix" ist degeneriert: min == max == 1`
`supported_manifest_schema_versions` präsentiert sich als Range, erlaubt aber exakt Schema 1 und exakt String-gleiche `CARGO_PKG_VERSION`; Prerelease-Builds (`1.2.3-beta`) gelten als Cross-Version und werden geblockt. Erzwingung ist echt und getestet — aber es gibt keine Stufenlogik, die eine v2 je aufnehmen könnte, ohne die Konstante zu ändern; die „Matrix" ist Präsentation.

`[LOW] snapshot:16976-16990 — inspect-manifest akzeptiert absolute Pfade aus dem Manifest`
`resolve_backup_manifest_artifact_path` übernimmt absolute `ciphertext.path`-Angaben ungeprüft und hasht die referenzierte Datei. Nur lokaler Lesezugriff durch einen Admin-CLI-Aufruf, aber ein präpariertes Manifest kann so Existenz/Hash beliebiger lokaler Dateien sondieren.

`[LOW] snapshot:16438, 15871-15900, 15798 — Der „Readiness-Dry-Run" ist nicht nebenwirkungsfrei`
Der Drill erzeugt beim ersten Lauf produktive Signing-/Encryption-Keys und schreibt Secret-Records; jeder Command-Lauf schreibt ein weiteres Export-Artefakt nach `runtime/business-os/restore-drills/`, für das es keinerlei Prune gibt (nur Drill-Dirs unter `backup/` werden gepruned). Mutierender Dry-Run plus unbegrenztes Artefakt-Wachstum.

## healthiest_aspects
- Verify-after-Encrypt-Roundtrip mit per-Chunk-AAD, Frame-Magic, Chunk-Längen-Deckel und Trailing-Bytes-Check (16307-16404); Nonce-Konstruktion mit Overflow-Guard statt stillem Wrap (16406-16418).
- Redaktions-Scanner `support_artifact_forbidden_paths` wird per `anyhow::ensure!` vor jedem Artifact-Write erzwungen (15785-15790, 15460-15465, 17846-17889) — kein Opt-out, deckt verschachtelte Pfade ab.
- Prune ist konservativ: Dry-Run-Modus, Dirs ohne Retention-Policy werden gemeldet, aber nie gelöscht (16664-16679) — genau so getestet (53119-53196).
- Crash-sichere Snapshot-Mechanik: VACUUM INTO (17531-17548), danach PRAGMA integrity_check auf den Restore-Kopien (17550-17568), Gesamt-`ok` UND-verknüpft aus Integrity und Readiness (16579-16586).

## coupling
Der Fokusbereich steht bewusst quer: `crate::secrets`, `crate::persistence` (Payload-Store für MCP-/Retention-Policy), `crate::paths::backup_dir`, rxdb_peer-Seite (`rxdb_store_path`, `rxdb_collection_table_name`), `mcp_channel`-Policy und Command-Dispatch (`write_rxdb_control_command_outcome`, `workspace_policy_decision`). Die Grenze ist überwiegend sauber: Die drei Command-Typen werden aus der generierten `business_command_inventory.json` konsumiert (include_str! bei 21382; alle drei Typen dort vorhanden) — **keine Parallel-Liste**. Zwei schmutzige Stellen: (1) Der Drill ruft `open_store` auf fremden Roots und erbt dessen Schema-Heilung — falsche Abstraktion für einen Prüfer. (2) `service/business_os.rs:352-384` ist die einzige Aufrufquelle für run/prune/inspect — die fehlende Scheduler-Kante ist der Retention-Befund.

## test_coverage
Vier substanzielle Tests, überwiegend Verhaltensebene über öffentliche Einstiegspunkte: `backup_restore_drill_copies_and_validates_isolated_restore` (52498, kompletter Drill inkl. Release-Seeding, Manifest-Hash, Signatur-, Eskrow-Assertions — der stärkste Test), `backup_manifest_version_compatibility_blocks_cross_version_and_downgrade` (53077, Unit-Pin), `backup_restore_drill_prune_deletes_expired_drill_dirs` (53119, Dry-Run + Löschverhalten), `backup_restore_drill_command_is_runtime_gated_and_sanitized` (53199, Policy-Gate + Sanitizing). Was fehlt, spiegelt exakt die Findings: kein Tampered-Manifest-Test gegen `inspect_business_os_backup_manifest` (Signatur-Theater wäre sofort sichtbar), kein Test auf Schlüssel-Abwesenheit im ZIP ab Lauf zwei, kein Preflight-Test auf schlüssellosem Fremd-Root, kein End-to-End-Active-Root-Restore-Test, kein Retention-Automatik-Test (Feature existiert nicht). Die Boundary-Assertions 52607-52645 pinnen Text-Strings in `remaining_boundaries` — Impl-Pinning, das bei Textänderung bricht, ohne Verhalten zu schützen.

---

# P7f — Querschnitt: Fehler-Semantik, Kopplung, Totholz


## grade

D+ — Der Kern funktioniert und ist ungewöhnlich stark getestet (288 Tests, ~23.700 Zeilen Testmodul ab snapshot:44621), aber die strukturelle Mitte ist eine 2.654-Zeilen-Gottfunktion (`accept_rxdb_business_command_with_origin`, snapshot:21481-24134), die Claim, Recovery, Policy, Dispatch und Outcome-Schreiben für ~15 Kommandofamilien inline kennt. Dominantes Verfallsmuster: Fehler- und Statussemantik wird per Substring auf Freitext entschieden (Fehlertext → error_code, Queue-Notiz → Terminalstatus, Sendefehler → Reason-Code, Natural-Language → Markdown-Edit), während gleichzeitig eine generierte Wahrheitsquelle existiert, die zirkulär aus eben dieser Datei regex-gescraped und per `include_str!` wieder eingelesen wird. Dazu eine ~250-Zeilen tote Fallback-DOCX-Kette, vier parallele Status-Vokabulare (die generated Lifecycle-Konstanten ignoriert store.rs komplett) und direkte Zugriffe auf zwei fremde SQLite-Schemata. Gut sind: durchdachte Failure-Semantik-Inseln (retryable Send-Blocks, Saga-Kompensation, RAII-Guard), Dry-Run-Disziplin aller Repair-Pfade und Performance-Regression-Pins per Zähler-Assertions.

## findings

`[HIGH] snapshot:22535-22541 — Office-Error-Code per Text-Sniffing über Modulgrenze`
Der Office-Arm klassifiziert `error.to_string().contains("version_conflict")` / `"feature_dependency_pending"` in einen persistierten `error_code`, auf dem Retry-Verhalten der UI hängt. Der String wird in office_engine.rs:8262 als plain anyhow-Text erzeugt (`"version_conflict: base editor payload hash does not match"`). Jede Umformulierung dort macht retriable Versionskonflikte lautlos zu `"office_engine_failed"` — im Feld: Editoren verlieren ihre Merge-/Retry-Spur, Nutzer sehen generische Engine-Fehler statt Konfliktauflösung.

`[HIGH] snapshot:39655-39676 — Queue-Terminalstatus aus Freitext-Notiz geraten`
`queue_status_note_is_terminal_success/failure` entscheidet per Substring (`"terminal-success"`, `" completed."` + `"changed "`, `"turn/start failed"`), ob ein geleaster Task als handled/failed gilt — Kontrollfluss, der an zwei Stellen (snapshot:39504-39516, 19468-19510) Route-Status umschreibt und im Repair-Pfad sogar ackt. Eine Notiz wie „turn/start failed to deliver, retrying" wird als terminal failure gewertet; eine Erfolgsnotiz ohne das Wort „changed" nicht als Erfolg. Im Feld: falsch terminale Commands, die die G3-Re-Drive-Logik der Diagnose nicht mehr erreicht, und Repair-Läufe, die auf geratenen Zuständen acken.

`[HIGH] snapshot:21379-21408 — Zirkuläre Wahrheit: Klassifizierer liest seine eigene Regex-Kopie`
`is_rxdb_control_command_type` lädt `exact_control_types` per `include_str!("business_command_inventory.json")` — einer Datei, die `tools/build_business_command_inventory.mjs` per Regex aus den match-Armen *derselben Funktionsdatei* scrapt (inkl. Abhängigkeit von exakt 8 Leerzeichen Einrückung des `_ => {}`-Arms), und OR-t dann 13 handgeschriebene Prädikate dazu. Code → Scrape → JSON → include_str → Laufzeit-Kontrollfluss. Bei stalem JSON routet die Laufzeit-Klassifikation anders als die match-Arme darunter; ein Parse-Fehler wird per `.ok().unwrap_or_default()` zur leeren Menge — dann fallen *alle* exakten Control-Commands still in die Queue (Doppel-Effekte nach Diagnose-Modus 6).

`[HIGH] snapshot:21481-24134 — 2.654 Zeilen Gottfunktion als Kopplungszentrum`
Accept, Claim, Idempotenz, Recovery-Resume, Session, Policy, Dispatch über ~40 Kommandofamilien (mailserver inkl. Direkt-SQL auf stalwart_users/stalwart_mailboxes, snapshot:24080-24086) und Outcome-Persistenz in einer Funktion. Jede neue Domäne vergrößert sie; die Inventory deklariert sie offiziell als „authoritative_router". Im Feld: jede Änderung an einer Domäne riskiert Regressionen in allen anderen; Review unmöglich fokussiert.

`[MED] snapshot:29214-29232 — Send-Block-Klassifikator parst, was er abschaffen soll`
Der Doc-Kommentar verspricht stabile Reason-Codes „statt Free-Form-Text zu parsen" — die Funktion selbst parst Free-Form-Text per `lowered.contains(...)`. Präzedenz ist fragil: ein Fehler mit „limit" und „approval" wird `sender_limit_exhausted`, `"approv"` matched vor `recipient_address`. Der Code wird auf Message *und* Engagement persistiert und steuert Retry-UX; Fehlklassifikation zeigt dem Nutzer den falschen Behebungspfad.

`[MED] snapshot:38711-38718 — Substring-„Idempotenz" verschluckt Nutzer-Edits`
`if existing_text.contains(markdown_to_append.trim())` → `section_already_present`, Edit übersprungen. Jede neue Notiz, die zufällig Teilstring des bestehenden Dokuments ist, wird lautlos fallengelassen — kein Fehler, kein Hinweis, kein Command-Failure. Gehört zusammen mit snapshot:38905-38986: Natural-Language-Intent-Parsing (deutsch/englisch, `contains("ergänz")`, `"add "`) als Edit-Trigger mitten im Kern-Store.

`[MED] snapshot:36044-36290 — Tote Fallback-DOCX-Kette (~250 Zeilen)`
`ensure_generated_docx_exists` hat null Aufrufer (einziger Treffer ist die Definition); damit sind `build_fallback_report_docx`, `render_fallback_document_xml`, `fallback_docx_table/styles/numbering_xml` komplett tot. Der Zweck wäre zudem bedenklich: fehlendes Agenten-Artefakt durch fabriziertes Platzhalter-DOCX ersetzen statt zu scheitern — „falsch schreiben" als Design.

`[MED] snapshot:44407-44418 — Schema-Migration per CREATE-TABLE-Text-Sniffing`
`migrate_business_users_roles` erkennt das Alt-Schema an `table_sql.contains("'admin', 'user'")`. Jede Alt-DB, deren CHECK-Rollenliste anders sortiert/formatiert ist (`'user', 'admin'`), wird still übersprungen — die Rollenmigration läuft nie, ohne Fehler, ohne Log. Im Feld: dauerhaft driftende Schemata auf Alt-Installationen, die spätere Grant-Logik falsch auswertet.

`[MED] snapshot:39628-39652, 41482-41492, 38623-38636 — Vier parallele Status-Vokabulare, generated Contract ignoriert`
`command_status_for_queue_route_status`, `projection_route_status_for_command_status`, `projection_status_is_active`, `normalize_queue_status`, `normalize_business_chat_tracking_status` pflegen je eigene Statuslisten (inkl. `"erledigt"` und `"canceled"`-Varianten). `command_lifecycle_generated.rs` existiert (TERMINAL_STATUSES, EXECUTION_PHASES) und wird von command_lifecycle.rs/rxdb_peer.rs genutzt — von store.rs aber null. Exakt der G8-Befund der Command-Bus-Diagnose, hier auf der nativen Seite belegt.

`[MED] snapshot:27732-27800, 21286-21306, 43853-43888 — Store greift in zwei fremde SQLite-Schemata`
Der Store öffnet und konfiguriert die Mailserver-DB samt Pragmas (21286-21306), reconciliert `outbound_messages` direkt gegen `stalwart_smtp_delivery_log` (27790) und spiegelt `iot_datapoints`-Zeilen aus der Core-DB, „which the projector cannot re-derive" (43853). Drei Parallelwahrheits-Kopien über DB-Grenzen, geklebt über implizite Fremdschema-Kenntnis; ein Stalwart-Schema-Update bricht outbound still (`unwrap_or(None)` auf der Log-Query).

`[MED] snapshot:6452-6453, 19401, 9210, 7616, 29766, 39828 — Repair/Legacy/Fallback ist institutionalisiert, nicht abgebaut`
~24 repair-/reconcile-/legacy-/backfill-Funktionen: zwei Backfills laufen bei *jedem* Startup (6452-6453: Preview-Grants, SemVer-Releases), Queue-Repair per Operator-CLI (`ctox business-os repair queue-projections`, service/business_os.rs:557), Lifecycle-Repair als Kommando, Grant-Migration, Inline-Artefakt-Redaktion, Provider-Reconcile, Shadow-Reconcile. Das ist exakt die Schuldklasse „falsch schreiben → später reparieren" des Schwestermoduls — hier mit sauberer Dry-Run-Disziplin, aber ohne Abbau-Pfad; die Backfills haben kein Sunset-Kriterium.

`[LOW] snapshot:41428-41472, 43775-43782, 5015-5019 — Kleinere Text-Dispatches`
Skill-Routing per `command_type.contains("knowledge"|"runbook"|"scoring")` (41428-41453) und `is_source_parse_command` per contains (41465) statt Registry; Redaktions-Keywordliste matched „token" auch in „tokenizer" (43775-43782); CSS-Sanitizer als Substring-Blacklist (5015-5019). Einzeln vertretbar, in Summe dasselbe Muster.

## healthiest_aspects

- snapshot:21343-21370 — `ActiveExternalSqlControlCommand`: RAII-Guard mit Drop und poison-tolerantem Mutex; In-Flight-Dedup von External-SQL-Commands ist korrekt und aufraumsicher gelöst.
- snapshot:29240-29303 — `outbound_record_send_failure`: Approved-Draft bleibt retryable, Reason/Attempt/Timestamp strukturiert persistiert, Spiegelung aufs Engagement — vorbildliche Failure-Semantik (abgesehen vom textbasierten Klassifikator darüber).
- snapshot:135-222 + 60517-60729 — Compile-time-Testinstrumentierung (Column-Load-/Writer-Open-/Transaktions-Zähler) mit Assertions wie „must not run PRAGMA table_info per upsert": Performance-Regressionen sind test-gepinnt.
- snapshot:23548-23620 — Saga-Schritte mit Claim/Kompensation für Modul-Sichtbarkeit (inkl. `saga_compensation_failed`-Aggregation); dazu die Dry-Run/Action-List-Disziplin aller Repair-Funktionen (9210-9300, 19401-19430) und die idempotenten Spalten-Migrationen mit Capability-Epoch-Triggern (44450-44520).

## coupling

Der Fokusbereich ist das Kopplungszentrum schlechthin. Nach außen (sauber bis vertretbar): `super::office_engine` (25), `ats_gates` (10), `external_sql_sync` (9), `threads`/`support`/`rxdb_peer` (je 5) — Richtung ok, aber office_engine zusätzlich per Fehlertext-Kontrakt (Befund 1). Grenzverletzend: direktes Öffnen/Konfigurieren der Mailserver-DB (stalwart-Schema-Kenntnis im Store, 21286, 24080, 27790), Core-DB-Lese-Spiegel für IoT (43853), Channel-DB-Zugriffe (26625, 28396). Fachlich fehlplatziert im Store: WebRTC/TURN/ICE-Infra (1710-2260), DKIM-Keygen (29553-29616), CSS-Sanitizer (5015), PDF-Textextraktion (40621-40684), ZIP/DOCX-Bau (tot, 36044+), Natural-Language-Edit-Parsing (38877-38986). **Natürlicher Schnitt** (12 Module): (1) store-core: Connection/Cache/Schema/Migration (125-1600, 43918-44620); (2) command-bus: accept/record_command/claim/outcome (13221-13377, 21343-24134, 30259-30398); (3) policy/session/capability (2261-2900, 29659-30079); (4) queue-projection + Status-Mapping (38560-40015, 41475-41500); (5) module-lifecycle (3269-10600); (6-10) Domänen outbound (24149-29616, ~5.500 Zeilen), customers (31055-32713), ATS (36753-38100), appsec (41726-43810), documents/office (32714-36043, 39088-39620); (11) sync/TURN-Infra (1710-2260); (12) repair-ops (alle Repair-/Backfill-Funktionen gesammelt, mit Sunset-Tests).

## test_coverage

Das Testmodul (ab snapshot:44621, 288 Tests, ~23.700 Zeilen) deckt den Fokus breit, aber asymmetrisch: **Verhaltenstests** existieren für Repair dry-run/apply-Paare (58583-59009), Terminal-Note-Ack über den wohlverhaltenen Marker-Pfad (58784), Send-Gate-Suppression-Reason (63931-64545), Capability-Epoch (47485) und Parallel-Send-Cap unter Last (64380-64420). **Lücken:** kein Direkttest der Terminal-Notiz-Klassifikatoren für Edge-Cases (nur der „business-os:terminal-success"-Pfad), kein Test der Präzedenz in `outbound_classify_send_block`, kein Test für die Rollen-Migrations-Reorder-Fragilität, und der zirkuläre Inventory-Pfad (stales JSON → Misrouting) ist nur extern per Smoke (`command-type-inventory-smoke.mjs`) gepinnt, nicht im Modul. **Impl-Pins:** ~40 Assertions der Form `error.to_string().contains(...)` (44662-46124, 59891-60093) ossifizieren genau die fragilen Textkontrakte, von denen der Kontrollfluss abhängt — sie sichern das Verhalten, verhaken aber die Schuld; die Test-infrastruktur spiegelt das Muster sogar selbst (64397: Test dispatchet auf `msg.contains("daily limit")`/`"locked"`).

