## 1. Datei-Liste

Die reproduzierbare Minimalmenge umfasst **13 Dateien**; keine tracked Datei muss `dirty-ganz` übernommen werden.

1. **`Cargo.toml` — Hunks reichen (HEAD-Zeile 25):** Aktiviert für `ctox-cliproxyapi` ausschließlich `anthropic-fingerprint-transport` und `antigravity-http-transport` zusätzlich zum bestehenden Codex-Transport.
2. **`Cargo.lock` — Hunks reichen (HEAD-Zeilen 2119, 2383 und 4551):** Ergänzt die aufgelösten Root-/CLIProxyAPI-Abhängigkeiten und den `gjson`-Packageblock; das opportunistische Upgrade von `libssh2-sys` wurde ausdrücklich ausgeschlossen.
3. **`src/core/execution/mod.rs` — Hunks reichen (HEAD-Zeile 3):** Registriert `cliproxyapi_integration` als Execution-Modul.
4. **`src/core/execution/cliproxyapi_integration/mod.rs` — untracked-ganz:** Definiert die typisierte, credential-freie Konfiguration und Validierung für Kimi-Coding-Subscription-Accounts.
5. **`src/core/execution/cliproxyapi_integration/kimi_host.rs` — untracked-ganz:** Implementiert Laden, Installation, Routing und Entfernung der Kimi-Subscription samt Secret-Referenzen und CLIProxyAPI-Topologie.
6. **`src/core/secrets.rs` — Hunks reichen (HEAD-Zeilen 33 und 169):** Ergänzt nur `credential_lifecycle_guard`, `SecretRecordWrite` und die transaktionalen APIs `read_secret_values`, `write_secret_records` und `delete_secret_records`; Master-Key-Migration, Rechteänderungen, Kimi-Katalogzeile und Tests wurden nicht übernommen.
7. **`src/core/execution/responses/gateway.rs` — Hunks reichen (HEAD-Zeile 21):** Fügt `MainResponsesGatewayPhase`, den Statusdatentyp sowie Getter und Setter des pro Root gehaltenen Gateway-Status hinzu.
8. **`src/core/business_os/store_outbound_commands.rs` — Hunks reichen (HEAD-Zeilen 754, 4151 und 4226):** Ergänzt HTML-Link-/Pixel-Tracking, Token-Erzeugung und `record_mail_tracking_event`; die unabhängige `delivered_at_ms`-Änderung wurde ausgeschlossen.
9. **`src/core/mailserver/src/config.rs` — Hunks reichen (HEAD-Zeile 2):** Definiert persistierbare Mailserver-Laufzeiteinstellungen einschließlich Tracking-Basis-URL und deren Umwandlung in `StalwartConfig`.
10. **`src/core/mailserver/src/store/sqlite.rs` — Hunks reichen (HEAD-Zeilen 1, 57 und 105):** Ergänzt Tracking-Token-Datentyp sowie Lade-/Speicher- und Tracking-Event-Methoden des `SqliteStore`; die neuen Tests wurden ausgeschlossen.
11. **`src/core/mailserver/src/store/sqlite_schema.rs` — Hunks reichen (HEAD-Zeile 35):** Legt Tabellen und Indizes für Mailserver-Laufzeitkonfiguration, Tracking-Tokens und append-only Tracking-Events an.
12. **`src/core/service/service.rs` — Hunks reichen (47 Produktionshunks mit HEAD-Startzeilen 996, 2196, 3334, 3358, 3415, 3624, 3648, 3705, 4111, 4138, 5963, 6287, 7347, 7394, 7429, 7725, 8398, 8643, 11060, 11093, 11330, 11447, 11609, 12002, 15094, 15201, 15905, 18304, 18670, 18702, 18727, 18831, 19094, 19214, 19290, 19354, 19380, 19391, 19429, 19458, 19494, 19847, 20839, 25175, 25184, 25202 und 25331):** Migriert Business-OS-App-Erkennung von Prompt-Markern auf typisierte Queue-Metadaten und behebt die bereits in HEAD vorhandenen Variablen-, Rückgabewert- und Signaturinkonsistenzen; Supervisor-Autostart, unabhängige Projection-Änderung und sämtliche Test-/Fixture-Hunks wurden ausgeschlossen.
13. **`src/core/mission/channels/mod.rs` — Hunks reichen (HEAD-Zeilen 283 und 6440):** Fügt `metadata: Value` zu `QueueTaskView` hinzu und übernimmt beim Laden die Nachrichtenmetadaten; der zusätzliche Testblock wurde ausgeschlossen.

Nicht benötigt wurden insbesondere `src/core/business_os/desktop_files.rs`, `src/core/mission/queue.rs`, `src/core/mailserver/src/lib.rs`, die ganze dirty `Cargo.toml`-Fassung sowie fünf Dokument-/Gate-/JSON-Dateien aus dem untracked Integrationsverzeichnis.

## 2. Beweis

Der exakte 13-Dateien-Patch wurde auf einen **zweiten frischen `git archive HEAD`** angewandt; danach wurde nur das vorgeschriebene `src/core/coding_agents/pi-sidecar/dist` hineinkopiert.

```text
Scratch:  /Volumes/tmp/ctox-pipeline/heal-replay
Command:  cargo check --locked --bin ctox
Result:   Exit 0
Cargo:    Finished `dev` profile
Warnings: `ctox` generated 593 warnings
```

Zusätzlich erschienen zwei Build-Script-Warnungen; das Rohlog enthält damit 596 `warning:`-Header, während `greppy bash-smart` einschließlich Zusammenfassungsmarkern 599 Warnungsmarker zählt.

Artefakte:

- Exakter Patch: `/Volumes/tmp/ctox-pipeline/heal-minimal.patch`
- Patch-SHA-256: `44f72e333f175016cd80027d133053b12ea6e542fbe2333917f5adba067be823`
- Replay-Buildlog: `/Volumes/tmp/ctox-pipeline/heal-replay-cargo-check.log`
- Fortschrittsjournal: `/Volumes/tmp/ctox-pipeline/heal-fortschritt.md`
- Checkout wurde nicht verändert; kein `git add` und kein Commit.

## 3. Delta-Risiken

- **Service-Semantik:** Die 47 `service.rs`-Hunks ändern Business-OS-App-Erkennung substanziell von heuristischen Prompt-Markern auf strukturierte Queue-Metadaten, einschließlich SQL-Abfragen, Recovery, Leasing und Validierungsrouting; das ist die größte fremde Semantik-Adoption.
- **Mail-Tracking:** Outbound-HTML wird verändert, Links werden umgeschrieben und ein unsichtbares Pixel wird eingefügt; das berührt Datenschutz, Einwilligung, Zustellbarkeit und Security-Review und ist mehr als nur die fehlende Compilerfunktion.
- **Persistenz:** Drei neue Mailserver-Tabellen und dazugehörige Token-/Event-APIs erweitern das gemeinsame SQLite-Schema dauerhaft.
- **Provider-Semantik:** Das untracked Integrationsmodul führt Kimi-Coding-Subscription-Routing und Credential-Topologien als neues Produktverhalten ein.
- **Dependency-Oberfläche:** Die beiden CLIProxyAPI-Features aktivieren eine größere Transport-/Krypto-/Kompressions-Abhängigkeitsmenge; der Patch übernimmt aber ausdrücklich nicht die dirty OpenSSL→Rustls-, `git2`- oder Workspace-Exclude-Umstellung.
- **Secrets:** Die übernommenen APIs ermöglichen atomare Credential-Tupel-Mutationen und Löschungen; die wesentlich fremdere dirty Master-Key-Dateimigration wurde dagegen ausgeschlossen.
- **Queue-Metadaten:** `QueueTaskView` transportiert nun die vollständigen Message-Metadaten weiter, was Speicherverbrauch und interne Datenexposition leicht erhöht.

## 4. Offene Bedenken

- Bewiesen ist ausschließlich der verlangte Binär-Build; `cargo check --workspace` wurde nicht ausgeführt.
- Test-/Fixture-Hunks wurden aus Gründen der Bin-Build-Minimalität ausgeschlossen, daher kann `cargo test` beziehungsweise `cargo check --tests` wegen fehlender neuer Metadatenfelder in alten Testkonstruktoren scheitern.
- Die fünf ausgeschlossenen Dateien `README.md`, `provider-integration.json` sowie die drei Gate-Skripte sind für den Rust-Build nicht referenziert, werden aber für den dokumentierten Track-B-Integrationsprozess benötigt.
- `pi-sidecar/dist` ist wie vorgeschrieben nur ein externer Scratch-Buildbestandteil und nicht Teil des 13-Dateien-Patches.
- Die im älteren Beweis genannte `desktop_file_chunk_completeness`-Fehlerklasse trat im aktuellen archivierten HEAD nicht auf; die Datei war für den reproduzierten Exit 0 nicht erforderlich.
- Die bereits vorhandene Warnungsmenge bleibt sehr hoch: 593 Warnungen im `ctox`-Binary.

Workjet-Completion-Receipt v1
status: completed
checkout_mutated: false
commit_created: false
deliverable: /Volumes/tmp/ctox-pipeline/heal-minimal.patch
deliverable_sha256: 44f72e333f175016cd80027d133053b12ea6e542fbe2333917f5adba067be823
proof_command: cargo check --locked --bin ctox
proof_exit: 0
proof_warnings_ctox: 593
proof_log: /Volumes/tmp/ctox-pipeline/heal-replay-cargo-check.log
