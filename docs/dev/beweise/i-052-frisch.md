# I-052 Report

## was_geaendert

- In `src/core/service/service.rs:24623` zentralisiert `record_queue_ack_failure_durably` fehlgeschlagene Queue-Acks als bestehende Harness-Flow-Ereignisse (`event_kind = queue.ack_failed`).
- Fuer jeden betroffenen Queue-Key wird ein eigener Eintrag mit gesetztem `message_key` geschrieben. Der Eintrag traegt Ack-Operation und Fehler sowohl lesbar als auch strukturiert in `metadata_json`.
- `record_queue_ack_and_refresh_business_os_projections_locked` schreibt einen Ack-Fehler nicht mehr in `shared.recent_events`, sondern benutzt den dauerhaften Pfad (`service.rs:24661`).
- Der Working-Hours-/Busy-Handoff in `enqueue_prompt` benutzt bei einem fehlgeschlagenen Ruecksetzen auf `pending` denselben dauerhaften Pfad (`service.rs:19509`).
- Neuer Regressionstest `queue_ack_failure_is_recorded_durably_with_message_key`: prueft die DB-Zeile, `event_kind`, Text, Metadaten, den konkreten `message_key` und dass der Prozess-Ringpuffer leer bleibt.
- Der bestehende Projektions-Test wurde zu `record_queue_ack_refreshes_business_os_command_projection` umbenannt, damit der vorgeschriebene enge Filter genau einen aussagekraeftigen Test trifft.

## ursache_belegt

- Vorher endete der Fehlerzweig in `record_queue_ack_and_refresh_business_os_projections_locked` nach `push_event_locked`; der Ringpuffer ist auf 24 Eintraege begrenzt und nicht persistent.
- Die Gegenprobe stellte genau dieses Verhalten wieder her: der Fehler wurde nur nach `shared.recent_events` geschrieben. Der neue Test wurde daraufhin rot und zeigte den Ringpufferinhalt im Assertion-Fehler.
- Nach Wiederherstellung des Fixes ist derselbe Test gruen und liest den Eintrag aus `ctox_harness_flow_events` anhand des `message_key`.

## kompensationen_geloescht

- Keine.
- Der generische TTL-Sweep wurde ausdruecklich nicht entfernt.
- Die Business-OS-App-Recovery wurde ebenfalls nicht entfernt, weil Punkt 2 noch offen ist und ein Ack-Fehler die Lease weiterhin physisch auf `leased` stehen lassen kann.

## verblieben

- Punkt 2 (In-Memory-Ownership erst nach erfolgreichem durablem Ack freigeben) bleibt offen.
- Begruendung: Im Worker-Abschlusspfad werden alle Keys bereits gemeinsam in `service.rs:6702` freigegeben, waehrend die status- und dispositionabhaengigen Queue-Acks erst spaeter in vielen getrennten Zweigen erfolgen. Ein korrektes Verschieben erfordert Ack-Erfolg pro Key statt des heutigen pauschalen `leases_released`-Zustands; andernfalls wuerden Teilerfolge, Fehlerpfade und `PromptWorkerActivity::Drop` falsch behandelt.
- Im Lease-Handoff wird bei Working-Hours-/Busy-Release vor dem `pending`-Ack gar keine langlebige Ownership fuer den Fehlerfall registriert. Das bestehende Live-Owner-Praedikat koppelt den Set-Eintrag zudem an `busy`, aktive Worker oder `durable_queue_lease_in_progress`. Ein lokales Umstellen nur der Reihenfolge wuerde daher keine verlaessliche Ownership nach Ack-Fehler erhalten.
- Folge: `recover_stale_business_os_app_queue_tasks` (`service.rs:18519`), die Worker-Finalization-Recovery (`service.rs:18747`) und `release_stale_queue_task_leases` (`src/core/mission/channels/mod.rs:3498`) bleiben als Netz erforderlich.

## tests

Alle Cargo-Aufrufe verwendeten `CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline-targets/I-052`.

- `cargo fmt --check`: erfolgreich; dieser Befehl erzeugt keine `test result`-Zeile und fuehrt 0 Tests aus.
- `cargo check --bin ctox`: erfolgreich; dieser Befehl erzeugt keine `test result`-Zeile und fuehrt 0 Tests aus.
- Pflicht nach Funktionsaenderung: `cargo check --bin ctox --tests`: erfolgreich (`Finished dev profile`); Cargo Check erzeugt keine `test result`-Zeile und fuehrt 0 Tests aus.
- `cargo test --bin ctox queue_ack_failure` — Filtertreffer: 1:
  `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2735 filtered out; finished in 0.01s`
- `cargo test --bin ctox record_queue_ack` — Filtertreffer: 1:
  `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2735 filtered out; finished in 0.12s`
- `git diff --check`: erfolgreich.

## gegenprobe

- Dauerhaften Aufruf im Fehlerzweig von `record_queue_ack_and_refresh_business_os_projections_locked` temporaer durch den alten, weiterhin kompilierenden `push_event_locked`-Pfad ersetzt.
- `cargo test --bin ctox queue_ack_failure` wurde wie gefordert rot:
  `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 2735 filtered out; finished in 0.06s`
- Assertion-Beleg: `queue ack failure evidence must not live only in the process ring buffer: ["Failed to ack handled queue lease(s) after completion: database is locked"]`.
- Danach die gesicherte Fix-Datei exakt zurueckkopiert und den Test erneut gruen ausgefuehrt.
- `git diff --stat` nach Wiederherstellung:
  `src/core/service/service.rs | 103 ++++++++++++++++++++++++++++++++++++++------`
  `1 file changed, 89 insertions(+), 14 deletions(-)`

## offene_bedenken

- `record_harness_flow_event_lossy` ist der vom Auftrag vorgegebene bestehende Persistenzweg. Bei einem SQLite-Schreibfehler kann auch dieser forensische Eintrag verworfen werden; der bestehende Implementierungsvertrag zaehlt und protokolliert solche Drops auf stderr, garantiert aber keinen Retry.
- Solange Punkt 2 offen ist, ist der Defekt von „unsichtbar“ zu „persistent messbar“ repariert, nicht aber die verwaiste Lease selbst automatisch geheilt. Deshalb sind die Kompensationen noch nicht ueberfluessig.
- Keine vorbestehend roten gezielten Tests beobachtet.

## pfade

Geaendert (Hard Whitelist eingehalten):

- `src/core/service/service.rs:19509` — dauerhafte Ack-Fehlererfassung im Lease-Handoff.
- `src/core/service/service.rs:24623` — gemeinsamer dauerhafter Queue-Ack-Fehlerpfad.
- `src/core/service/service.rs:24661` — Worker-Abschlusspfad nutzt den dauerhaften Fehlerpfad.
- `src/core/service/service.rs:35836` — Regressionstest mit DB- und `message_key`-Nachweis.

Fuer eine Folgewelle relevant, aber nicht geaendert:

- `src/core/service/service.rs:6702` — pauschale In-Memory-Freigabe vor den spaeteren Ack-Zweigen.
- `src/core/service/service.rs:18519` — Business-OS-App-Stale-Recovery bleibt erforderlich.
- `src/core/service/service.rs:18747` — Recovery nach Worker-Finalization bleibt erforderlich.
- `src/core/mission/channels/mod.rs:3498` — generischer TTL-Sweep; ausserhalb der Whitelist und absichtlich unveraendert.
