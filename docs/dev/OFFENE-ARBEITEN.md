# Offene Arbeiten — Sync-Engine, Refactoring, Performance

Stand: 14.08.2026. Alle Zahlen sind selbst gemessen; wo eine Zahl fehlt, steht
„ungemessen" statt einer Schätzung.

---

## 0. Der Befund, der alles andere überlagert

**Die App wird nicht langsamer, weil Code langsam ist, sondern weil die
transportierte Datenmenge wächst und niemand sie begrenzt.**

Auf der Kundeninstanz liegen (gemessen am 14.08.):

| Collection | Dokumente |
|---|---:|
| `sellify_activities` | 139.804 |
| `sellify_campaigns` | 86.551 |
| `sellify_people` | 60.640 |
| `sellify_companies` | 17.520 |
| **Summe Sellify** | **304.515** |

Bedarfsgesteuert repliziert werden im ganzen System **genau drei** Collections,
und alle drei sind Dateiblöcke: `desktop_file_chunks`, `document_blob_chunks`,
`spreadsheet_blob_chunks` (`shared/command-bus.js:107`). Für alles andere gilt
**Vollreplikation**. Ein bounded initial window existiert nirgends — Suche nach
`initialPullWindow`, `since_ms`, `maxInitialDocs` im gesamten Sync-Kern: **kein
Treffer**.

Bei Batchgröße 20 (`shared/sync-contract.js:48`) sind das **~15.225 Rundreisen**
allein für Sellify. Jede neue Aktivität verlängert jeden künftigen Erstsync.
Das ist die Erklärung für „immer langsamer statt schneller".

**Wichtig zur Einordnung meiner bisherigen Zahlen:** Die vielzitierten
„15,1 s Boot" wurden auf einem **leeren** Speicher gemessen und beschreiben den
Kundenfall **nicht**. Das Desktop-Modul ist nachweislich korrekt client-first
(liest lokal, rendert sofort). Meine bisherigen Optimierungen zielten auf
Taktraten und Backoffs — also auf Sekunden — während der bestimmende Faktor die
Datenmenge ist.

### OA-1 (P0) — Begrenztes Erstfenster + Bedarfsabruf für große Collections
Die Mechanik existiert bereits (`leaseCollection`, Demand-Query-Loader) und wird
für Geschäfts-Collections nur nicht benutzt. **Fünfter Fall desselben Musters:
die Reparatur ist da, sie wird nicht gerufen.**
- Erstsync auf ein begrenztes Fenster (z. B. neueste N nach `updated_at_ms`),
  Rest über den vorhandenen Demand-Pfad.
- Eigentümer-Konflikt beachten: der Browser-Pull-Pfad (`shared/sync.js`,
  `replication-webrtc.mjs`) gehört der Parallelsitzung; der native Master
  (`src/core/rxdb/**`) gehört dieser Spur. Muss abgestimmt werden.
- **Abnahme:** Rundreisen bis „Modul benutzbar" auf einem Store mit ≥100.000
  Dokumenten, vorher/nachher. Nicht: Tests grün.

### OA-2 (P0) — Messung auf populiertem Store
Es gibt bis heute **keine einzige** Messung auf realistischer Datenmenge. Alle
M1–M5-Zahlen stammen von leeren Instanzen.
- Store der Kundeninstanz kopieren oder synthetisch füllen (≥300.000 Dokumente).
- Dann: Zeit bis erster sichtbarer Inhalt je Modul, Rundreisen, IndexedDB-Größe.

---

## 1. Refactoring

### Erreicht (gemessen)
| Modul | vorher | jetzt |
|---|---:|---:|
| `rxdb_peer.rs` | 25.292 | **8.897** |
| `office_engine.rs` | 13.953 | 13.561 |
| `channels/mod.rs` | 7.221 | 7.035 |
| `service/business_os.rs` | 7.106 | 6.060 |
| `lcm/mod.rs` | 5.627 | 5.310 |
| `store_outbound_commands.rs` | 5.270 | 5.108 |

Sieben Module aus `rxdb_peer.rs` geschnitten. Kein Budget je angehoben, alle
Wächter grün, `main` baut aus der Historie, Arbeitsbaum leer.

### OA-3 (P1) — Die zwei verbliebenen Riesen
| Datei | Produktionszeilen |
|---|---:|
| `store.rs` | **27.342** |
| `service.rs` | **26.244** |
| `app.js` | **12.388** (physisch) |

Hier ist **nichts** Nennenswertes passiert (−174 bzw. −12 Zeilen). Das ist der
berechtigte Teil der Kritik „an der Codequalität sehe ich keine Fortschritte":
Wer ins Repo schaut, sieht zwei Dateien mit 27.000 und 26.000 Zeilen.
- Naht-Kandidaten in `store.rs`: Projektionen, Command-Handler, App-Runtime.
- `service.rs`: Queue-Dispatch, Recovery, Prompt-Aufbau.
- **Abnahme:** reiner Move mit Hash-Beweis je Funktion, Budgets nur gesenkt.

---

## 2. Performance — Bilanz der sechs geplanten Hebel

| Hebel | Ergebnis |
|---|---|
| P1 Keep-Alive | negativ / blockiert |
| P2 doc_cache-Deckel | **eingelöst** (fa1a18ab6) |
| P3 | durch Messung **widerlegt** |
| P4 | durch Messung **widerlegt** |
| P5 Batch-Matrix | existierte bereits |
| P6 | nur unter Churn relevant |
| Parallelstart (Parallelsitzung) | gemessen Faktor **1,25**, nicht 4 |
| Consumer-Backoff-Fix | Latenz **unverändert** |

**Für den Nutzer spürbar besser geworden ist: nichts.**

### Offene, gemessene Werte
| Größe | Wert | Ziel |
|---|---|---|
| Command-Roundtrip p50 | 1,50 s | < 300 ms warm |
| Command-Roundtrip p95 | 6,2 s | — |
| Boot bis alle Collections live (leerer Store) | 15,1 s | < 5 s |
| Bridge-Handshake je Collection | 0,5–2,3 s | Rundreisen senken |

### OA-4 (P1) — Etappenmessung fahren
Die Instrumentierung ist gelandet (AP1, cfc84ad61): sieben Zeitmarken je
Command, opt-in per `command_timing_probe`. **Gefahren wurde sie noch nicht.**
Erst danach dürfen AP5–AP7 (Terminalstatus ereignisnah, Intake-Wake,
Ein-Projektion-Abschluss) angefasst werden — sonst wieder Optimierung auf
Verdacht.

### OA-5 (P1) — Bridge-Handshake-Rundreisen senken
Von der Parallelsitzung als Haupthebel für den Boot identifiziert und gemessen
begründet. Liegt in `replication-webrtc.mjs` / `webrtc-native.mjs` (deren Spur).

---

## 3. Kundeninstanz

### Erledigt und beim Kunden wirksam (14.08.)
- **Schreibschleife 174 → 0 Rev/min**, Kontrolle nach 15 min ebenfalls 0.
- **Grabsteine ~302.000 → 275** (Fix der Parallelsitzung, mit meinem Bau
  ausgeliefert).
- Beides unabhängig gegengeprüft.

### OA-6 (P0) — Die Schleife läuft weiter, nur ohne Schreibvorgänge
Der Daemon verbrennt **58 % eines Kerns** im Dauerzustand (alter Daemon im
Mittel 11 %). Der Sweep läuft **29×/min, kein einziger Leerlauf**, greift jedes
Mal dieselben **6** Dokumente:

| command_type | status | Anzahl |
|---|---|---:|
| `web_stack.person_research` | failed | 4 |
| `outbound.research_source.generate_adapter` | failed | 1 |
| `outbound.research_source.test` | failed | 1 |

Meine Sperre feuert **nie**: sie verlangt `exhausted = 1`, und **keine** der 12
Intake-Failure-Zeilen hat das (Versuche stehen bei 1–2, Budget 5 nie erreicht).
Ich habe das Symptom behandelt, nicht die Ursache.
- **Richtiger Fix:** Dokumente müssen das Kandidatenfenster verlassen können —
  echter Versuchszähler mit Aufgabe, oder `terminal_status` beim endgültigen
  Scheitern. **Kein weiterer Deckel auf das Symptom.**
- **Abnahme:** CPU-Dauerlast < 5 %, `idle_ticks` > 0.

### OA-7 (P2) — Store-Datei kompaktieren
2.287.198.208 Bytes, unverändert. SQLite gibt gelöschte Seiten erst mit `VACUUM`
zurück — blockierender Schritt auf laufender Instanz, eigenes Wartungsfenster.

### OA-8 (P2) — `replicationUp` dauerhaft false
In **allen** Messpunkten beider Sitzungen. Ursache belegt:
`dataChannelOpen: false` bei `poolCreated`/`signalingJoinAccepted` = true — es
war schlicht kein Browser verbunden. Der Wert ist also ehrlich. Offen bleibt:
ist das „niemand schaut hin" oder „niemand *kann* sich verbinden"?

### OA-9 (P2) — Wirksamkeitsnachweis AP3 auf der Instanz
Die ehrliche Gesundheitsmeldung (b8fbf718f) ist ausgeliefert, aber auf der
Kundeninstanz nie unter Last erlebt worden. Braucht eine verbundene Sitzung.

---

## 4. Symptome der Parallelsitzung (deren Spur)

| # | Symptom | Stand |
|---|---|---|
| 1 | Boot 10–15 s | offen, Hebel = Handshake-Rundreisen |
| 2 | Replikation hängt bis zum Neuladen (`masterChangesSince`-Timeout) | offen |
| 3 | Schreibschleife | **erledigt** |
| 4 | Peer meldet Gesundheit, während nichts fließt | im Repo behoben (AP3) |

---

## 5. Methodische Regeln, die diese Kampagne erzwungen hat

1. **Beweiskette in drei Stufen:** im Repo gelandet → beim Kunden ausgeliefert →
   auf der Instanz gemessen wirksam. Keine Stufe ersetzt eine andere.
2. **Verbrauchte Cache-Buster-Namen sind dauerhaft verbrannt** — ein einziger
   Messabruf verbrennt einen Namen.
3. **Drei Fragen vor jedem Befund:** Gibt es das Symbol jetzt im Baum (`grep`)?
   Gab es das je in der Historie (`git log -S`)? Ist der Stand in einer Minute
   noch derselbe? (Bei zwei Sitzungen an einem Arbeitsbaum: nein.)
4. **Testläufe nur mit Trefferzahl zitieren.** `--exact` mit Kurznamen liefert
   „0 passed" und sieht aus wie Erfolg.
5. **`TMPDIR` mit umlenken**, nicht nur `CARGO_TARGET_DIR` — sonst stirbt der
   Bau an der vollen Systemplatte.
6. **Auf leerem Store gemessene Zahlen sagen nichts über den Kundenfall.**

---

## 6. Reihenfolge, die ich vorschlage

1. **OA-6** — CPU-Dauerlast beim Kunden (läuft seit heute, wächst nicht mit).
2. **OA-2** — Messung auf populiertem Store; ohne sie ist jede weitere
   Optimierung geraten.
3. **OA-1** — begrenztes Erstfenster; das ist der Hebel für „client first".
4. **OA-4** — Etappenmessung fahren, dann erst AP5–AP7.
5. **OA-3** — `store.rs` und `service.rs` schneiden.
