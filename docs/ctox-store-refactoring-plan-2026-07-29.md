# SYNC-C: store.rs → A-Grade (v1, zur adversarialen Prüfung)

Basis: `ctox-store-review-2026-07-29.md` (P7a–f, 15 HIGH). Gilt unter dem
A-Grade-Masterplan (Feature-Freeze, Schnitt-zuerst-Prinzip, Workjet-Regeln,
globale A-Definition). Ist-Zustand: 68.359 Zeilen, 1.052 Produktions-fns,
Noten C−/C−/C+/B−/C−/D+.

**Gate:** store.rs ist aktuell fremd-in-Arbeit. Jede Welle prüft vor Start
`git status` auf die Whitelist; Reviews/Analysen laufen gegen HEAD-Snapshots.

## Welle SEC — Sicherheits-Sofortmaßnahmen (VOR dem Schnitt, klein & gezielt)

| ID | Ticket | Befund |
|---|---|---|
| SEC1 ▲ | Eskalations-Guard vervollständigen: ein `users.manage`-Grant darf keine admin-User minten oder auf admin heben — admin wird wie chef behandelt; Regressionstest beweist Ablehnung | P7d, snapshot:18254-18264 |
| SEC2 ▲ | Backup-Schlüsselhygiene: Portable-Key darf ab dem zweiten Drill NICHT im Artefakt reisen; HMAC-Signaturschlüssel nicht im Backup selbst (Disaster-Fall sonst unverifizierbar). Schlüsselablage getrennt vom Artefaktpfad; Drill-Test prüft Abwesenheit | P7e, snapshot:16433-16467, 15990-16011 |
| SEC3 ▲ | Token-Ausstellung entkoppeln: stille User-Provisionierung/-Reaktivierung raus aus dem Ausstellungspfad — explizite, auditierte Provisionierung; Ausstellung schlägt bei fehlendem User strukturiert fehl | P7d, snapshot:29947-29981 |

Alle drei liegen außerhalb der Gottfunktion (policy-/backup-Flächen) und
kollidieren nicht mit dem Schnitt. Wer: Sol je Ticket, Kimi-Review für SEC2.

## Welle S-CUT — Der Schnitt (move-only, seriell, Muster D3)

Reihenfolge nach Risiko (in sich geschlossen → verwoben). Nach jedem Commit:
`cargo test business_os::` gezielt + `cargo check`. Der Commit-Pfad ist der
etablierte Temp-Index/CAS-Weg.

1. **S-CUT1** `backup_restore.rs` — Drill-/Restore-/Prune-Familie (~35 fns;
   P7e bescheinigt Geschlossenheit).
2. **S-CUT2** `module_lifecycle.rs` — Install/Release/Rollback/Katalog
   inkl. der drei Repair-Schichten (unverändert mitbewegen — Löschung ist
   S-FIX). **Karte 01.08.: ESCAPE-HATCH GEZOGEN, Reihenfolge ändern.**

   Gemessen: 52 Funktionen der Familie, 2.321 Zeilen, verteilt auf SIEBEN
   Cluster zwischen Zeile 68 und 14136 — anders als die geschlossene
   Backup-Familie aus S-CUT1. Entscheidend: **35 der 52 werden von
   ausserhalb aufgerufen**, also 35 neue `pub(super)`-Nähte gegen die im
   Plan festgelegte Stopgrenze von 20.

   Die Nähte ballen sich: `module_catalog_for_rxdb` (13 Aufrufe),
   `record_module_version` (10), `write_module_catalog_projection_to_rxdb`
   (9), `record_module_release` (7) tragen 39 der Aufrufstellen. Und sechs
   Funktionen der Familie sind projektionsartig
   (`*_for_rxdb`, `*_projection_*`) — sie gehören fachlich in S-CUT3,
   nicht hierher.

   **Empfehlung: S-CUT3 (Projektionen) VOR S-CUT2 ziehen.** Wandern die
   Projektionen zuerst, verlassen die beiden schwersten Nahtträger die
   Lebenszyklus-Familie, und S-CUT2 wird neu vermessen sehr wahrscheinlich
   unter die 20er-Grenze fallen. Die bisherige Reihenfolge S-CUT2 → S-CUT3
   war eine Annahme, keine Messung.

   **Neuvermessung 01.08., nach S-CUT3a+b: 35 → 20 Nähte.** Die Vorhersage
   hat gehalten; die Familie ist auf 38 Funktionen und 1.629 Zeilen
   geschrumpft, weil sechs projektionsartige Mitglieder mit S-CUT3
   abgeflossen sind. 20 ist aber genau die Grenze, also ohne Spielraum —
   und die Familie hat eine natürliche Fuge:

   - **S-CUT2a Release-Review/Audit** — 10 Funktionen, 343 Zeilen,
     **4 Nähte**. Klein genug, um sie direkt zu schneiden.
   - **S-CUT2b Version/Release/Install** — 28 Funktionen, 1.286 Zeilen,
     **17 Nähte**. Schwerster Träger `record_module_version` (5).

   Der Teil kostet zusammen 21 statt 20 Nähte, also genau eine mehr: die
   eine Stelle, an der Review-Code den Versionskern ruft. Ein Schnitt
   exakt auf der Grenze ist das nicht wert — beide Hälften einzeln sind
   prüfbar, der Gesamtschnitt wäre es kaum.
3. **S-CUT3** `store_projections.rs` — die business_records-/Queue-/Chat-
   Kompatibilitätsprojektionen (~74 fns) inkl. `repair_queue_projections`.
4. **S-CUT4** `command_plane.rs` — Acceptance/Routing/Outcomes/Outbox
   inkl. der 2.654-Zeilen-Funktion ALS GANZES (Zerlegung ist S-FIX;
   move-only heißt: auch Monster ziehen unzerlegt um).

   **Vermessen 01.08.: S-CUT4 ist KEIN Monolith-Schnitt, sondern sechs
   unabhängige.** Die Familie liegt zwischen Zeile 19.001 und 41.349 —
   rund 22.000 Zeilen, ein Drittel von store.rs. Entscheidend ist aber
   nicht die Größe, sondern die Nahtzahl, und die ist minimal: jeder
   Domänen-Handler (`handle_outbound_active_command`,
   `handle_customers_active_command`, `handle_office_control_command`,
   `handle_ats_active_command`, `handle_ats_mutating_command`,
   `handle_appsec_business_command`) hat **genau einen** Aufrufer, den
   Dispatcher, und **null** externe. Jeder zieht also einzeln um, mit
   einer Naht.

   Reihenfolge daher: erst die sechs Handler als je eigene Welle, zuletzt
   der Dispatcher `accept_rxdb_business_command_with_origin`. Ein Schnitt
   über alle 22.000 Zeilen auf einmal wäre nicht schneller, nur unprüfbarer.

   **KORREKTUR 01.08., nach den fünf Handler-Wellen.** Ich hatte hier
   geschrieben, der Dispatcher werde danach „eine Verzweigung über sechs
   Namen". Das war falsch, und es steht so auch in mehreren Commits.
   Gemessen nach S-CUT4e: der Dispatcher ist **2.653 Zeilen** — vorher
   2.654. Die fünf Wellen haben ihn um genau eine Zeile verkleinert.

   Der Grund: die sechs Handler waren Funktionen, die er *aufruft*. Sein
   eigener Rumpf enthält **49 weitere `ctox.*`-Kommandotypen inline**,
   nach Domäne: module (12), business_os (8), source (7), mailserver (5),
   secret (3), task/file/app_store/app (je 2), und neun weitere einzeln.
   Ich hatte die Nahtzahl gemessen und den Rumpf nicht gelesen — die
   Nahtzahl sagt, wie teuer ein Schnitt ist, nicht was in der Funktion
   steht.

   Für S-CUT4f folgt daraus: der Dispatcher zieht **unzerlegt** um, wie
   die Planregel es für Monster vorsieht. Das macht ihn nicht besser, es
   holt nur 2.653 Zeilen aus store.rs und legt die Command-Plane dorthin,
   wo ihre Zerlegung später stattfinden kann. Die Zerlegung der 49 Arme
   ist S-FIX, nicht diese Welle — und sie ist grösser, als dieser Plan
   bisher behauptet hat.

   **ZWEITE KORREKTUR, 01.08. — S-CUT4f ist BLOCKIERT, und zwar
   strukturell.** Auch nach S-CUT5 (Policy raus) geht der Schnitt bei
   KEINER Hüllentiefe unter die Grenze. Vollständig vermessen:

   | Hülle          | Familie          | neue `pub(super)` in store.rs |
   |----------------|------------------|-------------------------------|
   | nur der Kern   | 5 Fn / 2.812 Z   | **36** |
   | Tiefe 1        | 50 Fn / 4.529 Z  | **50** |
   | Tiefe 2        | 89 Fn / 5.827 Z  | **38** |
   | volle Hülle    | 170 Fn / 7.533 Z | **23** |

   Das Minimum ist 23, und es kostet ein Modul von 7.533 Zeilen, das 30 %
   von store.rs verschluckt — samt `bind_chatgpt_login_server`,
   `build_chatgpt_authorize_url` und `assign_module_founder`, die mit
   einer Command-Plane nichts zu tun haben. Sie landen nur deshalb in der
   Hülle, weil der Dispatcher ihr einziger Aufrufer ist; er ruft sie aus
   seinen 49 inline-Armen.

   **Damit kippt für diesen einen Fall die Planregel.** „Erst move-only,
   Zerlegung später" funktioniert, solange ein Monster wenige Fühler hat.
   Dieses hat 36. Die Zerlegung muss VOR dem Schnitt kommen, nicht danach.

   Zwei Messrichtungen, beide nötig — hier ist einmal Verwirrung
   entstanden, die im Plan festgehalten gehört: die API-Fläche (wie viele
   Familienfunktionen von aussen gerufen werden) ist nur **2**. Die für
   die Stopgrenze zählende Richtung ist die andere: wie viele private
   store.rs-Funktionen `pub(super)` werden müssen, damit die verschobene
   Familie sie noch erreicht. Wer nur die erste misst, hält einen
   unmöglichen Schnitt für trivial.

   **Nachfolgeticket S-DISPATCH:** die 49 Arme domänenweise in Handler
   extrahieren (Zielform sind die fünf bereits ausgelagerten Handler),
   danach S-CUT4f neu vermessen. Das ist Semantikarbeit mit Tests, keine
   Move-Welle.
5. **S-CUT5** Policy-Anteile — **Karte erstellt 01.08., Schnitt VERTAGT.**
   Gemessen: 16 Policy-Overlays in store.rs. Sechs davon greifen direkt auf
   den Store zu (Enforcement, bleibt per Ticket). Von den zehn übrigen sind
   vier transitiv an `scoped_policy_decision` gebunden, zwei an allgemeine
   Store-Helfer (`source_sanitize_slug`, `support_string`). Die letzten vier
   plus `policy_actor_from_session` liessen sich technisch verschieben —
   aber sie brauchen `BusinessOsSession` und die Session-Helfer, die store.rs
   besitzt. Ein Umzug nach policy.rs erzeugte damit **policy → store** und
   drehte genau die Schichtung um, die dieses Ticket herstellen soll.
   Sieben der 16 Overlays sind session-gekoppelt.

   **STAND 01.08.: Vorbedingung ERFÜLLT, S-CUT5 wird VORGEZOGEN — vor
   S-CUT4f.** `session.rs` liegt seit der Session-Welle unter store.rs und
   policy.rs. Und S-CUT4f hat gezeigt, warum die Reihenfolge zwingend ist:
   der Command-Plane-Schnitt stoppte bei 25 Nähten (Grenze 20), und **14
   der 25 sind Policy-Funktionen** — `reject_command_if_policy_denied`,
   `workspace_policy_decision`, `task_policy_decision`,
   `policy_audit_actor_context` und weitere.

   Die beiden Schnitte sind verschränkt: die Policy-Nähte zählen die
   Aufrufe des Dispatchers mit, die Dispatcher-Nähte die Policy-Aufrufe.
   Wer zuerst zieht, entlastet den anderen. Policy ist die kleinere und
   tiefere Schicht, also zieht Policy zuerst.

   Vermessen: 30 Funktionen, 550 Zeilen, **21 Nähte** — eine über der
   Grenze. Die Familie teilt sich sauber, und diesmal **ohne Zusatzkosten**
   (13 + 8 = 21, kein Aufschlag wie bei S-CUT2/3):

   - **S-CUT5a Entscheidungen** — 16 Funktionen, 276 Zeilen, 13 Nähte.
   - **S-CUT5b Audit/Summary/Rest** — 14 Funktionen, 274 Zeilen, 8 Nähte.

   Ziel ist ein NEUES Modul `store_policy.rs`, NICHT das bestehende
   `policy.rs`: die Overlays brauchen Store-Helfer, ein Umzug nach
   policy.rs erzeugte damit `policy → store` und drehte genau die
   Schichtung um, die dieses Ticket herstellen soll.

   **Zum fünften Mal dasselbe Muster.** Ein „hartes Ziel" löst sich auf,
   sobald das, was eine Ebene zu hoch einsortiert war, nach unten wandert.
   Vorher: core_state, command_lifecycle, communication_store, session.

   Ursprüngliche Notiz (historisch): `BusinessOsSession` samt
   `session_user_id`/`session_role` muss zuerst unter store.rs und
   policy.rs wandern. Erst danach
   ist der Policy-Schnitt ein Move und keine Zyklus-Erzeugung.
   Ein Probeumzug wurde ausgeführt und wieder zurückgenommen; der
   Compiler-Fehler (BusinessOsSession/session_user_id/session_role nicht in
   policy.rs sichtbar) ist der Beleg.

Erwartung danach: store.rs ≈ Restkern (Öffnung/Verwaltung/Verkabelung).
Escape-Hatch je Schnitt: Nähte > 20 pub(crate)-Einstiege ⇒ STOPP + Karte.

## Welle S-TRUTH — Eine Wahrheit statt vier (nach S-CUT, ▲)

| ID | Ticket | Befund |
|---|---|---|
| ST1 ▲ | Zirkular-Inventar beenden: `business_command_inventory.json` wird aus dem Klassifizierer-CODE generiert (eine Quelle), nicht per Regex aus der eigenen Datei gescraped und re-included. Browserfähige generierte Darstellung (löst G12); Drift-Check in die Validierung | P7f, snapshot:21379-21408 |
| ST2 ▲ | Status-Vokabulare 4→1: store konsumiert die generierten Lifecycle-Konstanten; Milestone-Mengen („accepted erreicht", Terminal) IM GENERATOR definieren (löst G11); `blocked`-Semantik explizit | P7f + G11 |
| ST3 ▲ | Terminal-Erkennung strukturiert: Queue-Terminalstatus aus Feld statt Substring auf `status_note`; `queue_task_payload` fälscht `command_type`/`inbound_channel` nicht mehr pauschal | P7b, snapshot:39655-39676, 39478-39502 |

## Welle S-FIX — Schuldklassen an der Quelle (nach S-CUT, PARALLEL pro Modul)

**command_plane.rs** (Sol ×2 seriell im Modul):
- SF1 ▲ Gottfunktion zerlegen: Claim → Authorize → Dispatch → Outcome als
  explizite Stufen; die 7 Autorisierungsstapel → 1 (Policy-Modul);
  „already accepted" 3 Formen → 1.
- SF2 ▲ Outcome-Write-Schlucken (8 Stellen) beheben — Fehler beim
  Outcome-Schreiben ist beobachtbar, nie still.
- SF3 ▲ Uncertain-Claim-Sackgasse generisch lösen (TTL-Re-Drive statt
  5-Typen-Allowlist — vereint mit G3).

**store_projections.rs**:
- SF4 ▲ Kanonischer Schreibpfad statt 6 paralleler Schreiber;
  **Akzeptanz: `repair_queue_projections` ist GELÖSCHT**; tote
  Schattenimplementierung (~140 Z.) raus; `waiting_dependencies` erhält
  Verhaltenstests.

**module_lifecycle.rs**:
- SF5 ▲ Uninstall/Delete räumt vollständig ab (Grants, ACL, Release-Zeilen);
  **Akzeptanz: Lesepfad-Backfill GELÖSCHT**, ersetzt durch einmalige echte
  Migration; „Modul geändert"-Hash 4 Definitionen → 1;
  `ctox-system`-Admin-Bypass → Policy-Entscheidung.

**backup_restore.rs**:
- SF6 ▲ Retention wird vollzogen (Scheduler/Hook, nicht Deklaration);
  Active-Root-Restore als ECHTER end-to-end Drill (nicht nur isolierte
  Kopie); Kompatibilitäts-Matrix real (min < max oder ehrlich Same-Version
  dokumentiert).

**policy.rs**:
- SF7 Enforcement-Idiom (~60 Kopien) → ein Helper/Makro;
  Legacy-Grant-Programm → Migration mit definiertem Endzustand statt
  permanentem Allow-All-Fundament (▲, mit Owner-Rückfrage vor Härtung:
  welche Rollen sollen die Legacy-Weite behalten?).

**Querschnitt (alle Module):**
- SF8 ▲ Die 42 Text-Dispatch-Stellen: je klassifizieren (Kontrollfluss →
  typisierter Code mit Konsumenten-Check; Log → belassen); MiniMax
  inventarisiert, Sol behebt modulweise.
- SF9 Totholz (~400 Z.: DOCX-Fallback-Kette, Schattenimpl, Rest) mit
  Call-Site-Beweis löschen.

## Welle S-GUARD — Zement

- SG1 Inventar-/Contract-Drift-Checks in `cargo test` (nicht nur Tool).
- SG2 Guard: kein `.contains(` auf Fehler-/Statustexten im Kontrollfluss
  (Allowlist für Logs), Rust-Seite — Pendant zum JS-Guard.
- SG3 Modulgrenzen-Sichtbarkeitstest (wer darf wen rufen).
- SG4 P7-Re-Review aller 6 Fokusse (frische Kontexte) — Ziel je ≥ A−.

## Serialisierung & Umfang

SEC (3 Tickets, parallel möglich) → S-CUT 1–5 (seriell, gleiche Datei) →
S-TRUTH (ST1/ST2 parallel, ST3 nach S-CUT3) → S-FIX (parallel pro Modul,
innerhalb Modul seriell) → S-GUARD. Geschätzt ~24 Tickets, Sol-lastig,
MiniMax für Inventare/Guards, Kimi für SEC2-Review, Plan-Adversarial und SG4.

---

# v2-Auflagen (adversariales Kimi-Review, 2026-07-29 — alle übernommen)

Reviewer-Verdikt: GO_MIT_AUFLAGEN. Die folgenden Änderungen ERSETZEN die
entsprechenden v1-Regelungen.

## Reihenfolge (ersetzt v1)

SEC1+SEC2 (parallel) → **S-CUT5 (Policy-Extraktion ZUERST** — 8 verstreute
Overlay-Anker, 103 Call-Sites: wer sie zuletzt zieht, re-touched vier fertige
Module) → S-CUT1 → S-CUT2 → S-CUT3 → S-CUT4 → S-TRUTH → SF10 → S-FIX
(parallel pro Modul) → SF8 → S-GUARD.
**SEC3 wird bis nach der D-Welle zurückgestellt** (Caller in rxdb_peer.rs —
Kollision mit laufendem Schnitt; zudem Owner-Frage 1 offen).

## Harte Gates (ersetzt „prüfen")

- S-CUT startet NICHT, solange store.rs fremd-dirty ist (aktuell +42 Zeilen
  Fremdarbeit). Landet die Fremdarbeit, werden alle snapshot-Anker neu
  aufgelöst.
- mod.rs-Registrierungen werden mit der D-Welle koordiniert (gleiche Datei,
  zwei Kampagnen — Kollisionshistorie bekannt): CAS-Commits, Claim in der
  Kampagnen-Memory vor jedem Schnitt.

## Ticket-Präzisierungen

- **Jeder S-CUT:** Das Testmodul (23.700 Zeilen, ~314 Tests, greift privat
  zu) wird MIT disponiert — Tests der Familie ziehen im selben Commit mit;
  Escape-Hatch-Metrik um „Test-Call-Sites über Modulgrenze" erweitert.
  Vor jedem Schnitt: Karten-Schritt mit exakter fn-Liste (Muster D3).
- **S-CUT1:** Scope explizit inkl. Entscheidung zu Audit-Retention/-Export
  und Redaktions-Scanner (3 Nicht-Backup-Nutzer → Scanner bleibt im Kern
  oder eigenes Modul; Karte entscheidet, Begründung im Commit).
- **S-CUT2:** Umfang korrigiert: ~210 fns (nachgezählt), nicht 130.
- **S-CUT4/SF1:** SF1 ist KEIN Ein-Commit-Ticket — Stufen-Extraktion
  (Claim→Authorize→Dispatch→Outcome) mit Sub-Commits je Stufe + Testlauf;
  dokumentierte Ausnahme von der Ein-Commit-Regel.
- **SEC2-Akzeptanz erweitert:** Manifest-Verifikation auf schlüssellosem
  Fremd-Root mit eskrowiertem Signing-Key erfolgreich + Tampered-Manifest-
  Test (sonst bleibt der Disaster-Fall strukturell kaputt).
- **SF4-Akzeptanz erweitert:** Regressionstest „Replay eines terminalen
  Commands ändert seinen Status nicht" (deckt record_command-ON-CONFLICT-
  Rückspulung und Markdown-Schnellpfad ab).
- **ST1-Akzeptanz erweitert:** die 13 Hand-Prädikate sind aus dem Inventory
  ableitbar oder als benannte Generator-Ausnahmen deklariert.
- **ST3:** spannt nach dem Schnitt zwei Module (store_projections +
  update_ctox_task im Kern) — Besitz: store_projections; update_ctox_task
  konsumiert dessen API.
- **NEU SF10 (vor SF8):** ~40 Test-Assertions der Form
  `error.to_string().contains(...)` auf Verhaltens-Pins umbauen — sonst
  blockieren SF8/SG2 an den eigenen Tests.
- **SG2 als eigenes Projekt:** 347 rohe `.contains(`-Stellen triagieren
  (Kontrollfluss/Log/Test) mit Allowlist-Mechanik; MiniMax inventarisiert,
  Sol baut den Guard.
- **SG4-Ziel angeglichen:** W1-Abschluss ≥ B+ je Fokus (A− folgt mit dem
  W5-Feinschliff — Masterplan-Maßstab gilt).
- **G-Wellen-Handshake:** ST1 schließt G12, ST2 schließt G11, SF3 schließt
  G3 — Übergabe wird in der Kampagnen-Memory vermerkt, damit nichts doppelt
  oder nie geschlossen wird.

## Neue Tickets aus heimatlosen Findings (SM-Serie)

SM1 Recoverable-Re-Exec: Re-Autorisierung bei gelöschtem/verändertem User
(P7a, snapshot:30452) · SM2 Outbox-Destination „business-os": fünfte
Repräsentation + accepted-Default + fehlende Tests (P7a, 30622; inkl.
Outbox-Flush-Fehler sichtbar, 13253/30760) · SM3 Markdown/Documents-
Synchronschluss-Idempotenz (P7b, 13451) · SM4 LWW-Stempel-Guard store-seitig
(P7b, 41537 — Pendant zur rxdb_peer-Stempel-Disziplin) · SM5 `superseded`
vs `rolled_back` entkonflationieren (P7c, 8757) · SM6 rollback_module_release:
module.json-Snapshot statt asymmetrischem Überschreiben — stiller
Datenverlust (P7c, 8844) · SM7 Restore-Blocking-Check heilt sich selbst
(open_store im Prüfpfad; P7e, 17160) · SM8 Audit-Retention nicht per Request
aushebelbar + Export-Wachstum begrenzen (DSGVO; P7e, 15345) · SM9
Fremd-Schema-Direktzugriffe (stalwart/IoT) hinter Ports (P7f, 27732) ·
SM10 migrate_business_users_roles ohne CREATE-TABLE-Sniffing (P7f, 44407) ·
SM11 Command-Type→Permission-Handlisten in die generierte Quelle (P7d,
21452) · SM12 „~42"-Zahl ersetzt durch das echte MiniMax-Inventar aus SG2.

SM13 **App-Store-Installation ist testfrei — beide Enden.** Gefunden 01.08.
beim Prüfen der roten Tests aus S-CUT2b, und grösser als „zwei veraltete
Fixtures":

- `app_store_install_uninstall_allows_admin_policy_path` und
  `..._allows_explicit_module_grants` bedienen sich über
  `serve_test_zip_once` bei einem lokalen HTTP-Server. Der SSRF-Guard in
  `module_lifecycle.rs` blockiert Loopback — zu Recht, sein Kommentar sagt
  genau das. Die Tests sind seit dem Guard rot.
- Damit läuft KEIN Test mehr über die Berechtigungszusicherungen von
  Install/Uninstall, obwohl beide Tests genau dafür geschrieben wurden.
- `serve_test_zip_once` hat exakt zwei Aufrufer: diese beiden. Und der
  Zip-Pfad über den RxDB-Chunk-Store — im Sicherheitskommentar der
  *sichere* Weg, "never over HTTP" — hat gar keinen Test.

Reparatur ist kein Fixture-Tausch: es fehlt ein Helfer, der eine
Desktop-Datei in den Chunk-Store setzt, damit `file_id` bedient werden
kann. Danach beide Tests auf `source_kind: "zip"` umstellen und
`serve_test_zip_once` samt HTTP-Fixture löschen. Erst dann prüfen die
Berechtigungs-Assertions wieder etwas, und der sichere Pfad ist zum
ersten Mal abgedeckt.

SM14 **`status` heisst zweierlei, und die Transportschicht gewinnt.**
Gefunden 01.08. bei S-CUT4a. `write_rxdb_control_command_state`
(store.rs:20080) schreibt bedingungslos

    object.insert("status", Value::String(status.to_string()));

in das Ergebnisobjekt des Handlers. `status` ist dort der *Lebenszyklus*-
Status des Kommandos (`completed`/`cancelled`/`failed`); im Handler-Ergebnis
steht aber der *fachliche* Status der Domäne. Ein Handler, der
`needs_review` oder `active` beantwortet, bekommt seine Antwort durch
`completed` ersetzt — nicht überstimmt, sondern überschrieben.

Zwei Tests zeigen das seit Längerem und sind deshalb rot:
`outbound_campaign_activation_requires_ready_channel` und
`customers_import_from_outbound_creates_account_contacts_and_dedupe`.

Dieselbe Form wie die bereits behobenen Vokabular-Befunde: ein Feldname
für zwei Bedeutungen. Reparatur ist NICHT „den Test anpassen" — es braucht
getrennte Felder (fachlicher Status im Ergebnis, Lebenszyklus daneben),
und dann müssen die Leser beider Seiten mitgezogen werden.

Zuordnung: SM2/SM1/SM11/SM14 → command_plane · SM3/SM4 → store_projections ·
SM5/SM6/SM13 → module_lifecycle · SM7/SM8 → backup_restore ·
SM9/SM10 → Kern.

## Offene Owner-Fragen (blockieren NUR die genannten Tickets)

1. **SEC3**: strukturiertes Fehlschlagen statt stiller Reaktivierung ist ein
   Breaking Change für lebende Tenants — wer provisioniert künftig, ist das
   ohne Migrationsankündigung akzeptabel? (SEC3 wartet ohnehin auf D-Welle.)
2. **SF5**: „module.json version 1.0.0 ⇒ Auto-Release ins Team" ist heute
   test-gepinntes Produktverhalten — darf das entfallen?
3. **SF4/SF5**: Einmal-Korrektur für bereits gedriftete Feld-DBs — Updater-
   Hook oder Startup? Risikoträger ohne Repair-Fallback?
4. **SF6**: destruktiver Active-Root-Drill — Umgebung/Blast-Radius?
5. **SF6**: Retention-Vollzug: Daemon-Scheduler oder Start-Hook?
   (Architektur-Entscheidung außerhalb store.rs.)
6. **P7d-LOWs** (`deny_supported:false`, Epoch-Flut): akzeptieren oder
   Roadmap?
