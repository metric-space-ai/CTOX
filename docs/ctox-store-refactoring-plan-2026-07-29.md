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

   **DRITTE MESSUNG, nach den sieben S-DISPATCH-Wellen.** Der Dispatcher
   ist von 2.653 auf 874 Zeilen gefallen, und der enge Kern kostet statt 36
   nur noch — nach meiner Messung — 19 Nähte. Der Worker mass 21 und stoppte.

   **Seine Zahl ist die richtige, meine war ein Werkzeugfehler.** Mein Skript
   zählt nur Funktionen. Die echte Naht umfasst auch private TYPEN, ihre
   assoziierten Funktionen und die Felder, die der Dispatcher liest:
   `ActiveExternalSqlControlCommand` samt `try_acquire`, und `ReportAccepted`,
   den privaten Rückgabetyp von `record_report_command`.

   Damit ist es der dritte Messfehler dieser Art in derselben Frage: erst die
   falsche Richtung (API-Fläche statt benötigter Sichtbarkeiten), dann die
   Hüllentiefe, jetzt Typen und Felder. Eine compilergestützte Messung schlägt
   ein Regex-Skript — das gilt ab hier als Regel, nicht als Einzelfall.

   Interessant nebenbei: nach der Zerlegung ist die transitive Hülle der
   SCHLECHTESTE Weg (Tiefe 1 kostet 35, Tiefe 2 sogar 46 Nähte), während sie
   vorher das Minimum lieferte. Die Helfer gehören jetzt sichtbar den
   Handlern.

   **S-CUT4g** zieht die geteilten Control-Command-Typen unter store.rs und
   die Command-Plane — zum sechsten Mal dasselbe Muster. Danach wird erneut
   gemessen, compilergestützt.

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
  **ENTSCHIEDEN 01.08.: NICHT GEBAUT, und das ist das Ergebnis.**

  Die Prämisse des Tickets trägt nicht. Die Fünferliste kodiert nicht
  „diese sind wichtig", sondern „diese sind auditiert idempotent". Ein
  TTL ersetzt das nicht — er tauscht die bekannte Sackgasse gegen
  mögliche doppelte Zahlungen, Mails oder Dateien.

  Vermessen, mit Codebelegen: es gibt heute KEIN allgemeines, im Code
  prüfbares Merkmal für „sicher erneut ausführbar".
  - `BusinessCommand` trägt keine Replay-Semantik (nur ID, Modul, Typ,
    Record-ID, Payload, Client-Context, Origin).
  - Der `idempotency_key` ist immer die `command_id`. Das schützt vor
    einem zweiten Claim mit anderem Intent, beweist aber nichts über die
    Idempotenz des Handler-Effekts.
  - Der Claim-Layer meldet für JEDEN nicht-terminalen Zustand pauschal
    `uncertain`; er kennt den Stand des externen Effekts nicht.
  - „Kein Resultat" ist mehrdeutig: noch nicht ausgeführt ODER Effekt
    ausgeführt und Outcome vor der Persistenz verloren.
  - `DataRead`/`DataWrite` trägt nicht: `person_research` ist Read,
    `external_sql.write` ist Write — **beide stehen auf der Liste**.
    Lesekommandos können externe Kosten und Artefakte erzeugen,
    Schreibkommandos können per Upsert sicher sein.

  **Was für einen sicheren generischen Re-Drive fehlt** (Owner-Frage 8):
  1. Eine verpflichtende Handler-Deklaration, zentral auswertbar, z. B.
     `ReplayPolicy::{Never, Idempotent, ReconcileByReceipt,
     ReadOnlyRecomputable}` — deklariert, nicht aus Namen oder
     Permission abgeleitet.
  2. Für `Idempotent`: eine stabile Effekt-/Dedupe-ID, die der
     tatsächliche Sink durchsetzt (DB, Mailprovider, Payment, Dateisystem).
  3. Für `ReconcileByReceipt`: ein dauerhaftes Effekt-Receipt an
     `command_id + payload_hash`, mit Phasen `not_started`,
     `pending_external`, `applied`, `completed`, plus einer
     handler-spezifischen Abfrage, ob der Effekt schon angewendet wurde.

  Bis dahin bleibt die Liste. Sie ist explizit, auditiert und ehrlich —
  eine generische Lösung ohne 1–3 wäre nur scheinbar allgemeiner.

**store_projections.rs**:
- SF4 ▲ Kanonischer Schreibpfad statt 6 paralleler Schreiber;
  **Akzeptanz: `repair_queue_projections` ist GELÖSCHT**; tote
  Schattenimplementierung (~140 Z.) raus; `waiting_dependencies` erhält
  Verhaltenstests.
  **VERMESSEN 01.08.: Prämisse hält nicht. Nichts gelöscht, nichts
  geändert — und das ist das Ergebnis.**

  Die Karte Schreiber → Reparaturzweig lässt sich nicht ziehen, weil die
  Reparatur etwas anderes heilt, als das Ticket annimmt:

  - `leased_terminal_success_status` / `..._failure_status` korrigieren
    NICHT die Projektion. Sie mutieren einen kanonischen Queue-Task, der
    noch auf `leased` steht, obwohl das Kommando terminal ist. Der
    kanonische `refresh_queue_task_projection` projiziert in genau dieser
    Lage bereits korrekt `completed`/`failed` — **ohne** den kanonischen
    Task anzufassen. Die Projektion ist also richtig und die QUELLE falsch.
  - Die `*_from_canonical`-Zweige gleichen Status, Note und Lease-Felder
    mit einem vorhandenen Task ab. Historische Ursache war laut
    Recovery-Dokument ein Worker-Ack ohne anschliessenden Refresh. Denselben
    Zustand erzeugen heute weiterhin der generische
    `push_collection_records`-Schreiber und `channels`-Mutationen ohne
    Store-Refresh.

  Die sechs „parallelen Schreiber" sind also nicht die Ursache. Die
  Ursache ist, dass kanonische Mutationen aus anderen Pfaden keinen
  Refresh auslösen. Wer die Schreiber vereinheitlicht und dann die
  Reparatur löscht, entfernt das Netz und lässt den Boden, wie er ist.

  **Neu geschnitten (SF4a/b):**
  - SF4a: `leased` + terminales Kommando ⇒ der kanonische Task wird beim
    Terminalwerden mitgeführt, statt später geheilt. Danach fallen die
    beiden `leased_terminal_*`-Zweige.
  - SF4b: `push_collection_records` und die `channels`-Mutationen lösen
    einen Refresh aus. Danach fallen die `*_from_canonical`-Zweige.

  Erst wenn beide liegen, ist `repair_queue_projections` löschbar. Vorher
  nicht — die Kampagne hat sieben Reparaturen gelöscht, jede NACH ihrer
  Ursache.

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

## SF-Prämissenprüfung (01.08.) — was noch steht

Nach drei Tickets, deren Prämisse unter der Messung wegbrach (SM13, SF3,
SF4), wurde der Rest der Serie vermessen, BEVOR gebaut wird. Ergebnis:

| Ticket | Urteil | Gemessen |
|---|---|---|
| SF1 | **STEHT** | Funktion 2.654 → 858 Z., aber weiterhin kein vierstufiger Ablauf; 19 Autorisierungsstellen in 5 Formen, 5 Return-Shapes |
| SF5 | **STEHT** | kein Cleanup in beiden Quellpfaden, beide Lesepfad-Backfills, 4 Hashdefinitionen, 2 hartkodierte Admin-Sessions |
| SF6 | **STEHT** | Retention wird berechnet, aber nie aufgerufen — weder Scheduler noch Start-Hook |
| SF7 | **STEHT** | 59 produktive Call-Sites (Schätzung war ~60), heute in store.rs (48) und command_plane.rs (11) |
| SF8 | **VERSCHOBEN** | Inventar + Ratchet erledigt (SG2); es bleiben 30 Entscheidungen, 15 davon in service.rs |
| SF9 | **STEHT** | 398 nachweislich tote Zeilen (Schätzung war ~400) |
| SF10 | **VERSCHOBEN** | 46 Assertions vorhanden, aber KEIN Gate für SF8/SG2 — die behauptete Serialisierung besteht nicht |

**Korrektur meiner eigenen Verallgemeinerung.** Ich hatte nach SM13/SF3/SF4
geschrieben, das Review habe „die Ursachen eine Ebene daneben verortet". Das
trägt nicht: bei fünf von sieben stimmen Symptom UND Ursache, zwei
Schätzungen (~60, ~400) sind fast exakt. Drei Treffer waren keine Regel.

**Reihenfolge, mit einer Abhängigkeit, die der Plan nicht hatte:**

1. **SF5** — der unmittelbarste Governance-Defekt: Delete/Uninstall kann alte
   Grants und alte Release-Evidenz unter derselben Modul-ID wieder wirksam
   machen (Grant-Resurrection).
2. **SF6** — Klartext-Backups liegen ohne Vollzug unbefristet; kleine
   Aufrufkante, hohe Datenschutzwirkung. Betriebssemantik ist eine
   Owner-Frage.
3. **SF9** — 398 tote Zeilen mit Call-Site-Beweis, risikoarm, verkleinert
   store.rs vor den grossen Schnitten.
4. **SF7 VOR SF1.** Erst der typisierte Command→Actor/Permission/Scope→
   Decision→Denial-Vertrag, dann die 59 Call-Sites darunter. **Ohne das
   verschiebt SF1 die Autorisierungsstapel nur räumlich.** Diese
   Abhängigkeit stand nirgends im Plan.
5. **SF1** stufenweise; die fünf Replay-Shapes in EINEN Receipt-Typ, ohne
   die terminale und die uncertain-Semantik im JSON zu verlieren.
6. **SF8** mit dem ausführbaren Bestand 30 planen, nicht mit 42 oder 22, und
   mit einer benannten Blindstellenliste.
7. **SF10** just-in-time zu SF8, nicht als globales Vorab-Gate.


### SF10 — geschlossen ohne Umbau (02.08.)

Das Ticket wollte 46 Message-Assertions umbauen, weil sie SG2 blockierten.
Beide Hälften halten nicht:

- **Die Blockade existiert nicht.** SG2 ignoriert Testmodule ausdrücklich und
  ist längst gelandet. Das stand schon in der Prämissenprüfung.
- **Die Zahl stimmt nicht.** Allein in `business_os/` und `service/` sind es
  **522**, nicht 46.

Und die Verteilung entscheidet die Sache. Der grösste erkennbare Block sind
**negative** Zusicherungen wie `assert!(!output.contains("sk-test-secret"))` —
sie belegen, dass ein Geheimnis NICHT durchsickert. Dort ist das Festnageln des
Literals nicht Schuld, sondern der Test selbst. Ein pauschaler Umbau hätte
Sicherheitstests beschädigt, um eine Metrik zu verbessern.

Die zielbezogene Fassung aus der Prämissenprüfung — „nur die Tests umbauen, die
den jeweils in SF8 typisierten Prosa-Vertrag festnageln" — wurde nach SF8
geprüft und ist **leer**: SF8s typisierte Fehler haben keine verwaisten
Assertions hinterlassen.

Was bleibt, ist keine mechanische Aufgabe, sondern eine Frage pro Stelle: Ist
diese Meldung Teil des öffentlichen Verhaltens? Wo ja, gehört sie festgenagelt.
Wo nein, ist ein typisierter Fehler die Antwort — und der entsteht bei SF8, nicht
hier. Kein Sammelticket.


### SG4 — unabhängige Nachprüfung (02.08., Kimi)

Geprüft wurde die Berechtigungsfläche, die ich und Sol gebaut haben — deshalb
von einem Dritten, mit dem Auftrag zu WIDERLEGEN.

**Der Hauptbefund korrigiert eine Behauptung von mir.** Nach SF7a/b hatte ich
geschrieben: „59 Stellen ohne eine einzige ungeschützte Mutation — die
Verdopplung war real, die fehlende Durchsetzung nicht." Das war wahr und
irreführend. Beide Wellen inventarisierten **Durchsetzungsstellen** und prüften,
ob jede ihre Ablehnung auswertet. **Ein Pfad ganz ohne Prüfung hat nichts, was
ein Inventar von Prüfungen finden könnte.** Zählen, was da ist, entdeckt nicht,
was fehlt.

Gefunden wurden zwei mutierende Pfade ohne jede Entscheidung:

- **C-1 `ctox.file.materialize`** — schrieb Datei-Chunks hinter blosser
  Authentifizierung, während das Geschwister `ctox.file.export` zum blossen
  LESEN eine Berechtigung verlangte. Der Schreibpfad war der schwächere.
  **Behoben** (Commit 852a6e644), mit Test und Gegenprobe.
- **C-2 `ctox.maintenance.client_ready`** — schliesst Wartungs-Leases ohne
  Entscheidung. Das fehlende Gate ist Fakt; die Auswirkung (vorzeitiges Beenden
  eines Wartungsfensters bei bekannter `lease_id`) ist als Vermutung markiert.
  **Offen** — braucht eine Owner-Entscheidung, ob dieser Pfad überhaupt
  aufruferbezogen sein soll.

Was der Prüfer angegriffen hat und was gehalten hat, gehört genauso zum
Ergebnis: `enforce_command_policy` liess sich nicht umgehen — die
`PolicyDecision` bleibt intern, der Aufrufer bekommt nur ein `#[must_use]`
gekennzeichnetes Ergebnis, und `on_allowed` läuft erst nach geprüfter Ablehnung.
Die groben Gates (`require_manage_all` für Rechnungen, IoT, ATS, Coding-Agent),
das Teilnehmer-Modell der Threads und die Read-only-Pfade wurden einzeln
durchgegangen und sind keine Lücken.

**Kleinere Befunde, offen:**

- Vier der elf bewussten SF7-Ausnahmen sind Persistenz-Konsistenz, nicht
  Entscheidungslogik — der Vertrag träfe dieselben Fälle. **Meine
  Commit-Begründung überzeichnet dort.** Kein Sicherheitsimpact.
- Collection-bezogene Grants überleben `ctox.module.delete` (SF5a räumt
  `scope_type='collection'` nicht ab). Stale Grants bei Namens-Wiederverwendung.
- Legacy-Manifeste mit `preview_user_ids` verlieren Preview-Zugriff bis zur
  expliziten Migration aus SF5b — die Migration existiert, muss aber gefahren
  werden.

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

**KORREKTUR 01.08. — meine Begründung war ein Lesefehler.** Ich hatte
den Zip-Pfad „den sicheren Weg" genannt und mich dabei auf den Kommentar
`never over HTTP` gestützt. Der beschreibt den TRANSPORT (RxDB/WebRTC
statt HTTP), nicht die Vertrauensstufe.

Gemessen beim Reparaturversuch: `install_app_module` stempelt für Zip die
Provenienz `{"kind":"zip","file_id":...}` OHNE `verified`, und derselbe
Pfad verlangt später `app_source.verified == true` (fehlend = false).
**Kein Zip-Upload kann also je installieren.** Und das ist ABSICHT: der
Kommentar an der Prüfung sagt, dass nicht vertrauenswürdige
Drittanbieter-Archive gesperrt bleiben, bis eine isolierte Sandbox
existiert. Nur `catalog` und der Erstanbieter-`github`-Pfad setzen
`verified: true`.

Der Zip-Pfad hat also keinen Test, weil er gesperrt ist — ein Test dort
könnte nur die Sperre bestätigen. Was bleibt, ist trotzdem ein Befund,
nur ein anderer:

- Die zwei Berechtigungstests brauchen eine Quelle, die das Trust-Gate
  BESTEHT (`catalog`), nicht Zip. Erst dann prüfen sie wieder, wofür sie
  geschrieben wurden.
- Für die Sperre selbst fehlt ein Test, der sie als Sperre festhält:
  ein Zip-Upload MUSS abgelehnt werden. Solange das niemand prüft, kann
  die Sperre unbemerkt fallen.

Brauchbar aus dem Versuch: `seed_rxdb_chat_attachment` existiert bereits
als Helfer, der Dokument und Chunks so schreibt, dass die Verifikation
besteht — die Fixture-Infrastruktur fehlt also nicht, wie ich annahm.

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

## Offene Owner-Fragen — entscheidungsreif (Stand 02.08.)

Alle technischen Tickets sind durch. Was hier steht, kann ich nicht entscheiden:
es sind Produkt-, Betriebs- und Risikoabwägungen. Je Frage: was gemessen ist,
die Optionen, meine Empfehlung, und was die Antwort freigibt.

---

**1. Stille Reaktivierung bei der Token-Ausstellung (SEC3)**

Gemessen: der Ausstellungspfad provisioniert und reaktiviert Benutzer stumm.
Strukturiertes Fehlschlagen wäre für lebende Tenants ein Breaking Change.
Optionen: (a) strukturiert scheitern, Provisionierung explizit und auditiert;
(b) beibehalten und dokumentieren; (c) Übergangsfrist mit Warnung.
**Empfehlung: (a) mit Ankündigung.** Eine Reaktivierung, die niemand angefordert
hat, ist dasselbe Muster wie die Grant-Wiederauferstehung aus SF5a.
Gibt frei: SEC3.

**2. ~~Auto-Release bei Version 1.0.0~~ — durch SF5b entschärft (02.08.)**

Die gefährliche Form gibt es nicht mehr. `backfill_semver_public_release_records`
liegt seit SF5b **innerhalb** `migrate_legacy_module_lifecycle_authority` — der
policy-geschützten, katalogweiten Migration mit Probelauf. Beim Lesen des
Katalogs passiert nichts mehr.

Was bleibt, ist zahm: die einmalige Migration legt für Alt-Module mit Major ≥ 1
OHNE vorhandenen Release-Datensatz einen an, gekennzeichnet als
`reviewed_by: ctox.release-record-migration`. Module mit bestehendem Release
werden übersprungen.

Das ist keine stille Veröffentlichung aus einer Versionsnummer, sondern das
Nachtragen fehlender Evidenz — und ohne es verlieren genau diese Module ihren
Release-Nachweis (der SF5b-Vorbehalt). **Nichts zu entscheiden.**

**ALTE FASSUNG (überholt):**

Gemessen: heute test-gepinntes Produktverhalten — eine 1.0.0 löst automatisch
ein Team-Release aus. Optionen: (a) entfällt; (b) bleibt, ausdrücklich
dokumentiert; (c) bleibt, aber hinter einer expliziten Freigabe.
**Empfehlung: (c).** Ein Release ist eine Veröffentlichung; sie sollte nicht aus
einer Versionsnummer folgen. Gibt frei: den Rest von SF5.

**3. ~~Einmal-Korrektur für gedriftete Feld-Datenbanken~~ — BEANTWORTET durch die
Implementierung (02.08.)**

Nachgemessen: der einzige produktive Auslöser ist das policy-geschützte Kommando
`ctox.module.repair_lifecycle_projection` mit
`migrate_legacy_manifest_lifecycle: true`. Es **verweigert einen `module_id`-Filter**
(„must run for the complete catalog") und kennt `dry_run`. Die beiden anderen
Aufrufstellen liegen im Testmodul.

Das ist genau die empfohlene Variante: manuell, katalogweit, mit Probelauf und
Evidenz. Ein Updater-Hook bleibt ein späterer Schritt, kein Blocker.

**ALTE FASSUNG (überholt):**

Gemessen: SF5b hat die Migration gebaut
(`business_os.legacy_module_lifecycle_authority.v1`, policy-geschützt, verweigert
Teilkataloge). SF4a/b haben die Reparaturen entfernt, die Altbestand bisher
still heilten. Offen ist nur der **Auslöser**. Optionen: (a) Updater-Hook;
(b) manuell pro Installation; (c) beim Start.
**Empfehlung: (b) für den ersten Durchlauf**, mit Evidenz je Installation —
danach (a). Ohne Reparatur-Netz ist ein automatischer Massenlauf riskant.
Gibt frei: den Abschluss von SF4/SF5 im Feld.

**4. ~~Destruktiver Active-Root-Drill~~ — LÖST SICH AUF (02.08.)**

Es gibt ihn nicht. `destructive_restore_performed` steht an allen drei
produktiven Stellen **fest auf `false`** und wird nirgends auf `true` gesetzt;
ein Test hält das fest. Der Active-Root-Restore ist als
`"status": "manual_operator_runbook"` mit `requires_operator_confirmation: true`
ausgewiesen — eine Anleitung für einen Menschen, keine automatisierte
Zerstörung.

Es ist also kein Radius festzulegen. Die Frage stammt aus einem Review, das
annahm, der Code führe den Drill selbst aus.

**ALTE FASSUNG (überholt):**

Gemessen: der Drill existiert; die Retention läuft jetzt (SF6). Offen sind
Umgebung und Radius. Optionen: (a) nur in isolierter Umgebung; (b) auf dem
aktiven Root mit Vorab-Sicherung; (c) gar nicht.
**Empfehlung: (a).** Ein Wiederherstellungstest, der die Produktion beschädigen
kann, prüft das falsche Risiko.

**5. ~~Retention-Vollzug: Scheduler oder Start-Hook~~ — BEANTWORTET (SF6)**

Entschieden für die Wartungsschleife mit Tagesmarker, nicht Start-Hook: der
Daemon läuft wochenlang, ein Start-Hook feuerte nie wieder. Der Worker hat die
Begründung geprüft und mitgetragen. Erledigt.

**6. P7d-LOWs: `deny_supported:false`, Epoch-Flut — AKZEPTIERT (02.08.)**

Beide sind bekannt, begrenzt und nicht sicherheitsrelevant. Akzeptiert und hier
dokumentiert, statt als Ticket ohne Termin geführt zu werden — ein Ticket, das
niemand einplant, ist eine Behauptung, keine Absicht. Wer sie später angeht,
findet sie hier; wer sie nicht angeht, hat nichts übersehen.

**ALTE FASSUNG:**

Optionen: akzeptieren und dokumentieren, oder auf die Roadmap.
**Empfehlung: dokumentieren.** Beide sind bekannt und begrenzt; ein Ticket ohne
Termin ist eine Behauptung, keine Absicht.

**7. ~~Woher darf ein wirksamer Laufzeitwert stammen?~~ — LÖST SICH AUF (02.08.)**

Es war keine Owner-Frage. Gemessen: `effective_operator_env_map` liest den
persistierten Laufzeit-Env und den Secret-Store und greift **null Mal** auf
`std::env` zu; kein Resolver erzeugt `CTOX_CUDA_HOME`; die Fixture legt einen
leeren Root an. Die vier Assertions erwarteten Werte, die auf diesem Pfad nicht
entstehen können — geschrieben gegen eine Fassung, die es nicht mehr gibt.

Die Produktion hält den Vertrag also bereits, und `AGENTS.md` schreibt ihn
ohnehin vor. Korrigiert wurde der TEST, nicht das Verhalten.

**Die Gegenprobe ist der Punkt:** baut man einen `std::env`-Zugriff ein, wird
der Test jetzt rot. Vorher behauptete er den Verstoss, statt ihn zu erkennen.
Deshalb blieb er die Kampagne über rot: das Umdrehen war zugleich der bequeme
grüne Weg, und ein Fix, der bequem ist, braucht mehr Belege als einer, der es
nicht ist.

**ALTE FASSUNG DER FRAGE (überholt):**

Gemessen: `env_or_config_reads_secrets_only_from_store` ist rot, und **vier
seiner Assertions widersprechen seinem eigenen Namen** — sie erwarten Werte, die
die Fixture nie persistiert, also durchsickerndes Prozess-Environment.
Optionen: (a) Vertrag gilt, die vier auf `None` drehen; (b) Prozess-Environment
ist erlaubt, Name und Vertrag korrigieren.
**Empfehlung: (a).** Aber es ist Ihre Entscheidung, weil sie festlegt, worauf
sich der Laufzeit-Vertrag stützt — deshalb habe ich den Test rot gelassen statt
ihn still grün zu ziehen.

**8. `ReplayPolicy` — ein Vertrag für sicheres Wiederanlaufen (SF3)**

Gemessen: es gibt heute KEIN allgemeines Merkmal für „sicher erneut ausführbar";
alle Surrogate scheitern (siehe SF3 oben). Ein sicherer generischer Re-Drive
bräuchte: deklarierte `ReplayPolicy` je Handler, eine Dedupe-Kennung, die der
Empfänger durchsetzt, und für abgleichbare Effekte ein Receipt an
`command_id + payload_hash`.
**Empfehlung: bauen, wenn die Fünferliste zur Last wird — nicht vorher.** Sie
ist explizit und auditiert; ein generischer Ersatz ohne diese drei Bausteine
wäre nur scheinbar allgemeiner.

**9. ~~`ctox.maintenance.client_ready` ohne Berechtigung~~ — BEANTWORTET (02.08.),
und die Prüfung fand etwas Schärferes**

Die Berechtigungsfrage ist entschieden: **kein Gate.** Nachgemessen ruft
`src/apps/business-os/app.js` diesen Pfad als Shell selbst
(`source: business-os-maintenance-readiness`), um die eigene Bereitschaft im
Wartungsfenster zu melden. `acknowledge_business_os_maintenance_ready` verlangt
bereits, dass das Lease das AKTUELLE und nicht terminal ist, dass der Dienst
läuft und die Replikation steht. Eine aufruferbezogene Berechtigung würde
legitime Clients aussperren und die Wiederanlauf-Schleife brechen — Option (b),
dokumentiert.

**Dabei kam ein anderer Befund heraus, mit Beleg statt Vermutung.**
`complete_maintenance_for_client` (`src/core/install/mod.rs`) führt eine LISTE
`client_readiness`, trägt den meldenden Client ein (und behält die anderen) —
und setzt anschliessend `phase = "completed"`, `status = "completed"`,
`percent = 100` für das GANZE Lease.

**Die erste Client-Meldung beendet das Wartungsfenster für alle.** Wenn nur die
erste zählte, bräuchte es keine Liste; die Liste ist der Beleg, dass hier
mehrere Clients gemeint sind. Der Funktionsname sagt „für einen Client", die
Wirkung gilt für alle — dieselbe Form wie SM14, wo ein Feld zwei Bedeutungen
trug.

**Nachgemessen (02.08.):** `client_readiness` wird **nur geschrieben und
nirgends gelesen** — weder in Rust noch im Browser. Es gibt also keine
„alle Clients"-Semantik, die verloren gegangen wäre; sie wurde nie gebaut. Es
existiert auch keine Menge erwarteter Clients, gegen die man vergleichen könnte.

Damit ist das gegenwärtige Verhalten in sich stimmig: der erste bereite Client
belegt, dass das Upgrade trägt. Die Liste ist unbenutzte Buchführung.

**Kein Blocker, sondern eine kleine Aufräumfrage:** entweder die Liste
auswerten (dann ist „alle erwarteten Clients" ein neues Feature, samt
Erwartungsmenge) oder sie entfernen. Solange sie geschrieben und nie gelesen
wird, verspricht sie eine Genauigkeit, die es nicht gibt — dieselbe Form wie
der Zähler aus SF4b, der auch mitzählte, was er nicht geändert hatte.

