# Sync-Engine-Feldbefunde von der SKF-Instanz (05.–10.08.2026)

Adressat: Sync-Engine-Refactoring-Worker (Readiness-Plan
`docs/ctox-sync-production-readiness-95.md`).

Absender: SKF-Übernahme-Session (Web-Research-Instandsetzung). Alle Befunde
stammen aus dem Produktionsbetrieb der Instanz `biz_4c00f223…` auf
`ctox-245acec8` und sind dort verifiziert. Notfixes sind gelandet und
commit-gebunden; die **Klassen** dahinter gehören in das Refactoring, nicht in
weitere Einzelfixes.

## Klasse 1: Projektions-Schreiber ohne RxDB-Umschlag (`DOC_CACHE_REV`)

Zwei native Schreiber legten Dokumente ohne `_rev`/`_meta` in den RxDB-Store;
der Dokument-Cache des Peers geriet in eine Endlosschleife (Spitze: 1.860
Fehler in 21 h, Journal dadurch unbrauchbar rotiert, Browser-Boot zeitweise
blockiert).

- `upsert_rxdb_collection_record_with_writer` (Store-Writer): gefixt in
  `d9c204a3d` mit Regressionstest.
- Modulkatalog-Projektor (`write_module_catalog_projection_to_rxdb`): entfernte
  den Umschlag sogar explizit und vergiftete den Store bei jedem Dienststart
  neu; gefixt in `820748c4c`.
- Repo-Sweep: verbleibende `remove("_rev")`-Stellen sind Vergleichskopien.

**Refactoring-Bedarf:** Es gibt keinen erzwungenen einheitlichen Schreibpfad in
den RxDB-Store. Jeder direkte `INSERT`/Writer kann die Umschlag-Disziplin
verletzen; ein Typ-/API-Zwang (ein einziger Envelope-stempelnder Writer, keine
rohen Table-Writes) würde die Klasse strukturell schließen.

## Klasse 2: Zustandsübergänge, die nur eine Wahrheit fortschreiben

Vier verifizierte Instanzen desselben Musters — Kanalzustand, Business-Command,
`ctox_queue_tasks`-Projektion und Fachdatensatz (`research_runs`) laufen
auseinander, weil Übergänge nicht atomar alle Sichten treiben:

1. Retry-Budget-Erschöpfung terminalisierte das Queue-Item in der
   Channels-Transaktion, ohne den Command-Fehlerpfad (`edfea32a6`).
2. Validierungs-Budget-Erschöpfung ackte `failed`, refreshte aber nur die
   Projektion; Command blieb ewig `accepted` (`0c9f4a0dc`).
3. CLI-Mutationen (`ctox queue cancel/fail/complete/release`) transitionierten
   Kanal und Command, nie die Projektion; der Browser zeigte einen stornierten
   Lauf dauerhaft als `running` und sperrte den Resume-Knopf (Fix in
   `src/core/mission/queue.rs`, 10.08.).
4. `research_runs`-Datensätze wurden nach Erzeugung nie fortgeschrieben;
   Spiegel nachgerüstet (`b7355cbe9`, `513f3fdc4`), liest den RxDB-Store,
   nicht `business_records`.

**Refactoring-Bedarf:** Der Readiness-Plan fordert idempotente Command-,
Projection- und Saga-Effekte — es fehlt die eine Übergangs-Primitive, die alle
Sichten gemeinsam schreibt. Solange jeder Aufrufer selbst daran denken muss,
entsteht die Klasse immer wieder (vier Fundstellen in einer Woche; die fünfte
ist statistisch unterwegs).

## Klasse 3: Browser-Push-Verlässlichkeit

Ein im Browser erzeugter `research_tasks`-Datensatz (UI-Dialog, 03.08.)
replizierte **nie** zum Server — er blieb tagelang nur im lokalen IndexedDB
sichtbar und erzeugte eine Geister-Aufgabe in genau einer Sitzung.
`research_tasks` ist weder read-only noch demand-only deklariert; der Push ist
schlicht verloren gegangen (vermutlich Race mit Sitzungsende). RPO 0 gilt laut
Plan für bestätigte journalisierte Writes — der UI-Write hatte keine sichtbare
Bestätigungs-/Retry-Semantik.

## Klasse 4: Wire-Budget-Kollisionen (256 KB)

Drei unabhängige Pfade liefen gegen dieselbe Grenze:

- `find({limit: 100000})` auf einer Demand-Collection: `fetch:start` ohne
  jede Antwort, Modul meldete „Store nicht verbunden" (Browser-Fix: Paginierung
  `e5c6ff821`). Ein Query, dessen Antwort das Chunk-Budget sprengt, sollte
  einen Fehler liefern, nicht schweigen.
- Business-OS-MCP: `query_records`/`get_record` auf `knowledge_tables` und
  `business_commands` scheitern regulär mit `ResponseTooLarge` (Einzeldokument
  409 KB) — die MCP-Seite kennt das Chunking der Engine nicht.
- Oversized-Dokumente in der Knowledge-Projektion (Juli-Fix
  `retain_projectable_knowledge_item` in `rxdb_peer.rs`).

## Kontextbefunde (keine Sync-Engine-Fehler, aber Betriebsrealität)

- Der OVH-Wirt zeigt intermittierend 60–70 % Paketverlust; ein Teil der
  WebRTC-/Stream-Abrisse ist Netz, nicht Engine.
- Deploy-Frequenz + 3-GB-Zustandssicherung pro Upgrade + Quellbau auf der VM
  füllten die Platte zweimal auf 100 % (Buildabbrüche im Linker). Backup-
  Rotation gehört ins Upgrade-Werkzeug.
- Client-seitiger Command-Dispatch: offene Alt-Tabs dispatchen nach einem
  Release weiter alten Payload-Bau (hier: hartes 100-Quellen-Ziel). Eine
  Versionsangabe im Command + serverseitige Ablehnung/Normalisierung veralteter
  Dispatches wäre ein sauberes Gate.

## Status der Notfixes

Alle genannten Commits liegen auf `origin/main` und laufen auf der Instanz
(Release `branch-main-20260810T082054Z`). Die Klassen 1–2 sind dort seit dem
10.08. fehlerfrei gemessen (0 `DOC_CACHE_REV`; Projektionswechsel strukturell
verifiziert am Datensatz `queue:system::14e1e28ed8cd2ee6c6577283`:
`cancelled|cancelled`). Klassen 3–4 sind unbehandelte Engine-Arbeit.
