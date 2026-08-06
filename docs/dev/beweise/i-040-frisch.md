# I-040 Report

## was_geaendert

- `src/apps/business-os/shared/command-bus.js:1129-1169`
  - Der Receipt wird jetzt aus den nativen Lifecycle-Feldern abgeleitet, bevor Legacy-`status` ausgewertet wird.
  - `execution_phase` ist fuer nicht-terminale native Phasen kanonisch; bei `execution_phase=terminal` gilt `terminal_status`.
  - Fuer den gemessenen Legacy-Fall `replication_phase=native_observed` plus `status=pending_sync` liefert der Bus eindeutig `status=accepted`.
  - Ein Legacy-`task_status=pending_sync` wird im Receipt ebenfalls nicht neben einer bereits erreichten nativen Annahme ausgegeben, sondern auf den kanonischen Receipt-Status normalisiert.
- `src/apps/business-os/modules/documents/index.js:2898`
  - Der Dokument-Submit verwendet direkt `state.ctx.commandBus.dispatch(command)`.
- `src/apps/business-os/modules/outbound/index.js:6478`
  - Der Outbound-Submit verwendet direkt `state.ctx.commandBus.dispatch(updateCommand, { timeoutMs: 5000 })`.
- `src/apps/business-os/shared/command-bus-receipt.test.mjs:22-76`
  - Repo-lokaler Verhaltenstest fuer genau den gemessenen Widerspruch.

## ursache_belegt

Vor der Reparatur war der neue Repo-Test rot:

- `node --test src/apps/business-os/shared/command-bus-receipt.test.mjs`
- Ergebnis: `tests 1`, `pass 0`, `fail 1`.
- Abweichung: Receipt `actual='pending_sync'`, erwartet `accepted`.
- Gleichzeitig enthielt der beobachtete Datensatz weiterhin unveraendert `status='pending_sync'` und `replication_phase='native_observed'`; damit belegt der Test, dass der Fehler in der Receipt-Ableitung lag und nicht im Testdatensatz versteckt wurde.

Entscheidung: `native_observed` gilt. Diese Phase belegt, dass die native Seite den Command bereits gesehen und die Ownership uebernommen hat. `pending_sync` ist in diesem Fall nur noch der alte Browser-Intent-Status und darf nicht als aktuelle Warte-Aussage in den angenommenen Receipt durchschlagen. Wenn eine genauere native `execution_phase` vorhanden ist, ist diese genauer als das generische `accepted` und wird deshalb bevorzugt.

## kompensationen_geloescht

- `dispatchDocumentCommandWithBackendFallback` entfernt.
- `waitForBusinessCommandProjection` samt 12-Sekunden-Polling und lokalem `delay` entfernt.
- `dispatchBusinessCommandWithRxdbFallback` entfernt.
- Abschlusspruefung: Suche nach den drei Kompensationssymbolen ergab `fallback symbols removed: 3` und keine verbleibenden Treffer.

## verblieben

Keine der zwei benannten Modulkompensationen wird nach der Bus-Reparatur noch gebraucht.

## tests

Alle Cargo-Aufrufe erfolgten mit `CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline-targets/I-040`.

1. `cargo fmt --check`
   - Final: Exit 0, keine Format-Diffs (`0` Befunde).
   - Der Befehl ist kein Test-Runner und erzeugt daher keine `test result`-Zeile.
2. `cargo check --bin ctox`
   - Final: Exit 0.
   - Abschlusszeile: `Finished dev profile [unoptimized + debuginfo] target(s) in 8m 49s`.
   - Geprueftes Binary-Ziel: `1` (`ctox`); `401` Warnungen, keine Fehler.
   - Die ersten zwei Laeufe erreichten das 10-Minuten-Limit waehrend des initialen Dependency-Aufbaus; der dritte Lauf auf demselben vorgeschriebenen Target-Verzeichnis schloss erfolgreich ab.
   - Der Befehl ist kein Test-Runner und erzeugt daher keine `test result`-Zeile.
3. `node --test src/apps/business-os/shared/command-bus-receipt.test.mjs`
   - Ergebniszeilen: `tests 1`, `pass 1`, `fail 0`.
4. `node --test src/apps/business-os/shared/command-bus.test.mjs`
   - Ergebniszeilen: `tests 22`, `pass 22`, `fail 0`.
5. Zusaetzlich: `node --check` fuer alle drei geaenderten `.js`-Quelldateien: `3/3` ohne Syntaxfehler.

Es wurde kein `cargo test` aufgerufen; die Cargo-Filterregel ist daher nicht anwendbar. Die beiden Node-Aufrufe verwenden jeweils den exakten Testdateipfad als Selektor und liefen mit `1` beziehungsweise `22` Treffern, also keine Null-Treffer-Laeufe und keiner der verbotenen Cargo-Filter.

## gegenprobe

Der Widerspruch wurde in `commandReceiptStatus` testweise wiederhergestellt, indem der native-observed-Zweig erneut `pending_sync` zurueckgab.

- `node --test src/apps/business-os/shared/command-bus-receipt.test.mjs`
- Gegenprobe: `tests 1`, `pass 0`, `fail 1`, Exit 1.
- Konkrete Abweichung: `actual='pending_sync'`, `expected='accepted'` in Zeile 72.

Danach wurde exakt zur reparierten Fassung zurueckgebaut:

- SHA-256-Pruefung: `4/4` Whitelist-Dateien `OK` gegen die vor der Gegenprobe aufgenommenen Hashes.
- `git diff --stat` danach:
  - Trackte Dateien: `3 files changed, 19 insertions(+), 64 deletions(-)`.
  - Neuer Test separat: `1 file changed, 76 insertions(+)`.

## offene_bedenken

Keine task-spezifischen offenen Bedenken. Der Checkout enthaelt weiterhin zahlreiche fremde, bereits vorhandene bzw. parallel entstandene Aenderungen ausserhalb der Whitelist; sie wurden nicht angefasst. Die finalen Pflichtpruefungen `cargo fmt --check`, `cargo check --bin ctox` und beide Node-Suiten sind in der aktuellen Endfassung gruen.

## pfade

Noetig und geaendert, ausschliesslich innerhalb der Hard Whitelist:

- `src/apps/business-os/shared/command-bus.js:1129-1169`
- `src/apps/business-os/modules/documents/index.js:2898`
- `src/apps/business-os/modules/outbound/index.js:6478`
- `src/apps/business-os/shared/command-bus-receipt.test.mjs:22-76`

Keine weiteren Repo-Pfade werden fuer diese Reparatur benoetigt.
