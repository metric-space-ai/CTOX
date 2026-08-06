# SYNC-F: service.rs — der angekündigte eigene Feldzug

Nachfolger von SYNC-D (Kompensationen, abgeschlossen) und SYNC-E (Migrationen,
abgeschlossen). `service.rs` stand seit dem ersten Dichteplan als „eigener
Feldzug" im Text und wurde nie begonnen: ~50 Kompensations-Funktionen auf
26.000 Zeilen, 9 unabhängige Belange.

## Belang-Inventar (nachgemessen 05.08.)

1. **App-Recovery (22 Fn)** — VERTAGT mit Bedingung (I-053): fällt erst, wenn
   `queue.ack_failed` über Zeit null zeigt oder die Pro-Key-Reihenfolge steht.
2. **State-Invariant-Repair** (`attempt_state_invariant_repair`, :1659) +
   Lease-Preservation (:2054) → Messung I-064.
3. **Stalled-Founder/External-Chat-Repair** (:20246, :20607, :21096) → I-065.
4. **Outcome-Witness- + Artifact-Timeout- + Agent-Failure-Recovery**
   (:15291ff, :24916ff, :25407) → I-066.
5. **Ticket-Reconcile** (`reconcile_ticket_runtime_state`, :19061 + Gates
   :17302ff) → I-067.
6. **CV-Print-Parser-Recovery** (:9667–:9782) → I-068.

## Verfahren

Wie in SYNC-D/E: zwei Runden je Belang (erst Ursache mit datei:zeile und
Persistenz-Zahlen, dann Fix mit korrekter Whitelist); die Lehren gelten
weiter — null beweist nur bei dauerhaft schreibenden Pfaden Totsein, eine
Doku-Behauptung ist keine Messung, Gegenprobe gegen das ausgeführte Artefakt,
`--tests` nach Signaturänderungen, Index-Diff-Pflicht vor commit-tree.

Runde 1 läuft parallel: I-064/065/066 auf Sol, I-067/068 auf Grok — reine
Messungen im geteilten Checkout, keine Worktrees, keine Builds nötig.

## Runde-1-Ergebnis (06.08. früh, alle fünf Messungen vollständig)

**Der Kernbefund (I-066): ein gemeinsamer Finalisierungsfehler, drei Auslöser.**
Der Worker-Abschluss ist keine atomare Transition. Der Erfolgs-Reply trägt kein
typisiertes Outcome — 895 Assistant-Zeilen, **0 mit `Success`**, 731 NULL; nur
die ausdrücklich verlustbehaftete `work.outcome`-Projektion trägt Erfolg (614).
Failure-Zähler werden VOR Review und Outcome-Zeuge zurückgesetzt; die
Timeout-Fortsetzung entsteht VOR der typisierten Fehlernachricht; Nachricht,
Proof und Queue-Ack sind je für sich transaktional, nie gemeinsam. Gemessene
Folgen: 10 inkonsistente Deferral-Zeilen (10/10 verletzen die beabsichtigte
Zustandskombination), 2 reale Timeout-Fänge mit 0 erfolgreichen Recoveries.

**Die übrigen Befunde:** Ticket-Reconcile legitim, aber Queue-Hälfte doppelt
zum 60s-Sweep und der Sweep ohne Audit (GROK-6). CV-Recovery legitim, aber
Fehlertext-Gate statt typisiertem Kanal und Ringpuffer-Telemetrie (GROK-7).
Stalled-Communications: Schreibpfade vorhanden und bewacht, Instanz hat null
Founder-Verkehr, Telemetrie flüchtig (I-065). State-Invariant-Repair feuert
real (54 Ereignisse) — Ursache ist fehlender Mission-Seed bei Queue-Anlage
plus Import-Split-Brain, 662 Zeilen; alte Review-Behauptung widerlegt (I-064).

## Runde 2 (Reihenfolge, `service.rs`-Kollisionen sequenziell)

1. **I-070** (läuft): Mission-Seed-Root-Fix — eliminiert die 54 Vorher-Zustände.
2. **I-071**: der atomare Attempt-Abschluss — dauerhafter Finalisierungs-
   Datensatz, typisiertes Success-Outcome, Zeuge VOR Zähler-Reset, idempotente
   Wiederaufnahme. Das große Los; danach können Timeout-Kind-Aufgabe und
   Recovery-Prompt zu Zustandsübergängen desselben Items schrumpfen.
3. **I-072**: dauerhafte Repair-Telemetrie — eine Mechanik für CV-Recovery,
   Preserve-Lease, Founder-Repair (heute Ringpuffer).
4. **I-073**: Sweep bekommt Audit; danach fällt die Queue-Hälfte aus dem
   Ticket-Reconcile (Dedupe).
5. **I-074**: CV-Gate typisiert statt Fehlertext-Substring.
6. **GROK-8** (läuft, unabhängig): die neun dauerhaft roten Browser-Smokes
   klassifizieren — Umgebung, veraltet oder echte Regression.
