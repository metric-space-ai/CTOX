# I-066 — Runde 1 Messbericht

## was_geaendert

- Im Arbeitsbaum wurde nichts geaendert. Es gab keine Whitelist und der Auftrag war reine Messung.
- `git diff --stat` war vor und nach der Messung identisch: **48 Dateien, 5106 Einfuegungen, 2594 Loeschungen**; das sind die bereits vorhandenen/fremden Aenderungen. Fuer die unmittelbar betroffenen Pfade blieb der Status `M src/core/context/lcm/mod.rs`, `MM src/core/mission/channels/mod.rs`, `MM src/core/service/service.rs`; `src/core/execution/agent/turn_loop.rs` blieb unveraendert.
- Nur dieser Bericht wurde ausserhalb des Repositories unter `/tmp/i-066-report.md` geschrieben.

## ursache_belegt

### Gesamturteil: ein gemeinsamer Finalisierungsfehler, aber drei verschiedene fachliche Ausloeser

Der Worker-Abschluss ist **nicht als eine atomare, autoritative Finalisierung** implementiert:

1. Der Erfolgs-Reply wird im Turn-Loop zuerst als normale Assistant-Nachricht geschrieben (`src/core/execution/agent/turn_loop.rs:869-870`, `src/core/execution/agent/turn_loop.rs:1147-1148`). Der Retry-Helper ruft dafuer `run_add_message`, also den Pfad **ohne** `AgentOutcome`, auf (`src/core/execution/agent/turn_loop.rs:1270-1280`).
2. Nur der Fehlerzweig legt spaeter explizit eine Assistant-Nachricht mit typisiertem Outcome an (`src/core/service/service.rs:6380-6395`). Die einzelne Nachrichtenoperation ist intern sauber transaktional fuer `messages` + FTS + `context_items` (`src/core/context/lcm/mod.rs:1146-1203`), sie umfasst aber weder Review, Outcome-Zeuge, Missionszustand noch Queue-Abschluss.
3. Der forensische `work.outcome` wird bereits direkt aus `Result<String>` abgeleitet (`src/core/service/service.rs:6360-6371`) und ist ausdruecklich nur best effort (`src/core/service/service.rs:5931-5955`, Fehler werden nur gezaehlt/logged in `src/core/service/harness_flow.rs:319-332`).
4. Failure-Counter/Deferral werden **vor** Completion Review und Outcome-Zeuge veraendert (`src/core/service/service.rs:6401-6465` gegenueber Outcome-Zeuge ab `src/core/service/service.rs:7043-7106` und Queue-Abschluss `src/core/service/service.rs:7313-7451`). Damit kann ein Prozess-`Ok(reply)` die Agentenfehler-Deferral loeschen, obwohl der spaetere Outcome-Zeuge den Abschluss wegen fehlendem Artefakt ablehnt. Ausserdem geht ein als transient klassifizierter `ExecutionError` wegen der Bedingung in `src/core/service/service.rs:6403-6448` ebenfalls in den Reset-Zweig.
5. Beim Timeout wird die Fortsetzung sogar unmittelbar nach dem Worker-`Err` erzeugt (`src/core/service/service.rs:6341-6353`), also vor der typisierten Fehlernachricht (`src/core/service/service.rs:6380-6395`) und vor der spaeteren Lease-Behandlung (`src/core/service/service.rs:7793-7848`). Die neue Queue-Aufgabe hat ihre eigene Transaktion (`src/core/mission/channels/mod.rs:2706-2718`).
6. Die einzelnen Queue-Acks sind zwar intern transaktional (`src/core/mission/channels/mod.rs:5397-5423`), aber nicht zusammen mit Assistant-Outcome, Mission-State, Outcome-Proof und ggf. Timeout-Fortsetzung.

Die Persistenz bestaetigt die Asymmetrie: In `runtime/ctox.sqlite3` gibt es **895 Assistant-Zeilen**, davon **731 mit `agent_outcome IS NULL`**, **0 mit `agent_outcome='Success'`** und **164 typisierte Fehler-Outcomes**. Parallel existieren **741** persistierte `work.outcome`-Projektionen, davon **614 `Success`**. Der in `src/core/service/service.rs:5934-5936` behauptete primaere dauerhafte Success-Eintrag in `messages.agent_outcome` existiert damit real nicht; nur die lossy Projektion traegt Success.

Das ist der gemeinsame Architekturfehler. Die drei Netze sind trotzdem **nicht drei Beobachtungen desselben konkreten Fehlers**: fehlendes Artefakt, harter Turn-Timeout und spaetere Erholung einer Missions-Deferral sind fachlich verschiedene Ereignisse.

### A. `outcome_witness_recovery_message`

**Welchen Zustand repariert es / wer haette richtig schreiben muessen?**

- Erwartete Artefakte werden aus Job/Prompt abgeleitet (`src/core/service/service.rs:14276-14364`) und aus Dateien bzw. Kommunikationszeilen als geliefert rekonstruiert (`src/core/service/service.rs:14366-14476`).
- Der terminale Proof wird fuer geleaste Queue-/Ticket-/Work-Entitaeten geschrieben (`src/core/service/service.rs:15158-15285`). Fehlende oder im falschen Zustand befindliche Artefakte erzeugen dauerhaft `WP-Outcome-Missing` bzw. `WP-Outcome-Wrong-State` (`src/core/core_state/guard.rs:216-290`); der Proof wird vor dem Fehler dauerhaft upserted (`src/core/core_state/guard.rs:177-207`, `src/core/core_state/guard.rs:459-469`). Null Rejections sind deshalb hier **nicht** bloss ein leerer Ringpuffer.
- Bei Ablehnung wird Feedback/Requeue/Hold erzeugt (`src/core/service/service.rs:7097-7257`, `src/core/service/service.rs:7345-7451`). Der fachliche Erzeuger des Artefakts ist der Worker; der Controller schreibt jedoch den Erfolgs-Reply bereits vorher untypisiert (`src/core/execution/agent/turn_loop.rs:1147-1148`) und klassifiziert/resetet den Turn vor dem Zeugen (`src/core/service/service.rs:6360-6465`). Die Ursache existiert also strukturell weiter.

**Feuert es real?**

- Live-DB: **6** dauerhafte Outcome-Witness-Proofs, **6 accepted / 0 rejected**, Zeitraum **2026-07-10 15:41:12 UTC bis 2026-07-14 02:38:44 UTC**.
- Alle drei verfuegbaren Update-Snapshots (`20260718T073042Z`, `20260718T101507Z`, `20260718T115230Z`) enthalten ebenfalls **6 accepted / 0 rejected** und **0** `WP-Outcome-Missing`/`WP-Outcome-Wrong-State`.
- Ergebnis: Der Zeugenpfad schreibt dauerhaft und wurde erfolgreich benutzt; fuer den eigentlichen Recovery-Zweig ist im beobachteten Bestand **kein realer Fang** belegt. Das beweist nicht, dass der Gate unnoetig ist.

### B. `queue_durable_artifact_timeout_recovery`

**Welchen Zustand repariert es / wer haette richtig schreiben muessen?**

- Es greift nur fuer timeoutende, geleaste, nicht-Mail-/nicht-Self-Work-Jobs mit deklarierten Datei-Artefakten (`src/core/service/service.rs:24875-24930`).
- Es erzeugt eine neue dauerhafte Queue-Aufgabe mit `durable_artifact_timeout_recovery=true` (`src/core/service/service.rs:24932-25017`) und einem Resume-Prompt (`src/core/service/service.rs:25455-25470`).
- Der eigentliche fehlende Schreibpfad ist die gemeinsame Finalisierung des **urspruenglichen** Versuchs: Timeout-Fortsetzung, typisiertes Outcome und Status des originalen Queue-Items werden in getrennten Operationen geschrieben (`src/core/service/service.rs:6341-6395`, `src/core/service/service.rs:7793-7848`). Ein harter Timeout selbst ist legitim; die zusaetzliche Kind-Aufgabe ist die Kompensation fuer den fehlenden atomaren/idempotenten Attempt-Abschluss.

**Feuert es real?**

- `governance_events`: **18** `turn_timeout_continuation`-Ereignisse im Zeitraum **2026-06-10 08:10:05 UTC bis 2026-07-24 06:19:00 UTC**; davon **2** mit Aktion `queued a durable artifact recovery task`, **16** unterdrueckte Spawns.
- Die zwei Recovery-Aufgaben wurden beide am **2026-07-18** fuer `business-os/appsec-pentest/F-001` erzeugt. Der Snapshot `20260718T073042Z` hatte **0**, `20260718T101507Z` und `20260718T115230Z` jeweils **1**, die Live-DB **2**: die Wirkung ist dauerhaft und zeitlich nachvollziehbar.
- Endzustand in der Live-DB: **beide Recovery-Aufgaben `failed`**; die erste hat `failure_attempt_count=5`, die zweite `0`. Von den zwei referenzierten Originalaufgaben ist eine `handled` und eine `failed`. Das Netz hat also **2 reale Timeouts gefangen, aber 0/2 Recovery-Aufgaben erfolgreich terminalisiert**.

### C. `record_agent_failure_recovery`

**Korrektur der Ausgangsthese:** Diese Funktion requeued oder repariert keinen Job. Sie schreibt nur das Governance-Gegenereignis, falls der vorherige `deferred_reason` gesetzt war (`src/core/service/service.rs:25407-25438`). Die Zustandsaenderung geschieht vorher in `reset_mission_agent_failure_count` (`src/core/service/service.rs:6447-6456`).

**Welchen Zustand repariert es / wer haette richtig schreiben muessen?**

- Der Threshold-Pfad setzt `mission_status='deferred'`, `deferred_reason`, `is_open=false`, `allow_idle=true` (`src/core/context/lcm/mod.rs:2140-2151`).
- Der Recovery-Writer setzt dagegen nur `agent_failure_count=0` und `deferred_reason=NULL`; `mission_status`, `is_open` und `allow_idle` werden nicht als zusammengehoerige Recovery-Transition restauriert (`src/core/context/lcm/mod.rs:2077-2103`).
- Zusaetzlich wird dieser Reset bereits anhand des rohen Worker-Ergebnisses und vor Review/Outcome-Zeuge ausgefuehrt (`src/core/service/service.rs:6401-6465`). Die Ursache existiert damit weiter und ist nicht bloss ein harter-Crash-Fall.

**Feuert es real?**

- Live-DB: **1** dauerhaftes `agent_failure_recovery`-Ereignis am **2026-06-21 09:18:10 UTC**, `agent_outcome='Success'`, Thread `business-os/creator`.
- Dem stehen **27** `agent_failure_threshold`-Ereignisse fuer **11** Conversations gegenueber; **16/27** wurden nach bereits erreichtem Threshold mit Failure-Count **3 oder 4** erneut geschrieben.
- Aktueller Mission-State: **10** Zeilen haben `deferred_reason='agent_failure_threshold'`, aber **alle 10** verletzen die heute vom Deferral-Writer beabsichtigte Kombination `mission_status='deferred' AND is_open=0 AND allow_idle=1`: acht sind `active/is_open=1/allow_idle=0`, zwei `deferred/is_open=0/allow_idle=0`.
- Ergebnis: Das Recovery-Ereignis hat **1-mal real gefeuert**. Der Missionszustand ist aber nicht als atomare Defer/Recover-State-Machine geschrieben; das Netz ist derzeit nicht ueberfluessig.

## kompensationen_geloescht

- Keine. Reine Messung; keine Datei wurde geaendert.

## verblieben

1. **Outcome-Witness-Gate bleibt legitim.** Es ist die terminale Invariante gegen falsche Completion (`src/core/core_state/guard.rs:216-290`). Nach einem atomaren Finalizer koennte der separate Recovery-Prompt entfallen, indem dasselbe durable Queue-/Work-Item im Zustand `missing_artifact/rework_required` bleibt. Der Gate selbst darf nicht entfernt werden. Realer Recovery-Fang im beobachteten Bestand: **0**.
2. **Timeout-Behandlung bleibt legitim**, weil ein harter Runtime-Timeout nicht durch Atomizitaet verschwindet (`src/core/service/service.rs:24875-24884`). Obsolet werden kann nur der Spawn einer separaten Recovery-Aufgabe, wenn der urspruengliche Attempt atomar als `timed_out + resumable/pending` abgeschlossen wird. Reale Spawns: **2**, erfolgreiche Recovery-Aufgaben: **0**.
3. **Agent-Failure-Recovery bleibt als echte Zustandsueberleitung/Auditpaar legitim**, bis Defer und Recover vollstaendig und atomar modelliert sind. Die aktuelle Funktion ist nur das Audit nach einem partiellen Reset (`src/core/context/lcm/mod.rs:2095-2098`, `src/core/service/service.rs:25415-25438`). Reale Recovery-Ereignisse: **1**; aktuell inkonsistente Deferral-Zeilen: **10**.
4. Ein externer Datei-Write kann nicht gemeinsam mit SQLite committed werden. Runde 2 braucht deshalb einen dauerhaften `finalizing`/Attempt-Datensatz und idempotente Wiederaufnahme nach Crash; die terminale SQLite-Transition darf erst nach erneutem Artefakt-Check erfolgen. Der heutige Code hat nur getrennte lokale Transaktionen fuer Nachricht (`src/core/context/lcm/mod.rs:1161-1203`), Proof (`src/core/core_state/guard.rs:177-207`) und Queue-Ack (`src/core/mission/channels/mod.rs:5397-5423`).

## tests

Alle Cargo-Aufrufe nutzten `CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline-targets/I-066`.

1. `cargo fmt --check` — **PASS**, Exit 0. Dieses Kommando erzeugt keine `test result`-Zeile.
2. `cargo check --bin ctox` — **PASS**, Exit 0: `Finished dev profile ... in 14m 15s`; **417 Warnungen** aus dem bereits veraenderten Arbeitsbaum. Dieses Kommando erzeugt keine `test result`-Zeile.
3. Targetierter Versuch mit erlaubtem Filter `outcome_witness_`: `cargo test --bin ctox outcome_witness_ -- --nocapture`. Die Test-Binaerkompilierung erreichte nach **82 Minuten** noch keinen Testlauf und damit **keine `test result`-Zeile**; bei **Load 88.83 auf 10 CPUs** wurde nur dieser eigene Cargo-Prozess beendet (Exit 144), um den geteilten Checkout nicht weiter zu belasten. Es werden ausdruecklich **nicht null Treffer** behauptet.
4. Die Filter `durable_artifact_timeout` und `agent_failure_` wurden danach nicht gestartet, weil sie dieselbe noch nicht fertiggestellte Test-Binaerkompilierung unter der gemessenen geteilten Last erneut fortgesetzt haetten. Daher gibt es dafuer keine Trefferzahl und keine `test result`-Zeile.

Es gab keinen roten Test und deshalb keine Clean-HEAD-Rotmengen-Gegenueberstellung; der Testlauf selbst begann nicht.

## gegenprobe

- Entfaellt laut Auftrag (reine Messung).
- Es gab keinen Fix und daher nichts zurueckzubauen.
- `git diff --stat` war vor und nach der Arbeit identisch: **48 Dateien, 5106 Einfuegungen, 2594 Loeschungen**. Die vorhandenen Fremdaenderungen wurden nicht beruehrt.

## offene_bedenken

1. Die dauerhaften Recovery-/Outcome-Daten in `runtime/ctox.sqlite3` enden je nach Tabelle am **2026-07-24**, obwohl die Datei am 2026-08-06 weiter beschrieben wurde. Fuer den Zeitraum 2026-07-25 bis 2026-08-06 ist daher kein Worker-Recovery-Verkehr belegt; die gemessenen Nullen duerfen nicht auf diesen Zeitraum extrapoliert werden.
2. Die drei verfuegbaren vollwertigen `ctox.sqlite3`-Snapshots stammen alle vom **2026-07-18**. Sie bestaetigen die zeitliche Entstehung der ersten Timeout-Recovery und die **0** rejected Outcome-Witness-Proofs, erweitern das Beobachtungsfenster aber nicht ueber die Live-DB hinaus.
3. `src/core/service/service.rs`, `src/core/mission/channels/mod.rs` und `src/core/context/lcm/mod.rs` tragen bereits fremde Aenderungen. Runde 2 muss diese drei Dateien konfliktbewusst bearbeiten; der gemessene Finalisierungsfehler liegt in genau diesem aktuell veraenderten Pfad.
4. Die Erfolgszahl **614** in `ctox_harness_flow_events` ist keine autoritative Terminalzahl: der Writer ist best effort (`src/core/service/harness_flow.rs:319-332`) und wird vor Review/Outcome-Zeuge aufgerufen (`src/core/service/service.rs:6360-6371`).

## pfade

Fuer Runde 2 sind ausserhalb der leeren Whitelist mindestens diese Pfade erforderlich; in Runde 1 wurde deshalb nur gemessen:

1. `src/core/execution/agent/turn_loop.rs:869-870`, `src/core/execution/agent/turn_loop.rs:1147-1148`, `src/core/execution/agent/turn_loop.rs:1270-1280`
   - Erfolgs-Assistant-Write nicht mehr als untypisierten, bereits abgeschlossenen Turn behandeln. Entweder typisierten `execution_finished/pending_finalization`-Datensatz schreiben oder den autoritativen Outcome-Write an den Finalizer uebergeben.
2. `src/core/service/service.rs:6341-6465`, `src/core/service/service.rs:7043-7451`, `src/core/service/service.rs:7793-7848`, `src/core/service/service.rs:24875-25017`, `src/core/service/service.rs:25407-25438`
   - Einen idempotenten Worker-Attempt-Finalizer einfuehren: Artefaktcheck/Proof, typisiertes finales Outcome, Mission Failure/Recovery, Queue-/Work-Status und Timeout-Resume in definierter Reihenfolge; `Success` und Recovery erst nach angenommenem terminalen Proof.
3. `src/core/context/lcm/mod.rs:1146-1203`, `src/core/context/lcm/mod.rs:2077-2103`, `src/core/context/lcm/mod.rs:2140-2151`
   - Assistant-Outcome-API fuer Finalisierung bereitstellen und Mission Defer/Recover als vollstaendige Transition schreiben; beim Recover nicht nur Counter/Reason, sondern auch die dazugehoerigen Control-Felder konsistent behandeln.
4. `src/core/mission/channels/mod.rs:5397-5423`, `src/core/mission/channels/mod.rs:2706-2718`
   - Transaktionsfaehige Queue-APIs fuer den gemeinsamen Finalizer bereitstellen; Timeout moeglichst am urspruenglichen Queue-Attempt persistieren statt vor dessen Abschluss eine unabhaengige Kind-Aufgabe zu erzeugen.
5. `src/core/core_state/guard.rs:177-207`, `src/core/core_state/guard.rs:216-290`, `src/core/core_state/guard.rs:459-469`
   - Den vorhandenen dauerhaften Outcome-Proof im Finalizer weiterverwenden. Der Guard ist nicht das zu loeschende Netz; er ist die terminale Invariante und nimmt bereits eine DB-Connection entgegen.
