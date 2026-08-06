# S-01 — RUNDE 1 (nur messen): warum startet der native Sync-Peer 78-mal neu?

Messdatum: 2026-08-06  
Arbeitsbaum (read-only): `/Users/michaelwelsch/Documents/ctox`  
Laufende Instanz: `/Users/michaelwelsch/.local/lib/ctox/current` → `releases/branch-main-20260724T072316Z`  
Log: `/Users/michaelwelsch/.local/lib/ctox/current/runtime/ctox_service.log` (556 388 Zeilen, ~30 MiB)  
Primärquelle Code: `src/core/business_os/rxdb_peer.rs`

Hinweis Messbasis: Die **laufende Binary** (Release 2026-07-24) loggt noch
`multiplexed WebRTC replication up for …` und `critical child exited`.
Der **Checkout** formuliert dasselbe als `pool ready for …` (rxdb_peer.rs:2568)
bzw. `critical task exited` (rxdb_peer.rs:1838). Architektur/Schwellen sind
dieselben; Zeilennummern unten beziehen sich auf den Checkout.

Disk zum Messzeitpunkt: APFS Data-Volume **100 %** (`df`: 6.3 Gi avail von 926 Gi,
Capacity 100 %). Status-Heartbeats schreiben zeitweise wieder (`updated_at_ms`
bewegt sich), `replicationUp=false`, `dataChannelOpen=false` im aktuellen
`business-os-rxdb-peer.status.json`.

---

## Messzahlen (Log)

### Letzte 80 000 Logzeilen (Fenster)

| Muster | n |
|---|---:|
| `multiplexed WebRTC replication up for` (Bring-up OK) | **78** |
| `native rxdb peer status heartbeat failed` | **248** |
| `No space left on device` | **258** |
| `status.status.json` (doppelte Endung im Temp-Pfad) | **233** |
| `watchdog: heartbeat stale` | **18** |
| `exited after stale heartbeat; respawning in` | **18** |
| `critical child exited` / `critical children exited` | **4 + 1 = 5** |
| `runtime app schemas changed` (Watch + Reconfig-Meldungen) | **22** |
| `sync config changed` | **4** |
| `native rxdb peer failed` / bring-up failed|timeout | **0** in diesem Fenster |

Vorherige Exit-Ursache je erfolgreichem Bring-up im Fenster (Lookback ≤200 Zeilen):

| prior cause | n / 78 |
|---|---:|
| unknown_prior (langer Abstand / Fensterstart) | 41 |
| after_stale_heartbeat | 16 |
| after_schema_change | 11 |
| after_critical_child | 4 |
| after_lock | 4 |
| after_sync_config | 2 |

### Gesamtes Log

| Muster | n |
|---|---:|
| `multiplexed WebRTC replication up for` | 307 |
| `native rxdb peer failed` | 509 |
| bring-up failed / timed out | 425 / 56 |
| heartbeat failed | 307 |
| No space left on device | 320 |
| exited after stale heartbeat | 29 |
| watchdog: heartbeat stale | 30 |
| critical child exited | 23 |
| runtime app schemas changed; reconfiguring | 12 |
| sync config changed; reconfiguring | 2 |
| lock held by another | 219 |

Stale-Heartbeat-Backoff im Fenster (sek):  
`5,5,20,5,5,5,5,5,10,20,40,10,20,5,5,10,20,40` — wächst bis 40 s, springt
nach längeren Pausen wieder auf 5 s, **nicht** auf 0 und **nicht** dauerhaft
zurück nach jedem erfolgreichen Bring-up.

Letzte Stale-Altersangaben (`watchdog: heartbeat stale (Some(N) ms)`), n=30 im
ganzen Log; letzte Kette:  
`92231, 97518, 126800, 160797, 206085` ms.  
Δ der letzten 4 Kills: `5287, 29282, 33997, 45288` ms ≈ Backoff 5/10/20/40 s plus
Watchdog-/Bring-up-Overhead, bei **eingefrorenem** `updated_at_ms` (Statusdatei
wurde zwischen den Kills nicht erfolgreich neu geschrieben).

---

## 1. Auslöser-Kette — wer entscheidet den Neustart?

### Owner

**`spawn_native_peer`** (`rxdb_peer.rs:1710–1901`) spawnt den Supervisor-Thread
`business-os-rxdb-peer`. Jeder Lauf ist `runtime.block_on(run_native_peer(…, true))`
(`:1784–1790`). Nach dem Exit:

1. Lifecycle → `supervised_run_exited` (`:1791–1794`, Maschine `:362–372`)
2. Runtime-Shutdown mit Timeout 10 s (`:1800–1803`, Const `:777`)
3. Circuit-Breaker Success/Failure (`:1804–1813`)
4. Match auf `NativePeerExit` / `Err` → Log + Sleep + Delay-Verdopplung
   (`:1815–1887`)
5. Loop → nächster Bring-up

`NativePeerExit` (`:646–663`):

| Variante | Bedeutung | Respawn? |
|---|---|---|
| `StopRequested` | absichtlicher Stop | nein (break) |
| `HeartbeatStale` | Watchdog: Status-HB zu alt | ja + backoff |
| `SyncConfigurationChanged` | Sync-Room/URLs/Passwort | ja, **sofort** |
| `RuntimeSchemasChanged` | installierte Module-Schemas | ja, **sofort** |
| `CriticalTaskExited` | kritisches Kind beendet | ja + backoff |
| `ProgressStalled` | durable backlog ohne Fortschritt | ja + backoff |
| `ProcessLockUnavailable` | fremder Lock | ja + backoff |
| `Err(_)` (Bring-up u.a.) | fataler Lauf-Fehler | ja + backoff (+ Circuit) |

### Wo der Lauf endet: Watchdog in `run_native_peer`

Nach erfolgreichem Pool-Bring-up läuft **eine** `tokio::select!`-Schleife
(`:2735–2861`) mit drei Armen:

**A. `shutdown_rx`** (`:2737–2738`) → `exit = StopRequested` (Default `:2732`).

**B. `watchdog.tick()`** alle **15 s** (`NATIVE_PEER_WATCHDOG_INTERVAL_SECS=15`,
`:768`, Interval `:2725–2727`), Reihenfolge **fest**:

1. **Critical children** (`:2742–2750`): `peer.finished_critical_tasks()` nicht
   leer → Log `critical children exited (…)` → `NativePeerExit::CriticalTaskExited`.
2. **Progress stall** (`:2752–2800`):
   - Warn ab `NATIVE_PEER_PROGRESS_WARN_AGE_MS = 60_000` (`:782`)
   - Zähler ab `NATIVE_PEER_PROGRESS_RESPAWN_AGE_MS = 180_000` (`:783`)
   - Kill nach `NATIVE_PEER_PROGRESS_RESPAWN_TICKS = 2` (`:784`) → `ProgressStalled`
3. **Heartbeat stale** (`:2802–2816`):
   - liest **nur** `updated_at_ms` aus `business-os-rxdb-peer.status.json`
     via `read_native_peer_heartbeat` (`:3263–3266`, Pfad `:3259–3261`)
   - `wedged` wenn age > `NATIVE_PEER_WATCHDOG_MAX_HEARTBEAT_AGE_MS = 90_000`
     **oder Datei/Feld fehlt** (`unwrap_or(true)`, `:780–781`, `:2806–2808`)
   - → `NativePeerExit::HeartbeatStale`
4. **Sync-Config-Change** (`:2818–2837`) → `SyncConfigurationChanged`

**C. `runtime_schema_watch.tick()`** alle **5 s** (`:772`, `:2728–2730`,
`:2840–2859`) → `RuntimeSchemasChanged`.

Bring-up-Timeout ist **kein** Watchdog-Exit, sondern `Err` vor Running:

- Const `NATIVE_COLLECTION_BRINGUP_TIMEOUT_SECS = 20` (`:670`)
- `timeout` um die Multiplex-Bring-up-Task (`:2553–2599`)
- Fehler über `native_peer_bring_up_failure` (`:672–675`) → Supervisor loggt
  `native rxdb peer failed` (`:1857–1863`)

### Zusammenspiel / gegenseitiges Triggern

- **Ein Lauf, ein Exit:** die select-Schleife bricht beim ersten Treffer ab;
  mehrere Exit-Gründe feuern **nicht parallel** im selben Lauf.
- **Thrash-Kopplung über den Supervisor:**
  1. Platte voll → HB-Write schlägt fehl (`:3411–3414`) → `updated_at_ms` friert
  2. nach ≥90 s (+ Watchdog-Raster 15 s) → `HeartbeatStale` → Tear-down
  3. Supervisor wartet `delay` (5…300 s) → neuer Bring-up (195 Collections)
  4. während/nach Bring-up wieder ENOSPC → erneut Stale → Backoff verdoppelt
     **ohne** 600‑s-Healthy-Reset (s. §3)
- **Schema-Change** umgeht den Backoff (`immediate_reconfigure`, `:1870–1883`)
  und kann Bring-ups in dichter Folge erzeugen (im Fenster 11× nach Schema;
  zuletzt 178→195 Collections in unmittelbarer Folge, Log ~556310–556325).
- **Critical child** ist unabhängiger zweiter Kill-Pfad (z. B.
  `desktop_file_index`, Log 526220); im Fenster 4×, nicht der Haupttreiber der 78.
- **Circuit-Breaker** (`:501–503`, `:594–614`) greift nur bei `Err` mit
  Signaling-/Bring-up-Klassifikation (`classify_native_peer_failure`, `:3063+`);
  `HeartbeatStale` ist `Ok(_)` und ruft **`record_success`** auf (`:1810–1813`).
  Disk-Full-Stale **öffnet den Circuit nicht**.

**Antwort:** Der Supervisor in `spawn_native_peer` entscheidet den Neustart.
Die Auslöser im Lauf sind (1) Heartbeat-Stale-Watchdog, (2) Critical-Child-
Watchdog, (3) Progress-Stall-Watchdog, (4) Sync-Config-Change, (5) Runtime-
Schema-Change, (6) Bring-up-`Err`/Timeout. Im Messfenster dominieren
**(1) Stale-HB wegen ENOSPC** und **(5) Schema-Reconfigs**; sie triggern sich
nicht gegenseitig im selben Tick, aber **verkettet über Respawn + volle Platte**.

---

## 2. Heartbeat als Auslöser — tötet Schreibfehler einen gesunden Peer?

### Belegt: ja

**Schreiben (dedizierter OS-Thread, nicht Tokio):**

- Start vor DB-Open/Bring-up: `spawn_native_peer_status_heartbeat` (`:2385–2389`,
  `:3394–3433`)
- Intervall 5 s (`NATIVE_PEER_HEARTBEAT_INTERVAL_SECS`, `:756`)
- Bei Fehler: **nur** `eprintln!(… heartbeat failed …)` (`:3411–3414`), Thread
  läuft weiter. Es gibt **keinen** „disk full“-Exit im Heartbeat-Thread.
- Payload-Write: `write_native_peer_heartbeat` (`:3299–3360`) — atomic write+rename.

**Watchdog-Leseseite (unterscheidet nicht Ursache):**

```text
rxdb_peer.rs:2802-2816
  age = now - heartbeat.updated_at_ms   // aus status.json
  wedged = age > 90_000  ||  updated_at fehlt
  → NativePeerExit::HeartbeatStale
```

Kein Zweig für:

- Heartbeat-Thread noch alive (`lifecycle.heartbeat_thread_alive` existiert
  im Lifecycle `:95–106`, `:258–273`, wird in Status-JSON unter
  `criticalTasks`/`status_heartbeat` gespiegelt, aber **nicht** für den Kill
  ausgewertet),
- ENOSPC / Write-Fehler,
- „Peer ansonsten gesund“ (Pools, critical tasks, replication signals).

`unwrap_or(true)` bei fehlender Datei/Feld (`:2808`) = „kein Status ⇒ tot“.

### Log-Kette (kanonisch, Zeilen 555937–555965 u.ä.)

1. Dutzende  
   `status heartbeat failed: … business-os-rxdb-peer.status.status.json.tmp: No space left on device`
2. dann  
   `watchdog: heartbeat stale (Some(97518) ms); shutting down for a supervised respawn`
3. dann  
   `exited after stale heartbeat; respawning in 5s`
4. dann wieder  
   `multiplexed WebRTC replication up for 195 collections…`

Der Peer hatte den Multiplex-Pool **gerade erst** hochgefahren; die Kill-
Entscheidung basiert ausschließlich auf dem Alter der Statusdatei, die wegen
ENOSPC nicht mehr aktualisiert werden konnte.

**Schwellen (Kommentar `:778–780` nennt 90 s „generously above“ 5 s-Write und
30 s-TTL):**

| Const | Wert | Zeile |
|---|---:|---:|
| `NATIVE_PEER_HEARTBEAT_INTERVAL_SECS` | 5 | 756 |
| `NATIVE_PEER_HEARTBEAT_TTL_MS` | 30 000 | 757 |
| `NATIVE_PEER_WATCHDOG_INTERVAL_SECS` | 15 | 768 |
| `NATIVE_PEER_WATCHDOG_MAX_HEARTBEAT_AGE_MS` | 90 000 | 781 |

**Fazit §2:** Der Watchdog unterscheidet **nicht** „Peer tot“ vs. „Statusdatei
nicht schreibbar“. Ein reiner Status-Schreibfehler (Platte voll) lässt
`updated_at_ms` veralten und tötet einen ansonsten laufenden Peer nach ~90 s+.

---

## 3. Fehlerzähler / Backoff — Reset nach erfolgreichem Bring-up?

### Respawn-Delay `delay` (Supervisor)

```text
rxdb_peer.rs:1737          delay = 5s   (NATIVE_PEER_RESPAWN_BASE_DELAY_SECS)
:1866-1868                 reset auf 5s NUR wenn started_at.elapsed() >= 600s
                           (NATIVE_PEER_RESPAWN_HEALTHY_RUN_SECS = 600, :500)
:1870-1883                 immediate_reconfigure (Schema|SyncConfig):
                             kein Sleep; delay := 5s
:1874-1880                 sonst: sleep(native_peer_retry_delay(delay))  // +jitter :3116-3120
:1884-1886                 sonst: delay = min(delay*2, 300s)
```

**Erfolgreicher Bring-up allein resettet `delay` nicht.**  
Reset nur wenn der Lauf **≥ 600 s** durchgehalten hat.  
Stale-Kills bei ~90–206 s Laufzeit ⇒ Delay wächst 5→10→20→40… (Log belegt).

### Circuit-Breaker (separat, nur `Err`)

| Const | Wert | Zeile |
|---|---:|---:|
| `NATIVE_PEER_CIRCUIT_FAILURE_THRESHOLD` | 5 | 501 |
| `NATIVE_PEER_CIRCUIT_OPEN_SECS` | 120 | 502 |
| `record_success` | consecutive_failures=0 | 586–592 |
| `record_failure` | +1, open ab threshold/permanent/half-open fail | 594–614 |

`HeartbeatStale` / `CriticalTaskExited` / Schema → `Ok(exit)` →
`record_success` (`:1810–1813`) — **kein** Failure-Zähler für Disk-Full-Stale.

### Lifecycle-Maschine

`supervisor_start` / `begin_run` / `supervised_run_exited` (`:227–255`, `:362+`)
halten Phase/Ownership; **kein** Failure-Counter dort.

**Antwort §3:** Ja — der Respawn-Backoff `delay` setzt sich nach Bring-up **nicht**
zurück, nur nach ≥600 s Laufzeit oder immediate reconfigure. Das verlängert die
ENOSPC-Thrash-Pause schrittweise, ohne die Ursache zu heilen.

---

## 4. Doppelte Endung `.status.status.json`

### Entstehung (datei:zeile)

```text
rxdb_peer.rs:3259-3261  native_peer_heartbeat_path
  → root/runtime/business-os-rxdb-peer.status.json

rxdb_peer.rs:3347
  let temporary_path = path.with_extension("status.json.tmp");
```

Rust `Path::with_extension` ersetzt **nur die letzte Extension** (nach dem
letzten `.`):

| Eingabe | with_extension(…) | Ergebnis |
|---|---|---|
| `…peer.status.json` | `"status.json.tmp"` | **`…peer.status.status.json.tmp`** |
| zum Vergleich korrekt in derselben Datei `:4142` | `"json.tmp"` auf `….json` | `….json.tmp` |

Log belegt exakt diesen Temp-Pfad + ENOSPC (248× im Fenster).  
Zusätzlich tritt `failed to publish … business-os-rxdb-peer.status.json`
auf (`rename`-Fehler, `:3354–3358`) — Finalpfad ist korrekt, Temp-Name falsch.

### Kosmetisch oder echt?

| Pfad | Verwendung | Korrekt? |
|---|---|---|
| `…peer.status.json` | final read+publish (`:3260`, `:3264`, `:3354`) | ja |
| `…peer.status.status.json.tmp` | nur atomic temp write (`:3347–3353`) | **nein (Bug)** |

- **Kein** Leser sucht die doppelte Endung. Watchdog/Status-API lesen den
  Finalpfad ohne Doppelung (`:2802–2804`, `:3263–3266`, `:1403+` /
  `store.rs:2104`).
- Die Doppelung **allein** erklärt kein „Datei nie gefunden“.
- Sie ist aber ein **echter Pfadbau-Fehler** (falsche `with_extension`-Nutzung),
  der Diagnose erschwert und von der **richtigen** Variante `:4142`
  (`with_extension("json.tmp")`) abweicht.
- Bei ENOSPC scheitert der Write auf dem (falsch benannten) Temp; `updated_at_ms`
  bleibt alt → Stale-Kill. Dieselbe ENOSPC würde auch bei korrekt benanntem
  Temp greifen — die Doppelung ist **nicht die Plattenursache**, aber der
  sichtbare Symptom-Pfad im Log.

---

## 5. Kosten je Neustart (195 Collections)

### Was ein Bring-up tut (Code)

Pro Supervisor-Iteration (`run_native_peer`, ab `:2335`):

1. Lifecycle `begin_run` (`:2342`)
2. Process-Lock (`:2344–2346`)
3. Heartbeat-Thread starten (`:2385–2389`)
4. Schema-Repair + SQLite/RxDB öffnen (groß: `business-os-rxdb.sqlite3` ≈ **2.1 Gi**)
5. Collections laden (Log: 195, zwischenzeitlich 178 bei Schema-Glitch)
6. **Ein** multiplexter WebRTC-Pool, Timeout **20 s** (`:2530–2600`, `:670`)
7. Demand-File-Sources registrieren (`:2577`)
8. Kritische Background-Tasks neu spawnen (`:2604–2676`): Commands, Outbox,
   Notes, Desktop-File-Index, Channel-State, Users, Settings, Branding,
   Module-Catalog, Tickets, Knowledge, Business-Record-Projections, IoT,
   Browser-Runtime
9. `bring_up_succeeded` → Running (`:2707–2714`)
10. Watchdog-Schleife

Tear-down bei Exit: `begin_stopping` + `peer.shutdown()` + Runtime
`shutdown_timeout(10s)` (`:2864–2876`, `:1800–1803`).

### Dauer laut Log

- Logzeilen **ohne** Wanduhr-Timestamps.
- Indirekte Messung über eingefrorene `updated_at_ms`-Altersangaben der letzten
  Stale-Kette (s. oben): Abstand Kill→Kill ≈ Backoff + ~5–15 s Overhead
  (Shutdown 10 s Cap + Bring-up + Watchdog-Raster 15 s).
- Nach `respawning in Ns` folgen oft **≤ 7 Logzeilen** bis
  `replication up for 195 collections` → der reine Multiplex-Join ist im
  Erfolgsfall **deutlich unter dem 20‑s-Timeout**.
- Historisch (ganzes Log): 56× Bring-up-Timeout (20 s) + 425× Bring-up-failed —
  im aktuellen 80k-Fenster **0** solcher Fehler; hier stirbt der Peer **nach**
  erfolgreichem Bring-up am Stale-HB.

### Für den Nutzer nicht verfügbar währenddessen

| Phase | Effekt |
|---|---|
| Shutdown bis neuer Pool | kein nativer WebRTC-Peer / Session-Id wechselt jedes Mal (`rxdb-rs-{uuid}`, `:2357`) |
| Heartbeat `running` | kann während Restart lückenhaft/stale sein (TTL 30 s für externe Leser, `:757`, `:3280–3287`) |
| `replicationUp` / `dataChannelOpen` | im Status getrennt vom Prozess-Leben (`:9–10`, `:3332–3341`); Browser-Sync braucht beides. Aktuell gemessen: `replicationUp=false`, `dataChannelOpen=false` trotz `running=true` |
| UI | `src/apps/business-os/app.js:8621` verlangt `serviceActive && replicationUp && status==='active'` |
| Background-Projektionen | alle Loops neu; große max_duration_ms in Status (z. B. business_records max ~1.3e6 ms, desktop_file_index ~1.2e6 ms) zeigen teure Erstticks nach Start |
| Disk | jeder Restart: Status ~16 KiB pretty-JSON alle 5 s, SQLite-WAL, Log-Spam (248 HB-Fails/80k), WebRTC-Reconnect-Last |

Bei 78 Bring-ups im 80k-Fenster und 18 Stale-Respawns allein durch
Status-Schreibfehler ist Sync für den Nutzer **wiederholt minutenweise**
unterbrochen (90 s bis Kill + Backoff 5…40 s + Bring-up), obwohl die
Replikationspipeline kurz zuvor „up“ war.

---

## Report-Sektionen

### ursache_belegt

1. **Haupttreiber im Messfenster:** volle Platte (`No space left on device`, 258×)
   → Status-Heartbeat kann `business-os-rxdb-peer.status.json` nicht atomar
   aktualisieren (`write_native_peer_heartbeat` `:3299–3360`, Fail-Log `:3414`)
   → `updated_at_ms` altert
   → Watchdog nach >90 s (`:2802–2816`) → `HeartbeatStale`
   → Supervisor respawnt mit wachsendem Backoff (`:1817–1822`, `:1884–1886`).

2. **Designlücke (belegt):** Watchdog prüft **nur Dateialter**, nicht
   Schreibbarkeit und nicht „Peer-Arbeit lebt“. Schreibfehler ≠ Thread-Tod,
   wird aber identisch bestraft.

3. **Backoff-Lücke (belegt):** `delay` resettet nicht nach Bring-up, nur nach
   ≥600 s Laufzeit (`:1866–1868` / `:500`). Stale-Kills bei ~100 s halten den
   Backoff am Wachsen (Log: bis 40 s).

4. **Nebentreiber:** Runtime-Schema-Changes (sofortiger Reconfigure, 11×
   Bring-ups danach im Fenster); Critical-Child (4×); Sync-Config (2×).
   Bring-up-Timeouts spielen im Fenster keine Rolle (0), historisch aber 56×.

5. **Pfadbau-Bug:** `:3347` `with_extension("status.json.tmp")` erzeugt
   `.status.status.json.tmp`. Finaler Lesepfad bleibt korrekt; Bug ist real,
   aber ENOSPC-Wirkkette existiert unabhängig vom Temp-Namen.

6. **Laufende Binary** ist Release 2026-07-24; Checkout hat Log-Wording
   geändert, nicht die hier gemessene Mechanik.

### folge_fuer_den_nutzer

- Native Sync-Session wird regelmäßig zerrissen (neue `peer_session_id`, Pool neu).
- Browser-Datenkanal/`replicationUp` bleiben unzuverlässig; UI-Pfade, die
  `replicationUp` verlangen, melden Sync nicht „active“.
- Alle nativen Background-Projektionen und File-Index laufen neu an (teure
  Erstticks, Sekunden bis Minuten laut Status-Metriken).
- Log- und IO-Last steigen durch HB-Fail-Spam + 78× Full-Bring-up gegen 195
  Collections und ~2.1 Gi RxDB-SQLite — bei bereits voller Platte selbstverstärkend.
- Nutzer-sichtbare Lücke pro Stale-Zyklus: ≥~90 s „totgesagter“ Peer + Backoff
  5–40 s + Bring-up, wiederholt.

### pfade

| Rolle | Pfad |
|---|---|
| Source (Messgrundlage Zeilen) | `/Users/michaelwelsch/Documents/ctox/src/core/business_os/rxdb_peer.rs` |
| Service-Log | `/Users/michaelwelsch/.local/lib/ctox/current/runtime/ctox_service.log` |
| Heartbeat final | `…/runtime/business-os-rxdb-peer.status.json` |
| Heartbeat temp (Bug) | `…/runtime/business-os-rxdb-peer.status.status.json.tmp` |
| Process lock | `…/runtime/business-os-rxdb-peer.lock` |
| RxDB SQLite | `…/runtime/business-os-rxdb.sqlite3` (~2.1 Gi) |
| Running release | `/Users/michaelwelsch/.local/lib/ctox/releases/branch-main-20260724T072316Z` |
| Dieser Report | `/tmp/s-01-report.md` |

---

## Kurzantwort auf die Leitfrage

Der native Sync-Peer startet ~78× neu, weil der **Supervisor** jeden
nicht-intentionalen Exit respawnt, und im Messfenster vor allem der
**Heartbeat-Stale-Watchdog** feuert: die **Statusdatei lässt sich bei voller
Platte nicht schreiben**, der Watchdog wertet das als „Peer tot“ und reißt
einen ansonsten bring-up-fähigen Peer nach ~90 s herunter — mit Backoff, der
sich nach kurzem Überleben **nicht** zurücksetzt. Schema-Reconfigs und
vereinzelte Critical-Child-Exits multiplizieren weitere Bring-ups. Die doppelte
Endung `.status.status.json.tmp` ist ein realer `with_extension`-Bug an
`rxdb_peer.rs:3347`, aber nicht der Lesepfad des Watchdogs.

Keine Datei geändert. Kein Commit. Kein cargo.
