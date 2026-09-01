# RFC: Workjet Session Transfer

- Status: **Entscheidungsvorlage, zur Umsetzung freigegeben**
- Stand: 2026-09-01
- Scope: CTOX Core, CTOX-Sync-Projektion, Workjet-MCP-Tools und lokale Harness-Adapter
- Ziel: Umzug einer laufenden Workjet-Session mit Chat, Worker-Autoritaet und Arbeitsbaum von Rechner A nach Rechner B
- Autoritaet: genau eine CTOX-Instanz je Session

## 1. Entscheidung in Kurzform
Diese RFC fuehrt zwei native, serverautoritative Collections ein: `workjet_sessions` als dauerhaftes Register der wandernden Session und `workjet_session_transfers` als unveraenderlich nachvollziehbares Transferjournal. `coding_agent_sessions` bleibt eine abgeleitete Workbench-/Turn-Projektion und wird nicht zur Location of Record erweitert.
Die exklusive Worker-Autoritaet wird mit einer monotonen `fence_epoch` gesichert. Uhrzeiten dienen nur fuer Fortschritts- und Abbruchfristen; sie entscheiden nie, welcher Rechner noch schreiben darf.
Das normative Mutationsprotokoll verwendet die sieben Lifecycle-Verben `start`, `pause_ack`, `pack_complete`, `apply_complete`, `confirm_working_copy`, `resume_ack` und `abort`. Ein Status-Tool liest die serverautoritative Projektion und ist kein achtes Zustandsverb.
Der normale Transport ist Git: Branch/HEAD plus Patch fuer versionierte lokale Aenderungen plus Bundle fuer nicht versionierte Dateien. Nur wenn die Quelle kein benutzbares Git-Repository ist, wird der komplette Baum ueber bestehende `desktop_files` und `desktop_file_chunks` uebertragen.
Der atomare Commit bindet die Session an die Ziel-Working-Copy, aktiviert diese, setzt die Quell-Working-Copy auf `detached` und vergibt die naechste Fence-Epoch. Vor diesem Commit wird bei Fehlern zur Quelle zurueckgerollt; danach gibt es nur einen Forward-Fix am Ziel und niemals ein automatisches Wiederbeleben der Quelle.

## 2. Kontext
Workjet-Projekte sind virtuell und werden durch CTOX Sync zwischen Apps sichtbar. Eine Working Copy ist dagegen eine physische Auspraegung auf genau einem Rechner; ihr `path` ist fuer CTOX opak und darf serverseitig nicht als Host-Pfad ausgewertet werden.
Desktop und Mobile sind Steuerpulte. Ein Worker laeuft nur auf dem Rechner, auf dem die fuer seine Session autoritative Working Copy liegt. Chat, Befehlsjournal, Audit und Sessionregister bleiben in CTOX und folgen deshalb ohne Dateikopie.
Ein Session-Transfer ist damit eine gekoppelte Operation aus:
1. Einfrieren der bisherigen Worker-Autoritaet.
2. Erfassen des Quell-Arbeitsbaums.
3. Uebertragen und Anwenden am Ziel.
4. Verifizieren des Zielbaums.
5. Atomarem Umschalten der Session und der Working-Copy-Status.
6. Starten und Quittieren des Ziel-Workers.

## 3. Nicht-Ziele
- Kein Merge zweier divergierter Arbeitsbaeume.
- Keine gleichzeitige aktive Working Copy fuer dieselbe Session.
- Keine direkte Dateikopie zwischen Steuerpulten ausserhalb CTOX Sync.
- Kein HTTP-Datenpfad fuer Session-, Transfer- oder Dateirecords.
- Keine Interpretation opaker Working-Copy-Pfade durch CTOX Core.
- Keine Foederation mehrerer autoritativer CTOX-Instanzen.
- Kein automatischer Wechsel auf den Copy-Pfad, nur weil ein Git-Transfer temporaer fehlschlaegt.
- Keine Neudefinition von Chat-Threads oder `coding_agent_events`.
- Keine automatische Konfliktaufloesung fuer eine bereits geaenderte Zielkopie.
- Kein stilles Weiterarbeiten auf der Quelle, waehrend das Ziel offline ist.

## 4. Verifizierte Bestandsaufnahme
Die folgenden Belege wurden direkt im Klon geprueft und nicht aus den drei Vorentwuerfen uebernommen.
| Bestand | Beleg | Folgerung |
|---|---|---|
| Projekt- und Working-Copy-Collections | `src/core/business_os/store_workjet_projects.rs:22-23` | Das vorhandene fachliche Modell bleibt Grundlage. |
| Opake Rechner-/Pfadkennungen | `src/core/business_os/store_workjet_projects.rs:4-9` | Core darf `path` weder normalisieren noch oeffnen. |
| Strikte Payloads | `src/core/business_os/store_workjet_projects.rs:25-62` | Neue Payloads verwenden ebenfalls `deny_unknown_fields`. |
| Owner kommt vom Aufrufer | `src/core/business_os/store_workjet_projects.rs:64-80` | `owner_user_id` darf nie Client-Payload sein. |
| Owner-Migration | `src/core/business_os/store_workjet_projects.rs:85-127` | Neue Collections muessen dieselbe kanonische Owner-ID verwenden. |
| Attach-/Detach-Semantik | `src/core/business_os/store_workjet_projects.rs:249-292` | Quelle kann nach Commit idempotent `detached` werden. |
| Projekt- und Computer-Konsistenz | `src/core/business_os/store_workjet_projects.rs:293-303` | Ziel-ID muss zum Projekt und Rechner passen. |
| Deterministische Working-Copy-ID | `src/core/business_os/store_workjet_projects.rs:349-356` | Ziel-ID ist vor dem Commit berechenbar. |
| Idempotente Persistenz | `src/core/business_os/store_workjet_projects.rs:358-400` | Transferhandler uebernehmen stabilen Inhaltsvergleich. |
| Bounded-Validierung | `src/core/business_os/store_workjet_projects.rs:413-451` | Alle Strings erhalten konkrete Obergrenzen. |
| Computer-Assignment-Store vorhanden | `src/core/business_os/store_workjet_computers.rs:21-68` | `workjet_computers` ist der fachliche Rechneranker. |
| Assignment lehnt Backend-Hosts ab | `src/core/business_os/store_workjet_computers.rs:109-154` | Nur `workstation` und `self_hosted` sind Worker-Ziele. |
| Assigned- und Owner-Pruefung | `src/core/business_os/store_workjet_computers.rs:218-234` | Ziel-Acks muessen an diesen Check gekoppelt werden. |
| Computer-Payload-Grenzen | `src/core/business_os/store_workjet_computers.rs:274-285` | Computer-IDs und Faehigkeiten bleiben bounded. |
| Computer-Store noch nicht verdrahtet | `src/core/business_os/mod.rs:37-54` | Datei existiert, ist aber im Modul nicht registriert. |
| Computer-Commands nur im Inventory | `src/core/business_os/business_command_inventory.json:327-332` | Dispatcher/Policy/Schema muessen konsistent nachgezogen werden. |
| Exakte Control-Type-Liste | `src/core/business_os/command_plane.rs:266-327` | Jedes neue Verb muss in den Inventory-Gate. |
| Heutige Workjet-Policy ist Workspace-scope | `src/core/business_os/command_plane.rs:798-808` | Normale Owner koennen dadurch ausgeschlossen werden. |
| Heutiges Workjet-Routing | `src/core/business_os/command_plane.rs:1047-1069` | Neue Handler folgen demselben autorisierten Dispatchpfad. |
| Replay gespeicherter Outcomes | `src/core/business_os/command_plane.rs:496-569` | `command_id` verhindert doppelte Side Effects. |
| Unsicherer Control-Claim wird nicht wiederholt | `src/core/business_os/command_plane.rs:636-660` | Fachliche Step-Idempotenz ist zusaetzlich erforderlich. |
| Coding-Turn-Handler | `src/core/business_os/store.rs:12506-12587` | Fence-Pruefung muss vor und nach dem laufenden Turn liegen. |
| Coding-Session-Schreiber | `src/core/business_os/store.rs:3820-3884` | Der Record ist pro App und wird pro Turn ueberschrieben. |
| Pi-Prozess ist pro Turn frisch | `src/core/coding_agents/pi_sidecar.rs:361-398` | Pause kann einen einzelnen Kindprozess gezielt beenden. |
| Pi-Turn hat 600-s-Limit | `src/core/coding_agents/pi_sidecar.rs:340-377` | Transfer braucht einen kuerzeren Abbruchpfad als das Turn-Limit. |
| Snapshot wird nach Turn angewandt | `src/core/coding_agents/pi_sidecar.rs:733-799` | Zweiter Fence-Check vor `apply_turn_snapshot` ist zwingend. |
| `coding_agent_sessions`-Schema | `src/core/business_os/business_os_schema_contract.json:3572-3628` | Es enthaelt nur Provider-/Workbench-Felder und ist offen. |
| Workjet-Schemata sind geschlossen | `src/core/business_os/business_os_schema_contract.json:11299-11432` | Neue fachliche Felder brauchen explizite Contracts. |
| `verified_at_ms` ist vorhanden | `src/core/business_os/business_os_schema_contract.json:11403-11405` | Verifikation braucht kein neues WC-Feld. |
| Direkte Peer-Writes werden blockiert | `src/core/business_os/threads.rs:200-228` | Session und Transfer werden ausschliesslich per Command mutiert. |
| Native Projektionsliste | `src/core/business_os/threads.rs:231-268` | Drei weitere Workjet-Collections muessen aufgenommen werden. |
| Record-Scope besitzt Owner-Bit | `src/core/business_os/policy.rs:176-218` | Die Policy kann Ownership explizit tragen. |
| `DataWrite` ignoriert Owner-Bit | `src/core/business_os/policy.rs:340-379` | Genau dieser Policy-Zweig ist zu erweitern. |
| Decision-Audit ist vorhanden | `src/core/business_os/policy.rs:232-283` | Transferentscheidungen nutzen bestehendes Audit-Shape. |
| Datei-/Chunk-Schemata | `src/core/business_os/business_os_schema_contract.json:5597-5751` | Eigene Transfer-Chunk-Collection ist nicht erforderlich. |
| Frame-Budget und Resume | `docs/ctox-rxdb.md:482-527` | Grosse Artefakte duerfen nicht in Command-Payloads. |
| Demand-Chunk-Regel | `docs/ctox-rxdb.md:579-602` | Ziel zieht nur benoetigte Generationen. |
| Sticky Materialisierung | `docs/ctox-rxdb.md:720-723` | Transferartefakte brauchen explizite Retention/GC. |
| Schema-Contract wird nativ geladen | `src/core/business_os/rxdb_peer.rs:8844-8864` | Jede Collection muss im Contract registriert sein. |
| Hash-Registry-Test | `src/core/business_os/rxdb_peer.rs:12157-12185` | Contract und Browser-Hash muessen atomar aktualisiert werden. |
| Browser-Hash-Regel | `src/core/business_os/AGENTS.md:26-28` | Einseitige Schema-Aenderungen sind verboten. |
| Desktop-Projektsteuerung | `src/apps/business-os/app.js:12591-12722` | Session-Control folgt dem Command-plus-Projection-Muster. |
| Mobile-Projektbridge | `src/apps/business-os/mobile-host.js:154-166` | Eine Session-Control-Bridge kann gleichartig ergaenzt werden. |
| MCP-Deskriptoren und Dispatch | `src/core/business_os/mcp_channel.rs:1098-1140`, `src/core/business_os/mcp_channel.rs:2860-2993` | Harnesses erhalten typisierte Tools, keinen Sondertransport. |

### 4.1 Bestehende Luecken
1. Es gibt kein serverautoritatives Register, das eine logische Workjet-Session an genau eine Working Copy bindet.
2. `coding_agent_sessions` ist kein solches Register: Der aktuelle Schreiber erzeugt `pi:<module_id>`, schreibt `workspace_root=module_id` und aktualisiert den Record best-effort nach jedem Turn.
3. Es gibt keinen Transferjournal-Record und keinen Fence gegen veraltete Turns.
4. Der Computer-Store existiert als Datei, ist aber in Modul, Dispatcher, Policy, Schema-Contract und nativer Projektionsliste noch nicht komplett integriert.
5. `ctox.coding.turn` kennt weder `workjet_session_id` noch `fence_epoch`.
6. Der Pi-Sidecar wird zwar sicher beendet, aber es gibt noch keinen von einem Session-Transfer ausloesbaren Cancellation-Handle.
7. `verified_at_ms` existiert, wird im vorhandenen Working-Copy-Upsert aber nicht als Transferbeweis gesetzt.

## 5. Entscheidung 1: Eigenes Session-Register

### 5.1 Beschluss
Es wird eine eigene Collection `workjet_sessions` eingefuehrt. `coding_agent_sessions` erhaelt keine autoritativen Transferfelder.

### 5.2 Begruendung
Der aktuelle Schreiber von `coding_agent_sessions` ist `record_coding_agent_session_turn`; er schreibt pro App eine Pi-Workbench-Session und ein Event pro Turn (`src/core/business_os/store.rs:3820-3884`). Der Write geschieht nach dem Datei-Apply best-effort; ein Fehler darf die bereits gelandete Aenderung nicht zuruecknehmen (`src/core/coding_agents/pi_sidecar.rs:782-793`). Das ist absichtlich eine UI-/Historien-Projektion, keine transaktionale Location of Record.
Ausserdem ist das Schema `additionalProperties: true`, hat keine Owner-, Projekt-, Computer- oder Working-Copy-Bindung und indexiert `workspace_root` (`src/core/business_os/business_os_schema_contract.json:3572-3628`). Transferautoritaet dort unterzubringen wuerde einen abgeleiteten, von jedem Turn ueberschriebenen Record mit einer sicherheitskritischen, atomaren Sessionbindung vermischen.
`workjet_sessions` ist dagegen geschlossen, owner-scoped, nativ persistiert und nur durch Session-/Transferhandler veraenderbar. `coding_agent_sessions` darf optional `metadata.workjet_session_id` spiegeln; diese Spiegelung ist niemals Source of Truth und darf einen Transfer nicht blockieren.

## 6. Entscheidung 2: Kausale Fence-Epoch statt Uhrzeit-Lease

### 6.1 Beschluss
Jede Workjet-Session besitzt eine monotone `fence_epoch` vom Typ nichtnegativer 64-Bit-Integer. Jeder Worker-Start, jeder `ctox.coding.turn`, jeder Agent-Ack und jeder Apply-Schritt traegt die aktuell gelesene Epoch.
Ein Core-Commit, der die Session einfriert oder auf einen anderen Rechner umschaltet, erhoeht die Epoch. Jede Operation mit kleinerer oder abweichender Epoch wird mit `session_fenced` abgewiesen, unabhaengig von lokaler Uhrzeit, Netzverzoegerung oder einem spaeten Retry.

### 6.2 Rolle von Uhrzeiten
Uhrzeiten sind nur Liveness-Metadaten:
- `deadline_at_ms` bestimmt, wann Core einen nicht fortschreitenden Transfer kompensiert. - `last_seen_at_ms` bestimmt, ob ein Ziel vor `start` als erreichbar gilt. - Kein `expires_at_ms` verleiht Schreibrecht. - Eine spaete Nachricht wird auch bei noch nicht abgelaufener Frist verworfen, wenn ihre Epoch alt ist. - Eine Nachricht mit aktueller Epoch wird nicht allein wegen Clock-Skew verworfen; fuer den Zustand gilt die serverseitige Empfangszeit.

### 6.3 Pause-Ack-Handshake
Der Pause-Handshake ist fuer alle Harnesses gleich, auch wenn die lokale Prozesssteuerung unterschiedlich ist.
1. `start` validiert Owner, Quelle, Ziel und Zustand in einer Core-Transaktion. 2. Core erhoeht `workjet_sessions.fence_epoch` von `n` auf `n+1`. 3. Core setzt Session auf `pausing` und Transfer auf `pause_requested`. 4. Ab diesem Commit lehnt der Admission-Check neue `ctox.coding.turn` fuer die Session ab, sofern sie nicht exakt Epoch `n+1` und den internen Pausepfad tragen. 5. Der Quelladapter erhaelt die Projektion `pause_requested`. 6. Der Adapter stoppt die Annahme neuer Prompts. 7. Falls
ein Turn laeuft, fordert der Adapter Cancellation an. 8. Fuer den Pi-Pfad wird der pro Turn gestartete Kindprozess beendet; `Drop` wartet auf den Prozess und entfernt den Socket (`src/core/coding_agents/pi_sidecar.rs:267-281`). 9. Der laufende `ctox.coding.turn` prueft vor `apply_turn_snapshot` erneut die Epoch. Ist sie nicht mehr die Start-Epoch, wird der Snapshot verworfen und der Command endet mit `session_fenced`. 10. Der Adapter leert/fsynct seine lokalen Schreibpuffer und liest den finalen Baumzustand. 11. Erst
dann sendet er `pause_ack` mit Epoch `n+1` und dem letzten terminalen Turn-Identifier. 12. Core akzeptiert den Ack nur vom verifizierten Quellrechner und wechselt zu `packing`.

### 6.4 Konkrete Pause-Fristen
- Soft deadline: 30 Sekunden nach `start`; Core markiert `pause_slow` im Audit. - Hard deadline: 45 Sekunden; Core fordert lokalen Hard-Cancel an. - Abschlussfrist: 60 Sekunden; ohne beweisbaren Prozessstopp geht der Transfer nach `aborting` und anschliessend `rolled_back`. - Die Quelle wird erst wieder `running`, wenn kein Prozess mit der alten Epoch registriert ist. - Der bestehende Pi-Turn darf bis zu 600 Sekunden lesen; diese Frist wird beim Transfer nicht abgewartet
(`src/core/coding_agents/pi_sidecar.rs:340-377`).

## 7. Entscheidung 3: Normative Befehlsschnittstelle

### 7.1 Beschluss
Die RFC uebernimmt die feingranularen Lifecycle-Acks:
1. `ctox.workjet.session.transfer.start` 2. `ctox.workjet.session.transfer.pause_ack` 3. `ctox.workjet.session.transfer.pack_complete` 4. `ctox.workjet.session.transfer.apply_complete` 5. `ctox.workjet.session.transfer.confirm_working_copy` 6. `ctox.workjet.session.transfer.resume_ack` 7. `ctox.workjet.session.transfer.abort`
`workjet.session.transfer.status` ist ein read-only MCP-/UI-Tool, das `workjet_session_transfers` und `workjet_sessions` liest. Es ist kein zustandsaendernder Business-Command und deshalb kein Lifecycle-Verb.

### 7.2 Begruendung
Generische Verben wie `request`, `progress`, `confirm` und `resume` verdecken, welcher Rechner welchen Beweis liefert. Die gewaehlten Acks bilden dagegen die verteilten Side Effects direkt ab und erlauben pro Uebergang klare Actor- Verifikation, Payload-Grenzen, Idempotenz und Fehlercodes.
Ein eigenes `progress`-Command wird bewusst nicht eingefuehrt. Byte-Fortschritt ist lokale Telemetrie des bestehenden Demand-File-Streams; autoritativ sind nur `pack_complete`, `apply_complete` und die unveraenderlichen Artefakt-Hashes.

## 8. Entscheidung 4: Record-scoped Policy und Rechner-Actor

### 8.1 Beschluss
Workjet-Session-Mutationen verwenden `DataWrite` auf einem Record-Scope mit `owned_by_actor=true`, wenn der serverseitig geladene Sessionrecord dem authentifizierten User gehoert.
Der `DataWrite`-Arm in `policy::evaluate` wird wie folgt spezialisiert:
- Chef/Admin bleiben erlaubt. - `assigned_to_actor=true` bleibt erlaubt. - `owned_by_actor=true` erlaubt `DataWrite` nur fuer `Record`-Scopes. - `owned_by_actor` auf Workspace-, Collection- oder Module-Scope erweitert keine Rechte.
Damit wird der normale Owner nicht mehr durch den heutigen Workspace-Scope blockiert (`src/core/business_os/policy.rs:370-373`), ohne Workspace-weite Schreibrechte zu erhalten.

### 8.2 Serverberechnung von `owned_by_actor`
Der Client darf `owned_by_actor` nie senden. Der Policy-Resolver:
1. authentifiziert den User aus `BusinessOsSession`; 2. laedt `workjet_sessions/<session_id>` aus Core SQLite; 3. vergleicht `owner_user_id` mit der kanonischen Session-User-ID; 4. konstruiert `BusinessOsScope { scope_type: Record, scope_id, owned_by_actor }`; 5. protokolliert die resultierende `PolicyDecision`.
Fuer `start` wird zusaetzlich das Projekt geladen und derselbe Ownervergleich fuer Projekt, Quell-Working-Copy und Zielcomputer verlangt.

### 8.3 Verifikation von Source- und Target-Agent
Agent-Acks benoetigen zwei unabhaengige Beweise:
1. Der User-Actor der Verbindung ist Owner der Session.
2. Die device-bound Peer-Identitaet ist an genau einen assigned `workjet_computers`-Record desselben Owners gekoppelt.
`workjet_computers` wird dafuer um folgende native Felder erweitert:
- `device_binding_id`: opake ID der aktiven Device-to-Instance-Bindung. - `actor_epoch`: vom nativen Capability-System gelesene Actor-Epoch. - `last_seen_at_ms`: serverseitig gestempelter letzter erfolgreicher Heartbeat. - `replication_up`: letzter serverseitig beobachteter Sync-Zustand.
Der Payload darf `computer_id` als Assertion enthalten; autoritativ ist die aus der Verbindung abgeleitete Computer-ID. Stimmen beide nicht ueberein, gilt `computer_actor_mismatch`.
`require_assigned_workjet_computer` bleibt das fachliche Gate fuer Status, Owner und Hosting-Mode (`src/core/business_os/store_workjet_computers.rs:218-234`). Zusaetzlich muss die aktuelle Binding-ID mit dem Record uebereinstimmen. Ein Mobile-Steuerpult ohne die Ziel-Working-Copy darf `start` oder `abort` als Owner anfordern, aber keinen Source-/Target-Ack vortaeuschen.

## 9. Entscheidung 5: Bestehende Files/Chunks wiederverwenden

### 9.1 Beschluss
Es wird keine Collection `transfer_chunks` eingefuehrt. Transferartefakte verwenden `desktop_files` und `desktop_file_chunks`.

### 9.2 Begruendung
Der vorhandene Pfad besitzt bereits:
- deterministische Generationen und Chunk-Indizes; - Frame-Budgets unter der 16-KiB-SCTP-Grenze; - `start/chunk/ack/resume` fuer grosse Frames; - `rxdb.file.fetch` mit begrenzten parallelen Streams; - demand-only Abruf statt Hintergrund-Vollreplikation; - Hashfelder auf Datei- und Chunk-Ebene; - sticky Materialisierung fuer verfuegbare Generationen.
Eine zweite Chunk-Collection wuerde denselben Transport, dieselben Limits, dieselben Hashpruefungen, Browserprofile und GC-Regeln duplizieren. Die Trennung erfolgt fachlich ueber Metadaten, nicht physisch ueber eine neue Chunk-Tabelle.

### 9.3 Transferartefakt-Konvention
Jedes Transferartefakt ist ein `desktop_files`-Record mit:
- `source = "workjet-session-transfer"` - `linked_collection = "workjet_session_transfers"` - `linked_record_id = <transfer_id>` - `virtual_path = "workjet-transfer://<transfer_id>/<artifact-name>"` - `content_state = "available"` - `content_generation_id = <generation_id>` - `content_hash_scheme = "sha256"` - keinem serverseitig interpretierbaren `local_path`
Chunks verwenden die vorhandene ID-Konvention aus `docs/ctox-rxdb.md:579-602`. Der Transferrecord verweist nur auf File-IDs, Generationen und Hashes; keine Datei-Bytes gelangen in Command-Payloads oder Audit-Events.

### 9.4 Retention
- Erfolgreiche Artefakte: 7 Tage ab `completed_at_ms`. - Abgebrochene Artefakte: 24 Stunden ab `rolled_back_at_ms`. - Artefakte fuer `manual_intervention`: bis zur manuellen Aufloesung, maximal 30 Tage ohne explizite Verlaengerung. - GC markiert zuerst `desktop_files.is_deleted`; Chunk-Generationen werden erst entfernt, wenn kein nicht geloeschter File-Record mehr darauf verweist.

## 10. Datenmodell-Delta

### 10.1 Collection-Uebersicht
| Collection | Delta | Source of Truth | Schreiber | Projektion |
|---|---|---|---|---|
| `workjet_sessions` | neu | Core SQLite | Session-/Transferhandler | native read-only |
| `workjet_session_transfers` | neu | Core SQLite | Transferhandler | native read-only |
| `workjet_computers` | integrieren und Actor-Felder | Core SQLite | Computer-/Heartbeat-Handler | native read-only |
| `workjet_working_copies` | Schreibregel fuer `verified_at_ms` | Core SQLite | Transfer-Commit | native read-only |
| `desktop_files` | Nutzungskonvention | RxDB/File-Store | verifizierter Source-Agent/Core | bestehend |
| `desktop_file_chunks` | Nutzungskonvention | RxDB/File-Store | verifizierter Source-Agent | demand-chunks |
| `coding_agent_sessions` | optionale Spiegel-ID | bestehende Projektion | Coding-Turn-Logger | nicht autoritativ |

### 10.2 `workjet_sessions`
| Feld | Typ/Grenze | Bedeutung | Source of Truth | Schreiber |
|---|---|---|---|---|
| `id` | String, 160 | `workjet_session_<opaque>` | Core | create/import |
| `project_id` | String, 128 | virtuelles Projekt | Core | create |
| `thread_id` | String, 160, optional | Chat-Thread | Core | create/link |
| `coding_session_id` | String, 128, optional | Workbench-Spiegel | Core | coding adapter |
| `working_copy_id` | String, 160 | einzige autoritative WC | Core | create/transfer commit |
| `computer_id` | String, 256 | aus WC denormalisiert | Core | create/transfer commit |
| `run_status` | Enum | siehe unten | Core | Session-/Transferhandler |
| `fence_epoch` | Integer >= 0 | kausaler Schreibzaun | Core | pause/switch |
| `active_transfer_id` | String, 160, optional | offener Transfer | Core | start/terminal |
| `last_terminal_turn_id` | String, 160, optional | Pause-Beweis | Core | turn/pause ack |
| `owner_user_id` | String, 256 | kanonischer Owner | Core | aus Auth-Session |
| `created_at_ms` | Integer | Serverzeit | Core | create |
| `updated_at_ms` | Integer | Serverzeit | Core | jeder Write |
| `is_deleted` | Boolean | Tombstone | Core | delete |
`run_status` ist geschlossen auf: `running`, `pausing`, `paused`, `transferring`, `resuming`, `transfer_failed`.
Invarianten:
- `working_copy_id` ist fuer jede nicht geloeschte Session gesetzt. - `computer_id` entspricht dem `computer_id` der gebundenen Working Copy. - `active_transfer_id` ist genau dann gesetzt, wenn der Transfer nicht terminal ist. - Eine Session mit `run_status != running` akzeptiert keinen normalen Turn. - Zwei nicht geloeschte Sessions duerfen nicht gleichzeitig `running` oder `resuming` auf derselben Working Copy sein.

### 10.3 `workjet_session_transfers`
| Feld | Typ/Grenze | Bedeutung | Source of Truth | Schreiber |
|---|---|---|---|---|
| `id` | String, 160 | fachlicher Idempotenzanker | Core | start |
| `session_id` | String, 160 | wandernde Session | Core | start |
| `project_id` | String, 128 | Snapshot | Core | start |
| `source_working_copy_id` | String, 160 | Quell-WC | Core | start |
| `source_computer_id` | String, 256 | Quellrechner | Core | start |
| `target_computer_id` | String, 256 | Zielrechner | Core | start |
| `target_path` | String, 4096 | opake Zielkennung | Core | start |
| `target_working_copy_id` | String, 160, optional | deterministische Ziel-WC | Core | confirm |
| `state` | Enum | Zustandsautomat | Core | Lifecycle-Verben |
| `fence_epoch` | Integer >= 1 | Transfer-Epoch | Core | start |
| `mode` | `git` oder `copy`, optional bis pack | Transportentscheidung | Core | pack_complete |
| `manifest_file_id` | String, 160, optional | Manifest | Core | pack_complete |
| `artifact_file_ids` | Array max. 64 | Artefakt-IDs | Core | pack_complete |
| `artifact_generation_id` | String, 160, optional | Chunk-Generation | Core | pack_complete |
| `manifest_sha256` | Hex, 64, optional | Gesamtbeweis | Core | pack_complete |
| `git` | Objekt, optional | Git-Beweisfelder | Core | pack_complete |
| `tree_sha256` | Hex, 64, optional | Copy-Beweis | Core | pack/apply |
| `error_code` | String, 128, optional | stabiler Fehler | Core | Handler |
| `error_detail` | String, 512, optional | redigierte Diagnose | Core | Handler |
| `deadline_at_ms` | Integer | aktuelle Liveness-Frist | Core | State-Wechsel |
| `created_at_ms` | Integer | Serverzeit | Core | start |
| `updated_at_ms` | Integer | Serverzeit | Core | jeder Wechsel |
| `completed_at_ms` | Integer, optional | Terminalzeit | Core | resume_ack |
| `rolled_back_at_ms` | Integer, optional | Kompensationszeit | Core | abort/recovery |
| `owner_user_id` | String, 256 | kanonischer Owner | Core | aus Session |
| `is_deleted` | Boolean | Tombstone | Core | Retention |
Das `git`-Objekt ist geschlossen auf:
- `head`: 40 oder 64 Hexzeichen, abhaengig vom Repositoryformat. - `branch`: String, maximal 256 Zeichen. - `base_commit`: 40 oder 64 Hexzeichen. - `bundle_file_id`: String, maximal 160 Zeichen, optional. - `patch_file_id`: String, maximal 160 Zeichen. - `patch_sha256`: 64 Hexzeichen. - `untracked_file_id`: String, maximal 160 Zeichen. - `untracked_sha256`: 64 Hexzeichen. - `dirty`: Boolean.
Remote-URLs sind kein Pflichtfeld des Core-Records. Der Zieladapter darf eine lokal bereits erlaubte Remote verwenden; ansonsten enthaelt das Git-Bundle die benoetigten Commit-Objekte.

### 10.4 `workjet_computers`
| Feld | Typ/Grenze | Bedeutung | Source of Truth | Schreiber |
|---|---|---|---|---|
| bestehende Assignment-Felder | unveraendert | Rechnerauswahl | Core | assign/unassign |
| `device_binding_id` | String, 160 | gebundene Peer-Identitaet | Core | bind/assign |
| `actor_epoch` | Integer >= 0 | Capability-Widerruf | Core | auth sync |
| `last_seen_at_ms` | Integer | Heartbeat-Empfang | Core | native peer |
| `replication_up` | Boolean | Sync erreichbar | Core | native peer |
`workjet_computers` wird nicht nur als Datei behalten, sondern vollstaendig in `mod.rs`, `EXACT_CONTROL_TYPES`, Dispatcher, Policy, Schema-Contract, Browser-Collections und `NATIVE_PROJECTION_COLLECTIONS` integriert.

### 10.5 Bestehende Collections
`workjet_working_copies` bekommt kein neues Feld. `verified_at_ms` darf nur der atomare Transfer-Commit fuer die Ziel-WC setzen. Ein normaler `working_copy.upsert` darf diesen Beweis nicht aus Clientdaten uebernehmen.
`coding_agent_sessions` bleibt unveraendert. Optional darf der bestehende Logger `metadata.workjet_session_id` und `metadata.fence_epoch` setzen, damit die Workbench Ereignisse korrelieren kann; ein Fehlen oder verzoegerter Write hat keine Auswirkung auf Autoritaet oder Transferzustand.

### 10.6 Native Projektion und Schema-Contract
`NATIVE_PROJECTION_COLLECTIONS` in `src/core/business_os/threads.rs:237-249` wird um folgende Eintraege erweitert:

```text
workjet_computers
workjet_sessions
workjet_session_transfers
```
Alle drei Collections sind fuer direkte Peer-Writes gesperrt. Browser, Mobile und Harnesses mutieren sie ausschliesslich ueber exakte Business-Commands.
Die Umsetzung muss gemeinsam aktualisieren:
1. `src/apps/business-os/modules/ctox/schema.js` 2. `src/apps/business-os/modules/ctox/collections.schema.json` 3. `src/apps/business-os/modules/ctox/module.json` 4. `src/apps/business-os/modules/registry.json` 5. `src/core/business_os/business_os_schema_contract.json` 6. `src/core/business_os/business_os_schema_hashes.json` 7. `src/apps/business-os/rxdb/src/schema.mjs` 8. die gebaute Browser-RxDB-Datei gemaess Repository-Regel
Der Contract wird nativ direkt eingebettet (`src/core/business_os/rxdb_peer.rs:8844-8849`). Der Hash-Registry-Test muss fuer jede neue Collection denselben Hash auf Rust- und Browserseite sehen (`src/core/business_os/rxdb_peer.rs:12157-12185`).

## 11. Globale Invarianten
1. Eine Session hat genau eine autoritative Working Copy.
2. Vor dem Commit bleibt die Quell-WC `active`; die Ziel-WC ist noch nicht autoritativ registriert.
3. Der Commit aktiviert die Ziel-WC und detacht die Quell-WC in einer Core-Transaktion.
4. Nach dem Commit darf kein Quell-Worker mit einer alten Epoch schreiben.
5. Ein Transfer hat hoechstens einen nichtterminalen Record je Session.
6. Ein Ziel-Ack kommt nur vom gebundenen Zielcomputer.
7. Ein Quell-Ack kommt nur vom gebundenen Quellcomputer.
8. Datei-Bytes stehen nie in Transfer-Commands oder Audit-Events.
9. Git wird verwendet, wenn die Quelle ein benutzbares Git-Repository meldet.
10. Copy wird nur verwendet, wenn kein benutzbares Git-Repository existiert.
11. Eine vorhandene geaenderte Zielkopie wird nie automatisch gemerged oder ueberschrieben.
12. Direktes RxDB-`masterWrite` kann Session- oder Transferautoritaet nie veraendern.
13. Ein Command-Replay erhoeht die Fence-Epoch nicht erneut.
14. Ein Step-Replay erzeugt keine zweite Artefaktgeneration.
15. Nach `switching` gibt es keinen automatischen Source-Rollback.

## 12. Zustandsautomat

### 12.1 Zustaende und Uebergaenge
| Zustand | Eintritt/Aktion | Naechster Beweis | Timeout | Abbruch/Rollback | Invariante |
|---|---|---|---|---|---|
| `pause_requested` | `start`; Epoch erhoeht, Turns gesperrt | `pause_ack` Quelle | 60 s | Prozess stoppen, Quelle wieder `running` | Quelle bleibt einzige WC |
| `packing` | Quelle ist quieszent | `pack_complete` Quelle | Git 5 min, Copy 30 min | Artefakte GC, Quelle `running` | kein alter Prozess laeuft |
| `packed` | Manifest und Artefakte fest | interner Versandstart | 30 s | Quelle `running` | Artefakthashes unveraenderlich |
| `shipping` | Ziel zieht Generationen | Ziel beginnt Apply | 10 min offline | Partialdaten loeschen, Quelle `running` | Ziel noch nicht autoritativ |
| `applying` | Ziel baut Baum | `apply_complete` Ziel | Git 15 min, Copy 60 min | Ziel quarantaenisiert, Quelle `running` | Apply nur im Partialpfad |
| `applied` | Hashes stimmen lokal | `confirm_working_copy` Ziel | 60 s | Ziel verwerfen, Quelle `running` | Quelle bleibt bis Commit aktiv |
| `switching` | atomarer Core-Commit | interner Commitabschluss | 10 s | nur Forward-Fix | WC-Bindung wechselt atomar |
| `resuming` | Ziel ist Location of Record | `resume_ack` Ziel | 45 s | Ziel bleibt autoritativ und `paused` | Quelle bleibt `detached` |
| `completed` | Ziel-Worker bestaetigt | keiner | terminal | keiner | genau ein Worker akzeptiert Turns |
| `aborting` | Abort/Timeout vor Commit | Kompensation | 120 s | bei Fehler `manual_intervention` | keine neue Zielautoritaet |
| `rolled_back` | Quelle wieder autoritativ | keiner | terminal | keiner | Ziel bleibt nicht autoritativ |
| `failed` | nicht kompensierbarer Fehler | Operator-Fix | terminal | kein Auto-Rollback | letzte Commitseite bleibt autoritativ |

### 12.2 `start` -> `pause_requested`
Core prueft atomar:
- Session existiert und gehoert dem Actor. - Session ist `running` oder bereits explizit `paused`. - Kein anderer Transfer ist offen. - Quell-WC ist die gebundene WC und `active`. - Zielcomputer ist assigned, gleicher Owner, nicht die Quelle. - Zielheartbeat ist hoechstens 30 Sekunden alt und `replication_up=true`. - Zielpfad ist bounded und opak. - Am Ziel ist kein bekannter dirty Tree fuer denselben Pfad gemeldet.
Dann erzeugt Core den Transferrecord, erhoeht die Epoch genau einmal, setzt `active_transfer_id` und publiziert beide Projektionen.

### 12.3 `pause_ack` -> `packing`
Der Ack beweist nicht nur, dass ein UI-Button gedrueckt wurde. Er beweist:
- kein Quellprozess der alten Epoch laeuft; - kein `ctox.coding.turn` der alten Epoch kann noch einen Snapshot anwenden; - der letzte terminale Turn ist benannt; - der Arbeitsbaum wurde nach dem Stop erneut gelesen; - die gemeldete Epoch entspricht dem Transfer.
Ein mehrfacher identischer Ack ist ein No-op mit identischem Outcome.

### 12.4 `pack_complete` -> `packed`
Der Quelladapter entscheidet lokal und deterministisch:
- Git, wenn `.git` lesbar ist und `HEAD` aufgeloest werden kann.
- Copy nur sonst.
Core akzeptiert `mode=copy` nicht, wenn die beim Pause-Ack festgehaltene Source-Capability `git_repository=true` war. Ein Git-Fehler fuehrt zu `git_pack_failed`, nicht zu stillem Copy-Fallback.
Nach erfolgreicher Payload- und Artefaktpruefung setzt Core `packed` und direkt anschliessend `shipping`.

### 12.5 `apply_complete` -> `applied`
Der Zieladapter meldet die lokal beobachteten Hashes. Core vergleicht sie mit dem unveraenderlichen Manifest:
- Git: `observed_head`, Patch-Hash und Untracked-Hash.
- Copy: `observed_tree_sha256`.
Ein Mismatch bleibt vor dem Commit und ist sicher wiederholbar. Nach drei identischen Mismatches wird der Transfer `failed` mit `apply_hash_mismatch`; die Quelle bleibt autoritativ.

### 12.6 `confirm_working_copy` und atomarer Commit
`confirm_working_copy` ist der einzige Commit-Punkt. Core berechnet die erwartete Working-Copy-ID aus Projekt, Zielcomputer und Zielpfad und vergleicht sie mit der Payload.
In einer Core-Transaktion werden dann:
1. die Ziel-WC mit `status=active` und `verified_at_ms=now` upserted; 2. `workjet_sessions.working_copy_id` auf die Ziel-ID gesetzt; 3. `workjet_sessions.computer_id` auf den Zielcomputer gesetzt; 4. `workjet_sessions.run_status` auf `resuming` gesetzt; 5. die Session-Epoch erneut erhoeht; 6. die Quell-WC auf `detached` gesetzt; 7. der Transfer auf `resuming` gesetzt.
Die RxDB-Projektionen folgen erst nach erfolgreichem Store-Commit. Schlaegt eine Projektion fehl, repariert der native Projektionspfad aus Core SQLite; er aendert nicht die fachliche Commit-Entscheidung.

### 12.7 `resume_ack` -> `completed`
Der Zieladapter startet den Worker nur, wenn:
- die Sessionprojektion den eigenen `computer_id` nennt; - die gebundene WC lokal vorhanden ist; - die Epoch exakt der neuen Session-Epoch entspricht; - `run_status=resuming` ist; - die Working Copy `active` und verifiziert ist.
`resume_ack` setzt Session `running`, leert `active_transfer_id` und beendet den Transfer. Ein ausbleibender Ack laesst die Session am Ziel `paused` beziehungsweise `transfer_failed`; die Quelle bleibt detached.

### 12.8 Abort und Recovery
`abort` ist bis einschliesslich `applied` als Source-Rollback erlaubt. In `switching`, `resuming` oder `completed` bedeutet `abort` keinen Source-Rollback, sondern setzt bei Bedarf `failed` und verlangt einen Forward-Fix am Ziel.
Ein Core-Recovery-Pass verarbeitet abgelaufene `deadline_at_ms`:
- vor `switching`: Wechsel nach `aborting`, Partialdaten markieren, Quelle nach Prozess-/Epoch-Pruefung wieder `running`, dann `rolled_back`; - ab `switching`: Ziel bleibt Location of Record, Session wird `transfer_failed`, Worker darf per erneutem Resume-Versuch gestartet werden; - fehlgeschlagene Kompensation nach 120 Sekunden: `manual_intervention` im Audit und Transfer `failed`.

## 13. Git-Transport

### 13.1 Pack auf der Quelle
Der Source-Agent erfasst nach dem Pause-Ack:
1. aktuellen Branchnamen; 2. aktuellen `HEAD`; 3. alle benoetigten Commit-Objekte als Git-Bundle, falls sie nicht garantiert ueber eine lokal erlaubte Remote erreichbar sind; 4. Patch gegen `HEAD` fuer versionierte, nicht committete Aenderungen; 5. separates binary-sicheres Bundle fuer nicht versionierte Dateien; 6. ein Manifest ueber Dateinamen, Modus, Groesse und SHA-256; 7. einen Gesamt-Hash ueber die kanonisch sortierte Manifestdarstellung.
Der Patch und das Untracked-Bundle sind Pflichtartefakte, auch wenn sie leer sind; der jeweilige Hash beweist dann explizit den leeren Inhalt.

### 13.2 Apply am Ziel
Der Target-Agent:
1. erstellt einen frischen temporaeren Zielordner; 2. stellt den Commit aus erlaubter Remote oder Git-Bundle her; 3. checkt den gemeldeten Branch/HEAD aus; 4. wendet den Patch an; 5. entpackt nicht versionierte Dateien; 6. berechnet alle Manifest-Hashes erneut; 7. verschiebt den Baum erst nach erfolgreicher Pruefung an `target_path`; 8. sendet `apply_complete`.
Ein bereits vorhandener nichtleerer Zielpfad wird standardmaessig abgewiesen. Es gibt in V1 kein `overwrite=true` und keinen automatischen Merge.

## 14. Copy-Transport

### 14.1 Auswahl
Copy ist nur zulaessig, wenn der Source-Agent nach dem Pause-Ack meldet, dass kein benutzbares Git-Repository existiert. Ein defektes oder temporaer nicht erreichbares Git-Remote macht ein vorhandenes Repository nicht zu "ohne Git"; das Git-Bundle ist der netzunabhaengige Pfad.

### 14.2 Pack
Der Source-Agent fuehrt einen deterministischen Tree-Walk aus:
- relative Pfade in UTF-8-normalisierter Manifestdarstellung; - Dateityp, Modus, Groesse und SHA-256; - keine absoluten Host-Pfade; - Symlinks als Linkziel, nicht durch rekursives Folgen; - Hardlinks duerfen als getrennte Dateien materialisiert werden; - Special Files, Sockets und Device Nodes fuehren zu `unsupported_file_type`; - maximal 1.000.000 Manifesteintraege; - maximal 64 Artefakt-File-Records pro Transfer; - Bytes ausschliesslich in `desktop_file_chunks`.

### 14.3 Apply
Das Ziel materialisiert in einen Transfer-spezifischen Partialordner, prueft jeden Chunk, jede Datei und abschliessend `tree_sha256`. Erst danach wird der Ordner atomar an die opake Zielkennung uebergeben und `apply_complete` gesendet.

## 15. Payload-Regeln und Idempotenz

### 15.1 Gemeinsame Regeln
Alle Payload-Strukturen verwenden `deny_unknown_fields`. Alle Strings werden getrimmt, auf Control-Zeichen geprueft und mit dem Muster `validate_bounded` validiert (`src/core/business_os/store_workjet_projects.rs:433-451`).
Gemeinsame Grenzen:
| Feldklasse | Grenze |
|---|---|
| Command-/Idempotenz-ID | 160 Zeichen |
| Session-/Transfer-/File-ID | 160 Zeichen |
| Projekt-ID | 128 Zeichen |
| Computer-ID | 256 Zeichen |
| Zielpfad | 4096 Zeichen |
| Branchname | 256 Zeichen |
| Reason/Error-Detail | 512 Zeichen |
| SHA-256 | exakt 64 lowercase Hexzeichen |
| Artefakt-IDs | maximal 64 Eintraege |
Jeder Mutationscommand besitzt:
- aeusseres `command_id` fuer Command-Plane-Replay; - fachliches `idempotency_key` im Payload, maximal 160 Zeichen; - den Transferrecord als dauerhaften Zustandsanker.
Regeln:
1. Gleicher `command_id` liefert das gespeicherte Outcome. 2. Gleicher `(transfer_id, action, idempotency_key)` und gleiche normalisierte Payload liefert dasselbe Outcome. 3. Gleicher Schluessel mit anderer Payload liefert `idempotency_conflict`. 4. Ein bereits erreichter Zielzustand macht die identische Wiederholung zum erfolgreichen No-op. 5. Ein Lifecycle-Step darf die Epoch hoechstens einmal erhoehen. 6. `pack_complete` darf fuer denselben Step keine zweite Generation referenzieren.

## 16. Commands und Payload-Beispiele

### 16.1 `start`

```json
{
  "session_id": "workjet_session_s1",
  "target_computer_id": "gpu3-a4500",
  "target_path": "opaque://gpu3/worktrees/project-x",
  "idempotency_key": "move-s1-to-gpu3-2026-09-01"
}
```
Source-WC und Source-Computer werden aus `workjet_sessions` geladen, nicht aus der Payload. `transfer_id` wird deterministisch als `workjet_xfer_<sha256(session_id, idempotency_key)>` gebildet.
Ergebnis enthaelt Transfer, Session, neue `fence_epoch` und `state`.

### 16.2 `pause_ack`

```json
{
  "transfer_id": "workjet_xfer_...",
  "computer_id": "mac-a",
  "fence_epoch": 42,
  "last_terminal_turn_id": "turn_019...",
  "git_repository": true,
  "idempotency_key": "pause-ack-42"
}
```
`computer_id` ist nur Assertion. Core vergleicht ihn mit der device-bound Verbindungsidentitaet und `source_computer_id`.

### 16.3 `pack_complete` im Git-Modus

```json
{
  "transfer_id": "workjet_xfer_...",
  "computer_id": "mac-a",
  "fence_epoch": 42,
  "mode": "git",
  "manifest_file_id": "desktop_file_xfer_manifest",
  "artifact_file_ids": [
    "desktop_file_xfer_bundle",
    "desktop_file_xfer_patch",
    "desktop_file_xfer_untracked"
  ],
  "artifact_generation_id": "xfer_gen_01",
  "manifest_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "git": {
    "head": "0123456789abcdef0123456789abcdef01234567",
    "branch": "workjet/session-s1",
    "base_commit": "0123456789abcdef0123456789abcdef01234567",
    "bundle_file_id": "desktop_file_xfer_bundle",
    "patch_file_id": "desktop_file_xfer_patch",
    "patch_sha256": "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "untracked_file_id": "desktop_file_xfer_untracked",
    "untracked_sha256": "2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "dirty": true
  },
  "idempotency_key": "pack-git-gen-01"
}
```

### 16.4 `pack_complete` im Copy-Modus

```json
{
  "transfer_id": "workjet_xfer_...",
  "computer_id": "mac-a",
  "fence_epoch": 42,
  "mode": "copy",
  "manifest_file_id": "desktop_file_tree_manifest",
  "artifact_file_ids": ["desktop_file_tree_archive"],
  "artifact_generation_id": "xfer_gen_02",
  "manifest_sha256": "3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "tree_sha256": "4123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "idempotency_key": "pack-copy-gen-02"
}
```

### 16.5 `apply_complete`

```json
{
  "transfer_id": "workjet_xfer_...",
  "computer_id": "gpu3-a4500",
  "fence_epoch": 42,
  "observed_head": "0123456789abcdef0123456789abcdef01234567",
  "observed_manifest_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "observed_tree_sha256": null,
  "idempotency_key": "apply-gpu3-attempt-1"
}
```
Bei Copy wird `observed_head` weggelassen und `observed_tree_sha256` gesetzt. Optionale Felder werden weggelassen statt mit beliebigen Typen befuellt.

### 16.6 `confirm_working_copy`

```json
{
  "transfer_id": "workjet_xfer_...",
  "computer_id": "gpu3-a4500",
  "fence_epoch": 42,
  "working_copy_id": "workjet_wc_...",
  "path": "opaque://gpu3/worktrees/project-x",
  "idempotency_key": "confirm-wc-gpu3"
}
```
Core berechnet `working_copy_id` erneut. Dieser Command fuehrt den atomaren Switch aus und liefert die neue Session-Epoch.

### 16.7 `resume_ack`

```json
{
  "transfer_id": "workjet_xfer_...",
  "computer_id": "gpu3-a4500",
  "fence_epoch": 43,
  "worker_instance_id": "worker_gpu3_019...",
  "idempotency_key": "resume-worker-gpu3"
}
```
Der Worker muss bereits lokal gestartet sein, darf aber vor dem Ack nur mit der aktuellen Epoch arbeiten. Ein Ack von Mac oder Mobile scheitert.

### 16.8 `abort`

```json
{
  "transfer_id": "workjet_xfer_...",
  "reason": "operator_cancel",
  "idempotency_key": "abort-user-1"
}
```
Owner, Chef oder Admin duerfen abbrechen. Vor dem Commit wird kompensiert; nach dem Commit wird nur ein Forward-Fix markiert.

### 16.9 Status-Tool
MCP-Name:

```text
workjet.session.transfer.status
```
Argumente: genau eines von `transfer_id` oder `session_id`. Das Tool liest Core-/RxDB-Projektion, mutiert nichts und liefert keine Artefaktbytes.

## 17. Fehlercodes
| Code | Retry | Bedeutung | Zustand |
|---|---|---|---|
| `role_or_scope_denied` | nein | Record-scope Policy abgewiesen | unveraendert |
| `session_not_found` | nein | Session fehlt | kein Transfer |
| `session_not_owned` | nein | Ownervergleich fehlgeschlagen | kein Transfer |
| `session_not_running` | nein | unzulaessiger Startzustand | kein Transfer |
| `session_already_transferring` | nein | offener Transfer existiert | bestehender Transfer |
| `session_fenced` | nein | Epoch alt/abweichend | unveraendert |
| `source_working_copy_missing` | nein | Bindung ungueltig | kein Transfer |
| `source_computer_offline` | ja | Quellheartbeat alt | kein Transfer |
| `target_computer_unassigned` | nein | kein Assignment | kein Transfer |
| `target_computer_offline` | ja | Zielheartbeat alt | kein Transfer/shipping |
| `target_computer_is_source` | nein | identischer Rechner | kein Transfer |
| `computer_actor_mismatch` | nein | Peer != Payload/Transferrolle | unveraendert |
| `device_binding_required` | nein | keine gueltige Bindung | unveraendert |
| `transfer_illegal_state` | nein | Verb passt nicht zum State | unveraendert |
| `transfer_expired` | nein | Deadline kompensiert | aborting/rolled_back |
| `pause_timeout` | ja | Quellprozess nicht rechtzeitig still | aborting |
| `git_pack_failed` | ja | Git-Erfassung fehlgeschlagen | packing |
| `copy_not_allowed_for_git_repo` | nein | unerlaubter Fallback | packing |
| `artifact_missing` | ja | File/Generation nicht vorhanden | packing/shipping |
| `artifact_hash_mismatch` | ja | gespeicherte Bytes falsch | packing/shipping |
| `unsupported_file_type` | nein | Copy-Tree nicht darstellbar | packing |
| `target_working_copy_dirty` | nein | Ziel hat eigene Aenderungen | applying |
| `apply_hash_mismatch` | ja | Zielbeweis passt nicht | applying |
| `working_copy_id_mismatch` | nein | deterministische ID falsch | applied |
| `resume_timeout` | ja | Zielworker quittiert nicht | resuming |
| `idempotency_conflict` | nein | gleicher Key, andere Payload | unveraendert |
| `dependency_missing` | nein | unsicherer Control-Claim | unveraendert |
| `manual_intervention` | nein | Kompensation nicht beweisbar | failed |
Fehleroutcomes enthalten stets `ok`, `error_code`, `retryable`, `transfer_id`, `state` und eine redigierte Meldung mit maximal 512 Zeichen.

## 18. Policy-Matrix
| Aktion | User-Policy | Rechner-Gate | Scope | Audit |
|---|---|---|---|---|
| `start` | Owner/Chef/Admin | Ziel assigned + online | Session-Record | decision |
| `pause_ack` | Owner-Kontext | Source-Binding exakt | Transfer-Record | decision |
| `pack_complete` | Owner-Kontext | Source-Binding exakt | Transfer-Record | decision |
| `apply_complete` | Owner-Kontext | Target-Binding exakt | Transfer-Record | decision |
| `confirm_working_copy` | Owner-Kontext | Target-Binding exakt | Session-Record | decision |
| `resume_ack` | Owner-Kontext | Target-Binding exakt | Session-Record | decision |
| `abort` | Owner/Chef/Admin | kein Agent-Gate | Transfer-Record | decision |
| `status` | Owner/Chef/Admin | keines | Record read | read |
`requires_approval` bleibt fuer normale Owner-Mutationen nicht der Ersatz fuer Ownership. Ein fremder User kann einen Session-Transfer nicht ueber eine allgemeine Approval-Kette ausfuehren, weil Rechnerbindung und Arbeitsbaum zum Owner gehoeren.

## 19. Audit-Ereignisse
| Event | Zeitpunkt | Pflichtfelder |
|---|---|---|
| `workjet.session.transfer.policy` | jeder Command-Eintritt | actor, permission, scope, decision |
| `workjet.session.transfer.started` | nach `start` | session, source, target, epoch |
| `workjet.session.transfer.pause_slow` | nach 30 s | transfer, elapsed_ms |
| `workjet.session.transfer.fenced` | nach `pause_ack` | epoch, last_turn |
| `workjet.session.transfer.packed` | nach Pack | mode, manifest hash, generation |
| `workjet.session.transfer.applied` | nach Apply | beobachtete Hashes |
| `workjet.session.transfer.switched` | atomarer Commit | alte/neue WC, neue Epoch |
| `workjet.session.transfer.completed` | nach Resume | target, worker instance |
| `workjet.session.transfer.aborted` | Abort/Timeout | alter State, reason, code |
| `workjet.session.transfer.denied` | Policy/Actor-Fail | reason_code |
| `workjet.session.transfer.manual_intervention` | Recovery-Fail | State, offene Side Effects |
Audit enthaelt keine Patches, Dateinamenlisten, Dateiinhalte, Remote-Credentials, Host-Pfade oder Device-Schluessel. Opaque IDs und Hashes sind zulaessig.

## 20. Konfliktregeln

### 20.1 Zwei Working Copies mit Aenderungen
Die Quelle der Session ist allein massgeblich. Eine andere Working Copy desselben Projekts wird nicht in den Pack einbezogen.
Wenn am Ziel fuer denselben opaken Pfad bereits ein geaenderter Baum existiert, meldet der Target-Agent `target_working_copy_dirty`. V1 bietet weder Auto-Merge noch Auto-Overwrite. Der Owner muss einen frischen Zielpfad waehlen oder die Zielkopie ausserhalb des Transferprotokolls bereinigen und danach neu starten.
Auch im Git-Modus erzeugt Core keinen Merge-Commit. Ein manueller Git-Merge kann vor einem neuen Transfer in einer eigenen Working Copy erfolgen; er ist nicht Teil dieser Saga.

### 20.2 Bearbeitung waehrend des Transfers
- Chat lesen und reine Chat-Nachrichten schreiben bleibt erlaubt, weil der Chat in CTOX liegt. - Ein neuer worker-ausloesender Turn wird ab `pause_requested` mit `session_fenced` abgewiesen. - Steuerpulte duerfen den Prompt lokal als noch nicht gesendet anzeigen und nach `completed` mit der neuen Epoch erneut dispatchen. - Ein bereits laufender Turn darf nach Fence-Erhoehung keinen Snapshot mehr anwenden. - Presence bleibt rein informativ und kann keine Autoritaet verleihen.

### 20.3 Ziel offline
Vor `start` muss der Zielheartbeat frisch sein. Wird das Ziel danach offline, bleibt die Quelle gefenced und der Transfer maximal zehn Minuten in `shipping`. Danach kompensiert Core und setzt die Quelle nach Prozesspruefung wieder `running`.
Nach dem atomaren Switch gibt es keinen Offline-Rollback. Bleibt das Ziel vor `resume_ack` weg, bleibt die Session am Ziel `transfer_failed`/paused; nach Reconnect kann der Target-Agent mit der aktuellen Epoch erneut starten und den Ack wiederholen.

### 20.4 Quelle offline
Faellt die Quelle vor `pause_ack` aus, kann Core nicht beweisen, dass der alte Worker stillsteht. Der Transfer wird nicht fortgesetzt. Erst wenn die alte Actor-Epoch widerrufen oder der Rechner mit einem neuen, beweisbaren Zustand wieder verbunden ist, darf kompensiert beziehungsweise neu gestartet werden.

### 20.5 Doppelter Start
Gleicher Idempotenzschluessel liefert denselben Transfer. Ein anderer Schluessel bei offenem Transfer liefert `session_already_transferring`; es entstehen weder eine zweite Epoch noch ein zweiter Pack.

## 21. Harness- und MCP-Integration
Alle Harnesses verwenden dieselben typisierten Workjet-MCP-Tools:
- `workjet.session.transfer.start` - `workjet.session.transfer.status` - `workjet.session.transfer.abort` - `workjet.session.transfer.resume`
`resume` ist ein Owner-Tool, das bei `resuming` einen erneuten lokalen Start am bereits autoritativen Ziel anfordert; es ist kein weiteres Core-Lifecycle-Verb. Der eigentliche Abschluss bleibt `resume_ack` des Zieladapters.
Die Tools sind duenne Wrapper ueber Business-Commands beziehungsweise read-only Queries. Codex, Claude Code, Cursor, Grok und OpenCode sehen damit dieselbe API und dieselben Fehlercodes.
Desktop erhaelt `globalThis.workjetSessionControl` nach dem Muster von `workjetProjectControl` (`src/apps/business-os/app.js:12591-12722`). Mobile erhaelt eine `session.control`-Bridge nach dem Muster `project.control` (`src/apps/business-os/mobile-host.js:154-166`). Keine Plattform bekommt einen privaten Transfertransport.
Agent-interne Acks werden nicht als frei aufrufbare MCP-Tools exponiert. Sie laufen im lokalen Workjet-Computer-Adapter und benoetigen device-bound Computeridentitaet.

## 22. Zwei-Rechner-Testplan

### 22.1 Topologie
- A: Mac, assigned Workjet-Computer, aktive Quell-WC. - B: `gpu3-a4500`, assigned Workjet-Computer, frischer Zielpfad. - M: Galaxy Fold, nur Steuerpult und Beobachter. - Eine CTOX-Instanz ist fuer die Session autoritativ. - Session `s1` besitzt einen bestehenden Chat-Thread.

### 22.2 Vorbereitung
1. Projekt `p1` anlegen. 2. Mac-Working-Copy `wc_mac` aktiv anlegen. 3. Session `s1` an `wc_mac` und den Chat-Thread binden. 4. Einen Worker auf Mac mit Epoch `e0` starten. 5. Einen Turn abschliessen und einen zweiten Turn zur Fence-Pruefung vorbereiten. 6. In der Git-WC eine versionierte Datei aendern. 7. Eine nicht versionierte Datei anlegen. 8. Galaxy Fold mit derselben CTOX-Instanz synchronisieren.

### 22.3 Mac nach `gpu3-a4500`: Git-Pfad
1. Fold oder Mac ruft `workjet.session.transfer.start` auf. 2. Auf allen Apps erscheint derselbe Transfer mit `pause_requested`. 3. Ein neuer Turn auf Mac wird mit `session_fenced` abgewiesen. 4. Ein absichtlich laufender Pi-Turn wird beendet; sein Snapshot darf nicht angewandt werden. 5. Mac sendet `pause_ack`; Audit zeigt `fenced` mit Epoch `e0+1`. 6. Mac erzeugt Git-Bundle, Patch und Untracked-Bundle. 7. `pack_complete` setzt `mode=git` und unveraenderliche Hashes. 8. `gpu3-a4500` zieht nur
die referenzierten File-Generationen. 9. Ziel stellt HEAD, Patch und nicht versionierte Datei her. 10. `apply_complete` stimmt mit dem Manifest ueberein. 11. `confirm_working_copy` fuehrt den atomaren Commit aus. 12. Core-Store zeigt `s1.working_copy_id=wc_gpu`. 13. `wc_mac.status=detached`, `wc_gpu.status=active`. 14. `wc_gpu.verified_at_ms` ist gesetzt. 15. Session-Epoch ist `e0+2`. 16. Mac-Worker kann mit `e0` oder `e0+1` nicht mehr schreiben. 17. Zielworker startet mit `e0+2` und sendet
`resume_ack`. 18. Ein Testturn auf Ziel sieht beide lokalen Aenderungen. 19. Chat-ID und Verlauf sind unveraendert. 20. Fold sieht `completed` und kann danach einen neuen Turn steuern.

### 22.4 `gpu3-a4500` nach Mac: Git-Rueckweg
1. Auf Ziel eine weitere versionierte und nicht versionierte Aenderung erzeugen. 2. Neuen Transfer mit neuem Idempotenzschluessel nach Mac starten. 3. Mac verwendet einen frischen Zielpfad oder eine nachweislich saubere vorbereitete Working Copy. 4. Derselbe Pause-/Pack-/Apply-/Commit-/Resume-Ablauf gilt. 5. Nach Abschluss ist `wc_gpu.status=detached` und die neue Mac-WC `active`. 6. Byte-/Hashvergleich beweist, dass beide Aenderungsrunden vorhanden sind. 7. Genau ein Worker akzeptiert Turns.

### 22.5 Copy-Pfad ohne Git
1. Separate Session auf einem Verzeichnis ohne benutzbares Git anlegen. 2. Transfer Mac nach `gpu3-a4500` starten. 3. `pause_ack.git_repository=false` beweisen. 4. `pack_complete.mode=copy` akzeptieren. 5. DataChannel-Unterbrechung in der Mitte eines File-Fetch simulieren. 6. Resume des Demand-Streams beobachten. 7. Zielbaum materialisieren und `tree_sha256` vergleichen. 8. Atomar umschalten und Worker starten. 9. Ruecktransfer nach Mac wiederholen.

### 22.6 Negativlaeufe
| Test | Erwartung |
|---|---|
| Ziel vor Start offline | `target_computer_offline`, keine Epoch-Aenderung |
| Ziel waehrend Shipping offline | nach 10 min Rollback zur Quelle |
| Source-Ack vom Fold | `computer_actor_mismatch` |
| Target-Ack vom Mac | `computer_actor_mismatch` |
| Copy trotz Git | `copy_not_allowed_for_git_repo` |
| Dirty Zielpfad | `target_working_copy_dirty` |
| Falscher Manifest-Hash | `apply_hash_mismatch` |
| Falsche WC-ID | `working_copy_id_mismatch` |
| Doppelter identischer Start | gleicher Transfer, gleiche Epoch |
| Gleicher Key, anderes Ziel | `idempotency_conflict` |
| Direktes Peer-Write | abgewiesen, Core unveraendert |
| Resume-Ack nach Actor-Widerruf | `device_binding_required` |

### 22.7 Harte Erfolgsbeobachtungen
Der Test ist nur erfolgreich, wenn alle folgenden Beobachtungen vorliegen:
- Core SQLite und alle drei Projektionen konvergieren auf dieselbe Session-WC.
- Zu keinem Zeitpunkt sind fuer die Session zwei WCs autoritativ aktiv.
- Die Source-WC ist nach jedem erfolgreichen Commit `detached`.
- Alte Epochs koennen weder neue Turns starten noch Snapshots anwenden.
- Git-HEAD, Patch und Untracked-Bundle sind am Ziel nachgewiesen.
- Copy-Tree-Hash stimmt im No-Git-Lauf.
- Command-Replay erzeugt keine zweite Epoch oder Generation.
- Audit enthaelt Start, Fence, Pack, Apply, Switch und Completion.
- Galaxy Fold beobachtet passiv alle Zustaende und steuert nach Abschluss einen Turn, ohne selbst Agent-Acks senden zu koennen.
- Kein HTTP-Datenpfad wird verwendet.

## 23. Umsetzungsschnitt in Produktionspakete

### Paket 1: Core-Register, Computerintegration und Policy
- Schwierigkeit: **3/5**
- Reihenfolge: **1**
- Inhalt:
  - Stores und Schemata fuer `workjet_sessions` und `workjet_session_transfers`.
  - `workjet_computers` vollstaendig verdrahten.
  - Record-scoped `DataWrite` mit `owned_by_actor`.
  - Native Projektionsliste und Owner-Migration.
- Whitelist-Vorschlag:
  - `src/core/business_os/mod.rs`, `policy.rs`, `threads.rs`, `command_plane.rs`
  - `src/core/business_os/store_workjet_computers.rs`
  - `src/core/business_os/store_workjet_sessions.rs` neu
  - Schema-Contract, Hash-Registry und Browser-Schema-/Registry-Dateien
- Gate-Test des Workers: `cargo test workjet_session -- --nocapture`, `cargo test record_owned_data_write -- --nocapture` und Schema-Hash-Registry-Smoke.

### Paket 2: Transferjournal, Commands und Idempotenz
- Schwierigkeit: **4/5**
- Reihenfolge: **2**
- Inhalt:
  - Sieben Lifecycle-Commands.
  - Zustandsautomat und Recovery-Pass.
  - Step-Idempotenz, stabile Fehlercodes und Audit-Ereignisse.
- Whitelist-Vorschlag:
  - `src/core/business_os/command_plane.rs`
  - `src/core/business_os/store_workjet_sessions.rs`
  - `src/core/business_os/store_policy_audit.rs`
  - `src/core/business_os/business_command_inventory.json` und Inventory-Builder/-Tests
- Gate-Test des Workers: `cargo test workjet_session_transfer -- --nocapture`, Replay ohne zweite Epoch und State-Table-Test aller erlaubten und verbotenen Uebergaenge.

### Paket 3: Fence im Coding-Pfad und lokale Harness-Adapter
- Schwierigkeit: **5/5**
- Reihenfolge: **3**
- Inhalt:
  - `ctox.coding.turn` mit Session-ID und Epoch.
  - Admission-Check, Cancellation-Registry und zweiter Fence-Check vor Snapshot-Apply.
  - Pi-Sidecar Hard-Cancel und Adaptervertrag fuer Codex, Claude Code, Cursor, Grok und OpenCode.
- Whitelist-Vorschlag:
  - `src/core/business_os/store.rs`
  - `src/core/coding_agents/mod.rs`, `pi_sidecar.rs` und `pi-sidecar/src/*`
  - Harness-Adapter-/Contract-Dateien und zugehoerige Tests
- Gate-Test des Workers: kontrollierter langer Faux-Turn, Fence waehrend des Turns, Prozess beendet, kein Snapshot angewandt, Outcome `session_fenced`.

### Paket 4: Git-Artefaktpfad und atomarer Switch
- Schwierigkeit: **4/5**
- Reihenfolge: **4**
- Inhalt:
  - Git-Bundle, Patch, Untracked-Bundle und Manifest/Hashing.
  - `desktop_files`-Verknuepfung und Target-Apply.
  - Atomarer WC-/Session-Switch, Source-Detach und `verified_at_ms`.
- Whitelist-Vorschlag:
  - Neuer Workjet-Transfer-Agent unter `src/core/coding_agents/` oder bestehendem Workjet-Adapterpfad
  - `src/core/business_os/store_workjet_sessions.rs`
  - `src/core/business_os/store_workjet_projects.rs`
  - Gezielte File-Store-Helfer und zugehoerige Tests
- Gate-Test des Workers: temporaeres Git-Repo mit dirty und untracked Inhalt, Pack/Apply zwischen zwei Tempdirs, Hashgleichheit, Source detached, Target active und Session umgebunden.

### Paket 5: Copy-Pfad, Projektion und MCP/UI
- Schwierigkeit: **5/5**
- Reihenfolge: **5**
- Inhalt:
  - Deterministischer Tree-Walk ohne Git, Files/Chunks, Resume und Retention/GC.
  - `workjetSessionControl` fuer Desktop/Mobile und typisierte MCP-Tools.
  - Zwei-Rechner-E2E Mac -> GPU -> Mac.
- Whitelist-Vorschlag:
  - Workjet-Transfer-Agent-Dateien
  - `src/core/business_os/mcp_channel.rs`
  - `src/apps/business-os/app.js` und `mobile-host.js`
  - Workjet-Control-, RxDB-File-Demand- und Zwei-Rechner-E2E-Tests
- Gate-Test des Workers: No-Git-Tree groesser als 8 MiB, unterbrochener Demand-Fetch mit Resume, Tree-Hash gleich, MCP-Descriptor-/Dispatch-Test; Mobile darf beobachten und steuern, aber keinen Agent-Ack senden.

## 24. Reihenfolge und Release-Gates
Die Reihenfolge ist bindend: **P1 -> P2 -> P3 -> P4 -> P5**.
P1 und P2 duerfen ohne Workerstart hinter einem Feature-Flag projektiert werden. P3 ist das Sicherheitsgate: Kein echter Transfer darf freigeschaltet werden, bevor ein laufender Turn nach Fence garantiert keinen Snapshot mehr anwenden kann.
P4 liefert den ersten produktionsfaehigen End-to-End-Pfad, weil Git der Normalfall ist. P5 ergaenzt den No-Git-Copy-Pfad und die vollstaendige plattformuebergreifende Bedienung.
Release-Gates:
1. Schema-Hashes konvergieren auf Rust und Browser. 2. Direct-Peer-Write-Tests lehnen alle drei nativen Collections ab. 3. Policy-Test erlaubt normalen Owner nur auf eigenem Sessionrecord. 4. Fence-Race-Test beweist keinen Apply nach Pause. 5. Atomarer Switch-Test beweist Target active und Source detached. 6. Zwei-Rechner-Git-E2E ist gruen. 7. Zwei-Rechner-Copy-E2E ist gruen. 8. Galaxy-Fold-Beobachtung und nachgelagerte Steuerung ist gruen.

## 25. Sicherheits- und Datenschutzfolgen
- Die neue Autoritaet liegt ausschliesslich in Core SQLite. - RxDB ist read-only Projektion fuer Sessions, Transfers und Computer. - Device-Binding und Actor-Epoch verhindern Acks von einem anderen Rechner. - Opaque Pfade bleiben opak; Core oeffnet sie nicht. - Audit speichert keine Arbeitsbauminhalte. - Artefakte haben begrenzte Retention. - File-Bytes nutzen den bestehenden WebRTC-/RxDB-Pfad. - Ein alter Worker kann durch die Epoch auch nach Netzpartition nicht wieder autoritativ werden. - Der
kritischste Punkt ist der zweite Fence-Check unmittelbar vor dem Dateischreibpfad; ohne ihn ist die RFC nicht implementiert.

## 26. Betriebsverhalten
Operatoren sehen pro Session:
- aktuellen Rechner und Working Copy; - `run_status` und Fence-Epoch; - offenen Transfer und Zustand; - letzte serverseitige Deadline; - redigierten Fehlercode; - Retry-/Abort-Moeglichkeit gemaess Zustand.
Ein Operator darf nie durch manuelles Editieren der Projektion reparieren. Recovery erfolgt durch `abort`, einen erneuten Target-Resume oder einen neuen Transfer nach terminalem Rollback.
Metriken:
- Dauer je Zustand; - Pause-Cancel-Latenz; - Artefaktbytes und Chunk-Retries; - Hash-Mismatch-Zaehler; - Rollback- und Forward-Fix-Zaehler; - Projektionslatenz je App; - Anzahl abgewiesener alter Epochs.

## 27. Kompatibilitaet und Migration
Bestehende Projekte und Working Copies bleiben gueltig. Eine Workjet-Session wird beim ersten Start/Import angelegt und explizit an eine bestehende aktive Working Copy gebunden.
Bestehende `coding_agent_sessions` werden nicht automatisch zu `workjet_sessions` migriert, weil ihr `workspace_root` kein verlaesslicher Working-Copy-Fremdschluessel ist. Eine UI darf passende Kandidaten anzeigen; die Bindung muss ueber Projekt, Computer und opaken Pfad bestaetigt werden.
Alte Harnesses ohne Epoch-Unterstuetzung duerfen Sessions lesen, aber keine transferfaehige Session treiben. Sobald `active_transfer_id` oder eine Fence-Epoch groesser null gesetzt ist, ist ein Epoch-freier Turn `session_fenced`.

## 28. Verworfene Alternativen

### 28.1 Felder an `coding_agent_sessions`
Verworfen, weil der Record vom Turn-Logger als Workbench-Projektion geschrieben wird, pro App statt pro wandernder Session modelliert ist und nach dem Apply nur best-effort aktualisiert wird.

### 28.2 Uhrzeit-Lease als Autoritaet
Verworfen, weil Clock-Skew, Suspend und Netzpartition ein abgelaufenes Lease nicht zu einem sicheren Beweis machen. Deadlines bleiben fuer Recovery, aber die Epoch entscheidet jede Schreibberechtigung.

### 28.3 Generische Commands mit `progress`
Verworfen, weil sie Actor-Rollen und Side-Effect-Beweise vermischen. Fortschritt ist Telemetrie; Lifecycle-Acks sind autoritative Zustandsbeweise.

### 28.4 Workspace-weites `DataWrite`
Verworfen, weil `src/core/business_os/policy.rs:370-373` damit normale Owner ohne Assignment sperrt oder alternativ zu breite Workspace-Rechte verlangen wuerde.

### 28.5 Eigene `transfer_chunks`
Verworfen, weil Files/Chunks bereits Frame-Budget, Resume, Demand-Fetch, Hashing und Materialisierung besitzen. Eine zweite Datenebene wuerde nur Drift erzeugen.

### 28.6 Source-WC bis nach Resume aktiv lassen
Verworfen. Der verbindliche Commit detacht die Quelle atomar mit dem Switch; sonst waere die Location of Record nach dem Commit mehrdeutig.

## 29. Anhang A: Abweichungen der drei Vorentwuerfe

### A.1 Sessionregister
- Grok und GLM schlugen eine eigene `workjet_sessions`-Collection vor. - Kimi schlug Felder an `coding_agent_sessions` vor. - Entscheidung: eigene Collection, weil der im Klon verifizierte Schreiber `coding_agent_sessions` nach jedem Pi-Turn als best-effort Workbench-Record aktualisiert.

### A.2 Lease und Fence
- Kimi bevorzugte eine Uhrzeit-Lease. - GLM bevorzugte eine kausale Epoch. - Grok kombinierte Transferzustand und Lease-Gedanken. - Entscheidung: Epoch fuer Autoritaet, serverseitige Deadlines nur fuer Liveness und Recovery.

### A.3 Commands
- Grok/GLM benannten konkrete Agent-Acks. - Kimi fasste Schritte in `request/status/progress/confirm/abort/resume` zusammen. - Entscheidung: konkrete Acks, weil Source und Target jeweils andere, device-bound Beweise liefern. - `status` bleibt als read-only Tool erhalten, aber ausserhalb des Mutationsautomaten.

### A.4 Policy
- Grok identifizierte, dass Workspace-`DataWrite` normale Owner ausschliesst. - Kimi betonte Device-Proof und Zielverifikation. - GLM schlug record-scoped Ownership vor. - Entscheidung: `DataWrite` darf `owned_by_actor` nur auf Record-Scope nutzen; Agent-Acks brauchen zusaetzlich eine aktive Bindung an `workjet_computers`.

### A.5 Transport
- Ein Entwurf bevorzugte eine eigene Transfer-Chunk-Collection. - Zwei Entwuerfe verwendeten bestehende Files/Chunks. - Entscheidung: Wiederverwendung von `desktop_files`/ `desktop_file_chunks` mit expliziter Transfer-Metadatenkonvention und Retention.

### A.6 Working-Copy-Lebenszyklus
- Ein Entwurf liess die Quell-WC als Cache weiter `active`. - Andere Entwuerfe empfahlen `detached` zur Split-Brain-Vermeidung. - Entscheidung: Quelle wird im atomaren Commit `detached`, entsprechend dem verbindlichen Zielbild.

### A.7 Ziel offline
- Ein Entwurf wollte unbegrenzt pausiert warten. - Andere Entwuerfe sahen kurze automatische Rollbacks vor. - Entscheidung: vor Commit maximal zehn Minuten Shipping-Wartezeit, danach Rollback; nach Commit bleibt das Ziel autoritativ und wird vorwaerts repariert.

### A.8 Copy-Fallback
- Einige Formulierungen liessen `auto|git|copy` als Benutzerwahl zu.
- Entscheidung: Der Source-Agent waehlt deterministisch Git, wenn Git vorhanden ist; Copy ist nur fuer No-Git-Arbeitsbaeume zulaessig.

## 30. Anhang B: Normative Implementierungscheckliste
- [ ] `workjet_sessions` geschlossenes Schema angelegt. - [ ] `workjet_session_transfers` geschlossenes Schema angelegt. - [ ] `workjet_computers` vollstaendig verdrahtet und schemaregistriert. - [ ] Drei Collections in `NATIVE_PROJECTION_COLLECTIONS`. - [ ] Schema-Contract und Hash-Registry beidseitig aktualisiert. - [ ] Record-owned `DataWrite` implementiert und getestet. - [ ] Device-Binding -> Computer-ID serverseitig aufgeloest. - [ ] Sieben Lifecycle-Commands registriert. - [ ] Fachliche
Idempotenz je Step implementiert. - [ ] Fence vor Turn und vor Snapshot-Apply geprueft. - [ ] Laufender Pi-/Harness-Prozess abbrechbar. - [ ] Git-Bundle, Patch und Untracked-Bundle implementiert. - [ ] Copy-Tree-Walk und Chunk-Resume implementiert. - [ ] Atomarer Switch aktiviert Ziel und detacht Quelle. - [ ] `verified_at_ms` nur im Transfer-Commit gesetzt. - [ ] Audit ohne Dateiinhalt implementiert. - [ ] Retention/GC fuer Transferartefakte implementiert. - [ ] Desktop-, Mobile- und
MCP-Control vorhanden. - [ ] Mac -> `gpu3-a4500` -> Mac Git-E2E gruen. - [ ] No-Git-Copy-E2E gruen. - [ ] Galaxy Fold beobachtet und steuert nach Abschluss. - [ ] Keine direkte Peer-Mutation moeglich. - [ ] Kein HTTP-Datenpfad hinzugefuegt.
