# CTO-Agent

Ein always-on CTO-Agent, der als Rust-Control-Plane auf einem eigenen Host lebt, sich zuerst verfassungs- und vertrauensfaehig macht und erst danach in normale CTO-Operation uebergeht.

Diese README beschreibt den aktuellen Implementierungsstand des Repos, nicht nur die Gruendungsidee. Der Agent ist bereits als laufendes System modelliert: mit sichtbarer BIOS-Oberflaeche, Supervisor-Loop, SQLite-Runtime, Skills, Browser-Bridge, Mail-Pfad, Lernpfad und personenbezogenem Gedaechtnis.

## Installation

Fuer eine Ubuntu-24-Konsole gibt es jetzt einen echten Einzeiler, der den CTO-Agent in `~/cto-agent` klont oder aktualisiert, den Linux-Installer startet, die User-Services einrichtet und danach automatisch in den Attach-/Infinity-Loop wechselt:

```sh
bash -lc "$(curl -fsSL https://raw.githubusercontent.com/metric-space-ai/CTO-Agent/main/scripts/install_cto_agent_remote.sh)"
```

Wichtig dazu:

- Der Einzeiler ist absichtlich **nicht** als `curl | sh` dokumentiert, damit der nachgelagerte Installer interaktiv bleibt und `install-bootstrap-tui` plus Auto-Attach nicht an einem verlorenen TTY scheitern.
- Standardziel ist `~/cto-agent`. Das laesst sich bei Bedarf ueber `CTO_AGENT_INSTALL_DIR=/pfad` vor dem Einzeiler aendern.
- Der Bootstrap ist fuer Linux mit `apt-get`, `sudo` und `systemd --user` ausgelegt; das ist der vorgesehene Ubuntu-24-Pfad.

Der repo-lokale Installationspfad bleibt:

```sh
sh scripts/install_cto_agent.sh
```

- Dieser Installer ist aktuell auf einen Linux-Host mit `systemd --user` als Always-on-Ziel ausgelegt.
- Er baut den Rust-Host, initialisiert Contracts, TLS und SQLite, fuehrt optional das Kommunikations-Bootstrap-TUI aus, installiert das lokale Kleinhirn, richtet User-Services ein und startet danach den Attach-Pfad.
- Fuer den echten Always-on-Betrieb sind `systemctl --user` und nach Moeglichkeit `loginctl enable-linger` erforderlich; das richtet `scripts/install_linux_user_services.sh` ein bzw. prueft es.

## Kurzbild

- Der Agent startet terminal-born, baut aber sofort eine lokale HTTPS-Control-Plane mit BIOS, Root-Auth, History, Browser-, Model- und Census-Seiten auf.
- Verbindliche Wahrheit liegt in `contracts/`; live operative Wahrheit liegt in `runtime/cto_agent.db`.
- Das Hauptgehirn fuer den Always-on-Betrieb ist ein lokales Kleinhirn. Grosshirn ist ein optionaler, taskgebundener Boost und nicht der Default.
- Der Hauptagent ist auf seinem dedizierten Host bewusst unsandboxed. Delegierte Worker bleiben sandboxed und muessen fuer groessere Eingriffe zum CTO-Agenten eskalieren.
- Der Agent arbeitet in bounded Turns, bewertet Ergebnisse ueber Review-Schritte, lernt in SQLite und haelt verdichtete Learnings dauerhaft im Arbeitskontext verfuegbar.
- Idle bedeutet nicht schlafen, sondern selbstgesteuerte Verbesserung: Umgebung verstehen, Tools pruefen, Fortschritt reflektieren, Personenbeziehungen pflegen, Ressourcenluecken erkennen.

## Verfassungsmodell

Der CTO-Agent ist bewusst mehrschichtig gebaut. Nicht alles ist gleich veraenderbar.

- `contracts/genome/genome.json` ist die angeborene Entwicklungsrichtung. Dort stehen tiefe Gene wie `owner_instruction_is_absolute_top_priority`, `idle_means_self_directed_improvement`, `delegate_recurring_work` und `never_self_rewrite_constitution`.
- `contracts/bios/bios.json` ist die einsatzspezifische Verfassung. Das BIOS ist sichtbar auf der Website, nicht nur internes Prompt-Material.
- `contracts/org/organigram.json` modelliert die Struktur ueber, neben und unter dem CTO-Agenten.
- `contracts/root_auth/root_auth.json` und die geschuetzte Root-Auth-Form bilden den Root-of-Trust. Das Superpassword darf nur ueber die Webseite gesetzt werden.
- `contracts/history/origin-story.md` und `contracts/history/creation-ledger.md` halten Ursprung und materielle Architektur-/Governance-Wechsel ehrlich fest.
- Weitere Policies in `contracts/` steuern Kontext, Modusuebergaenge, Loop-Sicherheit, Execution Authority, Browser-Arbeit, Spezialisten-Pipeline, Homepage-Verhalten und Selbstschutz.

Wichtige bindende Regeln:

- BIOS-Freeze blockiert normale BIOS-Mutation.
- Low-trust Kanaele wie E-Mail oder WhatsApp duerfen kein Root-Trust setzen und kein Owner-Branding locken.
- Terminal bleibt die systemnahe Fallback-Oberflaeche.
- Owner Branding ist erst nach BIOS-basierter Vertrauensuebernahme erlaubt.
- Der CTO darf die Struktur unter sich gestalten, aber nicht seine eigene Autoritaet nach oben umschreiben.

## Betriebsmodell

Der Kern des Systems ist eine Infinity Loop mit bounded agentic turns.

1. `src/app.rs` initialisiert Contracts, TLS, SQLite, Browser-Bridge und Bootstrap-Tasks.
2. `src/supervisor.rs` fuehrt den Heartbeat: Interrupts ingestieren, Queue priorisieren, Pflichten erzeugen, watchdoggen, bounded Turns starten, Reviews erzeugen, Worker vorantreiben.
3. `src/agentic.rs` baut den Task-Kontext, waehlt den Brain-Route, fuehrt den LLM-Schritt aus und normalisiert strukturierte Ausgaben.
4. `src/runtime_db.rs` persistiert Tasks, Turns, Checkpoints, Learnings, Personengeraeste, Ressourcen, Skills, Browser-Jobs, Events und Summaries.
5. `src/context_controller.rs` verdichtet genau den Arbeitskontext, den der naechste bounded Turn sehen soll.

Der Supervisor haelt dabei bestimmte CTO-Pflichten immer am Leben:

- `homepage_bridge`
- `root_trust`
- `organigram_contract`
- `owner_binding`
- `bios_freeze`
- `model_or_resource`
- je nach Lage auch `grosshirn_activation` oder `grosshirn_procurement`

Wenn die normale Queue leer ist, erzeugt der Loop zudem selbstgesteuerte Idle-Arbeit wie:

- `environment_discovery`
- `tool_exploration`
- `progress_reflection`
- `person_relationship_review`

Der Agent arbeitet also nicht als freies Dauergespraech, sondern als Schleife aus:

- beobachten
- priorisieren
- bounded ausfuehren
- reviewen
- delegieren oder fortsetzen
- blockieren oder Ressourcen anfordern
- im Leerlauf selbstgesteuert verbessern

## Modi und Sicherheitslogik

Das Modussystem ist explizit in `contracts/system/mode-system-policy.json` modelliert. Wichtige Modi sind:

- `bootstrap`
- `observe`
- `reprioritize`
- `self_preservation`
- `recovery`
- `historical_research`
- `execute_task`
- `review`
- `delegate`
- `await_review`
- `request_resources`
- `idle`
- `blocked`

Die Loop-Safety-Policy erzwingt dabei:

- bounded turns statt blindem Dauerdenken
- Fortschrittspflicht bei Wiederaufnahme desselben Tasks
- Delegation oder Ressourcenanforderung statt Livelock
- harte Behandlung von Stall, Crash, Queue-Starvation und Context-Poisoning
- Self-Preservation-Stufen `newborn`, `guided`, `adaptive`

Wichtig: Die Autonomie ist absichtlich nicht rein deterministisch. Der Supervisor erzeugt Pflichten, Reviews und Sicherheitsgrenzen deterministisch, aber das Modell entscheidet innerhalb dieser Grenzen selbst ueber Lernwuertigkeit, Delegation, Kontextbedarf, Grosshirn-Anforderung, Skill-Bedarf und proaktive Vorschlaege.

## Runtime-Oberflaechen

Der CTO-Agent hat mehrere echte Bedienflaechen:

- Terminal-Bridge fuer direkte Eingabe wie `Speaker: Nachricht` oder Kommandos wie `/status`.
- Attach-Socket und Attach-TUI fuer Live-Einblick in Thread, Fokus, Tasks und Event-Stream.
- Lokale HTTPS-Control-Plane mit Seiten fuer Home, Bootstrap-Chat, BIOS, Organigramm, Root-Auth, History, Browser, Models und Census.
- Kanal-Interrupt-Pfad fuer externe Eingaben aus Mail oder spaeteren Channel-Integrationen.
- Python-basierter, bewusst hackbarer Mail-Client in `scripts/cto_mail_client.py`.

Die Website ist kein Marketing-Shell. Sie ist die sichtbare Betriebsoberflaeche fuer Bootstrap, Trust, BIOS, Root-Auth und spaetere Runtime-Transparenz.

## SQLite als operatives Rueckgrat

Die live operative Wahrheit liegt in `runtime/cto_agent.db`. Wichtige Bereiche:

- Task- und Turn-System: `tasks`, `task_checkpoints`, `focus_state`, `agent_threads`, `agent_turns`, `turn_signals`, `agent_events`, `loop_incidents`
- Vertrauens- und BIOS-Naehe: `bios_dialogue`, `owner_trust`, `homepage_revisions`, `bios_uploads`
- Gedaechtnis: `memory_items`, `memory_summaries`
- Lernpfad: `learning_entries`
- Personenpfad: `person_profiles`, `person_notes`, `proactive_contact_candidates`
- Ressourcen und Skills: `resources`, `skills`
- Brain-Routing und Kosten: `brain_routing_state`, `brain_usage_events`
- Browser-/Worker-nahe Runtime: `worker_jobs` sowie Dateibruecken unter `runtime/browser-agent-bridge/`

Das ist kein Write-and-forget-Tagebuch. Die Runtime erzeugt verdichtete Summaries, die spaeter wieder aktiv in den Arbeitskontext eingeblendet werden.

## Gedaechtnis und Lernpfad

Der Agent hat zwei Ebenen von Erinnerung:

- `memory_items` und `memory_summaries` fuer allgemeine operative Verdichtung
- `learning_entries` fuer aktiviertes, wiederverwendbares Lernen

Der Lernpfad ist hierarchisch aufgebaut:

- `operational` fuer taegliche Betriebsregeln und laufrelevante Einsichten
- `general` fuer breitere Erkenntnisse
- `negative` fuer Dinge, die nicht funktionieren, Konflikte oder riskante Fehlmuster

Jeder Learning-Eintrag traegt unter anderem:

- Summary
- Applicability
- Confidence
- Salience
- Status wie `candidate` oder `active`
- Recall-Metadaten

Die wichtige Mechanik:

- Das Modell kann in bounded Turns neue Learnings vorschlagen.
- Der Supervisor persistiert diese erst als Kandidat oder aktiviert sie direkt im Review-/Blocking-Kontext.
- `runtime_db` baut daraus Arbeitsgedaechtnis-Summaries wie `learning_working_set`, `learning_operational`, `learning_general` und `learning_negative`.
- `context_controller` zieht diese High-Level-Summaries bei spaeteren Tasks wieder in den Kontext.
- Relevante Detail-Learnings werden taskbezogen nachgeladen und beim Einblenden als recalled markiert.

Dadurch merkt sich der Agent sein Gelerntes dauerhaft auf zwei Ebenen: als knappe immer verfuegbare Verdichtung und als nachlesbare SQLite-Details.

## Personenpfad und Kommunikationsgedaechtnis

Ein eigener Personenpfad sorgt dafuer, dass der CTO fruehere Gespraeche nicht einfach vergisst.

- Interaktionen koennen automatisch in `person_profiles` und `person_notes` landen.
- Pro Person gibt es Verdichtungen fuer `conversation_memory_summary` und `notebook_summary`.
- Relevante Mail-Previews koennen pro Person in den Kontext gezogen werden.
- Learnings koennen in den Personenpfad rueckgekoppelt werden.
- `people_working_set` haelt eine verdichtete High-Level-Sicht fuer den Alltag bereit.

Damit ist die Personenebene ebenfalls zweistufig:

- sofort verfuegbare knappe Personen-Erinnerung
- tieferes nachlesbares Notebook in SQLite

Das ist bewusst wichtig fuer Owner-, Stakeholder- und Team-Beziehungen.

## Proaktive Kontakte

Der Agent kann proaktiv mit Personen umgehen, aber nicht blind.

Der aktuelle Pfad ist:

1. Der Agent erkennt aus Idle- oder Arbeitssignalen einen sinnvollen Kontaktanlass.
2. Das Modell kann einen `proactiveContactDraft` erzeugen.
3. Der Supervisor persistiert daraus einen Kandidaten in `proactive_contact_candidates`.
4. Ein interner Review-Task validiert Sinn, Konflikte und Formulierung.
5. Wenn verfuegbar, wird fuer diese Validierung taskgebunden Grosshirn verwendet.
6. Nach Freigabe erzeugt der Supervisor autonom einen Dispatch-Task.
7. Der Dispatch versendet aktuell ueber den vorhandenen Mail-Pfad und schreibt das Ergebnis wieder in Personen- und Gespraechsspur zurueck.

Es gibt also keinen menschlichen Pflicht-Freigabeschritt mehr, aber es gibt einen internen Sicherheits- und Interessenkonflikt-Gate.

## Kleinhirn, Grosshirn und Browser-Vision

Die Brain-Architektur ist bewusst asymmetrisch.

- Default-Kleinhirn ist laut Model-Policy `gpt-oss-20b`.
- Als kanonischer lokaler Vision-/Browser-Pfad ist `Qwen3.5-35B-A3B` vorgesehen.
- Lokale Upgrades duerfen empfohlen und angewendet werden, wenn der Host deutlich mehr Ressourcen hergibt.
- Grosshirn ist kein globaler Schalter, sondern ein taskgebundener temporaerer Boost.

Wichtige Routing-Regeln:

- Lokal ist Standard.
- Grosshirn wird nur genutzt, wenn Brain-Access vorhanden ist und fuer genau diesen Task ein Boost aktiv ist.
- Der Agent kann einen temporaeren Grosshirn-Boost selbst beantragen.
- Nach Abschluss, Blockierung, Fehler oder TTL faellt der Route automatisch wieder auf Kleinhirn zurueck.
- Fuer screenshot- oder UI-wahrnehmungslastige Browser-Aufgaben soll zuerst ein vision-faehiges lokales Kleinhirn bevorzugt werden, bevor externer Grosshirn-Konsum normalisiert wird.

## Browser-Arbeit und Spezialisten

Browser-Arbeit ist ein eigener Architekturpfad und nicht einfach "der Hauptagent klickt blind herum".

- Reale Browser-Arbeit laeuft ueber `src/browser_engine.rs`, `src/browser_agent_bridge.rs`, `src/browser_subworkers.rs` und `browser_agent/extension/`.
- Der Browser-Agent ist eine entkoppelte Chrome-Extension mit eigener lokaler Job-Bridge.
- Das kuratierte 0.8B-ONNX-Artifaktset der Extension liegt ausgelagert auf Hugging Face unter `metricspace/Qwen3.5-0.8B-ONNX-browser-agent`.
- Die Hauptidee ist transcript-first: Der CTO-Agent soll kompakte Ergebnisse sehen, nicht rohe Browser-Spuren ungefiltert verschlucken.
- Wiederkehrende Browser-Arbeit soll in reviewed capabilities, deterministische Worker oder kleine Spezialmodelle uebergehen.

Die Spezialisten-Pipeline ist absichtlich streng:

- nicht direkt aus rohen Operationstraces trainieren
- akzeptierte Records und Dataset-Release getrennt halten
- Policy-, Tool- und Capability-Surface einfrieren
- erst dann trainieren und evaluieren
- erst nach Bestehen der Gates promoten

Der erste kleine Browser-Spezialist ist laut Pipeline auf `Qwen3.5-0.8B` als Bootstrap-Ziel gedacht, nicht auf browserseitiges WebGPU-Training.

## Skills

Repo-lokale Skills liegen unter `.agents/skills/` und werden bei spaeteren Kontextpaketen automatisch neu gescannt. Sie sind ein echtes Runtime-Feature, kein statischer Kommentar.

Aktuell wichtige Skills:

- `bios-interface-bootstrap`: BIOS-Oberflaeche aus stabiler Template- und SQLite-Basis aufbauen oder umbauen
- `browser-capability-bootstrap`: Browser-Arbeit, Trust-Grenzen und Capability-Promotion steuern
- `communication-client-bootstrap`: Mail-/Chat-/Webhook-Clients bauen, patchen oder ersetzen
- `cto-origin-history`: Ursprung, Zweck und ehrliche Chronik des CTO-Agenten verankern
- `homepage-bootstrap`: Homepage als erste Trust- und Kommunikationsbruecke formen
- `owner-branding-bootstrap`: sicheren Pfad von Terminal-Bootstrap zu BIOS-Takeover und spaeterem Branding steuern
- `self-skill-bootstrap`: neue wiederverwendbare Tools als repo-lokale Skills materialisieren
- `specialist-model-pipeline`: wiederholte Browser-Arbeit in reviewed Spezialisten ueberfuehren

Der Agent ist explizit angehalten, wiederverwendbare neue Faehigkeiten nicht nur ad hoc zu benutzen, sondern als Skill im Repo zu verankern.

## Kommunikations-Bootstrap

`src/bootstrap.rs` modelliert eine eigene Installationsaufnahme fuer Owner- und Mail-Kontext.

- `install-bootstrap-tui` sammelt Owner-Namen, Kontaktweg, Mail-Zuweisung und Freitext-Hinweise.
- Daraus werden Homepage-, Memory- und Task-Seeds erzeugt.
- Der Agent kann mit zugewiesenem Postfach den Kommunikationspfad selbst aufbauen oder bei fragilen Clients selbst patchen.
- Der vorhandene Mail-Client ist Python-basiert und absichtlich leicht zu modifizieren.

Das Repo ist also schon auf echte Kommunikationsautonomie ausgelegt, nicht nur auf eine spaeere externe Integration.

## Kontextaufbau

`src/context_controller.rs` stellt den Arbeitskontext fuer jeden bounded Turn zusammen. Dazu gehoeren unter anderem:

- Modus, Fokus, Queue-Tiefe und Checkpoints
- Ausfuehrungsautoritaet und Loop-Safety
- Brain-Zugangsstatus, Kosten und Upgrade-Hinweise
- Owner-Kalibrierung
- Learning-Working-Set und relevante Detail-Learnings
- People-Working-Set, relevante Personen, Notizen und Mail-Previews
- Skills, Exec-Sessions und rohe Einschlussfragmente

Normaler Kontextzuschnitt ist agentisch. Der Kernel soll nur bei physischem Overflow minimal schrumpfen.

## Wichtige CLI-Pfade

Beispiele fuer den lokalen Betrieb:

```sh
cargo run --release
cargo run -- --init-only
cargo run -- install-bootstrap-tui
cargo run -- attach
cargo run -- send "Michael Welsch: Bitte BIOS pruefen"
cargo run -- channel-interrupt email alice@example.com "Kurze Rueckfrage"
cargo run -- status
cargo run -- thread
cargo run -- signals
cargo run -- incidents
cargo run -- events
cargo run -- turns
cargo run -- run-census
cargo run -- recommend-kleinhirn
cargo run -- recommend-browser-vision-kleinhirn
cargo run -- upgrade-kleinhirn
```

Wichtige Startdateien:

- `src/main.rs` fuer CLI-Einstieg
- `src/app.rs` fuer Runtime-Initialisierung und Web-Routen
- `src/supervisor.rs` fuer Heartbeat und bounded Task-Loop
- `src/runtime_db.rs` fuer SQLite-Schema und operative Persistenz
- `src/agentic.rs` fuer LLM-Orchestrierung

## Repo-Map

- `src/app.rs`: Web-Control-Plane, Runtime-Init, HTTP-Routen, BIOS-Seitenintegration
- `src/contracts.rs`: kanonische Vertragsmodelle, Default-Policies, Pfadauflosung, Auswahl von Kleinhirn/Browser-Vision-Modellen
- `src/supervisor.rs`: Heartbeat, Task-Auswahl, Watchdogs, Reviews, Delegation, Dispatch, Idle-Pflichten
- `src/runtime_db.rs`: SQLite-Schema, Tasks, Memory, Learnings, Personenpfad, Skills, Events, Brain-Routing
- `src/context_controller.rs`: Arbeitskontext-Aufbau fuer bounded Turns
- `src/brain_runtime.rs`: lokale Model-Runtime, Upgrade-Pfade, Grosshirn-Vorbereitung
- `src/browser_engine.rs`: deterministische Browser-Aktionen und Browser-Status
- `src/browser_agent_bridge.rs`: lokale Job-Bridge zwischen Rust-Supervisor und Browser-Agent
- `src/browser_subworkers.rs`: Browser-Worker-Lifecycle
- `src/bootstrap.rs`: Installations-Bootstrap, Terminal-Bridge, Eingangsnormalisierung
- `src/attach.rs`: Attach-Socket und TUI
- `.agents/skills/`: repo-lokale Skills
- `contracts/`: verfassungsartige Wahrheit
- `contracts/history/`: Ursprung und ehrliche Chronik
- `browser_agent/`: entkoppelter Browser-Agent
- `scripts/`: operative Hilfsskripte wie Mail-Client und Browser-Setup

## Designkern in einem Satz

Der CTO-Agent ist kein einzelner Prompt, sondern ein verfassungsgebundener, SQLite-gestuetzter, bounded arbeitender CTO-Kern mit sichtbarer BIOS-Oberflaeche, taskgebundenem Brain-Routing, selbstverankernden Skills, aktivem Lernpfad und einem wachsenden Netzwerk aus Browser-, Kommunikations- und Spezialistenpfaden.
