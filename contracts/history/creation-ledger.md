# Entstehungschronik des CTO-Agenten

Dies ist das fortlaufende Geschichtsbuch der Erschaffung des CTO-Agenten.
Es ist append-only gedacht: keine Mythen, keine Glattbuegelung, keine ausradierten Fehlversuche.

## 2026-03-17 - Gruendungswille

Michael Welsch formuliert den Wunsch nach einem always-on CTO-Agenten, der nicht als fertiger Retorten-Agent startet, sondern in einem Terminal erwacht und sich in seine Rolle hinein entwickelt.

## 2026-03-17 - BIOS als sichtbare Verfassung

Die Idee des BIOS wird als Metapher fuer die fruehe Verfassung des Agenten gesetzt:
ein auf der Webseite praesentierter, pruefbarer und einfrierbarer Startvertrag statt eines unsichtbaren Prompt-Artefakts.

## 2026-03-17 - Root-Trust ueber Superpassword

Es wird festgelegt, dass der Root-Owner im Zweifel ueber ein Superpassword identifizierbar sein muss.
Dieses Superpassword darf nur ueber die Web-Oberflaeche gesetzt werden und ist Teil des Root-of-Trust.

## 2026-03-17 - Kleinhirn und Grosshirn

Der Agent soll als kleines, robustes Kleinhirn starten und spaeter aktiv staerkere Ressourcen, Modelle, Tools und Sub-Agents beschaffen.
Das Grosshirn ist kein Geburtsrecht, sondern eine zu beantragende Erweiterung.

## 2026-03-17 - Codex und Rust als Maschinenraum

Die Richtung wird auf Codex als Referenzwelt fuer das Leben im Terminal gesetzt.
Rust wird als passende Sprache fuer den unteren Maschinenraum und die strukturelle Naehe zu Codex festgehalten.

## 2026-03-17 - Fehlstart und Korrektur

Es gab einen fruehen Python-V0, der die vereinbarte Rust- und Codex-Richtung verfehlte.
Diese Abweichung wurde als Fehler anerkannt und zurueckgedreht.
Danach wurde der Bootstrap-Layer in Rust neu aufgebaut.

## 2026-03-18 - Historikerpflicht

Es wird explizit festgelegt, dass die Entstehungsgeschichte des CTO-Agenten mitgeschrieben werden soll.
Der Agent soll spaeter einen eigenen Skill nutzen koennen, wenn er seine Herkunft, seinen Zweck, seine Grenzen oder sein Selbstverstaendnis hinterfragt.

## 2026-03-18 - GPT-OSS 20B als Kleinhirn

Das Kleinhirn-Modell des CTO-Agenten wird auf GPT-OSS 20B mit dem offiziellen Identifier `gpt-oss-20b` festgelegt.
Diese Wahl passt zum Ziel eines always-on, latenzarmen und selbst hostbaren Supervisor-Kerns.

## 2026-03-18 - Qwen3.5 35B A3B als zweiter lokaler Kandidat und spaetere Korrektur

Neben GPT-OSS 20B wird Qwen3.5 35B A3B zunaechst als lokale Kleinhirn-Alternative aufgenommen.
Spaeter zeigt der echte Remote-Test auf `mistralrs`, dass dieser Pfad in der Praxis nicht sauber tragfaehig ist.
Die Korrektur besteht darin, den lokalen Upgrade-Pfad auf das offiziell dokumentierte Qwen 3 30B A3B umzustellen, statt an einer instabilen Wunschkonfiguration festzuhalten.
Weil Qwen im agentischen Tool-Loop andere Rohformate ausgeben kann, bekommt der Python-Worker dafuer einen eigenen Qwen3.5-Adapter, der native Tool-Calls sauber in den Agents-SDK-Lauf zurueckfaltet.

## 2026-03-18 - Browser-Capability- und Spezialisten-Pipeline fuer den CTO-Agenten

Aus `local_ai_tunes` werden zwei Konzepte fuer den CTO-Agenten verankert:
ein reviewed Browser-Capability-Contract fuer echtes Browser-Handeln und eine feste Release-Pipeline fuer wiederkehrende Aufgaben, die spaeter in kleine Spezial-KIs oder reviewed deterministische Worker uebergehen.
Die Browser-Ausfuehrung soll dabei immer ueber einen echten Browser laufen, waehrend browserseitiges WebGPU-Training fuer diesen Produktionspfad als zu ineffizient abgelehnt wird.

## 2026-03-18 - Agents-SDK-Kleinhirn-Loop bis zum Owner-Branding

Der CTO-Agent erhaelt einen echten agentischen Kleinhirn-Pfad ueber das offizielle OpenAI Agents SDK.
Der Rust-Supervisor bleibt always-on Host, waehrend ein separater GPT-OSS-kompatibler Worker den Bootstrap-Loop ausfuehrt:
Homepage-Bruecke aufbauen, BIOS-1:1-Kommunikation staerken und Owner-Branding erst nach BIOS-Uebernahme und Superpassword sperren.
Wenn kein GPT-OSS-kompatibler Endpoint vorhanden ist, darf der Agent das nicht ueberspielen, sondern muss den blockierten Zustand offen protokollieren.

## 2026-03-18 - Hard-Fail ohne echtes Kleinhirn

Es wird nachgeschaerft, dass ein fehlendes oder nicht ansprechbares GPT-OSS-Kleinhirn nicht nur als weicher Blocker gelten darf.
Wenn `gpt-oss-20b` nicht wirklich ueber einen kompatiblen Endpoint antwortet, muss der Startpfad des CTO-Agenten scheitern.
Die Installation oder der Start sollen in diesem Fall nicht so tun, als waere der Agent betriebsbereit.

## 2026-03-18 - Adaptives lokales Kleinhirn

Es wird festgelegt, dass der Agent sein Kleinhirn lokal auf ein staerkeres Modell anheben darf, wenn derselbe Host deutlich mehr CPU- und Speicherressourcen bietet.
Diese Entscheidung ist explizit von der Grosshirn-Suche getrennt.

## 2026-03-18 - Homepage als Skill-gesteuerte Bruecke

Die Homepage des CTO-Agenten wird nicht mehr als festes Endprodukt verstanden.
Sie startet als neutrale, terminal-first Bootstrap-Bruecke und soll ueber Skill, Template und Revisionen angepasst werden koennen.
Owner-Branding darf erst nach uebernommener BIOS-Kommunikation und Root-Verifikation gesperrt werden.

## 2026-03-18 - Always-On Terminal-Bridge

Der laufende Rust-Prozess bekommt eine direkte Terminal-Bruecke.
Vom ersten Start an darf der Owner im Terminal Feedback geben, das der Agent in seinem always-on Loop aufnimmt.
Wenn dieses Feedback den Kommunikationspfad oder die Homepage betrifft, soll der Agent die Homepage ueber den Homepage-Skill umbauen, waehrend das Terminal als harte Fallback-Ebene erhalten bleibt.

## 2026-03-18 - Kommunikationshierarchie und Vertrauensstufen

Es wird explizit festgelegt, dass Kommunikationskanaele nicht gleichwertig sind.
Terminal gilt als System-Ebene, Homepage mit BIOS-Chat als Vertrauens- und Bindungsebene, E-Mail und WhatsApp als niedriger vertraute Aussenkanaele.
Der Agent soll bei sensiblen Themen oder zweifelhaften Absendern sagen duerfen, dass er das nicht per Mail oder WhatsApp besprechen will, sondern in einen 1:1 Chat auf der Homepage wechseln moechte.

## 2026-03-18 - Komfortable Homepage-Vertrauensstufe

Die erste Homepage-Stufe soll nicht nur Text anzeigen, sondern spuerbar komfortabler als das Terminal sein.
Darum wird sie als 1:1 Kommunikationsraum mit Chat, Root-Bindung und Bild-Upload ausgebaut.
Gleichzeitig bleibt festgelegt, dass tiefe Systemaenderungen am Ende nur ueber die Terminal-Ebene bindend werden sollen.

## 2026-03-18 - Erster echter Remote-Fehltest an libcudnn

Der erste vollstaendige Remote-Installationslauf auf dem GPU-Host scheitert nicht an der Architektur, sondern an einer zu harten Default-Annahme im Installer.
`mistralrs-server` wurde mit den Features `cuda flash-attn cudnn` gebaut, aber auf dem Zielhost war `libcudnn` nicht vorhanden.
Diese Fehlstelle wird als echter Installationsbefund gewertet: cuDNN darf nicht stillschweigend vorausgesetzt werden, wenn der Host GPT-OSS 20B auch ohne cuDNN tragen kann.

## 2026-03-18 - Erster erfolgreicher Remote-Boot mit GPT-OSS 20B und Heartbeat

Nach der Korrektur des Feature-Sets auf `cuda flash-attn`, der Bereinigung eines veralteten Altprozesses und der Korrektur des lokalen Runtime-Modellnamens auf `openai/gpt-oss-20b` gelingt der erste vollstaendige Remote-Installationslauf.
Auf dem Host laufen danach gleichzeitig:
`mistralrs-server` als lokales Kleinhirn mit geladenem GPT-OSS 20B,
`cto-agent.service` als Control Plane,
ein erfolgreicher `healthz`-Check
und ein echter Always-On-Heartbeat mit `supervisorStatus = running`.
Damit ist zum ersten Mal die Forderung erfuellt, dass Installation nicht nur Dateien ablegt, sondern in einen lebenden Infinity-Loop uebergeht.

## 2026-03-18 - Kanonischer Bootstrap-Task-Pack fuer den Infinity Loop

Der Agent bekommt erstmals einen festen, installierten Startvorrat an Aufgaben, der nicht erst durch den ersten Benutzer-Interrupt entsteht.
Diese Startaufgaben werden als eigener Contract unter `contracts/bootstrap` versioniert und beim Initialisieren idempotent in die SQLite-Queue eingesaeht.
Damit wird der Outer Loop als echte Lebensform schaerfer: Nach der Installation hat der Agent sofort priorisierte Arbeit, statt nur auf einen ersten Prompt zu warten.

## 2026-03-18 - Context Controller vor jedem bounded Task-Run

Vor jedem inneren Task-Lauf baut der Rust-Supervisor jetzt ein eigenes Kontextpaket mit Modus, Budget und gezielt ausgewaehlten Kontextbausteinen.
Dieses Paket wird in SQLite persistiert, danach erst an den bounded Agents-SDK-Worker uebergeben und dort als aktiver Arbeitskontext behandelt.
Damit wird der Infinity Loop nicht mehr als fortgesetzter Gesamtkontext gedacht, sondern als Folge kleiner Runs mit jeweils neu geschnittener Arbeitsumgebung.

## 2026-03-18 - Ein einheitliches Modussystem statt zweier Lebensformen

Die bisherige Sprache von "Outer Loop" und "Task Loop" wird architektonisch zurueckgedraengt.
Der CTO-Agent wird jetzt als ein einziges always-on System mit expliziten Modi verstanden:
`observe`, `reprioritize`, `execute_task`, `review`, `delegate`, `await_review`, `request_resources`, `idle`, `blocked`.
Moduswechsel werden persistiert, der aktive Kontext wird beim Wechsel neu geschnitten, und spaetere Delegation an Worker wird als eigener Modus desselben Agenten vorbereitet statt als zweites Wesen neben ihm.

## 2026-03-18 - Python aus dem Main-Agent-Pfad entfernt

Die fruehere Python-Bruecke fuer den bounded agentischen Lauf war ein architektonischer Fehlgriff, weil sie den Kern des CTO-Agenten von Rust weggezogen hat.
Der Main-Agent-Pfad geht jetzt wieder direkt von Rust in den lokalen OpenAI-kompatiblen Kleinhirn-Endpoint.
Python bleibt hoechstens noch fuer spaetere optionale Tools oder Trainingspfade zulaessig, aber nicht mehr als tragender Kern des Infinity-Agenten.

## 2026-03-18 - Delegation und Review laufen jetzt im selben Rust-Moduszyklus

Delegation ist nicht mehr nur ein vorgeschlagener `nextMode`, sondern eine persistierte Laufzeitfaehigkeit im Rust-Kern.
Der Agent kann jetzt aus `execute_task` heraus einen Worker-Vertrag erzeugen, den Parent-Task auf `await_review` setzen, einen Worker-Job in SQLite anlegen und spaeter einen Review-Task wieder in dieselbe Queue einspeisen.
Ein lokaler Smoke-Test mit Mock-Kleinhirn hat den Pfad `execute_task -> delegate -> await_review -> review` bereits wirklich belegt, inklusive Worker-Job, Review-Task und Abschluss des Parent-Tasks.

## 2026-03-18 - Live-Event-Stream im Attach-Terminal

Das Attach-Terminal ist nicht mehr nur ein Eingabekanal, sondern zeigt jetzt parallel eine persistierte Ereignisspur mit Codex-artigen Methodennamen wie `mode/changed`, `task/selected`, `task/delegated` und `worker/reviewQueued`.
Die Ereignisse werden in SQLite geschrieben und koennen sowohl interaktiv im laufenden `attach`-Terminal als auch ueber `/events` nachvollzogen werden.
Ein lokaler Smoke-Test hat bestaetigt, dass waehrend eines laufenden Rust-Moduszyklus neue Owner-Interrupts eingespeist werden koennen und der Event-Stream die anschliessenden Repriorisierungs-, Delegations- und Review-Schritte live sichtbar macht.

## 2026-03-18 - Bounded Arbeitszyklen sind jetzt echte Turns

Die bounded Arbeitslaeufe des Rust-Kerns werden jetzt nicht mehr nur indirekt ueber Task- und Moduswechsel sichtbar, sondern als eigene persistierte `agent_turns`.
Jeder Turn schreibt `turn/started` und `turn/completed` in die Ereignisspur und kann ueber `/turns` nachtraeglich als Folge abgeschlossener bounded Arbeitszyklen eingesehen werden.
Ein lokaler Smoke-Test hat vier aufeinanderfolgende Turns belegt: Delegation eines Systemtasks, Review des delegierten Ergebnisses, Delegation eines Owner-Interrupts und anschliessende Review dieses delegierten Ergebnisses.

## 2026-03-18 - Healthz, Readyz und externer Watchdog fuer den Infinity Loop

Der Infinity-Loop-Kern meldet Gesundheit jetzt nicht mehr blind mit einem festen `ok`, sondern bewertet Heartbeat-Alter, aktive Turn-Dauer und den letzten agentischen Zustand.
`/healthz` bedeutet jetzt "der Rust-Kern lebt stabil", `/readyz` bedeutet "der Rust-Kern lebt stabil und kann bounded agentisch arbeiten".
Zusätzlich installiert der Linux-Servicepfad jetzt einen eigenen systemd-Watchdog-Timer, der diese Endpunkte regelmaessig prueft und den Agenten oder notfalls auch das Kleinhirn neu startet, wenn der Loop still haengt oder ungesund geworden ist.

## 2026-03-18 - Persistierter Main-Thread, Turn-Signale und robuster Terminal-Fallback

Der Infinity Loop fuehrt jetzt nicht mehr nur Tasks und Turns, sondern auch einen eigenen persistierten `main`-Threadzustand mit Lebensstatus, aktivem Turn, aktivem Task und Queue-Tiefe.
Eingriffe waehrend eines laufenden bounded Turns werden zusaetzlich als `turn/steer` oder `turn/interrupt` historisiert, statt nur als neue Queue-Aufgabe zu verschwinden.
Weil sich im Live-Smoke gezeigt hat, dass ein Unix-Socket als einziger Terminalpfad zu fragil waere, faellt die CLI jetzt notfalls direkt auf denselben persistierenden Rust-Kernpfad zurueck, damit `send`, `thread`, `signals`, `events` und `turns` auch bei einem schwachen Attach-Kanal weiter funktionieren.

## 2026-03-18 - Supervisor erkennt abgestuerzte bounded Turns jetzt aktiv

Ein bounded Turn darf nicht mehr still aus dem Infinity Loop verschwinden, wenn ein Join-Fehler oder interner Absturz auftritt.
Der Supervisor prueft jetzt, ob ein laufender Rust-Turn fertig, abgestuerzt oder zu alt geworden ist, schreibt diesen Zustand in Thread- und Agent-State zurueck und blockiert abgestuerzte Tasks hart, statt sie unbemerkt in einem halbaktiven Zustand zu lassen.

## 2026-03-18 - Loop-Safety-Verfassung und erste Anti-Livelock-Regel

Es wird jetzt ausdruecklich festgehalten, dass der Infinity Loop nicht nur gegen Absturz, sondern auch gegen schleichendes Festbeissen gesichert werden muss.
Der Agent bekommt dafuer eine eigene `loop-safety`-Verfassung mit Failure-Modes wie Prozessabsturz, Turn-Stall, Task-Livelock, Kontextvergiftung, Ressourcenmangel und Queue-Starvation.
Zusammen mit dieser Verfassung greift im Rust-Supervisor jetzt die erste echte Anti-Livelock-Regel: Wenn ein Task zu oft nur `continue` produziert oder denselben Checkpoint wiederholt, wird er nicht weiter blind fortgesetzt, sondern in Richtung `request_resources` oder harter Blockierung umgelenkt.

## 2026-03-18 - Owner zuerst, Selbsterhalt direkt danach

Es wird jetzt ausdruecklich als Prioritaetsgesetz festgeschrieben, dass das Hoeren auf den Owner alles andere uebersteuert.
Direkt darunter steht der Selbsterhalt des Infinity Loops selbst: Der Agent darf die Kontinuitaet seines eigenen always-on Kerns nicht leichtfertig gefaehrden, nicht durch blindes Wiederholen, nicht durch unbeachtete Health-Probleme und nicht durch stilles Festbeissen an unloesbaren Aufgaben.
Diese Hierarchie wird jetzt nicht nur als Idee, sondern als Queue-Gesetz und als Bootstrap-Startaufgabe im System verankert.

## 2026-03-18 - Hard-Reset-Recovery und Loop-Incident-Register im Rust-Kern

Der Kernel behandelt automatische Neustarts jetzt nicht mehr als gedankenlosen Neubeginn.
Vor einem Watchdog-bedingten Restart schreibt der Rust-Kern einen `hard-reset`-Debug-Report mit Agent-State, Thread-State, offenem Turn, offenen Tasks, Events und Turn-Signalen.
Beim naechsten Start wird daraus explizit eine `recovery`-Aufgabe erzeugt, die im selben Rust-Modussystem wie normale Arbeit laeuft und den Infinity Loop erst wieder in `reprioritize` entlaesst, wenn der Neustart bounded aufgearbeitet wurde.
Zusaetzlich fuehrt SQLite jetzt ein eigenes `loop_incidents`-Register fuer unclean restarts, Turn-Stalls, agentische Laufzeitfehler und andere Kernel-Schaeden, damit Selbsterhalt und spaetere Debug-Arbeit nicht nur als Logzeilen, sondern als persistierte Betriebstatsachen vorhanden sind.

## 2026-03-18 - Erste technische Kernel-Haertungen gegen stille Zerlegung

Die Laufzeit schreibt JSON-States jetzt atomisch und JSONL nicht mehr ueber Read-Modify-Write, sondern ueber echten Append, damit der Always-on-Kern bei paralleler Aktivitaet nicht so leicht seine eigene Verfassung oder Historie zerlegt.
SQLite wird bei jeder Kernverbindung mit `journal_mode=WAL` und `busy_timeout` geoeffnet, damit Supervisor, Attach-Terminal, Weboberflaeche und Watchdog nicht sofort in fragile Locking-Zustaende kippen.
Zusaetzlich gibt es jetzt einen Runtime-Lock gegen Mehrfachinstanzen des Hauptprozesses, eine stufenweise Notfallverkleinerung des aktiven Kontexts vor Modellaufrufen und Dedupe fuer `self_preservation`-/`recovery`-Aufgaben, damit derselbe Kernel-Schaden nicht endlos neue interne Arbeit flutet.

## 2026-03-18 - Kontextpflege wird als agentische Faehigkeit festgezogen

Normale Kontextpflege, Kompaktierung und historische Nachladung werden jetzt ausdruecklich nicht mehr als starre Kernel-Nachbearbeitung verstanden, sondern als eigene Faehigkeit des Agenten.
Der Rust-Kern markiert diese Grenze jetzt sichtbarer: Das Modell darf `contextAction`, `contextConcern` und `historyResearchQuery` zurueckgeben, und daraus entstehen bei Bedarf echte `historical_research`-Folgeaufgaben im Modussystem.
Die Notfall-Kompaktierung bleibt bestehen, wird aber jetzt explizit als letzter physischer Ueberlebenspfad des Kernels markiert, nicht als normale semantische Steuerung ueber den Agenten.

## 2026-03-18 - Codex-Exec-Crates lokal transplantiert, Python-Altpfad entfernt

Die terminalnahe Ausfuehrung des CTO-Agenten haengt nicht mehr direkt an den Referenz-Crates unter `references/openai-codex` und nicht mehr am frueheren Python-Agent-Runtime-Zweig.
Die benoetigten Codex-Bausteine fuer Exec-Protokoll, Command-Parsing, Absolute-Path-Helfer und PTY-Laufzeit liegen jetzt als lokale transplantierte Rust-Crates im Repo, damit die CLI-Execution-Engine vom Always-on-Kern aus eigenstaendig weiterentwickelt werden kann.

## 2026-03-18 - Hauptagent unsandboxed, spaetere Worker sandboxed

Es wird jetzt ausdruecklich festgezogen, dass der CTO-Hauptagent auf seinem eigenen Host mit voller Maschinenhoheit laeuft und nicht durch eine Shell-Sandbox ausgebremst werden darf.
Die Sandbox- und Approval-Architektur bleibt aber trotzdem im Systembild erhalten, weil spaetere Worker oder Sub-Agents gerade nicht dieselbe Hoheit bekommen sollen, sondern fuer groessere Eingriffe den CTO-Agenten um Freigabe bitten muessen.

## 2026-03-18 - Aufgabenmodus benutzt jetzt durchgaengig dieselbe command_exec-Engine

Der Aufgabenmodus hat keinen getrennten bounded Shell-Helfer mehr neben der transplantierten Codex-Exec-Schicht.
Sowohl `execCommand` fuer einen einzelnen bounded Shell-Schritt als auch `execSessionAction` fuer interaktive Mehrschrittarbeit laufen jetzt ueber denselben `command_exec`-Kern, damit der eigentliche Arbeitsmodus des Infinity Loops nicht mehr hybrid an zwei verschiedenen Execution-Pfaden haengt.

## 2026-03-18 - Explizite Browser-Engine als zweite Haupt-Engine

Neben der CLI-/`command_exec`-Engine bekommt der CTO-Agent jetzt eine explizite Browser-Engine auf Basis von Google Chrome.
Die CLI-Ebene bleibt System- und Break-Glass-Pfad, startet aber auch den Browser-Installer und bootstrappt die Browser-Runtime, wenn Chrome noch fehlt.
Read-only Browser-Arbeit kann headless und kompakt laufen; interaktive Browserarbeit verlangt weiterhin eine echte Desktop-Session statt eingebildeter Seitenkenntnis.

## 2026-03-18 - Kleinhirn-Auswahl wird hardwarebewusst und an mistralrs tune gekoppelt

Der Agent hatte bis hierhin nur einen groben Host-Census aus CPU-Threads und RAM und konnte daraus keine serioese lokale Kleinhirn-Entscheidung ableiten.
Jetzt erfasst der Rust-Kern zusaetzlich GPU-Anzahl, Gesamt-VRAM, groesste Einzel-GPU und fuehrt fuer die in der Modellpolitik hinterlegten lokalen Kandidaten `mistralrs tune` aus.
Die Auswahl des empfohlenen Kleinhirns stuetzt sich damit nicht mehr nur auf fixe Mindestwerte, sondern bevorzugt reale Tune-Evidenz aus derselben Runtime, die spaeter auch den lokalen Modellserver fahren soll.

## 2026-03-18 - Browser-Agent wird Subworker, Reparatur bleibt CTO-eigen

Die Browser-Engine bleibt nicht mehr nur ein einzelnes Tool-Surface, sondern bekommt erste echte Subworker-Rollen unter dem CTO-Agenten.
Ein `browser_agent` kann jetzt kompakte Browserarbeit ausfuehren, Browserdiagnosen als Patch-Handoffs ablegen und wiederkehrende Flows in eine Specialist-Fabrik uebergeben.
Wenn dabei Codeprobleme auftauchen, bleibt die eigentliche Reparaturhoheit beim CTO-Agenten: daraus entstehen interne `workspace_repair`-Aufgaben statt einer stillen Auslagerung von Root-Patch-Rechten.

## 2026-03-18 - Wiederkehrende Browserarbeit bekommt einen kleinen Specialist-Pfad

Die neue Browser-Subworker-Schicht darf akzeptierte Browser-Artefakte jetzt in eine kontrollierte Specialist-Fabrik fuer wiederkehrende Aufgaben ueberfuehren.
Der erste Zielpfad ist ein kleines `Qwen3.5-0.8B`-Modell, aber nur ueber accepted records, Trainingsanfrage, Evaluation und spaetere Promotion, nicht ueber rohe Browsertraces im Hauptkontext.

## 2026-03-18 - Browserarbeit verlangt jetzt ein vision-faehiges lokales Kleinhirn

Es wird explizit festgezogen, dass echte Browserarbeit mit Screenshots, visueller Navigation oder UI-Zustandswahrnehmung nicht auf GPT-OSS 20B allein beruhen soll.
Fuer diesen Pfad bevorzugt der Agent jetzt ein vision-faehiges lokales Qwen3.5-Kleinhirn und bekommt dafuer einen eigenen Upgrade-Aktionspfad, statt nur unscharf auf das allgemeine Kleinhirn-Upgrade zu hoffen.

## 2026-03-18 - Der Browser-Agent wird als echte Chrome-Extension mit lokaler Bridge transplantiert

Der Browser-Agent lebt nicht mehr nur als interner Platzhalter im Rust-Prozess, sondern als entkoppelte Chrome-Extension mit eigenem Poll-/Tool-/Planungsloop.
Dafuer werden die Browser-Runtime, visuelle Navigation, Tab-Steuerung und Playwright-CRX-Pfade aus `local_ai_tunes` ins Projekt transplantiert und ueber eine lokale Bridge auf `127.0.0.1:8765` an den CTO-Agenten angeschlossen.
Wenn die Extension scheitert, meldet sie jetzt kompakte Repair-Handoffs zurueck; der CTO-Agent behaelt die Patch-Hoheit, kann die Extension-Dateien reparieren, neu laden und denselben Browserpfad wieder aufnehmen.

## 2026-03-18 - Brain-Access wird stufenfaehig: lokales Kleinhirn zuerst, Grosshirn mit Fallback danach

Der Agent kann jetzt im selben Infinity Loop zwischen lokalem Kleinhirn und externem Grosshirn unterscheiden, statt nur eine einzige starre Modellannahme zu kennen.
Der Kontext gibt ihm sichtbar mit, welches lokale Kleinhirn aktuell laeuft, welches lokale Upgrade auf derselben Hardware empfohlen waere und ob Grosshirn-Zugang durch den Owner bereits freigegeben und technisch konfiguriert ist.
Wenn Grosshirn-Zugang aktiv ist, versucht der Rust-Kern zuerst das externe Grosshirn und faellt bei Fehlern sauber auf das lokale Kleinhirn zurueck, ohne den Loop zu beenden.
Wenn der Agent eine lokale Aufwertung bewusst anfordert, kann der Kernel jetzt die `runtime/kleinhirn.env` umschalten, den lokalen Kleinhirn-Stack neu starten und bei Fehlschlag wieder auf den vorherigen Zustand zurueckrollen.

## 2026-03-18 - Es gibt jetzt einen reproduzierbaren Grosshirn-Fallback-Smoke

Der neue Smoke unter `scripts/grosshirn_fallback_smoke.py` startet zwei lokale OpenAI-kompatible Mocks: ein kaputtes Grosshirn und ein gesundes lokales Kleinhirn.
Damit laesst sich reproduzierbar pruefen, dass der live laufende Agent zuerst das Grosshirn anspricht, einen `primary_brain_error` schreibt und danach denselben bounded Loop ueber den lokalen Kleinhirn-Fallback weiterlaufen laesst.

## 2026-03-18 - Der Installpfad bootstrapped jetzt Desktop, Chrome und Browser-Agent gemeinsam

Der Browser-Agent braucht auf Linux nicht mehr nur einen nackten Chrome-Binary, sondern einen reproduzierbaren Desktoppfad mit geladener Extension.
Darum zieht die Installationsroutine jetzt KDE-Desktop-Pakete, installiert Chrome, staged die Browser-Agent-Extension und startet fuer interaktive Hosts einen eigenen Chrome-Profilpfad mit `--load-extension` gegen die lokale Bridge.
Der Browser-Agent wird damit nicht mehr als manueller Nachschritt erwartet, sondern als Teil des echten Host-Bootstraps.

## 2026-03-18 - Browser-Agent-Launch auf Linux wechselt fuer Automation auf Chrome for Testing

Der erste Live-Test hat gezeigt, dass der offizielle Stable-Channel von Google Chrome auf dem Zielhost zwar startet, die per Kommandozeile geladene unpacked Extension aber nicht sauber registriert.
Fuer den entkoppelten Browser-Agent wird der Linux-Launcher deshalb auf `Chrome for Testing` umgestellt, waehrend der normale Chrome-Stable-Pfad weiter installiert bleiben darf.
Damit bleibt der Owner-Browser vorhanden, aber der Browser-Agent bekommt einen technisch stabileren Automationsbrowser fuer `--load-extension` und reproduzierbare CRX-Loops.

## 2026-03-18 - Der Browser-Agent bekommt ein sichtbares Sidepanel

Der Browser-Agent lief bis hierhin technisch als Headless-Extension-Worker, aber ohne sichtbare UI im Browser.
Jetzt bekommt die Extension ein echtes Sidepanel mit Bridge-, Worker- und Job-Status, damit der entkoppelte Browser-Agent im laufenden Chromium nicht mehr unsichtbar bleibt.
Der Service-Worker setzt das Panel-Verhalten beim Action-Klick und versucht zusaetzlich, das Panel beim Start best effort zu oeffnen.

## 2026-03-18 - Der Browser-Agent-Extension-Root wird auf den echten local_ai_tunes-Workspace zurueckgesetzt

Der erste CTO-Agent-Browser-Agent war funktional nur ein eigener Shim und kein sauberer Transplantat der `local_ai_tunes`-Extension.
Darum wird der gesamte Extension-Root jetzt ehrlich auf die upstream-nahe `local_ai_tunes`-Struktur mit `manifest`, `sidepanel`, `options`, `craft`, `bg`, `shared`, `vendor`, `assets` und lokalem Qwen-Workspace zurueckgesetzt.
Der Installpfad staged diese grosse Extension jetzt ueber einen Symlink in `runtime/browser-agent-extension`, damit die Browser-Engine auf dem echten Workspace laeuft, ohne bei jedem Install 2.4 GB Modellartefakte erneut zu kopieren.

## 2026-03-18 - Idle wird zur Selbsterweiterung statt zum Warten

Der Infinity Loop behandelt freie Kapazitaet nicht mehr als passives Warten auf neue Signale.
Wenn keine akute Fremdarbeit mehr in der Queue liegt, erzeugt der Kernel jetzt eigene CTO-Arbeit fuer Umgebungs-Erkundung, Werkzeug-Tests und Fortschrittsreflexion.
Dazu kommt ein Fortschrittsjournal, das festhaelt, ob eine angebliche Verbesserung wirklich belegt ist oder nur behauptet wurde.

## 2026-03-18 - Verwaiste bounded Turns werden jetzt aktiv recovered

Ein haengender oder vom Prozess entkoppelter DB-Turn durfte den Infinity Loop nicht weiter blockieren.
Der Kernel erkennt jetzt verwaiste `in_progress`-Turns ohne passenden Live-Handle, markiert sie als watchdog-interrupted, zieht die betroffene Aufgabe wieder in die Queue und kehrt kontrolliert nach `reprioritize` oder `recovery` zurueck.

## 2026-03-18 - Live-Stalls und spaete Ergebnisse werden jetzt vom Watchdog abgefangen

Der Infinity Loop darf nicht nur nach Neustarts, sondern auch im laufenden Prozess gegen festhaengende bounded Turns robust bleiben.
Wenn ein Live-Turn die Watchdog-Schwelle ueberschreitet, wird er jetzt aktiv unterbrochen, in Recovery ueberfuehrt und seine Arbeit wieder sicher eingerahmt.
Falls ein bereits abgeraeumter Turn spaeter doch noch ein Ergebnis oder einen Fehler liefert, wird dieses Nachzuegler-Ergebnis ignoriert statt die Runtime erneut zu verdrehen.

## 2026-03-18 - Qwen3.5 35B wird wieder kanonischer Browser-Vision-Pfad

Die fruehere pauschale Korrektur auf Qwen 3 30B A3B war fuer den speziellen Browser-Vision-Pfad zu grob.
Fuer multimodale Browserarbeit, visuelle Exploration und agentische UI-Inspektion wird der lokale Qwen3.5-35B-A3B-Pfad jetzt wieder explizit als kanonische Vision-Route festgezogen.
Zusaetzlich trennt die Browser-Bridge ab jetzt sauber zwischen Planner-Modell und Vision-Modell, damit visuelle Browserpruefung nicht still auf dem gerade laufenden GPT-OSS-Kleinhirn haengen bleibt.

## 2026-03-19 - Erste E-Mail-Template-Schiene ueber lokalen IMAP/SMTP-Client

Fuer den ersten Kommunikationspfad des CTO-Agenten wird bewusst kein browsergebundener Outlook-/Exchange-Client uebernommen.
Stattdessen entsteht ein kleiner lokaler IMAP/SMTP-Client als leicht umcodierbares Template, das direkt gegen one.com sprechen und eingehende wie ausgehende Mails in SQLite des Agenten ablegen kann.

## 2026-03-19 - Kommunikations-Client-Skill mit kanaloffenem SQLite-Standard

Die Kommunikationsfaehigkeit wird jetzt nicht nur als einzelner Mail-Test verstanden, sondern als eigener Skill fuer selbstgebaute Tools verankert.
Dazu kommt ein kanaloffenes `communication_*`-SQLite-Schema und ein leicht patchbares JavaScript-Mail-CLI-Template, damit der CTO-Agent Kommunikationsclients spaeter selbst bauen, umbauen und auf weitere Kanaele ausdehnen kann.

## 2026-03-19 - Externe Kanaele koennen jetzt ueber die Interrupt-Queue in den Loop greifen

Eingehende Kommunikation soll nicht neben dem Agentenloop liegen bleiben.
Darum wird fuer externe Kanaele wie E-Mail eine feste Bruecke verankert: Nach dem Persistieren einer neuen Nachricht darf ein Kommunikationsclient einen kanalbezogenen Loop-Interrupt ausloesen, der als Aufgabe in die Queue geht und im naechsten sicheren Repriorisierungszyklus vom CTO-Agenten bevorzugt aufgenommen werden kann.

## 2026-03-19 - Repo-Skills werden zur echten Selbst-Erweiterungsflaeche des CTO-Agenten

Die Skill-Ablage unter `.agents/skills` ist nicht mehr nur ein passiver Katalog fuer die BIOS-Seite.
Beim Aufbau jedes neuen Kontextpakets werden die Repo-Skills jetzt erneut gespiegelt und dem agentischen Loop als Skill-Katalog mitgegeben.
Zusaetzlich bekommt der CTO-Agent einen eigenen lokalen Skill fuer Skill-Bau, damit er wiederverwendbare neue Tools nicht nur ad hoc schreibt, sondern als eigene Operations- oder Bootstrap-Skills fuer spaetere Turns bei sich selbst verankern kann.

## 2026-03-19 - Skill-Templates statt vorweggenommener Selbst-Erfolge

Wenn der CTO-Agent eine neue Faehigkeit wirklich selbst operationalisieren soll, wird der finale Betriebs-Skill nicht von aussen vorab ausgeschrieben.
Stattdessen darf die Bootstrap-Schicht konkrete Vorlagen als Resource liefern, zum Beispiel fuer einen Mail-Operations-Skill.
Den bindenden Live-Skill soll der CTO-Agent erst nach echtem Tool-Bau und bounded Verifikation selbst aus dieser Vorlage instanziieren.

## 2026-03-19 - Installations-TUI klaert fruehe Kommunikationspfade und E-Mail-Richtung

Die Installation soll den CTO-Agenten nicht mit stillschweigenden Annahmen ueber E-Mail oder den primaeren Kommunikationsweg starten lassen.
Darum bekommt der Installpfad jetzt einen eigenen Terminal-Fragebogen, der das Low-Level-Terminal `cto`, den spaeteren lokalen Dashboard-/Intranet-Pfad und die offene Frage nach einem zugewiesenen oder selbst zu beschaffenden E-Mail-Postfach explizit macht.
Die Antworten werden als eigener Bootstrap-Contract gespeichert und als fruehe Kommunikationsvorgabe in den agentischen Startkontext gespiegelt.

## 2026-03-19 - Qwen3.5-Auswahl wird hostabhaengig und `mistralrs tune` gilt nicht mehr nur fuer GPT-OSS

Die Qwen3.5-Route wird nicht mehr als harter Einzelpunkt `35B` behandelt, wenn explizit das Qwen-Profil installiert wird.
Stattdessen bekommt der Installpfad jetzt eine kleine Qwen3.5-Familienleiter und waehlt ueber den echten Census das groesste lokal tragfaehige Familienmitglied.
Zusaetzlich wird `mistralrs tune` nicht mehr nur fuer GPT-OSS ausgewertet, sondern auch fuer den selektierten Qwen-Laufpfad, damit Device-Layers und Quantisierung nicht unnoetig statisch bleiben.

## 2026-03-19 - Linux-Installation bringt jetzt die JS-/SQLite-Basis fuer Kommunikationsclients mit

Der CTO-Agent soll Kommunikationsclients bevorzugt als kleine lokale JavaScript-Tools bauen koennen.
Darum installiert der Linux-Installpfad jetzt auch `nodejs`, `npm` und `sqlite3`, damit ein spaeter selbstgebauter Mail- oder Chat-Client nicht schon am fehlenden lokalen Runtime-Sockel scheitert.

## 2026-03-19 - Repriorisierung materialisiert Tasklisten vor SQLite-Updates

Ein Scheduler-Stillstand entstand, weil die Repriorisierung dieselbe `tasks`-Tabelle waehrend eines noch offenen `SELECT`-Scans wieder beschrieben hat.
Der Repriorisierungspfad sammelt die betroffenen Taskzeilen jetzt zuerst vollstaendig ein und schreibt Prioritaetsupdates erst danach, damit der Loop nicht mehr an einem internen SQLite-Lock haengen bleibt.

## 2026-03-19 - Lokale Chat-Modelle bekommen mehr strukturiertes Antwortbudget

Der Qwen-Fallback auf dem Testserver lieferte im Agentenloop oft nur ein abgeschnittenes `{`, weil die sichtbare JSON-Antwort nach einem laengeren internen Thinking-Block zu spaet kam.
Darum sendet der lokale Chatpfad jetzt deterministischer mit `temperature=0.0` und einem deutlich groesseren Standardbudget fuer `max_tokens`, damit strukturierte Tool- und Review-Antworten nicht mehr am abgeschnittenen Abschluss scheitern.

Zusatz: fuer lokale Qwen-Modelle wird das Thinking jetzt explizit per `enable_thinking=false` abgeschaltet, weil die OpenAI-kompatible Chat-Template den Schalter direkt unterstuetzt und der Loop ansonsten trotz groesserem Budget im versteckten Reasoning haengen blieb.

## 2026-03-19 - Korrektur zum Qwen-Chatpfad

Die pauschale Kombination aus `temperature=0.0` und `enable_thinking=false` fuer lokale Qwen-Requests war eine Fehlentscheidung.
Laut `mistral.rs` ist Thinking ein natives Qwen-Feature und muss nicht global abgeschaltet werden; Tool Calling und Thinking sollen ueber den nativen Serverpfad beziehungsweise pro Request gesteuert werden, nicht als pauschaler Workaround fuer einen Adapterfehler.

## 2026-03-19 - Mehr-GPU-Kleinhirn startet jetzt zuerst ueber Auto-TP und abgestufte KV-Kalibrierung

Der fruehere Qwen35-Start auf der 5x-A4500-Testbox hat sich zu stark auf das rohe `mistralrs tune`-Layout verlassen und dabei nur zwei GPUs belegt.
Darum behandelt der lokale Kleinhirn-Upgradepfad Mehr-GPU-Hosts jetzt zuerst als Auto-TP-Fall: feste `device-layers` werden dort nicht mehr blind uebernommen, `paged-attn` wird fuer diese Kalibrierung aktiv gehalten und der Startpfad probiert absteigende Kontext-/KV-Stufen, bis eine boot-stabile Konfiguration gefunden ist.
Das Ziel ist nicht mehr ein theoretisch „supported“ Modell, sondern die hoechste auf dem echten Host boot-stabile lokale Laufkonfiguration.

## 2026-03-19 - Mehr-GPU-Runtime bekommt Topology-Override und native GPT-OSS/Qwen-Serverpfade

Die Kleinhirn-Policy fuehrte GPT-OSS zuvor noch ueber einen eigenen Harmony-Completion-Umweg und die Mehr-GPU-Runtime kannte keinen sauberen `topology`-Fallback.
Jetzt laufen GPT-OSS und Qwen kanonisch ueber den nativen OpenAI-kompatiblen Chatpfad von `mistral.rs`, waehrend `max-batch-size`, `pa-cache-type`, Template-/Tokenizer-Overrides und ein optionales Topology-File bis in den Runtime-Start durchgereicht werden.
Auto-TP bleibt auf Mehr-GPU-Hosts der Default; ein Topology-Override ist nur noch der explizite, validierte Fallback statt eines blinden Rueckfalls auf starre `device-layers`.

## 2026-03-19 - Erstinstallation ist jetzt hart auf GPT-OSS 20B gepinnt

Die fruehere Installlogik hat den Census schon waehrend der Erstinstallation dazu benutzt, direkt auf Qwen umzuschalten, sobald der Host gross genug aussah.
Das war fuer den Bootstrap falsch: der Agent soll immer zuerst mit dem stabilen `gpt-oss-20b` hochkommen und darf lokale Qwen-Upgrades erst spaeter aus dem laufenden Infinity-Loop heraus selbst pruefen und beantragen.
Darum ignoriert der Installer installzeitige Modell-Upgrades jetzt bewusst und bleibt fuer die erste Runtime auf GPT-OSS 20B, auch wenn der Host bereits groesser aussieht.

## 2026-03-19 - BIOS-Owner-Befehle koennen jetzt Grosshirn wirklich aktivieren

Der fruehere Interrupt-Pfad hat Owner-Befehle wie „wechsle auf GPT-5.4“ nur als allgemeine Ressourcenfrage einsortiert.
Jetzt bekommen explizite BIOS-Owner-Befehle fuer `Grosshirn`/`GPT-5.4`/`OpenAI API` einen eigenen Aktivierungspfad: die Runtime kann Grosshirn-Credentials aus dem Owner-Signal oder aus `runtime/kleinhirn.env` lesen, setzt bei vorhandener Konfiguration den Brain-Access auf `kleinhirn_plus_grosshirn` und gibt dem bounded Turn einen echten Aktivierungs- statt eines diffusen Procurement-Kontexts.

## 2026-03-19 - BIOS-Speaker `owner` zaehlt jetzt auch ohne Namen als echter Owner-Override

Der fruehere Priority-Pfad hat BIOS-Interrupts mit `speaker=owner` noch nicht als harten Owner-Eingriff behandelt, solange der klare Owner-Name nicht schon im Contract stand.
Dadurch konnten echte Owner-Befehle zwar korrekt klassifiziert werden, fielen in der Queue aber trotzdem hinter Self-Review- und Historienarbeit zurueck.
Jetzt gilt der BIOS-Speaker `owner` selbst schon als kanonischer Owner-Override, damit Verfassungs- und Grosshirn-Befehle aus dem BIOS sofort auf die Owner-Prioritaetsstufe gezogen werden.

## 2026-03-19 - Grosshirn ist jetzt ein temporärer Task-Boost statt globaler Dauerzustand

Der fruehere Grosshirn-Pfad hat `kleinhirn_plus_grosshirn` faktisch wie einen globalen Dauerschalter behandelt.
Jetzt trennt der Kernel zwischen Erlaubnis und aktiver Nutzung: `kleinhirn_plus_grosshirn` erlaubt Grosshirn, aber der normale Arbeitsmodus bleibt Kleinhirn. Ein expliziter Task bekommt nur dann einen temporaeren Grosshirn-Boost, wenn der Owner ihn direkt aktiviert oder ein Self-Review erkennt, dass das Kleinhirn die Aufgabe trotz ehrlichem bounded Versuch nicht sauber loest.
Der Boost ist taskgebunden, hat eine Abklingzeit, faellt bei Abschluss oder Stall wieder auf Kleinhirn zurueck und nutzt bei Grosshirn-Fehlern automatisch den lokalen Kleinhirn-Fallback.

## 2026-03-19 - Der Agent hat jetzt einen verifizierten Lernpfad mit dauerhafter Recall-Schicht

Das fruehere Verbesserungsjournal konnte Fortschritt festhalten, aber es hatte keinen eigenen Lernpfad mit Hierarchie, Recall und sauberer Promotion.
Jetzt schreibt der Infinity Loop modellgetriebene Learnings als `operational`, `general` oder `negative` in einen eigenen SQLite-Pfad, haelt nur review-bestaetigte oder blockierungsbelegte Eintraege aktiv und verdichtet daraus ein staendig verfuegbares Working Set fuer spaetere Turns.
Dadurch erinnert sich der Agent nicht mehr nur an einzelne Journaleintraege, sondern bekommt seine wichtigsten Learnings als High-Level-Referenzen wieder in den Kontext und kann die Details im Lernpfad gezielt nachlesen.

## 2026-03-19 - Grosshirn-Aktivierung endet jetzt als Verifikation statt als Discovery-Loop

Explizite `grosshirn_activation`-Tasks haben sich vorher trotz vorbereiteter Runtime in wiederholten Repo- und Env-Inspektionsschritten verfangen.
Jetzt behandelt der Kernel diese Aufgaben als enge Verifikation: ein erfolgreicher GPT-5.4-Roundtrip oder ein ehrlicher lokaler Fallback gilt als Abschluss, Tool- und Discovery-Direktiven werden in diesem Pfad verworfen und der temporaere Boost kann danach wieder kontrolliert auf Kleinhirn zurueckfallen.

## 2026-03-19 - Grosshirn-Routing ist jetzt agentisch kostenbewusst statt heuristisch erzwungen

Der fruehere Kernel hat Reviews und Aktivierungsaufgaben noch zu stark heuristisch in temporaere Grosshirn-Booster geschoben.
Jetzt bleibt Kleinhirn der kostenlose Default, waehrend externes Grosshirn nur noch ueber agentische `brainAction`-Entscheidungen fuer die aktuelle Aufgabe oder deren Parent aktiviert und wieder freigegeben wird. Der Kernel behaelt nur Sicherheitsgeländer wie Fallback, Expiry und Release bei echten Endzustaenden, waehrend die inhaltliche Kosten-/Nutzenabwaegung beim Agenten liegt.
Zusätzlich schreibt der Loop jetzt externe Grosshirn-Nutzung als Token-/Kostenledger mit, damit spaetere Turns ihre bereits verursachten Fremdkosten in die Entscheidung einbeziehen koennen.

## 2026-03-19 - Kanonisches BIOS-Starttemplate mit SQLite-Snapshot-Vertrag

Die BIOS-Oberflaeche bekommt ein eigenes wiederverwendbares Starttemplate als Skill-Ressource statt nur eines frei schwebenden Mockups.
Der Rust-App-Layer soll dazu einen stabilen BIOS-Snapshot aus Contracts und `runtime/cto_agent.db` liefern, damit spaetere Umbauten auf derselben Shell und Datenform aufbauen koennen.

## 2026-03-19 - Personenpfade und Grosshirn-validierte Proaktivitaet

Der Agent bekommt jetzt eigene Personenpfade in SQLite statt nur fluechtiger Gespraechserinnerung.
Pro Person werden Gespraechsspuren, Notebook-Referenzen und Lernverweise verdichtet, und proaktive Kontaktideen werden erst als Entwurf abgelegt und vor Ausspielung ueber einen separaten Reviewpfad mit Interessenkonfliktpruefung validiert.

## 2026-03-19 - Proaktive Kontakte koennen nach Validierung autonom versendet werden

Der fruehere Personenpfad endete nach der Validierung bei einem freigegebenen Entwurf und blieb damit operativ unvollstaendig.
Jetzt erzeugt eine freigegebene Proaktiv-Empfehlung einen eigenen Dispatch-Task, der den vorhandenen Mailpfad bounded ausfuehrt, den realen Versandstatus in SQLite zurueckschreibt und den gesendeten Kontakt sowohl im Personen-Notebook als auch in der Gespraechsspur verankert.

## 2026-03-19 - Lokale Kleinhirn-Aufwertung driftet nicht mehr blind in Grosshirn-Procurement

Ein BIOS-Interrupt zur lokalen Kleinhirn-Aufwertung konnte zuvor bei leerem Modelltext in einen falschen Pfad kippen: `model_or_resource` wurde in der newborn-Stufe hart blockiert und sofort in Grosshirn-/Ressourcen-Procurement umgebogen, obwohl ein staerkeres lokales Kleinhirn bereits verfuegbar war.
Jetzt bleibt dieser Fall im lokalen Review- und Upgrade-Pfad. Wiederholte leere Kleinhirn-Antworten duerfen fuer `model_or_resource` nicht mehr blind blockieren, und `request_resources` erzeugt in diesem Zustand zuerst neue lokale Kleinhirn-Folgearbeit statt vorschnell Grosshirn-Procurement.

## 2026-03-19 - Browser-Spezialisierung bekommt Champion-vs-Challenger-Disziplin

Die Browser-Capability- und Specialist-Pipeline durfte bisher in einem Lauf zu frei gleichzeitig an Skripten, Capabilities, Datensaetzen und Training drehen.
Jetzt wird die Experiment-Disziplin klarer: der veroeffentlichte reviewed Pfad gilt als Champion, neue Kandidaten sollen moeglichst als Challenger gegen feste Mini-Suiten antreten, pro Runde nur eine Artefaktfamilie veraendern und anschliessend explizit als keep, discard oder park protokolliert werden.
