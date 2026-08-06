# Playbook: bestehende Software belastbar nach Rust portieren

Dieses Dokument ist eine allgemein nutzbare Arbeitsanweisung für größere
Software-Ports nach Rust. Es abstrahiert die Erkenntnisse aus dem
CLIProxyAPI-Port; projektspezifische Historie bleibt in `PORTING.md`.

Ein Port ist nicht fertig, weil alle Dateien existieren, ein gewichteter
Funktionsumfang 100 % erreicht oder die aktuelle Rust-Suite grün ist. Fertig
ist er erst, wenn der vollständige vereinbarte Umfang gegen eine benannte
Upstream-Version implementiert, semantisch geprüft, sicher in die Zielarchitektur
eingepasst und mit reproduzierbarer Evidenz freigegeben wurde.

## 1. Vor Beginn: Portvertrag statt Übersetzungsauftrag

Vor dem ersten Rust-Body werden fünf Entscheidungen schriftlich fixiert:

1. **Upstream-Identität:** Repository, unveränderlicher Commit, Datum und
   Lizenz. Branch-Namen oder `HEAD` sind keine Baseline.
2. **Scope:** Produktionsdateien, Tests, Abhängigkeitsmanifeste, Build- und
   Release-Dateien, Runtime-Assets, Dokumentation, Beispiele und Plattformcode.
3. **Paritätsbegriff:** byte-identisch, strukturell identisch, beobachtbar
   verhaltensgleich oder bewusst zielsystem-adaptiert.
4. **Zielautorität:** Wer besitzt Netzwerk, Prozesse, Dateisystem, Secrets,
   Persistenz, Zeit, Scheduling und Shutdown im Zielsystem?
5. **Abschlussbeweis:** Welche Inventare, Testmatrizen, Differentials, Gates und
   Receipts müssen vor einer Freigabe vorliegen?

Jede bewusste Abweichung erhält eine benannte Disposition, eine Begründung und
Tests. „Idiomatic Rust“ ist allein keine ausreichende Begründung für anderes
Verhalten.

## 2. Metrik-Ontologie: Prozentwerte dürfen nicht lügen

Große Ports brauchen mehrere getrennte Ledgers. Diese Achsen messen
unterschiedliche Tatsachen und dürfen weder addiert noch still gemittelt werden.

| Metrik | Zähler / Nenner | Beweist | Beweist ausdrücklich nicht |
|---|---|---|---|
| Datei-Inventar | klassifizierte / vollständige Upstream-Dateien | Jede Datei wurde gesehen | Ein Body ist implementiert |
| Mechanische Closure | Dateien ohne Scaffold/Partial / alle Produktions- bzw. Testdateien | Marker und Zuordnung sind geschlossen | Semantische Parität oder grüne Gates |
| Strikte Closure | evidenzgedeckte Dateien / alle Produktions- bzw. Testdateien | Jede Gutschrift ist durch zulässige Evidenz gedeckt | Ein neuer Upstream-Kandidat ist geprüft |
| Semantischer Capability-Ledger | akzeptierte / eingefrorene Capability-Punkte | Vereinbarte Nutzerfähigkeiten funktionieren | Repository-weite Dateiparität |
| Kandidatenreview | vollständig disponierte Delta-Pfade / alle Delta-Pfade | Der konkrete Upstream-Delta wurde einzeln geprüft | Globale Regressionfreiheit |
| Promotion-Gates | attestierte / geforderte Gates | Die vollständige Kandidatenmatrix ist gelaufen | Dass der Pin bereits promoviert ist |
| Baseline-Status | boolesch: akzeptierter Pin geprüft | Aktuelle freigegebene Basis ist belastbar | Kandidat ist freigegeben |
| Promotion-Status | boolesch: Kandidat promoviert | Baseline wurde atomar weitergeschaltet | Zukünftige Upstream-Parität |

### Regeln für jede Fortschrittsanzeige

- Jede Zahl zeigt **Name, Zähler, Nenner, Commit und Evidenzklasse**, zum
  Beispiel: `Kandidatenreview a88197f: 14/111 (12,6 %), Gates 0/10`.
- Produktion und Tests werden getrennt ausgewiesen. Ein gemeinsamer Wert darf
  höchstens zusätzlich und mit expliziter Formel erscheinen.
- `100 %` steht nie ohne Gegenstand. Zulässig ist etwa
  `Accepted-Pin strikte Produktions-Closure: 605/605`; unzulässig ist
  `Port: 100 %`, solange ein Kandidat offen ist.
- Ein gewichteter Capability-Ledger und ein Datei-Ledger bleiben sichtbar
  getrennt. 1.000/1.000 Capability-Punkte können neben hunderten Scaffolds
  stehen.
- Scaffolds, Signaturen, generierte Platzhalter, Dokumentation und ignorierte
  Tests erhalten keine semantischen Punkte.
- Ein Dashboard berechnet keinen synthetischen „Gesamtfortschritt“, wenn es
  dafür keinen vorab eingefrorenen Nenner gibt. Es zeigt stattdessen die
  Achsen und einen konservativen Zustandsnamen wie `in_progress`,
  `ready_for_gates` oder `promoted`.
- Die oberste Statuskarte darf nur dann abgeschlossen anzeigen, wenn alle
  Abschlussbedingungen erfüllt sind. Untergeordnete 100-%-Karten tragen immer
  ihren Scope im Titel.
- Zähler werden aus maschinenlesbaren Inventaren abgeleitet, nicht aus
  Fließtext oder einer Folge kumulativer Worker-Berichte.

### Prozent-Fail-Closed-Regel

Wenn Nenner, Baseline oder Evidenzklasse unklar sind, wird **kein Prozentwert**
angezeigt. Stattdessen erscheint `nicht berechenbar` mit dem fehlenden
Artefakt. Eine scheinbar präzise falsche Zahl ist schädlicher als eine sichtbare
Messlücke.

### Port und Produktintegration sind getrennte Release-Lanes

Ein portables Zielartefakt und seine Einbettung in ein Produkt besitzen
unterschiedliche Änderungsquellen und Abschlussbeweise:

- Die **Port-Lane** folgt Upstream-Pin, Dateiinventar, semantischer Parität,
  Differentials, Kandidatenreview und Promotion.
- Die **Integrations-Lane** folgt Zielsystem-Konfiguration, Secrets,
  Persistenz, UI/Policy, Runtime-Routing, Consumer-Auswahl und echtem E2E.

Diese Ledgers werden nicht addiert oder gemittelt. `Port 100 %` darf bedeuten,
dass der vereinbarte Upstream-Pin vollständig freigegeben ist, obwohl noch
kein Consumer ihn produktiv verdrahtet. Umgekehrt kann eine Produktintegration
gegen den Accepted Pin verifiziert bleiben, während ein neuer Upstream-
Kandidat offen ist. Ein historischer Ledger, der beide Lanes vermischt, wird
eingefroren und als historische Evidenz beschriftet; er darf nicht rückwirkend
zur aktuellen Gesamtmetrik umgedeutet werden.

Die Integrations-Lane braucht ein vorab geschlossenes Provider-/Access-Mode-
Inventar. Ein gelöschter Provider darf den Nenner nicht verkleinern. Jeder
`verified`-Status referenziert Gate-kompatible Evidenz, die an Accepted Pin,
Pfad und Hash gebunden ist; eine bloße Liste beteiligter Quelldateien genügt
nicht. Nach einer Pin-Promotion werden betroffene Integrationsgates automatisch
ungültig oder müssen gegen den neuen Pin erneut attestiert werden.

## 3. Datei-Mirror und Rust-Architektur sind zwei verschiedene Ebenen

Eine file-basierte Spiegelung ist für Upstream-Nachverfolgung wertvoll:

- Jeder relevante Upstream-Pfad erhält einen stabilen Zielpfad oder eine
  explizite Ersatz-Disposition.
- Source-Anker nennen Upstream-Datei, Zeilenbereich und Commit.
- Added, Modified, Deleted und Renamed bleiben im nächsten Delta auffindbar.
- Upstream-Tests besitzen eigene Spiegel und eigene Closure-Zähler.

Sie ist aber keine geeignete Rust-Modularchitektur. Go-Pakete, `init()`-Logik,
Dateisplitting und Build-Tags lassen sich nicht eins zu eins in den
Rust-Modulgraphen übertragen. Daher gilt:

- Der Mirror ist die **Traceability-Schicht**.
- Idiomatische, fachlich geschnittene Rust-Module sind die
  **Ausführungsschicht**.
- `mod.rs`/Fassaden und Registry-Aufbau sind zielsystemeigene
  Integrationspunkte.
- Große Go-Dateien dürfen hinter ihrer gespiegelten Fassade in kleinere pure
  Rust-Komponenten zerlegt werden.
- Viele kleine Go-Dateien, die gemeinsam einen Owner oder eine
  Zustandsmaschine bilden, werden als eine modulare Porting Unit bearbeitet.
- Beispiele, Binaries und Benchmarks werden explizit aktiviert; automatische
  Cargo-Target-Erkennung wird abgeschaltet, wenn gespiegelte Pfade sonst
  versehentlich ausführbare Targets erzeugen.

### Status eines Mirror-Pfads

Ein Pfad sollte mindestens einen der folgenden Zustände tragen:

- `scaffold`: Pfad vorhanden, keine Gutschrift;
- `partial`: echte Teilsemantik aktiv, Rest benannt, keine Vollgutschrift;
- `ported`: direkte, vollständige Übersetzung mit Referenztests;
- `adapted_to_target`: vollständige Zielsystem-Adaption mit gleicher
  beobachtbarer Semantik und begründeter Architekturabweichung;
- `replaced_by_target`: Upstream-Funktion bewusst durch eine sichere
  Zielkomponente ersetzt, inklusive Integrationsbeweis;
- `excluded`: nicht relevant, mit überprüfbarer Begründung;
- `removed`: Upstream-Pfad gelöscht und Zielauswirkung geprüft.

Der Status beschreibt nicht automatisch strikte Evidenz. Marker-Closure und
Strict-Closure bleiben getrennt.

## 4. Signature-first: nützlich als Inventar, gefährlich als Fortschritt

Alle Dateien und Signaturen zuerst anzulegen kann Abhängigkeiten und öffentliche
API-Flächen sichtbar machen. Es erzeugt aber drei typische Täuschungen:

1. Der Port sieht vollständig aus, obwohl nur kompilierende Attrappen bestehen.
2. Früh fixierte Go-förmige Signaturen zementieren falsche Ownership- und
   Lebenszeitmodelle in Rust.
3. Ein riesiger Modulgraph erzeugt Lint-, Target- und Buildarbeit, bevor eine
   einzige Capability ausführbar ist.

Deshalb:

- Scaffolds bleiben außerhalb des aktiven Modulgraphs, soweit sie nicht für
  eine aktuelle Slice benötigt werden.
- `todo!()`, `unimplemented!()`, konstante Erfolgswerte und leere Adapter sind
  keine Implementierung und werden durch Guards erfasst.
- Signaturen werden nur so weit stabilisiert, wie die nächste semantische
  Slice sie benötigt.
- Traits werden nach tatsächlichen Autoritäts- und Test-Seams gestaltet, nicht
  mechanisch aus jedem Go-Interface erzeugt.
- Kein Prozentkredit für „Datei angelegt“, „Signatur kompiliert“ oder
  „Modul exportiert“.

Ein vollständiges Scaffold-Inventar darf parallel erzeugt werden. Die Bodies
werden dagegen nach fachlichen, testbaren Einheiten portiert.

## 5. Arbeitsstrategie: semantische vertikale Slices

Die effektivste Porting Unit ist eine beobachtbare Fähigkeit von Eingabe bis
Ausgabe. Für einen Protokoll-Gateway umfasst eine Slice beispielsweise:

- Request-Konvertierung;
- nicht-streamende Response;
- Streaming-State-Machine und Terminierung;
- Tools, Reasoning, Usage und Fehler;
- Auth-/Refresh-Grenze;
- Registrierung und realen Aufrufpfad;
- Upstream- und Rust-Tests sowie Differentials.

Eine Slice ist erst geschlossen, wenn sie über den echten Produktionsgraphen
läuft. Ein isolierter Testhelper, der dieselbe Fixture nachbaut, beweist den
Port nicht.

### Empfohlene Reihenfolge

1. **Inventar und Pin einfrieren.** Alle Dateiklassen zählen und konservieren.
2. **Kompatibilitätssubstrat portieren.** Raw Bytes/JSON, Missing/Null/Value,
   SSE-Framing, Cancellation, Fehler- und Usage-Verträge.
3. **Differential-Runner bauen.** Dieselben Fixtures durch gepinnten Upstream
   und Rust ausführen.
4. **Erste kleine End-to-End-Slice.** Sie deckt falsche Architekturannahmen
   früh auf.
5. **Forensisch auswerten.** Gefundene Differenzen und Buildprobleme als neue
   Regeln und Tests festhalten.
6. **Nach Fan-out priorisieren.** Gemeinsame Helper nur zusammen mit einer
   nutzenden Slice aktivieren.
7. **Owner zusammen portieren.** Dateien eines gemeinsamen Lifecycle-,
   Scheduler- oder Registry-Owners nicht auf unabhängige Worker verteilen.
8. **Zugehörige Upstream-Testspiegel schließen.** Inline-Zusatztests ersetzen
   ihre Mirror-Disposition nicht.
9. **Wellenweise Vollgates.** Nach kleinen Prozent-/Capability-Schwellen die
   Strategie bewusst neu prüfen.
10. **Repository-Closure und Promotion.** Erst nach vollständigem Inventar,
    Full Matrix und Receipt.

Parallelisierung eignet sich für unabhängige pure Translatoren, Fixtures,
Testspiegel und read-only Audits. Shared Registries, Modulgraphen, Statusmarker,
Pins und Dashboard-Dateien haben einen eindeutigen Owner. Parallelität ohne
Ownership erzeugt mehr Integrationsarbeit als Portfortschritt.

## 6. Differential Testing: Verhalten vor Codeähnlichkeit

Ein Compiler beweist Typkonsistenz, nicht Upstream-Parität. Differentialtests
senden dieselbe Eingabe an den gepinnten Upstream und den Rust-Port und
vergleichen die relevante Beobachtung.

### Vergleichsmodi

- **Byte-exakt:** No-op-Pfade, Raw Tool Arguments, eingebettete Assets, Header
  und Framing, wenn der Vertrag Bytes verspricht.
- **Event-exakt:** Ereignisname, Reihenfolge, Index, Terminierung und Usage bei
  Streams.
- **Strukturell:** kanonisches JSON, wenn Whitespace/Key-Order nicht Teil des
  Vertrags ist.
- **Normalisiert:** ausschließlich benannte nondeterministische Felder wie
  Zeitstempel oder zufällige Request-IDs entfernen. Eine globale
  „sort-and-ignore“-Politik versteckt Fehler.
- **Zieladaptiert:** gleiche externe Wirkung, aber andere sichere Ownership;
  die Abweichung wird separat getestet.

Jeder im Differential gefundene Fehler wird zu einer dauerhaften Regression.
Ein Fixture-Korpus sollte Request, Non-Stream, Stream, Fehler, Cancellation,
fragmentierte Eingabe, unbekannte Erweiterungen und Grenzfälle umfassen.

## 7. Go→Rust-Differenzen, die regelmäßig Portfehler verursachen

### JSON und Bytes

- `[]byte` bleibt byte-orientiert. Eine Runde durch `serde_json::Value` kann
  Zahlenlexeme (`1.00`), Whitespace, Objektordnung oder unbekannte Felder
  verändern.
- No-op-Pfade geben die ursprünglichen Bytes zurück.
- Missing, explizites `null`, leer und konkreter Wert sind getrennte Zustände.
- `RawValue` oder byte-backed Carrier erhalten rohe Teilbäume an Stellen, an
  denen nur die Hülle validiert werden muss.
- JSON-Pfadlogik braucht semantischen Traversal-Kontext. Ein Property mit dem
  Namen `properties` oder `propertyNames` ist nicht automatisch ein
  Schema-Keyword.

### Streaming und Zustandsmaschinen

- SSE ist zuerst eine Transportgrammatik: beliebige Chunk-Grenzen, CRLF,
  mehrere `data:`-Zeilen, Kommentare und ein nicht terminiertes letztes Event.
- Transport-Framing und Modellaggregation besitzen getrennten Zustand.
- Fehlerhafte Transportdaten dürfen keinen Modellzustand teilweise mutieren.
- Event-Reihenfolge, Output-Indizes, Block-Finalisierung, `[DONE]`, Usage und
  Terminalpfade werden explizit geprüft.
- Cancellation wird sowohl vor dem Transform als auch zwischen Events und an
  Socket-/Task-Grenzen beachtet.
- Go `*any` als Streamzustand wird request-lokal typisiert; globaler
  übersetzungsbezogener Zustand ist verboten.

### Registrierung, Interfaces und Fehler

- Go `init()` wird durch explizite, testbare Registrierung ersetzt.
- Implizite globale Singletons werden instance-owned. Rust-Traits bilden
  Fähigkeiten und Autorität ab, nicht bloß Methodensätze.
- Go-Fehlerstring, Wrapping und Statusklassifikation können beobachtbare API
  sein; `anyhow`-Text allein ist kein Paritätsbeweis.
- Map-Iteration ist nicht stabil. Wo Ausgabeordnung sichtbar ist, muss sie
  explizit sortiert oder in Upstream-Reihenfolge erhalten werden.
- `defer`, frühe Returns und Goroutine-Abbruch werden als vollständige
  Lifecycle-Tabelle portiert: success, error, cancellation, timeout, drop und
  replacement.

### Datenbanken, Tabellen und Plattformpolicy

- Eine etablierte Rust-Crate ist nicht automatisch kompatibel mit der
  Upstream-Policy. MIME-Datenbanken, URL-Normalisierung, Headerdefaults,
  Zeitzonen oder TLS-Fingerprints können zusätzliche oder andere Werte haben.
- Geschlossene Upstream-Tabellen werden bei beobachtbarer Semantik vollständig
  portiert und differential geprüft; „bessere“ Extra-Unterstützung ist Drift.
- Plattform- oder Build-Tag-Code erhält eine explizite Disposition und einen
  passenden Plattformgate; Kompilieren auf nur einem Host genügt nicht für
  allgemeine Parität.

### Sicherheitsrelevante Carrier

- Signaturen, Thought-Carriers, Bypass-Sentinels und Replay-Metadaten sind
  Autorität, nicht bloß Strings.
- Syntaxgültigkeit beweist keine Herkunft. Provider, Richtung, Zieltyp,
  Nachbarschaft, Größe und synthetische Provenienz werden validiert.
- Temporäre synthetische Sentinels dürfen durch eine spätere generische
  Normalisierung nicht zu dauerhaftem Providerzustand werden.

## 8. Zielarchitektur und Autorität sicher adaptieren

Ein sicherer Rust-Port kopiert nicht automatisch die Autorität des
Upstream-Prozesses. Besonders kritisch sind globale Stores, Home-Verzeichnisse,
Umgebungsvariablen, Credential Helper, Shell-Aufrufe, Listener und Plugins.

### Grundregeln

- Netzwerk, Prozess, Dateisystem, Uhr, RNG, Secret Store, Persistenz,
  Scheduling und Reload werden typisiert injiziert.
- Produktionskonfiguration kommt aus dem autoritativen Config-/Runtime-/Secret
  Store des Zielsystems, nicht aus neuen Umgebungsvariablen. Isolierte
  Testkonfiguration ist zulässig.
- Credentials, Tokens, Header, Request-/Response-Bodies und Pfade erscheinen
  nicht in `Debug`, Logs oder Assertion-Fehlern. Sichere IDs und Zähler genügen.
- Provider-Fähigkeiten erhalten nur die Metadaten, die sie benötigen, nicht
  das gesamte Secret-bearing Auth-Objekt.
- Konfigurations- und Registry-Publikation erfolgt build-then-swap bzw.
  transaktional. Leser dürfen keine halbe Revision sehen.
- Compound Owner-Operationen werden serialisiert; externe Callbacks laufen nie
  unter Manager-/Registry-Locks.
- Handles und Streams sind an Erzeuger, Generation und Lifecycle gebunden.
  Stale Generationen dürfen Ersatzobjekte nicht schließen.
- Plugin/cgo-Grenzen werden als isolierter Prozess, IPC oder WASM neu gebaut;
  Prozessisolation ersetzt jedoch keine per-plugin Autorisierung und atomare
  Snapshot-Publikation.
- Ein SDK-Facade darf die im Inneren entfernte Autorität nicht wieder
  erschaffen. Loader, Transport und Stores bleiben Konstruktorpflichten.
- Management-HTTP bleibt Control Plane. Ein Port darf nicht nebenbei eine
  verbotene Datenbrücke in das Zielsystem einführen.

Eine Disposition `adapted_to_target` verdient erst Vollgutschrift, wenn der
reale Zielowner die Capability konsumiert. Ein sicherer, aber ungebundener
Fail-closed-Adapter ist noch keine abgeschlossene Integration.

## 9. Nachhaltiger Upstream-Prozess: Baseline plus Candidate Overlay

Der akzeptierte Pin bleibt während der Kandidatenarbeit unverändert. Ein
Kandidat ist ein separates, commit-adressiertes Overlay:

```text
akzeptierter Pin
  ├─ unveränderliches Mirror-/Strict-Ledger
  └─ Kandidat <commit>
       ├─ vollständiger Delta
       ├─ fail-closed Review
       ├─ Impact Summary
       ├─ Gate-Evidenz
       └─ Promotion oder verwerfbarer Kandidat
```

Discovery darf automatisiert werden; Promotion bleibt bewusst autorisiert und
gated. Runtime-Code lädt oder aktualisiert Upstream niemals selbständig.

### Delta-Inventar: Conservation vor Review

Der Generator verarbeitet mindestens:

- Added, Modified, Deleted und Renamed;
- Produktions-Go und Go-Tests;
- alle verschachtelten `go.mod`, `go.sum`, Workspaces und Lockdateien;
- Build/Release, CI, Container, Runtime-Assets, Konfiguration;
- Lizenz, Dokumentation, Beispiele und sonstige Pfade.

Für jede Stufe gilt die Conservation-Invariante:

```text
Git name-status records == normalisierte Zeilen == JSON changes == Review rows
```

Die Gleichheit wird maschinell geprüft. Besonders wichtig:

- Ein Added-Pfad besitzt keinen alten Pfad; `old_upstream` ist explizit `null`.
  Ein Filter auf einen leeren Objektwert darf nicht den ganzen Datensatz
  verwerfen.
- Rename besitzt alte und neue Identität und eine eigene Aktion.
- Unbekannte Dateiklassen fallen in `other`, niemals aus dem Inventar.
- Pfade sind eindeutig; Duplikate oder fehlende Rows brechen fail-closed ab.
- Der Kandidatendelta wird gegen einen synthetischen Git-Fixture mit allen
  Status- und Dateiklassen getestet.

Erst nachdem diese Invariante grün ist, beginnt semantisches Review. Sonst ist
jeder Prozentwert des Reviews ungültig.

### Review-Ledger

Jeder Delta-Pfad enthält mindestens:

- stabile `upstream`-Identität, Modul, Änderungsart und Source-Kind;
- geforderte Aktion;
- Status `pending|complete`;
- Disposition und Begründung;
- Upstream-Evidenz;
- Rust-Evidenz für Produktions- und Testcode.

Das initiale Ledger ist fail-closed: alle Pfade pending, alle Gates false und
alle Gate-Evidenzen leer. Resume darf ein identisches Review bytegenau
erhalten; ein nicht passendes vorhandenes Artefakt wird niemals überschrieben.

### Inventarreparatur und Remaps

Wird ein fehlerhaftes Inventar entdeckt:

1. Alte Delta-, Review- und Summary-Artefakte unverändert archivieren.
2. Korrigierten Delta und ein frisches fail-closed Review erzeugen.
3. Fertige Dispositionen ausschließlich über eindeutige stabile Pfade
   übernehmen, nie über Array-Index oder Position.
4. Falsch zugeordnete Evidenz nur über eine explizite Remap-Datei mit
   `from_upstream`, `to_upstream` und Begründung retten.
5. Inventory-Felder am Ziel gegen den korrigierten Delta prüfen.
6. Direkte Übernahmen + Remaps müssen exakt der Zahl gültiger alter
   Completions entsprechen; kein Eintrag darf verschwinden oder doppelt
   erscheinen.
7. Alle globalen Gates und Gate-Evidenzen zurück auf pending setzen.
8. Hashes von altem Review, korrigiertem Delta und frischem Review sowie alle
   Remaps im Reconciliation-Metadatensatz speichern.

Eine Reparatur verändert niemals still den akzeptierten Pin und übernimmt
keine alten globalen Gates, weil deren Scope auf dem falschen Inventar beruhte.

## 10. Testmatrizen und beweisbare Receipts

### Minimale Full Matrix vor Promotion

Das genaue Set wird projektspezifisch eingefroren. Für einen Rust-Port mit Go
als Referenz umfasst es typischerweise:

- alle aus dem gepinnten Git-Baum abgeleiteten Go-Module;
- Go-Tests nach kontrollierter Dependency-Hydration offline mit
  `-mod=readonly`;
- Rust ohne Default Features;
- Rust mit allen produktiven Features;
- Clippy `--all-targets` in beiden Matrizen mit `-D warnings`;
- Formatierung;
- Integration in den äußeren Host/Workspace;
- Tracking-, Anchor-, Inventar- und Dashboard-Validatoren;
- gezielte Plattform-, Security-, Differential- und Liveness-Gates.

Focused Tests sind Entwicklungsfeedback. Full-Matrix-Gates sind
Promotionsbeweise; keines ersetzt das andere.

### Forensische Infrastruktur-Learnings

- **Accepted Checkout bleibt sauber.** Tests laufen aus einem `git archive`
  in einer Sandbox. Dependency-Hydration darf nur diese Kopie ändern.
- **Offline-Phase nach Hydration.** Erst fehlende Summen/Module im Sandboxbaum
  auflösen, danach mit deaktiviertem Proxy und Readonly-Modus testen. Der
  hydratisierte Manifestzustand wird gehasht.
- **Alle Go-Module aus dem Git-Baum ableiten.** Eine hart codierte Rootliste
  übersieht verschachtelte Beispiel-/Pluginmodule.
- **Module seriell, Tests intern normal parallel.** `go test -p=1` begrenzt
  Paket-/Modulparallelität. `-parallel=1` kann Tests deadlocken, die parallele
  Children bewusst synchronisieren.
- **Kurzer kanonischer Temp-Root.** Lange Temp-Pfade im Repository können
  Filesystem-/Git-Grenztests brechen. System-Temp oder ein dediziertes
  beschreibbares Volume verwenden.
- **Exklusives Cargo Target pro Gate-Run.** Ein geteiltes Target kann
  inkrementelle Cache-Races und fehlende Artefakte erzeugen. Das Target wird
  exakt isoliert und anschließend entfernt.
- **Speicher vor einer Vollmatrix messen.** Rust-All-Features und mehrere
  Host-Harnesses können zweistellige GiB-Mengen erzeugen. Root- und
  Temp-Volume werden vor dem Lauf geprüft; ausschließlich eindeutig
  regenerierbare, portbezogene Target-Verzeichnisse dürfen nach beendetem
  Build über ihren exakten Pfad bereinigt werden. `ENOSPC` ist ein
  Infrastrukturfehler und niemals ein negativer Semantikbeweis.
- **macOS-Provenance von Test-Hangs unterscheiden.** Auf externen Volumes kann
  `com.apple.provenance` ein frisch gelinktes großes Testbinary bereits vor
  `main` blockieren. Ein externer Timeout plus Prozess-Footprint trennt diesen
  Loaderzustand von hängender Testlogik. Für fokussierte Diagnose darf eine
  Kopie in einen kurzen lokalen Temp-Pfad verwendet und ausschließlich dort
  das Provenance-xattr entfernt werden; der eigentliche Promotion-Gate muss
  weiterhin auf dem dokumentierten Produktionspfad erfolgreich enden.
- **Native Loader separat beweisen.** Ein Binary zu verschieben oder nur das
  Target-Verzeichnis zu wechseln ist bei intermittierenden macOS Loader-/TLS-
  Stalls keine Evidenz. Der reale Execution Gate muss erfolgreich enden.
- **Testinterne Parallelitätsannahmen respektieren.** Infrastruktur-Tuning darf
  keine Semantik des Testframeworks ändern.
- **Fehlversuche diagnostizierbar, aber sauber.** Kleine Logs können erhalten
  bleiben; große Sandboxes und Target-Verzeichnisse werden über exakte Pfade
  bereinigt.

### Receipt-Anforderungen

Ein Promotions- oder Umbrella-Receipt ist commit-adressiert,
nicht überschreibbar und enthält:

- Schema-Version, Upstream-Commit, Start-/Endzeit und Toolchain-Versionen;
- Clean-before/Clean-after-Belege;
- exakte argv-Arrays statt eines nur lesbaren Shell-Strings;
- Gate-Status, relative sichere Logpfade und SHA-256 jedes Logs;
- Source-Manifest und gegebenenfalls hydratisiertes Dependency-Manifest mit
  SHA-256;
- vollständigen Delta, Dispositionen und Gate-Evidenzen bei Promotion;
- Validatorversion bzw. ein stabiles Schema.

Der Validator prüft Schema, erwartete Gate-Menge, Eindeutigkeit, sichere
relative Pfade, Existenz und Hash jedes Artefakts sowie Commit-Identität. Ein
manuell auf `true` gesetztes Gate ist keine Evidenz.

Historische Receipts binden den damaligen Source-Snapshot. Sie werden nicht
gegen einen später legitimerweise geänderten Kandidaten-Worktree umgedeutet.

## 11. Promotion-Checkliste

Ein Kandidat darf nur promoviert werden, wenn alle Punkte nachweislich erfüllt
sind:

- [ ] Repository, alter Pin und Kandidaten-Commit sind eindeutig.
- [ ] Accepted Checkout ist sauber und entspricht dem alten Pin.
- [ ] Delta-Conservation ist grün; alle Dateiklassen sind enthalten.
- [ ] Jeder Delta-Pfad ist eindeutig und vollständig dispositioniert.
- [ ] Added/Deleted/Renamed und Non-Go-Auswirkungen sind explizit geprüft.
- [ ] Alle direkten und adaptierten Rust-Bodies besitzen Upstream- und
      Rust-Evidenz.
- [ ] Provider-/Format-Differentials sind grün, inklusive Streams und Fehler.
- [ ] Authority-, Secret-, Redaction- und Lifecycle-Regeln sind geprüft.
- [ ] Alle Go-Module und beide Rust-Featurematrizen sind grün.
- [ ] Beide Clippy-All-Targets-Matrizen und Formatierung sind grün.
- [ ] Äußerer Host, Integrationen, Tracking, Anchors und Dashboard sind grün.
- [ ] Jedes Gate wurde ausgeführt, geloggt und gehasht.
- [ ] Review-Status ist maschinell `ready_for_promotion`.
- [ ] Promotion hat einen vollständigen Pre-Mutation-Snapshot und Rollback.
- [ ] Pin, Dokumentation, Source-Anker und generierte Maps werden atomar auf
      denselben Commit aktualisiert.
- [ ] Nach der Mutation laufen Anchor- und Tracking-Checks erneut.
- [ ] Ein nicht überschreibbares, geprüftes History-Receipt wurde geschrieben.
- [ ] Dashboard zeigt den neuen Accepted Pin und keinen offenen Kandidaten als
      abgeschlossen an.

Schlägt eine Mutation oder ein Generator fehl, wird der alte Zustand vollständig
wiederhergestellt. Ein gemischter Baseline-Zustand ist schwerwiegender als eine
nicht erfolgte Promotion.

## 12. Tracking-Dashboard für Ports

Das Dashboard ist eine generierte Sicht auf validierte Projektartefakte, nicht
selbst die Wahrheit. Bediener dürfen das HTML niemals von Hand korrigieren, und
Worker dürfen Prozentwerte nicht als freie Texte eintragen.

### Source of Truth und Schema

Eine kleine kanonische JSON-Datei beschreibt Identität, Metriken und
Abschlussprädikate. Die detaillierten Pfadlisten, Reviews und Receipts bleiben
in eigenen Artefakten; die Statusdatei verweist auf sie und wiederholt nur
validierte Summen.

Minimalbeispiel:

```json
{
  "schema": "org.example.rust-port.status.v1",
  "accepted": {
    "commit": "0123456789abcdef0123456789abcdef01234567",
    "mechanical": {
      "production": { "complete": 605, "total": 605 },
      "tests": { "complete": 418, "total": 418 }
    },
    "strict": {
      "production": { "complete": 605, "total": 605 },
      "tests": { "complete": 418, "total": 418 }
    },
    "receipt": {
      "path": "receipts/accepted.json",
      "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    }
  },
  "candidate": {
    "commit": "89abcdef0123456789abcdef0123456789abcdef",
    "review": { "complete": 14, "total": 111 },
    "gates": { "passed": 0, "total": 10 },
    "promoted": false
  },
  "completion": {
    "scope_inventory_complete": true,
    "accepted_strict_complete": true,
    "semantic_scope_complete": true,
    "candidate_resolved": false,
    "all_required_gates_attested": false,
    "is_complete": false
  },
  "artifacts": {
    "port_map": "port-map.json",
    "candidate_review": "reviews/89abcdef/upstream-review.json"
  }
}
```

Ein JSON Schema oder gleichwertiger Validator erzwingt mindestens:

- bekannte Schema-Version und vollständige Commit-Hashes;
- nichtnegative ganzzahlige Zähler und positive Nenner;
- `complete <= total` bzw. `passed <= total`;
- getrennte Nenner für Produktion, Tests, Strict, Review und Gates;
- sichere relative Artefaktpfade und gültige SHA-256-Werte;
- Candidate entweder vollständig `null` oder mit Commit, Review, Gates und
  Promotion-Status;
- Übereinstimmung der Summen mit den referenzierten Detailartefakten;
- Übereinstimmung des Accepted Commit in Lock, Map, Receipt und Status;
- Übereinstimmung von `completion.is_complete` mit der eingefrorenen
  booleschen Abschlussformel.

### Port-Lane: Accepted und Candidate

Das UI zeigt zwei visuell und semantisch getrennte Spuren:

1. **Accepted Baseline:** Pin, mechanische Produktion/Tests, strikte
   Produktion/Tests, semantischer Scope und letztes valides Receipt.
2. **Upstream Candidate:** Kandidaten-Commit, Delta-Inventar, Review-Zähler,
   Gates, offene Module und Promotion-Status.

Eine zu 100 % geschlossene Accepted Baseline bleibt ein positiver historischer
Status. Sie darf aber weder die Candidate-Lane überdecken noch den
Projektzustand auf abgeschlossen setzen. Gibt es keinen Kandidaten, zeigt die
zweite Lane ausdrücklich `kein offener Kandidat`; sie wird nicht still
weggelassen.

Außerhalb dieser beiden Upstream-Spuren steht eine eigene
**Produktintegrations-Lane**. Sie besitzt eine separate maschinenlesbare
Provider-/Consumer-Matrix und boolesche Capability-Gates. Weder ihre
Gate-Zähler noch ihr Status fließen in den Port-Prozentwert ein. Das Dashboard
muss die Grenze in der obersten Ansicht erklären; ein Leser darf eine
untergeordnete 100-%-Karte nicht als Abschluss des zusammengesetzten Produkts
missverstehen können.

### Boolesche Completion Rule

Projektabschluss ist keine Prozentformel, sondern eine Konjunktion. Zum
Beispiel:

```text
is_complete =
  scope_inventory_complete
  AND accepted_mechanical_production_complete
  AND accepted_mechanical_tests_complete
  AND accepted_strict_production_complete
  AND accepted_strict_tests_complete
  AND semantic_scope_complete
  AND candidate_resolved
  AND all_required_gates_attested
  AND tracking_consistent
  AND receipt_valid
```

`candidate_resolved` bedeutet: kein offener Kandidat gehört zum vereinbarten
Scope **oder** der Kandidat wurde vollständig geprüft und promoviert. Ein
Reviewwert von 100 % allein genügt nicht; Gates und Promotion sind eigene
Prädikate.

Eine Prozentanzeige für die **Port-Freigabe** ist nur zulässig, wenn ihr Nenner vor dem Lauf aus
atomaren, maschinenprüfbaren Abschlussaktionen eingefroren wurde und 100 %
genau dann erreichbar sind, wenn auch die boolesche Completion Rule wahr ist.
Für einen aktiven Upstream-Promotionszyklus lautet die konservative Formel:

```text
port_freigabe =
  (vollständige Pfadreviews + attestierte Gates + Promotion + Post-Full-Gate)
  / (alle Pfadreviews + alle Gates + 1 + 1)
```

Accepted-Closure und Capability-Punkte werden nicht erneut hineingemischt;
sie bleiben boolesche Vorbedingungen bzw. eigene historische Ledgers. Das
Dashboard zeigt Zähler, Nenner und Formel direkt neben dem Prozentwert. Ein
Guard muss jeden Zustand ablehnen, in dem `100,0 %` und
`project_completion.complete` voneinander abweichen. Untergeordnete Lanes
zeigen weiterhin ihre eigenen Brüche, aber keine konkurrierende große
100-%-Karte.

### Standalone-HTML-Generator

Das Artefakt ist eine einzelne lokal öffnungsfähige HTML-Datei ohne Server-
oder Netzwerkabhängigkeit. Der Generator:

1. validiert zuerst alle Source-of-truth-Artefakte;
2. berechnet Summen deterministisch aus den Detailledgers;
3. bettet den validierten JSON-Snapshot, bevorzugt Base64-kodiert, in ein
   statisches Template ein;
4. rendert beide Lanes und die booleschen Abschlussprädikate;
5. schreibt über eine temporäre Datei atomar an das Ziel;
6. trägt `generated_at`, Input-Hashes und Generator-Schema in den Snapshot ein;
7. verändert niemals Source-of-truth-Dateien.

Eine Auto-Reload-Funktion darf die lokale Datei regelmäßig neu laden, ist aber
kein Ersatz für Regeneration. Im sichtbaren Footer stehen Generierungszeit,
Accepted-/Candidate-Commit und Input-Hashes, damit ein Bediener einen alten
Tab erkennen kann.

### Renderer testen, nicht HTML-Strings suchen

Ein Dashboardtest führt das eingebettete JavaScript in einer DOM-Umgebung oder
einem echten Browser aus. Er prüft mindestens:

- Accepted `605/605` plus Candidate `14/111` rendert global
  `in_progress`, niemals `100 % abgeschlossen`;
- ohne Candidate wird `kein offener Kandidat` sichtbar;
- `is_complete=false` dominiert jede untergeordnete 100-%-Metrik;
- Zähler, Nenner, Commit und Metrikname erscheinen gemeinsam;
- ungültige/fehlende Daten führen zu einer sichtbaren Fehlerkarte und nicht zu
  `0 %` oder `100 %`;
- HTML funktioniert als `file://`-Artefakt ohne Fetch;
- der im DOM sichtbare Stand entspricht dem eingebetteten JSON, nicht
  hardcodierten Templatewerten.

Ein `grep` nach einem erwarteten String beweist nur, dass Text im Quell-HTML
steht. Er beweist weder, dass der Renderer ihn auswählt, noch dass eine falsche
100-%-Karte verborgen bleibt.

### Stale-State verhindern

- Der Generator bricht ab, wenn Lock, Checkout, Maps, Delta, Review oder
  Receipt unterschiedliche Commits nennen.
- Regenerierte Port-/Module-/Closure-Maps werden bytegenau mit den
  eingecheckten bzw. autoritativen Maps verglichen.
- Candidate-Summen werden aus dem aktuellen vollständigen Delta berechnet;
  ein Review mit abweichender Pfadmenge ist unrenderbar.
- Input-Hashes werden im HTML eingebettet. Ein Check-Kommando vergleicht sie
  mit den aktuellen Artefakten und meldet ein veraltetes Dashboard.
- Eine Generierungszeit ist nur Frischeinformation, keine Evidenz. Hash und
  Commit entscheiden.
- Ein vorhandenes Review mit Operatorfortschritt wird bei Regeneration nicht
  überschrieben; Identitätsabweichung bricht fail-closed ab.
- Nach Promotion wird das Dashboard erst nach erneuten Anchor-, Tracking- und
  Receipt-Checks veröffentlicht.

### Update-Takt für Worker

Jeder Worker aktualisiert nach einer abgeschlossenen, integrierten Welle:

1. Pfad-/Review-Dispositionen und Evidenz;
2. relevante Gate-Ergebnisse;
3. Strategie-/Forensiknotiz;
4. validierte Projektartefakte;
5. das generierte Standalone-Dashboard.

Bei langen Gates wird zusätzlich nach einem Zustandswechsel aktualisiert, etwa
`review abgeschlossen, Full Matrix läuft` oder `Gate fehlgeschlagen`. Ein
Worker darf das Dashboard nicht vor den Ledgers aktualisieren und keine
voraussichtliche Gutschrift anzeigen. Parallel arbeitende Worker schreiben
nicht direkt in dasselbe Dashboard; ein Integrationsowner regeneriert es nach
dem Zusammenführen ihrer Evidenz.

### Logs und Receipts im Dashboard

Das Dashboard zeigt nur attestierte Gatezustände. `passed` erfordert ein
Receipt oder einen Gate-Evidence-Eintrag mit ausgeführtem argv, Abschlusszeit,
relativem Logpfad und SHA-256. Links dürfen auf lokale Receipt-/Logartefakte
zeigen; Secrets und vollständige Request-/Response-Payloads werden nicht
eingebettet. Ein fehlender oder nicht passender Hash wird als `ungültig`, nicht
als `passed`, gerendert.

Ein Hash allein authentisiert nur Bytes, nicht deren Bedeutung. Der Generator
muss deshalb jedes Receipt erneut mit dem zu seiner `receipt_schema` gehörenden
Validator prüfen und Commit/Pin explizit binden. Leere oder unbekannte Schemas
sind ungültig. Produktintegrationsevidenz wird auf `(Provider-Modus, Gate)`
gebucht und nennt die konkreten bestandenen Checks; ein global grüner Lauf darf
keinen pauschalen Credit auf andere Modi oder Gates verteilen.

Ein Current-Source-Receipt enthält ein Manifest aller beweisrelevanten Quellen,
Tests, Runner und Validatoren. Der spätere Dashboardlauf hasht diese Dateien
erneut. Damit invalidieren Änderungen am getesteten Code ebenso wie Änderungen
am Evidence-Checker selbst einen alten grünen Beleg. Das eigentliche Ledger darf
wegen der zyklischen Receipt-Hash-Referenz außerhalb dieses Manifests liegen,
wird aber separat schema-, pin- und hashgebunden validiert.

Ein Gate-Test darf fehlende Laufzeitvoraussetzungen nicht still als Erfolg
verbuchen. Wenn etwa Node, ein Sidecar-Bundle, ein Provider-Konto oder eine
Credential fehlt, muss der Gate-Runner vor dem Test fail-closed prüfen oder der
Test einen maschinenlesbaren `skipped`-Status erzeugen, den der Receipt-Checker
nicht als `passed` akzeptiert. Ein Prozess-Exitcode 0 nach einer bloßen
`SKIP`-Meldung ist keine Capability-Evidenz.

Produktnahe UI-Belege gehören ebenfalls in die Source-Closure: Komponenten,
Styles, lokale Logos, Provenienzmanifest, Lizenztext und UI-Test. Ein grüner
UI-Test bei nicht gebundenen Assets beweist sonst nicht das ausgelieferte
Standalone-/Business-OS-Artefakt.

In einem Dirty-Workspace muss die Evidence-Closure explizit sein: relevante
Crates, Adapter, Stores, UIs, Tests, Runner und Validatoren gehören hinein. Das
Manifest bindet für diese Closure den binären Delta der getrackten Quellen gegen
`HEAD` sowie Inventar und Hashes der ungetrackten Quellen; clean getrackte Dateien
sind durch den Commit gebunden. Der Receipt konsumierende Ledger ist die
explizite Ausnahme. Den gesamten Dirty-Workspace pauschal zu binden ist ebenfalls
ein Fehler: fachfremde Änderungen machen dann dauerhaft jede Produktevidenz
ungültig. Bei einer monolithischen Crate ist für stärkere Isolation langfristig
ein eigener Test-Harness oder ein unveränderlicher Snapshot-Checkout vorzuziehen.

Große Grenzwerttests sollten die Produktionskonstante ausdrücklich prüfen und
den Transport mit einem injizierten kleineren Testlimit belasten. Acht Megabyte
Testdaten durch einen stark ausgelasteten CI-Socket zu schieben misst sonst eher
Maschinenlast als die deterministische Reject-Semantik. Das Produktionslimit
bleibt unverändert und sichtbar; der schnelle Test beweist denselben Codepfad.

Bei sehr großen Testbinärlingen wird das Testprofil einmal gebaut und der exakt
aufgelöste Binärpfad anschließend direkt mit den einzelnen Filtern ausgeführt.
Jeder Filter behält sein eigenes gehashtes Log. Wiederholte `cargo test`-Aufrufe
pro Filter können auf externen Volumes minutenlang nur Fingerprints scannen und
fügen dabei keine neue Compile-Evidenz hinzu.

### Dashboard-Abnahmecheckliste

- [ ] JSON Schema und Cross-Artifact-Validator laufen vor dem Renderer.
- [ ] Jede Prozentzahl nennt Metrik, Zähler, Nenner und Commit/Lane.
- [ ] Port-Track und Produktintegrations-Track sind getrennt sichtbar.
- [ ] Innerhalb des Port-Tracks sind Accepted und Candidate getrennt sichtbar.
- [ ] Es existiert keine zusammengesetzte Gesamt-% über Port und Integration.
- [ ] Port-Completion und Integrations-Completion folgen jeweils ihrer eigenen booleschen Regel.
- [ ] Offene Abschlussprädikate werden namentlich angezeigt.
- [ ] Das HTML ist standalone und atomar generiert.
- [ ] Footer zeigt Zeitpunkt, Input-Hashes und Commit-Identitäten.
- [ ] Renderer-Execution-Test deckt 100-%-Accepted plus offenen Kandidaten ab.
- [ ] Stale-, Schema-, Commit- und Hashfehler sind fail-closed sichtbar.
- [ ] Worker-Regeneration erfolgt nach jeder integrierten Welle.
- [ ] Gatekarten verweisen nur auf hashgebundene Evidenz.

## 13. Wiederverwendbares Artefaktmodell

Für künftige Ports sollte das Projekt mindestens diese Artefakte führen:

| Artefakt | Pflichtinhalt | Lebensdauer |
|---|---|---|
| `upstream-lock.json` | Repository, Commit, Datum, Policy | akzeptierte Baseline |
| `port-map.json` | jeder Upstream-Pfad, Rust-Pfad, Status, Anchor | regenerierbar, geprüft |
| `module-map.json` | Pfade→fachliche Porting Units/Owner | regenerierbar, geprüft |
| `semantic-ledger.json/md` | eingefrorene Capabilities, Punkte, Acceptance | dauerhaft |
| `mirror-closure.json` | mechanische und strikte Produktion/Test-Zähler | regenerierbar |
| `upstream-delta.json` | vollständiger Kandidatendelta und Summen | je Kandidaten-Commit |
| `upstream-review.json` | Disposition je Pfad, Gates, Evidenz | resumefähig, fail-closed |
| `impact-summary.json` | Module, Dateiklassen, Aktionen | je Kandidaten-Commit |
| `reconciliation.json` | alte/neue Hashes, stabile Replays, Remaps | nur bei Reparatur |
| `product-integration.json` | Provider-/Consumer-Gates außerhalb des Ports | je Accepted Pin und Zielsystemrelease |
| Gate-Logs | argv, Output, Status, Hash | je Kandidat/Receipt |
| Promotion-Receipt | Delta, Review, Gates, Manifeste, Hashes | dauerhaft, immutable |
| Dashboard | klar getrennte Metrikachsen und Status | regelmäßig generiert |
| `PORTING.md` | Entscheidungen, Forensik, Strategieänderungen | dauerhaft |

### Mindestinvarianten für Generatoren

- deterministische Ausgabe bei identischem Input;
- Schema-Version und Commit-Identität;
- vollständige Mengen- und Eindeutigkeitsprüfungen;
- atomisches Publish ohne Clobbering operatorgepflegter Arbeit;
- temporäre Dateien neben dem Ziel oder auf einem kontrollierten Temp-Root;
- keine Baseline-Mutation bei Discovery/Review;
- Validatoren werden zusammen mit Schemaänderungen getestet;
- Dashboard wird ausschließlich aus validierten Artefakten gebaut.

## 14. Porting-Worker-Protokoll

Jede kleine Arbeitswelle hinterlässt eine prüfbare Übergabe:

1. exakt benannte Upstream-Pfade und Zielmodule;
2. implementierte Semantik und bewusst offener Rest;
3. neue oder übersetzte Tests und Differential-Fixtures;
4. ausgeführte Kommandos, Featurematrix und Ergebnis;
5. Pfadstatus nur nach erfolgreichem Gate;
6. gefundene Paritäts-, Architektur- oder Infrastrukturfehler;
7. daraus abgeleitete Strategieänderung;
8. aktualisierte maschinenlesbare Ledgers und Dashboard;
9. keine Aussage „fertig“ außerhalb des eigenen nachgewiesenen Scopes.

Nach jeder substanziellen Welle wird geprüft, ob die Strategie noch trägt:

- Werden Dateien geschlossen oder echte Owner/Capabilities?
- Finden Differentials weiterhin reale Fehler?
- Wächst `partial` ohne benannten Rest?
- Sind Testspiegel und Produktion synchron?
- Gibt es einen Shared Owner, der Parallelisierung begrenzt?
- Sind neue Upstream-Dateien im Inventar sichtbar?
- Kann jede Gutschrift auf Pfade und Gate-Evidenz zurückgeführt werden?
- Zeigt das Dashboard den tatsächlichen Stand ohne Scope-Verwechslung?

## 15. Definition of Done

Der vollständige Port ist abgeschlossen, wenn gleichzeitig gilt:

1. Das Scope-Inventar ist vollständig und konserviert.
2. Jeder Produktions- und Testpfad hat eine geprüfte End-Disposition.
3. Die vereinbarten Capabilities laufen über den realen Zielgraphen.
4. Direkte Ports sind differential oder durch gleich starke Referenzevidenz
   gegen den akzeptierten Upstream geprüft.
5. Zielsystem-Adaptionen bewahren Semantik und verschärfen keine Autorität.
6. Mechanische und strikte Closure sind vollständig.
7. Alle eingefrorenen Full-Matrix-, Security-, Plattform-, Tracking- und
   Integrationsgates sind attestiert.
8. Ein offener Kandidat ist entweder vollständig promoviert oder ausdrücklich
   außerhalb des Abschluss-Scopes; er wird niemals durch die 100-%-Anzeige der
   alten Baseline verdeckt.
9. Pin, Maps, Anchors, Dokumentation, Receipt und Dashboard stimmen überein.
10. Ein unabhängiger Completion-Audit kann jede Anforderung auf aktuelle,
    konkrete Evidenz zurückführen.

Fehlt einer dieser Beweise, lautet der Projektstatus `in_progress` – auch dann,
wenn einzelne Ledgers korrekt 100 % anzeigen.
