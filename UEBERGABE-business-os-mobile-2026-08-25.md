# ÜBERGABE · CTOX Business OS Mobile — Stand 2026-08-25 ~11:30

Für den übernehmenden Agenten. Dieses Dokument ist die Wiederaufnahme-Quelle:
es setzt KEINEN Chatverlauf voraus. Alles hier ist selbst verifiziert, außer
wo ausdrücklich „unverifiziert" steht.

---

## 0 · Die eine Falle zuerst: es gibt ZWEI „CTOX Desktop"-Apps

1. **CTOX Business OS** — dieses Repo, Shell in `src/apps/business-os/`
   (no-build HTML/JS/CSS), Electron-Launcher in `src/apps/business-os-desktop/`.
   Datenebene AUSSCHLIESSLICH RxDB/WebRTC über Signaling (`wss://signaling.ctox.dev`
   bzw. lokaler Signaling-Server). **HTTP ist nur zur Auslieferung statischer
   Shell-Dateien erlaubt (Loopback), niemals als Datenpfad.** Das ist die App,
   die der Operator meint, wenn er „CTOX Desktop" sagt.
2. **„CTOX Desktop App"** im workjet-Repo (`~/Documents/workjet/apps/desktop`,
   umbenanntes T3 Code, Electron + HTTP/WebSocket-Backend). Anderes Produkt,
   andere Architektur. Verwechslung hat einen halben Arbeitstag gekostet.

## 1 · Erledigt und committet (verifizierte Fakten)

**ctox-Repo, Branch main** — Business OS mobil bedienbar gemacht:

| Commit | Inhalt |
| --- | --- |
| `7876ab4bd` | Phase 1: Shell (scrollender Desktop, Fenster→Vollbild-Sheets <768px, Topbar/Chat-Dock/Drawer viewport-sicher, ≥44px Tap-Ziele) + Module desktop, threads, tickets, app-store |
| `9b8bab176` | Phase 2: 18 Modul-Frontends (browser, knowledge, creator, credentials, reports, conversations, mail, ctox, calendar, notes, buchhaltung, spreadsheets, interviews, matching, esign, intake, submissions, coding-agents) |
| `74cb83127` | Phase 2b: documents, support, importer, consent, customers, invoices |
| `5665df8aa` | Phase 2c: appsec-pentest, cv-print-builder, iot, nachweise, outbound, placements, research, shiftflow |

Damit hat **jedes** Business-OS-Modul einen Mobile-Pass. Konventionen: ein
Breakpoint bei **767px**, Spalten stapeln, Tabellen scrollen im eigenen
Container (nie die Seite), Tap-Ziele ≥44px, CSS-only (fremd-modifizierte
JS/Schema-Dateien bewusst unangetastet). Umgesetzt von Kimi · UI/UX über
Workjet (Run-IDs in §6), von mir per Diff, Zeitstempel, Klammer-Bilanz und
Simulator-Stichproben unabhängig verifiziert. Kein einziger neuer
`http://`/`fetch(`-Pfad im gesamten Sweep.

**workjet-Repo, Branch codex/workjet-native-foundation** (anderes Produkt,
T3-Code-Mobile): CTOX-Identität committet (`7431f120e` Rebrand, `6d710849c`
Icons vom neuen Turbofan-Master). iOS-Dev-Build lief end-to-end auf dem
Simulator (Pairing, Projekt sichtbar). Nicht Teil dieses Streams, nur damit
du es nicht doppelt machst.

## 2 · AKTIVER ZUSTAND auf dieser Maschine (bitte zuerst lesen)

- **Overlay auf der Produktiv-Installation:** Die 8 Shell-Dateien (Phase 1)
  und alle 36 Modul-`index.css` sind über
  `~/.local/lib/ctox/current/business-os/` gelegt, damit der Operator den
  Stand sofort sieht. Originale liegen als Backup unter
  `~/.local/state/ctox-mobile-handover/bos-shell-backup/`. Das Overlay wird
  vom nächsten regulären ctox-Update überschrieben — das ist okay, der Stand
  ist committet. Restore bei Bedarf: Backup zurückkopieren.
- **Shell-Serve** ggf. aktiv: `ctox business-os serve --addr 127.0.0.1:8799`
  (liefert nur statische Dateien; der native RxDB-Peer läuft im
  ctox-real-Daemon und meldet „lock held" — das ist korrekt so).
- **iOS-Simulator:** Gerät „CTOX-Test-iPhone", UDID
  `CFE1E053-1B1D-435E-AECF-E8208A508A49`, iOS 26.5. Business OS dort via
  Safari auf `http://127.0.0.1:8799/business-os/` — Sync gegen die lokale
  Instanz funktioniert (live gesehen: Desktop, Threads, Credentials, Mail).
- Der ctox-Checkout `~/Documents/ctox` enthält **fremde uncommitted
  Änderungen** (mind. eine weitere aktive Agent-Session committet dort auch,
  z. B. `be70fbb92`). Nichts davon anfassen, nichts „aufräumen", Commits nur
  mit expliziten Pfaden.

## 3 · DEINE AUFGABEN (Reihenfolge = Priorität)

### A · Visuelle QS-Runde über alle 36 Module (sofort startbar)
Nur Stichproben sind visuell geprüft (desktop, threads, credentials, mail).
Öffne jedes Modul im Simulator (Safari → `http://127.0.0.1:8799/business-os/`,
Serve ggf. neu starten) und prüfe: nichts ragt aus dem Viewport, Listen
scrollen, Formulare bedienbar, Sheet-Schließen erreichbar. Befunde als Liste
sammeln; kleine Fixes direkt (CSS im jeweiligen Modul), größere als
Kimi-Nachlauf-Brief (Muster siehe §5). Bekannte, dokumentierte Restlücken:
- SuperDoc-Vendor-Toolbar-Buttons < 44px (vendor/ ist tabu — Entscheidung
  nötig, ob Override per Modul-CSS zulässig ist)
- Browser-Modul: Kontextmenü auf Touch nur Long-Press; Remote-Canvas-Keyboard
  auf Touch ungetestet
- Mail: viewport-basierte statt container-basierte Queries (schmales Fenster
  auf breitem Desktop stapelt nicht)

### B · `shared/window-manager.js`: Drag/Resize mobil deaktivieren
Phase 1 hat Fenster mobil per CSS zu Sheets gemacht; das Drag-JS läuft
wirkungslos mit. Sauberer Fix gehört in `src/apps/business-os/shared/window-manager.js`.
**TRIGGER/BLOCKER:** Die Datei trägt fremde uncommitted Änderungen (Aug 21).
Erst wenn diese committet sind (beobachte `git status`), einen kleinen
Sol-oder-Kimi-Brief schneiden: Pointer-Drag/Resize-Handler unter 768px zu
No-Ops, Desktop unverändert, bestehende Tests (`window-manager.test.mjs`)
grün halten.

### C · Der große Block: native Schale `src/apps/business-os-mobile`
Ziel: CTOX Business OS auf echtem iPhone/Android-Gerät OHNE Mac. Architektur
(vom Operator bestätigt, aus `src/apps/business-os/ARCHITECTURE.md` §Local
Hosting + `src/apps/business-os-desktop/README.md`):
- Dünne native App (Expo/React Native ODER Swift/Kotlin-WebView — Entscheidung
  über Discovery offen): WKWebView/WebView lädt die gebündelte,
  versionsgleiche Shell; Launch-Kontext wird injiziert
  (`instance_id`, `peer_id`, `peer_role`, `sync_room`, `signaling_urls:
  ["wss://signaling.ctox.dev"]`, `transport: "webrtc"`).
- Pairing wie Desktop: `ctox business-os desktop invite --format link`
  (`ctox-business-os-desktop://pair?...`; für Mobile eigenes Schema wählen,
  NIEMALS `ctox:` — gehört dem Daemon). Room-Secret in Keychain/Keystore,
  nie in URLs/Registry. Referenzcode: `src/apps/business-os-desktop/src/common/invites.cjs`,
  `src/main/sources.cjs`, `session-view.cjs`, `bundled-shell.cjs` (insg. ~2.100 Zeilen).
- HTTP nur für lokal gebündelte statische Dateien (WKURLSchemeHandler oder
  Mini-Loopback wie Electron), NIE als Datenpfad.
- Machbarkeit ist BEWIESEN: Shell + RxDB/WebRTC-Sync laufen in iOS-WebKit
  (Simulator-Safari gegen lokale Instanz, 2026-08-24/25).
Vorgehen nach Workjet-Prozess: identischer Discovery-Brief an Prototype
A/Grok, B/Luna, C/GLM (bounded, Prototyp-Pfad z. B. `/tmp`-Expo-App, die einen
hartkodierten Invite konsumiert und die Shell lädt), dann konsolidierter
Produktions-Brief an Sol; Kimi nur für die native UI der Schale (Instanz-Liste,
Pairing-Screen).

### D · Auslieferung des Mobile-Sweeps
Beobachten, wann die Business-OS-Commits in ein ctox-Release gelangen
(`ctox --version`, aktuell 0.3.22-Installation). Nach dem Update: Overlay-Reste
sind automatisch weg; Simulator-Kurztest wiederholen (A-Checkliste, 3 Module).

### E · workjet-Repo-Nebenstrang (nur wenn Kapazität)
T3-Mobile: Dev-/Preview-Varianten zeigen noch T3-Blueprint-Icons/Splash;
Production nutzt CTOX (seit `6d710849c` Turbofan). Falls der Operator will,
dass auch Dev CTOX zeigt: `apps/mobile/app.config.ts` DEVELOPMENT_/
PREVIEW_ASSETS auf die ctox-Ableitungen stellen + nativer Rebuild.

## 4 · Umgebungs-Fallen (jede hat real Zeit gekostet)

1. `~/Documents/ctox/runtime` ist ein **Symlink** auf `~/.local/state/ctox`
   (Live-State!). Quelle der Shell ist `src/apps/business-os/`. Nie in
   runtime/ „Quellen" editieren.
2. **Workjet-Snapshot-Limit:** ctox-Repo-Archiv ≈ 398 MB > 64 MiB → Runs aus
   einem Launchpad-Mini-Repo starten, Brief schickt den Worker per explizitem
   `cd ~/Documents/ctox` in den echten Checkout (dokumentiertes Muster).
   **Launchpad an durablem Ort anlegen** (`~/.local/state/workjet-launchpads/…`),
   NICHT im Session-Scratchpad — sonst nach Neustart `workspace_rejected`
   bei result import (Learning ist in Workjet gespeichert).
3. **Kimi-Runs sterben am 200k-Kontextlimit** des Claude-Code-Harness für
   Fremdmodelle — immer erst spät im Lauf. Gegenmittel: ≤6 Module pro Brief,
   „Bericht KOMPAKT" erzwingen, Arbeit vor Bericht priorisieren. Nach jedem
   Fehlschlag: Working-Tree prüfen — die Arbeit ist meist trotzdem da
   (per mtime + `{`/`}`-Bilanz + `git diff` verifizieren).
4. **CocoaPods braucht `LANG=en_US.UTF-8`**, sonst „Unicode Normalization
   not appropriate for ASCII-8BIT" hinter einem Error-Report-Crash.
5. **Fremder Prozess-Killer:** Im Aug-24-Zeitraum wurden frisch gestartete
   `node apps/server/src/bin.ts serve`-Prozesse (workjet-Repo) maschinenweit
   per SIGKILL beendet — Verursacher unidentifiziert (Verdacht: fremde
   ChatGPT-Agent-Session, cwd ~/Documents/ctox; Operator-eslogger stand aus).
   Wenn Server still sterben: erst `ps` nach fremden Agenten, dann debuggen.
6. `workjet health --probe-workers --json` VOR jeder Verfügbarkeitsaussage;
   Kimi-Provider erlaubt max. 3 parallele Runs.
7. Port 8081 gehört ctox-real, 8765 einem fremden Python-Prozess, 9300 der
   laufenden Desktop-App (CDP — nicht anfassen, Absprache mit der
   Desktop-Session „workjet-0c" über /tmp/cc-socks/62630.sock).
8. Simulator-Koordinaten: Screenshots sind größer als der 402×874-Punktraum —
   Bildpixel × (402/Bildbreite) umrechnen, sonst gehen Taps daneben.

## 5 · Arbeitsregeln für Worker-Briefs (bewährt in 6 Runs)

Jeder Brief enthält: Arbeitsort-Block (Launchpad-Hinweis + `cd`), ein
Objective, harte Datei-Whitelist, Verbote (shared/, vendor/, runtime/,
Sync/RxDB/WebRTC, kein Build-Tooling, keine neuen HTTP-Pfade), die Regel
**„JS/HTML nur ändern, wenn `git status --short -- <datei>` leer ist"**
(der Checkout trägt fremde uncommitted Änderungen), Akzeptanzkommandos
(`node --check` je JS), KOMPAKT-Bericht, kein git add/commit durch Worker,
keine Subagenten, Receipt-Pflicht. Integration immer selbst: Diff prüfen,
mtime-Abgrenzung, Balance-Check, explizite Pfad-Commits, dann
`workjet result import` + `runs mark integrated`.

## 6 · Evidenz-Karte

- Kimi-Run-IDs (alle Worker „Kimi · UI/UX", Briefs waren im Scratchpad und
  sind mit dem Neustart verloren — Inhalte sind durch §5 reproduzierbar):
  Phase 1 `…103b569e` (integriert); Phase 2 A `…a5e2c5ac` (completed,
  Lifecycle-Markierung wegen Launchpad-Verlust nicht möglich), B `…9c9400eb`
  (failed→abandoned, calendar+notes übernommen), C `…af51eba9` (completed,
  Markierung s. A), B2 `…ca1e2594` (integriert), D `…d60080f7` (integriert),
  E `…84917b65` (failed→abandoned, Arbeit vollständig übernommen).
- Overlay-Backups: `~/.local/state/ctox-mobile-handover/bos-shell-backup/`
- T3-Testumgebung (workjet-Strang): Wegwerf-Backend-Basis lag im Session-
  Scratchpad und ist weg — bei Bedarf neu nach Skill `test-t3-mobile`;
  `serve` braucht `WORKJET_PROVIDER_GATEWAY_HOST_EXECUTABLE=
  <workjet-repo>/native/provider-gateway-workjet-host/target/release/workjet-provider-gateway-host`.
- Koordination: Desktop-Session „workjet-0c" (SendMessage an
  `uds:/tmp/cc-socks/62630.sock`) hält apps/desktop, apps/web, apps/server,
  native/provider-gateway*; Mobile-/Business-OS-Pfade gehören diesem Strang.

## 7 · Geklärte Rückfragen

Diese Antworten sind Arbeitsanweisungen für die Fortsetzung. Nur Punkt 7
benötigt eine Entscheidung des Operators.

1. **SuperDoc-Toolbar per CSS-Override: erlaubt.** Der Override gehört in
   `modules/documents/index.css`, muss streng auf den Documents-Container
   begrenzt bleiben und darf `vendor/` nicht verändern. Das bewusst akzeptierte
   Risiko: Ein Vendor-Update kann Klassennamen ändern. Den Override deshalb mit
   einem kurzen Kommentar auf die verwendete SuperDoc-Version datieren.
2. **Mail-Lücke: nur dokumentierter Folgepunkt.** Die viewport-basierten statt
   container-basierten Queries werden in dieser mobilen QS-Runde nicht geändert,
   weil ein mobiles Sheet die volle Viewport-Breite nutzt. Falls die QS zeigt,
   dass auch das mobile Sheet betroffen ist, die Lücke hochstufen und als
   eigenen kleinen Brief schneiden.
3. **Einmaliger Secret-Transport im Deep Link: erlaubt und etabliert.** Beim
   Import das Room-Secret sofort in Keychain/Keystore verschieben und die
   Link-Payload danach nirgends speichern. Es darf insbesondere nicht in
   persistierten URLs, Registry, Logs, Verlauf, Screenshots, Berichten oder
   Receipts auftauchen. QR-Scan oder Einfügen bevorzugen; nach einem
   Pasteboard-Import die Zwischenablage bereinigen. Invite-TTL verwenden.
4. **Mobile-URL-Schema:** `ctox-business-os-mobile://pair` ist korrekt und frei.
   `ctox:` bleibt dem Daemon vorbehalten. `ctox-desktop*` und `ctox-mobile*`
   gehören zur Workjet/T3-Produktlinie und sind wegen Verwechslungsgefahr tabu.
5. **Discovery und QS parallel ausführen.** Sobald der byte-identische
   Discovery-Brief steht, Grok, Luna und GLM in getrennten
   Wegwerf-Prototyp-Verzeichnissen starten. Der Brief enthält mindestens den
   WebKit-Beweis, das Launch-Kontext-Format, den Invite-Referenzcode und die
   Verbote aus §3C. Die Prototypen sind keine Produktionslieferung und dürfen
   den Checkout nicht verändern. Die QS-Runde läuft währenddessen weiter.
6. **Kanonische QS-Liste:** aus
   `src/apps/business-os/system-apps.json` und
   `src/apps/business-os/modules/registry.json` ableiten. Reihenfolge nach
   Gewicht: zuerst die zehn System-Apps (`desktop`, `ctox`, `tickets`,
   `threads`, `knowledge`, `browser`, `credentials`, `app-store`, `creator`,
   `reports`), dann die geschäftskritischen Module (`mail`, `conversations`,
   `customers`, `invoices`, `buchhaltung`, `documents`), danach der Rest. Pro
   Modul Viewport, Scrollen, Formulare/Tap-Ziele und Sheet-Bedienung prüfen.
   Ein geprüfter Leerzustand zählt bei leeren Operator-Modulen als QS.
7. **OPERATOR-ENTSCHEIDUNG — Aufgabe B bleibt blockiert.** Die fremden
   uncommitted Änderungen an `shared/window-manager.js` und den übrigen
   Aug-20/21-Dateien weder committen noch verwerfen. Michael lässt sie von der
   verursachenden Session landen oder ausdrücklich verwerfen. Erst danach darf
   der Mobile-No-Op-Fix aus Aufgabe B beginnen.

## 8 · Fortsetzung am 2026-08-25

### A · Mobile-QS abgeschlossen

Die vier Mobile-Commits betreffen **36**, nicht 32, eindeutige Module. Die
ältere Zahl in §2/§3 wurde korrigiert. Geprüft wurde bei 402×874:

- Live in der Operator-Shell: `desktop`, `ctox`, `tickets`, `threads`,
  `knowledge`, `browser`, `credentials`, `app-store`, `reports`, `mail`,
  `coding-agents`, `appsec-pentest`; Mail zusätzlich in Safari auf dem iOS-26.5-
  Simulator.
- Die übrigen bzw. nicht direkt aus dem Launcher erreichbaren Oberflächen als
  statisches Modul-Markup mit den echten Shell-/Base-/Modul-Styles:
  `buchhaltung`, `calendar`, `consent`, `conversations`, `creator`, `customers`,
  `cv-print-builder`, `documents`, `esign`, `importer`, `intake`, `interviews`,
  `invoices`, `iot`, `matching`, `nachweise`, `notes`, `outbound`, `placements`,
  `research`, `shiftflow`, `spreadsheets`, `submissions`, `support`.

Ergebnis: kein Seitenüberlauf; Sheet-Schließen blieb erreichbar. Die zunächst
auffälligen Überbreiten in `matching`, `notes` und `shiftflow` sind erwartete,
intern geschlossene Side-Panes bzw. eigene horizontale Scroll-Owner. Leere und
statische Ladezustände wurden als solche gewertet. Die statische Prüfung ersetzt
bei datenabhängigen Detailansichten keinen späteren echten Geräte-E2E-Test.

Der bestätigte SuperDoc-Befund wurde als erlaubter Modul-CSS-Fix behoben:
`modules/documents/index.css` überschreibt unter `pointer: coarse` die 32px-
Toolbar von **SuperDoc 1.32.0** auf 44px, streng auf
`.documents-superdoc-toolbar` gescoped. `vendor/superdoc.{css,mjs}` blieb
byte-identisch. Ein Guard in `documents.test.mjs` schützt Scope, Version-Kommentar
und Zielgröße.

Verifikation:

- `node --test src/apps/business-os/modules/documents/documents.test.mjs` —
  37/37 grün; nur bereits vorhandene Duplicate-Key-Warnungen aus
  `vendor/document-format.mjs`.
- `node --test src/apps/business-os/modules/documents/documents-layout.browser.test.mjs`
  — 1/1 grün.
- `git diff --check` — grün.

Offen dokumentiert, nicht Teil dieses Fixes: Browser-Long-Press/Remote-Keyboard,
die nur im schmalen Desktop-Fenster relevante Mail-Container-Query und ein in
der Live-Shell wiederholt sichtbarer `ctox/index.js`-Fehler beim Zugriff auf
`executionPhase`. Letzterer liegt außerhalb des CSS-Sweeps und die betroffene
JS-Arbeit ist fremd/dirty.

### B · Weiter blockiert

`shared/window-manager.js` und `shared/window-manager.test.mjs` sind weiterhin
fremd verändert. Keine Änderung und kein Commit durch diesen Strang.

### Koordination mit der parallelen „CTOX Desktop app"

Der Workjet-Desktop-Worker hat seinen getrennten Packaging-Scope als
`c805e3e35` abgeschlossen. Er änderte keine Datei unter diesem CTOX-Checkout,
keine Mobile-/Contract-/Pairing-Datei und meldete Port 9300 wieder laufend. Es
gab keine Überschneidung.

### C · Discovery-Panel ausgewertet

Byte-identischer Brief und Launchpad:
`~/.local/state/workjet-launchpads/ctox-business-os-mobile-discovery-20260825/`
(Brief-Commit `9fb8c27`).

- **Grok 4.6:** nicht gestartet; zwei Health-Probes scheiterten reproduzierbar
  mit `API Error: 400 unknown provider for model grok-4.6`. Kein Ersatzmodell
  wurde stillschweigend verwendet.
- **Luna**, Run `local-2026-08-25T092649Z-03a6c0ab-f922-40cd-a56a-9ab698d728de`:
  isoliert und 8/8 eigene Tests grün, aber als Produktionsgrundlage verworfen
  und `abandoned` markiert. Der Prototyp erfand `room`/`secret`/numerische TTL
  statt des Desktop-v1-Vertrags und gab dem bestehenden WebRTC-Client das
  Room-Passwort nicht; damit wäre kein echter Sync möglich.
- **GLM**, Run `local-2026-08-25T092649Z-0dfc4f86-07c8-4cae-a2be-e28965007b85`:
  Ergebnis importiert und `integrated`. Wegwerf-Prototyp unter
  `/tmp/ctox-business-os-mobile-discovery.DPe9dx`; 9/9 Tests grün, Swift-App
  mit Xcode 26.6 gebaut und im iOS-26.5-Simulator per synthetischem Deep Link
  gepairt. Korrektes Invite-v1-Schema, Keychain-Grenze, opaque Registry-Refs
  und IndexedDB-Persistenz über Relaunch wurden belegt. Android ist nur als
  Quellarchitektur vorhanden, da Java/SDK/Gradle fehlen.

Konsolidierte Empfehlung: dünne native Hosts — SwiftUI/WKWebView plus
`WKURLSchemeHandler` und Kotlin/WebView plus `WebViewAssetLoader`; kein Expo/RN.
Ein wechselnder Loopback-Port setzte im Versuch den IndexedDB-Zähler zurück,
weil der Origin wechselte. Der Custom-Scheme-Handler behielt den Store und ist
auf iOS die Primärroute; Loopback bleibt höchstens ein stabil-portiger Fallback.

Pflichtkorrekturen für den Produktionsbrief gegenüber dem GLM-Prototyp:

1. Das Room-Passwort nach Keychain/Keystore-Rücklesen nur in-memory und am
   Dokumentstart injizieren bzw. beim Ausliefern von `index.html` einbetten;
   **kein erneutes `ctox_config` mit Secret im Shell-URL-Query**.
2. Desktop-v1-Felder exakt spiegeln (`display_name`, `sync_room`,
   `signaling_room_password`, ISO-`expires_at`, optional Session-/Capability-
   Material); ein gemeinsamer Fixture-Korpus muss JS, Swift und Kotlin binden.
3. Android-Build und echter WebRTC/IndexedDB-Smoke sind ein Akzeptanzkriterium,
   nicht nur eine Quellcode-Notiz.
4. Die reale Shell umfasst ca. 289 MB/2058 Dateien; davon entfallen ca. 217 MB
   auf `vendor/ctox-office`. Dies ist vor der Produktionsimplementation als
   Paketierungsentscheidung zu klären.

### Neue Operator-Entscheidungen vor dem Sol-Produktionslauf

1. **Shell-Paket:** Soll v0 die vollständigen ca. 289 MB bündeln, oder darf
   `vendor/ctox-office` als On-Demand-Resource/Play-Asset-Pack ausgelagert
   werden? Empfehlung: schlanke Basis plus signiertes, versionsgleiches
   On-Demand-Paket; andernfalls ist die Store-/Download-Größe ein hohes Risiko.
2. **Plattform-Floor/Isolation:** Darf v0 iOS 17 voraussetzen und Android-Geräte
   ohne `MULTI_PROFILE` zunächst auf genau eine gepairte Instanz begrenzen?
   Empfehlung: ja; damit bleibt die IndexedDB-Isolation fail-closed, statt
   Instanzdaten in einem gemeinsamen Profil zu vermischen.
