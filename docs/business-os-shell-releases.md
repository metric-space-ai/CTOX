# Business OS shell releases

Stand: 2026-08-26

Business-OS-Shells werden unabhängig vom CTOX-Backend als unveränderliche
Release-Artefakte veröffentlicht. Stable-Tags folgen
`business-os-shell-v<SemVer>`; Vorabversionen werden den Kanälen `beta` oder
`nightly` zugeordnet.

## Vertrauenskette

Der Release-Workflow baut ein deterministisches USTAR/Gzip-Artefakt, prüft den
abgelösten SHA-256, erzeugt eine vollständige SPDX-2.3-Dateiliste und signiert
anschließend zwei Dokumente unabhängig voneinander:

- `ctox.business-os-shell.release.v2`: unveränderliches Release-Manifest mit
  Artefakt, Dateiliste, Kompatibilität, Provenienz und SBOM-Verweis.
- `ctox.business-os-shell.channel.v1`: kleiner signierter Pointer auf das
  unveränderliche Manifest des aktuellen Kanals.

Signaturen sind Ed25519 über die UTF-8-Bytes der kanonischen JSON-Nutzlast ohne
das Feld `signature`. Der private current-Key und der vorbereitete next-Key
liegen ausschließlich als GitHub-Repository-Secrets. Clients bündeln genau
die beiden zugehörigen öffentlichen SPKI-Keys. Ein unbekannter Key, eine
ungültige Signatur, ein abweichender Manifest-Hash oder eine inkompatible
Version muss geschlossen fehlschlagen.

## Stable v0.1.0

Der unveränderliche Tag `business-os-shell-v0.1.0` zeigt auf CTOX-Commit
`a79a83ce7`. Er enthält die konsolidierte Shell ohne zweite Desktop-Theme-
Schicht, die kurze Shell-Version mit Statusanzeige sowie den begrenzten,
inhaltsbreiten Chat-Dock.

GitHub-Actions-Lauf `32909885557` wurde am 2026-08-26 angestoßen. Solange der
Lauf keinen erfolgreichen Abschluss besitzt, darf der Stable-Channel-Pointer
nicht als veröffentlicht gelten und Workjet darf seinen Recovery-/RC-Pin nicht
auf v0.1.0 umstellen.

## Lokale Verifikation

`npm run test:shell-artifact --prefix src/apps/business-os` prüft
deterministischen Build, Pfad-/Symlink-/Budgetgrenzen, atomare Publikation,
SPDX-Inventar sowie Release- und Channel-Signaturen. Am 2026-08-26 waren alle
12 Tests grün. Der Chat-/Chrome-Slice wurde zusätzlich mit dem statischen
Layout-Guard, 49 Business-Chat-Verhaltenstests und zwei Shell-Status-Tests
geprüft.

## Aktueller Stable v0.1.6

Der aktuelle signierte Stable-Pointer zeigt auf `business-os-shell-v0.1.6`
und CTOX-Commit `cf2c63285f844acc73b82bca429e907d7faf5687`.
GitHub-Actions-Lauf `32926668617` hat Build, fokussierte Artefakttests,
SHA-256-Prüfung, SPDX-SBOM, Signatur, Release-Upload und Pointer-Publikation
erfolgreich abgeschlossen.

- Veröffentlichung: `2026-08-26T05:29:12+02:00`
- Release-Manifest-SHA-256: `1ca9bc3ffa4f2967ec2c02d4378e89afddaf5e18deb2e4f0ce513e32cde9089b`
- Artefakt-SHA-256: `cca3562e7860b5fdaf95ab5cc724399448d10d847bd3eedcdf4ad725e5e02ec9`
- Artefaktgröße: 122282535 Bytes
- Inventar: 1651 reguläre Dateien
- Signing-Key-ID: `shell-current-2026-08`
- Kompatibilität: Workjet ab 0.0.33, CTOX ab 0.3.22,
  `workjet.business-os-shell.v1`

Die realen Workjet-CDP-Proben der Zwischenstände fanden und korrigierten drei
Releasefehler: `shared/shell-release-status.js` wurde vom generischen
`release`-Dateifilter erfasst, `mobile-host.css/js` fehlten in der Root-
Inventarliste, und statisch gehostete Workjet-Shells lösten ihre bereits
vorhandene kurzlebige Desktop-Einladung nicht vor dem Hosted-Control-Fallback
auf. v0.1.4 war der erste Stable mit diesen drei Korrekturen. v0.1.5 ergänzte
die Shell-Kontinuität über Backend-Neustarts; v0.1.6 ergänzt einen begrenzten
Recovery-Heartbeat für einzelne Collections, deren einmaliger Reconnect-Timer
bereits verbraucht wurde.

## Reale v0.1.6-Abnahme

Die lokale CTOX-Instanz wurde über `check`, `stage` und `activate` atomar von
v0.1.5 auf v0.1.6 aktualisiert. Status danach: Stable, `current`, `healthy`,
administrierbar, kein Recovery-Fallback. Workjet löste exakt diese aktive
Version aus seinem verifizierten Versionscache auf.

Unter laufender v0.1.6-Shell wurde der CTOX-Service anschließend neu gestartet.
Alle zehn aktiven RxDB/WebRTC-Collections wechselten ohne manuellen Eingriff in
14 Sekunden auf dieselbe neue native Peer-Session; insbesondere blieb
`ctox_queue_tasks` nicht mehr auf der vorherigen Session hängen. Danach wurde
eine neue Workspace-Branding-Änderung über den normalen Business-Command-Pfad
projiziert und wieder auf `Meridian Supply Co.` zurückgesetzt. Beide Commands
endeten `completed`; es wurde kein HTTP-Business-Datenfallback verwendet.

Die fokussierte lokale Suite war mit 65 Shell-/Chat-/Reconnect-Tests und zwölf
Artefakt-/SBOM-/Signaturtests grün. Im realen Workjet-CDP-Target wurden keine
neuen Console-, Page-, Request- oder HTTP-Fehler beobachtet.

## Instanz-Lifecycle

CTOX speichert Shell-Zustand in der vorhandenen SQLite-Runtime: aktiver Kanal,
aktive/gewünschte/angebotene Version, current-/previous-Slot, letzte Prüfung,
letzte erfolgreiche Aktivierung, Health, Phase, Fehler und Rollbackstatus.
`ctox business-os shell-update status|check|stage|activate|rollback` verwendet
denselben Lifecycle wie die authentifizierten Control-Plane-Routen.

Stage und Activate prüfen Pointer/Manifest, Ed25519-Key-ID/-Signatur,
Kompatibilität, Archivgröße/-SHA-256, vollständige Dateiliste und Smoke-Test.
Der neue Slot wird erst danach atomar aktiviert. Unterbrochene Downloads,
ungültige Dateien oder ein fehlgeschlagener Smoke-Test verändern den aktiven
Slot nicht; der vorherige Slot bleibt für Rollback erhalten. Backend-Update und
Shell-Update sind absichtlich getrennte Operationen. Die Shell-Artefakte sind
statische Distribution; Business-Daten bleiben ausschließlich auf dem
RxDB/WebRTC-Pfad.

## Abgrenzung instanzspezifischer Module

`installed-modules/**` und `local-modules/**` gehören nicht in ein globales
Shell-Release. Workjet löst diese Assets nur für die lokale Instanz aus deren
bestehendem CTOX-Runtime-Root auf. Fehlt ausschließlich `icon.svg`, liefert der
Desktop-Host ein neutrales Workjet-App-Raster; fehlender Modulcode oder ein
fehlendes Schema bleibt fail-closed. Die drei real fehlenden Icons von
`vite-imported-app`, `vite-react-demo` und `vite-react-typescript-starter`
antworteten in der v0.1.6-Abnahme jeweils mit 200 und `image/svg+xml`, ohne
instanzspezifischen Code in Stable einzubauen.
