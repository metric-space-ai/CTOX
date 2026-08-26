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

## Aktueller Stable v0.1.4

Der aktuelle signierte Stable-Pointer zeigt auf `business-os-shell-v0.1.4`
und CTOX-Commit `c2d25b33115311ac45162df58a7d9e09686da4af`.
GitHub-Actions-Lauf `32917548865` hat Build, fokussierte Artefakttests,
SHA-256-Prüfung, SPDX-SBOM, Signatur, Release-Upload und Pointer-Publikation
erfolgreich abgeschlossen.

- Veröffentlichung: `2026-08-26T03:02:40+02:00`
- Release-Manifest-SHA-256: `a9bdcc7cabea6498f75d1003ac44f5977e97fa23fbb91ddaa3c1cfa1684bb4cf`
- Artefakt-SHA-256: `89875506f7de123c47497df5c6936838a53288eb05ce162097f65c0e5b7044ac`
- Artefaktgröße: 122281365 Bytes
- Inventar: 1651 reguläre Dateien
- Signing-Key-ID: `shell-current-2026-08`
- Kompatibilität: Workjet ab 0.0.33, CTOX ab 0.3.22,
  `workjet.business-os-shell.v1`

Die realen Workjet-CDP-Proben der Zwischenstände fanden und korrigierten drei
Releasefehler: `shared/shell-release-status.js` wurde vom generischen
`release`-Dateifilter erfasst, `mobile-host.css/js` fehlten in der Root-
Inventarliste, und statisch gehostete Workjet-Shells lösten ihre bereits
vorhandene kurzlebige Desktop-Einladung nicht vor dem Hosted-Control-Fallback
auf. v0.1.4 ist der erste Stable, der diese drei Korrekturen gemeinsam enthält.

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

`installed-modules/**` gehört nicht in ein globales Shell-Release. Die reale
v0.1.4-Probe zeigt deshalb bei katalogisierten Instanzmodulen ohne separat
bereitgestellten Code einen klaren 404 statt heimlich Kundenmodule in Stable zu
veröffentlichen. Der professionelle Folgepfad ist ein eigener signierter,
versionsgebundener Module-Pack-/Resolver-Vertrag mit derselben fail-closed
Trust-Strategie. Bis dieser existiert, gilt die globale Shell als abgenommen,
instanzspezifische Offline-/Restart-Parität jedoch ausdrücklich nicht.
