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
