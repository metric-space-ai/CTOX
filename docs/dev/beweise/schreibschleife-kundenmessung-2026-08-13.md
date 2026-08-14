# Schreibschleife: Vorher/Nachher auf dem Kundensystem (13.08.2026)

Erste faktisch beim Kunden wirksame Reparatur dieser Kampagne. Alle Zahlen
selbst erhoben, beide Messungen unter identischer Bedingung.

## Messverfahren (von der Parallelsitzung übernommen, wortgetreu)

```bash
D=~/.local/state/ctox/business-os-rxdb.sqlite3
T=ctox_business_os__business_commands__v1
Q="SELECT SUM(CAST(substr(revision,1,instr(revision,'-')-1) AS INTEGER)) FROM $T;"
a=$(sqlite3 "file:$D?mode=ro" "$Q"); sleep 60
b=$(sqlite3 "file:$D?mode=ro" "$Q"); echo "Zuwachs: $((b-a)) in 60s"
```

Zwei Bedingungen, ohne die die Zahl wertlos ist:
1. **SUMME** über alle Dokumente, nicht MAX — MAX zeigt nur das lauteste.
2. `replicationUp` bei **jedem** Messpunkt mitschreiben. Bei `true` zählt man
   Browser-Nutzlast mit. Beide Messungen unten liefen bei `false`.

## Ergebnis

| Zeitpunkt | replicationUp | Revisionen/min |
|---|---|---:|
| unmittelbar vor dem Austausch | false / false | **174** |
| unmittelbar nach dem Austausch | false / false | **0** |
| Kontrolle, 15 min später | false / false | **0** |

Zum Vergleich die älteren Basiswerte derselben Messung: 92,0/min
(Parallelsitzung) und 93,0/min (eigene Messung) — die Schleife hatte bis zum
Austausch also noch zugelegt.

Zählerstand blieb im Nachher-Fenster exakt stehen: a = b = 3.291.459.

## Was ausgeliefert wurde

- Binary aus `main` (Stand mit AP1/AP3/AP4 und dem Schleifenfix), gebaut auf
  Linux/x86_64 (glibc 2.35 — Zielsystem hat 2.39).
- SHA-256 (Präfix) `2204dcea23f94b438c899dd2dae9bc61`, identisch an drei
  Stellen geprüft: Bau-Maschine, lokale Kopie, Zielort auf der Kundenmaschine.
- Vorheriges Binary vom 10.08. gesichert als
  `/home/ctox/ctox-real.sicherung-20260810` (Rückweg vorhanden).
- Dienst gestoppt, getauscht, gestartet; Rollback-Zweig war vorbereitet und
  wurde nicht gebraucht: `systemctl --user is-active ctox.service` = `active`.
- Peer nach dem Neustart: `running true`.

## Warum die Instanz das vorher nicht hatte

Das Daemon-Binary auf der Kundenmaschine stammte vom **10.08.** — vor allen
Schnitten und Fixes. Jeder Commit im Repo war dort ohne Wirkung. Das ist die
dritte Stufe der Beweiskette („auf der Kundeninstanz wirksam"), die bis heute
offen war.

## Offen

- Die 2,29-GB-Store-Datei schrumpft dadurch **nicht** von selbst; die
  Revisionsgeschichte bleibt. Kompaktierung ist ein eigener Schritt.
- Kontrollmessung 15 min nach dem Neustart: **0/min**, und der Zählerstand
  war unverändert **3.291.459** — derselbe Wert wie unmittelbar nach dem
  Austausch. In über 15 Minuten also **kein einziger** Schreibvorgang auf
  diesen Dokumenten. Damit ist ausgeschlossen, dass die Null nur eine
  Ruhephase direkt nach dem Neustart war.
