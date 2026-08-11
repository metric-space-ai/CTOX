# Produktmessung M1–M5 — erste Fakten (12.08.2026, 21:00–21:35)

Messaufbau: isolierter Root `/Volumes/tmp/ctox-pipeline/m1-root` (CTOX_ROOT),
Debug-Binary aus `b9ed00757`-Nachfolge, `ctox business-os serve --addr
127.0.0.1:8917`, Browser = Claude-Browser-Pane, Shell mit `?rxdbSmoke`.
Die Produktivinstanz des Owners wurde nicht berührt. Release-Nachmessung
läuft (Debug-Zahlen sind obere Schranken, die Struktur der Befunde ist
davon unabhängig).

## M1 Kaltstart (Daemon-Start → HTTP antwortet)

| Lauf | Sekunden |
|---|---|
| Erststart, leerer Root (einmalig, inkl. Store-Anlage) | ~88 |
| Neustart 1 | 8,3 |
| Neustart 2 | 27,0 |
| Neustart 3 | 23,9 |
| Neustart 4 | 13,5 |
| Neustart 5 | 15,5 |

**Median 15,5 s, Streuung 8–27 s** — bei praktisch leerem Store. Der Server
bindet den Port absichtlich erst nach Store-Parse und Peer-Vorbereitung.
Das „Peer bereit“-Kriterium der Messreihe war unbrauchbar (`replicationUp`
wird erst mit verbundenem Browser wahr); nicht gewertet.

## M2 Browser-Boot (Navigation → Collections live; leerer Store)

- Shell DOM-interaktiv: **1,75 s**
- Alle 15 Boot-Collections initial repliziert: **15,9 s**
- Verursacher: `desktop_icons` allein **11,1 s** Erstreplikation (bei ~null
  Daten); die übrigen Collections starten **seriell im ~1-s-Takt**
  (serialisierte Start-Queue), Fertigstellungszeiten 6,8 / 8,8 / 10,8 /
  11,0 / 11,8 / 12,8 / 15,9 s.

Die vom Owner benannten „vielen Sekunden“ sind damit lokalisierbar:
serielle Collection-Starts plus ein einzelner Ausreißer.

## M3 Command-Bus — DEFEKT, Latenz nicht messbar

30 Dispatches `ctox.provider_subscription.status` über den offiziellen
Modul-Command-Bus (`createModuleContext({id:'ctox',
collections:['business_commands']})`): **alle scheitern deterministisch** mit

```
QUERY_NOT_SUPPORTED / SQLITE_QUERY_STREAM_UNSUPPORTED
```

Wurzel: `src/core/rxdb/src/storage/sqlite/instance.rs` (~1613) verweigert
auf dem WebRTC-Query-Fetch-Hotpfad jede Mango-Query, die nicht nach SQL
kompilierbar ist („refusing Rust matcher fallback“); der Tracking-Pfad des
Command-Bus stellt genau so eine Query. Die Command-Plane-Diagnostik zählt
`counters: {}` — **in dieser frischen Instanz ist nie ein Command
durchgekommen.** p50/p95 sind erst nach dem Fix messbar.

Offene Einordnung: ob der reguläre Modul-UI-Pfad dieselbe Query stellt
(dann Totalausfall des Command-Plane auf frischen Instanzen) oder nur der
Smoke-Kontext, ist der erste Prüfschritt des Fixes.

## M4 Abbruch-Churn (Zyklus 1 von 10)

Daemon hart getötet, 20 s Ausfall, Neustart (~18 s bis HTTP). Browser:
15 Collections getrennt → kontinuierlicher Wiederaufbau → **~30 s nach
Serverrückkehr alle wieder verbunden**. Kein hängender Zustand, Room-Circuit
blieb geschlossen, `lastError` null. (Vier Collections melden dauerhaft
Status `reused` — das ist Bridge-Wiederverwendung, kein Fehler.)
Ausstehend: 9 weitere Zyklen, Schreib-vor-Kill-Datenverlustprüfung.

## M5 Multi-User — ausstehend

Braucht zweite echte Nutzersitzung (Invite-Flow); Multi-Tab-Sync wäre nur
ein Ersatzmaß.

## Vorläufige Produkt-Einordnung (nur Fakten)

- Boot bis „alles live“ auf leerem Store: ~16 s → für ein Desktop-OS-Gefühl
  zu langsam; Hebel sind benannt (serielle Starts, desktop_icons).
- Command-Plane auf frischer Instanz: defekt (M3).
- Abbruch-Erholung: funktioniert, ~30 s (Zyklus 1).

Produktnote bleibt bis zum vollständigen M1–M5-Satz offen; nach aktueller
Faktenlage wäre sie schlecht („funktioniert, aber langsam, ein Kernpfad
defekt“). Repo-Note (B) und Produktnote sind getrennte Größen.
