# Native Shell-Generationen: reproduzierbare Regression

Stand: 2026-09-06. **Offener Fehler, keine Produktionsfreigabe.**

## Ergebnis

Ein echter lokaler CTOX-HTTP-Server liefert einem bereits geladenen Dokument A
nach einem Wechsel der Quelldateien und Serverneustart das RxDB-Bundle B.
Im Browserlauf `shell-generation-native-browser-20260906-r3` trat das bei
**10 von 10 Dokumenten** auf. Alle Bundle-Antworten waren HTTP 200; es gab
keinen JavaScript-Pageerror. Der Browser kann also eine falsche Kombination
laden, ohne dass der Transport oder der Import einen Fehler meldet.

Der Test verwendet den unveränderten kanonischen `shared/rxdb-runtime.js`
aus diesem Checkout und das reale `rxdb/dist/ctox-rxdb-js.mjs`. Ausschließlich
in den privaten Testkopien wird dem Bundle ein Export mit A beziehungsweise B
angehängt. Ein kleines Testdokument löst den echten dynamischen Import erst
nach dem Wechsel aus. Die manuelle APP_BUILD-Kennung bleibt bei A und B gleich,
wie bei den zuvor beobachteten unterschiedlichen Releases mit derselben Kennung.

**Dieser Nachweis benötigt ctox.dev nicht.** Der temporär wiederhergestellte
Proxy-Fallback bleibt eine weitere Vertragsverletzung, aber sein Entfernen
allein repariert keine fehlende Bindung der nativen Abhängigkeiten.

## Genaue Reichweite

- Echte Chromium-ESM-Ausführung und echter nativer `serve_static`-Pfad.
- Zwei private Quellbaum-Fixtures; der Test stoppt und startet ausschließlich
  seinen eigenen nativen Prozess und erhält die alten Browserdokumente.
- Die HTML-Testseite wird als eigene statische Datei ausgeliefert. Sie fordert
  keinen personalisierten Launch-Kontext an und öffnet keine Business-Datenbank.
- Kein signierter Slot-Wechsel, keine Authentifizierungs-, Sync-, Mobil- oder
  vollständige Business-OS-Abnahme. Diese Prüfungen bleiben erforderlich.
- Native Binärdatei: `/Users/michaelwelsch/.local/bin/ctox-real`, SHA-256
  `f4f1a734e7b4858467cb09d235de68ba7c6831c5e4a6615d23c922ecab59def5`.
  Das war eine bereits installierte ältere Binärdatei, **kein Build des aktuellen main**.
- Loader-SHA-256:
  `d9869e219c44ddd18ee959b8fb026b70813a58849d5feba56fe7654ffe7f7301`.
- Der aktuelle native Quellcode lässt den betroffenen `rxdb/`-Pfad ebenfalls
  außerhalb des Generation-Guards; der Browserlauf ersetzt dennoch keinen
  Lauf mit einer daraus neu gebauten Binärdatei.

## Wiederholen

Im Repository, mit expliziter nativer Binärdatei und eigenem Ausgabeverzeichnis:

```sh
greppy bash-smart -- npm run qa:shell-generation-native --prefix src/apps/business-os -- /absolute/path/to/native-ctox /absolute/path/to/evidence
```

Das Programm übernimmt keine Operator-Credentials oder Laufzeitkonfiguration.
Es erstellt einen eigenen temporären CTOX-Root, nutzt einen freien Loopback-Port
und beendet seine nativen Prozesse sowie den eigenen Chromium am Ende.
Launcher-Skripte werden abgelehnt, weil sie einen anderen Root erzwingen können.
Der Test ist für Unix-Prozessgruppen ausgelegt.

Es speichert `result.json`, `trace.zip` und
`old-document-after-switch.png`. Der Gate-Exitcode bleibt bei einer gemischten
Generation **1**. Er darf nicht auf „bekannter Fehler ist grün“ umgestellt werden.
Kontrollläufe mit vollständig frischen A- und B-Dokumenten unterscheiden einen
kaputten Testaufbau vom eigentlichen Wechselproblem.

Die lokale Evidenz des ersten abgeschlossenen Wechsellaufs liegt unter:
`/Volumes/tmp/dev-artifacts/ctox/sync-core-offensive/shell-generation-native-browser-20260906-r3/`.

Der abschließende Lauf mit Kontrollfällen liegt unter
`/Volumes/tmp/dev-artifacts/ctox/sync-core-offensive/shell-generation-native-browser-20260906-r5/`:
frisches A lädt A, frisches B lädt B, alle zehn offengebliebenen A-Dokumente
laden B. Beide Kontrollfälle bestehen; der Wechsel-Gate endet mit Exit 1.
Der native Testprozess beendet sich anschließend mit Exit 0. Der Trace und
die Screenshots verbleiben als lokale Evidenz; Testcode und dieser Befund
werden auf main versioniert.

## Messungen und weiterer Befund

Der Lauf r3 maß vom Klick-Handler bis zum Ende des dynamischen Imports bei zehn
alten Dokumenten p50 **107,6 ms**, p95 **362 ms**. Das sind Zeiten eines
**fachlich falschen** Imports im lokalen Test, keine Erfolgsmessung, kein
Command-p50 und kein Boot-p95. Die vollständigen Einzelwerte stehen im JSON.
Im abschließenden Lauf r5 betrugen dieselben Importmetriken bei n=10
p50 **203 ms** und p95 **449,6 ms**. Sie bleiben Messungen der Regression.

Zwei frühere Varianten mit personalisiertem `index.html` erreichten den Wechsel
nicht: r1 lief beim Navigieren in 30 Sekunden ins Timeout; in r2 antworteten
zwei HTML-Navigationen, die dritte endete nach 30 Sekunden ohne abgeschlossenen
DOMContentLoaded. Dieses Verhalten im minimalen Testbestand ist ungeklärt;
es darf weder als normaler Produktions-Boot noch als dessen Ursache behauptet
werden. r2 enthält die Requestfolge und einen Browser-Trace.

## Nächste notwendige Änderung

Die native Instanz muss jede Shell-Abhängigkeit unter einer unveränderlichen,
verifizierten Release-Identität auflösen. Ein manuelles APP_BUILD oder ein
beliebiges `?v=` reicht nicht. Das Dokument, statische und dynamische Imports,
CSS-Referenzen, Worker und weitere Shell-Ressourcen müssen dieselbe Identität
behalten. Freigegebene frühere Dateien müssen bei einem Wechsel weiter unter
ihrer eigenen Identität erreichbar sein. Fehlende oder beschädigte Versionen
müssen ausdrücklich scheitern; aktuelle Bytes unter einer alten Identität sind
verboten.

Dazu gehören ein gemeinsam geprüfter URL-Vertrag für native Instanz und
Workjet-Hosts, eine vollständige Inventur absoluter beziehungsweise dynamisch
gebauter Ressourcen-URLs sowie getrennte Resolver für Shell und installierte
Apps. Erst nach dem nativen Vertrag, einer Prüfung mit signierten Slots,
aktuellem Native-Build und vollständigem Browser-/Sync-/Performance-Lauf darf
der Proxy-Kompatibilitätspfad erneut produktiv entfernt werden.
