# Command-Roundtrip-Abnahme vom 17.08.2026

## Ergebnis

Das lokale OA-4-Latenztor ist bestanden. Ein Warmup und anschließend 30 echte
Browser→WebRTC→Rust→RxDB→Browser-Commands vom Typ
`ctox.provider_subscription.status` ergaben auf Commit `5947d102a`:

| Kennzahl | Ergebnis | Tor |
|---|---:|---:|
| vollständige Samples | 30/30 | 30/30 |
| Gesamt-p50 | 238 ms | < 300 ms |
| Gesamt-p95 | 445,15 ms | sinkend |
| Gesamt-Maximum | 667 ms | keine neue Ausreißerklasse |
| Commit→Browser p50 | 94 ms | Diagnosewert |
| Commit→Browser p95 | 150,55 ms | Diagnosewert |

Gegen den unmittelbar vorherigen gültigen Stand `4728042e1` sanken p50 von
587,5 ms, p95 von 661,15 ms und das Maximum von 2.126 ms. Damit sind sowohl
Median als auch Tail besser.

## Belegter Engpass und Korrektur

Die exakten Query-Fetch-RPCs benötigten typischerweise nur 15–100 ms. Der
Command-Bus wartete vor jeder solchen ID-Abfrage jedoch zusätzlich auf
collectionweite Pulls der Command- und Queue-Bridge. Commit `5947d102a`
entfernt diesen redundanten Vorlauf nur aus der normalen endlichen
Terminal-Revalidierung. Die AP3-Stallreparatur behält ihren umfassenderen Pull.
Vier exakte Abfragen bleiben mit 25/50/100/200 ms begrenzt.

Der zugehörige Command-Bus-Test ist 26/26 grün und prüft ausdrücklich, dass
der normale Terminalpfad keinen collectionweiten Pull mehr ausführt.

## Reproduzierbarkeit

- Isolierter Smoke-Root:
  `/Volumes/tmp/ctox-command-repeat-500f416c6.Kz7rfu`
- Sauberer Archiv-Release des Rust-Stands `4728042e1`:
  SHA-256
  `d6eaac9b0effe0f9b0aad34e62e150a4f307408727d02330a5c61649eaf63f6d`
- Browserdatei `command-bus.js` aus `5947d102a` war vor dem Lauf bytegleich
  zwischen Arbeitsbaum und Smoke-Root (SHA-256
  `75a040b303e944f6dbb719050493a8b98c1ed73a1ae4064f75c1c0877f6b2883`).
- `5947d102a` ändert ausschließlich die Browserdatei und ihren Test; daher ist
  der unmittelbar vorherige saubere Rust-Release der passende Binärbeleg.
- Der lokale Produktionsdienst und `launchctl` wurden nicht verwendet oder
  verändert.

Rohdaten:

- `raw/command-roundtrip-warm-5947d102a-2026-08-17-marks.json`
  (SHA-256
  `3a1be57831185df2457e4af6b4fd3e66dc17bfc40ee68afd37d90cbb43629095`)
- `raw/command-roundtrip-warm-5947d102a-2026-08-17-report.json`
  (SHA-256
  `f606ea6c600b7259ee4dc63c6c1c585cf0f66f2da0dccea942106536fcb4dc27`)

Zwischenstände `281290b6b`, `3f1faf32c`, `7522d8988`, `692996f6f` und
`4728042e1` sind mit demselben Namensschema unter `raw/` versioniert. Zwei
Diagnoseläufe mit einer veralteten statischen `command-bus.js`-Kopie im
wiederverwendeten Smoke-Root wurden verworfen und nicht als Abnahmebeleg
versioniert.
