# P5-Befund: Die Batch-/Checkpoint-Matrix existiert bereits und ist begründet

Statische Messung 11.08.2026 der Wire-Konfiguration
(src/apps/business-os/shared/sync-contract.js:48-66, docs/ctox-rxdb.md:478-509).

| Collection-Klasse | batchSize | Begründung im Code |
|---|---:|---|
| desktop_file_chunks | 6 | ~22 KB/Doc; Master cappt masterChangesSince bei 96 KiB → 1,26 MB-Datei fällt von ~52 auf ~26 Pull-Roundtrips; Drain ist truncation-aware |
| knowledge_tables | 1 | Root+payload-Daten, hunderte KiB möglich → batch 20 würde das Frame-Ceiling reißen und spätere Docs stranden |
| *attachment*/*chunk* | 8 | byte-bounded, aber groß |
| reguläre Business-Docs | 20 | ≤~2 KB; halbiert Initial-Catch-up-Roundtrips ohne Frame-Limit-Nähe |
| nativer Multiplex | 20/20/5000 | pull/push/retry (rxdb_peer.rs:2535ff) |

Feste Transferrahmen (docs/ctox-rxdb.md): MAX_TRANSFER_BYTES 8 MiB,
FRAME_ACK_WINDOW 4, 16-KiB-SCTP-Ceiling mit transparentem Chunking,
idempotente ACK-Caches gegen Doppelzustellung, 200 Docs/256 KiB je Query-Chunk.

**Urteil:** Der Discovery-Hebel P5 („zu kleine/große Batches, Chattiness")
beschreibt einen Zustand, der NICHT vorliegt: die Batch-Größen sind pro
Collection-Klasse an den gemessenen Dokumentgrößen und den harten
Frame-/Transfer-Grenzen ausgerichtet, mit Roundtrip-Zahlen im Code belegt.
P5 als Bauauftrag entfällt; verbleibt nur optionales Feintuning unter
echter Transfer-Last (eigenes, kleines Ticket — braucht einen Mandanten
mit großem Chunk-Volumen als Messquelle, nicht die ruhige Idle-Instanz).

## Hebel-Gesamtbilanz (6 Discovery-Hebel)
- P1 Keep-Alive: NEGATIV — Connection:close ist Absicht (Chromium-Regression a429b596d).
- P2 DocumentCache-Deckel: GELANDET (fa1a18ab6).
- P3 Idle-Backoff: WIDERLEGT — 7/12 Loops schlafen, Consumer 30-s-Intervall.
- P4 Initial-Sync-Priorisierung: WIDERLEGT — Boot synct 15 statt 178, Demand-Loading priorisiert.
- P5 Batch-Matrix: EXISTIERT bereits, begründet.
- P6 Projektions-Scan: nur unter Änderungslast relevant (Idle still, s. P3-Nebenbefund).

Fünf der sechs statisch/messtechnisch geklärten Hebel zeigen: die
Sync-Engine ist an den vermuteten Performance-Fronten bereits solide. Der
reale Wert von Plan v2 lag im Refactoring (Peer 25.292→9.941, service.rs-
Semantik SYNC-F) und in der HEAD-Heilung, nicht in nachträglicher
Performance-Nachrüstung — die Mess-Tore haben das ehrlich gezeigt.
