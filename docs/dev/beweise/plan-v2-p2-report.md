## Ergebnis

Geändert wurde ausschließlich `src/core/rxdb/src/doc_cache.rs`.

- Modulkonstante `DOC_CACHE_MAX_ENTRIES = 10_000` ergänzt.
- Jeder Cache-Eintrag erhält eine monotone Einfügesequenz.
- Beim Einfügen einer neuen Dokument-ID greift zusätzlich zum bestehenden 256er-Sweep der Größen-Deckel.
- Verdrängt werden ausschließlich Einträge, bei denen für **alle** `by_rev`-Weaks `strong_count() == 0` gilt.
- Verdrängung erfolgt deterministisch nach Einfügesequenz; Dokument-ID dient als Tie-Breaker.
- Falls mehr als 10.000 Einträge lebende Handles haben, bleibt der Deckel bewusst weich: Lebende Handles werden nie verdrängt.
- Keine neue Dependency, Env-Variable oder öffentliche Config-Fläche.

## Vorher/Nachher

Baseline mit dem vorgegebenen Testkommando:

- **5 bestanden**
- **0 fehlgeschlagen**
- **339 gefiltert**

Nachher:

- **6 bestanden**
- **0 fehlgeschlagen**
- **339 gefiltert**
- Keine neuen roten Tests.

Der neue Test verwendet `k = 8` und `N = 64` tote IDs, bleibt mit insgesamt 66 Cache-Aufrufen bewusst unter dem 256er-Sweep und belegt daher gezielt die Einfüge-Verdrängung. Das lebende Handle bleibt pointer-identisch und behält sowohl Handle-Inhalt als auch `latest`-Inhalt.

## Gegenprobe und Stabilität

- Den Test-Deckel temporär mit `usize::MAX` deaktiviert.
- Erwartetes Ergebnis: Test rot mit `dead-id churn exceeded the cache limit: 65 > 8`.
- Danach den Deckel wiederhergestellt.
- Der finale Leck-Test lief anschließend **3× unmittelbar hintereinander grün**.
- `cargo fmt --check --manifest-path src/core/rxdb/Cargo.toml` ist grün.
- Vor jedem Cargo-Lauf wurden freier Speicher und Load geprüft; Minimum waren 15 GiB frei, die Load blieb deutlich unter 30.
- Kein `git add` oder Commit ausgeführt.

## Default-Begründung

10.000 Einträge lassen typische aktive Arbeitsmengen im Cache, begrenzen aber tote churn-getriebene IDs samt geklontem `latest`-Payload. Es handelt sich absichtlich nicht um ein hartes Gesamt-RAM-Limit: Payloadgrößen variieren und lebende Handles dürfen den Deckel überschreiten.

## Offene Bedenken

- Bei einem außergewöhnlich großen Bestand lebender Handles kann der Cache weiterhin über 10.000 Einträge wachsen; das ist die notwendige Folge der Handle-Semantik.
- Die Verdrängung prüft beim Überschreiten des Deckels die Weak-Handles unter dem bestehenden Cache-Mutex. Bei dauerhaft sehr großen Live-Sets kann dies zusätzliche CPU-/Lock-Zeit verursachen.
- Der `u64`-Sequenzüberlauf ist praktisch unerreichbar; nach `2^64` Einfügungen wäre die chronologische Ordnung nicht mehr exakt.

```workjet-completion-receipt-v1
{"schemaVersion":1,"status":"completed","summary":"DocumentCache entry ceiling implemented in src/core/rxdb/src/doc_cache.rs. The production default is 10000 entries; insertion-time eviction deterministically removes only entries whose by_rev weak handles all have strong_count()==0, while the existing 256-call sweep remains unchanged. A regression test proves dead-ID churn is capped and a live handle remains pointer-identical with unchanged content.","changedFiles":["src/core/rxdb/src/doc_cache.rs"],"verification":[{"command":"cargo test --manifest-path src/core/rxdb/Cargo.toml doc_cache -- --test-threads=4","result":"baseline: 5 passed, 0 failed, 339 filtered; final: 6 passed, 0 failed, 339 filtered"},{"command":"cargo test --manifest-path src/core/rxdb/Cargo.toml doc_cache::tests::size_limit_evicts_dead_ids_and_preserves_live_handle -- --exact --test-threads=1","result":"passed 3 consecutive final runs"},{"command":"temporary test simulation with max_entries=usize::MAX","result":"expected failure: dead-id churn exceeded the cache limit: 65 > 8; original limit restored afterward"},{"command":"cargo fmt --check --manifest-path src/core/rxdb/Cargo.toml","result":"passed"},{"command":"git status --short -- src/core/rxdb/src/doc_cache.rs","result":"only the whitelisted file is modified by this work; no git add or commit performed"}],"concerns":["The ceiling is intentionally soft when more than 10000 document IDs have live handles.","Insertion-time eviction scans weak handles while holding the existing cache mutex; unusually large live working sets can add CPU and lock time.","The u64 insertion sequence would lose exact chronological ordering only after practically unreachable wraparound."],"producedPaths":[]}
```
