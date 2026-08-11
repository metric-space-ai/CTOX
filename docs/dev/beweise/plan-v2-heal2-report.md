## Ergebnis

`heal-minimal-v2.patch` wurde erstellt und in einem vollständig frischen Archiv-Replay erfolgreich geprüft.

- **Patch:** `/Volumes/tmp/ctox-pipeline/heal-minimal-v2.patch`
- **SHA-256:** `c3a467046527fdb4106d7ba60ecfd022e3b6357ecd84c45e83c672f7918f1c06`
- **Größe:** 183980 Bytes
- **Umfang:** 15 Dateien
- **Basis:** `HEAD` `75797527803eaf33b043dec22699586dccadefc1`
- Enthält **V1 und V2 gemeinsam in einer Datei**
- `git diff --check`: sauber

## Ergänzte Hunks gegenüber V1

Insgesamt **161 zusätzliche Diff-Hunks in drei Dateien**:

| Datei | Hunks | Art |
|---|---:|---|
| `src/core/service/service.rs` | 159 | 145× `QueuedPrompt.queue_task_metadata`; 4× Anpassung des entfernten Prompt-Parsers; 4× neues Signaturargument; 3× Typed-Metadata-Companion-Fixture; 1× Typed-Metadata-Testhelfer; 2× `QueueTaskView.metadata` |
| `src/core/business_os/store_projections.rs` | 1 | `QueueTaskView.metadata: Value::Null` im Testliteral |
| `src/core/business_os/desktop_files.rs` | 1 | Zwei Chunk-Completeness-Testhelfer im gemeinsamen Hunk |

Supplement-Diffstat:

```text
10   0  src/core/business_os/desktop_files.rs
1    0  src/core/business_os/store_projections.rs
238 17  src/core/service/service.rs
```

Ein zunächst ebenfalls übernommenes `queue_task_metadata`-Feld wurde wieder entfernt: Der betreffende Literal nutzt `..job.clone()` und benötigt das Feld nicht explizit. Final verbleiben damit exakt die **145 compilerrelevanten Initialisierer**.

## Produktions-Zusatz

Der einzige neue Hunk in einer Produktionsdatei ist:

- `src/core/business_os/desktop_files.rs`
  - `reset_desktop_file_chunk_completeness_checks`
  - `desktop_file_chunk_completeness_check_count`

Beide Funktionen sind mit `#[cfg(test)]` versehen. Sie werden ausschließlich für die vorhandenen Tests gebaut und ändern keine Laufzeitsemantik. Weitere Produktions-Hunks gegenüber V1 wurden nicht ergänzt.

## Frischer Replay-Beweis

`/Volumes/tmp/ctox-pipeline/heal2-replay` wurde abschließend erneut vollständig aufgebaut:

1. `git archive HEAD`
2. kompletter V2-Patch in einem Schritt angewandt
3. Patch-Reverse-Check erfolgreich
4. Pi-Sidecar-`dist` aus dem Checkout kopiert
5. gemeinsames Target `/Volumes/tmp/ctox-pipeline/heal-target` verwendet

Beweisbefehl:

```bash
CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline/heal-target \
cargo check --locked --tests --bin ctox
```

Ergebnis:

```text
Exit 0
Finished `dev` profile [unoptimized + debuginfo] target(s) in 1m 32s
```

- Greppy-Verdikt: `ok — exit 0, 647 warnings`
- Cargo-Endzusammenfassung für das Bin-Testtarget: 311 Warnungen, davon 266 Duplikate
- Rohlog: `/Volumes/tmp/ctox-pipeline/heal2-replay-cargo-check.log`

## V1-Receipt

- **V1:** `/Volumes/tmp/ctox-pipeline/heal-minimal.patch`
- **SHA-256:** `44f72e333f175016cd80027d133053b12ea6e542fbe2333917f5adba067be823`
- **Größe:** 106904 Bytes
- **Umfang:** 13 Dateien
- V1-Replay: `cargo check --locked --bin ctox` Exit 0
- V1-Log: `/Volumes/tmp/ctox-pipeline/heal-replay-cargo-check.log`

## Weitere Artefakte

- Fortschrittsprotokoll: `/Volumes/tmp/ctox-pipeline/heal2-fortschritt.md`
- Supplement-Diff gegenüber V1: `/Volumes/tmp/ctox-pipeline/heal2-supplement.diff`

## Offene Bedenken

- Keine offenen Compilerfehler.
- Die Warnungen wurden nicht bearbeitet, da sie außerhalb des geforderten minimalen Test-Target-Patches liegen.
- `cargo check --tests` kompiliert die Tests, führt sie aber nicht aus.
- Das Pi-Sidecar-`dist` bleibt wie bei V1 eine Replay-Voraussetzung und ist nicht Bestandteil des Patches.
- Der dirty Checkout wurde nur gelesen. Sein Status-Fingerprint blieb unverändert: `c9359b89812e19a1932c268c9ad63c027c56daa9bdc90246de59c3ba6dd7fc8e`.
