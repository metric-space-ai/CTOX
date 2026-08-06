# GROK-5 — fuenf restliche Browser-Smokes fuer Versions-Invalidierung

## was_geaendert

Nur die Hard-Whitelist:

- `src/apps/business-os/rxdb/tests/version-invalidation-wal-pending-smoke.mjs` (neu)
- `src/apps/business-os/rxdb/tests/version-invalidation-multi-tab-smoke.mjs` (neu)
- `src/apps/business-os/rxdb/tests/version-invalidation-sidecar-window-smoke.mjs` (neu)
- `src/apps/business-os/rxdb/tests/version-invalidation-full-pull-smoke.mjs` (neu)
- `src/apps/business-os/rxdb/tests/version-invalidation-reset-wal-smoke.mjs` (neu)

Kein `src/`-Produktionscode, kein `dist/`, kein `run-all.mjs`, kein Commit.
`git status --porcelain` zeigt ausschliesslich die fuenf untracked Smokes.
Bundle-SHA unveraendert: `bc86c5bae1e4bcf56eeadd84588ba204ad481b42`.

Vorlage: `collection-version-invalidation-smoke.mjs` (Playwright, Bundle-HTTP,
`assert`, eigener DB-Name). Produktionspfad aus Commit `24e4f9dc8` /
I-061-Report bereits verdrahtet.

## tests

| Smoke | Assertions | Lauf |
|---|---:|---|
| `version-invalidation-wal-pending-smoke.mjs` | 11 | gruen |
| `version-invalidation-multi-tab-smoke.mjs` | 8 | gruen |
| `version-invalidation-sidecar-window-smoke.mjs` | 9 | gruen |
| `version-invalidation-full-pull-smoke.mjs` | 14 | gruen |
| `version-invalidation-reset-wal-smoke.mjs` | 12 | gruen |

**Summe: 54 neue Assertions.**

### 1. wal-pending (11)
pushable=0 (Master-Origin-Zeile), pending-WAL-Batch in `__recovery_v2` via
`openRecoveryJournal.appendBatch`. v0→v1 blockiert mit
`collection_version_invalidation_blocked`, Evidence
`pushableRows=0, pendingBatches=1`, Marker v0/ready, Primary-Zeile und WAL-Batch
unveraendert, Message enthaelt „Nothing was discarded“.

### 2. multi-tab (8)
Drei Seiten im selben Context: Setup (v0 Marker + Master-Cache), Writer (live
v0-Handle), Invalidator (v1 `addCollections`). Race unter Web-Lock: erlaubte
Kombinationen nur (Invalidierung gewinnt + Write fail-closed) oder (Write
gewinnt + Invalidierung blocked) oder (beide blocked). **TOCTOU-Kern:**
`invalidationWon && hasConcurrentLocal` ist verboten — kein Write zwischen
Dirty-Pruefung und Clear.

### 3. sidecar-window (9)
Ready-v0-Marker zuerst, dann Demand-Fenster in
`ctox_business_os_v1_5_meta_widgets` und `_tickets`. Nach widgets v0→v1:
widgets-Fenster leer/Scan 0, tickets-Fenster complete unberuehrt, tickets
Primary unberuehrt, widgets Primary geloescht.

### 4. full-pull (14)
Mock-Peer-Harness wie `replication-recovery-smoke.mjs` /
`first-pull-readiness-smoke.mjs` (Import aus `../src/replication-webrtc.mjs`).
`versionInvalidation.invalidated=true` + geräumter Primary + retained
LocalStorage-Checkpoint. Construction: `retainedCheckpoints=null`,
`firstPullCompletedAtMs=0`, LS-Key weg. Pull: 3 Passes (null → batch → batch →
empty), retained-old bleibt weg, frisches `firstPullCompletedAtMs` gestempelt
und persistiert.

### 5. reset-wal (12)
v0 Marker + pending WAL, dann `removeRxDatabase(name)` (resetBusinessDb-artig:
nur Primary). WAL bleibt (`pendingBatches=1`, gleiche batchId). Post-Reset
missing-marker-Bring-up und explizites v1 blockieren fail-closed auf live WAL
(`pushableRows=0, pendingBatches=1`), WAL unveraendert, nichts verworfen.

## gegenprobe

Bundle-SHA vorher/nachher je Schritt: `bc86c5bae1e4bcf56eeadd84588ba204ad481b42`
(match YES). dist/ am Ende unveraendert (`git status` clean fuer dist).

### wal-pending
- **Biegung (Bundle):**
  `if (pushable > 0 || Number(pending?.pendingBatches || 0) > 0)` →
  `if (pushable > 0 || false /* gegenprobe */)`
- **Rot:** `WAL-pending guard did not fail closed:` —
  `version-invalidation-wal-pending-smoke.mjs:142`
- **Restore:** byte-genau, SHA match, 11/11 gruen.

### multi-tab
- **Biegung (Bundle):** Guard `if (false && ...)`, Clear uebersprungen
  (`clearedRows=1` ohne delete), `assertCollectionSchemaReady` no-op,
  `runSerializedCollectionMutation` ohne Lock/Tail.
- **Rot:** `TOCTOU race produced an illegal combination: ... hasConcurrentLocal:true`
  — `version-invalidation-multi-tab-smoke.mjs:241`
  (Write ok + Invalidierung ok, concurrent-local + master-cache liegen noch).
- **Restore:** byte-genau, SHA match, 8/8 gruen.

### sidecar-window
- **Biegung (Bundle):** `invalidateQueryMetaCollection` → `return false`.
- **Rot:** `widgets demand/query sidecar window survived invalidation` —
  `version-invalidation-sidecar-window-smoke.mjs:165`
- **Restore:** byte-genau, SHA match, 9/9 gruen.

### full-pull
- Smoke importiert **src**, nicht Bundle → Bundle-Edit greift nicht.
- **Biegung (Testaufbau):** `versionInvalidation: { invalidated: false, ... }`
- **Rot:** `versionInvalidation must force retained checkpoints to null at construction`
  — `version-invalidation-full-pull-smoke.mjs:85`
- **Restore:** Test-SHA byte-genau, Bundle/src unberuehrt, 14/14 gruen.

### reset-wal
- **Biegung (Bundle):** pendingBatches-Zweig der Guard-Bedingung auf `false`.
- **Rot:** `post-reset missing-marker bring-up did not fail closed on live WAL:`
  — `version-invalidation-reset-wal-smoke.mjs:208`
- **Restore:** byte-genau, SHA match, 12/12 gruen.

## verblieben

- (leer) — alle fuenf Smokes sauber beweisbar geliefert.

## offene_bedenken

- Multi-Tab-Smoke akzeptiert die legale Serialisierungs-Outcome-Menge; die
  TOCTOU-Kernassertion (`invalidationWon ⇒ !hasConcurrentLocal`) ist die
  harte Kausalitaetspruefung. Ohne kuenstliche Bundle-Biegung bleibt der
  Produktionspfad unter Web-Locks fail-closed.
- Full-Pull deckt den Replikations-Constructor + Drain-Schleife mit Mock-Peer
  ab (analog existing recovery/readiness smokes), nicht einen echten
  WebRTC-Netzpfad. Fuer den Guard/Clear/Sidecar/WAL reichen die Browser-Smokes.
- Sidecar-DB-Namen sind global pro Collection (`ctox_business_os_v1_5_meta_<col>`),
  nicht pro Primary-DB-Name — Tests seeded/cleanupen dieselben Store-Namen und
  muessen Marker vor dem Fenster-Seeding etablieren (sonst loescht der
  Missing-Marker-Pfad die Fenster schon beim v0-Bring-up).
