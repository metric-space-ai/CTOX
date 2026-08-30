# Workjet and CTOX product matrix

This matrix is the release and naming authority for the Workjet/CTOX product family.

| Component | Canonical source | Status | User-facing name |
| --- | --- | --- | --- |
| Desktop and web client | `metric-space-ai/workjet` | Supported product | Workjet |
| iOS and Android client | `metric-space-ai/workjet/apps/mobile` | Supported product | Workjet |
| Durable daemon and Business OS backend | `metric-space-ai/ctox` | Supported backend | CTOX Backend |
| `src/apps/business-os-desktop` | this repository | Legacy migration donor; releases disabled | None |
| `src/apps/business-os-mobile` | this repository | Mobile implementation donor; not a product | None |

## Naming rules

- App windows, installers, stores, navigation and onboarding use **Workjet**.
- **CTOX** appears only for the backend, its instances, installation, health and diagnostics.
- Existing legacy bundle identifiers and incoming URL schemes may remain temporarily for update and data continuity. New links and release artifacts use Workjet naming.

## Release rules

- User-facing desktop and mobile artifacts ship only from `metric-space-ai/workjet`.
- CTOX releases contain the backend, machine-readable installer metadata and backend runtime assets.
- The legacy Electron and native mobile donors in this repository must not publish app artifacts.
- Donor removal requires documented feature parity, signed Workjet artifacts, RxDB/WebRTC restart persistence, and update/rollback evidence.
