# CTOX sync and managed-service security review

Date: 2026-08-21

## Executive summary

The CTOX WebRTC/RxDB sync engine and the ctox.dev managed service were reviewed
as one trust boundary. All identified reachable critical, high, and medium
findings are fixed. The remaining RustSec entries are either unreachable
lockfile-only dependencies or warnings without a fixed upstream release; CI
proves the former stay unreachable and fails on any newly published unsoundness
advisory.

## Findings

### SEC-001 — Critical — signaling identity and legacy room authentication

- Location: `src/core/rxdb/src/plugins/replication_webrtc/signaling_client.rs`
  and the ctox.dev signaling worker.
- Impact: legacy room credentials and insufficient role binding could allow an
  unauthorized signaling participant to attempt peer confusion or room abuse.
- Resolution: signaling now uses the cryptographically signed `/v2` contract,
  binds browser/native roles into the credential, rejects legacy joins, limits
  role cardinality, and re-derives credentials on reconnect.
- Status: fixed and covered by native, browser, and live-production signaling
  tests.

### SEC-002 — High — browser credential lifetime and revocation

- Location: CTOX Business OS bootstrap/sync code and ctox.dev token issuance.
- Impact: long-lived or browser-visible room secrets would increase replay and
  disclosure impact.
- Resolution: the browser no longer receives a room password; scoped signaling
  tokens are short-lived, rotatable, and revocable. Native peer lifecycle and
  reconnect behavior remain server-authoritative.
- Status: fixed.

### SEC-003 — High — managed-service web trust boundaries

- Location: ctox.dev Next.js routes, Cloudflare signaling worker, and managed
  control-plane handlers.
- Impact: permissive cross-origin behavior, unsafe redirects/fetches, missing
  request provenance checks, or unescaped output could enable CSRF, SSRF, open
  redirects, credential abuse, or XSS.
- Resolution: strict origin/CORS policy, CSRF validation, bounded input and
  rate limits, DNS-rebinding-resistant SSRF checks, URL allowlists, nonce CSP,
  security headers, output escaping, and redacted errors/logging.
- Status: fixed and deployed.

### SEC-004 — High — installer and remote-host supply chain

- Location: ctox.dev installer generation and CTOX Desktop SSH/update paths.
- Impact: mutable downloads or permissive SSH host-key handling could permit a
  compromised artifact or machine-in-the-middle to replace trusted code.
- Resolution: installers pin an immutable CTOX commit plus SHA-256; SSH uses
  strict host-key verification and explicit trust onboarding.
- Status: fixed. The production installer pin is refreshed as part of every
  release deployment.

### SEC-005 — High — reachable vulnerable dependencies

- Location: root and nested Cargo manifests/locks plus
  `src/apps/business-os-desktop/package.json`.
- Evidence: the release now runs `src/scripts/audit-rust-dependencies.sh` from
  `.github/workflows/ci.yml` and `.github/workflows/release.yml`; desktop and
  Business OS jobs run `npm audit --audit-level=low`.
- Resolution: upgraded Electron and js-yaml, RMCP, WebSocket, HTTP/2, QUIC,
  serialization, XML, browser-launch, and supporting crates; replaced direct
  RSA operations with AWS-LC; patched Rama DNS to fixed Hickory releases; and
  vendored the SQL Server TLS adapter on rustls 0.23.
- Status: fixed. All three npm trees report zero vulnerabilities and the Rust
  audit reports no vulnerability or denied-unsoundness failures.

### SEC-006 — Medium — stale release and security guard coverage

- Location: Business OS inventory generators, command/data-plane smoke tests,
  and `src/apps/business-os/scripts/run-app-story-tests.mjs`.
- Impact: stale timeouts, source inventories, or app counts could make a red
  guard uninformative and allow security-sensitive drift to escape CI.
- Resolution: guards now match the 120-second production command budget,
  demand-only exact-ID replication, current native source inventory, 35-app
  catalog, and current schema cache-busters. Expected fail-soft test warnings
  no longer flood or truncate test evidence.
- Status: fixed.

## Reviewed dependency exceptions

The exact RustSec exceptions, reachability proof, affected APIs, and update
policy are documented in `docs/security-dependency-exceptions.md`. These are
not known reachable P0/P1/P2 findings in the current feature graph.

## Verification

- Native RxDB: 359 unit tests plus conformance and guard suites passed.
- Browser RxDB: 104 tests passed with the real cross-process wire daemon.
- Network proxy: 123 tests passed.
- CLIProxy: 2,509 unit tests passed; plugin supervisor 4/4 on isolated rerun.
- Business OS: full test command passed; app stories 621 passed, 1 skipped;
  shared, App Store, and module bundle suites passed.
- Desktop: syntax/config checks, 167 unit tests, and five Electron smoke tests
  passed.
- Dependency scans: npm audit reports zero vulnerabilities for Business OS,
  Desktop, and ctox.dev; cargo audit has no vulnerability or denied-unsoundness
  failures across the root, harness, and SQL Server tool lockfiles.

## Residual risk

No security review can prove the absence of every defect. The current residual
risk is limited to explicitly tracked upstream maintenance/unsoundness warnings,
future dependency or configuration drift, and operational compromise outside
the application trust boundary. CI, Dependabot, immutable installer pins, token
rotation, and live signaling checks are the controls for those risks.
