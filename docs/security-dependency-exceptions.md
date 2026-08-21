# Rust dependency audit exceptions

CTOX's Rust dependency gate scans every versioned Rust lockfile. It permits
three lockfile-only advisories only in the root lockfile and, for the RSA
advisory, the harness lockfile. The gate independently fails if an affected
package becomes reachable in either active dependency graph. Other standalone
lockfiles do not inherit these exceptions.

- `quick-xml 0.39.4` (`RUSTSEC-2026-0194`, `RUSTSEC-2026-0195`) is recorded by
  the optional `object_store` cloud feature of `polars-io`. CTOX disables the
  Polars cloud/streaming feature chain and enables only local CSV, JSON, and
  Parquet I/O. The built XML parser is `quick-xml >= 0.41`.
- `rsa 0.9.10` (`RUSTSEC-2023-0071`) is recorded by the optional MySQL driver
  in the `sqlx` meta-package. CTOX does not enable MySQL. Direct CTOX RSA key
  generation and key-format validation use `aws-lc-rs` instead.

`src/scripts/audit-rust-dependencies.sh` proves these packages are unreachable
before passing their advisory IDs to `cargo audit`; the harness has a separate
RSA reachability proof. Any future feature change that activates either package
therefore fails CI before the exception applies.

## RustSec unsoundness warnings

RustSec currently publishes unsoundness warnings without patched releases for
`anyhow`, `event-listener`, `git2`, `lru`, `memmap2`, and `rand`. CTOX does not
use the affected `anyhow::Error::downcast_mut`, `git2::Remote::list`,
buffer-backed `BlameHunk`, `lru::LruCache::iter_mut`, or custom-logger
`rand::rng()` patterns. The remaining packages enter through upstream runtime
dependencies and have no unaffected replacement release available.

The audit gate uses `--deny unsound` so a newly published unsoundness advisory
fails CI. The currently reviewed advisory IDs are explicit exceptions until an
upstream patched release exists:

- `RUSTSEC-2026-0002`, `RUSTSEC-2026-0253` (`lru`)
- `RUSTSEC-2026-0097` (`rand`)
- `RUSTSEC-2026-0183`, `RUSTSEC-2026-0184` (`git2`)
- `RUSTSEC-2026-0186` (`memmap2`)
- `RUSTSEC-2026-0190` (`anyhow`)
- `RUSTSEC-2026-0221` (`event-listener`)

Unmaintained and yanked dependency warnings are tracked separately from
security vulnerabilities and unsoundness. They do not currently have fixed
releases and are not accepted as evidence that an exploit is reachable.
