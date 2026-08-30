# CTOX object_store 0.13.2 patch

This directory is the published `object_store` 0.13.2 crate, whose crates.io
archive has SHA-256 `622acbc9100d3c10e2ee15804b0caa40e55c933d5aa53814cd520805b7958a49`.

CTOX keeps this version because Polars 0.53 constrains `object_store` to the
0.13 API. The only source-package change is the compatible `quick-xml`
dependency update from 0.39 to 0.41. This removes the reachable vulnerable
0.39 release without disabling Polars' local Parquet streaming executor or
loosening any Cargo feature.

The upstream license and notice files are retained unchanged. Delete this
patch when CTOX upgrades to a Polars release that supports `object_store`
0.14 or newer on the repository's pinned Rust toolchain.
