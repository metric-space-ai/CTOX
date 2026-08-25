# CTOX remote installation contract

Workjet provisions CTOX from the official GitHub release channel. The target
machine downloads all metadata and artifacts itself; Workjet does not relay
release payloads in version 1.

Each release publishes `ctox-install-manifest-v1.json` with schema
`ctox.install-manifest.v1`. It contains the stable release tag, a fixed
platform/architecture matrix, artifact URLs, SHA-256 digests and compatibility
flags. The manifest and referenced artifacts are covered by the release's
GitHub build-provenance attestation.

Supported targets are macOS arm64/x64, Linux arm64/x64 and Windows x64. A
consumer must reject an absent target, checksum mismatch, incompatible schema,
non-HTTPS artifact URL, or a manifest whose repository is not
`metric-space-ai/ctox`.

Business OS readiness is not established by an HTTP page. After installation,
Workjet must verify the CTOX service, native peer and RxDB/WebRTC replication.
The manifest explicitly advertises `httpDataBridge: false` to prevent an HTTP
fallback from becoming an accidental data path.
