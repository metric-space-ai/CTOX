# CTOX isolated plugin process protocol

This file records the intentional Rust-port delta from the pinned upstream
`internal/pluginhost` implementation. Upstream loads Go shared objects into the
host process and calls native symbols carrying JSON. CTOX will not reproduce
that trust boundary. Plugins will run out of process and retain the upstream
JSON payload schemas inside a CTOX-owned frame.

Version 1 uses a four-byte unsigned big-endian length followed by one JSON
message. Frames are limited to 8 MiB. Every message carries the protocol
version and a bounded request ID. Requests additionally carry a bounded method,
an optional absolute deadline and the untouched upstream JSON payload.

The message state machine is:

1. The host sends one `request`.
2. The plugin sends either one `response`, or ordered `stream_chunk` messages
   followed by exactly one `stream_end`.
3. Either side may send `cancel` for an active request. Cancellation is
   idempotent; messages arriving after cancellation or terminal completion are
   ignored and audited by the future process host.

The codec and transport-independent session state are active and tested. The
async reader rejects the size prefix before payload allocation. The session
allows at most 256 inflight requests and enforces duplicate IDs, deadlines,
idempotent cancellation, exact stream sequence and one terminal response/end.

Unix IPC is active at an instance-scoped private socket with a compact,
preflight-limited path, `0700` directories, `0600` socket permissions and a
five-second nonce-correlated schema/plugin-claim handshake. Cleanup is bound to
the socket device/inode so it cannot delete a path replacement. The claim is
not process authentication; peer-to-child binding belongs to the supervisor.

Unix process spawning is now active behind an explicit supervisor. The child
receives no ambient environment or free-form arguments. A 256-bit one-shot
token travels only through stdin; the response proof binds that token to the
host nonce, plugin ID and schema. The pipe is closed before the socket
handshake. Graceful shutdown, crash-with-inflight, kill-on-drop and capped,
budgeted restart backoff have real child-process tests.

Windows named-pipe IPC, a production installed-plugin selector and registered
capability calls are not implemented yet. The supervisor is therefore not
wired into CTOX service startup and the safe plugin replacement remains
partial.
