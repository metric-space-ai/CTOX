commit 9a9e043f2cd0e6bf2ffa305a988adbb5e3d1f371
Author: Codex <codex@openai.com>
Date:   Tue Jul 28 10:45:36 2026 +0200

    rxdb(fetch): unify the in-flight stream-limit code across both handlers
    
    E4 rework R5. The final review caught an unfulfilled B2 promise that my
    verification of that ticket missed: both handlers reject on the same
    condition — max in-flight streams reached — but the file side reported
    RATE_LIMITED while the query side reported STREAM_LIMIT_EXCEEDED. B2's
    brief named STREAM_LIMIT_EXCEEDED as the unified code and the fix only
    reached one copy, which is exactly the drift the ticket existed to
    remove.
    
    Behaviour-neutral today: both were already retryable and routeFileError
    passes code and retryable through without an allowlist, so no JS
    consumer distinguished them. FILE_FETCH_ERROR_RATE_LIMITED had no other
    use — the file handler never had a real rate-limit case, it mislabelled
    the stream limit — so the constant is replaced rather than kept.
    
    Also records the open E4 adjudication at the resolved-conflict wait: the
    reviewer argues queue-head semantics leave a window this port has and
    upstream RxDB does not, because we publish activity before queue entry
    while upstream enqueues synchronously. Plausible and low severity
    (self-healing via the next conflict round), but switching to
    ActivityAndQueue makes
    downstream_waits_for_upstream_queue_when_fork_is_resolved_conflict time
    out — that test pins the current semantics on purpose. Changing
    behaviour by rewriting its guard is exactly what this campaign refused
    all along, so the requirement stays and the comment names the experiment
    that would settle it.
    
    371 tests green, clippy clean, fmt clean.
    
    Backlog: SYNC-A-R5
    
    Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

diff --git a/src/core/rxdb/src/plugins/replication_webrtc/file_fetch_handler.rs b/src/core/rxdb/src/plugins/replication_webrtc/file_fetch_handler.rs
index 87d2874cb..1640427b8 100644
--- a/src/core/rxdb/src/plugins/replication_webrtc/file_fetch_handler.rs
+++ b/src/core/rxdb/src/plugins/replication_webrtc/file_fetch_handler.rs
@@ -71,7 +71,7 @@ pub struct FileFetchChunk {
 pub const FILE_FETCH_ERROR_NOT_FOUND: &str = "FILE_NOT_FOUND";
 pub const FILE_FETCH_ERROR_SOURCE: &str = "FILE_SOURCE_ERROR";
 pub const FILE_FETCH_ERROR_UNAUTHORIZED: &str = "UNAUTHORIZED";
-pub const FILE_FETCH_ERROR_RATE_LIMITED: &str = "RATE_LIMITED";
+pub const FILE_FETCH_ERROR_STREAM_LIMIT: &str = "STREAM_LIMIT_EXCEEDED";
 pub const FILE_FETCH_ERROR_FEATURE_DISABLED: &str = "FEATURE_DISABLED";
 pub const FILE_FETCH_ERROR_REMOTE_TIMEOUT: &str = "REMOTE_TIMEOUT";
 
@@ -269,7 +269,7 @@ pub async fn run_file_fetch<H: WebRTCConnectionHandler>(
                 &peer,
                 &message.id,
                 &request.request_id,
-                FILE_FETCH_ERROR_RATE_LIMITED,
+                FILE_FETCH_ERROR_STREAM_LIMIT,
                 "max in-flight file streams reached",
                 true,
             )
diff --git a/src/core/rxdb/src/replication_protocol/downstream.rs b/src/core/rxdb/src/replication_protocol/downstream.rs
index 0c20856fc..68d41b8ce 100644
--- a/src/core/rxdb/src/replication_protocol/downstream.rs
+++ b/src/core/rxdb/src/replication_protocol/downstream.rs
@@ -373,9 +373,23 @@ async fn persist_from_master(
                     .and_then(Value::as_str)
                     == fork.get("_rev").and_then(Value::as_str)
                 {
-                    // This is deliberately queue-head semantics: upstream RxDB
-                    // awaits `streamQueue.up` here so the resolved write is sent,
-                    // without waiting for unrelated future upstream activity.
+                    // Deliberately queue-head semantics: upstream RxDB awaits
+                    // `streamQueue.up` here so the resolved write is sent, without
+                    // waiting for unrelated future upstream activity. Pinned by
+                    // `downstream_waits_for_upstream_queue_when_fork_is_resolved_conflict`.
+                    //
+                    // OPEN (E4 adjudication, SYNC-A-R9): upstream enqueues its task
+                    // synchronously on event arrival, while this port publishes
+                    // activity *before* queue entry — so there may be a window
+                    // (activity set, queue still empty) that the original does not
+                    // have, letting an upstream push with a stale assumedMasterState
+                    // overtake this fork update. Self-healing via the next conflict
+                    // round, hence low severity. Switching to `ActivityAndQueue`
+                    // makes the test above time out, so the window must be
+                    // demonstrated before the semantics change: park an upstream
+                    // task between activity-track and queue acquisition via the
+                    // `wait_before_persist` hook while a resolved-conflict batch
+                    // runs. Do not flip the requirement without that evidence.
                     crate::replication_protocol::index_mod::await_rx_storage_replication_direction_idle(
                         state,
                         RxStorageReplicationDirection::Up,
