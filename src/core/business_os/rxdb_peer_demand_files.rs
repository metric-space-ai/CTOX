// Origin: CTOX
// License: Apache-2.0

use super::rxdb_peer::{
    active_desktop_file_chunk_rows_from_sqlite, demand_file_chunk_rows_for_key_from_sqlite,
    runtime_module_demand_chunk_sources, DemandFileFetchRequestStats, WebRtcPool,
    DEMAND_FILE_FETCH_METRICS,
};
use base64::Engine;
use rxdb::plugins::replication_webrtc::file_fetch_handler::FileRange;
use rxdb::rx_database::RxDatabase;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

/// Phase 4: file-demand sources exposed to the browser. The request collection
/// can be metadata (`desktop_files`) while the bytes still live in a separate
/// chunk collection (`desktop_file_chunks`); this keeps large payloads off the
/// normal background replication path.
pub(super) struct DemandFileChunkCollection {
    pub(super) request_collection: &'static str,
    pub(super) storage_collection: &'static str,
    pub(super) key_field: &'static str,
}

pub(super) const DEMAND_FILE_CHUNK_COLLECTIONS: &[DemandFileChunkCollection] = &[
    DemandFileChunkCollection {
        request_collection: "desktop_files",
        storage_collection: "desktop_file_chunks",
        key_field: "file_id",
    },
    DemandFileChunkCollection {
        request_collection: "desktop_file_chunks",
        storage_collection: "desktop_file_chunks",
        key_field: "file_id",
    },
    DemandFileChunkCollection {
        request_collection: "document_blob_chunks",
        storage_collection: "document_blob_chunks",
        key_field: "blob_id",
    },
    DemandFileChunkCollection {
        request_collection: "spreadsheet_blob_chunks",
        storage_collection: "spreadsheet_blob_chunks",
        key_field: "blob_id",
    },
    DemandFileChunkCollection {
        request_collection: "business_module_source_blob_chunks",
        storage_collection: "business_module_source_blob_chunks",
        key_field: "blob_id",
    },
];

/// SYNC-32: one resolved demand-file source — either a built-in entry from
/// `DEMAND_FILE_CHUNK_COLLECTIONS` or a runtime-module collection declared
/// with `"syncProfile": "demand-chunks"`.
pub(super) struct DemandFileSourceConfig {
    pub(super) request_collection: String,
    pub(super) storage_collection: String,
    pub(super) key_field: String,
}

/// SYNC-32: built-in demand-file sources plus the sources runtime-installed
/// modules declare. Built-ins come first and are unchanged; declared sources
/// are appended, serve their own collection (request == storage), and are
/// deduped on the request collection so a declaration can never shadow a
/// built-in source.
pub(super) fn demand_file_source_configs(root: &Path) -> Vec<DemandFileSourceConfig> {
    let mut configs: Vec<DemandFileSourceConfig> = DEMAND_FILE_CHUNK_COLLECTIONS
        .iter()
        .map(|source| DemandFileSourceConfig {
            request_collection: source.request_collection.to_owned(),
            storage_collection: source.storage_collection.to_owned(),
            key_field: source.key_field.to_owned(),
        })
        .collect();
    let mut seen: HashSet<String> = configs
        .iter()
        .map(|config| config.request_collection.clone())
        .collect();
    for (collection, key_field) in runtime_module_demand_chunk_sources(root) {
        if !seen.insert(collection.clone()) {
            continue;
        }
        configs.push(DemandFileSourceConfig {
            request_collection: collection.clone(),
            storage_collection: collection,
            key_field,
        });
    }
    configs
}

/// File bytes are authoritative storage, never a lossy business projection.
pub(super) fn is_demand_file_storage_collection(root: &Path, collection: &str) -> bool {
    if DEMAND_FILE_CHUNK_COLLECTIONS
        .iter()
        .any(|source| source.storage_collection == collection)
    {
        return true;
    }
    // Installed modules cannot shadow built-in schemas. Avoid rescanning
    // every installed module when opening an ordinary projection writer.
    if super::rxdb_peer::business_os_schema_contract().contains_key(collection) {
        return false;
    }
    runtime_module_demand_chunk_sources(root)
        .iter()
        .any(|(name, _)| name == collection)
}

/// Phase 4: register a bounded-memory file stream source on the pool's file
/// fetch registry for each file-bearing chunk collection that is actually
/// registered on this database. Without this, `rxdb.file.fetch` always returns
/// FILE_NOT_FOUND (no source). The source closure is sync and reads the local
/// RxDB SQLite store through read-only queries; the file-fetch dispatcher runs
/// it on a blocking worker and applies async transport backpressure.
/// SYNC-32: the source list is the built-in set plus runtime-module
/// `demand-chunks` declarations (`demand_file_source_configs`).
pub(super) fn register_demand_file_sources(
    pool: &WebRtcPool,
    database: &Arc<RxDatabase>,
    root: &Path,
) {
    for source_config in demand_file_source_configs(root) {
        // Only register sources whose backing storage collection exists (the
        // catalog is fault-tolerant; optional chunk collections may be absent).
        if database
            .collection(&source_config.storage_collection)
            .is_none()
        {
            continue;
        }
        let root = root.to_path_buf();
        let request_collection = source_config.request_collection;
        let storage_collection = source_config.storage_collection;
        let key_field = source_config.key_field;
        let closure_storage_collection = storage_collection.clone();
        let closure_key_field = key_field.clone();
        let source: Arc<rxdb::plugins::replication_webrtc::file_fetch_handler::FileChunkStreamFn> =
            Arc::new(move |_collection, file_id, range, emit| {
                stream_demand_file_chunks(
                    &root,
                    &closure_storage_collection,
                    &closure_key_field,
                    file_id,
                    range,
                    emit,
                )
            });
        pool.file_fetch_registry
            .register_stream_source(&request_collection, source);
        eprintln!(
            "[business-os] demand-fetch file source registered for `{request_collection}` \
             via `{storage_collection}` (key `{key_field}`)"
        );
    }
}

/// Phase 4: stream the bytes of `file_id` from `collection`'s chunk documents.
/// Reads the chunk docs by the collection's key field, orders by `idx`,
/// base64-decodes each `data`, and emits one chunk of raw bytes at a time
/// (honoring an optional byte range). Returns `Err` when the collection is
/// missing or the query fails; emits nothing (→ FILE_NOT_FOUND upstream) when
/// the file has no chunks.
pub(super) fn stream_demand_file_chunks(
    root: &Path,
    collection: &str,
    key_field: &str,
    file_id: &str,
    range: Option<&FileRange>,
    emit: &mut dyn FnMut(&[u8]) -> rxdb::rx_error::RxResult<bool>,
) -> rxdb::rx_error::RxResult<()> {
    let mut stats = DemandFileFetchRequestStats::new(range);
    let result = stream_demand_file_chunks_inner(
        root, collection, key_field, file_id, range, emit, &mut stats,
    );
    stats.finish();
    DEMAND_FILE_FETCH_METRICS.record(&stats, result.is_ok());
    result
}

fn stream_demand_file_chunks_inner(
    root: &Path,
    collection: &str,
    key_field: &str,
    file_id: &str,
    range: Option<&FileRange>,
    emit: &mut dyn FnMut(&[u8]) -> rxdb::rx_error::RxResult<bool>,
    stats: &mut DemandFileFetchRequestStats,
) -> rxdb::rx_error::RxResult<()> {
    let (mut chunk_rows, loaded_base_offset) = if collection == "desktop_file_chunks" {
        active_desktop_file_chunk_rows_from_sqlite(root, file_id, range, stats)?
    } else {
        (
            demand_file_chunk_rows_for_key_from_sqlite(
                root, collection, key_field, file_id, None, None, stats,
            )?,
            0,
        )
    };
    // Order by `idx` so the reassembled byte stream is correct.
    chunk_rows.sort_by_key(|chunk: &Value| chunk.get("idx").and_then(Value::as_u64).unwrap_or(0));

    // Range support: skip/take a byte window across the decoded chunk stream.
    let (range_start, range_end) = match range {
        Some(r) => (r.offset, r.offset.saturating_add(r.length)),
        None => (0u64, u64::MAX),
    };
    let mut emitted_offset: u64 = loaded_base_offset;
    for chunk in chunk_rows {
        // Skip redacted/pruned chunks (empty data) so they do not corrupt the
        // stream; the browser tracks presence separately.
        let data = chunk.get("data").and_then(Value::as_str).unwrap_or("");
        if data.is_empty() {
            continue;
        }
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(data.as_bytes())
            .map_err(|err| {
                rxdb::rx_error::new_rx_error(
                    "RC_WEBRTC_PEER",
                    Some(json!({
                        "message": format!("decode {collection} chunk for {file_id}: {err}"),
                    })),
                )
            })?;
        stats.chunks_decoded = stats.chunks_decoded.saturating_add(1);
        stats.bytes_decoded = stats
            .bytes_decoded
            .saturating_add(u64::try_from(decoded.len()).unwrap_or(u64::MAX));
        let chunk_start = emitted_offset;
        let chunk_end = emitted_offset.saturating_add(decoded.len() as u64);
        emitted_offset = chunk_end;
        // Clip this chunk to the requested byte window.
        if chunk_end <= range_start || chunk_start >= range_end {
            continue;
        }
        let slice_start = range_start.saturating_sub(chunk_start) as usize;
        let slice_end = (range_end.min(chunk_end) - chunk_start) as usize;
        let slice = &decoded[slice_start.min(decoded.len())..slice_end.min(decoded.len())];
        if slice.is_empty() {
            continue;
        }
        // `emit` returns Ok(false) to stop early (cancel / known-sequence skip).
        if !emit(slice)? {
            break;
        }
        stats.bytes_emitted = stats
            .bytes_emitted
            .saturating_add(u64::try_from(slice.len()).unwrap_or(u64::MAX));
    }
    Ok(())
}
