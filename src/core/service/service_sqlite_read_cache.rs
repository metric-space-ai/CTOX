use std::cell::RefCell;

#[cfg(unix)]
type ServiceSqliteReadCacheKey = (PathBuf, u64, u64);
#[cfg(not(unix))]
type ServiceSqliteReadCacheKey = PathBuf;

const SERVICE_SQLITE_READ_CACHE_MAX_ENTRIES: usize = 8;

thread_local! {
    static SERVICE_SQLITE_READ_CACHE: RefCell<BTreeMap<ServiceSqliteReadCacheKey, Connection>> =
        const { RefCell::new(BTreeMap::new()) };
    #[cfg(test)]
    static SERVICE_SQLITE_READ_OPEN_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(unix)]
fn service_sqlite_read_cache_key(path: &Path) -> ServiceSqliteReadCacheKey {
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let metadata = std::fs::metadata(&canonical)
        .or_else(|_| std::fs::metadata(path))
        .ok();
    let (device, inode) = metadata
        .map(|metadata| {
            (
                std::os::unix::fs::MetadataExt::dev(&metadata),
                std::os::unix::fs::MetadataExt::ino(&metadata),
            )
        })
        .unwrap_or((0, 0));
    (canonical, device, inode)
}

#[cfg(not(unix))]
fn service_sqlite_read_cache_key(path: &Path) -> ServiceSqliteReadCacheKey {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn with_cached_service_sqlite_read_only<T>(
    path: &Path,
    purpose: &str,
    f: impl FnOnce(Option<&Connection>) -> Result<T>,
) -> Result<T> {
    SERVICE_SQLITE_READ_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        if !path.exists() {
            cache.clear();
            return f(None);
        }
        let key = service_sqlite_read_cache_key(path);
        if !cache.contains_key(&key) {
            if cache.len() >= SERVICE_SQLITE_READ_CACHE_MAX_ENTRIES {
                cache.clear();
            }
            let conn = Connection::open_with_flags(
                path,
                OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
            )
            .with_context(|| {
                format!("failed to open SQLite db {} for {purpose}", path.display())
            })?;
            conn.busy_timeout(crate::persistence::sqlite_busy_timeout_duration())
                .with_context(|| {
                    format!("failed to configure SQLite busy_timeout for {purpose}")
                })?;
            conn.execute_batch("PRAGMA query_only = ON;")
                .with_context(|| {
                    format!("failed to configure read-only SQLite mode for {purpose}")
                })?;
            #[cfg(test)]
            SERVICE_SQLITE_READ_OPEN_COUNT.with(|count| count.set(count.get() + 1));
            cache.insert(key.clone(), conn);
        }
        let result = f(cache.get(&key));
        if result.is_err() {
            cache.remove(&key);
        }
        result
    })
}

#[cfg(test)]
fn reset_service_sqlite_read_cache_for_tests() {
    SERVICE_SQLITE_READ_CACHE.with(|cell| cell.borrow_mut().clear());
    SERVICE_SQLITE_READ_OPEN_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
fn service_sqlite_read_open_count_for_tests() -> usize {
    SERVICE_SQLITE_READ_OPEN_COUNT.with(std::cell::Cell::get)
}

#[cfg(test)]
mod service_sqlite_read_cache_tests {
    use super::*;

    fn seed(path: &Path, value: i64) {
        let conn = Connection::open(path).expect("open cache test database");
        conn.execute_batch("CREATE TABLE probe(value INTEGER NOT NULL);")
            .expect("create cache test schema");
        conn.execute("INSERT INTO probe(value) VALUES (?1)", [value])
            .expect("seed cache test value");
    }

    fn read(path: &Path) -> i64 {
        with_cached_service_sqlite_read_only(path, "cache regression", |conn| {
            conn.expect("cache test database exists")
                .query_row("SELECT value FROM probe", [], |row| row.get(0))
                .map_err(Into::into)
        })
        .expect("read cached value")
    }

    #[test]
    fn reuses_multiple_databases_observes_commits_and_reopens_replacements() {
        let root = tempfile::tempdir().expect("create cache test root");
        let first = root.path().join("first.sqlite3");
        let second = root.path().join("second.sqlite3");
        seed(&first, 1);
        seed(&second, 2);
        reset_service_sqlite_read_cache_for_tests();

        assert_eq!(read(&first), 1);
        assert_eq!(read(&second), 2);
        assert_eq!(read(&first), 1);
        assert_eq!(service_sqlite_read_open_count_for_tests(), 2);

        Connection::open(&first)
            .expect("open first writer")
            .execute("UPDATE probe SET value = 11", [])
            .expect("update first database");
        assert_eq!(read(&first), 11);
        assert_eq!(service_sqlite_read_open_count_for_tests(), 2);

        let failed_query =
            with_cached_service_sqlite_read_only(&first, "cache error regression", |conn| {
                conn.expect("cache test database exists")
                    .query_row("SELECT value FROM missing_probe", [], |row| {
                        row.get::<_, i64>(0)
                    })
                    .map_err(Into::into)
            });
        assert!(failed_query.is_err());
        assert_eq!(read(&first), 11);
        assert_eq!(service_sqlite_read_open_count_for_tests(), 3);

        std::fs::rename(&second, root.path().join("second.previous.sqlite3"))
            .expect("move previous second database");
        seed(&second, 22);
        assert_eq!(read(&second), 22);
        assert_eq!(service_sqlite_read_open_count_for_tests(), 4);
        reset_service_sqlite_read_cache_for_tests();
    }
}
