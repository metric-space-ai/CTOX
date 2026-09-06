use ctox_sync::authority::{store::SqliteStore, TypeConfig};
use openraft::{
    testing::{StoreBuilder, Suite},
    StorageError,
};
struct Builder;
impl StoreBuilder<TypeConfig, SqliteStore, SqliteStore, tempfile::TempDir> for Builder {
    async fn build(
        &self,
    ) -> Result<(tempfile::TempDir, SqliteStore, SqliteStore), StorageError<u64>> {
        let root = tempfile::tempdir().unwrap();
        let store = SqliteStore::open(&root.path().join("authority.sqlite")).unwrap();
        Ok((root, store.clone(), store))
    }
}
#[test]
fn sqlite_implements_openraft_storage_contract() {
    Suite::<TypeConfig, SqliteStore, SqliteStore, Builder, tempfile::TempDir>::test_all(Builder)
        .unwrap();
}
