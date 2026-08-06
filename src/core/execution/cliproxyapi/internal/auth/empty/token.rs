// ref: internal/auth/empty/token.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::internal::auth::models::{TokenStorage, TokenStorageError};

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct EmptyStorage {
    #[serde(rename = "type")]
    pub storage_type: String,
}

impl TokenStorage for EmptyStorage {
    fn save_token_to_file(&mut self, _auth_file_path: &Path) -> Result<(), TokenStorageError> {
        self.storage_type = "empty".to_owned();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{EmptyStorage, TokenStorage};

    #[test]
    fn save_is_a_noop_but_marks_the_storage_type() {
        let mut storage = EmptyStorage::default();
        storage
            .save_token_to_file(Path::new("must-not-be-created.json"))
            .expect("empty storage should not perform I/O");
        assert_eq!(storage.storage_type, "empty");
        assert_eq!(
            serde_json::to_value(storage).expect("serialize empty storage"),
            serde_json::json!({"type": "empty"})
        );
        assert!(!Path::new("must-not-be-created.json").exists());
    }
}
