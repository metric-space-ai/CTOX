// ref: internal/misc/credentials.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::path::{Component, Path, PathBuf};

use serde::Serialize;
use serde_json::{Map, Value};

/// Separator used by a host audit sink to group credential lifecycle events.
pub const CREDENTIAL_SEPARATOR: &str =
    "-------------------------------------------------------------------";

/// Non-secret audit event for an impending credential-store write.
///
/// Upstream prints the path directly. The Rust port returns structured data so
/// the CTOX host remains the sole logging authority and can apply its path
/// policy before emitting evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SavingCredentialsEvent {
    cleaned_path: PathBuf,
}

impl SavingCredentialsEvent {
    pub fn cleaned_path(&self) -> &Path {
        &self.cleaned_path
    }

    pub fn message(&self) -> String {
        format!("Saving credentials to {}", self.cleaned_path.display())
    }
}

/// Builds the upstream saving-credentials event without writing to stdout.
pub fn saving_credentials_event(path: impl AsRef<Path>) -> Option<SavingCredentialsEvent> {
    let path = path.as_ref();
    if path.as_os_str().is_empty() {
        return None;
    }
    Some(SavingCredentialsEvent {
        cleaned_path: clean_path(path),
    })
}

/// Serializes a source using its JSON field names and overlays metadata.
/// Metadata wins on duplicate keys, matching upstream.
pub fn merge_metadata<T>(
    source: &T,
    metadata: Option<&Map<String, Value>>,
) -> Result<Map<String, Value>, MergeMetadataError>
where
    T: Serialize + ?Sized,
{
    let value = serde_json::to_value(source).map_err(|_| MergeMetadataError::SerializeSource)?;
    let mut data = match value {
        Value::Object(data) => data,
        _ => return Err(MergeMetadataError::SourceIsNotObject),
    };
    if let Some(metadata) = metadata {
        data.extend(metadata.clone());
    }
    Ok(data)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MergeMetadataError {
    SerializeSource,
    SourceIsNotObject,
}

impl fmt::Display for MergeMetadataError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::SerializeSource => "failed to marshal credential metadata source",
            Self::SourceIsNotObject => "failed to unmarshal credential metadata source to map",
        })
    }
}

impl std::error::Error for MergeMetadataError {}

fn clean_path(path: &Path) -> PathBuf {
    let mut cleaned = PathBuf::new();
    let anchored = path.is_absolute();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => match cleaned.components().next_back() {
                Some(Component::Normal(_)) => {
                    cleaned.pop();
                }
                Some(Component::ParentDir) | None if !anchored => {
                    cleaned.push(component.as_os_str());
                }
                _ => {}
            },
            _ => cleaned.push(component.as_os_str()),
        }
    }
    cleaned
}

#[cfg(test)]
mod tests {
    use serde::Serialize;
    use serde_json::json;

    use super::*;

    #[derive(Serialize)]
    struct TaggedSource {
        #[serde(rename = "access_token")]
        token: String,
        count: u64,
    }

    #[test]
    fn metadata_overlays_json_tagged_source_without_mutating_input() {
        let source = TaggedSource {
            token: "source".to_owned(),
            count: 3,
        };
        let metadata = serde_json::from_value::<Map<String, Value>>(json!({
            "access_token": "metadata",
            "extra": true
        }))
        .unwrap();
        let merged = merge_metadata(&source, Some(&metadata)).unwrap();
        assert_eq!(merged["access_token"], "metadata");
        assert_eq!(merged["count"], 3);
        assert_eq!(merged["extra"], true);
        assert_eq!(source.token, "source");
    }

    #[test]
    fn map_source_is_cloned_and_nil_metadata_is_a_noop() {
        let source = serde_json::from_value::<Map<String, Value>>(json!({"a": 1})).unwrap();
        let mut merged = merge_metadata(&source, None).unwrap();
        assert_eq!(merged, source);
        merged.insert("a".to_owned(), json!(2));
        assert_eq!(source["a"], 1);
    }

    #[test]
    fn non_object_source_returns_typed_redacted_error() {
        let error = merge_metadata(&["secret"], None).unwrap_err();
        assert_eq!(error, MergeMetadataError::SourceIsNotObject);
        assert!(!format!("{error:?} {error}").contains("secret"));
    }

    #[test]
    fn saving_event_cleans_path_and_never_logs_implicitly() {
        assert!(saving_credentials_event("").is_none());
        let event = saving_credentials_event("credentials/./nested/../auth.json").unwrap();
        assert_eq!(event.cleaned_path(), Path::new("credentials/auth.json"));
        assert_eq!(
            event.message(),
            "Saving credentials to credentials/auth.json"
        );
        assert_eq!(CREDENTIAL_SEPARATOR.len(), 67);
        assert_eq!(
            saving_credentials_event("../../auth.json")
                .unwrap()
                .cleaned_path(),
            Path::new("../../auth.json")
        );
        assert_eq!(
            saving_credentials_event("/../auth.json")
                .unwrap()
                .cleaned_path(),
            Path::new("/auth.json")
        );
    }
}
