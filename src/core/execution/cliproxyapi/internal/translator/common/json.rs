// refs: sdk/translator/registry.go, internal/translator/** @ ffdb9c9fbc78a6235d59c9ccbdc4243ba35ecdcd
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawJson(Vec<u8>);

impl RawJson {
    pub fn parse(bytes: impl Into<Vec<u8>>) -> Result<Self, serde_json::Error> {
        let bytes = bytes.into();
        serde_json::from_slice::<&serde_json::value::RawValue>(&bytes)?;
        Ok(Self(bytes))
    }

    pub fn from_value(value: &Value) -> Result<Self, serde_json::Error> {
        serde_json::to_vec(value).map(Self)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JsonField<T> {
    Missing,
    Null,
    Value(T),
}

impl<T> JsonField<T> {
    pub fn from_value(value: Option<Option<T>>) -> Self {
        match value {
            None => Self::Missing,
            Some(None) => Self::Null,
            Some(Some(value)) => Self::Value(value),
        }
    }
}

/// Updates a top-level string while preserving byte identity on every no-op or
/// invalid-input path. Mutation may normalize formatting; callers that require
/// raw subtree identity must carry those subtrees as `RawJson`.
pub fn set_top_level_string(data: &[u8], key: &str, value: &str) -> Vec<u8> {
    let Ok(mut root) = serde_json::from_slice::<Value>(data) else {
        return data.to_vec();
    };
    let Some(object) = root.as_object_mut() else {
        return data.to_vec();
    };
    if object.get(key).and_then(Value::as_str) == Some(value) {
        return data.to_vec();
    }
    object.insert(key.to_owned(), Value::String(value.to_owned()));
    serde_json::to_vec(&root).unwrap_or_else(|_| data.to_vec())
}
