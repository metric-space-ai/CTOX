// ref: sdk/access/types.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const ACCESS_PROVIDER_TYPE_CONFIG_API_KEY: &str = "config-api-key";
pub const DEFAULT_ACCESS_PROVIDER_NAME: &str = "config-inline";

/// Groups request authentication providers.
///
/// `None` is Go's nil slice. `Some(Vec::new())` is a non-nil empty slice. Both
/// serialize as omitted because upstream uses `omitempty`, but they remain
/// distinguishable while the configuration is in memory.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AccessConfig {
    #[serde(
        skip_serializing_if = "option_vec_is_none_or_empty",
        rename = "providers"
    )]
    pub providers: Option<Vec<AccessProvider>>,
}

impl AccessConfig {
    #[must_use]
    pub fn providers(&self) -> &[AccessProvider] {
        self.providers.as_deref().unwrap_or_default()
    }
}

/// Describes a request authentication provider entry.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AccessProvider {
    /// Instance identifier for the provider.
    #[serde(rename = "name")]
    pub name: String,

    /// Provider implementation registered via the SDK.
    #[serde(rename = "type")]
    pub provider_type: String,

    /// Optional third-party SDK module providing this provider.
    #[serde(skip_serializing_if = "String::is_empty", rename = "sdk")]
    pub sdk: String,

    /// Inline keys for providers that require them.
    ///
    /// The option preserves nil versus non-nil slices in memory.
    #[serde(
        skip_serializing_if = "option_vec_is_none_or_empty",
        rename = "api-keys"
    )]
    pub api_keys: Option<Vec<String>>,

    /// Provider-specific JSON-compatible options.
    ///
    /// The option preserves nil versus non-nil maps in memory.
    #[serde(skip_serializing_if = "option_map_is_none_or_empty", rename = "config")]
    pub config: Option<BTreeMap<String, Value>>,
}

fn option_vec_is_none_or_empty<T>(value: &Option<Vec<T>>) -> bool {
    value.as_ref().is_none_or(Vec::is_empty)
}

fn option_map_is_none_or_empty<K, V>(value: &Option<BTreeMap<K, V>>) -> bool {
    value.as_ref().is_none_or(BTreeMap::is_empty)
}

/// Constructs an inline API-key provider configuration.
///
/// As upstream does, this returns `None` for both nil-equivalent and empty key
/// slices and owns a clone of every supplied key.
#[must_use]
pub fn make_inline_api_key_provider(keys: &[String]) -> Option<AccessProvider> {
    if keys.is_empty() {
        return None;
    }
    Some(AccessProvider {
        name: DEFAULT_ACCESS_PROVIDER_NAME.to_owned(),
        provider_type: ACCESS_PROVIDER_TYPE_CONFIG_API_KEY.to_owned(),
        api_keys: Some(keys.to_vec()),
        ..AccessProvider::default()
    })
}
