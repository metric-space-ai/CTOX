// ref: internal/config/parse.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::path::Path;

use super::ProviderCompatConfig;

pub fn parse_provider_compat_config(
    data: &[u8],
) -> Result<ProviderCompatConfig, ProviderCompatConfigError> {
    if data.is_empty() {
        return Err(ProviderCompatConfigError::Empty);
    }
    let mut config = serde_yaml::from_slice::<ProviderCompatConfig>(data)
        .map_err(|error| ProviderCompatConfigError::Parse(error.to_string()))?;
    config.sanitize();
    Ok(config)
}

/// Parses and resolves host-owned paths against an explicit CTOX data root.
/// No process environment is consulted.
pub fn parse_provider_compat_config_with_root(
    data: &[u8],
    data_root: &Path,
) -> Result<ProviderCompatConfig, ProviderCompatConfigError> {
    let mut config = parse_provider_compat_config(data)?;
    config
        .resolve_plugins_dir(data_root)
        .map_err(ProviderCompatConfigError::Path)?;
    Ok(config)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProviderCompatConfigError {
    Empty,
    Parse(String),
    Path(String),
}

impl fmt::Display for ProviderCompatConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("config payload is empty"),
            Self::Parse(message) => write!(formatter, "parse config payload: {message}"),
            Self::Path(message) => write!(formatter, "resolve config path: {message}"),
        }
    }
}

impl std::error::Error for ProviderCompatConfigError {}
