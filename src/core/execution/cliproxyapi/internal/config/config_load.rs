// ref: internal/config/config_load.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::parse::{parse_provider_compat_config_with_root, ProviderCompatConfigError};
use super::ProviderCompatConfig;

pub trait TypedConfigSource {
    fn read(&self) -> io::Result<Vec<u8>>;
    fn description(&self) -> String;
}

/// Writable counterpart to [`TypedConfigSource`]. The embedding host decides
/// whether this is a file, database record, object-store object, or test
/// buffer; configuration helpers never discover persistence authority.
pub trait TypedConfigSink: TypedConfigSource {
    fn write(&self, data: &[u8]) -> io::Result<()>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FileConfigSource {
    path: PathBuf,
}

impl FileConfigSource {
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl TypedConfigSource for FileConfigSource {
    fn read(&self) -> io::Result<Vec<u8>> {
        fs::read(&self.path)
    }

    fn description(&self) -> String {
        self.path.display().to_string()
    }
}

/// Explicit caller-constructed filesystem document. Merely parsing or
/// normalizing configuration never constructs this adapter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FileConfigDocument {
    path: PathBuf,
}

impl FileConfigDocument {
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl TypedConfigSource for FileConfigDocument {
    fn read(&self) -> io::Result<Vec<u8>> {
        fs::read(&self.path)
    }

    fn description(&self) -> String {
        self.path.display().to_string()
    }
}

impl TypedConfigSink for FileConfigDocument {
    fn write(&self, data: &[u8]) -> io::Result<()> {
        fs::write(&self.path, data)
    }
}

pub fn load_config(
    source: &dyn TypedConfigSource,
    data_root: &Path,
    optional: bool,
) -> Result<ProviderCompatConfig, ConfigLoadError> {
    let bytes = match source.read() {
        Ok(bytes) => bytes,
        Err(error)
            if optional
                && matches!(
                    error.kind(),
                    io::ErrorKind::NotFound | io::ErrorKind::IsADirectory
                ) =>
        {
            return empty_config(data_root);
        }
        Err(error) => {
            return Err(ConfigLoadError::Read {
                source: source.description(),
                error,
            });
        }
    };
    if optional && bytes.iter().all(u8::is_ascii_whitespace) {
        return empty_config(data_root);
    }
    match parse_provider_compat_config_with_root(&bytes, data_root) {
        Ok(config) => Ok(config),
        Err(_) if optional => empty_config(data_root),
        Err(error) => Err(ConfigLoadError::Parse(error)),
    }
}

fn empty_config(data_root: &Path) -> Result<ProviderCompatConfig, ConfigLoadError> {
    parse_provider_compat_config_with_root(b"{}", data_root).map_err(ConfigLoadError::Parse)
}

#[derive(Debug)]
pub enum ConfigLoadError {
    Read { source: String, error: io::Error },
    Parse(ProviderCompatConfigError),
}

impl fmt::Display for ConfigLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { source, error } => {
                write!(
                    formatter,
                    "failed to read config source {source:?}: {error}"
                )
            }
            Self::Parse(error) => write!(formatter, "failed to parse config: {error}"),
        }
    }
}

impl std::error::Error for ConfigLoadError {}

#[cfg(test)]
mod tests {
    use super::*;

    struct MemorySource(Result<Vec<u8>, io::ErrorKind>);

    impl TypedConfigSource for MemorySource {
        fn read(&self) -> io::Result<Vec<u8>> {
            self.0.clone().map_err(io::Error::from)
        }

        fn description(&self) -> String {
            "memory".into()
        }
    }

    #[test]
    fn optional_sources_fail_closed_to_typed_defaults() {
        let root = Path::new("/typed/root");
        let missing = MemorySource(Err(io::ErrorKind::NotFound));
        assert_eq!(
            load_config(&missing, root, true).unwrap().plugins.dir,
            "plugins"
        );
        assert!(load_config(&missing, root, false).is_err());
        let invalid = MemorySource(Ok(b"unknown-authority: true\n".to_vec()));
        assert!(load_config(&invalid, root, false).is_err());
        assert_eq!(
            load_config(&invalid, root, true).unwrap().plugins.dir,
            "plugins"
        );
    }
}
