// ref: internal/watcher/synthesizer/context.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::watcher::config_reload::WatcherConfig;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SynthesizedAuth {
    pub id: String,
    pub provider: String,
    pub prefix: String,
    pub file_name: String,
    pub label: String,
    pub disabled: bool,
    pub priority: i32,
    pub weight: Option<i64>,
    pub proxy_url: String,
    pub attributes: BTreeMap<String, String>,
    pub metadata: BTreeMap<String, Value>,
    pub excluded_models: Vec<String>,
    pub model_aliases: Vec<ModelAlias>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelAlias {
    pub name: String,
    pub alias: String,
    pub display_name: String,
    pub fork: bool,
}

pub trait PluginAuthParser: Send + Sync {
    fn parse(&self, path: &Path, data: &[u8]) -> io::Result<Option<Vec<SynthesizedAuth>>>;
}

pub struct SynthesisContext<'a> {
    pub config: &'a WatcherConfig,
    pub auth_dir: &'a Path,
    pub files: Vec<PathBuf>,
    pub filesystem: Arc<dyn crate::internal::watcher::WatchFilesystem>,
    pub parser: Option<Arc<dyn PluginAuthParser>>,
}
