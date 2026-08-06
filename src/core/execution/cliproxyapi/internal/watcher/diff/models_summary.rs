// ref: internal/watcher/diff/models_summary.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::watcher::config_reload::ModelRoute;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelsSummary {
    pub count: usize,
    pub names: Vec<String>,
    pub aliases: Vec<String>,
}
pub fn summarize_models(models: &[ModelRoute]) -> ModelsSummary {
    let mut names = models
        .iter()
        .map(|model| model.name.trim().to_owned())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    let mut aliases = models
        .iter()
        .map(|model| model.alias.trim().to_owned())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    aliases.sort();
    aliases.dedup();
    ModelsSummary {
        count: models.len(),
        names,
        aliases,
    }
}
