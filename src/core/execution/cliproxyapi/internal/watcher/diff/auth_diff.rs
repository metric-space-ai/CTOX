// ref: internal/watcher/diff/auth_diff.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::watcher::synthesizer::context::SynthesizedAuth;
pub fn build_auth_change_details(old: &SynthesizedAuth, new: &SynthesizedAuth) -> Vec<String> {
    let mut changes = Vec::new();
    macro_rules! changed {
        ($field:ident) => {
            if old.$field != new.$field {
                changes.push(format!(
                    "{}: {:?} -> {:?}",
                    stringify!($field),
                    old.$field,
                    new.$field
                ));
            }
        };
    }
    changed!(provider);
    changed!(prefix);
    changed!(label);
    changed!(disabled);
    changed!(priority);
    changed!(weight);
    if old.proxy_url != new.proxy_url {
        changes.push("proxy_url: updated".into());
    }
    if old.excluded_models != new.excluded_models {
        changes.push(format!(
            "excluded_models: {} -> {}",
            old.excluded_models.len(),
            new.excluded_models.len()
        ));
    }
    if old.model_aliases != new.model_aliases {
        changes.push(format!(
            "model_aliases: {} -> {}",
            old.model_aliases.len(),
            new.model_aliases.len()
        ));
    }
    if old.attributes != new.attributes {
        changes.push("attributes: updated (values redacted)".into());
    }
    changes
}
