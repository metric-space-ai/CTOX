// ref: internal/watcher/clients.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::runtime::{AuthUpdate, AuthUpdateAction, WatcherDependencies, WatcherState};
use super::synthesizer::config::ConfigSynthesizer;
use super::synthesizer::context::{SynthesisContext, SynthesizedAuth};
use super::synthesizer::file::FileSynthesizer;
use super::synthesizer::interface::AuthSynthesizer;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub(super) fn reload_clients(
    dependencies: &WatcherDependencies,
    auth_dir: &Path,
    state: &mut WatcherState,
) {
    let mut files = dependencies
        .filesystem
        .list_files(auth_dir)
        .unwrap_or_default();
    files.retain(|path| super::events::is_auth_file(path));
    files.sort();
    let context = SynthesisContext {
        config: &state.config,
        auth_dir,
        files: files.clone(),
        filesystem: dependencies.filesystem.clone(),
        parser: dependencies.plugin_parser.clone(),
    };
    let mut auths = ConfigSynthesizer::new()
        .synthesize(&context)
        .unwrap_or_default();
    auths.extend(
        FileSynthesizer::new()
            .synthesize(&context)
            .unwrap_or_default(),
    );
    let new_auths = auth_slice_to_map(auths);
    let updates = compute_updates(&state.auths, &new_auths);
    state.auths = new_auths;
    state.auth_hashes = files
        .iter()
        .filter_map(|path| {
            dependencies
                .filesystem
                .read(path)
                .ok()
                .map(|bytes| (path.clone(), format!("{:x}", Sha256::digest(bytes))))
        })
        .collect();
    if !updates.is_empty() {
        dependencies.dispatcher.dispatch(updates);
        let _ = dependencies.persistence_sink.persist_auth(&files);
    }
}

pub fn auth_slice_to_map(auths: Vec<SynthesizedAuth>) -> BTreeMap<String, SynthesizedAuth> {
    auths
        .into_iter()
        .filter(|auth| !auth.id.trim().is_empty())
        .map(|auth| (auth.id.clone(), auth))
        .collect()
}

pub fn compute_updates(
    old: &BTreeMap<String, SynthesizedAuth>,
    new: &BTreeMap<String, SynthesizedAuth>,
) -> Vec<AuthUpdate> {
    let mut updates = Vec::new();
    for (id, auth) in old {
        if !new.contains_key(id) {
            updates.push(AuthUpdate {
                action: AuthUpdateAction::Delete,
                auth: auth.clone(),
            });
        }
    }
    for (id, auth) in new {
        match old.get(id) {
            None => updates.push(AuthUpdate {
                action: AuthUpdateAction::Add,
                auth: auth.clone(),
            }),
            Some(previous) if !super::dispatcher::auth_equal(previous, auth) => {
                updates.push(AuthUpdate {
                    action: AuthUpdateAction::Modify,
                    auth: auth.clone(),
                })
            }
            _ => {}
        }
    }
    updates
}

pub fn auth_file_unchanged(path: &Path, bytes: &[u8], hashes: &BTreeMap<PathBuf, String>) -> bool {
    hashes
        .get(path)
        .is_some_and(|hash| *hash == format!("{:x}", Sha256::digest(bytes)))
}
