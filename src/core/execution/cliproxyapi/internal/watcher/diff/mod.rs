// Origin: CTOX module graph for watcher diff helpers.
// License: AGPL-3.0-only

pub mod auth_diff;
pub mod config_diff;
pub mod model_hash;
pub mod models_summary;
pub mod oauth_excluded;
pub mod oauth_model_alias;
pub mod openai_compat;

#[cfg(test)]
mod config_diff_test;
#[cfg(test)]
mod model_hash_test;
#[cfg(test)]
mod oauth_excluded_test;
#[cfg(test)]
mod oauth_model_alias_test;
#[cfg(test)]
mod openai_compat_test;
