// ref: examples/plugin/claude-web-search-router/go/penalty.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only
use super::Backend;
use std::collections::BTreeMap;
#[derive(Default)]
pub struct Penalties(BTreeMap<Backend, u32>);
impl Penalties {
    pub fn failure(&mut self, b: Backend) {
        *self.0.entry(b).or_default() += 5
    }
    pub fn success(&mut self, b: Backend) {
        let score = self.0.entry(b).or_default();
        *score = score.saturating_sub(1)
    }
    pub fn sorted(&self, input: &[Backend]) -> Vec<Backend> {
        let mut indexed = input.iter().copied().enumerate().collect::<Vec<_>>();
        indexed.sort_by_key(|(i, b)| (self.0.get(b).copied().unwrap_or(0), *i));
        indexed.into_iter().map(|(_, b)| b).collect()
    }
}
