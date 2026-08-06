// ref: examples/plugin/claude-web-search-router/go/tavily.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only
use super::Hit;
use std::cell::Cell;
pub trait SearchTransport {
    fn search(
        &self,
        key: &str,
        query: &str,
        max_results: usize,
    ) -> Result<(Vec<Hit>, String), String>;
}
pub struct TavilyClient<T> {
    keys: Vec<String>,
    next: Cell<usize>,
    transport: T,
}
impl<T: SearchTransport> TavilyClient<T> {
    pub fn new(keys: Vec<String>, transport: T) -> Self {
        Self {
            keys: keys
                .into_iter()
                .filter_map(|k| {
                    let k = k.trim().to_owned();
                    (!k.is_empty()).then_some(k)
                })
                .collect(),
            next: Cell::new(0),
            transport,
        }
    }
    pub fn available(&self) -> bool {
        !self.keys.is_empty()
    }
    pub fn search(&self, query: &str, max: usize) -> Result<(Vec<Hit>, String), String> {
        if !self.available() {
            return Err("tavily_api_keys is empty".into());
        }
        let query = query.trim();
        if query.is_empty() {
            return Err("web search query is empty".into());
        }
        let i = self.next.get();
        self.next.set(i + 1);
        self.transport.search(
            &self.keys[i % self.keys.len()],
            query,
            if max == 0 { 5 } else { max },
        )
    }
}
