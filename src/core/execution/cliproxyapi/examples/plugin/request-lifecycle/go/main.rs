// ref: examples/plugin/request-lifecycle/go/main.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only
use super::core::{registration, ExampleRegistration};
use std::collections::BTreeSet;
pub fn example() -> ExampleRegistration {
    registration(
        "example-request-lifecycle",
        &["request_interceptor", "request_lifecycle_plugin"],
    )
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Admission {
    Accepted,
    Duplicate,
    RejectedPolicy,
    RejectedBusy,
}
#[derive(Debug)]
pub struct Lifecycle {
    max: usize,
    keyword: String,
    active: BTreeSet<String>,
}
impl Lifecycle {
    pub fn new(max: usize, keyword: impl Into<String>) -> Result<Self, &'static str> {
        if max == 0 {
            return Err("max_concurrency must be greater than zero");
        }
        Ok(Self {
            max,
            keyword: keyword.into().trim().to_owned(),
            active: BTreeSet::new(),
        })
    }
    pub fn admit(&mut self, id: &str, body: &[u8]) -> Result<Admission, &'static str> {
        if id.is_empty() {
            return Err("request ID is required");
        } else if self.active.contains(id) {
            return Ok(Admission::Duplicate);
        } else if !self.keyword.is_empty() && String::from_utf8_lossy(body).contains(&self.keyword)
        {
            return Ok(Admission::RejectedPolicy);
        } else if self.active.len() >= self.max {
            return Ok(Admission::RejectedBusy);
        }
        self.active.insert(id.to_owned());
        Ok(Admission::Accepted)
    }
    pub fn complete(&mut self, id: &str) {
        self.active.remove(id);
    }
}
#[path = "main_test.rs"]
mod main_test;
