// ref: internal/httpfetch @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: supplemental
// License: AGPL-3.0-only

#[path = "httpfetch.rs"]
mod implementation;

pub use implementation::{
    get_bytes, BodyChunkFuture, FetchFuture, FetchResponse, Headers, HttpDoer, HttpFetchError,
    ResponseBody,
};

#[cfg(test)]
#[path = "httpfetch_test.rs"]
mod httpfetch_test;
