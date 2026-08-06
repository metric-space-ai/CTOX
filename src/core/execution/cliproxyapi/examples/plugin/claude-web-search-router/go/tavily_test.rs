// ref: examples/plugin/claude-web-search-router/go/tavily_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only
use super::{
    claude_response,
    tavily::{SearchTransport, TavilyClient},
    Hit,
};
use std::{cell::RefCell, rc::Rc};
#[derive(Clone, Default)]
struct Mock {
    keys: Rc<RefCell<Vec<String>>>,
}
impl SearchTransport for Mock {
    fn search(&self, key: &str, query: &str, max: usize) -> Result<(Vec<Hit>, String), String> {
        self.keys.borrow_mut().push(key.into());
        Ok((
            vec![Hit {
                title: query.into(),
                url: "https://example.test".into(),
                snippet: max.to_string(),
            }],
            "answer".into(),
        ))
    }
}
#[test]
fn empty_keys_fail_without_transport() {
    let client = TavilyClient::new(vec![], Mock::default());
    assert!(client.search("q", 5).is_err());
}
#[test]
fn keys_rotate_and_response_is_claude_shaped() {
    let transport = Mock::default();
    let keys = transport.keys.clone();
    let client = TavilyClient::new(vec!["a".into(), "b".into()], transport);
    let (hits, answer) = client.search("query", 2).unwrap();
    client.search("again", 2).unwrap();
    assert_eq!(&*keys.borrow(), &["a", "b"]);
    let message = claude_response::message("", "query", &hits, &answer);
    assert_eq!(message["type"], "message");
}
