// ref: examples/plugin/claude-web-search-router/go/claude_response.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only
use super::Hit;
use serde_json::json;
pub fn answer_text(answer: &str, hits: &[Hit]) -> String {
    if !answer.trim().is_empty() {
        return answer.into();
    }
    if hits.is_empty() {
        return "No web search results were returned.".into();
    }
    hits.iter()
        .map(|h| {
            [h.title.as_str(), h.url.as_str(), h.snippet.as_str()]
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}
pub fn message(model: &str, query: &str, hits: &[Hit], answer: &str) -> serde_json::Value {
    json!({"id":"msg_fixture","type":"message","role":"assistant","model":if model.trim().is_empty(){"claude-sonnet-4-6"}else{model},"content":[{"type":"server_tool_use","id":"srvtoolu_fixture","name":"web_search","input":{"query":query}},{"type":"web_search_tool_result","tool_use_id":"srvtoolu_fixture","content":hits.iter().map(|h|json!({"type":"web_search_result","title":h.title,"url":h.url,"page_age":null})).collect::<Vec<_>>()},{"type":"text","text":answer_text(answer,hits)}],"stop_reason":"end_turn"})
}
