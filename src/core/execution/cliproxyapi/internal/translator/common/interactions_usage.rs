// ref: internal/translator/common/interactions_usage.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

pub fn interactions_usage(root: &Value) -> Option<&Value> {
    [
        "/interaction/usage",
        "/usage",
        "/metadata/total_usage",
        "/metadata/usage",
        "/interaction/metadata/total_usage",
        "/interaction/metadata/usage",
    ]
    .into_iter()
    .find_map(|path| root.pointer(path))
}

#[cfg(test)]
mod tests {
    use super::interactions_usage;
    use serde_json::json;

    #[test]
    fn uses_the_first_existing_usage_path_in_upstream_order() {
        let root = json!({"interaction":{"usage":{"input":1}},"usage":{"input":2}});
        assert_eq!(interactions_usage(&root), Some(&json!({"input":1})));
        assert!(interactions_usage(&json!({})).is_none());
    }
}
