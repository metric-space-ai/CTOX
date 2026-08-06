// ref: internal/runtime/executor/helps/cloak_obfuscate.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

const ZERO_WIDTH_SPACE: char = '\u{200b}';

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SensitiveWordMatcher {
    words: Vec<String>,
}

pub fn build_sensitive_word_matcher(words: &[String]) -> Option<SensitiveWordMatcher> {
    let mut words: Vec<String> = words
        .iter()
        .map(|word| word.trim())
        .filter(|word| word.chars().count() >= 2 && !word.contains(ZERO_WIDTH_SPACE))
        .map(str::to_owned)
        .collect();
    words.sort_by_key(|word| std::cmp::Reverse(word.len()));
    words.dedup_by(|left, right| left.eq_ignore_ascii_case(right));
    (!words.is_empty()).then_some(SensitiveWordMatcher { words })
}

pub fn obfuscate_sensitive_words(
    payload: &[u8],
    matcher: Option<&SensitiveWordMatcher>,
) -> Vec<u8> {
    let Some(matcher) = matcher else {
        return payload.to_vec();
    };
    let Ok(mut root) = serde_json::from_slice::<Value>(payload) else {
        return payload.to_vec();
    };
    let mut changed = false;
    if let Some(system) = root.get_mut("system") {
        match system {
            Value::String(text) => changed |= matcher.obfuscate_string(text),
            Value::Array(blocks) => {
                for block in blocks {
                    if block.get("type").and_then(Value::as_str) == Some("text") {
                        if let Some(text) = block
                            .get_mut("text")
                            .and_then(|value| value.as_str())
                            .map(str::to_owned)
                        {
                            let obfuscated = matcher.obfuscate_text(&text);
                            if obfuscated != text {
                                block["text"] = Value::String(obfuscated);
                                changed = true;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    if let Some(messages) = root.get_mut("messages").and_then(Value::as_array_mut) {
        for message in messages {
            let Some(content) = message.get_mut("content") else {
                continue;
            };
            match content {
                Value::String(text) => changed |= matcher.obfuscate_string(text),
                Value::Array(blocks) => {
                    for block in blocks {
                        if block.get("type").and_then(Value::as_str) == Some("text") {
                            if let Some(text) = block
                                .get_mut("text")
                                .and_then(|value| value.as_str())
                                .map(str::to_owned)
                            {
                                let obfuscated = matcher.obfuscate_text(&text);
                                if obfuscated != text {
                                    block["text"] = Value::String(obfuscated);
                                    changed = true;
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    if changed {
        serde_json::to_vec(&root).unwrap_or_else(|_| payload.to_vec())
    } else {
        payload.to_vec()
    }
}

impl SensitiveWordMatcher {
    pub fn obfuscate_text(&self, text: &str) -> String {
        let mut output = String::with_capacity(text.len());
        let mut offset = 0;
        while offset < text.len() {
            let remainder = &text[offset..];
            let matched = self.words.iter().find(|word| {
                remainder
                    .get(..word.len())
                    .is_some_and(|candidate| candidate.eq_ignore_ascii_case(word))
            });
            if let Some(word) = matched {
                let original = &remainder[..word.len()];
                let mut chars = original.chars();
                if let Some(first) = chars.next() {
                    output.push(first);
                    output.push(ZERO_WIDTH_SPACE);
                    output.extend(chars);
                } else {
                    output.push_str(original);
                }
                offset += word.len();
            } else {
                let next = remainder.chars().next().expect("non-empty remainder");
                output.push(next);
                offset += next.len_utf8();
            }
        }
        output
    }

    fn obfuscate_string(&self, text: &mut String) -> bool {
        let output = self.obfuscate_text(text);
        if output == *text {
            return false;
        }
        *text = output;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matcher_filters_sorts_and_obfuscates_case_insensitively() {
        let matcher = build_sensitive_word_matcher(&[
            " a ".to_owned(),
            "secret".to_owned(),
            "secret phrase".to_owned(),
        ])
        .unwrap();
        assert_eq!(
            matcher.obfuscate_text("SECRET PHRASE and secret"),
            "S\u{200b}ECRET PHRASE and s\u{200b}ecret"
        );
    }

    #[test]
    fn only_system_and_text_message_blocks_are_obfuscated() {
        let matcher = build_sensitive_word_matcher(&["secret".to_owned()]).unwrap();
        let output = obfuscate_sensitive_words(
            br#"{"system":[{"type":"text","text":"secret"},{"type":"image","text":"secret"}],"messages":[{"content":"SECRET"},{"content":[{"type":"text","text":"a secret"},{"type":"tool_result","content":"secret"}]}]}"#,
            Some(&matcher),
        );
        let output: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(output["system"][0]["text"], "s\u{200b}ecret");
        assert_eq!(output["system"][1]["text"], "secret");
        assert_eq!(output["messages"][0]["content"], "S\u{200b}ECRET");
        assert_eq!(output["messages"][1]["content"][1]["content"], "secret");
    }
}
