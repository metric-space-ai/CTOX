// ref: internal/runtime/executor/claude_signing.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;
use url::Url;

const CLAUDE_CCH_SEED: u64 = 0x4D65_9218_E32A_3268;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClaudeCchUpstreamKind {
    Other,
    Anthropic,
    Vertex,
}

pub fn claude_cch_signing_enabled(
    api_key: &str,
    kind: ClaudeCchUpstreamKind,
    endpoint: &str,
) -> bool {
    if api_key.trim().starts_with("sk-ant-oat") {
        return true;
    }
    if kind == ClaudeCchUpstreamKind::Vertex {
        return true;
    }
    if kind != ClaudeCchUpstreamKind::Anthropic {
        return false;
    }
    Url::parse(endpoint.trim()).ok().is_some_and(|url| {
        url.username().is_empty()
            && url.password().is_none()
            && url.scheme().eq_ignore_ascii_case("https")
            && url
                .host_str()
                .is_some_and(|host| host.eq_ignore_ascii_case("api.anthropic.com"))
            && url.port().is_none_or(|port| port == 443)
            && url.path().contains("/v1/messages")
    })
}

pub fn finalize_anthropic_messages_body_cch(body: &[u8], fallback_billing: &str) -> Vec<u8> {
    let mut root = match serde_json::from_slice::<Value>(body) {
        Ok(root) => root,
        Err(_) => return body.to_vec(),
    };
    let billing = root
        .pointer("/system/0/text")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !billing.starts_with("x-anthropic-billing-header:") && !fallback_billing.is_empty() {
        let block = serde_json::json!({"type":"text", "text": fallback_billing});
        let system = root
            .as_object_mut()
            .and_then(|object| object.remove("system"));
        let mut items = vec![block];
        match system {
            Some(Value::Array(existing)) => items.extend(existing),
            Some(Value::String(text)) => {
                items.push(serde_json::json!({"type":"text", "text":text}))
            }
            _ => {}
        }
        if let Some(object) = root.as_object_mut() {
            object.insert("system".to_owned(), Value::Array(items));
        }
    }
    let billing = root
        .pointer("/system/0/text")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    if !billing.contains("cch=") {
        if let Some(index) = billing
            .find("cc_entrypoint=")
            .and_then(|start| billing[start..].find(';').map(|end| start + end + 1))
        {
            let mut updated = billing;
            updated.insert_str(index, " cch=00000;");
            if let Some(value) = root.pointer_mut("/system/0/text") {
                *value = Value::String(updated);
            }
        }
    }
    let encoded = serde_json::to_vec(&root).unwrap_or_else(|_| body.to_vec());
    sign_anthropic_messages_body(&encoded)
}

pub fn sign_anthropic_messages_body(body: &[u8]) -> Vec<u8> {
    let valid_header = serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|root| {
            root.pointer("/system/0/text")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .is_some_and(|text| text.starts_with("x-anthropic-billing-header:"));
    if !valid_header {
        return body.to_vec();
    }
    let Some(cch_start) = find_cch_value(body) else {
        return body.to_vec();
    };
    let mut unsigned = body.to_vec();
    unsigned[cch_start..cch_start + 5].copy_from_slice(b"00000");
    let normalized = normalize_claude_cch_input(&unsigned).unwrap_or_else(|| unsigned.clone());
    let cch = format!("{:05x}", xxhash64(&normalized, CLAUDE_CCH_SEED) & 0xF_FFFF);
    let mut signed = unsigned;
    signed[cch_start..cch_start + 5].copy_from_slice(cch.as_bytes());
    signed
}

/// Produces Claude Code's byte-preserving hash view: every `model` string is
/// emptied and dispatch-only members are removed at every object depth.
pub fn normalize_claude_cch_input(body: &[u8]) -> Option<Vec<u8>> {
    serde_json::from_slice::<Value>(body).ok()?;
    let mut scanner = CchScanner {
        body,
        pos: 0,
        edits: Vec::new(),
    };
    scanner.parse_value(true)?;
    scanner.skip_ws();
    if scanner.pos != body.len() {
        return None;
    }
    scanner.edits.sort_unstable_by_key(|edit| edit.0);
    let mut output = Vec::with_capacity(body.len());
    let mut last = 0;
    for (start, end) in scanner.edits {
        if start < last || end > body.len() {
            return None;
        }
        output.extend_from_slice(&body[last..start]);
        last = end;
    }
    output.extend_from_slice(&body[last..]);
    Some(output)
}

#[derive(Clone, Copy)]
struct CchMember {
    start: usize,
    end: usize,
    before: Option<usize>,
    after: Option<usize>,
    excluded: bool,
}
struct CchScanner<'a> {
    body: &'a [u8],
    pos: usize,
    edits: Vec<(usize, usize)>,
}
impl CchScanner<'_> {
    fn parse_value(&mut self, collect: bool) -> Option<()> {
        self.skip_ws();
        match *self.body.get(self.pos)? {
            b'{' => self.parse_object(collect),
            b'[' => self.parse_array(collect),
            b'"' => self.parse_string().map(|_| ()),
            _ => {
                let start = self.pos;
                while self.body.get(self.pos).is_some_and(|byte| {
                    !matches!(byte, b',' | b'}' | b']' | b' ' | b'\t' | b'\r' | b'\n')
                }) {
                    self.pos += 1;
                }
                (self.pos > start).then_some(())
            }
        }
    }
    fn parse_object(&mut self, collect: bool) -> Option<()> {
        self.pos += 1;
        self.skip_ws();
        if self.consume(b'}') {
            return Some(());
        }
        let mut members = Vec::new();
        let mut before = None;
        loop {
            self.skip_ws();
            let start = self.pos;
            let (key_start, key_end) = self.parse_string()?;
            self.skip_ws();
            if !self.consume(b':') {
                return None;
            }
            self.skip_ws();
            let key = &self.body[key_start..key_end];
            let excluded = collect
                && matches!(
                    key,
                    b"\"max_tokens\"" | b"\"fallbacks\"" | b"\"fallback_credit_token\""
                );
            if collect && key == b"\"model\"" && self.body.get(self.pos) == Some(&b'"') {
                let (value_start, value_end) = self.parse_string()?;
                self.edits.push((value_start + 1, value_end - 1));
            } else {
                self.parse_value(collect && !excluded)?;
            }
            let end = self.pos;
            self.skip_ws();
            let after = if self.consume(b',') {
                Some(self.pos - 1)
            } else {
                None
            };
            members.push(CchMember {
                start,
                end,
                before,
                after,
                excluded,
            });
            if after.is_some() {
                before = after;
                continue;
            }
            if !self.consume(b'}') {
                return None;
            }
            break;
        }
        if collect {
            self.exclude_members(&members);
        }
        Some(())
    }
    fn parse_array(&mut self, collect: bool) -> Option<()> {
        self.pos += 1;
        self.skip_ws();
        if self.consume(b']') {
            return Some(());
        }
        loop {
            self.parse_value(collect)?;
            self.skip_ws();
            if self.consume(b',') {
                continue;
            }
            if !self.consume(b']') {
                return None;
            }
            return Some(());
        }
    }
    fn parse_string(&mut self) -> Option<(usize, usize)> {
        if !self.consume(b'"') {
            return None;
        }
        let start = self.pos - 1;
        while self.pos < self.body.len() {
            match self.body[self.pos] {
                b'\\' => self.pos += 2,
                b'"' => {
                    self.pos += 1;
                    return Some((start, self.pos));
                }
                _ => self.pos += 1,
            }
        }
        None
    }
    fn exclude_members(&mut self, members: &[CchMember]) {
        let mut start = 0;
        while start < members.len() {
            if !members[start].excluded {
                start += 1;
                continue;
            }
            let mut end = start;
            while end + 1 < members.len() && members[end + 1].excluded {
                end += 1;
            }
            let edit = if end + 1 < members.len() {
                (
                    members[start].start,
                    members[end]
                        .after
                        .map_or(members[end].end, |comma| comma + 1),
                )
            } else if start > 0 && end > start {
                (members[start].start, members[end].end)
            } else if start > 0 {
                (
                    members[start].before.unwrap_or(members[start].start),
                    members[end].end,
                )
            } else {
                (members[start].start, members[end].end)
            };
            if edit.0 < edit.1 {
                self.edits.push(edit);
            }
            start = end + 1;
        }
    }
    fn skip_ws(&mut self) {
        while self.body.get(self.pos).is_some_and(u8::is_ascii_whitespace) {
            self.pos += 1;
        }
    }
    fn consume(&mut self, byte: u8) -> bool {
        if self.body.get(self.pos) == Some(&byte) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
}

fn find_cch_value(body: &[u8]) -> Option<usize> {
    body.windows(4)
        .position(|window| window == b"cch=")
        .map(|index| index + 4)
        .filter(|start| {
            body.get(*start..*start + 5)
                .is_some_and(|value| value.iter().all(u8::is_ascii_hexdigit))
                && body.get(*start + 5) == Some(&b';')
        })
}

fn xxhash64(input: &[u8], seed: u64) -> u64 {
    const P1: u64 = 11_400_714_785_074_694_791;
    const P2: u64 = 14_029_467_366_897_019_727;
    const P3: u64 = 1_609_587_929_392_839_161;
    const P4: u64 = 9_650_029_242_287_828_579;
    const P5: u64 = 2_870_177_450_012_600_261;
    fn round(mut accumulator: u64, lane: u64) -> u64 {
        accumulator = accumulator.wrapping_add(lane.wrapping_mul(P2));
        accumulator.rotate_left(31).wrapping_mul(P1)
    }
    fn merge(accumulator: u64, lane: u64) -> u64 {
        (accumulator ^ round(0, lane))
            .wrapping_mul(P1)
            .wrapping_add(P4)
    }
    fn read(bytes: &[u8]) -> u64 {
        u64::from_le_bytes(bytes.try_into().expect("fixed-width xxhash lane"))
    }

    let mut offset = 0;
    let mut hash = if input.len() >= 32 {
        let mut v1 = seed.wrapping_add(P1).wrapping_add(P2);
        let mut v2 = seed.wrapping_add(P2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(P1);
        while offset <= input.len() - 32 {
            v1 = round(v1, read(&input[offset..offset + 8]));
            v2 = round(v2, read(&input[offset + 8..offset + 16]));
            v3 = round(v3, read(&input[offset + 16..offset + 24]));
            v4 = round(v4, read(&input[offset + 24..offset + 32]));
            offset += 32;
        }
        let accumulator = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));
        merge(merge(merge(merge(accumulator, v1), v2), v3), v4)
    } else {
        seed.wrapping_add(P5)
    };
    hash = hash.wrapping_add(input.len() as u64);
    while offset + 8 <= input.len() {
        let lane = round(0, read(&input[offset..offset + 8]));
        hash = (hash ^ lane)
            .rotate_left(27)
            .wrapping_mul(P1)
            .wrapping_add(P4);
        offset += 8;
    }
    if offset + 4 <= input.len() {
        let lane = u32::from_le_bytes(input[offset..offset + 4].try_into().unwrap()) as u64;
        hash = (hash ^ lane.wrapping_mul(P1))
            .rotate_left(23)
            .wrapping_mul(P2)
            .wrapping_add(P3);
        offset += 4;
    }
    while offset < input.len() {
        hash = (hash ^ u64::from(input[offset]).wrapping_mul(P5))
            .rotate_left(11)
            .wrapping_mul(P1);
        offset += 1;
    }
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(P2);
    hash ^= hash >> 29;
    hash = hash.wrapping_mul(P3);
    hash ^ (hash >> 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xxhash_matches_reference_vector() {
        assert_eq!(xxhash64(b"", 0), 0xef46_db37_51d8_e999);
    }

    #[test]
    fn signing_is_byte_stable_and_verifiable() {
        let body = br#"{"system":[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.63.abc; cc_entrypoint=cli; cch=00000;"}],"messages":[]}"#;
        let signed = sign_anthropic_messages_body(body);
        assert_eq!(signed.len(), body.len());
        assert_ne!(signed, body);
        assert_eq!(sign_anthropic_messages_body(&signed), signed);
        assert_eq!(
            sign_anthropic_messages_body(br#"{"system":[]}"#),
            br#"{"system":[]}"#
        );
    }
}
