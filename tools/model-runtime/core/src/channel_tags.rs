//! Parser for Gemma 4 channel-tag reasoning blocks.
//!
//! Gemma 4 emits hidden reasoning in the form:
//! `<|channel>thought\n...<channel|>`
//! and may start generation with a bare `thought\n` prefix after `<|think|>`.

const CHANNEL_OPEN_TAG: &str = "<|channel>";
const CHANNEL_CLOSE_TAG: &str = "<channel|>";
const THOUGHT_PREFIX: &str = "thought\n";

pub fn is_channel_tag_template(template: &str) -> bool {
    template.contains(CHANNEL_OPEN_TAG) && template.contains(CHANNEL_CLOSE_TAG)
}

pub struct ChannelTagContext {
    accumulated_content: String,
    accumulated_reasoning: String,
    in_reasoning_block: bool,
    buffer: String,
    sent_content_len: usize,
    sent_reasoning_len: usize,
    utf8_buffer: Vec<u8>,
    pending_strip_prefix: bool,
    allow_implicit_thought_prefix: bool,
}

impl ChannelTagContext {
    pub fn new() -> Self {
        Self {
            accumulated_content: String::new(),
            accumulated_reasoning: String::new(),
            in_reasoning_block: false,
            buffer: String::new(),
            sent_content_len: 0,
            sent_reasoning_len: 0,
            utf8_buffer: Vec::new(),
            pending_strip_prefix: false,
            allow_implicit_thought_prefix: false,
        }
    }

    pub fn new_with_implicit_thinking() -> Self {
        Self {
            allow_implicit_thought_prefix: true,
            ..Self::new()
        }
    }

    pub fn process_bytes(&mut self, bytes: &[u8]) {
        self.utf8_buffer.extend_from_slice(bytes);
        let buffer = std::mem::take(&mut self.utf8_buffer);

        let valid_up_to = match std::str::from_utf8(&buffer) {
            Ok(text) => {
                self.process_text(text);
                return;
            }
            Err(err) => err.valid_up_to(),
        };

        if valid_up_to > 0 {
            let valid_str = unsafe { std::str::from_utf8_unchecked(&buffer[..valid_up_to]) };
            self.process_text(valid_str);
        }

        self.utf8_buffer = buffer[valid_up_to..].to_vec();
    }

    pub fn process_text(&mut self, text: &str) {
        self.buffer.push_str(text);

        loop {
            if self.in_reasoning_block {
                if self.pending_strip_prefix {
                    if self.buffer.starts_with(THOUGHT_PREFIX) {
                        self.buffer = self.buffer[THOUGHT_PREFIX.len()..].to_string();
                        self.pending_strip_prefix = false;
                    } else if THOUGHT_PREFIX.starts_with(&self.buffer) {
                        break;
                    } else {
                        self.pending_strip_prefix = false;
                    }
                }

                if let Some(end_pos) = self.buffer.find(CHANNEL_CLOSE_TAG) {
                    self.accumulated_reasoning.push_str(&self.buffer[..end_pos]);
                    self.in_reasoning_block = false;
                    self.buffer = self.buffer[end_pos + CHANNEL_CLOSE_TAG.len()..].to_string();
                    continue;
                }

                let partial_len = self.potential_partial_tag_len(CHANNEL_CLOSE_TAG);
                if partial_len > 0 {
                    let safe_len = self.buffer.len() - partial_len;
                    if safe_len > 0 {
                        self.accumulated_reasoning
                            .push_str(&self.buffer[..safe_len]);
                        self.buffer = self.buffer[safe_len..].to_string();
                    }
                } else {
                    self.accumulated_reasoning.push_str(&self.buffer);
                    self.buffer.clear();
                }
                break;
            }

            if self.allow_implicit_thought_prefix && self.accumulated_content.is_empty() {
                if self.buffer.starts_with(THOUGHT_PREFIX) {
                    self.buffer = self.buffer[THOUGHT_PREFIX.len()..].to_string();
                    self.in_reasoning_block = true;
                    self.allow_implicit_thought_prefix = false;
                    continue;
                }
                if THOUGHT_PREFIX.starts_with(&self.buffer) {
                    break;
                }
                self.allow_implicit_thought_prefix = false;
            }

            if let Some(start_pos) = self.buffer.find(CHANNEL_OPEN_TAG) {
                self.accumulated_content.push_str(&self.buffer[..start_pos]);
                self.in_reasoning_block = true;
                self.pending_strip_prefix = true;
                self.buffer = self.buffer[start_pos + CHANNEL_OPEN_TAG.len()..].to_string();
                continue;
            }

            let partial_len = self.potential_partial_tag_len(CHANNEL_OPEN_TAG);
            if partial_len > 0 {
                let safe_len = self.buffer.len() - partial_len;
                if safe_len > 0 {
                    self.accumulated_content.push_str(&self.buffer[..safe_len]);
                    self.buffer = self.buffer[safe_len..].to_string();
                }
            } else {
                self.accumulated_content.push_str(&self.buffer);
                self.buffer.clear();
            }
            break;
        }
    }

    fn potential_partial_tag_len(&self, tag: &str) -> usize {
        let buffer_bytes = self.buffer.as_bytes();
        let tag_bytes = tag.as_bytes();

        for suffix_len in 1..=tag.len().min(self.buffer.len()) {
            let suffix_start = self.buffer.len() - suffix_len;
            if buffer_bytes[suffix_start..] == tag_bytes[..suffix_len] {
                return suffix_len;
            }
        }
        0
    }

    pub fn get_content_delta(&mut self) -> Option<String> {
        if self.accumulated_content.len() > self.sent_content_len {
            let delta = self.accumulated_content[self.sent_content_len..].to_string();
            self.sent_content_len = self.accumulated_content.len();
            (!delta.is_empty()).then_some(delta)
        } else {
            None
        }
    }

    pub fn get_reasoning_delta(&mut self) -> Option<String> {
        if self.accumulated_reasoning.len() > self.sent_reasoning_len {
            let delta = self.accumulated_reasoning[self.sent_reasoning_len..].to_string();
            self.sent_reasoning_len = self.accumulated_reasoning.len();
            (!delta.is_empty()).then_some(delta)
        } else {
            None
        }
    }

    pub fn content(&self) -> Option<String> {
        (!self.accumulated_content.is_empty()).then(|| self.accumulated_content.clone())
    }

    pub fn reasoning_content(&self) -> Option<String> {
        (!self.accumulated_reasoning.is_empty()).then(|| self.accumulated_reasoning.clone())
    }

    pub fn finalize(&mut self) {
        if !self.utf8_buffer.is_empty() {
            let pending = std::mem::take(&mut self.utf8_buffer);
            if let Ok(text) = String::from_utf8(pending) {
                self.process_text(&text);
            }
        }

        if self.in_reasoning_block {
            if self.pending_strip_prefix {
                if self.buffer.starts_with(THOUGHT_PREFIX) {
                    self.buffer = self.buffer[THOUGHT_PREFIX.len()..].to_string();
                }
                self.pending_strip_prefix = false;
            }
            self.accumulated_reasoning.push_str(&self.buffer);
        } else {
            self.accumulated_content.push_str(&self.buffer);
        }
        self.buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gemma_channel_blocks() {
        let mut ctx = ChannelTagContext::new();
        ctx.process_text("<|channel>thought\nreasoning<channel|>final");
        assert_eq!(ctx.reasoning_content().as_deref(), Some("reasoning"));
        assert_eq!(ctx.content().as_deref(), Some("final"));
    }

    #[test]
    fn parses_implicit_thinking_prefix() {
        let mut ctx = ChannelTagContext::new_with_implicit_thinking();
        ctx.process_text("thought\nreasoning<channel|>final");
        ctx.finalize();
        assert_eq!(ctx.reasoning_content().as_deref(), Some("reasoning"));
        assert_eq!(ctx.content().as_deref(), Some("final"));
    }
}
