// refs: internal/translator/*/*/*_response.go @ ffdb9c9fbc78a6235d59c9ccbdc4243ba35ecdcd
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: Vec<u8>,
    pub id: Option<String>,
    pub retry_millis: Option<u64>,
}

#[derive(Debug, Default)]
pub struct SseDecoder {
    buffer: Vec<u8>,
    event: SseEvent,
    data_lines: Vec<Vec<u8>>,
}

impl SseDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        self.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some(newline) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line: Vec<u8> = self.buffer.drain(..=newline).collect();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if let Some(event) = self.consume_line(&line) {
                events.push(event);
            }
        }
        events
    }

    pub fn finish(&mut self) -> Vec<SseEvent> {
        let mut events = Vec::new();
        if !self.buffer.is_empty() {
            let mut line = std::mem::take(&mut self.buffer);
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if let Some(event) = self.consume_line(&line) {
                events.push(event);
            }
        }
        if let Some(event) = self.dispatch() {
            events.push(event);
        }
        events
    }

    fn consume_line(&mut self, line: &[u8]) -> Option<SseEvent> {
        if line.is_empty() {
            return self.dispatch();
        }
        if line.first() == Some(&b':') {
            return None;
        }
        let (field, value) =
            line.iter()
                .position(|byte| *byte == b':')
                .map_or((line, &[][..]), |colon| {
                    let value = &line[colon + 1..];
                    (&line[..colon], value.strip_prefix(b" ").unwrap_or(value))
                });
        match field {
            b"event" => self.event.event = Some(String::from_utf8_lossy(value).into_owned()),
            b"data" => self.data_lines.push(value.to_vec()),
            b"id" if !value.contains(&0) => {
                self.event.id = Some(String::from_utf8_lossy(value).into_owned())
            }
            b"retry" => {
                if let Some(retry) = std::str::from_utf8(value)
                    .ok()
                    .and_then(|value| value.parse().ok())
                {
                    self.event.retry_millis = Some(retry);
                }
            }
            _ => {}
        }
        None
    }

    fn dispatch(&mut self) -> Option<SseEvent> {
        if self.data_lines.is_empty() {
            self.event.event = None;
            self.event.retry_millis = None;
            return None;
        }
        self.event.data = self.data_lines.join(&b'\n');
        self.data_lines.clear();
        Some(std::mem::take(&mut self.event))
    }
}
