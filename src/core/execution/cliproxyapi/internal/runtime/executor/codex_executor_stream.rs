// ref: internal/runtime/executor/codex_executor_stream.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::time::SystemTime;

use crate::internal::translator::common::SseDecoder;

use super::codex_executor_terminal::{
    CodexIncompleteStreamError, CodexTerminalAccumulator, CodexTerminalError, CodexTerminalEvent,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodexStreamTerminal {
    Pending,
    Completed(Vec<u8>),
    Failed(CodexTerminalError),
}

/// Request-owned SSE processor. A response becomes committed only at a
/// terminal event; malformed/non-terminal chunks cannot produce success.
#[derive(Debug, Default)]
pub struct CodexSseTerminalStream {
    decoder: SseDecoder,
    terminal: CodexTerminalAccumulator,
}

impl CodexSseTerminalStream {
    pub fn push(&mut self, chunk: &[u8], now: SystemTime) -> CodexStreamTerminal {
        for event in self.decoder.push(chunk) {
            if event.data == b"[DONE]" {
                continue;
            }
            match self.terminal.ingest(&event.data, now) {
                CodexTerminalEvent::Continue => {}
                CodexTerminalEvent::Completed(payload) => {
                    return CodexStreamTerminal::Completed(payload)
                }
                CodexTerminalEvent::Failed(error) => return CodexStreamTerminal::Failed(error),
            }
        }
        CodexStreamTerminal::Pending
    }

    pub fn finish(
        mut self,
        now: SystemTime,
    ) -> Result<CodexStreamTerminal, CodexIncompleteStreamError> {
        for event in self.decoder.finish() {
            match self.terminal.ingest(&event.data, now) {
                CodexTerminalEvent::Continue => {}
                CodexTerminalEvent::Completed(payload) => {
                    return Ok(CodexStreamTerminal::Completed(payload))
                }
                CodexTerminalEvent::Failed(error) => return Ok(CodexStreamTerminal::Failed(error)),
            }
        }
        self.terminal.finish()?;
        Ok(CodexStreamTerminal::Pending)
    }

    pub fn committed(&self) -> bool {
        self.terminal.committed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fragmented_sse_requires_terminal_completion() {
        let mut stream = CodexSseTerminalStream::default();
        assert_eq!(
            stream.push(b"data: {\"type\":\"response.com", SystemTime::UNIX_EPOCH),
            CodexStreamTerminal::Pending
        );
        let result = stream.push(
            b"pleted\",\"response\":{\"output\":[]}}\n\n",
            SystemTime::UNIX_EPOCH,
        );
        assert!(matches!(result, CodexStreamTerminal::Completed(_)));
        assert!(stream.committed());
    }
}
