// ref: sdk/api/handlers/stream_forwarder.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{HandlerCancellation, HandlerResponse};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ForwardedStreamEvent {
    Chunk(Vec<u8>),
    TerminalError(HandlerResponse),
    Complete,
}

#[derive(Clone, Debug, Default)]
pub struct StreamForwarder {
    committed: bool,
    finished: bool,
}

impl StreamForwarder {
    #[must_use]
    pub fn committed(&self) -> bool {
        self.committed
    }

    #[must_use]
    pub fn push_chunk(&mut self, chunk: Vec<u8>) -> Option<ForwardedStreamEvent> {
        if self.finished || chunk.is_empty() {
            return None;
        }
        self.committed = true;
        Some(ForwardedStreamEvent::Chunk(chunk))
    }

    #[must_use]
    pub fn fail(&mut self, response: HandlerResponse) -> ForwardedStreamEvent {
        self.finished = true;
        ForwardedStreamEvent::TerminalError(response)
    }

    #[must_use]
    pub fn complete(&mut self) -> Option<ForwardedStreamEvent> {
        if self.finished {
            return None;
        }
        self.finished = true;
        Some(ForwardedStreamEvent::Complete)
    }

    pub fn cancel(&mut self, cancellation: &HandlerCancellation) {
        cancellation.cancel();
        self.finished = true;
    }
}
