// Origin: CTOX
// License: AGPL-3.0-only

mod http;
mod manager;
mod message;
mod session;

pub use http::{
    decode_chunk, decode_error, decode_response, encode_request, HeaderMap, HttpRequest,
    HttpResponse, PrincipalAuthority, RelayAuthority, RelayHandshake, StreamEvent,
};
pub use manager::{Manager, ManagerOptions, RelayEventSink, RelayLimits, SystemRelayClock};
pub use message::{
    Message, MESSAGE_TYPE_ERROR, MESSAGE_TYPE_HTTP_REQUEST, MESSAGE_TYPE_HTTP_RESPONSE,
    MESSAGE_TYPE_PING, MESSAGE_TYPE_PONG, MESSAGE_TYPE_STREAM_CHUNK, MESSAGE_TYPE_STREAM_END,
    MESSAGE_TYPE_STREAM_START,
};
pub use session::{RelayCancellation, RelayClock, RelayError, RelayTransport, WebSocketTransport};

#[cfg(test)]
mod tests;
