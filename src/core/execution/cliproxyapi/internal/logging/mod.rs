// Origin: CTOX module graph for the upstream logging package.
// License: AGPL-3.0-only

pub mod cpa_trace;
pub mod gin_logger;
pub mod global_logger;
pub mod home_app_log_forwarder;
pub mod log_dir_cleaner;
pub mod request_logger;
pub mod request_logger_body_source;
pub mod request_logger_format;
pub mod request_logger_home;
pub mod request_logger_streaming;
pub mod request_logger_writer;
pub mod requestid;
pub mod requestmeta;

pub use cpa_trace::{
    format_cpa_trace_id, get_handler_cpa_trace_id, handler_cpa_trace_id_callback,
    set_handler_cpa_trace_id, CpaTraceIdCallback, CpaTraceResponseWriter, CPA_TRACE_ID_HEADER,
};
pub use requestid::{
    generate_request_id, get_handler_request_id, get_request_id, set_handler_request_id,
    with_request_id,
};
pub use requestmeta::{
    get_client_request_metadata, get_endpoint, get_response_headers, get_response_status,
    set_response_headers, set_response_status, with_client_request_metadata, with_endpoint,
    with_response_headers_holder, with_response_status_holder, ClientRequestMetadata,
    RequestContext, ResponseHeaders,
};

#[cfg(test)]
mod cpa_trace_test;

#[cfg(test)]
mod gin_logger_test;

#[cfg(test)]
mod global_logger_test;

#[cfg(test)]
mod home_app_log_forwarder_test;

#[cfg(test)]
mod log_dir_cleaner_test;

#[cfg(test)]
mod request_logger_home_test;

#[cfg(test)]
mod request_logger_test;

#[cfg(test)]
mod request_logger_streaming_test;
