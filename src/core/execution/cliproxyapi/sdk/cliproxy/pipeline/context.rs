// ref: sdk/cliproxy/pipeline/context.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::sync::Arc;

use crate::sdk::cliproxy::auth::Auth;
use crate::sdk::cliproxy::executor::{ExecutionError, Options, Request, Response, StreamChunk};
use crate::sdk::cliproxy::rtprovider::DefaultRoundTripperProvider;
use crate::sdk::proxyutil::HttpTransport;
use crate::sdk::translator::{Pipeline, TranslationContext};

/// Execution state shared by middleware, translators and executors.
///
/// Go stores a mutable `http.Client` here. CTOX keeps socket/TLS/proxy
/// authority outside this value and carries only the selected, immutable
/// transport route. The component that performs I/O remains injected by the
/// executor host.
pub struct Context {
    pub request: Request,
    pub options: Options,
    pub auth: Option<Arc<Auth>>,
    pub translator: Arc<Pipeline>,
    pub http_transport: Option<Arc<HttpTransport>>,
}

impl fmt::Debug for Context {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PipelineContext")
            .field("request", &self.request)
            .field("options", &self.options)
            .field("auth_selected", &self.auth.is_some())
            .field("translator", &"Pipeline")
            .field("http_transport", &self.http_transport)
            .finish()
    }
}

/// Middleware callbacks around one execution.
pub trait Hook: Send + Sync {
    fn before_execute(&self, context: &TranslationContext, execution: &mut Context);

    fn after_execute(
        &self,
        context: &TranslationContext,
        execution: &mut Context,
        response: &Response,
        error: Option<&ExecutionError>,
    );

    fn on_stream_chunk(
        &self,
        context: &TranslationContext,
        execution: &mut Context,
        chunk: &StreamChunk,
    );
}

pub type BeforeHook = Arc<dyn Fn(&TranslationContext, &mut Context) + Send + Sync + 'static>;
pub type AfterHook = Arc<
    dyn Fn(&TranslationContext, &mut Context, &Response, Option<&ExecutionError>)
        + Send
        + Sync
        + 'static,
>;
pub type StreamHook =
    Arc<dyn Fn(&TranslationContext, &mut Context, &StreamChunk) + Send + Sync + 'static>;

/// Optional callback aggregation matching upstream's `HookFunc` nil behavior.
#[derive(Clone, Default)]
pub struct HookFunc {
    pub before: Option<BeforeHook>,
    pub after: Option<AfterHook>,
    pub stream: Option<StreamHook>,
}

impl fmt::Debug for HookFunc {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HookFunc")
            .field("before", &self.before.is_some())
            .field("after", &self.after.is_some())
            .field("stream", &self.stream.is_some())
            .finish()
    }
}

impl Hook for HookFunc {
    fn before_execute(&self, context: &TranslationContext, execution: &mut Context) {
        if let Some(callback) = &self.before {
            callback(context, execution);
        }
    }

    fn after_execute(
        &self,
        context: &TranslationContext,
        execution: &mut Context,
        response: &Response,
        error: Option<&ExecutionError>,
    ) {
        if let Some(callback) = &self.after {
            callback(context, execution, response, error);
        }
    }

    fn on_stream_chunk(
        &self,
        context: &TranslationContext,
        execution: &mut Context,
        chunk: &StreamChunk,
    ) {
        if let Some(callback) = &self.stream {
            callback(context, execution, chunk);
        }
    }
}

/// Supplies a proxy/TLS route for a selected credential without transferring
/// process-global network ownership into the pipeline context.
pub trait RoundTripperProvider: Send + Sync {
    fn round_tripper_for(&self, auth: Option<&Auth>) -> Option<Arc<HttpTransport>>;
}

impl RoundTripperProvider for DefaultRoundTripperProvider {
    fn round_tripper_for(&self, auth: Option<&Auth>) -> Option<Arc<HttpTransport>> {
        DefaultRoundTripperProvider::round_tripper_for(self, auth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdk::translator::Registry;
    use std::sync::Mutex;

    fn execution_context() -> Context {
        Context {
            request: Request::default(),
            options: Options::default(),
            auth: None,
            translator: Arc::new(Pipeline::new(Arc::new(Registry::new()))),
            http_transport: None,
        }
    }

    #[test]
    fn hook_func_preserves_optional_callback_order() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let hook = HookFunc {
            before: Some({
                let calls = Arc::clone(&calls);
                Arc::new(move |_, execution| {
                    execution.request.model = "prepared".to_owned();
                    calls.lock().unwrap().push("before");
                })
            }),
            after: Some({
                let calls = Arc::clone(&calls);
                Arc::new(move |_, execution, _, error| {
                    assert_eq!(execution.request.model, "prepared");
                    assert!(error.is_none());
                    calls.lock().unwrap().push("after");
                })
            }),
            stream: Some({
                let calls = Arc::clone(&calls);
                Arc::new(move |_, _, chunk| {
                    assert_eq!(chunk.payload, b"delta");
                    calls.lock().unwrap().push("stream");
                })
            }),
        };

        let translation = TranslationContext::default();
        let mut execution = execution_context();
        hook.before_execute(&translation, &mut execution);
        hook.on_stream_chunk(
            &translation,
            &mut execution,
            &StreamChunk {
                payload: b"delta".to_vec(),
                error: None,
            },
        );
        hook.after_execute(&translation, &mut execution, &Response::default(), None);

        assert_eq!(
            calls.lock().unwrap().as_slice(),
            ["before", "stream", "after"]
        );
    }

    #[test]
    fn context_debug_does_not_render_auth_or_proxy_secrets() {
        let mut auth = Auth::default();
        auth.id = "private-account".to_owned();
        auth.proxy_url = "http://proxy-user:proxy-pass@example.invalid:8080".to_owned();
        auth.attributes
            .insert("api_key".to_owned(), "private-api-key".to_owned());

        let provider = new_provider();
        let transport = RoundTripperProvider::round_tripper_for(&provider, Some(&auth));
        let mut execution = execution_context();
        execution.auth = Some(Arc::new(auth));
        execution.http_transport = transport;

        let debug = format!("{execution:?}");
        assert!(!debug.contains("private-account"));
        assert!(!debug.contains("proxy-pass"));
        assert!(!debug.contains("private-api-key"));
        assert!(debug.contains("auth_selected: true"));
    }

    fn new_provider() -> DefaultRoundTripperProvider {
        crate::sdk::cliproxy::rtprovider::new_default_round_tripper_provider()
    }
}
