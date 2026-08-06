// ref: internal/translator/translator/translator.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::sdk::translator::{
    Format, Registry, RequestTransform, ResponseTransform, TranslationContext, TranslationState,
};

/// Owns the default registry and context without a mutable process-global.
pub struct Translator {
    registry: Registry,
    context: TranslationContext,
}

impl Default for Translator {
    fn default() -> Self {
        Self::registered()
    }
}

impl Translator {
    pub fn new(registry: Registry, context: TranslationContext) -> Self {
        Self { registry, context }
    }

    pub fn registered() -> Self {
        let registry = Registry::new();
        crate::internal::translator::register_all(&registry);
        Self::new(registry, TranslationContext::default())
    }

    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    pub fn context(&self) -> &TranslationContext {
        &self.context
    }

    pub fn register(
        &self,
        from: &str,
        to: &str,
        request: RequestTransform,
        response: ResponseTransform,
    ) {
        register(&self.registry, from, to, request, response);
    }

    pub fn request(
        &self,
        from: &str,
        to: &str,
        model_name: &str,
        raw_json: &[u8],
        stream: bool,
    ) -> Vec<u8> {
        request(
            &self.registry,
            &self.context,
            from,
            to,
            model_name,
            raw_json,
            stream,
        )
    }

    pub fn need_convert(&self, from: &str, to: &str) -> bool {
        need_convert(&self.registry, from, to)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn response(
        &self,
        from: &str,
        to: &str,
        model_name: &str,
        original_request: &[u8],
        translated_request: &[u8],
        raw_json: &[u8],
        state: Option<&mut TranslationState>,
    ) -> Vec<Vec<u8>> {
        response(
            &self.registry,
            &self.context,
            from,
            to,
            model_name,
            original_request,
            translated_request,
            raw_json,
            state,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn response_non_stream(
        &self,
        from: &str,
        to: &str,
        model_name: &str,
        original_request: &[u8],
        translated_request: &[u8],
        raw_json: &[u8],
        state: Option<&mut TranslationState>,
    ) -> Vec<u8> {
        response_non_stream(
            &self.registry,
            &self.context,
            from,
            to,
            model_name,
            original_request,
            translated_request,
            raw_json,
            state,
        )
    }
}

pub fn register(
    registry: &Registry,
    from: &str,
    to: &str,
    request: RequestTransform,
    response: ResponseTransform,
) {
    registry.register(
        Format::from(from),
        Format::from(to),
        Some(request),
        response,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn request(
    registry: &Registry,
    context: &TranslationContext,
    from: &str,
    to: &str,
    model_name: &str,
    raw_json: &[u8],
    stream: bool,
) -> Vec<u8> {
    registry.translate_request(
        context,
        &Format::from(from),
        &Format::from(to),
        model_name,
        raw_json,
        stream,
    )
}

pub fn need_convert(registry: &Registry, from: &str, to: &str) -> bool {
    registry.has_response_transformer(&Format::from(from), &Format::from(to))
}

#[allow(clippy::too_many_arguments)]
pub fn response(
    registry: &Registry,
    context: &TranslationContext,
    from: &str,
    to: &str,
    model_name: &str,
    original_request: &[u8],
    translated_request: &[u8],
    raw_json: &[u8],
    state: Option<&mut TranslationState>,
) -> Vec<Vec<u8>> {
    let mut local_state = None;
    let state = state.unwrap_or(&mut local_state);
    registry.translate_stream(
        context,
        &Format::from(from),
        &Format::from(to),
        model_name,
        original_request,
        translated_request,
        raw_json,
        state,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn response_non_stream(
    registry: &Registry,
    context: &TranslationContext,
    from: &str,
    to: &str,
    model_name: &str,
    original_request: &[u8],
    translated_request: &[u8],
    raw_json: &[u8],
    state: Option<&mut TranslationState>,
) -> Vec<u8> {
    let mut local_state = None;
    let state = state.unwrap_or(&mut local_state);
    registry.translate_non_stream(
        context,
        &Format::from(from),
        &Format::from(to),
        model_name,
        original_request,
        translated_request,
        raw_json,
        state,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn registered_facade_routes_request_and_response_without_global_state() {
        let translator = Translator::registered();
        let request = translator.request(
            "claude",
            "interactions",
            "model-test",
            br#"{"messages":[{"role":"user","content":"hi"}]}"#,
            false,
        );
        let request: Value = serde_json::from_slice(&request).unwrap();
        assert_eq!(
            request.pointer("/input/0/content/0/text"),
            Some(&Value::String("hi".into()))
        );
        assert!(translator.need_convert("claude", "interactions"));

        let mut state = None;
        let response = translator.response_non_stream(
            "interactions",
            "claude",
            "model-test",
            &[],
            &[],
            br#"{"id":"interaction_1","steps":[{"type":"model_output","content":[{"type":"text","text":"ok"}]}]}"#,
            Some(&mut state),
        );
        let response: Value = serde_json::from_slice(&response).unwrap();
        assert_eq!(
            response.pointer("/content/0/text"),
            Some(&Value::String("ok".into()))
        );
    }
}
