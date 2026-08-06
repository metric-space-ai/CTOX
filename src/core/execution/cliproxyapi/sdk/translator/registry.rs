// ref: sdk/translator/registry.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{
    Format, PluginHooks, RequestTransform, ResponseTransform, TranslationContext, TranslationState,
};
use crate::internal::thinking::{apply_summary_config_for_model, extract_summary_config};
use crate::internal::translator::common::set_top_level_string;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Default)]
struct RegistryState {
    requests: HashMap<(Format, Format), RequestTransform>,
    responses: HashMap<(Format, Format), ResponseTransform>,
    hooks: Option<Arc<dyn PluginHooks>>,
}

#[derive(Default)]
pub struct Registry {
    state: RwLock<RegistryState>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &self,
        from: Format,
        to: Format,
        request: Option<RequestTransform>,
        response: ResponseTransform,
    ) {
        let mut state = self.state.write().expect("translator registry poisoned");
        if let Some(request) = request {
            state.requests.insert((from.clone(), to.clone()), request);
        }
        state.responses.insert((from, to), response);
    }

    /// Registers one client→provider protocol pair.
    ///
    /// As in upstream, request and response transforms share the same stored
    /// `(client, provider)` key. Response translation receives
    /// `(provider, client)` and performs the reverse lookup at dispatch time.
    pub fn register_pair(
        &self,
        client: Format,
        provider: Format,
        request: RequestTransform,
        response: ResponseTransform,
    ) {
        let mut state = self.state.write().expect("translator registry poisoned");
        state
            .requests
            .insert((client.clone(), provider.clone()), request);
        state.responses.insert((client, provider), response);
    }

    pub fn set_plugin_hooks(&self, hooks: Option<Arc<dyn PluginHooks>>) {
        self.state
            .write()
            .expect("translator registry poisoned")
            .hooks = hooks;
    }

    pub fn has_request_transformer(&self, from: &Format, to: &Format) -> bool {
        self.state
            .read()
            .expect("translator registry poisoned")
            .requests
            .contains_key(&(from.clone(), to.clone()))
    }

    pub fn has_response_transformer(&self, from: &Format, to: &Format) -> bool {
        self.response_transform(from, to)
            .is_some_and(|transform| transform.has_any())
    }

    pub fn has_stream_response_transformer(&self, from: &Format, to: &Format) -> bool {
        self.response_transform(from, to)
            .is_some_and(|transform| transform.stream.is_some())
    }

    pub fn has_non_stream_response_transformer(&self, from: &Format, to: &Format) -> bool {
        self.response_transform(from, to)
            .is_some_and(|transform| transform.non_stream.is_some())
    }

    fn response_transform(&self, from: &Format, to: &Format) -> Option<ResponseTransform> {
        self.state
            .read()
            .expect("translator registry poisoned")
            .responses
            .get(&(from.clone(), to.clone()))
            .cloned()
    }

    pub fn translate_request(
        &self,
        context: &TranslationContext,
        from: &Format,
        to: &Format,
        model: &str,
        raw_json: &[u8],
        stream: bool,
    ) -> Vec<u8> {
        let (transform, hooks) = {
            let state = self.state.read().expect("translator registry poisoned");
            (
                state.requests.get(&(from.clone(), to.clone())).cloned(),
                state.hooks.clone(),
            )
        };

        if let Some(transform) = transform {
            let summary = extract_summary_config(raw_json, from.as_str());
            let translated = transform(model, raw_json, stream);
            let body = apply_summary_config_for_model(&translated, to.as_str(), model, &summary);
            return hooks.map_or(body.clone(), |hooks| {
                hooks.normalize_request(context, from, to, model, body, stream)
            });
        }

        let mut body = normalize_model(raw_json, model);
        let Some(hooks) = hooks else { return body };
        body = hooks.normalize_request(context, from, to, model, body, stream);
        let summary = extract_summary_config(&body, from.as_str());
        hooks
            .translate_request(context, from, to, model, &body, stream)
            .map_or(body, |translated| {
                apply_summary_config_for_model(&translated, to.as_str(), model, &summary)
            })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn translate_stream(
        &self,
        context: &TranslationContext,
        from: &Format,
        to: &Format,
        model: &str,
        original_request: &[u8],
        translated_request: &[u8],
        raw_json: &[u8],
        state: &mut TranslationState,
    ) -> Vec<Vec<u8>> {
        if context.is_cancelled() {
            return Vec::new();
        }
        // Response registrations are provider -> client. This preserves the
        // direction used by upstream TranslateStream.
        let (transform, hooks) = {
            let guard = self.state.read().expect("translator registry poisoned");
            (
                guard.responses.get(&(to.clone(), from.clone())).cloned(),
                guard.hooks.clone(),
            )
        };
        let body = hooks.as_ref().map_or_else(
            || raw_json.to_vec(),
            |hooks| {
                hooks.normalize_response_before(
                    context,
                    from,
                    to,
                    model,
                    original_request,
                    translated_request,
                    raw_json.to_vec(),
                    true,
                )
            },
        );

        let used_native = transform.as_ref().is_some_and(|item| item.stream.is_some());
        let mut outputs = if let Some(native) = transform.and_then(|item| item.stream) {
            native(
                context,
                model,
                original_request,
                translated_request,
                &body,
                state,
            )
        } else if let Some(translated) = hooks.as_ref().and_then(|hooks| {
            hooks.translate_response(
                context,
                from,
                to,
                model,
                original_request,
                translated_request,
                &body,
                true,
            )
        }) {
            vec![translated]
        } else if used_native {
            Vec::new()
        } else {
            vec![body]
        };

        if let Some(hooks) = hooks {
            for output in &mut outputs {
                *output = hooks.normalize_response_after(
                    context,
                    from,
                    to,
                    model,
                    original_request,
                    translated_request,
                    std::mem::take(output),
                    true,
                );
            }
        }
        outputs
    }

    #[allow(clippy::too_many_arguments)]
    pub fn translate_non_stream(
        &self,
        context: &TranslationContext,
        from: &Format,
        to: &Format,
        model: &str,
        original_request: &[u8],
        translated_request: &[u8],
        raw_json: &[u8],
        state: &mut TranslationState,
    ) -> Vec<u8> {
        let (transform, hooks) = {
            let guard = self.state.read().expect("translator registry poisoned");
            (
                guard.responses.get(&(to.clone(), from.clone())).cloned(),
                guard.hooks.clone(),
            )
        };
        let mut body = hooks.as_ref().map_or_else(
            || raw_json.to_vec(),
            |hooks| {
                hooks.normalize_response_before(
                    context,
                    from,
                    to,
                    model,
                    original_request,
                    translated_request,
                    raw_json.to_vec(),
                    false,
                )
            },
        );
        if let Some(native) = transform.and_then(|item| item.non_stream) {
            body = native(
                context,
                model,
                original_request,
                translated_request,
                &body,
                state,
            );
        } else if let Some(translated) = hooks.as_ref().and_then(|hooks| {
            hooks.translate_response(
                context,
                from,
                to,
                model,
                original_request,
                translated_request,
                &body,
                false,
            )
        }) {
            body = translated;
        }
        hooks.map_or(body.clone(), |hooks| {
            hooks.normalize_response_after(
                context,
                from,
                to,
                model,
                original_request,
                translated_request,
                body,
                false,
            )
        })
    }

    pub fn translate_token_count(
        &self,
        context: &TranslationContext,
        from: &Format,
        to: &Format,
        count: i64,
        raw_json: &[u8],
    ) -> Vec<u8> {
        self.response_transform(to, from)
            .and_then(|transform| transform.token_count)
            .map_or_else(|| raw_json.to_vec(), |transform| transform(context, count))
    }
}

fn normalize_model(raw_json: &[u8], model: &str) -> Vec<u8> {
    if model.is_empty() {
        return raw_json.to_vec();
    }
    set_top_level_string(raw_json, "model", model)
}
