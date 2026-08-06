// ref: sdk/translator/types.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Clone, Debug, Default)]
pub struct TranslationContext {
    cancelled: Arc<AtomicBool>,
}

impl TranslationContext {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

pub type TranslationState = Option<Box<dyn Any + Send>>;

pub type RequestTransform = Arc<dyn Fn(&str, &[u8], bool) -> Vec<u8> + Send + Sync + 'static>;
pub type ResponseStreamTransform = Arc<
    dyn Fn(&TranslationContext, &str, &[u8], &[u8], &[u8], &mut TranslationState) -> Vec<Vec<u8>>
        + Send
        + Sync
        + 'static,
>;
pub type ResponseNonStreamTransform = Arc<
    dyn Fn(&TranslationContext, &str, &[u8], &[u8], &[u8], &mut TranslationState) -> Vec<u8>
        + Send
        + Sync
        + 'static,
>;
pub type ResponseTokenCountTransform =
    Arc<dyn Fn(&TranslationContext, i64) -> Vec<u8> + Send + Sync + 'static>;

#[derive(Clone, Default)]
pub struct ResponseTransform {
    pub stream: Option<ResponseStreamTransform>,
    pub non_stream: Option<ResponseNonStreamTransform>,
    pub token_count: Option<ResponseTokenCountTransform>,
}

impl ResponseTransform {
    pub fn has_any(&self) -> bool {
        self.stream.is_some() || self.non_stream.is_some() || self.token_count.is_some()
    }
}
