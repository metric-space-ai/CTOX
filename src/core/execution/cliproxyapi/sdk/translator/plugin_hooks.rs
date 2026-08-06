// ref: sdk/translator/plugin_hooks.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{Format, TranslationContext};

/// Extension boundary matching upstream's plugin hooks without committing the
/// Rust port to Go's in-process C plugin ABI.
#[allow(clippy::too_many_arguments)]
pub trait PluginHooks: Send + Sync {
    fn normalize_request(
        &self,
        _context: &TranslationContext,
        _from: &Format,
        _to: &Format,
        _model: &str,
        body: Vec<u8>,
        _stream: bool,
    ) -> Vec<u8> {
        body
    }

    fn translate_request(
        &self,
        _context: &TranslationContext,
        _from: &Format,
        _to: &Format,
        _model: &str,
        _body: &[u8],
        _stream: bool,
    ) -> Option<Vec<u8>> {
        None
    }

    fn normalize_response_before(
        &self,
        _context: &TranslationContext,
        _from: &Format,
        _to: &Format,
        _model: &str,
        _original_request: &[u8],
        _translated_request: &[u8],
        body: Vec<u8>,
        _stream: bool,
    ) -> Vec<u8> {
        body
    }

    fn translate_response(
        &self,
        _context: &TranslationContext,
        _from: &Format,
        _to: &Format,
        _model: &str,
        _original_request: &[u8],
        _translated_request: &[u8],
        _body: &[u8],
        _stream: bool,
    ) -> Option<Vec<u8>> {
        None
    }

    fn normalize_response_after(
        &self,
        _context: &TranslationContext,
        _from: &Format,
        _to: &Format,
        _model: &str,
        _original_request: &[u8],
        _translated_request: &[u8],
        body: Vec<u8>,
        _stream: bool,
    ) -> Vec<u8> {
        body
    }
}
