// ref: internal/watcher/synthesizer/interface.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::context::{SynthesisContext, SynthesizedAuth};
use std::io;

pub trait AuthSynthesizer {
    fn synthesize(&self, context: &SynthesisContext<'_>) -> io::Result<Vec<SynthesizedAuth>>;
}
