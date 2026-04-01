#[allow(dead_code)]
mod acoustic_transformer;
#[allow(dead_code)]
mod audio_tokenizer;
mod config;
#[allow(dead_code)]
mod language_model;
mod prompt;
mod voice_embedding;
#[allow(dead_code)]
mod wire;

pub use acoustic_transformer::{AudioSpecialToken, VoxtralTtsAcousticTransformer};
pub use audio_tokenizer::VoxtralTtsAudioTokenizer;
pub use config::{
    VoxtralTtsAudioEncodingArgs, VoxtralTtsAudioModelArgs, VoxtralTtsAudioTokenizerArgs,
    VoxtralTtsConfig, VoxtralTtsMultimodalConfig,
};
pub use language_model::VoxtralTtsLanguageModel;
pub use prompt::{
    build_input_ids as build_voxtral_tts_input_ids, prepare_request, VoxtralTtsPreparedRequest,
};
pub use voice_embedding::{load_voice_embedding_pt, TorchStorageKind, TorchTensorArchive};
#[allow(unused_imports)]
pub use wire::{
    apply_ctx_frames_cutting, parse_batched_audio_input, prepare_decode_chunks, AudioCodeChunkPlan,
    AudioCodeSequence,
};
