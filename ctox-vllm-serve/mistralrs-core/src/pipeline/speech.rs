use super::text_models_inputs_processor::PagedAttentionMeta;
use super::{
    AdapterPaths, AnyMoePipelineMixin, Cache, CacheManagerMixin, EitherCache, ForwardInputsResult,
    GeneralMetadata, InputProcessorOutput, InputsProcessor, InputsProcessorType, IsqPipelineMixin,
    Loader, MessagesAction, MetadataMixin, ModelCategory, ModelKind, ModelPaths,
    PreProcessingMixin, Processor, TokenSource,
};
use crate::device_map::{self, DeviceMapper};
use crate::distributed::{WorkerTransferData, use_ring};
use crate::pipeline::{ChatTemplate, EmbeddingModulePaths, Modalities, SupportedModality};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::speech_models::{
    DiaConfig, DiaPipeline, Qwen3TtsAudioProcessor, Qwen3TtsConfig, Qwen3TtsSpeakerEncoder,
    Qwen3TtsTalker, Qwen3TtsTokenizerConfig, Qwen3TtsTokenizerDecoder,
    Qwen3TtsTokenizerEncoder, SpeechGenerationOutput, SpeechLoaderType, prepare_request,
};
use crate::speech_models::utils::normalize_loudness;
use crate::utils::progress::ProgressScopeGuard;
use crate::utils::varbuilder_utils::DeviceForLoadTensor;
use crate::utils::{tokens::get_token, varbuilder_utils::from_mmaped_safetensors};
use crate::{
    DeviceMapSetting, MessageContent, PagedAttentionConfig, Pipeline, SpeechGenerationConfig,
    TryIntoDType, api_get_file, distributed, request::SpeechGenerationRequest,
};
use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use indexmap::IndexMap;
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use regex::Regex;
use serde::Deserialize;
use std::time::Instant;
use std::any::Any;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::{
    Tokenizer,
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::BPE,
    pre_tokenizers::byte_level::ByteLevel,
    tokenizer::AddedToken,
};
use tokio::sync::Mutex;

#[derive(Clone, Debug)]
pub struct SpeechModelPaths {
    weights: Vec<PathBuf>,
    config: PathBuf,
    tokenizer_vocab: Option<PathBuf>,
    tokenizer_merges: Option<PathBuf>,
    tokenizer_config: Option<PathBuf>,
    generation_config: Option<PathBuf>,
    speech_tokenizer_config: Option<PathBuf>,
    speech_tokenizer_weights: Vec<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct HfAddedTokenSpec {
    content: String,
    #[serde(default)]
    special: bool,
}

#[derive(Debug, Deserialize)]
struct HfTokenizerConfig {
    #[serde(default)]
    added_tokens_decoder: std::collections::HashMap<String, HfAddedTokenSpec>,
}

fn load_qwen3_tts_text_tokenizer(
    vocab_path: &PathBuf,
    merges_path: &PathBuf,
    tokenizer_config_path: &PathBuf,
) -> Result<Tokenizer> {
    let vocab_path = vocab_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid UTF-8 path for Qwen3-TTS vocab.json"))?;
    let merges_path = merges_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid UTF-8 path for Qwen3-TTS merges.txt"))?;
    let model = BPE::from_file(vocab_path, merges_path)
        .build()
        .map_err(anyhow::Error::msg)?;
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
    tokenizer.with_decoder(Some(ByteLevelDecoder::new(false, false, false)));

    let tokenizer_cfg: HfTokenizerConfig =
        serde_json::from_str(&std::fs::read_to_string(tokenizer_config_path)?)?;
    let mut special_tokens = tokenizer_cfg
        .added_tokens_decoder
        .into_iter()
        .filter_map(|(id, tok)| {
            tok.special.then_some((
                id.parse::<u32>().unwrap_or(u32::MAX),
                AddedToken::from(tok.content, true),
            ))
        })
        .collect::<Vec<_>>();
    special_tokens.sort_unstable_by_key(|(id, _)| *id);
    let special_tokens = special_tokens
        .into_iter()
        .map(|(_, tok)| tok)
        .collect::<Vec<_>>();
    tokenizer.add_special_tokens(&special_tokens);

    Ok(tokenizer)
}

impl ModelPaths for SpeechModelPaths {
    fn get_config_filename(&self) -> &PathBuf {
        &self.config
    }
    fn get_tokenizer_filename(&self) -> &PathBuf {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_weight_filenames(&self) -> &[PathBuf] {
        &self.weights
    }
    fn get_template_filename(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_gen_conf_filename(&self) -> Option<&PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_preprocessor_config(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_processor_config(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_chat_template_explicit(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_adapter_paths(&self) -> &AdapterPaths {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_modules(&self) -> Option<&[EmbeddingModulePaths]> {
        unreachable!("Use `std::any::Any`.")
    }
}

pub struct SpeechProcessor;

impl Processor for SpeechProcessor {
    fn process(
        &self,
        _pipeline: &dyn Pipeline,
        _messages: Vec<IndexMap<String, MessageContent>>,
        _add_generation_prompt: bool,
        _add_special_tokens: bool,
        _enable_thinking: Option<bool>,
        _reasoning_effort: Option<crate::request::ReasoningEffort>,
        _tools: Vec<crate::Tool>,
    ) -> Result<(Vec<u32>, String)> {
        anyhow::bail!(
            "SpeechProcessor::process should not be used. It does not expect chat messages."
        )
    }
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(SpeechInputsProcessor)
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }
    fn template_action(&self) -> MessagesAction {
        // Just a default
        MessagesAction::FlattenOnlyText
    }
}

pub struct SpeechInputsProcessor;

#[derive(Clone)]
pub struct ModelInputs {
    pub(crate) requests: Vec<SpeechGenerationRequest>,
}

impl InputsProcessor for SpeechInputsProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Text
    }

    fn process_inputs(
        &self,
        _tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        _is_prompt: bool,
        _is_xlora: bool,
        _device: &Device,
        _no_kv_cache: bool,
        _last_n_context_len: Option<(usize, usize)>,
        _return_raw_logits: bool,
        _other_config: Option<Arc<dyn Any>>,
        _paged_attn_metadata: Option<PagedAttentionMeta>,
        _mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputProcessorOutput> {
        let inputs = ModelInputs {
            requests: input_seqs
                .iter()
                .map(|seq| {
                    seq.speech_request()
                        .cloned()
                        .unwrap_or(SpeechGenerationRequest {
                            input: seq.get_initial_prompt().to_string(),
                            speaker: None,
                            language: None,
                            instructions: None,
                            task_type: None,
                            ref_audio: None,
                            ref_audio_input: None,
                            ref_text: None,
                            ref_code: None,
                            icl_mode: None,
                            x_vector_only_mode: None,
                            max_new_tokens: None,
                            temperature: None,
                            top_p: None,
                            top_k: None,
                            repetition_penalty: None,
                        })
                })
                .collect(),
        };
        Ok(InputProcessorOutput {
            inputs: Box::new(inputs),
            seq_indices: (0..input_seqs.len()).collect::<Vec<_>>(),
        })
    }
}

pub struct SpeechPipeline {
    model_id: String,
    model: SpeechRuntimeModel,
    metadata: Arc<GeneralMetadata>,
    dummy_cache: EitherCache,
    cfg: SpeechGenerationConfig,
    tokenizer: Option<Arc<Tokenizer>>,
}

enum SpeechRuntimeModel {
    Dia(DiaPipeline),
    Qwen3Tts(Qwen3TtsPipeline),
}

impl SpeechRuntimeModel {
    fn generate(
        &self,
        request: &SpeechGenerationRequest,
        cfg: &SpeechGenerationConfig,
    ) -> candle_core::Result<SpeechGenerationOutput> {
        match self {
            Self::Dia(model) => model.generate(&request.input, cfg),
            Self::Qwen3Tts(model) => model.generate(request, cfg),
        }
    }

    fn device(&self) -> &Device {
        match self {
            Self::Dia(model) => model.device(),
            Self::Qwen3Tts(model) => model.device(),
        }
    }
}

fn merged_speech_cfg(
    base: &SpeechGenerationConfig,
    request: &SpeechGenerationRequest,
) -> SpeechGenerationConfig {
    match *base {
        SpeechGenerationConfig::Dia {
            max_tokens,
            cfg_scale,
            temperature,
            top_p,
            top_k,
        } => SpeechGenerationConfig::Dia {
            max_tokens: request.max_new_tokens.or(max_tokens),
            cfg_scale,
            temperature,
            top_p,
            top_k,
        },
        SpeechGenerationConfig::Qwen3Tts {
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            subtalker_do_sample,
            subtalker_temperature,
            subtalker_top_p,
            subtalker_top_k,
        } => {
            let force_greedy = matches!(request.temperature, Some(v) if v == 0.0);
            SpeechGenerationConfig::Qwen3Tts {
            max_new_tokens: request.max_new_tokens.or(max_new_tokens),
            temperature: request.temperature.unwrap_or(temperature),
            top_p: request.top_p.unwrap_or(top_p),
            top_k: request.top_k.or(top_k),
            repetition_penalty: request.repetition_penalty.unwrap_or(repetition_penalty),
            subtalker_do_sample: if force_greedy {
                false
            } else {
                subtalker_do_sample
            },
            subtalker_temperature: if force_greedy {
                0.0
            } else {
                subtalker_temperature
            },
            subtalker_top_p,
            subtalker_top_k,
        }
        }
    }
}

struct Qwen3TtsPipeline {
    cfg: Qwen3TtsConfig,
    tokenizer_cfg: Qwen3TtsTokenizerConfig,
    tokenizer: Arc<Tokenizer>,
    audio_processor: Qwen3TtsAudioProcessor,
    speaker_encoder: Option<Qwen3TtsSpeakerEncoder>,
    tokenizer_encoder: Qwen3TtsTokenizerEncoder,
    talker: Qwen3TtsTalker,
    decoder: Qwen3TtsTokenizerDecoder,
    device: Device,
}

impl Qwen3TtsPipeline {
    fn new(
        cfg: Qwen3TtsConfig,
        tokenizer_cfg: Qwen3TtsTokenizerConfig,
        tokenizer: Arc<Tokenizer>,
        model_vb: mistralrs_quant::ShardedVarBuilder,
        speech_tokenizer_vb: mistralrs_quant::ShardedVarBuilder,
        speech_tokenizer_weights: std::path::PathBuf,
    ) -> Result<Self> {
        let audio_processor = Qwen3TtsAudioProcessor::new(&cfg.speaker_encoder_config);
        let speaker_encoder = if cfg.tts_model_type.trim().eq_ignore_ascii_case("base") {
            Some(Qwen3TtsSpeakerEncoder::new(
                &cfg.speaker_encoder_config,
                model_vb.clone().pp("speaker_encoder"),
            )?)
        } else {
            None
        };
        let tokenizer_encoder = Qwen3TtsTokenizerEncoder::new(
            &tokenizer_cfg.encoder_config,
            speech_tokenizer_vb.clone(),
            &speech_tokenizer_weights,
            tokenizer_cfg.decoder_config.num_quantizers,
        )?;
        let talker = Qwen3TtsTalker::new(&cfg, model_vb)?;
        let decoder =
            Qwen3TtsTokenizerDecoder::new(&tokenizer_cfg.decoder_config, speech_tokenizer_vb.pp("decoder"))?;
        Ok(Self {
            device: talker.device().clone(),
            cfg,
            tokenizer_cfg,
            tokenizer,
            audio_processor,
            speaker_encoder,
            tokenizer_encoder,
            talker,
            decoder,
        })
    }

    fn generate(
        &self,
        request: &SpeechGenerationRequest,
        cfg: &SpeechGenerationConfig,
    ) -> candle_core::Result<SpeechGenerationOutput> {
        let started = Instant::now();
        let mut prepared = prepare_request(&self.cfg, &self.tokenizer, request)
            .map_err(candle_core::Error::msg)?;
        tracing::info!(
            "Qwen3-TTS pipeline generate start: task={:?} input_chars={} x_vector_only_mode={}",
            prepared.task_type,
            request.input.chars().count(),
            prepared.x_vector_only_mode
        );
        if prepared.requires_ref_codes && prepared.ref_codes.is_none() {
            let ref_audio = prepared
                .ref_audio_input
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("Qwen3-TTS ICL path requires ref_audio_input".into()))?;
            let waveform = self
                .audio_processor
                .prepare_waveform(ref_audio, &self.device)
                .map_err(candle_core::Error::msg)?;
            let ref_codes = self.tokenizer_encoder.encode_ref_codes(
                &waveform,
                self.tokenizer_cfg.encode_downsample_rate,
                self.tokenizer_cfg.decoder_config.num_quantizers,
            )?;
            let ref_code_preview = ref_codes
                .iter()
                .take(3)
                .map(|frame| frame.iter().take(6).copied().collect::<Vec<_>>())
                .collect::<Vec<_>>();
            tracing::info!(
                "Qwen3-TTS tokenizer encoder produced {} ref_code frames preview={:?}",
                ref_codes.len(),
                ref_code_preview
            );
            prepared.ref_codes = Some(ref_codes);
        }
        let speaker_embedding = match (prepared.ref_audio_input.as_ref(), self.speaker_encoder.as_ref()) {
            (Some(audio), Some(encoder)) => {
                let speaker_started = Instant::now();
                let mel = self
                    .audio_processor
                    .process_audio(audio, &self.device)
                    .map_err(candle_core::Error::msg)?;
                let embedding = encoder.forward(&mel)?;
                tracing::info!(
                    "Qwen3-TTS speaker embedding ready in {} ms",
                    speaker_started.elapsed().as_millis()
                );
                Some(embedding)
            }
            _ => None,
        };
        let codes_started = Instant::now();
        let codes = self
            .talker
            .generate_codes(&prepared, speaker_embedding.as_ref(), cfg)
            .map_err(candle_core::Error::msg)?;
        tracing::info!(
            "Qwen3-TTS talker returned codes shape={:?} in {} ms",
            codes.dims(),
            codes_started.elapsed().as_millis()
        );
        let decode_started = Instant::now();
        let wav = self.decoder.decode(&codes)?;
        tracing::info!(
            "Qwen3-TTS decoder returned wav shape={:?} in {} ms",
            wav.dims(),
            decode_started.elapsed().as_millis()
        );
        let wav = wav.i((0, 0))?;
        let wav = normalize_loudness(&wav, self.tokenizer_cfg.output_sample_rate as u32, true)?;
        tracing::info!(
            "Qwen3-TTS pipeline generate done in {} ms",
            started.elapsed().as_millis()
        );
        Ok(SpeechGenerationOutput {
            pcm: Arc::new(wav.to_vec1::<f32>()?),
            rate: self.tokenizer_cfg.output_sample_rate,
            channels: 1,
        })
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

pub struct SpeechLoader {
    pub model_id: String,
    pub dac_model_id: Option<String>,
    pub arch: SpeechLoaderType,
    pub cfg: Option<SpeechGenerationConfig>,
}

impl Loader for SpeechLoader {
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let paths: anyhow::Result<Box<dyn ModelPaths>> = {
            // Main weights first, DAC is the final one.
            let mut weights = Vec::new();
            let mut tokenizer_vocab = None;
            let mut tokenizer_merges = None;
            let mut tokenizer_config = None;
            let mut generation_config = None;
            let mut speech_tokenizer_config = None;
            let mut speech_tokenizer_weights = Vec::new();

            // Main model
            let config = {
                let api = ApiBuilder::new()
                    .with_progress(!silent)
                    .with_token(get_token(&token_source)?)
                    .build()?;
                let revision = revision.clone().unwrap_or("main".to_string());
                let api = api.repo(Repo::with_revision(
                    self.model_id.to_string(),
                    RepoType::Model,
                    revision.clone(),
                ));
                let model_id = std::path::Path::new(&self.model_id);

                let weight = api_get_file!(api, "model.safetensors", &model_id);
                let config = api_get_file!(api, "config.json", &model_id);
                weights.push(weight);
                if matches!(self.arch, SpeechLoaderType::Qwen3Tts) {
                    tokenizer_vocab = Some(api_get_file!(api, "vocab.json", &model_id));
                    tokenizer_merges = Some(api_get_file!(api, "merges.txt", &model_id));
                    tokenizer_config = Some(api_get_file!(api, "tokenizer_config.json", &model_id));
                    generation_config =
                        Some(api_get_file!(api, "generation_config.json", &model_id));
                    speech_tokenizer_config =
                        Some(api_get_file!(api, "speech_tokenizer/config.json", &model_id));
                    speech_tokenizer_weights
                        .push(api_get_file!(api, "speech_tokenizer/model.safetensors", &model_id));
                }
                config
            };

            // Auxiliary model weights (currently DAC for Dia only).
            if matches!(self.arch, SpeechLoaderType::Dia) {
                let api = ApiBuilder::new()
                    .with_progress(!silent)
                    .with_token(get_token(&token_source)?)
                    .build()?;
                let revision = revision.unwrap_or("main".to_string());

                let dac_model = self
                    .dac_model_id
                    .clone()
                    .unwrap_or_else(|| "EricB/dac_44khz".to_string());

                let api = api.repo(Repo::with_revision(
                    dac_model.clone(),
                    RepoType::Model,
                    revision.clone(),
                ));
                let model_id = std::path::Path::new(&dac_model);

                let weight = api_get_file!(api, "model.safetensors", &model_id);
                weights.push(weight);
            }

            Ok(Box::new(SpeechModelPaths {
                weights,
                config,
                tokenizer_vocab,
                tokenizer_merges,
                tokenizer_config,
                generation_config,
                speech_tokenizer_config,
                speech_tokenizer_weights,
            }))
        };
        self.load_model_from_path(
            &paths?,
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        )
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_path(
        &self,
        paths: &Box<dyn ModelPaths>,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        _paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let paths = &paths
            .as_ref()
            .as_any()
            .downcast_ref::<SpeechModelPaths>()
            .expect("Path downcast failed.");

        if matches!(mapper, DeviceMapSetting::Map(_)) {
            anyhow::bail!("Device mapping is not supported for speech models.")
        }

        mistralrs_quant::set_immediate_isq(in_situ_quant, vec![Regex::new(".*")?]);

        let raw_config = std::fs::read_to_string(&paths.config)?;

        #[cfg(feature = "cuda")]
        if let Device::Cuda(dev) = &device {
            unsafe { dev.disable_event_tracking() };
        }
        let use_nccl = mistralrs_quant::distributed::use_nccl();
        let available_devices = if let Ok(payload) = env::var(distributed::IS_DAEMON_FLAG) {
            let payload: WorkerTransferData = serde_json::from_str(&payload)?;
            let WorkerTransferData::Init { id: _, worker_rank } = payload;
            vec![candle_core::Device::new_cuda(worker_rank + 1)?]
        } else if use_nccl || use_ring() {
            vec![candle_core::Device::new_cuda(0)?]
        } else {
            device_map::get_all_similar_devices(device)?
        };

        let mapper =
            DeviceMapSetting::dummy().into_mapper(usize::MAX, device, None, &available_devices)?;
        let dtype = mapper.get_min_dtype(dtype)?;

        let tokenizer = match (
            &self.arch,
            &paths.tokenizer_vocab,
            &paths.tokenizer_merges,
            &paths.tokenizer_config,
        ) {
            (
                SpeechLoaderType::Qwen3Tts,
                Some(vocab),
                Some(merges),
                Some(tokenizer_config),
            ) => Some(Arc::new(load_qwen3_tts_text_tokenizer(
                vocab,
                merges,
                tokenizer_config,
            )?)),
            (SpeechLoaderType::Qwen3Tts, _, _, _) => {
                anyhow::bail!("Qwen3-TTS requires `vocab.json`, `merges.txt`, and `tokenizer_config.json`.")
            }
            _ => None,
        };

        let model = match self.arch {
            SpeechLoaderType::Dia => {
                let cfg: DiaConfig = serde_json::from_str(&raw_config)?;
                let model_weights = paths.weights[..paths.weights.len() - 1].to_vec();
                let vb = from_mmaped_safetensors(
                    model_weights,
                    Vec::new(),
                    Some(dtype),
                    device,
                    vec![None],
                    silent,
                    None,
                    |_| true,
                    Arc::new(|_| DeviceForLoadTensor::Base),
                )?;

                let dac_vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        &[paths.weights.last().expect("Dia DAC weight missing")],
                        dtype,
                        device,
                    )?
                };

                SpeechRuntimeModel::Dia(DiaPipeline::new(&cfg, vb, dac_vb)?)
            }
            SpeechLoaderType::Qwen3Tts => {
                let cfg: Qwen3TtsConfig = serde_json::from_str(&raw_config)?;
                let Some(speech_tokenizer_config_path) = &paths.speech_tokenizer_config else {
                    anyhow::bail!(
                        "Qwen3-TTS requires `speech_tokenizer/config.json`, but it was not resolved during model load."
                    );
                };
                let speech_tokenizer_config: Qwen3TtsTokenizerConfig =
                    serde_json::from_str(&std::fs::read_to_string(speech_tokenizer_config_path)?)?;
                let vb = from_mmaped_safetensors(
                    paths.weights.clone(),
                    Vec::new(),
                    Some(dtype),
                    device,
                    vec![None],
                    silent,
                    None,
                    |_| true,
                    Arc::new(|_| DeviceForLoadTensor::Base),
                )?;
                let speech_tokenizer_vb = from_mmaped_safetensors(
                    paths.speech_tokenizer_weights.clone(),
                    Vec::new(),
                    Some(dtype),
                    device,
                    vec![None],
                    silent,
                    None,
                    |_| true,
                    Arc::new(|_| DeviceForLoadTensor::Base),
                )?;
                SpeechRuntimeModel::Qwen3Tts(Qwen3TtsPipeline::new(
                    cfg,
                    speech_tokenizer_config,
                    tokenizer.clone().expect("Qwen3-TTS tokenizer missing"),
                    vb,
                    speech_tokenizer_vb,
                    paths
                        .speech_tokenizer_weights
                        .first()
                        .cloned()
                        .expect("Qwen3-TTS speech tokenizer weights missing"),
                )?)
            }
        };

        let loaded_generation_cfg = if matches!(self.arch, SpeechLoaderType::Qwen3Tts) {
            paths
                .generation_config
                .as_ref()
                .and_then(|path| std::fs::read_to_string(path).ok())
                .as_deref()
                .and_then(SpeechGenerationConfig::from_qwen3_tts_generation_config)
        } else {
            None
        };

        Ok(Arc::new(Mutex::new(SpeechPipeline {
            model_id: self.model_id.clone(),
            model,
            metadata: Arc::new(GeneralMetadata {
                max_seq_len: 1024,
                llg_factory: None,
                is_xlora: false,
                no_prefix_cache: false,
                num_hidden_layers: 1, // FIXME(EricLBuehler): we know this is only for caching, so its OK.
                eos_tok: vec![],
                kind: ModelKind::Normal,
                no_kv_cache: true, // NOTE(EricLBuehler): no cache for these.
                activation_dtype: dtype,
                sliding_window: None,
                cache_config: None,
                cache_engine: None,
                model_metadata: None,
                modalities: Modalities {
                    input: vec![SupportedModality::Text],
                    output: vec![SupportedModality::Audio],
                },
            }),
            dummy_cache: EitherCache::Full(Cache::new(0, false)),
            cfg: self
                .cfg
                .or(loaded_generation_cfg)
                .unwrap_or_else(|| SpeechGenerationConfig::default(self.arch)),
            tokenizer,
        })))
    }

    fn get_id(&self) -> String {
        self.model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        ModelKind::Normal
    }
}

impl PreProcessingMixin for SpeechPipeline {
    fn get_processor(&self) -> Arc<dyn Processor> {
        Arc::new(SpeechProcessor)
    }
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        None
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for SpeechPipeline {
    fn re_isq_model(&mut self, _dtype: IsqType) -> Result<()> {
        anyhow::bail!("Speech models do not support ISQ for now.")
    }
}

impl CacheManagerMixin for SpeechPipeline {
    fn clone_in_cache(&self, _seqs: &mut [&mut Sequence]) {}
    fn clone_out_cache(&self, _seqs: &mut [&mut Sequence]) {}
    fn set_none_cache(
        &self,
        _seqs: &mut [&mut Sequence],
        _reset_non_granular: bool,
        _modify_draft_cache: bool,
        _load_preallocated_cache: bool,
    ) {
    }
    fn cache(&self) -> &EitherCache {
        &self.dummy_cache
    }
}

impl MetadataMixin for SpeechPipeline {
    fn device(&self) -> Device {
        self.model.device().clone()
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn reset_non_granular_state(&self) {}
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        self.tokenizer.clone()
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        None
    }
}

#[async_trait::async_trait]
impl Pipeline for SpeechPipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> candle_core::Result<ForwardInputsResult> {
        assert!(!return_raw_logits);

        let ModelInputs { requests } = *inputs.downcast().expect("Downcast failed.");
        let mut pcms = Vec::new();
        let mut rates = Vec::new();
        let mut channels_all = Vec::new();
        for request in requests {
            let effective_cfg = merged_speech_cfg(&self.cfg, &request);
            let SpeechGenerationOutput {
                pcm,
                rate,
                channels,
            } = self.model.generate(&request, &effective_cfg)?;
            pcms.push(pcm);
            rates.push(rate);
            channels_all.push(channels);
        }

        Ok(ForwardInputsResult::Speech {
            pcms,
            rates,
            channels: channels_all,
        })
    }

    async fn sample_causal_gen(
        &self,
        _seqs: &mut [&mut Sequence],
        _logits: Vec<Tensor>,
        _prefix_cacher: &mut PrefixCacheManagerV2,
        _disable_eos_stop: bool,
        _srng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error> {
        candle_core::bail!("`sample_causal_gen` is incompatible with `SpeechPipeline`");
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::Speech
    }
}

impl AnyMoePipelineMixin for SpeechPipeline {}
