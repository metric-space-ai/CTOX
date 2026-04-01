#![allow(dead_code)]

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::Embedding;
use mistralrs_quant::{
    linear as quant_linear, linear_b as quant_linear_b, linear_no_bias as quant_linear_no_bias,
    QuantMethod, QuantizedConfig, ShardedVarBuilder,
};
use rand::{
    distr::{weighted::WeightedIndex, Distribution},
    rngs::StdRng,
    SeedableRng,
};
use std::sync::Arc;
use std::time::Instant;

use crate::{
    layers::{self, repeat_kv, Activation, F32RmsNorm, RotaryEmbedding},
    ops::apply_triangular,
    speech_models::{Qwen3TtsPreparedRequest, Qwen3TtsTaskType, SpeechGenerationConfig},
};

use super::{Qwen3TtsConfig, Qwen3TtsTalkerConfig};

fn hidden_act(name: &str) -> Result<Activation> {
    match name.trim().to_ascii_lowercase().as_str() {
        "gelu" => Ok(Activation::Gelu),
        "gelu_new" | "gelu_pytorch_tanh" => Ok(Activation::NewGelu),
        "relu" => Ok(Activation::Relu),
        "silu" | "swish" => Ok(Activation::Silu),
        other => anyhow::bail!("Unsupported Qwen3-TTS activation `{other}`."),
    }
}

fn ids_tensor(ids: &[u32], device: &Device) -> candle_core::Result<Tensor> {
    Tensor::from_vec(ids.to_vec(), (1, ids.len()), device)
}

fn ensure_ids_in_range(ids: &[u32], vocab_size: usize, label: &str) -> Result<()> {
    if ids.is_empty() {
        anyhow::bail!("Qwen3-TTS {label} embedding lookup received an empty id slice.");
    }
    if let Some((position, token_id)) = ids
        .iter()
        .copied()
        .enumerate()
        .find(|(_, token_id)| *token_id as usize >= vocab_size)
    {
        anyhow::bail!(
            "Qwen3-TTS {label} embedding lookup received out-of-range token id {token_id} at position {position} (vocab_size={vocab_size}, len={}, max_id={}).",
            ids.len(),
            ids.iter().copied().max().unwrap_or(0)
        );
    }
    Ok(())
}

fn no_quant_config() -> &'static Option<QuantizedConfig> {
    static NO_QUANT_CONFIG: Option<QuantizedConfig> = None;
    &NO_QUANT_CONFIG
}

fn causal_attention_mask(seq_len: usize, device: &Device) -> candle_core::Result<Tensor> {
    let mask = apply_triangular(
        &Tensor::ones((seq_len, seq_len), DType::F32, device)?,
        0,
        false,
    )?;
    let neg_inf = Tensor::full(f32::NEG_INFINITY, (seq_len, seq_len), device)?;
    let zeros = Tensor::zeros((seq_len, seq_len), DType::F32, device)?;
    let mask = mask.to_dtype(DType::U8)?;
    mask.where_cond(&zeros, &neg_inf)?
        .unsqueeze(0)?
        .unsqueeze(0)
}

fn sample_next_token(
    logits: &Tensor,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    repetition_penalty: f32,
    context: &[u32],
    suppressed_tokens: &[u32],
    rng: &mut StdRng,
) -> candle_core::Result<u32> {
    let apply_repetition_penalty = |values: &mut [f32]| {
        if repetition_penalty == 1.0 {
            return;
        }
        let mut counts = vec![0u16; values.len()];
        for &token_id in context {
            if let Some(count) = counts.get_mut(token_id as usize) {
                *count = count.saturating_add(1);
            }
        }
        for (token_id, logit) in values.iter_mut().enumerate() {
            if counts[token_id] == 0 {
                continue;
            }
            if *logit > 0.0 {
                *logit /= repetition_penalty;
            } else {
                *logit *= repetition_penalty;
            }
        }
    };

    let suppress = |probs: &mut [f32]| {
        for &token_id in suppressed_tokens {
            if let Some(slot) = probs.get_mut(token_id as usize) {
                *slot = 0.0;
            }
        }
    };

    let fallback_argmax = |values: &[f32]| -> candle_core::Result<u32> {
        let mut values = values.to_vec();
        apply_repetition_penalty(&mut values);
        suppress(&mut values);
        values
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| candle_core::Error::msg("Qwen3-TTS sampler saw empty logits."))
    };

    if temperature == 0.0 {
        return fallback_argmax(&logits.to_dtype(DType::F32)?.to_vec1::<f32>()?);
    }

    let mut logits = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    apply_repetition_penalty(&mut logits);
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = logits
        .into_iter()
        .map(|logit| ((logit - max_logit) / temperature).exp())
        .collect::<Vec<_>>();
    suppress(&mut probs);
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_unstable_by(|&i, &j| probs[j].partial_cmp(&probs[i]).unwrap());

    if let Some(top_k) = top_k {
        for (rank, idx) in indices.iter().enumerate() {
            if rank >= top_k {
                probs[*idx] = 0.0;
            }
        }
    }

    if top_p < 1.0 {
        let total_prob = probs.iter().copied().sum::<f32>();
        if total_prob.is_finite() && total_prob > 0.0 {
            let mut cumulative = 0.0;
            for idx in &indices {
                let normalized = probs[*idx] / total_prob;
                if cumulative >= top_p {
                    probs[*idx] = 0.0;
                } else {
                    cumulative += normalized;
                }
            }
        }
    }

    if probs.iter().all(|prob| *prob <= 0.0) {
        return fallback_argmax(&probs);
    }

    let dist = WeightedIndex::new(&probs).map_err(candle_core::Error::msg)?;
    Ok(dist.sample(rng) as u32)
}

fn top_logits_preview(logits: &Tensor, top_n: usize) -> candle_core::Result<Vec<(u32, f32)>> {
    let mut indexed = logits
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?
        .into_iter()
        .enumerate()
        .collect::<Vec<_>>();
    indexed.sort_unstable_by(|(_, lhs), (_, rhs)| {
        rhs.partial_cmp(lhs).unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(indexed
        .into_iter()
        .take(top_n)
        .map(|(idx, value)| (idx as u32, value))
        .collect())
}

fn tensor_stats_preview(tensor: &Tensor) -> candle_core::Result<(f32, f32, Vec<f32>)> {
    let values = tensor
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    if values.is_empty() {
        return Ok((0.0, 0.0, Vec::new()));
    }
    let mean = values.iter().copied().sum::<f32>() / values.len() as f32;
    let var = values
        .iter()
        .map(|value| {
            let diff = *value - mean;
            diff * diff
        })
        .sum::<f32>()
        / values.len() as f32;
    Ok((
        mean,
        var.sqrt(),
        values.into_iter().take(8).collect::<Vec<_>>(),
    ))
}

#[derive(Debug, Clone)]
struct ResizeMlp {
    linear_fc1: Arc<dyn QuantMethod>,
    linear_fc2: Arc<dyn QuantMethod>,
    act: Activation,
}

impl ResizeMlp {
    fn new(
        input_size: usize,
        intermediate_size: usize,
        output_size: usize,
        act: Activation,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            linear_fc1: quant_linear(
                input_size,
                intermediate_size,
                no_quant_config(),
                vb.pp("linear_fc1"),
            )?,
            linear_fc2: quant_linear(
                intermediate_size,
                output_size,
                no_quant_config(),
                vb.pp("linear_fc2"),
            )?,
            act,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        self.linear_fc2
            .forward(&self.act.forward(&self.linear_fc1.forward(hidden_states)?)?)
    }
}

#[derive(Debug, Clone)]
struct TalkerMlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act: Activation,
}

impl TalkerMlp {
    fn new(cfg: &Qwen3TtsTalkerConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: quant_linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                no_quant_config(),
                vb.pp("gate_proj"),
            )?,
            up_proj: quant_linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                no_quant_config(),
                vb.pp("up_proj"),
            )?,
            down_proj: quant_linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                no_quant_config(),
                vb.pp("down_proj"),
            )?,
            act: hidden_act(&cfg.hidden_act)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.act.forward(&self.gate_proj.forward(hidden_states)?)?;
        let up = self.up_proj.forward(hidden_states)?;
        self.down_proj.forward(&gate.broadcast_mul(&up)?)
    }
}

#[derive(Debug, Clone)]
struct SelfAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    q_norm: F32RmsNorm,
    k_norm: F32RmsNorm,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_interleaved: bool,
}

impl SelfAttention {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_theta: f64,
        max_position_embeddings: usize,
        rms_norm_eps: f64,
        rotary_interleaved: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        Ok(Self {
            q_proj: quant_linear_b(
                hidden_size,
                num_heads * head_dim,
                false,
                no_quant_config(),
                vb.pp("q_proj"),
            )?,
            k_proj: quant_linear_b(
                hidden_size,
                num_kv_heads * head_dim,
                false,
                no_quant_config(),
                vb.pp("k_proj"),
            )?,
            v_proj: quant_linear_b(
                hidden_size,
                num_kv_heads * head_dim,
                false,
                no_quant_config(),
                vb.pp("v_proj"),
            )?,
            o_proj: quant_linear_b(
                num_heads * head_dim,
                hidden_size,
                false,
                no_quant_config(),
                vb.pp("o_proj"),
            )?,
            q_norm: F32RmsNorm::new(head_dim, rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: F32RmsNorm::new(head_dim, rms_norm_eps, vb.pp("k_norm"))?,
            rotary_emb: RotaryEmbedding::new(
                rope_theta as f32,
                head_dim,
                max_position_embeddings,
                &device,
                false,
                dtype,
            )?,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_interleaved,
        })
    }

    fn apply_rotary(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_offset: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let (cos, sin) = self.rotary_emb.get_cos_sin()?;
        let cos = cos.narrow(0, position_offset, seq_len)?;
        let sin = sin.narrow(0, position_offset, seq_len)?;
        let q = if self.rotary_interleaved {
            candle_nn::rotary_emb::rope_i(&q.contiguous()?, &cos, &sin)?
        } else {
            candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?
        };
        let k = if self.rotary_interleaved {
            candle_nn::rotary_emb::rope_i(&k.contiguous()?, &cos, &sin)?
        } else {
            candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?
        };
        Ok((q, k))
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let q = self
            .q_proj
            .forward(hidden_states)?
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .apply(&self.q_norm)?;
        let k = self
            .k_proj
            .forward(hidden_states)?
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .apply(&self.k_norm)?;
        let v = self
            .v_proj
            .forward(hidden_states)?
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // The main talker uses interleaved MRoPE, while the code predictor uses the
        // regular rotary path. Mixing them breaks subgroup parity.
        let (q, k) = self.apply_rotary(&q, &k, 0)?;
        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let q = q.contiguous()?;
        let mut attn_weights = (q.matmul(&k_t)? / (self.head_dim as f64).sqrt())?;
        if let Some(mask) = attention_mask {
            attn_weights = attn_weights.broadcast_add(mask)?;
        }
        let attn_probs =
            candle_nn::ops::softmax(&attn_weights.to_dtype(DType::F32)?, candle_core::D::Minus1)?
                .to_dtype(q.dtype())?
                .contiguous()?;
        let v = v.contiguous()?;
        let attn_output = attn_probs.matmul(&v)?.transpose(1, 2)?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;
        self.o_proj.forward(&attn_output)
    }

    fn forward_with_past(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        past_kv: Option<&(Tensor, Tensor)>,
    ) -> candle_core::Result<(Tensor, (Tensor, Tensor))> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let q = self
            .q_proj
            .forward(hidden_states)?
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .apply(&self.q_norm)?;
        let k = self
            .k_proj
            .forward(hidden_states)?
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .apply(&self.k_norm)?;
        let v = self
            .v_proj
            .forward(hidden_states)?
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let position_offset = past_kv.map(|(past_k, _)| past_k.dims()[2]).unwrap_or(0);
        let (q, k) = self.apply_rotary(&q, &k, position_offset)?;

        let (k_cache, v_cache) = match past_kv {
            Some((past_k, past_v)) => (
                Tensor::cat(&[past_k, &k], 2)?,
                Tensor::cat(&[past_v, &v], 2)?,
            ),
            None => (k.clone(), v.clone()),
        };

        let k = repeat_kv(k_cache.clone(), self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v_cache.clone(), self.num_heads / self.num_kv_heads)?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let q = q.contiguous()?;
        let mut attn_weights = (q.matmul(&k_t)? / (self.head_dim as f64).sqrt())?;
        if let Some(mask) = attention_mask {
            attn_weights = attn_weights.broadcast_add(mask)?;
        }
        let attn_probs =
            candle_nn::ops::softmax(&attn_weights.to_dtype(DType::F32)?, candle_core::D::Minus1)?
                .to_dtype(q.dtype())?
                .contiguous()?;
        let v = v.contiguous()?;
        let attn_output = attn_probs.matmul(&v)?.transpose(1, 2)?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;
        Ok((self.o_proj.forward(&attn_output)?, (k_cache, v_cache)))
    }
}

#[derive(Debug, Clone)]
struct TalkerDecoderLayer {
    self_attn: SelfAttention,
    mlp: TalkerMlp,
    input_layernorm: F32RmsNorm,
    post_attention_layernorm: F32RmsNorm,
}

impl TalkerDecoderLayer {
    fn new(cfg: &Qwen3TtsTalkerConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: SelfAttention::new(
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.rope_theta,
                cfg.max_position_embeddings,
                cfg.rms_norm_eps,
                cfg.rope_scaling
                    .as_ref()
                    .and_then(|cfg| cfg.interleaved)
                    .unwrap_or(false),
                vb.pp("self_attn"),
            )?,
            mlp: TalkerMlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: F32RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: F32RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward(&hidden_states, attention_mask)?;
        let hidden_states = (hidden_states + residual)?;
        let residual = &hidden_states;
        let hidden_states = self
            .mlp
            .forward(&hidden_states.apply(&self.post_attention_layernorm)?)?;
        residual + hidden_states
    }

    fn forward_with_past(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        past_kv: Option<&(Tensor, Tensor)>,
    ) -> candle_core::Result<(Tensor, (Tensor, Tensor))> {
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let (hidden_states, kv_cache) =
            self.self_attn
                .forward_with_past(&hidden_states, attention_mask, past_kv)?;
        let hidden_states = (hidden_states + residual)?;
        let residual = &hidden_states;
        let hidden_states = self
            .mlp
            .forward(&hidden_states.apply(&self.post_attention_layernorm)?)?;
        Ok(((residual + hidden_states)?, kv_cache))
    }
}

#[derive(Debug, Clone)]
struct TalkerModel {
    codec_embedding: Embedding,
    text_embedding: Embedding,
    layers: Vec<TalkerDecoderLayer>,
    norm: F32RmsNorm,
    device: Device,
}

impl TalkerModel {
    fn new(cfg: &Qwen3TtsTalkerConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let model_vb = vb.pp("model");
        let layers_vb = model_vb.pp("layers");
        Ok(Self {
            codec_embedding: layers::embedding(
                cfg.vocab_size,
                cfg.hidden_size,
                model_vb.pp("codec_embedding"),
                &None,
            )?,
            text_embedding: layers::embedding(
                cfg.text_vocab_size,
                cfg.text_hidden_size,
                model_vb.pp("text_embedding"),
                &None,
            )?,
            layers: (0..cfg.num_hidden_layers)
                .map(|idx| TalkerDecoderLayer::new(cfg, layers_vb.pp(idx)))
                .collect::<Result<Vec<_>>>()?,
            norm: F32RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, model_vb.pp("norm"))?,
            device,
        })
    }

    fn forward_embeds(&self, mut hidden_states: Tensor) -> candle_core::Result<Tensor> {
        let attention_mask = causal_attention_mask(hidden_states.dim(1)?, &self.device)?
            .to_dtype(hidden_states.dtype())?;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, Some(&attention_mask))?;
        }
        hidden_states.apply(&self.norm)
    }
}

#[derive(Debug, Clone)]
struct CodePredictorModel {
    codec_embeddings: Vec<Embedding>,
    input_projection: Option<Arc<dyn QuantMethod>>,
    layers: Vec<TalkerDecoderLayer>,
    norm: F32RmsNorm,
    hidden_size: usize,
    device: Device,
}

impl CodePredictorModel {
    fn new(cfg: &Qwen3TtsTalkerConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let cp_cfg = &cfg.code_predictor_config;
        let device = vb.device().clone();
        let root = vb.pp("code_predictor");
        let model_vb = root.pp("model");
        let layers_vb = model_vb.pp("layers");
        Ok(Self {
            codec_embeddings: (0..cp_cfg.num_code_groups - 1)
                .map(|idx| {
                    layers::embedding(
                        cp_cfg.vocab_size,
                        cfg.hidden_size,
                        model_vb.pp("codec_embedding").pp(idx),
                        &None,
                    )
                })
                .collect::<candle_core::Result<Vec<_>>>()?,
            input_projection: if cp_cfg.hidden_size != cfg.hidden_size {
                Some(quant_linear(
                    cfg.hidden_size,
                    cp_cfg.hidden_size,
                    no_quant_config(),
                    root.pp("small_to_mtp_projection"),
                )?)
            } else {
                None
            },
            layers: (0..cp_cfg.num_hidden_layers)
                .map(|idx| {
                    let layer_cfg = Qwen3TtsTalkerConfig {
                        hidden_size: cp_cfg.hidden_size,
                        intermediate_size: cp_cfg.intermediate_size,
                        num_hidden_layers: cp_cfg.num_hidden_layers,
                        num_attention_heads: cp_cfg.num_attention_heads,
                        num_key_value_heads: cp_cfg.num_key_value_heads,
                        max_position_embeddings: cp_cfg.max_position_embeddings,
                        head_dim: cp_cfg.head_dim,
                        hidden_act: cp_cfg.hidden_act.clone(),
                        initializer_range: cp_cfg.initializer_range,
                        rms_norm_eps: cp_cfg.rms_norm_eps,
                        rope_theta: cp_cfg.rope_theta,
                        rope_scaling: cp_cfg.rope_scaling.clone(),
                        attention_bias: cp_cfg.attention_bias,
                        use_cache: cp_cfg.use_cache,
                        use_sliding_window: cp_cfg.use_sliding_window,
                        sliding_window: cp_cfg.sliding_window,
                        attention_dropout: cp_cfg.attention_dropout,
                        text_hidden_size: cfg.text_hidden_size,
                        text_vocab_size: cfg.text_vocab_size,
                        vocab_size: cp_cfg.vocab_size,
                        num_code_groups: cp_cfg.num_code_groups,
                        position_id_per_seconds: cfg.position_id_per_seconds,
                        codec_bos_id: cfg.codec_bos_id,
                        codec_eos_token_id: cfg.codec_eos_token_id,
                        codec_think_id: cfg.codec_think_id,
                        codec_nothink_id: cfg.codec_nothink_id,
                        codec_pad_id: cfg.codec_pad_id,
                        codec_think_bos_id: cfg.codec_think_bos_id,
                        codec_think_eos_id: cfg.codec_think_eos_id,
                        codec_language_id: cfg.codec_language_id.clone(),
                        spk_id: cfg.spk_id.clone(),
                        spk_is_dialect: cfg.spk_is_dialect.clone(),
                        code_predictor_config: cp_cfg.clone(),
                    };
                    TalkerDecoderLayer::new(&layer_cfg, layers_vb.pp(idx))
                })
                .collect::<Result<Vec<_>>>()?,
            norm: F32RmsNorm::new(cp_cfg.hidden_size, cp_cfg.rms_norm_eps, model_vb.pp("norm"))?,
            hidden_size: cp_cfg.hidden_size,
            device,
        })
    }

    fn embed_group(
        &self,
        group_idx: usize,
        token_id: u32,
        vocab_size: usize,
        device: &Device,
    ) -> Result<Tensor> {
        ensure_ids_in_range(&[token_id], vocab_size, "generated predictor codec")?;
        Ok(self.codec_embeddings[group_idx].forward(&ids_tensor(&[token_id], device)?)?)
    }

    fn forward_embeds(&self, mut hidden_states: Tensor) -> candle_core::Result<Tensor> {
        if let Some(proj) = &self.input_projection {
            hidden_states = proj.forward(&hidden_states)?;
        }
        let attention_mask = causal_attention_mask(hidden_states.dim(1)?, &self.device)?
            .to_dtype(hidden_states.dtype())?;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, Some(&attention_mask))?;
        }
        hidden_states.apply(&self.norm)
    }

    fn forward_embeds_cached(
        &self,
        mut hidden_states: Tensor,
        caches: &mut [Option<(Tensor, Tensor)>],
        debug_label: Option<&str>,
    ) -> candle_core::Result<Tensor> {
        if let Some(proj) = &self.input_projection {
            hidden_states = proj.forward(&hidden_states)?;
        }
        if let Some(label) = debug_label {
            let (mean, std, preview) = tensor_stats_preview(&hidden_states)?;
            tracing::info!(
                "Qwen3-TTS code_predictor {} projected mean={:.6} std={:.6} vec8={:?}",
                label,
                mean,
                std,
                preview
            );
        }
        let attention_mask = if hidden_states.dim(1)? > 1 {
            Some(
                causal_attention_mask(hidden_states.dim(1)?, &self.device)?
                    .to_dtype(hidden_states.dtype())?,
            )
        } else {
            None
        };
        for (layer_idx, (layer, cache_slot)) in
            self.layers.iter().zip(caches.iter_mut()).enumerate()
        {
            let (next_hidden_states, next_cache) = layer.forward_with_past(
                &hidden_states,
                attention_mask.as_ref(),
                cache_slot.as_ref(),
            )?;
            hidden_states = next_hidden_states;
            *cache_slot = Some(next_cache);
            if let Some(label) = debug_label {
                if layer_idx < 2 || layer_idx + 1 == self.layers.len() {
                    let (mean, std, preview) = tensor_stats_preview(&hidden_states)?;
                    tracing::info!(
                        "Qwen3-TTS code_predictor {} layer{} mean={:.6} std={:.6} vec8={:?}",
                        label,
                        layer_idx,
                        mean,
                        std,
                        preview
                    );
                }
            }
        }
        let hidden_states = hidden_states.apply(&self.norm)?;
        if let Some(label) = debug_label {
            let (mean, std, preview) = tensor_stats_preview(&hidden_states)?;
            tracing::info!(
                "Qwen3-TTS code_predictor {} norm mean={:.6} std={:.6} vec8={:?}",
                label,
                mean,
                std,
                preview
            );
        }
        Ok(hidden_states)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3TtsTalker {
    cfg: Qwen3TtsTalkerConfig,
    tts_bos_token_id: u32,
    tts_eos_token_id: u32,
    tts_pad_token_id: u32,
    model: TalkerModel,
    text_projection: ResizeMlp,
    codec_head: Arc<dyn QuantMethod>,
    code_predictor: CodePredictorModel,
    code_predictor_heads: Vec<Arc<dyn QuantMethod>>,
    device: Device,
}

impl Qwen3TtsTalker {
    pub fn new(cfg: &Qwen3TtsConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let root = vb.pp("talker");
        let code_predictor_root = root.pp("code_predictor");
        let talker_cfg = &cfg.talker_config;
        let cp_cfg = &talker_cfg.code_predictor_config;
        Ok(Self {
            cfg: talker_cfg.clone(),
            tts_bos_token_id: cfg.tts_bos_token_id,
            tts_eos_token_id: cfg.tts_eos_token_id,
            tts_pad_token_id: cfg.tts_pad_token_id,
            model: TalkerModel::new(talker_cfg, root.clone())?,
            text_projection: ResizeMlp::new(
                talker_cfg.text_hidden_size,
                talker_cfg.text_hidden_size,
                talker_cfg.hidden_size,
                hidden_act(&talker_cfg.hidden_act)?,
                root.pp("text_projection"),
            )?,
            codec_head: quant_linear_b(
                talker_cfg.hidden_size,
                talker_cfg.vocab_size,
                false,
                no_quant_config(),
                root.pp("codec_head"),
            )?,
            code_predictor: CodePredictorModel::new(talker_cfg, root.clone())?,
            code_predictor_heads: (0..cp_cfg.num_code_groups - 1)
                .map(|idx| {
                    quant_linear_b(
                        cp_cfg.hidden_size,
                        cp_cfg.vocab_size,
                        false,
                        no_quant_config(),
                        code_predictor_root.pp("lm_head").pp(idx),
                    )
                })
                .collect::<candle_core::Result<Vec<_>>>()?,
            device,
        })
    }

    fn text_embed_checked(&self, ids: &[u32], label: &str) -> Result<Tensor> {
        ensure_ids_in_range(ids, self.cfg.text_vocab_size, label)?;
        Ok(self.text_projection.forward(
            &self
                .model
                .text_embedding
                .forward(&ids_tensor(ids, &self.device)?)?,
        )?)
    }

    fn codec_embed_checked(&self, ids: &[u32], label: &str) -> Result<Tensor> {
        ensure_ids_in_range(ids, self.cfg.vocab_size, label)?;
        Ok(self
            .model
            .codec_embedding
            .forward(&ids_tensor(ids, &self.device)?)?)
    }

    fn ensure_generated_codec_id(&self, token_id: u32, label: &str) -> Result<()> {
        ensure_ids_in_range(&[token_id], self.cfg.vocab_size, label)
    }

    fn ensure_generated_predictor_id(&self, token_id: u32, label: &str) -> Result<()> {
        ensure_ids_in_range(
            &[token_id],
            self.cfg.code_predictor_config.vocab_size,
            label,
        )
    }

    fn text_embed(&self, ids: &[u32]) -> candle_core::Result<Tensor> {
        self.text_projection.forward(
            &self
                .model
                .text_embedding
                .forward(&ids_tensor(ids, &self.device)?)?,
        )
    }

    fn codec_embed(&self, ids: &[u32]) -> candle_core::Result<Tensor> {
        self.model
            .codec_embedding
            .forward(&ids_tensor(ids, &self.device)?)
    }

    fn ref_code_embed_checked(&self, ref_codes: &[Vec<u32>]) -> Result<Tensor> {
        if ref_codes.is_empty() {
            anyhow::bail!("Qwen3-TTS ICL prompt requires non-empty `ref_code` frames.");
        }
        if ref_codes
            .iter()
            .any(|frame| frame.len() != self.cfg.num_code_groups)
        {
            anyhow::bail!(
                "Qwen3-TTS `ref_code` frames must each contain exactly {} code groups.",
                self.cfg.num_code_groups
            );
        }

        let first_group = ref_codes.iter().map(|frame| frame[0]).collect::<Vec<_>>();
        let mut summed = self.codec_embed_checked(&first_group, "icl ref_code group 0")?;

        for group_idx in 1..self.cfg.num_code_groups {
            let group_codes = ref_codes
                .iter()
                .map(|frame| frame[group_idx])
                .collect::<Vec<_>>();
            ensure_ids_in_range(
                &group_codes,
                self.cfg.code_predictor_config.vocab_size,
                "icl predictor ref_code",
            )?;
            let embed = self.code_predictor.codec_embeddings[group_idx - 1]
                .forward(&ids_tensor(&group_codes, &self.device)?)?;
            summed = summed.broadcast_add(&embed)?;
        }

        let codec_bos = self.codec_embed_checked(&[self.cfg.codec_bos_id], "icl codec bos")?;
        Ok(Tensor::cat(&[&codec_bos, &summed], 1)?)
    }

    fn build_icl_prompt_embeddings(
        &self,
        prepared: &Qwen3TtsPreparedRequest,
        tts_pad_embed: &Tensor,
        tts_eos_embed: &Tensor,
        non_streaming_mode: bool,
    ) -> Result<(Tensor, Tensor)> {
        let ref_ids = prepared
            .ref_ids
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Qwen3-TTS ICL prompt requires `ref_ids`."))?;
        let ref_codes = prepared
            .ref_codes
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Qwen3-TTS ICL prompt requires `ref_code`."))?;

        if prepared.input_ids.len() < 8 || ref_ids.len() < 5 {
            anyhow::bail!(
                "Qwen3-TTS ICL prompt received unexpectedly short input/ref token sequences."
            );
        }

        let mut text_tokens = ref_ids[3..ref_ids.len() - 2].to_vec();
        text_tokens.extend_from_slice(&prepared.input_ids[3..prepared.input_ids.len() - 5]);
        let text_embed = Tensor::cat(
            &[
                &self.text_embed_checked(&text_tokens, "icl text tokens")?,
                tts_eos_embed,
            ],
            1,
        )?;

        let codec_embed = self.ref_code_embed_checked(ref_codes)?;
        let text_lens = text_embed.dim(1)?;
        let codec_lens = codec_embed.dim(1)?;

        if non_streaming_mode {
            let pad_ids = vec![self.cfg.codec_pad_id; text_lens];
            let icl_input_embed = text_embed.broadcast_add(
                &self.codec_embed_checked(&pad_ids, "icl non-streaming codec pad")?,
            )?;
            let codec_pad = tts_pad_embed.expand((1, codec_lens, self.cfg.hidden_size))?;
            return Ok((
                Tensor::cat(
                    &[&icl_input_embed, &codec_embed.broadcast_add(&codec_pad)?],
                    1,
                )?,
                tts_pad_embed.clone(),
            ));
        }

        if text_lens > codec_lens {
            Ok((
                text_embed
                    .narrow(1, 0, codec_lens)?
                    .broadcast_add(&codec_embed)?,
                text_embed.narrow(1, codec_lens, text_lens - codec_lens)?,
            ))
        } else {
            let text_embed = if codec_lens > text_lens {
                let pad =
                    tts_pad_embed.expand((1, codec_lens - text_lens, self.cfg.hidden_size))?;
                Tensor::cat(&[&text_embed, &pad], 1)?
            } else {
                text_embed
            };
            Ok((
                text_embed.broadcast_add(&codec_embed)?,
                tts_pad_embed.clone(),
            ))
        }
    }

    fn speaker_embed_from_request(
        &self,
        prepared: &Qwen3TtsPreparedRequest,
        speaker_embedding: Option<&Tensor>,
    ) -> Result<Option<Tensor>> {
        if let Some(speaker_embedding) = speaker_embedding {
            let speaker_embedding = match speaker_embedding.rank() {
                3 => speaker_embedding.clone(),
                2 => speaker_embedding.unsqueeze(1)?,
                rank => {
                    anyhow::bail!(
                        "Qwen3-TTS speaker embedding must be rank 2 or 3, got rank {rank} with shape {:?}.",
                        speaker_embedding.dims()
                    );
                }
            };
            return Ok(Some(speaker_embedding));
        }
        let Some(speaker) = prepared.speaker.as_deref() else {
            return Ok(None);
        };
        let ids = self
            .cfg
            .spk_id
            .get(&speaker.to_ascii_lowercase())
            .ok_or_else(|| anyhow::anyhow!("Unknown Qwen3-TTS speaker `{speaker}`."))?;
        if ids.len() != 1 {
            anyhow::bail!(
                "Qwen3-TTS speaker `{speaker}` resolved to {} ids; the native port currently expects exactly one.",
                ids.len()
            );
        }
        Ok(Some(
            self.codec_embed_checked(ids, "speaker codec")?.reshape((
                1,
                1,
                self.cfg.hidden_size,
            ))?,
        ))
    }

    fn resolve_language_id(&self, prepared: &Qwen3TtsPreparedRequest) -> Result<Option<u32>> {
        let normalized = prepared.language.trim().to_ascii_lowercase();
        if normalized == "auto" {
            return Ok(None);
        }
        let language_id = self
            .cfg
            .codec_language_id
            .get(&normalized)
            .copied()
            .ok_or_else(|| {
                anyhow::anyhow!("Unsupported Qwen3-TTS language `{}`.", prepared.language)
            })?;
        Ok(Some(language_id))
    }

    fn build_prompt_embeddings(
        &self,
        prepared: &Qwen3TtsPreparedRequest,
        speaker_embedding: Option<&Tensor>,
        non_streaming_mode: bool,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        tracing::info!(
            "Qwen3-TTS input-ids head10={:?} len={}",
            prepared
                .input_ids
                .iter()
                .take(10)
                .copied()
                .collect::<Vec<_>>(),
            prepared.input_ids.len()
        );
        let speaker_embed = self.speaker_embed_from_request(prepared, speaker_embedding)?;
        if let Some(speaker_embed) = &speaker_embed {
            let (mean, std, head8) = tensor_stats_preview(speaker_embed)?;
            tracing::info!(
                "Qwen3-TTS speaker stats mean={:.6} std={:.6} head8={:?}",
                mean,
                std,
                head8
            );
        }
        let language_id = self.resolve_language_id(prepared)?;

        let tts_special = self.text_embed_checked(
            &[
                self.tts_bos_token_id,
                self.tts_eos_token_id,
                self.tts_pad_token_id,
            ],
            "tts special",
        )?;
        let tts_bos_embed = tts_special.narrow(1, 0, 1)?;
        let tts_eos_embed = tts_special.narrow(1, 1, 1)?;
        let tts_pad_embed = tts_special.narrow(1, 2, 1)?;

        let codec_prefill = if let Some(language_id) = language_id {
            vec![
                self.cfg.codec_think_id,
                self.cfg.codec_think_bos_id,
                language_id,
                self.cfg.codec_think_eos_id,
            ]
        } else {
            vec![
                self.cfg.codec_nothink_id,
                self.cfg.codec_think_bos_id,
                self.cfg.codec_think_eos_id,
            ]
        };
        let codec_input_embedding_0 = self.codec_embed_checked(&codec_prefill, "codec prefill")?;
        let codec_input_embedding_1 = self.codec_embed_checked(
            &[self.cfg.codec_pad_id, self.cfg.codec_bos_id],
            "codec prefix",
        )?;
        let codec_input_embedding = if let Some(speaker_embed) = speaker_embed {
            Tensor::cat(
                &[
                    &codec_input_embedding_0,
                    &speaker_embed,
                    &codec_input_embedding_1,
                ],
                1,
            )?
        } else {
            Tensor::cat(&[&codec_input_embedding_0, &codec_input_embedding_1], 1)?
        };
        let (codec_mean, codec_std, codec_head8) = tensor_stats_preview(&codec_input_embedding)?;
        tracing::info!(
            "Qwen3-TTS codec-input stats mean={:.6} std={:.6} head8={:?}",
            codec_mean,
            codec_std,
            codec_head8
        );

        let role_raw = self
            .model
            .text_embedding
            .forward(&ids_tensor(&prepared.input_ids[..3], &self.device)?)?;
        let (role_raw_mean, role_raw_std, role_raw_head8) = tensor_stats_preview(&role_raw)?;
        tracing::info!(
            "Qwen3-TTS role-raw stats mean={:.6} std={:.6} head8={:?}",
            role_raw_mean,
            role_raw_std,
            role_raw_head8
        );
        let role_embed = self.text_embed_checked(&prepared.input_ids[..3], "role prefix")?;
        let (role_mean, role_std, role_head8) = tensor_stats_preview(&role_embed)?;
        tracing::info!(
            "Qwen3-TTS role-embed stats mean={:.6} std={:.6} head8={:?}",
            role_mean,
            role_std,
            role_head8
        );
        let pad_prefix =
            tts_pad_embed.expand((1, codec_input_embedding.dim(1)? - 2, self.cfg.hidden_size))?;
        let talker_prefix = Tensor::cat(&[&pad_prefix, &tts_bos_embed], 1)?.broadcast_add(
            &codec_input_embedding.narrow(1, 0, codec_input_embedding.dim(1)? - 1)?,
        )?;
        let (prefix_mean, prefix_std, prefix_head8) = tensor_stats_preview(&talker_prefix)?;
        tracing::info!(
            "Qwen3-TTS talker-prefix stats mean={:.6} std={:.6} head8={:?}",
            prefix_mean,
            prefix_std,
            prefix_head8
        );
        let mut talker_input_embed = Tensor::cat(&[&role_embed, &talker_prefix], 1)?;

        let trailing_text_hidden = if matches!(prepared.task_type, Qwen3TtsTaskType::Base)
            && prepared.requires_ref_codes
        {
            let (icl_input_embed, trailing_text_hidden) = self.build_icl_prompt_embeddings(
                prepared,
                &tts_pad_embed,
                &tts_eos_embed,
                non_streaming_mode,
            )?;
            talker_input_embed = Tensor::cat(&[&talker_input_embed, &icl_input_embed], 1)?;
            trailing_text_hidden
        } else {
            let text_first_raw = self
                .model
                .text_embedding
                .forward(&ids_tensor(&prepared.input_ids[3..4], &self.device)?)?;
            let (text_first_raw_mean, text_first_raw_std, text_first_raw_head8) =
                tensor_stats_preview(&text_first_raw)?;
            tracing::info!(
                "Qwen3-TTS text-first-raw stats mean={:.6} std={:.6} head8={:?}",
                text_first_raw_mean,
                text_first_raw_std,
                text_first_raw_head8
            );
            let text_first = self
                .text_embed_checked(&prepared.input_ids[3..4], "first text token")?
                .broadcast_add(&codec_input_embedding.narrow(
                    1,
                    codec_input_embedding.dim(1)? - 1,
                    1,
                )?)?;
            let (text_first_mean, text_first_std, text_first_head8) =
                tensor_stats_preview(&text_first)?;
            tracing::info!(
                "Qwen3-TTS text-first stats mean={:.6} std={:.6} head8={:?}",
                text_first_mean,
                text_first_std,
                text_first_head8
            );
            talker_input_embed = Tensor::cat(&[&talker_input_embed, &text_first], 1)?;

            if non_streaming_mode {
                let text_embed = self.text_embed_checked(
                    &prepared.input_ids[3..prepared.input_ids.len().saturating_sub(5)],
                    "non-streaming text tail",
                )?;
                let text_pad = self.codec_embed_checked(
                    &vec![self.cfg.codec_pad_id; text_embed.dim(1)?],
                    "non-streaming text pad",
                )?;
                talker_input_embed =
                    talker_input_embed.narrow(1, 0, talker_input_embed.dim(1)? - 1)?;
                let prompt_tail = Tensor::cat(
                    &[
                        &text_embed.broadcast_add(&text_pad)?,
                        &tts_eos_embed.broadcast_add(&self.codec_embed_checked(
                            &[self.cfg.codec_pad_id],
                            "tts eos codec pad",
                        )?)?,
                        &tts_pad_embed.broadcast_add(&self.codec_embed_checked(
                            &[self.cfg.codec_bos_id],
                            "tts pad codec bos",
                        )?)?,
                    ],
                    1,
                )?;
                talker_input_embed = Tensor::cat(&[&talker_input_embed, &prompt_tail], 1)?;
                tts_pad_embed.clone()
            } else {
                Tensor::cat(
                    &[
                        &self.text_embed_checked(
                            &prepared.input_ids[4..prepared.input_ids.len().saturating_sub(5)],
                            "streaming text tail",
                        )?,
                        &tts_eos_embed,
                    ],
                    1,
                )?
            }
        };

        let (prompt_mean, prompt_std, prompt_head8) = tensor_stats_preview(&talker_input_embed)?;
        tracing::info!(
            "Qwen3-TTS prompt-full stats mean={:.6} std={:.6} head8={:?}",
            prompt_mean,
            prompt_std,
            prompt_head8
        );

        let (prompt_last_mean, prompt_last_std, prompt_last_head8) = tensor_stats_preview(
            &talker_input_embed.narrow(1, talker_input_embed.dim(1)? - 1, 1)?,
        )?;
        tracing::info!(
            "Qwen3-TTS prompt-last stats mean={:.6} std={:.6} head8={:?}",
            prompt_last_mean,
            prompt_last_std,
            prompt_last_head8
        );

        Ok((talker_input_embed, trailing_text_hidden, tts_pad_embed))
    }

    fn sample_cfg(
        cfg: &SpeechGenerationConfig,
        _prepared: &Qwen3TtsPreparedRequest,
    ) -> Result<(
        usize,
        f32,
        f32,
        Option<usize>,
        f32,
        bool,
        f32,
        f32,
        Option<usize>,
    )> {
        let SpeechGenerationConfig::Qwen3Tts {
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            subtalker_do_sample,
            subtalker_temperature,
            subtalker_top_p,
            subtalker_top_k,
        } = cfg
        else {
            anyhow::bail!("Qwen3-TTS talker received a non-Qwen3-TTS sampling config.")
        };
        Ok((
            max_new_tokens.unwrap_or(2048),
            *temperature,
            *top_p,
            *top_k,
            *repetition_penalty,
            *subtalker_do_sample,
            *subtalker_temperature,
            *subtalker_top_p,
            *subtalker_top_k,
        ))
    }

    pub fn generate_codes(
        &self,
        prepared: &Qwen3TtsPreparedRequest,
        speaker_embedding: Option<&Tensor>,
        cfg: &SpeechGenerationConfig,
    ) -> Result<Tensor> {
        let total_started = Instant::now();
        let (
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            subtalker_do_sample,
            subtalker_temperature,
            subtalker_top_p,
            subtalker_top_k,
        ) = Self::sample_cfg(cfg, prepared)?;
        // Qwen3-TTS Base/voice-clone defaults to the simulated streaming prompt path
        // in the Python reference. The non-streaming path is primarily an explicit opt-in.
        let non_streaming_mode = false;
        let (mut sequence_embeds, trailing_text_hidden, tts_pad_embed) =
            self.build_prompt_embeddings(prepared, speaker_embedding, non_streaming_mode)?;
        let mut generated = Vec::new();
        let mut first_codes_context = Vec::new();
        let mut rng = StdRng::from_os_rng();
        let suppressed_first_head_tokens = ((self.cfg.vocab_size.saturating_sub(1024)) as u32
            ..self.cfg.vocab_size as u32)
            .filter(|token_id| *token_id != self.cfg.codec_eos_token_id)
            .collect::<Vec<_>>();
        tracing::info!(
            "Qwen3-TTS generate_codes start: max_new_tokens={} prompt_seq_len={} trailing_text_len={}",
            max_new_tokens,
            sequence_embeds.dim(1)?,
            trailing_text_hidden.dim(1)?
        );

        for step in 0..max_new_tokens {
            let step_started = Instant::now();
            let hidden_states = self.model.forward_embeds(sequence_embeds.clone())?;
            let last_hidden = hidden_states.narrow(1, hidden_states.dim(1)? - 1, 1)?;
            let first_logits = self
                .codec_head
                .forward(&last_hidden)?
                .squeeze(0)?
                .squeeze(0)?;
            if step == 0 {
                tracing::info!(
                    "Qwen3-TTS first-head top10={:?}",
                    top_logits_preview(&first_logits, 10)?
                );
            }
            let first_code = sample_next_token(
                &first_logits,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                &first_codes_context,
                &suppressed_first_head_tokens,
                &mut rng,
            )?;
            self.ensure_generated_codec_id(first_code, "generated first codec")?;
            if step < 3 {
                tracing::info!(
                    "Qwen3-TTS generate_codes step={} first_code={} elapsed_ms={}",
                    step,
                    first_code,
                    step_started.elapsed().as_millis()
                );
            }

            let mut frame_codes = vec![first_code];
            let mut frame_embeds =
                vec![self.codec_embed_checked(&[first_code], "generated first codec")?];
            if first_code == self.cfg.codec_eos_token_id {
                tracing::info!("Qwen3-TTS generate_codes hit eos at step={}", step);
                break;
            }
            first_codes_context.push(first_code);

            let mut predictor_cache = vec![None; self.code_predictor.layers.len()];
            let mut predictor_hidden = self.code_predictor.forward_embeds_cached(
                Tensor::cat(&[&last_hidden, &frame_embeds[0]], 1)?,
                &mut predictor_cache,
                if step == 0 { Some("prefill_g0") } else { None },
            )?;
            for group_idx in 0..self.cfg.num_code_groups - 1 {
                let group_started = Instant::now();
                let predictor_last = predictor_hidden.narrow(1, predictor_hidden.dim(1)? - 1, 1)?;
                if step == 0 && (group_idx == 3 || group_idx == 4) {
                    let (mean, std, preview) = tensor_stats_preview(&predictor_last)?;
                    tracing::info!(
                        "Qwen3-TTS generate_codes step={} group={} predictor_last mean={:.6} std={:.6} vec8={:?}",
                        step,
                        group_idx,
                        mean,
                        std,
                        preview
                    );
                }
                let group_logits = self.code_predictor_heads[group_idx]
                    .forward(&predictor_last)?
                    .squeeze(0)?
                    .squeeze(0)?;
                if step == 0 {
                    tracing::info!(
                        "Qwen3-TTS generate_codes step={} group={} top10={:?}",
                        step,
                        group_idx,
                        top_logits_preview(&group_logits, 10)?
                    );
                }
                let code = sample_next_token(
                    &group_logits,
                    if subtalker_do_sample {
                        subtalker_temperature
                    } else {
                        0.0
                    },
                    subtalker_top_p,
                    subtalker_top_k,
                    1.0,
                    &[],
                    &[],
                    &mut rng,
                )?;
                self.ensure_generated_predictor_id(code, "generated predictor codec")?;
                if step == 0 {
                    tracing::info!(
                        "Qwen3-TTS generate_codes step={} group={} code={} elapsed_ms={}",
                        step,
                        group_idx,
                        code,
                        group_started.elapsed().as_millis()
                    );
                }
                frame_codes.push(code);
                let embed = self.code_predictor.embed_group(
                    group_idx,
                    code,
                    self.cfg.code_predictor_config.vocab_size,
                    &self.device,
                )?;
                if step == 0 && (group_idx == 2 || group_idx == 3) {
                    let (mean, std, preview) = tensor_stats_preview(&embed)?;
                    tracing::info!(
                        "Qwen3-TTS generate_codes step={} group={} embed mean={:.6} std={:.6} vec8={:?}",
                        step,
                        group_idx,
                        mean,
                        std,
                        preview
                    );
                }
                frame_embeds.push(embed.clone());
                if group_idx + 1 < self.cfg.num_code_groups - 1 {
                    predictor_hidden = self.code_predictor.forward_embeds_cached(
                        embed,
                        &mut predictor_cache,
                        if step == 0 && group_idx < 4 {
                            Some(match group_idx {
                                0 => "step0_embed_g0",
                                1 => "step0_embed_g1",
                                2 => "step0_embed_g2",
                                3 => "step0_embed_g3",
                                _ => unreachable!(),
                            })
                        } else {
                            None
                        },
                    )?;
                }
            }

            let mut frame_hidden = frame_embeds[0].clone();
            for embed in frame_embeds.iter().skip(1) {
                frame_hidden = frame_hidden.broadcast_add(embed)?;
            }
            let conditioning = if step < trailing_text_hidden.dim(1)? {
                trailing_text_hidden.narrow(1, step, 1)?
            } else {
                tts_pad_embed.clone()
            };
            frame_hidden = frame_hidden.broadcast_add(&conditioning)?;
            sequence_embeds = Tensor::cat(&[&sequence_embeds, &frame_hidden], 1)?;
            generated.push(frame_codes);
        }

        let codes = generated.into_iter().flatten().collect::<Vec<_>>();
        let frames = if self.cfg.num_code_groups == 0 {
            0
        } else {
            codes.len() / self.cfg.num_code_groups
        };
        let out = Tensor::from_vec(codes, (1, frames, self.cfg.num_code_groups), &self.device)?;
        tracing::info!(
            "Qwen3-TTS generate_codes done: frames={} total_elapsed_ms={}",
            frames,
            total_started.elapsed().as_millis()
        );
        Ok(out)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
