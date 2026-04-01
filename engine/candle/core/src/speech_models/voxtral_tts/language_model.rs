use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use mistralrs_quant::{
    linear_b as quant_linear_b, QuantMethod, QuantizedConfig, ShardedVarBuilder,
};
use std::sync::Arc;

use crate::layers::{embedding, repeat_kv, MatMul, RmsNorm, RotaryEmbedding};

use super::config::VoxtralTtsConfig;

fn no_quant_config() -> &'static Option<QuantizedConfig> {
    static NO_QUANT_CONFIG: Option<QuantizedConfig> = None;
    &NO_QUANT_CONFIG
}

fn causal_mask(seq_len: usize, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
    let mut values = Vec::with_capacity(seq_len * seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            values.push(if j > i { f32::NEG_INFINITY } else { 0.0 });
        }
    }
    Tensor::from_vec(values, (1, 1, seq_len, seq_len), device)?.to_dtype(dtype)
}

fn qmethod_matmul_autocast(xs: &Tensor, layer: &dyn QuantMethod) -> candle_core::Result<Tensor> {
    let original_dtype = xs.dtype();
    let mut xs = xs.clone();
    if let Some(t) = layer.quantized_act_type() {
        xs = xs.to_dtype(t)?;
    }
    let mut ys = MatMul.qmethod_matmul(&xs, layer)?;
    if layer.quantized_act_type().is_some() {
        ys = ys.to_dtype(original_dtype)?;
    }
    Ok(ys)
}

#[derive(Debug, Clone)]
struct DecoderAttention {
    wq: Arc<dyn QuantMethod>,
    wk: Arc<dyn QuantMethod>,
    wv: Arc<dyn QuantMethod>,
    wo: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: RotaryEmbedding,
}

impl DecoderAttention {
    fn new(cfg: &VoxtralTtsConfig, vb: ShardedVarBuilder, device: &Device) -> Result<Self> {
        Ok(Self {
            wq: quant_linear_b(
                cfg.dim,
                cfg.n_heads * cfg.head_dim,
                false,
                no_quant_config(),
                vb.pp("wq"),
            )?,
            wk: quant_linear_b(
                cfg.dim,
                cfg.n_kv_heads * cfg.head_dim,
                false,
                no_quant_config(),
                vb.pp("wk"),
            )?,
            wv: quant_linear_b(
                cfg.dim,
                cfg.n_kv_heads * cfg.head_dim,
                false,
                no_quant_config(),
                vb.pp("wv"),
            )?,
            wo: quant_linear_b(
                cfg.n_heads * cfg.head_dim,
                cfg.dim,
                false,
                no_quant_config(),
                vb.pp("wo"),
            )?,
            num_heads: cfg.n_heads,
            num_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            rotary_emb: RotaryEmbedding::new(
                cfg.rope_theta as f32,
                cfg.head_dim,
                cfg.max_position_embeddings.unwrap_or(cfg.max_seq_len),
                device,
                false,
                vb.dtype(),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, seq_len, _) = xs.dims3()?;
        let q = qmethod_matmul_autocast(xs, &*self.wq)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = qmethod_matmul_autocast(xs, &*self.wk)?
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = qmethod_matmul_autocast(xs, &*self.wv)?
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let (q, k) = self.rotary_emb.forward(&q, &k, &[0])?;
        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;
        let attn = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?
            / (self.head_dim as f64).sqrt())?
        .broadcast_add(&causal_mask(seq_len, q.dtype(), q.device())?)?;
        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let ys = attn.matmul(&v)?.transpose(1, 2)?.reshape((
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;
        qmethod_matmul_autocast(&ys, &*self.wo)
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        vec![&mut self.wq, &mut self.wk, &mut self.wv, &mut self.wo]
    }
}

#[derive(Debug, Clone)]
struct DecoderMlp {
    w1: Arc<dyn QuantMethod>,
    w2: Arc<dyn QuantMethod>,
    w3: Arc<dyn QuantMethod>,
}

impl DecoderMlp {
    fn new(cfg: &VoxtralTtsConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            w1: quant_linear_b(
                cfg.dim,
                cfg.hidden_dim,
                false,
                no_quant_config(),
                vb.pp("w1"),
            )?,
            w2: quant_linear_b(
                cfg.hidden_dim,
                cfg.dim,
                false,
                no_quant_config(),
                vb.pp("w2"),
            )?,
            w3: quant_linear_b(
                cfg.dim,
                cfg.hidden_dim,
                false,
                no_quant_config(),
                vb.pp("w3"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let gate = candle_nn::ops::silu(&qmethod_matmul_autocast(xs, &*self.w1)?)?;
        let up = qmethod_matmul_autocast(xs, &*self.w3)?;
        qmethod_matmul_autocast(&gate.broadcast_mul(&up)?, &*self.w2)
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        vec![&mut self.w1, &mut self.w2, &mut self.w3]
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    attention: DecoderAttention,
    feed_forward: DecoderMlp,
    attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &VoxtralTtsConfig, vb: ShardedVarBuilder, device: &Device) -> Result<Self> {
        Ok(Self {
            attention: DecoderAttention::new(cfg, vb.pp("attention"), device)?,
            feed_forward: DecoderMlp::new(cfg, vb.pp("feed_forward"))?,
            attention_norm: RmsNorm::new(cfg.dim, cfg.norm_eps, vb.pp("attention_norm"))?,
            ffn_norm: RmsNorm::new(cfg.dim, cfg.norm_eps, vb.pp("ffn_norm"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let hidden = (xs + self.attention.forward(&self.attention_norm.forward(xs)?)?)?;
        let ff = self
            .feed_forward
            .forward(&self.ffn_norm.forward(&hidden)?)?;
        hidden + ff
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        let mut layers = self.attention.get_isq_layers();
        layers.extend(self.feed_forward.get_isq_layers());
        layers
    }
}

#[derive(Debug, Clone)]
pub struct VoxtralTtsLanguageModel {
    tok_embeddings: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    dtype: DType,
}

impl VoxtralTtsLanguageModel {
    pub fn new(cfg: &VoxtralTtsConfig, vb: ShardedVarBuilder, device: &Device) -> Result<Self> {
        Ok(Self {
            tok_embeddings: embedding(
                cfg.vocab_size,
                cfg.dim,
                vb.pp("mm_audio_embeddings").pp("tok_embeddings"),
                &None,
            )?,
            layers: (0..cfg.n_layers)
                .map(|idx| DecoderLayer::new(cfg, vb.pp("layers").pp(idx), device))
                .collect::<Result<Vec<_>>>()?,
            norm: RmsNorm::new(cfg.dim, cfg.norm_eps, vb.pp("norm"))?,
            dtype: vb.dtype(),
        })
    }

    pub fn forward_with_voice(
        &self,
        input_ids: &Tensor,
        audio_token_id: u32,
        audio_embeddings: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let text_embeds = self.tok_embeddings.forward(input_ids)?;
        let input_embeds = if let Some(audio_embeddings) = audio_embeddings {
            self.splice_audio_embeddings(text_embeds, input_ids, audio_token_id, audio_embeddings)?
        } else {
            text_embeds
        };
        let mut hidden = if input_embeds.dtype() == self.dtype {
            input_embeds
        } else {
            input_embeds.to_dtype(self.dtype)?
        };
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        self.norm.forward(&hidden)
    }

    pub fn last_hidden_state(
        &self,
        input_ids: &Tensor,
        audio_token_id: u32,
        audio_embeddings: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let hidden = self.forward_with_voice(input_ids, audio_token_id, audio_embeddings)?;
        hidden.i((.., hidden.dim(1)? - 1, ..))
    }

    fn splice_audio_embeddings(
        &self,
        text_embeds: Tensor,
        input_ids: &Tensor,
        audio_token_id: u32,
        audio_embeddings: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let positions = input_ids.flatten_all()?.to_vec1::<u32>()?;
        let expected = audio_embeddings.dim(0)?;
        let mut audio_positions = Vec::with_capacity(expected);
        for (idx, token) in positions.iter().enumerate() {
            if *token == audio_token_id {
                audio_positions.push(idx);
            }
        }
        if audio_positions.is_empty() {
            return Ok(text_embeds);
        }
        if audio_positions.len() != expected {
            candle_core::bail!(
                "audio embedding/audio placeholder mismatch: prompt has {} [AUDIO] tokens, embedding has {} frames",
                audio_positions.len(),
                expected
            );
        }

        let audio_embeddings = audio_embeddings
            .to_device(text_embeds.device())?
            .to_dtype(text_embeds.dtype())?
            .unsqueeze(0)?;
        let mut segments = Vec::with_capacity(audio_positions.len() * 2 + 1);
        let mut cursor = 0usize;
        for (frame_idx, position) in audio_positions.into_iter().enumerate() {
            if position > cursor {
                segments.push(text_embeds.narrow(1, cursor, position - cursor)?);
            }
            segments.push(audio_embeddings.narrow(1, frame_idx, 1)?);
            cursor = position + 1;
        }
        if cursor < text_embeds.dim(1)? {
            segments.push(text_embeds.narrow(1, cursor, text_embeds.dim(1)? - cursor)?);
        }
        let segment_refs = segments.iter().collect::<Vec<_>>();
        Tensor::cat(&segment_refs, 1)
    }

    pub fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        let mut layers = Vec::new();
        for layer in &mut self.layers {
            layers.extend(layer.get_isq_layers());
        }
        layers
    }
}
