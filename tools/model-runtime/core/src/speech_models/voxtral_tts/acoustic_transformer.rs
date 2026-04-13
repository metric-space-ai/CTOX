use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use engine_quant::{
    linear_b as quant_linear_b, linear_no_bias as quant_linear_no_bias, QuantMethod,
    QuantizedConfig, ShardedVarBuilder,
};
use std::sync::Arc;

use crate::layers::{repeat_kv, Activation, RmsNorm};

use super::config::VoxtralTtsAudioModelArgs;

fn no_quant_config() -> &'static Option<QuantizedConfig> {
    static NO_QUANT_CONFIG: Option<QuantizedConfig> = None;
    &NO_QUANT_CONFIG
}

fn qforward_autocast(xs: &Tensor, layer: &dyn QuantMethod) -> candle_core::Result<Tensor> {
    let original_dtype = xs.dtype();
    let mut xs = xs.clone();
    if let Some(t) = layer.quantized_act_type() {
        xs = xs.to_dtype(t)?;
    }
    let mut ys = layer.forward(&xs)?;
    if layer.quantized_act_type().is_some() {
        ys = ys.to_dtype(original_dtype)?;
    }
    Ok(ys)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioSpecialToken {
    EmptyAudio = 0,
    EndAudio = 1,
}

impl AudioSpecialToken {
    pub const COUNT: u32 = 2;

    pub const fn id(self) -> u32 {
        self as u32
    }
}

#[derive(Debug, Clone)]
struct FeedForward {
    w1: Arc<dyn QuantMethod>,
    w2: Arc<dyn QuantMethod>,
    w3: Arc<dyn QuantMethod>,
}

impl FeedForward {
    fn new(dim: usize, hidden_dim: usize, use_biases: bool, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            w1: quant_linear_no_bias(dim, hidden_dim, no_quant_config(), vb.pp("w1"))?,
            w2: quant_linear_b(hidden_dim, dim, use_biases, no_quant_config(), vb.pp("w2"))?,
            w3: quant_linear_no_bias(dim, hidden_dim, no_quant_config(), vb.pp("w3"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let gate = Activation::Silu.forward(&qforward_autocast(xs, &*self.w1)?)?;
        let value = qforward_autocast(xs, &*self.w3)?;
        qforward_autocast(&gate.broadcast_mul(&value)?, &*self.w2)
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        vec![&mut self.w1, &mut self.w2, &mut self.w3]
    }
}

#[derive(Debug, Clone)]
struct BidirectionalAttention {
    wq: Arc<dyn QuantMethod>,
    wk: Arc<dyn QuantMethod>,
    wv: Arc<dyn QuantMethod>,
    wo: Arc<dyn QuantMethod>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl BidirectionalAttention {
    fn new(args: &VoxtralTtsAudioModelArgs, vb: ShardedVarBuilder) -> Result<Self> {
        let attn = &args.acoustic_transformer_args;
        Ok(Self {
            wq: quant_linear_b(
                attn.dim,
                attn.n_heads * attn.head_dim,
                attn.use_biases,
                no_quant_config(),
                vb.pp("wq"),
            )?,
            wk: quant_linear_b(
                attn.dim,
                attn.n_kv_heads * attn.head_dim,
                false,
                no_quant_config(),
                vb.pp("wk"),
            )?,
            wv: quant_linear_b(
                attn.dim,
                attn.n_kv_heads * attn.head_dim,
                attn.use_biases,
                no_quant_config(),
                vb.pp("wv"),
            )?,
            wo: quant_linear_b(
                attn.n_heads * attn.head_dim,
                attn.dim,
                attn.use_biases,
                no_quant_config(),
                vb.pp("wo"),
            )?,
            n_heads: attn.n_heads,
            n_kv_heads: attn.n_kv_heads,
            head_dim: attn.head_dim,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, seq_len, _) = xs.dims3()?;
        let q = qforward_autocast(xs, &*self.wq)?.reshape((
            batch,
            seq_len,
            self.n_heads,
            self.head_dim,
        ))?;
        let k = qforward_autocast(xs, &*self.wk)?.reshape((
            batch,
            seq_len,
            self.n_kv_heads,
            self.head_dim,
        ))?;
        let v = qforward_autocast(xs, &*self.wv)?.reshape((
            batch,
            seq_len,
            self.n_kv_heads,
            self.head_dim,
        ))?;
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = repeat_kv(
            k.transpose(1, 2)?.contiguous()?,
            self.n_heads / self.n_kv_heads,
        )?;
        let v = repeat_kv(
            v.transpose(1, 2)?.contiguous()?,
            self.n_heads / self.n_kv_heads,
        )?;
        let attn = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? / (self.head_dim as f64).sqrt())?;
        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let ys = attn.matmul(&v)?.transpose(1, 2)?.reshape((
            batch,
            seq_len,
            self.n_heads * self.head_dim,
        ))?;
        qforward_autocast(&ys, &*self.wo)
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        vec![&mut self.wq, &mut self.wk, &mut self.wv, &mut self.wo]
    }
}

#[derive(Debug, Clone)]
struct AcousticTransformerBlock {
    attention: BidirectionalAttention,
    feed_forward: FeedForward,
    attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
}

impl AcousticTransformerBlock {
    fn new(args: &VoxtralTtsAudioModelArgs, vb: ShardedVarBuilder) -> Result<Self> {
        let attn = &args.acoustic_transformer_args;
        Ok(Self {
            attention: BidirectionalAttention::new(args, vb.pp("attention"))?,
            feed_forward: FeedForward::new(
                attn.dim,
                attn.hidden_dim,
                attn.use_biases,
                vb.pp("feed_forward"),
            )?,
            attention_norm: RmsNorm::new(attn.dim, attn.sigma, vb.pp("attention_norm"))?,
            ffn_norm: RmsNorm::new(attn.dim, attn.sigma, vb.pp("ffn_norm"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let residual = xs;
        let xs = self.attention.forward(&self.attention_norm.forward(xs)?)?;
        let hidden = (residual + xs)?;
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
struct TimeEmbedding {
    inv_freq: Tensor,
}

impl TimeEmbedding {
    fn new(dim: usize, theta: f64, device: &Device) -> candle_core::Result<Self> {
        let half_dim = dim / 2;
        let values = (0..half_dim)
            .map(|i| (-(theta.ln()) * i as f64 / half_dim as f64).exp() as f32)
            .collect::<Vec<_>>();
        Ok(Self {
            inv_freq: Tensor::from_vec(values, half_dim, device)?,
        })
    }

    fn forward(&self, t: &Tensor) -> candle_core::Result<Tensor> {
        let inv_freq = self.inv_freq.to_device(t.device())?.to_dtype(t.dtype())?;
        let args = t.matmul(&inv_freq.unsqueeze(0)?)?;
        Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)
    }
}

#[derive(Debug, Clone)]
pub struct VoxtralTtsAcousticTransformer {
    semantic_codebook_size: usize,
    acoustic_embeddings_levels: usize,
    n_acoustic_codebook: usize,
    semantic_codebook_output: Arc<dyn QuantMethod>,
    acoustic_codebook_output: Arc<dyn QuantMethod>,
    input_projection: Arc<dyn QuantMethod>,
    time_projection: Arc<dyn QuantMethod>,
    llm_projection: Arc<dyn QuantMethod>,
    time_embedding: TimeEmbedding,
    layers: Vec<AcousticTransformerBlock>,
    norm: RmsNorm,
    timesteps: Vec<f32>,
    cfg_alpha: f64,
    noise_scale: f64,
}

impl VoxtralTtsAcousticTransformer {
    pub fn new(
        args: &VoxtralTtsAudioModelArgs,
        vb: ShardedVarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let attn = &args.acoustic_transformer_args;
        let timesteps = (0..8).map(|idx| idx as f32 / 7f32).collect::<Vec<_>>();
        Ok(Self {
            semantic_codebook_size: args.semantic_codebook_size,
            acoustic_embeddings_levels: args.acoustic_codebook_size,
            n_acoustic_codebook: args.n_acoustic_codebook,
            semantic_codebook_output: quant_linear_b(
                attn.dim,
                round_up_to_multiple(
                    args.semantic_codebook_size + AudioSpecialToken::COUNT as usize,
                    128,
                ),
                attn.use_biases,
                no_quant_config(),
                vb.pp("semantic_codebook_output"),
            )?,
            acoustic_codebook_output: quant_linear_no_bias(
                attn.dim,
                args.n_acoustic_codebook,
                no_quant_config(),
                vb.pp("acoustic_codebook_output"),
            )?,
            input_projection: quant_linear_no_bias(
                args.n_acoustic_codebook,
                attn.dim,
                no_quant_config(),
                vb.pp("input_projection"),
            )?,
            time_projection: quant_linear_no_bias(
                attn.dim,
                attn.dim,
                no_quant_config(),
                vb.pp("time_projection"),
            )?,
            llm_projection: quant_linear_no_bias(
                attn.input_dim,
                attn.dim,
                no_quant_config(),
                vb.pp("llm_projection"),
            )?,
            time_embedding: TimeEmbedding::new(attn.dim, attn.rope_theta, device)?,
            layers: (0..attn.n_layers)
                .map(|idx| AcousticTransformerBlock::new(args, vb.pp("layers").pp(idx)))
                .collect::<Result<Vec<_>>>()?,
            norm: RmsNorm::new(attn.dim, attn.sigma, vb.pp("norm"))?,
            timesteps,
            cfg_alpha: 1.2,
            noise_scale: 1.0,
        })
    }

    pub fn forward(&self, llm_hidden: &Tensor) -> candle_core::Result<(Tensor, bool)> {
        let semantic_logits = qforward_autocast(llm_hidden, &*self.semantic_codebook_output)?;
        let semantic_code = self.select_semantic_code(&semantic_logits)?;
        let is_end = semantic_code.first().copied() == Some(AudioSpecialToken::EndAudio.id());
        let acoustic_codes = self.decode_one_frame(&semantic_code, llm_hidden)?;
        let semantic =
            Tensor::from_vec(semantic_code, (llm_hidden.dim(0)?, 1), llm_hidden.device())?;
        Ok((Tensor::cat(&[semantic, acoustic_codes], 1)?, is_end))
    }

    fn select_semantic_code(&self, logits: &Tensor) -> candle_core::Result<Vec<u32>> {
        let shape = logits.shape();
        let logits = logits.to_dtype(DType::F32)?;
        let mut rows = logits.to_vec2::<f32>()?;
        let cutoff = AudioSpecialToken::COUNT as usize + self.semantic_codebook_size;
        for row in &mut rows {
            if !row.is_empty() {
                row[AudioSpecialToken::EmptyAudio.id() as usize] = f32::NEG_INFINITY;
            }
            for value in row.iter_mut().skip(cutoff) {
                *value = f32::NEG_INFINITY;
            }
        }
        let flat = rows.into_iter().flatten().collect::<Vec<_>>();
        let masked = Tensor::from_vec(flat, shape, logits.device())?;
        masked
            .argmax_keepdim(D::Minus1)?
            .to_vec2::<u32>()
            .map(|v| v.into_iter().map(|mut row| row.remove(0)).collect())
    }

    fn decode_one_frame(
        &self,
        semantic_code: &[u32],
        llm_hidden: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let batch = llm_hidden.dim(0)?;
        let mut sampled = Tensor::randn(
            0f32,
            self.noise_scale as f32,
            (batch, self.n_acoustic_codebook),
            llm_hidden.device(),
        )?
        .to_dtype(llm_hidden.dtype())?;
        let llm_hidden_zero = Tensor::zeros_like(llm_hidden)?;
        for window in self.timesteps.windows(2) {
            let t = window[0];
            let dt = window[1] - window[0];
            let t_tensor =
                Tensor::full(t, (batch, 1), llm_hidden.device())?.to_dtype(llm_hidden.dtype())?;
            let t_emb = self
                .time_embedding
                .forward(&t_tensor)?
                .to_dtype(llm_hidden.dtype())?;
            let x_batched = Tensor::cat(&[sampled.clone(), sampled.clone()], 0)?;
            let llm_batched = Tensor::cat(&[llm_hidden.clone(), llm_hidden_zero.clone()], 0)?;
            let t_emb_batched = Tensor::cat(&[t_emb.clone(), t_emb], 0)?;
            let velocity = self.predict_velocity(&x_batched, &llm_batched, &t_emb_batched)?;
            let cond = velocity.narrow(0, 0, batch)?;
            let uncond = velocity.narrow(0, batch, batch)?;
            let cond_scale = Tensor::full(self.cfg_alpha as f32, cond.shape(), cond.device())?
                .to_dtype(cond.dtype())?;
            let uncond_scale = Tensor::full(
                (1.0 - self.cfg_alpha) as f32,
                uncond.shape(),
                uncond.device(),
            )?
            .to_dtype(uncond.dtype())?;
            let guided =
                (cond.broadcast_mul(&cond_scale)? + uncond.broadcast_mul(&uncond_scale)?)?;
            let dt_scale =
                Tensor::full(dt, guided.shape(), guided.device())?.to_dtype(guided.dtype())?;
            sampled = (sampled + guided.broadcast_mul(&dt_scale)?)?;
        }
        let sampled = sampled.clamp(-1f64, 1f64)?;
        let scaled = ((sampled + 1.0)? / 2.0)?;
        let level_scale = Tensor::full(
            (self.acoustic_embeddings_levels.saturating_sub(1)) as f32,
            scaled.shape(),
            scaled.device(),
        )?
        .to_dtype(scaled.dtype())?;
        let scaled = scaled.broadcast_mul(&level_scale)?;
        let rounded = scaled.round()?.to_dtype(DType::U32)?;
        let specials_offset = AudioSpecialToken::COUNT;
        let mut rows = rounded.to_vec2::<u32>()?;
        for (row, semantic) in rows.iter_mut().zip(semantic_code) {
            if *semantic == AudioSpecialToken::EndAudio.id() {
                for token in row.iter_mut() {
                    *token = AudioSpecialToken::EmptyAudio.id();
                }
            } else {
                for token in row.iter_mut() {
                    *token = token.saturating_add(specials_offset);
                }
            }
        }
        Tensor::from_vec(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            (batch, self.n_acoustic_codebook),
            llm_hidden.device(),
        )
    }

    fn predict_velocity(
        &self,
        x_t: &Tensor,
        llm_output: &Tensor,
        t_emb: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let x_t = x_t.to_dtype(llm_output.dtype())?;
        let t_emb = qforward_autocast(t_emb, &*self.time_projection)?;
        let llm_output = qforward_autocast(llm_output, &*self.llm_projection)?;
        let acoustic = qforward_autocast(&x_t, &*self.input_projection)?.unsqueeze(1)?;
        let t_emb = t_emb.unsqueeze(1)?;
        let llm_output = llm_output.unsqueeze(1)?;
        let mut hidden = Tensor::cat(&[acoustic, t_emb, llm_output], 1)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        let hidden = self.norm.forward(&hidden)?;
        qforward_autocast(&hidden.i((.., 0, ..))?, &*self.acoustic_codebook_output)
    }

    pub fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        let mut layers = vec![
            &mut self.semantic_codebook_output,
            &mut self.acoustic_codebook_output,
            &mut self.input_projection,
            &mut self.time_projection,
            &mut self.llm_projection,
        ];
        for layer in &mut self.layers {
            layers.extend(layer.get_isq_layers());
        }
        layers
    }
}

fn round_up_to_multiple(value: usize, multiple: usize) -> usize {
    multiple * value.div_ceil(multiple)
}
