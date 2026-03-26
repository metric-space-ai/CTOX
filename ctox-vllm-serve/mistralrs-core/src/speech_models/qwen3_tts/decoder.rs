#![allow(dead_code)]

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Embedding, LayerNorm};
use mistralrs_quant::{
    linear as quant_linear, linear_b as quant_linear_b, linear_no_bias as quant_linear_no_bias,
    Convolution, QuantMethod, QuantizedConfig, ShardedVarBuilder,
};
use std::sync::Arc;

use crate::{
    layers::{self, Activation, RmsNorm, RotaryEmbedding, conv1d, layer_norm, repeat_kv},
    ops::apply_triangular,
};

use super::Qwen3TtsTokenizerDecoderConfig;

fn hidden_act(name: &str) -> Result<Activation> {
    match name.trim().to_ascii_lowercase().as_str() {
        "gelu" => Ok(Activation::Gelu),
        "gelu_new" | "gelu_pytorch_tanh" => Ok(Activation::NewGelu),
        "relu" => Ok(Activation::Relu),
        "silu" | "swish" => Ok(Activation::Silu),
        other => anyhow::bail!("Unsupported Qwen3-TTS decoder activation `{other}`."),
    }
}

fn conv_transpose1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    bias: bool,
    cfg: ConvTranspose1dConfig,
    vb: ShardedVarBuilder,
) -> candle_core::Result<ConvTranspose1d> {
    let ws = vb.get((in_channels, out_channels, kernel_size), "weight")?;
    let bs = if bias { Some(vb.get(out_channels, "bias")?) } else { None };
    Ok(ConvTranspose1d::new(ws, bs, cfg))
}

fn no_quant_config() -> &'static Option<QuantizedConfig> {
    static NO_QUANT_CONFIG: Option<QuantizedConfig> = None;
    &NO_QUANT_CONFIG
}

fn ensure_codes_in_range(codes: &Tensor, codebook_size: usize, label: &str) -> candle_core::Result<()> {
    let flat = codes.flatten_all()?.to_dtype(DType::U32)?.to_device(&Device::Cpu)?;
    let flat = flat.to_vec1::<u32>()?;
    if let Some((position, code)) = flat
        .iter()
        .copied()
        .enumerate()
        .find(|(_, code)| *code as usize >= codebook_size)
    {
        candle_core::bail!(
            "Qwen3-TTS {label} received out-of-range code {code} at position {position} (codebook_size={codebook_size}, len={}, max_code={}).",
            flat.len(),
            flat.iter().copied().max().unwrap_or(0)
        );
    }
    Ok(())
}

fn causal_attention_mask(seq_len: usize, device: &Device) -> candle_core::Result<Tensor> {
    let mask = apply_triangular(
        &Tensor::ones((seq_len, seq_len), DType::F32, device)?,
        0,
        false,
    )?;
    let neg_inf = Tensor::full(f32::NEG_INFINITY, (seq_len, seq_len), device)?;
    let zeros = Tensor::zeros((seq_len, seq_len), DType::F32, device)?;
    mask.to_dtype(DType::U8)?
        .where_cond(&zeros, &neg_inf)?
        .unsqueeze(0)?
        .unsqueeze(0)
}

#[derive(Debug, Clone)]
struct SnakeBeta {
    alpha: Tensor,
    beta: Tensor,
}

impl SnakeBeta {
    fn new(channels: usize, vb: ShardedVarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            alpha: vb.get(channels, "alpha")?,
            beta: vb.get(channels, "beta")?,
        })
    }
}

impl Module for SnakeBeta {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let alpha = self.alpha.exp()?.unsqueeze(0)?.unsqueeze(2)?;
        let beta = self.beta.exp()?.unsqueeze(0)?.unsqueeze(2)?;
        let periodic = xs.broadcast_mul(&alpha)?.sin()?.sqr()?;
        xs + (&beta + 1e-9)?.recip()?.broadcast_mul(&periodic)?
    }
}

#[derive(Debug, Clone)]
struct CausalConv1d {
    conv: Conv1d,
    stride: usize,
    effective_kernel: usize,
}

impl CausalConv1d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        stride: usize,
        groups: usize,
        vb: ShardedVarBuilder,
    ) -> candle_core::Result<Self> {
        let effective_kernel = (kernel_size - 1) * dilation + 1;
        let cfg = Conv1dConfig {
            padding: 0,
            stride,
            dilation,
            groups,
            ..Default::default()
        };
        Ok(Self {
            conv: conv1d(in_channels, out_channels, kernel_size, cfg, vb.pp("conv"))?,
            stride,
            effective_kernel,
        })
    }

    fn extra_padding(&self, len: usize) -> usize {
        let padding = self.effective_kernel.saturating_sub(self.stride);
        let n_frames =
            (len as f64 - self.effective_kernel as f64 + padding as f64) / self.stride as f64 + 1.0;
        let ideal_length =
            ((n_frames.ceil() - 1.0) * self.stride as f64 + (self.effective_kernel - padding) as f64)
                as usize;
        ideal_length.saturating_sub(len)
    }
}

impl Module for CausalConv1d {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let padding = self.effective_kernel.saturating_sub(self.stride);
        let padded = xs.pad_with_zeros(D::Minus1, padding, self.extra_padding(xs.dim(D::Minus1)?))?;
        Convolution.forward_1d(&self.conv, &padded)
    }
}

#[derive(Debug, Clone)]
struct CausalTransposedConv1d {
    conv: ConvTranspose1d,
    right_pad: usize,
}

impl CausalTransposedConv1d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        vb: ShardedVarBuilder,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            conv: conv_transpose1d(
                in_channels,
                out_channels,
                kernel_size,
                true,
                ConvTranspose1dConfig {
                    padding: 0,
                    stride,
                    ..Default::default()
                },
                vb.pp("conv"),
            )?,
            right_pad: kernel_size.saturating_sub(stride),
        })
    }
}

impl Module for CausalTransposedConv1d {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let ys = xs.apply(&self.conv)?;
        if self.right_pad > 0 {
            ys.narrow(D::Minus1, 0, ys.dim(D::Minus1)?.saturating_sub(self.right_pad))
        } else {
            Ok(ys)
        }
    }
}

#[derive(Debug, Clone)]
struct ConvNeXtBlock {
    dwconv: CausalConv1d,
    norm: LayerNorm,
    pwconv1: Arc<dyn QuantMethod>,
    pwconv2: Arc<dyn QuantMethod>,
    gamma: Tensor,
    act: Activation,
}

impl ConvNeXtBlock {
    fn new(dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            dwconv: CausalConv1d::new(dim, dim, 7, 1, 1, dim, vb.pp("dwconv"))?,
            norm: layer_norm(dim, 1e-6, vb.pp("norm"))?,
            pwconv1: quant_linear(dim, 4 * dim, no_quant_config(), vb.pp("pwconv1"))?,
            pwconv2: quant_linear(4 * dim, dim, no_quant_config(), vb.pp("pwconv2"))?,
            gamma: vb.get(dim, "gamma")?,
            act: Activation::Gelu,
        })
    }
}

impl Module for ConvNeXtBlock {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let residual = xs;
        let xs = self.dwconv.forward(xs)?.transpose(1, 2)?;
        let xs = self.norm.forward(&xs)?;
        let xs = self.pwconv2.forward(&self.act.forward(&self.pwconv1.forward(&xs)?)?)?;
        let xs = self.gamma.unsqueeze(0)?.broadcast_mul(&xs)?.transpose(1, 2)?;
        residual + xs
    }
}

#[derive(Debug, Clone)]
struct DecoderSelfAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl DecoderSelfAttention {
    fn new(cfg: &Qwen3TtsTokenizerDecoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        Ok(Self {
            q_proj: quant_linear_b(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim,
                cfg.attention_bias,
                no_quant_config(),
                vb.pp("q_proj"),
            )?,
            k_proj: quant_linear_b(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                cfg.attention_bias,
                no_quant_config(),
                vb.pp("k_proj"),
            )?,
            v_proj: quant_linear_b(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                cfg.attention_bias,
                no_quant_config(),
                vb.pp("v_proj"),
            )?,
            o_proj: quant_linear_b(
                cfg.num_attention_heads * cfg.head_dim,
                cfg.hidden_size,
                cfg.attention_bias,
                no_quant_config(),
                vb.pp("o_proj"),
            )?,
            rotary_emb: RotaryEmbedding::new(
                cfg.rope_theta as f32,
                cfg.head_dim,
                cfg.max_position_embeddings,
                &device,
                false,
                dtype,
            )?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> candle_core::Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let q = self
            .q_proj
            .forward(hidden_states)?
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(hidden_states)?
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(hidden_states)?
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (q, k) = self.rotary_emb.forward(&q, &k, &vec![0; batch_size])?;
        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let q = q.contiguous()?;
        let v = v.contiguous()?;
        let attn = (q.matmul(&k_t)? / (self.head_dim as f64).sqrt())?
            .broadcast_add(attention_mask)?;
        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?.contiguous()?;
        let attn = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&attn)
    }
}

#[derive(Debug, Clone)]
struct DecoderMlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act: Activation,
}

impl DecoderMlp {
    fn new(cfg: &Qwen3TtsTokenizerDecoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
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
struct LayerScale {
    scale: Tensor,
}

impl LayerScale {
    fn new(dim: usize, vb: ShardedVarBuilder) -> candle_core::Result<Self> {
        Ok(Self { scale: vb.get(dim, "scale")? })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.scale.unsqueeze(0)?.broadcast_mul(xs)
    }
}

#[derive(Debug, Clone)]
struct DecoderTransformerLayer {
    self_attn: DecoderSelfAttention,
    mlp: DecoderMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    self_attn_layer_scale: LayerScale,
    mlp_layer_scale: LayerScale,
}

impl DecoderTransformerLayer {
    fn new(cfg: &Qwen3TtsTokenizerDecoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: DecoderSelfAttention::new(cfg, vb.pp("self_attn"))?,
            mlp: DecoderMlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            self_attn_layer_scale: LayerScale::new(cfg.hidden_size, vb.pp("self_attn_layer_scale"))?,
            mlp_layer_scale: LayerScale::new(cfg.hidden_size, vb.pp("mlp_layer_scale"))?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> candle_core::Result<Tensor> {
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward(&hidden_states, attention_mask)?;
        let hidden_states = (residual + self.self_attn_layer_scale.forward(&hidden_states)?)?;
        let residual = &hidden_states;
        let hidden_states = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&hidden_states)?)?;
        Ok((residual + self.mlp_layer_scale.forward(&hidden_states)?)?)
    }
}

#[derive(Debug, Clone)]
struct DecoderTransformer {
    input_proj: Arc<dyn QuantMethod>,
    layers: Vec<DecoderTransformerLayer>,
    norm: RmsNorm,
    output_proj: Arc<dyn QuantMethod>,
    device: Device,
}

impl DecoderTransformer {
    fn new(cfg: &Qwen3TtsTokenizerDecoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            input_proj: quant_linear(
                cfg.latent_dim,
                cfg.hidden_size,
                no_quant_config(),
                vb.pp("input_proj"),
            )?,
            layers: (0..cfg.num_hidden_layers)
                .map(|idx| DecoderTransformerLayer::new(cfg, vb.pp("layers").pp(idx)))
                .collect::<Result<Vec<_>>>()?,
            norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?,
            output_proj: quant_linear(
                cfg.hidden_size,
                cfg.latent_dim,
                no_quant_config(),
                vb.pp("output_proj"),
            )?,
            device: vb.device().clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        let mut hidden_states = self.input_proj.forward(hidden_states)?;
        let attention_mask = causal_attention_mask(hidden_states.dim(1)?, &self.device)?
            .to_dtype(hidden_states.dtype())?;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &attention_mask)?;
        }
        let hidden_states = self.norm.forward(&hidden_states)?;
        self.output_proj.forward(&hidden_states)
    }
}

#[derive(Debug, Clone)]
struct EuclideanCodebook {
    cluster_usage: Tensor,
    embedding_sum: Tensor,
    epsilon: f64,
}

impl EuclideanCodebook {
    fn new(dim: usize, codebook_size: usize, epsilon: f64, vb: ShardedVarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            cluster_usage: vb.get(codebook_size, "cluster_usage")?,
            embedding_sum: vb.get((codebook_size, dim), "embedding_sum")?,
            epsilon,
        })
    }

    fn decode(&self, codes: &Tensor) -> candle_core::Result<Tensor> {
        ensure_codes_in_range(codes, self.embedding_sum.dim(0)?, "decoder codebook")?;
        let codes = codes
            .to_device(self.embedding_sum.device())?
            .to_dtype(DType::U32)?;
        let usage = self.cluster_usage.clamp(self.epsilon, f64::INFINITY)?.unsqueeze(1)?;
        let embedding = self.embedding_sum.broadcast_div(&usage)?;
        Embedding::new(embedding, self.embedding_sum.dim(1)?).forward(&codes)
    }
}

#[derive(Debug, Clone)]
struct VectorQuantization {
    project_out: Option<Arc<dyn QuantMethod>>,
    codebook: EuclideanCodebook,
}

impl VectorQuantization {
    fn new(
        dim: usize,
        codebook_size: usize,
        codebook_dim: usize,
        epsilon: f64,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            project_out: if codebook_dim != dim {
                Some(quant_linear(
                    codebook_dim,
                    dim,
                    no_quant_config(),
                    vb.pp("project_out"),
                )?)
            } else {
                None
            },
            codebook: EuclideanCodebook::new(codebook_dim, codebook_size, epsilon, vb.pp("_codebook"))?,
        })
    }

    fn decode(&self, codes: &Tensor) -> candle_core::Result<Tensor> {
        let mut quantized = self.codebook.decode(codes)?;
        if let Some(project_out) = &self.project_out {
            quantized = project_out.forward(&quantized)?;
        }
        quantized.transpose(1, 2)
    }
}

#[derive(Debug, Clone)]
struct ResidualVectorQuantization {
    layers: Vec<VectorQuantization>,
}

impl ResidualVectorQuantization {
    fn new(
        num_quantizers: usize,
        dim: usize,
        codebook_size: usize,
        codebook_dim: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            layers: (0..num_quantizers)
                .map(|idx| VectorQuantization::new(dim, codebook_size, codebook_dim, 1e-5, vb.pp("layers").pp(idx)))
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn decode(&self, codes: &Tensor) -> candle_core::Result<Tensor> {
        let mut quantized = None;
        for (idx, layer) in self.layers.iter().enumerate() {
            let layer_codes = codes.i((idx, .., ..))?;
            let decoded = layer.decode(&layer_codes)?;
            quantized = Some(match quantized {
                None => decoded,
                Some(q) => (q + decoded)?,
            });
        }
        quantized.ok_or_else(|| candle_core::Error::Msg("empty quantizer stack".into()))
    }
}

#[derive(Debug, Clone)]
struct ResidualVectorQuantizer {
    output_proj: Conv1d,
    vq: ResidualVectorQuantization,
}

impl ResidualVectorQuantizer {
    fn new(
        dimension: usize,
        output_dimension: usize,
        n_q: usize,
        bins: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            output_proj: conv1d_no_bias_wrapper(dimension, output_dimension, vb.pp("output_proj"))?,
            vq: ResidualVectorQuantization::new(n_q, dimension, bins, dimension, vb.pp("vq"))?,
        })
    }

    fn decode(&self, codes: &Tensor) -> candle_core::Result<Tensor> {
        let codes = codes.transpose(0, 1)?;
        let quantized = self.vq.decode(&codes)?;
        Convolution.forward_1d(&self.output_proj, &quantized)
    }
}

fn conv1d_no_bias_wrapper(
    in_channels: usize,
    out_channels: usize,
    vb: ShardedVarBuilder,
) -> candle_core::Result<Conv1d> {
    layers::conv1d_no_bias(
        in_channels,
        out_channels,
        1,
        Conv1dConfig {
            padding: 0,
            ..Default::default()
        },
        vb,
    )
}

#[derive(Debug, Clone)]
struct SplitResidualVectorQuantizer {
    n_q_semantic: usize,
    rvq_first: ResidualVectorQuantizer,
    rvq_rest: ResidualVectorQuantizer,
}

impl SplitResidualVectorQuantizer {
    fn new(cfg: &Qwen3TtsTokenizerDecoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            n_q_semantic: 1,
            rvq_first: ResidualVectorQuantizer::new(
                cfg.codebook_dim / 2,
                cfg.codebook_dim,
                1,
                cfg.codebook_size,
                vb.pp("rvq_first"),
            )?,
            rvq_rest: ResidualVectorQuantizer::new(
                cfg.codebook_dim / 2,
                cfg.codebook_dim,
                cfg.num_quantizers - 1,
                cfg.codebook_size,
                vb.pp("rvq_rest"),
            )?,
        })
    }

    fn decode(&self, codes: &Tensor) -> candle_core::Result<Tensor> {
        let mut quantized = self.rvq_first.decode(&codes.i((.., ..self.n_q_semantic, ..))?)?;
        if codes.dim(1)? > self.n_q_semantic {
            quantized = (quantized + self.rvq_rest.decode(&codes.i((.., self.n_q_semantic.., ..))?)?)?;
        }
        Ok(quantized)
    }
}

#[derive(Debug, Clone)]
struct DecoderResidualUnit {
    act1: SnakeBeta,
    conv1: CausalConv1d,
    act2: SnakeBeta,
    conv2: CausalConv1d,
}

impl DecoderResidualUnit {
    fn new(dim: usize, dilation: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            act1: SnakeBeta::new(dim, vb.pp("act1"))?,
            conv1: CausalConv1d::new(dim, dim, 7, dilation, 1, 1, vb.pp("conv1"))?,
            act2: SnakeBeta::new(dim, vb.pp("act2"))?,
            conv2: CausalConv1d::new(dim, dim, 1, 1, 1, 1, vb.pp("conv2"))?,
        })
    }
}

impl Module for DecoderResidualUnit {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let residual = xs;
        let xs = self.act1.forward(xs)?;
        let xs = self.conv1.forward(&xs)?;
        let xs = self.act2.forward(&xs)?;
        let xs = self.conv2.forward(&xs)?;
        residual + xs
    }
}

#[derive(Debug, Clone)]
struct DecoderBlock {
    snake: SnakeBeta,
    transposed: CausalTransposedConv1d,
    residuals: Vec<DecoderResidualUnit>,
}

impl DecoderBlock {
    fn new(cfg: &Qwen3TtsTokenizerDecoderConfig, layer_idx: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let in_dim = cfg.decoder_dim / 2usize.pow(layer_idx as u32);
        let out_dim = cfg.decoder_dim / 2usize.pow((layer_idx + 1) as u32);
        let stride = cfg.upsample_rates[layer_idx];
        Ok(Self {
            snake: SnakeBeta::new(in_dim, vb.pp("block").pp(0))?,
            transposed: CausalTransposedConv1d::new(
                in_dim,
                out_dim,
                2 * stride,
                stride,
                vb.pp("block").pp(1),
            )?,
            residuals: [(1usize, 2usize), (3, 3), (9, 4)]
                .into_iter()
                .map(|(dilation, slot)| DecoderResidualUnit::new(out_dim, dilation, vb.pp("block").pp(slot)))
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

impl Module for DecoderBlock {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut xs = self.snake.forward(xs)?;
        xs = self.transposed.forward(&xs)?;
        for residual in &self.residuals {
            xs = residual.forward(&xs)?;
        }
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3TtsTokenizerDecoder {
    cfg: Qwen3TtsTokenizerDecoderConfig,
    total_upsample: usize,
    device: Device,
    quantizer: SplitResidualVectorQuantizer,
    pre_conv: CausalConv1d,
    pre_transformer: DecoderTransformer,
    upsample: Vec<(CausalTransposedConv1d, ConvNeXtBlock)>,
    decoder_pre: CausalConv1d,
    decoder_blocks: Vec<DecoderBlock>,
    decoder_post_snake: SnakeBeta,
    decoder_post: CausalConv1d,
}

impl Qwen3TtsTokenizerDecoder {
    pub fn new(cfg: &Qwen3TtsTokenizerDecoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let total_upsample = cfg
            .upsample_rates
            .iter()
            .chain(cfg.upsampling_ratios.iter())
            .product::<usize>();
        Ok(Self {
            cfg: cfg.clone(),
            total_upsample,
            device: vb.device().clone(),
            quantizer: SplitResidualVectorQuantizer::new(cfg, vb.pp("quantizer"))?,
            pre_conv: CausalConv1d::new(
                cfg.codebook_dim,
                cfg.latent_dim,
                3,
                1,
                1,
                1,
                vb.pp("pre_conv"),
            )?,
            pre_transformer: DecoderTransformer::new(cfg, vb.pp("pre_transformer"))?,
            upsample: cfg
                .upsampling_ratios
                .iter()
                .enumerate()
                .map(|(idx, factor)| {
                    Ok((
                        CausalTransposedConv1d::new(
                            cfg.latent_dim,
                            cfg.latent_dim,
                            *factor,
                            *factor,
                            vb.pp("upsample").pp(idx).pp(0),
                        )?,
                        ConvNeXtBlock::new(cfg.latent_dim, vb.pp("upsample").pp(idx).pp(1))?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?,
            decoder_pre: CausalConv1d::new(
                cfg.latent_dim,
                cfg.decoder_dim,
                7,
                1,
                1,
                1,
                vb.pp("decoder").pp(0),
            )?,
            decoder_blocks: (0..cfg.upsample_rates.len())
                .map(|idx| DecoderBlock::new(cfg, idx, vb.pp("decoder").pp(idx + 1)))
                .collect::<Result<Vec<_>>>()?,
            decoder_post_snake: SnakeBeta::new(
                cfg.decoder_dim / 2usize.pow(cfg.upsample_rates.len() as u32),
                vb.pp("decoder").pp(cfg.upsample_rates.len() + 1),
            )?,
            decoder_post: CausalConv1d::new(
                cfg.decoder_dim / 2usize.pow(cfg.upsample_rates.len() as u32),
                1,
                7,
                1,
                1,
                1,
                vb.pp("decoder").pp(cfg.upsample_rates.len() + 2),
            )?,
        })
    }

    fn forward_chunk(&self, codes: &Tensor) -> candle_core::Result<Tensor> {
        if codes.dim(1)? != self.cfg.num_quantizers {
            candle_core::bail!(
                "Expected {} quantizer groups, got {}.",
                self.cfg.num_quantizers,
                codes.dim(1)?
            );
        }
        let hidden = self.quantizer.decode(codes)?;
        let hidden = self.pre_conv.forward(&hidden)?.transpose(1, 2)?;
        let mut hidden = self.pre_transformer.forward(&hidden)?.transpose(1, 2)?;
        for (transposed, convnext) in &self.upsample {
            hidden = transposed.forward(&hidden)?;
            hidden = convnext.forward(&hidden)?;
        }
        hidden = self.decoder_pre.forward(&hidden)?;
        for block in &self.decoder_blocks {
            hidden = block.forward(&hidden)?;
        }
        self.decoder_post.forward(&self.decoder_post_snake.forward(&hidden)?)?.clamp(-1.0, 1.0)
    }

    pub fn chunked_decode(
        &self,
        codes: &Tensor,
        chunk_size: usize,
        left_context_size: usize,
    ) -> candle_core::Result<Tensor> {
        let total_len = codes.dim(D::Minus1)?;
        let mut start_index = 0usize;
        let mut wavs = Vec::new();
        while start_index < total_len {
            let end_index = (start_index + chunk_size).min(total_len);
            let context_size = if start_index > left_context_size {
                left_context_size
            } else {
                start_index
            };
            let chunk = codes.narrow(D::Minus1, start_index - context_size, end_index - (start_index - context_size))?;
            let wav = self.forward_chunk(&chunk)?;
            let skip = context_size * self.total_upsample;
            wavs.push(wav.narrow(D::Minus1, skip, wav.dim(D::Minus1)?.saturating_sub(skip))?);
            start_index = end_index;
        }
        let refs = wavs.iter().collect::<Vec<_>>();
        Tensor::cat(&refs, D::Minus1)
    }

    pub fn decode(&self, audio_codes: &Tensor) -> candle_core::Result<Tensor> {
        let audio_codes = audio_codes
            .to_device(&self.device)?
            .to_dtype(DType::U32)?
            .clamp(0u32, u32::MAX)?;
        let audio_codes = audio_codes.transpose(1, 2)?;
        self.chunked_decode(&audio_codes, 300, 25)
    }
}
