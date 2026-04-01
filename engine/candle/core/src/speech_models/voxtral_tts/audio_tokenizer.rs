use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{
    ops::softmax, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Embedding,
};
use mistralrs_quant::{
    linear_b as quant_linear_b, linear_no_bias as quant_linear_no_bias, Convolution, QuantMethod,
    QuantizedConfig, ShardedVarBuilder,
};
use std::sync::Arc;

use crate::layers::{embedding, repeat_kv, Activation, ReflectionPad1d, RmsNorm};

use super::config::VoxtralTtsAudioTokenizerArgs;

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

#[derive(Debug, Clone, Copy)]
enum PadMode {
    Reflect,
    Replicate,
}

fn replication_pad1d(
    xs: &Tensor,
    pad_left: usize,
    pad_right: usize,
) -> candle_core::Result<Tensor> {
    let xs = xs.contiguous()?;
    let (_n, _c, w) = xs.dims3()?;

    let left = if pad_left > 0 {
        let indices = Tensor::new(vec![0i64; pad_left], xs.device())?;
        Some(xs.index_select(&indices, 2)?)
    } else {
        None
    };
    let right = if pad_right > 0 {
        let indices = Tensor::new(vec![(w - 1) as i64; pad_right], xs.device())?;
        Some(xs.index_select(&indices, 2)?)
    } else {
        None
    };

    match (left, right) {
        (Some(l), Some(r)) => Tensor::cat(&[l, xs.clone(), r], 2),
        (Some(l), None) => Tensor::cat(&[l, xs.clone()], 2),
        (None, Some(r)) => Tensor::cat(&[xs.clone(), r], 2),
        (None, None) => Ok(xs.clone()),
    }?
    .contiguous()
}

fn conv1d_weight_norm(
    out_c: usize,
    in_c: usize,
    kernel_size: usize,
    bias: bool,
    config: Conv1dConfig,
    vb: ShardedVarBuilder,
) -> candle_core::Result<Conv1d> {
    let weight_g = vb.get((out_c, 1, 1), "parametrizations.weight.original0")?;
    let weight_v = vb.get(
        (out_c, in_c, kernel_size),
        "parametrizations.weight.original1",
    )?;
    let norm_v = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?;
    let bias = if bias {
        Some(vb.get(out_c, "bias")?)
    } else {
        None
    };
    Ok(Conv1d::new(weight, bias, config))
}

fn conv_transpose1d_weight_norm(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    bias: bool,
    config: ConvTranspose1dConfig,
    vb: ShardedVarBuilder,
) -> candle_core::Result<ConvTranspose1d> {
    let weight_g = vb.get((in_c, 1, 1), "parametrizations.weight.original0")?;
    let weight_v = vb.get(
        (in_c, out_c, kernel_size),
        "parametrizations.weight.original1",
    )?;
    let norm_v = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?;
    let bias = if bias {
        Some(vb.get(out_c, "bias")?)
    } else {
        None
    };
    Ok(ConvTranspose1d::new(weight, bias, config))
}

#[derive(Debug, Clone)]
struct CausalConv1d {
    conv: Conv1d,
    stride: usize,
    effective_kernel: usize,
    pad_mode: PadMode,
}

impl CausalConv1d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        pad_mode: PadMode,
        use_weight_norm: bool,
        use_bias: bool,
        vb: ShardedVarBuilder,
    ) -> candle_core::Result<Self> {
        let cfg = Conv1dConfig {
            stride,
            padding: 0,
            ..Default::default()
        };
        let conv = if use_weight_norm {
            conv1d_weight_norm(
                out_channels,
                in_channels,
                kernel_size,
                use_bias,
                cfg,
                vb.pp("conv"),
            )?
        } else {
            let weight = vb
                .pp("conv")
                .get((out_channels, in_channels, kernel_size), "weight")?;
            let bias = if use_bias {
                Some(vb.pp("conv").get(out_channels, "bias")?)
            } else {
                None
            };
            Conv1d::new(weight, bias, cfg)
        };
        Ok(Self {
            conv,
            stride,
            effective_kernel: kernel_size,
            pad_mode,
        })
    }

    fn extra_padding(&self, len: usize) -> usize {
        let total_padding = self.effective_kernel.saturating_sub(self.stride);
        let n_frames = (len as f64 - self.effective_kernel as f64 + total_padding as f64)
            / self.stride as f64
            + 1.0;
        let target_length = ((n_frames.ceil() - 1.0) * self.stride as f64
            + (self.effective_kernel - total_padding) as f64) as usize;
        target_length.saturating_sub(len)
    }
}

impl Module for CausalConv1d {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let total_padding = self.effective_kernel.saturating_sub(self.stride);
        let padded = match self.pad_mode {
            PadMode::Reflect => {
                ReflectionPad1d::new((total_padding, self.extra_padding(xs.dim(D::Minus1)?)))
                    .forward(xs)?
            }
            PadMode::Replicate => {
                replication_pad1d(xs, total_padding, self.extra_padding(xs.dim(D::Minus1)?))?
            }
        };
        Convolution.forward_1d(&self.conv, &padded)
    }
}

#[derive(Debug, Clone)]
struct CausalConvTranspose1d {
    conv: ConvTranspose1d,
    left_trim: usize,
    right_trim: usize,
}

impl CausalConvTranspose1d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        use_weight_norm: bool,
        use_bias: bool,
        vb: ShardedVarBuilder,
    ) -> candle_core::Result<Self> {
        let conv = if use_weight_norm {
            conv_transpose1d_weight_norm(
                in_channels,
                out_channels,
                kernel_size,
                use_bias,
                ConvTranspose1dConfig {
                    padding: 0,
                    stride,
                    ..Default::default()
                },
                vb.pp("conv"),
            )?
        } else {
            let weight = vb
                .pp("conv")
                .get((in_channels, out_channels, kernel_size), "weight")?;
            let bias = if use_bias {
                Some(vb.pp("conv").get(out_channels, "bias")?)
            } else {
                None
            };
            ConvTranspose1d::new(
                weight,
                bias,
                ConvTranspose1dConfig {
                    padding: 0,
                    stride,
                    ..Default::default()
                },
            )
        };
        let total_padding = kernel_size.saturating_sub(stride);
        Ok(Self {
            conv,
            left_trim: 0,
            right_trim: total_padding,
        })
    }
}

impl Module for CausalConvTranspose1d {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let ys = xs.apply(&self.conv)?;
        let ys = if self.left_trim > 0 {
            ys.narrow(
                D::Minus1,
                self.left_trim,
                ys.dim(D::Minus1)? - self.left_trim,
            )?
        } else {
            ys
        };
        if self.right_trim > 0 {
            ys.narrow(
                D::Minus1,
                0,
                ys.dim(D::Minus1)?.saturating_sub(self.right_trim),
            )
        } else {
            Ok(ys)
        }
    }
}

#[derive(Debug, Clone)]
struct SemanticCodebook {
    cluster_usage: Tensor,
    embedding_sum: Tensor,
}

impl SemanticCodebook {
    fn new(codebook_size: usize, dim: usize, vb: ShardedVarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            cluster_usage: vb.get(codebook_size, "cluster_usage")?,
            embedding_sum: vb.get((codebook_size, dim), "embedding_sum")?,
        })
    }

    fn decode(&self, codes: &Tensor, dtype: DType) -> candle_core::Result<Tensor> {
        let usage = self
            .cluster_usage
            .clamp(1e-5, f64::INFINITY)?
            .unsqueeze(1)?;
        let embedding = self.embedding_sum.broadcast_div(&usage)?.to_dtype(dtype)?;
        let codes = codes.to_device(embedding.device())?.to_dtype(DType::U32)?;
        Embedding::new(embedding, self.embedding_sum.dim(1)?)
            .forward(&codes.squeeze(1)?)?
            .transpose(1, 2)
    }
}

#[derive(Debug, Clone)]
struct AcousticCodebook {
    n_levels: usize,
}

impl AcousticCodebook {
    fn decode(&self, codes: &Tensor, dtype: DType) -> candle_core::Result<Tensor> {
        let codes = codes.to_dtype(dtype)?;
        let scaled = (&codes * 2.)?;
        let scaled = (scaled / (self.n_levels.saturating_sub(1) as f64))?;
        scaled.broadcast_sub(&Tensor::ones_like(&scaled)?)
    }
}

#[derive(Debug, Clone)]
struct MistralAudioCodebook {
    semantic_codebook: SemanticCodebook,
    acoustic_codebook: AcousticCodebook,
}

impl MistralAudioCodebook {
    fn new(
        args: &VoxtralTtsAudioTokenizerArgs,
        vb: ShardedVarBuilder,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            semantic_codebook: SemanticCodebook::new(
                args.semantic_codebook_size,
                args.semantic_dim,
                vb.pp("semantic_codebook"),
            )?,
            acoustic_codebook: AcousticCodebook {
                n_levels: args.acoustic_codebook_size,
            },
        })
    }

    fn decode(&self, codes: &Tensor, dtype: DType) -> candle_core::Result<Tensor> {
        let semantic_codes = codes.i((.., ..1, ..))?;
        let acoustic_codes = codes.i((.., 1.., ..))?;
        let semantic = self.semantic_codebook.decode(&semantic_codes, dtype)?;
        let acoustic = self.acoustic_codebook.decode(&acoustic_codes, dtype)?;
        Tensor::cat(&[semantic, acoustic], 1)
    }

    fn num_codebooks(&self) -> usize {
        1
    }
}

#[derive(Debug, Clone)]
struct MultiVocabEmbeddings {
    offsets: Vec<u32>,
    embeddings: Embedding,
}

impl MultiVocabEmbeddings {
    fn new(
        args: &VoxtralTtsAudioTokenizerArgs,
        embedding_dim: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let mut codebook_sizes = Vec::with_capacity(1 + args.acoustic_dim);
        codebook_sizes.push((args.semantic_codebook_size + 2) as u32);
        codebook_sizes.extend(std::iter::repeat_n(
            (args.acoustic_codebook_size + 2) as u32,
            args.acoustic_dim,
        ));
        let mut offsets = Vec::with_capacity(codebook_sizes.len());
        let mut running = 0u32;
        for size in &codebook_sizes {
            offsets.push(running);
            running = running.saturating_add(*size);
        }
        let padded_size = usize::try_from(((running + 127) / 128) * 128)
            .expect("padded multi-vocab embedding size exceeds usize");
        Ok(Self {
            offsets,
            embeddings: embedding(
                padded_size,
                embedding_dim,
                vb.pp("mm_audio_embeddings")
                    .pp("audio_codebook_embeddings")
                    .pp("embeddings"),
                &None,
            )?,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, codebooks, frames) = input_ids.dims3()?;
        if codebooks != self.offsets.len() {
            candle_core::bail!(
                "audio token embedding expected {} codebooks, got {}",
                self.offsets.len(),
                codebooks
            );
        }
        let offsets = Tensor::from_vec(self.offsets.clone(), codebooks, input_ids.device())?
            .reshape((1, codebooks, 1))?;
        let offset_ids = input_ids
            .to_device(self.embeddings.embeddings().device())?
            .to_dtype(DType::U32)?
            .broadcast_add(&offsets)?;
        let flat = offset_ids
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch * frames * codebooks,))?;
        let hidden = self.embeddings.forward(&flat)?;
        let hidden_dim = hidden.dim(1)?;
        hidden
            .reshape((batch, frames, codebooks, hidden_dim))?
            .sum(D::Minus2)
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

fn get_alibi_slopes(n_heads: usize) -> Vec<f32> {
    fn power_of_two_slopes(n: usize) -> Vec<f32> {
        let ratio = 2f32.powf(-8.0 / n as f32);
        (0..n).map(|i| ratio.powi(i as i32)).collect()
    }
    if (n_heads as f32).log2().fract() == 0.0 {
        power_of_two_slopes(n_heads)
    } else {
        let m = 2usize.pow(((n_heads as f32).log2().floor()) as u32);
        let mut slopes = power_of_two_slopes(m);
        let mut extra = power_of_two_slopes(2 * m);
        extra = extra.into_iter().step_by(2).take(n_heads - m).collect();
        slopes.extend(extra);
        slopes
    }
}

fn make_attention_bias(
    seq_len: usize,
    n_heads: usize,
    sliding_window: usize,
    causal: bool,
    dtype: DType,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let slopes = get_alibi_slopes(n_heads);
    let mut values = Vec::with_capacity(n_heads * seq_len * seq_len);
    for slope in slopes {
        for i in 0..seq_len {
            for j in 0..seq_len {
                let rel = j as isize - i as isize;
                let masked = (causal && rel > 0)
                    || rel < -(sliding_window as isize)
                    || rel > if causal { 0 } else { sliding_window as isize };
                values.push(if masked {
                    f32::NEG_INFINITY
                } else {
                    slope * rel as f32
                });
            }
        }
    }
    Tensor::from_vec(values, (1, n_heads, seq_len, seq_len), device)?.to_dtype(dtype)
}

#[derive(Debug, Clone)]
struct Attention {
    wq: Arc<dyn QuantMethod>,
    wk: Arc<dyn QuantMethod>,
    wv: Arc<dyn QuantMethod>,
    wo: Arc<dyn QuantMethod>,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
    causal: bool,
}

impl Attention {
    fn new(args: &VoxtralTtsAudioTokenizerArgs, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            wq: quant_linear_b(
                args.dim,
                args.n_heads * args.head_dim,
                false,
                no_quant_config(),
                vb.pp("wq"),
            )?,
            wk: quant_linear_b(
                args.dim,
                args.n_kv_heads * args.head_dim,
                false,
                no_quant_config(),
                vb.pp("wk"),
            )?,
            wv: quant_linear_b(
                args.dim,
                args.n_kv_heads * args.head_dim,
                false,
                no_quant_config(),
                vb.pp("wv"),
            )?,
            wo: quant_linear_b(
                args.n_heads * args.head_dim,
                args.dim,
                args.use_biases,
                no_quant_config(),
                vb.pp("wo"),
            )?,
            q_norm: args
                .qk_norm
                .then(|| {
                    RmsNorm::new(
                        args.n_heads * args.head_dim,
                        args.qk_norm_eps,
                        vb.pp("q_norm"),
                    )
                })
                .transpose()?,
            k_norm: args
                .qk_norm
                .then(|| {
                    RmsNorm::new(
                        args.n_kv_heads * args.head_dim,
                        args.qk_norm_eps,
                        vb.pp("k_norm"),
                    )
                })
                .transpose()?,
            n_heads: args.n_heads,
            n_kv_heads: args.n_kv_heads,
            head_dim: args.head_dim,
            sliding_window: args.attn_sliding_window_size,
            causal: args.causal,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, seq_len, _) = xs.dims3()?;
        let mut q = qforward_autocast(xs, &*self.wq)?;
        let mut k = qforward_autocast(xs, &*self.wk)?;
        let v = qforward_autocast(xs, &*self.wv)?;
        if let Some(norm) = &self.q_norm {
            q = norm.forward(&q)?;
        }
        if let Some(norm) = &self.k_norm {
            k = norm.forward(&k)?;
        }

        let q = q
            .reshape((batch, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = repeat_kv(k, self.n_heads / self.n_kv_heads)?.contiguous()?;
        let v = repeat_kv(v, self.n_heads / self.n_kv_heads)?.contiguous()?;

        let attn_bias = make_attention_bias(
            seq_len,
            self.n_heads,
            self.sliding_window,
            self.causal,
            q.dtype(),
            q.device(),
        )?;
        let attn = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? / (self.head_dim as f64).sqrt())?
            .broadcast_add(&attn_bias)?;
        let attn = softmax(&attn, D::Minus1)?;
        let ys = attn.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((
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
struct TransformerLayer {
    attention: Attention,
    feed_forward: FeedForward,
    attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    attention_scale: Option<Tensor>,
    ffn_scale: Option<Tensor>,
}

impl TransformerLayer {
    fn new(args: &VoxtralTtsAudioTokenizerArgs, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            attention: Attention::new(args, vb.pp("attention"))?,
            feed_forward: FeedForward::new(
                args.dim,
                args.hidden_dim,
                args.use_biases,
                vb.pp("feed_forward"),
            )?,
            attention_norm: RmsNorm::new(args.dim, args.norm_eps, vb.pp("attention_norm"))?,
            ffn_norm: RmsNorm::new(args.dim, args.norm_eps, vb.pp("ffn_norm"))?,
            attention_scale: args
                .layer_scale
                .then(|| vb.get(args.dim, "attention_scale"))
                .transpose()?,
            ffn_scale: args
                .layer_scale
                .then(|| vb.get(args.dim, "ffn_scale"))
                .transpose()?,
        })
    }

    fn scale(xs: Tensor, scale: &Option<Tensor>) -> candle_core::Result<Tensor> {
        if let Some(scale) = scale {
            scale.unsqueeze(0)?.unsqueeze(0)?.broadcast_mul(&xs)
        } else {
            Ok(xs)
        }
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let attn = Self::scale(
            self.attention.forward(&self.attention_norm.forward(xs)?)?,
            &self.attention_scale,
        )?;
        let hidden = (xs + attn)?;
        let ff = Self::scale(
            self.feed_forward
                .forward(&self.ffn_norm.forward(&hidden)?)?,
            &self.ffn_scale,
        )?;
        hidden + ff
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        let mut layers = self.attention.get_isq_layers();
        layers.extend(self.feed_forward.get_isq_layers());
        layers
    }
}

#[derive(Debug, Clone)]
struct Transformer {
    layers: Vec<TransformerLayer>,
}

impl Transformer {
    fn new(
        args: &VoxtralTtsAudioTokenizerArgs,
        n_layers: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            layers: (0..n_layers)
                .map(|idx| TransformerLayer::new(args, vb.pp("layers").pp(idx)))
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut hidden = xs.clone();
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        Ok(hidden)
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        let mut layers = Vec::new();
        for layer in &mut self.layers {
            layers.extend(layer.get_isq_layers());
        }
        layers
    }
}

#[derive(Debug, Clone)]
enum DecoderBlock {
    Conv1d(CausalConv1d),
    ConvTranspose1d(CausalConvTranspose1d),
    Transformer(Transformer),
}

#[derive(Debug, Clone)]
pub struct VoxtralTtsAudioTokenizer {
    patch_size: usize,
    downsample_factor: usize,
    quantizer: MistralAudioCodebook,
    audio_token_embedding: MultiVocabEmbeddings,
    decoder_blocks: Vec<DecoderBlock>,
    output_proj: CausalConv1d,
}

impl VoxtralTtsAudioTokenizer {
    pub fn new(
        args: &VoxtralTtsAudioTokenizerArgs,
        embedding_dim: usize,
        vb: ShardedVarBuilder,
        embeddings_vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let latent_dim = args.semantic_dim + args.acoustic_dim;
        let decoder_transformer_lengths = args.decoder_transformer_lengths();
        let decoder_convs_kernels = args.decoder_convs_kernels();
        let decoder_convs_strides = args.decoder_convs_strides();
        let mut decoder_blocks = Vec::new();

        decoder_blocks.push(DecoderBlock::Conv1d(CausalConv1d::new(
            latent_dim,
            args.dim,
            decoder_convs_kernels[0],
            decoder_convs_strides[0],
            PadMode::Replicate,
            args.conv_weight_norm,
            false,
            vb.pp("decoder_blocks").pp(0),
        )?));

        for (idx, n_layers) in decoder_transformer_lengths.iter().copied().enumerate() {
            decoder_blocks.push(DecoderBlock::Transformer(Transformer::new(
                args,
                n_layers,
                vb.pp("decoder_blocks").pp(decoder_blocks.len()),
            )?));
            if idx + 1 != decoder_transformer_lengths.len()
                && (decoder_convs_kernels[idx + 1] != 1 || decoder_convs_strides[idx + 1] != 1)
            {
                decoder_blocks.push(DecoderBlock::ConvTranspose1d(CausalConvTranspose1d::new(
                    args.dim,
                    args.dim,
                    decoder_convs_kernels[idx + 1],
                    decoder_convs_strides[idx + 1],
                    args.conv_weight_norm,
                    false,
                    vb.pp("decoder_blocks").pp(decoder_blocks.len()),
                )?));
            }
        }

        Ok(Self {
            patch_size: args.pretransform_patch_size,
            downsample_factor: args.downsample_factor(),
            quantizer: MistralAudioCodebook::new(args, vb.pp("quantizer"))?,
            audio_token_embedding: MultiVocabEmbeddings::new(args, embedding_dim, embeddings_vb)?,
            decoder_blocks,
            output_proj: CausalConv1d::new(
                args.dim,
                args.pretransform_patch_size,
                args.patch_proj_kernel_size,
                1,
                PadMode::Reflect,
                args.conv_weight_norm,
                false,
                vb.pp("output_proj"),
            )?,
        })
    }

    pub fn downsample_factor(&self) -> usize {
        self.downsample_factor
    }

    pub fn num_codebooks(&self) -> usize {
        self.quantizer.num_codebooks() + 36
    }

    pub fn encode_tokens(&self, codes: &Tensor) -> candle_core::Result<Tensor> {
        self.audio_token_embedding.forward(codes)
    }

    fn forward_decoder(&self, emb: &Tensor) -> candle_core::Result<Tensor> {
        let mut hidden = emb.transpose(1, 2)?.contiguous()?;
        for block in &self.decoder_blocks {
            hidden = match block {
                DecoderBlock::Conv1d(block) => {
                    block.forward(&hidden.transpose(1, 2)?)?.transpose(1, 2)?
                }
                DecoderBlock::ConvTranspose1d(block) => {
                    block.forward(&hidden.transpose(1, 2)?)?.transpose(1, 2)?
                }
                DecoderBlock::Transformer(block) => block.forward(&hidden)?,
            };
        }
        let hidden = self.output_proj.forward(&hidden.transpose(1, 2)?)?;
        let batch = hidden.dim(0)?;
        let combined_channels = hidden.dim(1)?;
        let frames = hidden.dim(2)?;
        if combined_channels % self.patch_size != 0 {
            candle_core::bail!(
                "decoder output channels {combined_channels} are not divisible by patch size {}",
                self.patch_size
            );
        }
        let channels = combined_channels / self.patch_size;
        hidden
            .reshape((batch, channels, self.patch_size, frames))?
            .transpose(2, 3)?
            .contiguous()?
            .reshape((batch, channels, frames * self.patch_size))
    }

    pub fn decode(&self, codes: &Tensor, dtype: DType) -> candle_core::Result<Tensor> {
        let emb = self.quantizer.decode(codes, dtype)?;
        self.forward_decoder(&emb)
    }

    pub fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        let mut layers = Vec::new();
        for block in &mut self.decoder_blocks {
            if let DecoderBlock::Transformer(block) = block {
                layers.extend(block.get_isq_layers());
            }
        }
        layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn acoustic_codebook_rescales_codes() {
        let codebook = AcousticCodebook { n_levels: 21 };
        let codes = Tensor::new(vec![0u32, 10, 20], &Device::Cpu)
            .unwrap()
            .reshape((1, 1, 3))
            .unwrap();
        let decoded = codebook.decode(&codes, DType::F32).unwrap();
        let values = decoded.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(values[0], -1.0);
        assert!((values[1] - 0.0).abs() < 1e-6);
        assert_eq!(values[2], 1.0);
    }

    #[test]
    fn replicate_pad_repeats_edge_values() {
        let xs = Tensor::new(vec![1f32, 2., 3.], &Device::Cpu)
            .unwrap()
            .reshape((1, 1, 3))
            .unwrap();
        let padded = replication_pad1d(&xs, 2, 1).unwrap();
        let values = padded.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(values, vec![1.0, 1.0, 1.0, 2.0, 3.0, 3.0]);
    }
}
