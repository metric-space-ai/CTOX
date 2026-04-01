#![allow(dead_code)]

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, Linear};
use mistralrs_quant::{Convolution, QuantizedConfig, ShardedVarBuilder};
use std::sync::Mutex;

use crate::{
    layers::{repeat_kv, Activation},
    speech_models::qwen3_tts::Qwen3TtsTokenizerEncoderConfig,
};

fn tensor_debug_summary(name: &str, tensor: &Tensor) -> candle_core::Result<String> {
    let shape = tensor.shape().dims().to_vec();
    let flat = tensor.flatten_all()?.to_dtype(DType::F32)?;
    let values = flat.to_vec1::<f32>()?;
    let count = values.len().max(1) as f64;
    let mean = values.iter().map(|v| *v as f64).sum::<f64>() / count;
    let var = values
        .iter()
        .map(|v| {
            let dv = *v as f64 - mean;
            dv * dv
        })
        .sum::<f64>()
        / count;
    let preview = values.iter().take(8).copied().collect::<Vec<_>>();
    Ok(format!(
        "{name} shape={shape:?} mean={mean:.6} std={std:.6} slice={preview:?}",
        std = var.sqrt()
    ))
}

fn tensor_vec8_summary(name: &str, tensor: &Tensor) -> candle_core::Result<String> {
    let vals = tensor.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    Ok(format!("{name} vec8={:?}", vals))
}

fn mimi_elu(xs: &Tensor) -> candle_core::Result<Tensor> {
    xs.to_dtype(DType::F32)?.elu(1.0)?.to_dtype(xs.dtype())
}

fn rotate_half(xs: &Tensor) -> candle_core::Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let first = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let second = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&second.neg()?, &first], D::Minus1)
}

#[derive(Debug, Clone)]
struct MimiRotaryEmbedding {
    inv_freq: Tensor,
}

impl MimiRotaryEmbedding {
    fn new(base: f32, head_dim: usize, device: &Device) -> candle_core::Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / head_dim as f32))
            .collect();
        Ok(Self {
            inv_freq: Tensor::from_vec(inv_freq, head_dim / 2, device)?,
        })
    }

    fn forward(&self, q: &Tensor, k: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let positions = Tensor::arange(0u32, seq_len as u32, q.device())?
            .to_dtype(DType::F32)?
            .reshape((seq_len, 1))?;
        let freqs = positions.matmul(&self.inv_freq.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        let cos = emb.cos()?.to_dtype(q.dtype())?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = emb.sin()?.to_dtype(q.dtype())?.unsqueeze(0)?.unsqueeze(0)?;
        let q_embed = (q.broadcast_mul(&cos)? + rotate_half(q)?.broadcast_mul(&sin)?)?;
        let k_embed = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin)?)?;
        Ok((q_embed, k_embed))
    }
}

fn hidden_act(name: &str) -> Result<Activation> {
    match name.trim().to_ascii_lowercase().as_str() {
        "gelu" => Ok(Activation::Gelu),
        "gelu_new" | "gelu_pytorch_tanh" => Ok(Activation::NewGelu),
        "relu" => Ok(Activation::Relu),
        "silu" | "swish" => Ok(Activation::Silu),
        other => anyhow::bail!("Unsupported Qwen3-TTS encoder activation `{other}`."),
    }
}

fn no_quant_config() -> &'static Option<QuantizedConfig> {
    static NO_QUANT_CONFIG: Option<QuantizedConfig> = None;
    &NO_QUANT_CONFIG
}

fn vb_conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv1dConfig,
    bias: bool,
    vb: candle_nn::VarBuilder<'_>,
) -> candle_core::Result<Conv1d> {
    let ws = vb.get(
        (out_channels, in_channels / cfg.groups, kernel_size),
        "weight",
    )?;
    let bs = if bias {
        Some(vb.get(out_channels, "bias")?)
    } else {
        None
    };
    Ok(Conv1d::new(ws, bs, cfg))
}

fn vb_linear(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: candle_nn::VarBuilder<'_>,
) -> candle_core::Result<Linear> {
    let ws = vb.get((out_dim, in_dim), "weight")?;
    let bs = if bias {
        Some(vb.get(out_dim, "bias")?)
    } else {
        None
    };
    Ok(Linear::new(ws, bs))
}

#[derive(Debug, Clone)]
struct MimiLinear {
    linear: Linear,
}

impl MimiLinear {
    fn new(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: candle_nn::VarBuilder<'_>,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            linear: vb_linear(in_dim, out_dim, bias, vb)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.linear.forward(xs)
    }
}

fn conv1d_to_f32(conv: &Conv1d) -> candle_core::Result<Conv1d> {
    Ok(Conv1d::new(
        conv.weight().to_dtype(DType::F32)?,
        match conv.bias() {
            Some(bias) => Some(bias.to_dtype(DType::F32)?),
            None => None,
        },
        conv.config().clone(),
    ))
}

#[derive(Debug, Clone)]
struct MimiLayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl MimiLayerNorm {
    fn new(size: usize, eps: f64, vb: candle_nn::VarBuilder<'_>) -> candle_core::Result<Self> {
        Ok(Self {
            weight: vb.get(size, "weight")?,
            bias: vb.get(size, "bias")?,
            eps,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mean = xs.mean_keepdim(D::Minus1)?;
        let centered = xs.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = centered.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        normed
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)
    }
}

fn sliding_causal_attention_mask(
    seq_len: usize,
    sliding_window: Option<usize>,
    device: &Device,
    dtype: DType,
) -> candle_core::Result<Tensor> {
    let mut data = Vec::with_capacity(seq_len * seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            let blocked = j > i
                || sliding_window
                    .map(|window| i.saturating_sub(j) >= window)
                    .unwrap_or(false);
            data.push(if blocked { f32::NEG_INFINITY } else { 0.0 });
        }
    }
    Tensor::from_vec(data, (1, 1, seq_len, seq_len), device)?.to_dtype(dtype)
}

fn replicate_pad_1d(xs: &Tensor, left: usize, right: usize) -> candle_core::Result<Tensor> {
    let xs = xs.contiguous()?;
    let (_b, _c, width) = xs.dims3()?;

    let left_pad = if left > 0 {
        let indices = vec![0i64; left];
        Some(xs.index_select(&Tensor::new(indices, xs.device())?, 2)?)
    } else {
        None
    };

    let right_pad = if right > 0 {
        let last = width.saturating_sub(1) as i64;
        let indices = vec![last; right];
        Some(xs.index_select(&Tensor::new(indices, xs.device())?, 2)?)
    } else {
        None
    };

    match (left_pad, right_pad) {
        (Some(l), Some(r)) => Tensor::cat(&[l, xs.clone(), r], 2),
        (Some(l), None) => Tensor::cat(&[l, xs.clone()], 2),
        (None, Some(r)) => Tensor::cat(&[xs.clone(), r], 2),
        (None, None) => Ok(xs),
    }
}

fn reflect_pad_1d(xs: &Tensor, left: usize, right: usize) -> candle_core::Result<Tensor> {
    let mut xs = xs.contiguous()?;
    let (_b, _c, width) = xs.dims3()?;
    let max_pad = left.max(right);
    let mut extra_pad = 0usize;
    if width <= max_pad {
        extra_pad = max_pad - width + 1;
        xs = xs.pad_with_zeros(D::Minus1, 0, extra_pad)?;
    }
    let width = xs.dim(D::Minus1)?;

    let left_pad = if left > 0 {
        let indices = (1..=left).rev().map(|i| i as i64).collect::<Vec<_>>();
        Some(xs.index_select(&Tensor::new(indices, xs.device())?, 2)?)
    } else {
        None
    };

    let right_pad = if right > 0 {
        let indices = (0..right)
            .map(|i| (width.saturating_sub(2 + i)) as i64)
            .collect::<Vec<_>>();
        Some(xs.index_select(&Tensor::new(indices, xs.device())?, 2)?)
    } else {
        None
    };

    let padded = match (left_pad, right_pad) {
        (Some(l), Some(r)) => Tensor::cat(&[l, xs.clone(), r], 2)?,
        (Some(l), None) => Tensor::cat(&[l, xs.clone()], 2)?,
        (None, Some(r)) => Tensor::cat(&[xs.clone(), r], 2)?,
        (None, None) => xs,
    };

    if extra_pad > 0 {
        padded.narrow(D::Minus1, 0, padded.dim(D::Minus1)? - extra_pad)
    } else {
        Ok(padded)
    }
}

#[derive(Debug, Clone)]
struct MimiConv1dPaddingCache {
    per_layer_padding: Vec<usize>,
    per_layer_padding_mode: Vec<String>,
    per_layer_in_channels: Vec<usize>,
    padding_cache: Vec<Option<Tensor>>,
}

impl MimiConv1dPaddingCache {
    fn new(
        per_layer_padding: Vec<usize>,
        per_layer_padding_mode: Vec<String>,
        per_layer_in_channels: Vec<usize>,
    ) -> candle_core::Result<Self> {
        let num_layers = per_layer_padding.len();
        if per_layer_padding_mode.len() != num_layers || per_layer_in_channels.len() != num_layers {
            candle_core::bail!(
                "Expected identical lengths for MimiConv1dPaddingCache layer metadata."
            );
        }
        if per_layer_padding_mode
            .iter()
            .any(|mode| mode != "constant" && mode != "replicate")
        {
            candle_core::bail!(
                "`padding_cache` is only supported for `constant` and `replicate` pad modes."
            );
        }
        Ok(Self {
            per_layer_padding,
            per_layer_padding_mode,
            per_layer_in_channels,
            padding_cache: vec![None; num_layers],
        })
    }

    fn update(&mut self, hidden_states: &Tensor, layer_idx: usize) -> candle_core::Result<Tensor> {
        let hidden_states = hidden_states.contiguous()?;
        let (batch_size, in_channels, seq_len) = hidden_states.dims3()?;
        let padding = self.per_layer_padding[layer_idx];
        let padding_mode = &self.per_layer_padding_mode[layer_idx];
        let expected_channels = self.per_layer_in_channels[layer_idx];
        if in_channels != expected_channels {
            candle_core::bail!(
                "Padding cache channel mismatch for layer {layer_idx}: got {in_channels}, expected {expected_channels}."
            );
        }

        let current_cache = if let Some(cache) = &self.padding_cache[layer_idx] {
            cache.clone()
        } else if padding_mode == "constant" {
            Tensor::zeros(
                (batch_size, in_channels, padding),
                hidden_states.dtype(),
                hidden_states.device(),
            )?
        } else {
            Tensor::ones(
                (batch_size, in_channels, padding),
                hidden_states.dtype(),
                hidden_states.device(),
            )?
            .broadcast_mul(&hidden_states.i((.., .., ..1))?)?
        };

        let padding_states = if padding > 0 {
            hidden_states.i((.., .., seq_len.saturating_sub(padding)..))?
        } else {
            Tensor::zeros(
                (batch_size, in_channels, 0usize),
                hidden_states.dtype(),
                hidden_states.device(),
            )?
        };
        self.padding_cache[layer_idx] = Some(padding_states);
        Ok(current_cache)
    }
}

#[derive(Debug, Clone)]
struct MimiConv1d {
    conv: Conv1d,
    conv_f32: Option<Conv1d>,
    causal: bool,
    pad_mode: String,
    layer_idx: Option<usize>,
    in_channels: usize,
    stride: usize,
    effective_kernel_size: usize,
    padding_total: usize,
    padding_left: usize,
    padding_right: usize,
}

impl MimiConv1d {
    fn new(
        cfg: &Qwen3TtsTokenizerEncoderConfig,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        pad_mode: Option<&str>,
        bias: bool,
        layer_idx: Option<usize>,
        vb: candle_nn::VarBuilder<'_>,
    ) -> candle_core::Result<Self> {
        let conv_cfg = Conv1dConfig {
            padding: 0,
            stride,
            dilation,
            groups,
            ..Default::default()
        };
        let effective_kernel_size = (kernel_size - 1) * dilation + 1;
        let padding_total = effective_kernel_size.saturating_sub(stride);
        let padding_right = if cfg.use_causal_conv {
            0
        } else {
            padding_total / 2
        };
        let padding_left = padding_total.saturating_sub(padding_right);
        let conv = vb_conv1d(
            in_channels,
            out_channels,
            kernel_size,
            conv_cfg,
            bias,
            vb.pp("conv"),
        )?;
        Ok(Self {
            conv_f32: None,
            conv,
            causal: cfg.use_causal_conv,
            pad_mode: pad_mode.unwrap_or(&cfg.pad_mode).to_string(),
            layer_idx,
            in_channels,
            stride,
            effective_kernel_size,
            padding_total,
            padding_left,
            padding_right,
        })
    }

    fn extra_padding(&self, len: usize) -> usize {
        let n_frames = (len as f64 - self.effective_kernel_size as f64 + self.padding_total as f64)
            / self.stride as f64
            + 1.0;
        let ideal_length = ((n_frames.ceil() - 1.0) * self.stride as f64
            + (self.effective_kernel_size - self.padding_total) as f64)
            as usize;
        ideal_length.saturating_sub(len)
    }

    fn get_output_length(&self, input_length: usize) -> usize {
        let extra_padding = self.extra_padding(input_length);
        let padded = if self.causal {
            input_length + self.padding_total + extra_padding
        } else {
            input_length + self.padding_left + self.padding_right + extra_padding
        };
        (padded.saturating_sub(self.effective_kernel_size) / self.stride) + 1
    }

    fn forward_with_cache(
        &self,
        xs: &Tensor,
        padding_cache: Option<&mut MimiConv1dPaddingCache>,
    ) -> candle_core::Result<Tensor> {
        let input_len = xs.dim(D::Minus1)?;
        let cached_causal = self.causal && padding_cache.is_some();
        let padded = match padding_cache {
            Some(padding_cache) => {
                if !self.causal {
                    candle_core::bail!(
                        "`padding_cache` is not supported for non-causal convolutions."
                    );
                }
                let layer_idx = self.layer_idx.ok_or_else(|| {
                    candle_core::Error::Msg(
                        "Missing MimiConv1d layer_idx for padding-cache encode path.".into(),
                    )
                })?;
                let layer_padding_cache = padding_cache.update(xs, layer_idx)?;
                Tensor::cat(&[&layer_padding_cache, xs], 2)?
            }
            None => {
                let extra_padding = self.extra_padding(xs.dim(D::Minus1)?);
                let (left_pad, right_pad) = if self.causal {
                    (self.padding_total, extra_padding)
                } else {
                    (self.padding_left, self.padding_right + extra_padding)
                };
                match self.pad_mode.as_str() {
                    "constant" => xs.pad_with_zeros(D::Minus1, left_pad, right_pad)?,
                    "replicate" => replicate_pad_1d(xs, left_pad, right_pad)?,
                    "reflect" => reflect_pad_1d(xs, left_pad, right_pad)?,
                    other => {
                        candle_core::bail!("Unsupported Qwen3-TTS MimiConv1d pad_mode `{other}`.");
                    }
                }
            }
        };
        let _ = input_len;
        let _ = cached_causal;
        if let Some(conv_f32) = &self.conv_f32 {
            Convolution
                .forward_1d(conv_f32, &padded.to_dtype(DType::F32)?)?
                .to_dtype(xs.dtype())
        } else {
            Convolution.forward_1d(&self.conv, &padded)
        }
    }

    fn with_f32_forward(mut self) -> candle_core::Result<Self> {
        self.conv_f32 = Some(Conv1d::new(
            self.conv.weight().to_dtype(DType::F32)?,
            match self.conv.bias() {
                Some(bias) => Some(bias.to_dtype(DType::F32)?),
                None => None,
            },
            self.conv.config().clone(),
        ));
        Ok(self)
    }
}

impl Module for MimiConv1d {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.forward_with_cache(xs, None)
    }
}

#[derive(Debug, Clone)]
struct MimiResnetBlock {
    conv1: MimiConv1d,
    conv2: MimiConv1d,
    shortcut: Option<MimiConv1d>,
}

impl MimiResnetBlock {
    fn new(
        cfg: &Qwen3TtsTokenizerEncoderConfig,
        dim: usize,
        dilations: (usize, usize),
        first_layer_idx: usize,
        vb: candle_nn::VarBuilder<'_>,
    ) -> Result<Self> {
        let hidden = dim / cfg.compress;
        Ok(Self {
            conv1: MimiConv1d::new(
                cfg,
                dim,
                hidden,
                cfg.residual_kernel_size,
                1,
                dilations.0,
                1,
                None,
                true,
                Some(first_layer_idx),
                vb.pp("block").pp(1),
            )?,
            conv2: MimiConv1d::new(
                cfg,
                hidden,
                dim,
                1,
                1,
                dilations.1,
                1,
                None,
                true,
                Some(first_layer_idx + 1),
                vb.pp("block").pp(3),
            )?,
            shortcut: if cfg.use_conv_shortcut {
                Some(MimiConv1d::new(
                    cfg,
                    dim,
                    dim,
                    1,
                    1,
                    1,
                    1,
                    None,
                    true,
                    None,
                    vb.pp("shortcut"),
                )?)
            } else {
                None
            },
        })
    }
}

impl Module for MimiResnetBlock {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.forward_with_cache(xs, None)
    }
}

impl MimiResnetBlock {
    fn forward_with_cache(
        &self,
        xs: &Tensor,
        mut padding_cache: Option<&mut MimiConv1dPaddingCache>,
    ) -> candle_core::Result<Tensor> {
        let residual = if let Some(shortcut) = &self.shortcut {
            shortcut.forward_with_cache(xs, None)?
        } else {
            xs.clone()
        };
        let hidden = mimi_elu(xs)?;
        if self.conv1.layer_idx == Some(1) {
            tracing::info!("{}", tensor_debug_summary("Qwen3-TTS res1 input", xs)?);
            tracing::info!("{}", tensor_debug_summary("Qwen3-TTS res1 elu1", &hidden)?);
        }
        let hidden = self
            .conv1
            .forward_with_cache(&hidden, padding_cache.as_deref_mut())?;
        if self.conv1.layer_idx == Some(1) {
            tracing::info!("{}", tensor_debug_summary("Qwen3-TTS res1 conv1", &hidden)?);
        }
        let hidden = mimi_elu(&hidden)?;
        if self.conv1.layer_idx == Some(1) {
            tracing::info!("{}", tensor_debug_summary("Qwen3-TTS res1 elu2", &hidden)?);
        }
        let hidden = self
            .conv2
            .forward_with_cache(&hidden, padding_cache.as_deref_mut())?;
        if self.conv1.layer_idx == Some(1) {
            tracing::info!("{}", tensor_debug_summary("Qwen3-TTS res1 conv2", &hidden)?);
        }
        let out = (residual + hidden)?;
        if self.conv1.layer_idx == Some(1) {
            tracing::info!("{}", tensor_debug_summary("Qwen3-TTS res1 out", &out)?);
        }
        Ok(out)
    }

    fn get_output_length(&self, input_length: usize) -> usize {
        let hidden = self.conv1.get_output_length(input_length);
        self.conv2.get_output_length(hidden)
    }

    fn append_cache_specs(&self, out: &mut Vec<(usize, String, usize)>) {
        out.push((
            self.conv1.padding_total,
            self.conv1.pad_mode.clone(),
            self.conv1.in_channels,
        ));
        out.push((
            self.conv2.padding_total,
            self.conv2.pad_mode.clone(),
            self.conv2.in_channels,
        ));
    }
}

#[derive(Debug, Clone)]
enum MimiEncoderLayer {
    Conv(MimiConv1d),
    Resnet(MimiResnetBlock),
    Elu,
}

impl MimiEncoderLayer {
    fn forward_with_cache(
        &self,
        xs: &Tensor,
        padding_cache: Option<&mut MimiConv1dPaddingCache>,
    ) -> candle_core::Result<Tensor> {
        match self {
            Self::Conv(layer) => layer.forward_with_cache(xs, padding_cache),
            Self::Resnet(layer) => layer.forward_with_cache(xs, padding_cache),
            Self::Elu => mimi_elu(xs),
        }
    }

    fn get_output_length(&self, input_length: usize) -> usize {
        match self {
            Self::Conv(layer) => layer.get_output_length(input_length),
            Self::Resnet(layer) => layer.get_output_length(input_length),
            Self::Elu => input_length,
        }
    }

    fn append_cache_specs(&self, out: &mut Vec<(usize, String, usize)>) {
        match self {
            Self::Conv(layer) => out.push((
                layer.padding_total,
                layer.pad_mode.clone(),
                layer.in_channels,
            )),
            Self::Resnet(layer) => layer.append_cache_specs(out),
            Self::Elu => {}
        }
    }
}

#[derive(Debug, Clone)]
struct MimiEncoder {
    layers: Vec<MimiEncoderLayer>,
}

impl MimiEncoder {
    fn new(cfg: &Qwen3TtsTokenizerEncoderConfig, vb: candle_nn::VarBuilder<'_>) -> Result<Self> {
        let mut next_layer_idx = 0usize;
        let mut weight_layer_idx = 0usize;
        let vb_layers = vb.pp("layers");
        let mut layers = vec![MimiEncoderLayer::Conv(MimiConv1d::new(
            cfg,
            cfg.audio_channels,
            cfg.num_filters,
            cfg.kernel_size,
            1,
            1,
            1,
            None,
            true,
            Some(next_layer_idx),
            vb_layers.pp(weight_layer_idx),
        )?)];
        next_layer_idx += 1;
        weight_layer_idx += 1;
        let mut scaling = 1usize;

        for ratio in cfg.upsampling_ratios.iter().rev().copied() {
            let current_scale = scaling * cfg.num_filters;
            for j in 0..cfg.num_residual_layers {
                layers.push(MimiEncoderLayer::Resnet(MimiResnetBlock::new(
                    cfg,
                    current_scale,
                    (cfg.dilation_growth_rate.pow(j as u32), 1),
                    next_layer_idx,
                    vb_layers.pp(weight_layer_idx),
                )?));
                next_layer_idx += 2;
                weight_layer_idx += 1;
            }
            layers.push(MimiEncoderLayer::Elu);
            let downsample_conv = MimiConv1d::new(
                cfg,
                current_scale,
                current_scale * 2,
                ratio * 2,
                ratio,
                1,
                1,
                None,
                true,
                Some(next_layer_idx),
                // Candle's SeaNet weight numbering skips one slot before each downsample conv.
                vb_layers.pp(weight_layer_idx + 1),
            )?;
            let downsample_conv = if current_scale * 2 > cfg.hidden_size {
                downsample_conv.with_f32_forward()?
            } else {
                downsample_conv
            };
            layers.push(MimiEncoderLayer::Conv(downsample_conv));
            next_layer_idx += 1;
            weight_layer_idx += 2;
            scaling *= 2;
        }

        layers.push(MimiEncoderLayer::Elu);
        layers.push(MimiEncoderLayer::Conv(MimiConv1d::new(
            cfg,
            scaling * cfg.num_filters,
            cfg.hidden_size,
            cfg.last_kernel_size,
            1,
            1,
            1,
            None,
            true,
            Some(next_layer_idx),
            // Candle's final conv is stored one slot after the last parameterized stage.
            vb_layers.pp(weight_layer_idx + 1),
        )?));
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.forward_with_cache(xs, None)
    }

    fn forward_with_cache(
        &self,
        xs: &Tensor,
        mut padding_cache: Option<&mut MimiConv1dPaddingCache>,
    ) -> candle_core::Result<Tensor> {
        let mut hidden = xs.clone();
        for (idx, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_with_cache(&hidden, padding_cache.as_deref_mut())?;
            tracing::info!(
                "{}",
                tensor_debug_summary(&format!("Qwen3-TTS encoder layer {idx}"), &hidden)?
            );
            if let Ok((_, channels, frames)) = hidden.dims3() {
                if channels >= 8 && frames > 1 {
                    tracing::info!(
                        "{}",
                        tensor_vec8_summary(
                            &format!("Qwen3-TTS encoder layer {idx} frame1"),
                            &hidden.i((0, ..8, 1))?,
                        )?
                    );
                }
            }
        }
        Ok(hidden)
    }

    fn collect_cache_specs(&self) -> Vec<(usize, String, usize)> {
        let mut out = Vec::new();
        for layer in &self.layers {
            layer.append_cache_specs(&mut out);
        }
        out
    }

    fn get_output_length(&self, mut input_length: usize) -> usize {
        for layer in &self.layers {
            input_length = layer.get_output_length(input_length);
        }
        input_length
    }
}

#[derive(Debug, Clone)]
struct MimiLayerScale {
    scale: Tensor,
}

impl MimiLayerScale {
    fn new(dim: usize, vb: candle_nn::VarBuilder<'_>) -> candle_core::Result<Self> {
        Ok(Self {
            scale: vb.get(dim, "scale")?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.scale.unsqueeze(0)?.broadcast_mul(xs)
    }
}

#[derive(Debug, Clone)]
struct MimiAttention {
    q_proj: MimiLinear,
    k_proj: MimiLinear,
    v_proj: MimiLinear,
    o_proj: MimiLinear,
    rotary_emb: MimiRotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
}

impl MimiAttention {
    fn new(cfg: &Qwen3TtsTokenizerEncoderConfig, vb: candle_nn::VarBuilder<'_>) -> Result<Self> {
        let device = vb.device().clone();
        Ok(Self {
            q_proj: MimiLinear::new(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim,
                cfg.attention_bias,
                vb.pp("q_proj"),
            )?,
            k_proj: MimiLinear::new(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                cfg.attention_bias,
                vb.pp("k_proj"),
            )?,
            v_proj: MimiLinear::new(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                cfg.attention_bias,
                vb.pp("v_proj"),
            )?,
            o_proj: MimiLinear::new(
                cfg.num_attention_heads * cfg.head_dim,
                cfg.hidden_size,
                cfg.attention_bias,
                vb.pp("o_proj"),
            )?,
            rotary_emb: MimiRotaryEmbedding::new(cfg.rope_theta as f32, cfg.head_dim, &device)?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            sliding_window: cfg.sliding_window,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
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
        let (q, k) = self.rotary_emb.forward(&q, &k)?;
        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;
        let attn_mask = sliding_causal_attention_mask(
            seq_len,
            Some(self.sliding_window),
            hidden_states.device(),
            hidden_states.dtype(),
        )?;
        let attn = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?
            / (self.head_dim as f64).sqrt())?
        .broadcast_add(&attn_mask)?;
        let attn = candle_nn::ops::softmax(&attn.to_dtype(DType::F32)?, D::Minus1)?
            .to_dtype(q.dtype())?
            .contiguous()?;
        let attn = attn.matmul(&v.contiguous()?)?.transpose(1, 2)?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;
        self.o_proj.forward(&attn)
    }
}

#[derive(Debug, Clone)]
struct MimiMlp {
    fc1: MimiLinear,
    fc2: MimiLinear,
    act: Activation,
}

impl MimiMlp {
    fn new(cfg: &Qwen3TtsTokenizerEncoderConfig, vb: candle_nn::VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            fc1: MimiLinear::new(cfg.hidden_size, cfg.intermediate_size, false, vb.pp("fc1"))?,
            fc2: MimiLinear::new(cfg.intermediate_size, cfg.hidden_size, false, vb.pp("fc2"))?,
            act: hidden_act(&cfg.hidden_act)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.fc2.forward(&self.act.forward(&self.fc1.forward(xs)?)?)
    }
}

#[derive(Debug, Clone)]
struct MimiTransformerLayer {
    layer_idx: usize,
    self_attn: MimiAttention,
    mlp: MimiMlp,
    input_layernorm: MimiLayerNorm,
    post_attention_layernorm: MimiLayerNorm,
    self_attn_layer_scale: MimiLayerScale,
    mlp_layer_scale: MimiLayerScale,
}

impl MimiTransformerLayer {
    fn new(
        cfg: &Qwen3TtsTokenizerEncoderConfig,
        layer_idx: usize,
        vb: candle_nn::VarBuilder<'_>,
    ) -> Result<Self> {
        Ok(Self {
            layer_idx,
            self_attn: MimiAttention::new(cfg, vb.pp("self_attn"))?,
            mlp: MimiMlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: MimiLayerNorm::new(
                cfg.hidden_size,
                cfg.norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: MimiLayerNorm::new(
                cfg.hidden_size,
                cfg.norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            self_attn_layer_scale: MimiLayerScale::new(
                cfg.hidden_size,
                vb.pp("self_attn_layer_scale"),
            )?,
            mlp_layer_scale: MimiLayerScale::new(cfg.hidden_size, vb.pp("mlp_layer_scale"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let residual = xs;
        let input_ln = self.input_layernorm.forward(xs)?;
        let attn_hidden = self.self_attn.forward(&input_ln)?;
        let hidden = (residual + self.self_attn_layer_scale.forward(&attn_hidden)?)?;
        let residual = &hidden;
        let post_attn_ln = self.post_attention_layernorm.forward(&hidden)?;
        let mlp_hidden = self.mlp.forward(&post_attn_ln)?;
        if self.layer_idx == 0 || self.layer_idx == 7 {
            let prefix = format!("Qwen3-TTS xf{}", self.layer_idx);
            tracing::info!(
                "{}",
                tensor_debug_summary(&format!("{prefix} input_ln"), &input_ln)?
            );
            if let Ok((_, seq_len, hidden_size)) = input_ln.dims3() {
                if seq_len > 1 && hidden_size >= 8 {
                    tracing::info!(
                        "{}",
                        tensor_vec8_summary(
                            &format!("{prefix} input_ln frame1"),
                            &input_ln.i((0, 1, 0..8))?,
                        )?
                    );
                }
            }
            tracing::info!(
                "{}",
                tensor_debug_summary(&format!("{prefix} attn_raw"), &attn_hidden)?
            );
            if let Ok((_, seq_len, hidden_size)) = attn_hidden.dims3() {
                if seq_len > 1 && hidden_size >= 8 {
                    tracing::info!(
                        "{}",
                        tensor_vec8_summary(
                            &format!("{prefix} attn_raw frame1"),
                            &attn_hidden.i((0, 1, 0..8))?,
                        )?
                    );
                }
            }
            tracing::info!(
                "{}",
                tensor_debug_summary(&format!("{prefix} post_attn_ln"), &post_attn_ln)?
            );
            if let Ok((_, seq_len, hidden_size)) = post_attn_ln.dims3() {
                if seq_len > 1 && hidden_size >= 8 {
                    tracing::info!(
                        "{}",
                        tensor_vec8_summary(
                            &format!("{prefix} post_attn_ln frame1"),
                            &post_attn_ln.i((0, 1, 0..8))?,
                        )?
                    );
                }
            }
            tracing::info!(
                "{}",
                tensor_debug_summary(&format!("{prefix} mlp_raw"), &mlp_hidden)?
            );
            if let Ok((_, seq_len, hidden_size)) = mlp_hidden.dims3() {
                if seq_len > 1 && hidden_size >= 8 {
                    tracing::info!(
                        "{}",
                        tensor_vec8_summary(
                            &format!("{prefix} mlp_raw frame1"),
                            &mlp_hidden.i((0, 1, 0..8))?,
                        )?
                    );
                }
            }
        }
        residual + self.mlp_layer_scale.forward(&mlp_hidden)?
    }
}

#[derive(Debug, Clone)]
struct MimiTransformer {
    layers: Vec<MimiTransformerLayer>,
}

impl MimiTransformer {
    fn new(cfg: &Qwen3TtsTokenizerEncoderConfig, vb: candle_nn::VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            layers: (0..cfg.num_hidden_layers)
                .map(|idx| MimiTransformerLayer::new(cfg, idx, vb.pp("layers").pp(idx)))
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut hidden = xs.clone();
        for (idx, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden)?;
            tracing::info!(
                "{}",
                tensor_debug_summary(
                    &format!("Qwen3-TTS encoder transformer layer {idx}"),
                    &hidden
                )?
            );
            if let Ok((_, seq_len, hidden_size)) = hidden.dims3() {
                if seq_len > 1 && hidden_size >= 8 {
                    tracing::info!(
                        "{}",
                        tensor_vec8_summary(
                            &format!("Qwen3-TTS encoder transformer layer {idx} frame1"),
                            &hidden.i((0, 1, ..8))?,
                        )?
                    );
                }
            }
        }
        Ok(hidden)
    }
}

#[derive(Debug, Clone)]
struct MimiEuclideanCodebook {
    cluster_usage: Tensor,
    embed_sum: Tensor,
    epsilon: f64,
}

impl MimiEuclideanCodebook {
    fn new(
        cfg: &Qwen3TtsTokenizerEncoderConfig,
        vb: candle_nn::VarBuilder<'_>,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            cluster_usage: vb.get(cfg.codebook_size, "cluster_usage")?,
            embed_sum: vb.get((cfg.codebook_size, cfg.codebook_dim), "embed_sum")?,
            epsilon: 1e-5,
        })
    }

    fn embed(&self) -> candle_core::Result<Tensor> {
        let usage = self
            .cluster_usage
            .clamp(self.epsilon, f64::INFINITY)?
            .unsqueeze(1)?;
        self.embed_sum.broadcast_div(&usage)
    }

    fn encode(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, seq_len, dim) = hidden_states.dims3()?;
        let flat = hidden_states
            .reshape((batch * seq_len, dim))?
            .to_dtype(DType::F32)?;
        let embed = self
            .embed()?
            .to_device(flat.device())?
            .to_dtype(DType::F32)?;
        let diff = flat.unsqueeze(1)?.broadcast_sub(&embed.unsqueeze(0)?)?;
        let dists = diff.sqr()?.sum(D::Minus1)?.sqrt()?;
        dists.argmin(D::Minus1)?.reshape((batch, seq_len))
    }

    fn decode(&self, indices: &Tensor) -> candle_core::Result<Tensor> {
        let embed = self.embed()?;
        candle_nn::Embedding::new(embed, self.embed_sum.dim(1)?).forward(indices)
    }

    fn topk_for_vector(&self, vector: &Tensor, k: usize) -> candle_core::Result<Vec<(u32, f32)>> {
        let v = vector.to_dtype(DType::F32)?.reshape((1, vector.dim(0)?))?;
        let embed = self.embed()?.to_device(v.device())?.to_dtype(DType::F32)?;
        let diff = v.unsqueeze(1)?.broadcast_sub(&embed.unsqueeze(0)?)?;
        let dists = diff.sqr()?.sum(D::Minus1)?.sqrt()?.squeeze(0)?;
        let vals = dists.to_vec1::<f32>()?;
        let mut pairs = vals
            .into_iter()
            .enumerate()
            .map(|(idx, dist)| (idx as u32, dist))
            .collect::<Vec<_>>();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(k);
        Ok(pairs)
    }
}

#[derive(Debug, Clone)]
struct MimiVectorQuantization {
    codebook: MimiEuclideanCodebook,
}

impl MimiVectorQuantization {
    fn new(
        cfg: &Qwen3TtsTokenizerEncoderConfig,
        vb: candle_nn::VarBuilder<'_>,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            codebook: MimiEuclideanCodebook::new(cfg, vb.pp("codebook"))?,
        })
    }

    fn encode(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        self.codebook.encode(&hidden_states.permute((0, 2, 1))?)
    }

    fn decode(&self, indices: &Tensor) -> candle_core::Result<Tensor> {
        self.codebook.decode(indices)?.permute((0, 2, 1))
    }
}

#[derive(Debug, Clone)]
struct MimiResidualVectorQuantizer {
    input_proj: Option<Conv1d>,
    output_proj: Option<Conv1d>,
    layers: Vec<MimiVectorQuantization>,
}

impl MimiResidualVectorQuantizer {
    fn new(
        cfg: &Qwen3TtsTokenizerEncoderConfig,
        num_quantizers: usize,
        runtime_dtype: DType,
        vb: candle_nn::VarBuilder<'_>,
    ) -> Result<Self> {
        let input_proj = if cfg.vector_quantization_hidden_dimension != cfg.hidden_size {
            Some(Conv1d::new(
                vb.pp("input_proj")
                    .get(
                        (cfg.vector_quantization_hidden_dimension, cfg.hidden_size, 1),
                        "weight",
                    )?
                    .to_dtype(runtime_dtype)?,
                None,
                Conv1dConfig::default(),
            ))
        } else {
            None
        };
        let output_proj = if cfg.vector_quantization_hidden_dimension != cfg.hidden_size {
            Some(Conv1d::new(
                vb.pp("output_proj")
                    .get(
                        (cfg.hidden_size, cfg.vector_quantization_hidden_dimension, 1),
                        "weight",
                    )?
                    .to_dtype(runtime_dtype)?,
                None,
                Conv1dConfig::default(),
            ))
        } else {
            None
        };
        Ok(Self {
            input_proj,
            output_proj,
            layers: (0..num_quantizers)
                .map(|idx| MimiVectorQuantization::new(cfg, vb.pp("layers").pp(idx)))
                .collect::<candle_core::Result<Vec<_>>>()?,
        })
    }

    fn encode(
        &self,
        label: &str,
        embeddings: &Tensor,
        num_quantizers: Option<usize>,
    ) -> candle_core::Result<Tensor> {
        let mut residual = if let Some(input_proj) = &self.input_proj {
            Convolution.forward_1d(input_proj, embeddings)?
        } else {
            embeddings.clone()
        };
        tracing::info!(
            "{}",
            tensor_debug_summary(&format!("Qwen3-TTS quantizer {label} input"), &residual,)?
        );
        if label == "acoustic" {
            tracing::info!(
                "{}",
                tensor_vec8_summary(
                    &format!("Qwen3-TTS quantizer {label} input frame1"),
                    &residual.i((0, ..8, 1))?,
                )?
            );
        }
        let n_q = num_quantizers
            .unwrap_or(self.layers.len())
            .min(self.layers.len());
        let mut all_indices = Vec::with_capacity(n_q);
        for (layer_idx, layer) in self.layers.iter().take(n_q).enumerate() {
            if label == "acoustic" && layer_idx == 2 {
                let frame0_top5 = layer
                    .codebook
                    .topk_for_vector(&residual.i((0, .., 0))?, 5)?;
                let frame1_top5 = layer
                    .codebook
                    .topk_for_vector(&residual.i((0, .., 1))?, 5)?;
                tracing::info!(
                    "Qwen3-TTS quantizer {label} layer {layer_idx} frame0 top5={frame0_top5:?}"
                );
                tracing::info!(
                    "Qwen3-TTS quantizer {label} layer {layer_idx} frame1 top5={frame1_top5:?}"
                );
                tracing::info!(
                    "{}",
                    tensor_vec8_summary(
                        &format!("Qwen3-TTS quantizer {label} layer {layer_idx} residual frame0"),
                        &residual.i((0, ..8, 0))?,
                    )?
                );
                tracing::info!(
                    "{}",
                    tensor_vec8_summary(
                        &format!("Qwen3-TTS quantizer {label} layer {layer_idx} residual frame1"),
                        &residual.i((0, ..8, 1))?,
                    )?
                );
            }
            let indices = layer.encode(&residual)?;
            let quantized = layer.decode(&indices)?;
            let preview = indices
                .i((0, ..3))?
                .to_dtype(DType::U32)?
                .to_vec1::<u32>()?;
            tracing::info!("Qwen3-TTS quantizer {label} layer {layer_idx} codes={preview:?}");
            if label == "acoustic" && layer_idx == 2 {
                let first_idx = indices
                    .i((0, 0))?
                    .to_dtype(DType::U32)?
                    .to_scalar::<u32>()? as usize;
                let code_vec = layer.codebook.embed()?.i((first_idx, ..8))?;
                let frame_vec = quantized.i((0, ..8, 0))?;
                tracing::info!(
                    "{}",
                    tensor_vec8_summary(
                        &format!("Qwen3-TTS quantizer {label} layer {layer_idx} codebook_row"),
                        &code_vec,
                    )?
                );
                tracing::info!(
                    "{}",
                    tensor_vec8_summary(
                        &format!("Qwen3-TTS quantizer {label} layer {layer_idx} frame0"),
                        &frame_vec,
                    )?
                );
                tracing::info!(
                    "{}",
                    tensor_vec8_summary(
                        &format!("Qwen3-TTS quantizer {label} layer {layer_idx} frame1"),
                        &quantized.i((0, ..8, 1))?,
                    )?
                );
            }
            tracing::info!(
                "{}",
                tensor_debug_summary(
                    &format!("Qwen3-TTS quantizer {label} layer {layer_idx} quantized"),
                    &quantized,
                )?
            );
            residual = (&residual - &quantized)?;
            tracing::info!(
                "{}",
                tensor_debug_summary(
                    &format!("Qwen3-TTS quantizer {label} layer {layer_idx} residual"),
                    &residual,
                )?
            );
            if label == "acoustic" && layer_idx == 4 {
                tracing::info!(
                    "{}",
                    tensor_vec8_summary(
                        &format!(
                            "Qwen3-TTS quantizer {label} layer {layer_idx} post-residual frame0"
                        ),
                        &residual.i((0, ..8, 0))?,
                    )?
                );
                tracing::info!(
                    "{}",
                    tensor_vec8_summary(
                        &format!(
                            "Qwen3-TTS quantizer {label} layer {layer_idx} post-residual frame1"
                        ),
                        &residual.i((0, ..8, 1))?,
                    )?
                );
            }
            all_indices.push(indices);
        }
        let refs = all_indices.iter().collect::<Vec<_>>();
        Tensor::stack(&refs, 0)
    }

    fn decode(&self, indices: &Tensor) -> candle_core::Result<Tensor> {
        let n_q = indices.dim(0)?.min(self.layers.len());
        let mut quantized = None;
        for (idx, layer) in self.layers.iter().take(n_q).enumerate() {
            let decoded = layer.decode(&indices.i(idx)?)?;
            quantized = Some(match quantized {
                Some(acc) => (acc + decoded)?,
                None => decoded,
            });
        }
        let quantized = quantized.ok_or_else(|| {
            candle_core::Error::Msg(
                "MimiResidualVectorQuantizer::decode got zero quantizers".into(),
            )
        })?;
        if let Some(output_proj) = &self.output_proj {
            Convolution.forward_1d(output_proj, &quantized)
        } else {
            Ok(quantized)
        }
    }
}

#[derive(Debug, Clone)]
struct MimiSplitResidualVectorQuantizer {
    max_num_quantizers: usize,
    num_semantic_quantizers: usize,
    semantic_residual_vector_quantizer: MimiResidualVectorQuantizer,
    acoustic_residual_vector_quantizer: MimiResidualVectorQuantizer,
}

impl MimiSplitResidualVectorQuantizer {
    fn new(
        cfg: &Qwen3TtsTokenizerEncoderConfig,
        runtime_dtype: DType,
        vb: candle_nn::VarBuilder<'_>,
    ) -> Result<Self> {
        Ok(Self {
            max_num_quantizers: cfg.num_quantizers,
            num_semantic_quantizers: cfg.num_semantic_quantizers,
            semantic_residual_vector_quantizer: MimiResidualVectorQuantizer::new(
                cfg,
                cfg.num_semantic_quantizers,
                runtime_dtype,
                vb.pp("semantic_residual_vector_quantizer"),
            )?,
            acoustic_residual_vector_quantizer: MimiResidualVectorQuantizer::new(
                cfg,
                cfg.num_quantizers - cfg.num_semantic_quantizers,
                runtime_dtype,
                vb.pp("acoustic_residual_vector_quantizer"),
            )?,
        })
    }

    fn encode(
        &self,
        embeddings: &Tensor,
        num_quantizers: Option<usize>,
    ) -> candle_core::Result<Tensor> {
        let num_quantizers = num_quantizers.unwrap_or(self.max_num_quantizers);
        let semantic = self
            .semantic_residual_vector_quantizer
            .encode("semantic", embeddings, None)?;
        if num_quantizers <= self.num_semantic_quantizers {
            return Ok(semantic);
        }
        let acoustic = self.acoustic_residual_vector_quantizer.encode(
            "acoustic",
            embeddings,
            Some(num_quantizers - self.num_semantic_quantizers),
        )?;
        Tensor::cat(&[&semantic, &acoustic], 0)
    }
}

#[derive(Debug)]
pub struct Qwen3TtsTokenizerEncoder {
    cfg: Qwen3TtsTokenizerEncoderConfig,
    encoder: MimiEncoder,
    encoder_transformer: Mutex<MimiTransformer>,
    downsample: Option<MimiConv1d>,
    quantizer: MimiSplitResidualVectorQuantizer,
    device: Device,
    runtime_dtype: DType,
}

impl Qwen3TtsTokenizerEncoder {
    pub fn new(
        cfg: &Qwen3TtsTokenizerEncoderConfig,
        vb: ShardedVarBuilder,
        speech_tokenizer_weights: &std::path::Path,
        _output_num_quantizers: usize,
    ) -> Result<Self> {
        let runtime_dtype = vb.device().bf16_default_to_f32();
        let raw_tensors = candle_core::safetensors::load(speech_tokenizer_weights, vb.device())?;
        let encoder_weights = raw_tensors
            .into_iter()
            .filter_map(|(key, tensor)| {
                key.strip_prefix("encoder.")
                    .map(|stripped| (stripped.to_string(), tensor))
            })
            .collect::<std::collections::HashMap<_, _>>();
        if encoder_weights.is_empty() {
            anyhow::bail!(
                "No Qwen3-TTS speech tokenizer encoder weights found under `encoder.*` in `{}`.",
                speech_tokenizer_weights.display()
            );
        }
        let module_weights = encoder_weights
            .iter()
            .filter(|(key, _)| !key.starts_with("quantizer."))
            .map(|(key, tensor)| (key.clone(), tensor.clone()))
            .collect::<std::collections::HashMap<_, _>>();
        let quantizer_weights = encoder_weights
            .iter()
            .filter(|(key, _)| key.starts_with("quantizer."))
            .map(|(key, tensor)| (key.clone(), tensor.clone()))
            .collect::<std::collections::HashMap<_, _>>();
        let module_vb =
            candle_nn::VarBuilder::from_tensors(module_weights, runtime_dtype, vb.device());
        let quantizer_vb =
            candle_nn::VarBuilder::from_tensors(quantizer_weights, runtime_dtype, vb.device());
        let encoder = MimiEncoder::new(cfg, module_vb.pp("encoder"))?;
        let encoder_transformer = MimiTransformer::new(cfg, module_vb.pp("encoder_transformer"))?;
        let encodec_frame_rate =
            cfg.sampling_rate as f64 / cfg.upsampling_ratios.iter().product::<usize>() as f64;
        let downsample = if (encodec_frame_rate - cfg.frame_rate).abs() > f64::EPSILON {
            Some(MimiConv1d::new(
                cfg,
                cfg.hidden_size,
                cfg.hidden_size,
                2 * (encodec_frame_rate / cfg.frame_rate) as usize,
                2,
                1,
                1,
                Some("replicate"),
                false,
                Some(encoder.collect_cache_specs().len()),
                module_vb.pp("downsample"),
            )?)
        } else {
            None
        };
        let quantizer = MimiSplitResidualVectorQuantizer::new(
            cfg,
            runtime_dtype,
            quantizer_vb.pp("quantizer"),
        )?;
        Ok(Self {
            cfg: cfg.clone(),
            encoder,
            encoder_transformer: Mutex::new(encoder_transformer),
            downsample,
            quantizer,
            device: vb.device().clone(),
            runtime_dtype,
        })
    }

    pub fn encode_ref_codes(
        &self,
        waveform: &Tensor,
        _encode_downsample_rate: usize,
        encoder_valid_num_quantizers: usize,
    ) -> candle_core::Result<Vec<Vec<u32>>> {
        let waveform = waveform
            .to_device(&self.device)?
            .to_dtype(self.runtime_dtype)?;
        tracing::info!(
            "{}",
            tensor_debug_summary("Qwen3-TTS encoder stage wave", &waveform)?
        );
        // Python's tokenizer encode path constructs and passes a causal padding cache even for
        // the non-streaming ref-audio encode path. This matters for odd-length downsampling.
        let encoder_cache_specs = self.encoder.collect_cache_specs();
        let mut per_layer_padding = encoder_cache_specs
            .iter()
            .map(|(padding, _, _)| *padding)
            .collect::<Vec<_>>();
        let mut per_layer_padding_mode = encoder_cache_specs
            .iter()
            .map(|(_, mode, _)| mode.clone())
            .collect::<Vec<_>>();
        let mut per_layer_in_channels = encoder_cache_specs
            .iter()
            .map(|(_, _, channels)| *channels)
            .collect::<Vec<_>>();
        if let Some(downsample) = &self.downsample {
            per_layer_padding.push(downsample.padding_total);
            per_layer_padding_mode.push(downsample.pad_mode.clone());
            per_layer_in_channels.push(downsample.in_channels);
        }
        let mut padding_cache = Some(MimiConv1dPaddingCache::new(
            per_layer_padding,
            per_layer_padding_mode,
            per_layer_in_channels,
        )?);
        let embeddings = self
            .encoder
            .forward_with_cache(&waveform, padding_cache.as_mut())?;
        tracing::info!(
            "{}",
            tensor_debug_summary("Qwen3-TTS encoder stage enc", &embeddings)?
        );
        tracing::info!(
            "{}",
            tensor_vec8_summary(
                "Qwen3-TTS encoder stage enc frame1",
                &embeddings.i((0, ..8, 1))?
            )?
        );
        let embeddings = embeddings.transpose(1, 2)?;
        let embeddings = self
            .encoder_transformer
            .lock()
            .unwrap()
            .forward(&embeddings)?;
        let mut embeddings = embeddings.transpose(1, 2)?;
        tracing::info!(
            "{}",
            tensor_debug_summary("Qwen3-TTS encoder stage xf", &embeddings)?
        );
        tracing::info!(
            "{}",
            tensor_vec8_summary(
                "Qwen3-TTS encoder stage xf frame1",
                &embeddings.i((0, ..8, 1))?
            )?
        );
        let mut frame_len = self.encoder.get_output_length(waveform.dim(D::Minus1)?);
        if let Some(downsample) = &self.downsample {
            embeddings = downsample.forward_with_cache(&embeddings, padding_cache.as_mut())?;
            frame_len = downsample.get_output_length(frame_len);
        }
        tracing::info!(
            "{}",
            tensor_debug_summary("Qwen3-TTS encoder stage down", &embeddings)?
        );
        tracing::info!(
            "{}",
            tensor_vec8_summary(
                "Qwen3-TTS encoder stage down frame1",
                &embeddings.i((0, ..8, 1))?
            )?
        );
        let mut codes = self
            .quantizer
            .encode(&embeddings, Some(encoder_valid_num_quantizers))?
            .permute((1, 0, 2))?;
        if encoder_valid_num_quantizers < codes.dim(1)? {
            codes = codes.i((.., ..encoder_valid_num_quantizers, ..))?;
        }
        let frame_len = frame_len.min(codes.dim(2)?).max(1);
        let frame_len = frame_len.min(codes.dim(2)?);
        let codes = codes
            .i((0, .., ..frame_len))?
            .transpose(0, 1)?
            .to_dtype(DType::U32)?;
        let rows = codes.to_vec2::<u32>()?;
        Ok(rows)
    }
}
