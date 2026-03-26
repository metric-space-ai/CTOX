#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    dead_code
)]

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, Module};
use mistralrs_quant::ShardedVarBuilder;

use crate::layers::{conv1d, ReflectionPad1d};

use super::Qwen3TtsSpeakerEncoderConfig;

fn tensor_stats_preview(tensor: &Tensor, label: &str) -> Result<()> {
    let flat = tensor.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    if flat.is_empty() {
        tracing::info!("Qwen3-TTS {label} stats empty");
        return Ok(());
    }
    let len = flat.len() as f64;
    let mean = flat.iter().map(|&v| f64::from(v)).sum::<f64>() / len;
    let variance = flat
        .iter()
        .map(|&v| {
            let diff = f64::from(v) - mean;
            diff * diff
        })
        .sum::<f64>()
        / len;
    let head = flat.iter().take(8).copied().collect::<Vec<_>>();
    tracing::info!(
        "Qwen3-TTS {label} stats mean={:.6} std={:.6} head8={:?}",
        mean,
        variance.sqrt(),
        head
    );
    Ok(())
}

struct TimeDelayNetBlock {
    pad: ReflectionPad1d,
    conv: Conv1d,
}

impl TimeDelayNetBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let pad = ((kernel_size - 1) * dilation) / 2;
        let conv = conv1d(
            in_channels,
            out_channels,
            kernel_size,
            Conv1dConfig {
                padding: 0,
                stride: 1,
                dilation,
                groups: 1,
                ..Default::default()
            },
            vb.pp("conv"),
        )?;
        Ok(Self {
            pad: ReflectionPad1d::new((pad, pad)),
            conv,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.conv.forward(&self.pad.forward(hidden_states)?)?.relu()
    }
}

struct Res2NetBlock {
    blocks: Vec<TimeDelayNetBlock>,
    scale: usize,
}

impl Res2NetBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        scale: usize,
        kernel_size: usize,
        dilation: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let in_channel = in_channels / scale;
        let hidden_channel = out_channels / scale;
        let mut blocks = Vec::with_capacity(scale.saturating_sub(1));
        for i in 0..scale.saturating_sub(1) {
            blocks.push(TimeDelayNetBlock::new(
                in_channel,
                hidden_channel,
                kernel_size,
                dilation,
                vb.pp(format!("blocks.{i}")),
            )?);
        }
        Ok(Self { blocks, scale })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut outputs = Vec::with_capacity(self.scale);
        let mut previous_output: Option<Tensor> = None;
        let split_size = hidden_states.dim(1)? / self.scale;
        for i in 0..self.scale {
            let hidden_part = hidden_states.narrow(1, i * split_size, split_size)?;
            let output_part = if i == 0 {
                hidden_part
            } else if i == 1 {
                self.blocks[i - 1].forward(&hidden_part)?
            } else {
                self.blocks[i - 1].forward(&(hidden_part + previous_output.as_ref().unwrap())?)?
            };
            previous_output = Some(output_part.clone());
            outputs.push(output_part);
        }
        Tensor::cat(&outputs, 1)
    }
}

struct SqueezeExcitationBlock {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl SqueezeExcitationBlock {
    fn new(
        in_channels: usize,
        se_channels: usize,
        out_channels: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        Ok(Self {
            conv1: conv1d(in_channels, se_channels, 1, cfg, vb.pp("conv1"))?,
            conv2: conv1d(se_channels, out_channels, 1, cfg, vb.pp("conv2"))?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states_mean = hidden_states.mean_keepdim(2)?;
        let hidden_states_mean = self.conv1.forward(&hidden_states_mean)?.relu()?;
        let hidden_states_mean =
            candle_nn::ops::sigmoid(&self.conv2.forward(&hidden_states_mean)?)?;
        hidden_states.broadcast_mul(&hidden_states_mean)
    }
}

struct AttentiveStatisticsPooling {
    eps: f64,
    tdnn: TimeDelayNetBlock,
    conv: Conv1d,
}

impl AttentiveStatisticsPooling {
    fn new(channels: usize, attention_channels: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        Ok(Self {
            eps: 1e-12,
            tdnn: TimeDelayNetBlock::new(channels * 3, attention_channels, 1, 1, vb.pp("tdnn"))?,
            conv: conv1d(attention_channels, channels, 1, cfg, vb.pp("conv"))?,
        })
    }

    fn compute_statistics(&self, x: &Tensor, weights: &Tensor) -> Result<(Tensor, Tensor)> {
        let mean = weights.broadcast_mul(x)?.sum(2)?;
        let centered = x.broadcast_sub(&mean.unsqueeze(2)?)?;
        let eps = Tensor::new(self.eps as f32, x.device())?.to_dtype(x.dtype())?;
        let std = weights
            .broadcast_mul(&centered.sqr()?)?
            .sum(2)?
            .broadcast_maximum(&eps)?
            .sqrt()?;
        Ok((mean, std))
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch_size, _channels, seq_length) = hidden_states.dims3()?;
        let device = hidden_states.device();
        let mask = Tensor::ones((batch_size, 1, seq_length), hidden_states.dtype(), device)?;
        let total = mask.sum_keepdim(2)?;

        let (mean, std) = self.compute_statistics(hidden_states, &mask.broadcast_div(&total)?)?;
        let mean = mean.unsqueeze(2)?.repeat((1, 1, seq_length))?;
        let std = std.unsqueeze(2)?.repeat((1, 1, seq_length))?;
        let attention = Tensor::cat(&[hidden_states, &mean, &std], 1)?;
        let attention = self.conv.forward(&self.tdnn.forward(&attention)?.tanh()?)?;
        let attention = candle_nn::ops::softmax(&attention, D::Minus1)?;
        let (mean, std) = self.compute_statistics(hidden_states, &attention)?;
        Tensor::cat(&[&mean, &std], 1)?.unsqueeze(2)
    }
}

struct SqueezeExcitationRes2NetBlock {
    tdnn1: TimeDelayNetBlock,
    res2net_block: Res2NetBlock,
    tdnn2: TimeDelayNetBlock,
    se_block: SqueezeExcitationBlock,
}

impl SqueezeExcitationRes2NetBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        res2net_scale: usize,
        se_channels: usize,
        kernel_size: usize,
        dilation: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            tdnn1: TimeDelayNetBlock::new(in_channels, out_channels, 1, 1, vb.pp("tdnn1"))?,
            res2net_block: Res2NetBlock::new(
                out_channels,
                out_channels,
                res2net_scale,
                kernel_size,
                dilation,
                vb.pp("res2net_block"),
            )?,
            tdnn2: TimeDelayNetBlock::new(out_channels, out_channels, 1, 1, vb.pp("tdnn2"))?,
            se_block: SqueezeExcitationBlock::new(
                out_channels,
                se_channels,
                out_channels,
                vb.pp("se_block"),
            )?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let hidden_states = self.tdnn1.forward(hidden_states)?;
        let hidden_states = self.res2net_block.forward(&hidden_states)?;
        let hidden_states = self.tdnn2.forward(&hidden_states)?;
        let hidden_states = self.se_block.forward(&hidden_states)?;
        residual.broadcast_add(&hidden_states)
    }
}

pub struct Qwen3TtsSpeakerEncoder {
    blocks: Vec<SpeakerBlock>,
    mfa: TimeDelayNetBlock,
    asp: AttentiveStatisticsPooling,
    fc: Conv1d,
    device: Device,
}

enum SpeakerBlock {
    Tdnn(TimeDelayNetBlock),
    SeRes2Net(SqueezeExcitationRes2NetBlock),
}

impl SpeakerBlock {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        match self {
            Self::Tdnn(block) => block.forward(hidden_states),
            Self::SeRes2Net(block) => block.forward(hidden_states),
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Self::Tdnn(block) => block.conv.weight().dtype(),
            Self::SeRes2Net(block) => block.tdnn1.conv.weight().dtype(),
        }
    }
}

impl Qwen3TtsSpeakerEncoder {
    pub fn new(cfg: &Qwen3TtsSpeakerEncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        if cfg.enc_channels.len() < 2 {
            candle_core::bail!(
                "Qwen3-TTS speaker encoder expects at least two encoder channel stages."
            );
        }
        if cfg.enc_channels.len() != cfg.enc_kernel_sizes.len()
            || cfg.enc_channels.len() != cfg.enc_dilations.len()
        {
            candle_core::bail!(
                "Qwen3-TTS speaker encoder expects enc_channels, enc_kernel_sizes and enc_dilations to have the same length."
            );
        }

        let mut blocks = Vec::with_capacity(cfg.enc_channels.len().saturating_sub(1));
        blocks.push(SpeakerBlock::Tdnn(TimeDelayNetBlock::new(
            cfg.mel_dim,
            cfg.enc_channels[0],
            cfg.enc_kernel_sizes[0],
            cfg.enc_dilations[0],
            vb.pp("blocks.0"),
        )?));
        for i in 1..cfg.enc_channels.len() - 1 {
            blocks.push(SpeakerBlock::SeRes2Net(SqueezeExcitationRes2NetBlock::new(
                cfg.enc_channels[i - 1],
                cfg.enc_channels[i],
                cfg.enc_res2net_scale,
                cfg.enc_se_channels,
                cfg.enc_kernel_sizes[i],
                cfg.enc_dilations[i],
                vb.pp(format!("blocks.{i}")),
            )?));
        }

        let conv_cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        Ok(Self {
            blocks,
            mfa: TimeDelayNetBlock::new(
                cfg.enc_channels[cfg.enc_channels.len() - 1],
                cfg.enc_channels[cfg.enc_channels.len() - 1],
                cfg.enc_kernel_sizes[cfg.enc_kernel_sizes.len() - 1],
                cfg.enc_dilations[cfg.enc_dilations.len() - 1],
                vb.pp("mfa"),
            )?,
            asp: AttentiveStatisticsPooling::new(
                cfg.enc_channels[cfg.enc_channels.len() - 1],
                cfg.enc_attention_channels,
                vb.pp("asp"),
            )?,
            fc: conv1d(
                cfg.enc_channels[cfg.enc_channels.len() - 1] * 2,
                cfg.enc_dim,
                1,
                conv_cfg,
                vb.pp("fc"),
            )?,
            device: vb.device().clone(),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let input_dtype = self
            .blocks
            .first()
            .map(SpeakerBlock::dtype)
            .unwrap_or_else(|| self.fc.weight().dtype());
        let mut hidden_states = hidden_states
            .transpose(1, 2)?
            .to_dtype(input_dtype)?;
        tensor_stats_preview(&hidden_states, "speaker.input")?;
        let mut hidden_states_list = Vec::with_capacity(self.blocks.len());
        for (idx, block) in self.blocks.iter().enumerate() {
            hidden_states = block.forward(&hidden_states)?;
            tensor_stats_preview(&hidden_states, &format!("speaker.block{idx}"))?;
            hidden_states_list.push(hidden_states.clone());
        }
        let aggregated = Tensor::cat(&hidden_states_list[1..], 1)?;
        tensor_stats_preview(&aggregated, "speaker.aggregated")?;
        let hidden_states = self.mfa.forward(&aggregated)?;
        tensor_stats_preview(&hidden_states, "speaker.mfa")?;
        let hidden_states = self.asp.forward(&hidden_states)?;
        tensor_stats_preview(&hidden_states, "speaker.asp")?;
        let hidden_states = self.fc.forward(&hidden_states)?;
        tensor_stats_preview(&hidden_states, "speaker.fc")?;
        hidden_states.squeeze(2)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
