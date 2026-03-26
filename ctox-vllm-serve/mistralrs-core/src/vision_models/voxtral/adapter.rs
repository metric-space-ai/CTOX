#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{DType, Module, Result, Tensor};
use candle_nn::Linear;
use mistralrs_quant::ShardedVarBuilder;

/// Temporal adapter that performs 4x downsampling via reshape + MLP.
///
/// Input: [B, T, encoder_dim] (e.g., [B, T, 1280])
/// Reshape: [B, T/4, encoder_dim*4] (e.g., [B, T/4, 5120])
/// Output: [B, T/4, decoder_dim] (e.g., [B, T/4, 3072])
pub struct VoxtralTemporalAdapter {
    pub(super) w_in: Linear,
    pub(super) w_out: Linear,
    downsample_factor: usize,
}

impl VoxtralTemporalAdapter {
    pub fn new(
        encoder_dim: usize,
        decoder_dim: usize,
        downsample_factor: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let in_features = encoder_dim * downsample_factor;
        let vb_in = vb.pp("audio_language_projection").pp("0");
        let w_in = Linear::new(
            vb_in
                .get((decoder_dim, in_features), "weight")?
                .to_dtype(DType::F32)?,
            None,
        );
        let vb_out = vb.pp("audio_language_projection").pp("2");
        let w_out = Linear::new(
            vb_out
                .get((decoder_dim, decoder_dim), "weight")?
                .to_dtype(DType::F32)?,
            None,
        );
        Ok(Self {
            w_in,
            w_out,
            downsample_factor,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, d) = xs.dims3()?;
        let t_trunc = t - (t % self.downsample_factor);
        let xs = if t_trunc < t {
            xs.narrow(1, 0, t_trunc)?
        } else {
            xs.clone()
        };
        let t_new = t_trunc / self.downsample_factor;
        let xs = xs.reshape((b, t_new, d * self.downsample_factor))?;
        let xs = if xs.dtype() == DType::F32 {
            xs
        } else {
            xs.to_dtype(DType::F32)?
        };
        let xs = self
            .w_in
            .forward(&xs)
            .map_err(|e| candle_core::Error::Msg(format!("voxtral adapter w_in matmul: {e}")))?;
        let xs = if xs.dtype() == DType::F32 {
            xs
        } else {
            xs.to_dtype(DType::F32)?
        };
        let xs = xs
            .gelu_erf()
            .map_err(|e| candle_core::Error::Msg(format!("voxtral adapter gelu: {e}")))?;
        let xs = self
            .w_out
            .forward(&xs)
            .map_err(|e| candle_core::Error::Msg(format!("voxtral adapter w_out matmul: {e}")))?;
        if xs.dtype() == DType::F32 {
            Ok(xs)
        } else {
            xs.to_dtype(DType::F32)
        }
    }
}
