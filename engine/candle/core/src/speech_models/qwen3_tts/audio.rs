#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    dead_code
)]

use anyhow::Result;
use candle_core::{Device, Tensor};
use mistralrs_audio::AudioInput;
use rustfft::{num_complex::Complex32, FftPlanner};
use std::f32::consts::PI;

use super::Qwen3TtsSpeakerEncoderConfig;

const N_FFT: usize = 1024;
const HOP_SIZE: usize = 256;
const WIN_SIZE: usize = 1024;
const F_MIN: f32 = 0.0;

fn vector_stats_preview(values: &[f32], preview: usize) -> (f32, f32, Vec<f32>) {
    if values.is_empty() {
        return (0.0, 0.0, Vec::new());
    }
    let len = values.len() as f64;
    let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / len;
    let variance = values
        .iter()
        .map(|&v| {
            let diff = f64::from(v) - mean;
            diff * diff
        })
        .sum::<f64>()
        / len;
    let head = values.iter().take(preview).copied().collect::<Vec<_>>();
    (mean as f32, variance.sqrt() as f32, head)
}

pub struct Qwen3TtsAudioProcessor {
    sampling_rate: usize,
    num_mel_bins: usize,
}

impl Qwen3TtsAudioProcessor {
    pub fn new(cfg: &Qwen3TtsSpeakerEncoderConfig) -> Self {
        Self {
            sampling_rate: cfg.sample_rate,
            num_mel_bins: cfg.mel_dim,
        }
    }

    pub fn process_audio(&self, audio: &AudioInput, device: &Device) -> Result<Tensor> {
        let mono = audio.to_mono();
        let samples = if audio.sample_rate as usize != self.sampling_rate {
            self.resample(&mono, audio.sample_rate as usize, self.sampling_rate)?
        } else {
            mono
        };
        self.process_samples(&samples, device)
    }

    pub fn prepare_waveform(&self, audio: &AudioInput, device: &Device) -> Result<Tensor> {
        let mono = audio.to_mono();
        let samples = if audio.sample_rate as usize != self.sampling_rate {
            self.resample(&mono, audio.sample_rate as usize, self.sampling_rate)?
        } else {
            mono
        };
        let len = samples.len();
        Ok(Tensor::from_vec(samples, (1, 1, len), device)?)
    }

    pub fn process_samples(&self, samples: &[f32], device: &Device) -> Result<Tensor> {
        let mel = self.compute_mel_spectrogram(samples)?;
        let num_frames = mel.len();
        if num_frames == 0 {
            anyhow::bail!("Audio too short to produce mel frames");
        }
        let flat = mel.iter().flatten().copied().collect::<Vec<_>>();
        let first_frame = mel.first().cloned().unwrap_or_default();
        let (mel_mean, mel_std, mel_head) = vector_stats_preview(&flat, 8);
        let (_frame_mean, _frame_std, first_frame_head) = vector_stats_preview(&first_frame, 8);
        tracing::info!(
            "Qwen3-TTS mel stats frames={} bins={} mean={:.6} std={:.6} first_frame_head8={:?} flat_head8={:?}",
            num_frames,
            self.num_mel_bins,
            mel_mean,
            mel_std,
            first_frame_head,
            mel_head
        );
        let data = mel.into_iter().flatten().collect::<Vec<_>>();
        Ok(Tensor::from_vec(
            data,
            (1, num_frames, self.num_mel_bins),
            device,
        )?)
    }

    fn resample(&self, samples: &[f32], from_rate: usize, to_rate: usize) -> Result<Vec<f32>> {
        if from_rate == to_rate {
            return Ok(samples.to_vec());
        }
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let gcd = Self::gcd(from_rate, to_rate);
        let orig = from_rate / gcd;
        let new = to_rate / gcd;
        let lowpass_filter_width = 6.0f32;
        let rolloff = 0.99f32;
        let base_freq = (orig.min(new) as f32) * rolloff;
        let width = (lowpass_filter_width * orig as f32 / base_freq).ceil() as usize;
        let kernel_len = width * 2 + orig;

        let scale = base_freq / orig as f32;
        let mut kernels = vec![vec![0.0f32; kernel_len]; new];
        for (phase, kernel) in kernels.iter_mut().enumerate() {
            for (offset, weight) in kernel.iter_mut().enumerate() {
                let idx = (offset as isize - width as isize) as f32 / orig as f32;
                let mut t = -(phase as f32) / new as f32 + idx;
                t *= base_freq;
                t = t.clamp(-lowpass_filter_width, lowpass_filter_width);
                let window = (t * PI / lowpass_filter_width / 2.0).cos().powi(2);
                let t_pi = t * PI;
                let sinc = if t_pi == 0.0 { 1.0 } else { t_pi.sin() / t_pi };
                *weight = sinc * window * scale;
            }
        }

        let mut padded = vec![0.0f32; width + samples.len() + width + orig];
        padded[width..width + samples.len()].copy_from_slice(samples);

        let num_steps = (padded.len() - kernel_len) / orig + 1;
        let target_len = (new * samples.len()).div_ceil(orig);
        let mut out = Vec::with_capacity(num_steps * new);
        for step in 0..num_steps {
            let start = step * orig;
            let frame = &padded[start..start + kernel_len];
            for kernel in &kernels {
                let value = frame
                    .iter()
                    .zip(kernel.iter())
                    .map(|(&sample, &weight)| f64::from(sample) * f64::from(weight))
                    .sum::<f64>() as f32;
                out.push(value);
            }
        }
        out.truncate(target_len);
        Ok(out)
    }

    fn gcd(mut a: usize, mut b: usize) -> usize {
        while b != 0 {
            let tmp = a % b;
            a = b;
            b = tmp;
        }
        a.max(1)
    }

    fn compute_mel_spectrogram(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let n_fft = N_FFT;
        let hop = HOP_SIZE;
        let pad = (n_fft - hop) / 2;
        let n_freqs = n_fft / 2 + 1;

        let mut padded = vec![0.0f32; pad + samples.len() + pad];
        for (i, p) in padded.iter_mut().enumerate().take(pad) {
            let src_idx = (pad - i).min(samples.len().saturating_sub(1));
            *p = samples[src_idx];
        }
        padded[pad..pad + samples.len()].copy_from_slice(samples);
        for i in 0..pad {
            let src_idx = samples.len().saturating_sub(2 + i);
            padded[pad + samples.len() + i] = samples[src_idx];
        }

        let total_frames = (padded.len().saturating_sub(n_fft)) / hop + 1;
        if total_frames == 0 {
            return Ok(Vec::new());
        }

        let window = (0..WIN_SIZE)
            .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / WIN_SIZE as f32).cos()))
            .collect::<Vec<_>>();
        let mel_filters = self.create_mel_filterbank(n_fft)?;
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);

        let mut mel_features = Vec::with_capacity(total_frames);
        for frame_idx in 0..total_frames {
            let start = frame_idx * hop;
            let mut buf: Vec<Complex32> = padded[start..start + n_fft]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| Complex32::new(s * w, 0.0))
                .collect();
            fft.process(&mut buf);

            let magnitude: Vec<f32> = buf[..n_freqs]
                .iter()
                .map(|c| (c.norm_sqr() + 1e-9).sqrt())
                .collect();

            let mut mel_frame = vec![0.0f32; self.num_mel_bins];
            for (mel_idx, filter) in mel_filters.iter().enumerate() {
                let mut sum = 0.0f32;
                for (freq_idx, &coeff) in filter.iter().enumerate() {
                    if freq_idx < magnitude.len() {
                        sum += magnitude[freq_idx] * coeff;
                    }
                }
                mel_frame[mel_idx] = sum.max(1e-5).ln();
            }
            mel_features.push(mel_frame);
        }

        Ok(mel_features)
    }

    fn create_mel_filterbank(&self, n_fft: usize) -> Result<Vec<Vec<f32>>> {
        let n_freqs = n_fft / 2 + 1;
        let sr = self.sampling_rate as f32;
        let mel_min = Self::hertz_to_mel(F_MIN);
        let mel_max = Self::hertz_to_mel(sr / 2.0);

        let fft_freqs: Vec<f32> = (0..n_freqs)
            .map(|i| i as f32 * (sr / 2.0) / (n_freqs - 1) as f32)
            .collect();
        let filter_freqs: Vec<f32> = (0..self.num_mel_bins + 2)
            .map(|i| {
                let mel = mel_min + (mel_max - mel_min) * i as f32 / (self.num_mel_bins + 1) as f32;
                Self::mel_to_hertz(mel)
            })
            .collect();
        let filter_diff: Vec<f32> = filter_freqs.windows(2).map(|w| w[1] - w[0]).collect();

        let mut filterbank = vec![vec![0.0f32; n_freqs]; self.num_mel_bins];
        for m in 0..self.num_mel_bins {
            for (j, &fft_f) in fft_freqs.iter().enumerate() {
                let down = (fft_f - filter_freqs[m]) / filter_diff[m];
                let up = (filter_freqs[m + 2] - fft_f) / filter_diff[m + 1];
                filterbank[m][j] = 0.0f32.max(down.min(up));
            }
        }
        for m in 0..self.num_mel_bins {
            let enorm = 2.0 / (filter_freqs[m + 2] - filter_freqs[m]);
            for value in &mut filterbank[m] {
                *value *= enorm;
            }
        }
        Ok(filterbank)
    }

    fn hertz_to_mel(freq: f32) -> f32 {
        const MIN_LOG_HERTZ: f32 = 1000.0;
        const MIN_LOG_MEL: f32 = 15.0;
        const LOGSTEP: f32 = 27.0 / 1.856_298;
        if freq >= MIN_LOG_HERTZ {
            MIN_LOG_MEL + (freq / MIN_LOG_HERTZ).ln() * LOGSTEP
        } else {
            3.0 * freq / 200.0
        }
    }

    fn mel_to_hertz(mel: f32) -> f32 {
        const MIN_LOG_HERTZ: f32 = 1000.0;
        const MIN_LOG_MEL: f32 = 15.0;
        const LOGSTEP: f32 = 1.856_298 / 27.0;
        if mel >= MIN_LOG_MEL {
            MIN_LOG_HERTZ * (LOGSTEP * (mel - MIN_LOG_MEL)).exp()
        } else {
            200.0 * mel / 3.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_non_empty_mel_spectrogram() {
        let cfg = Qwen3TtsSpeakerEncoderConfig {
            mel_dim: 128,
            enc_dim: 1024,
            enc_channels: vec![512, 512, 512, 512, 1536],
            enc_kernel_sizes: vec![5, 3, 3, 3, 1],
            enc_dilations: vec![1, 2, 3, 4, 1],
            enc_attention_channels: 128,
            enc_res2net_scale: 8,
            enc_se_channels: 128,
            sample_rate: 24_000,
        };
        let processor = Qwen3TtsAudioProcessor::new(&cfg);
        let device = Device::Cpu;
        let samples = vec![0.1f32; 24_000];
        let mel = processor.process_samples(&samples, &device).unwrap();
        assert_eq!(mel.dims3().unwrap().0, 1);
        assert_eq!(mel.dims3().unwrap().2, 128);
        assert!(mel.dims3().unwrap().1 > 0);
    }
}
