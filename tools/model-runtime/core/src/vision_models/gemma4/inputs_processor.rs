#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{any::Any, sync::Arc};

use candle_core::{Device, Result, Tensor};
use engine_vision::{ApplyTransforms, Rescale, ToTensorNoNorm, Transforms};
use image::{DynamicImage, GenericImageView};
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    sequence::{build_mm_features_from_ranges, find_image_placeholder_ranges, Sequence},
    vision_models::{
        image_processor::{ImagePreProcessor, PreprocessedImages},
        preprocessor_config::{PreProcessorConfig, ToFilter},
        processor_config::ProcessorConfig,
        ModelInputs,
    },
};

use super::Gemma4SpecificArgs;

const IMAGE_TOKEN: &str = "<|image|>";
const BOI_TOKEN: &str = "<|image>";
const EOI_TOKEN: &str = "<image|>";
pub const IMAGE_TOKEN_ID: u32 = 258880;

pub struct Gemma4Processor {
    patch_size: usize,
    pooling_kernel_size: usize,
    default_output_length: usize,
    max_patches: usize,
    supports_images: bool,
}

impl Gemma4Processor {
    pub fn new(
        _processor_config: ProcessorConfig,
        patch_size: usize,
        pooling_kernel_size: usize,
        default_output_length: usize,
        supports_images: bool,
    ) -> Self {
        let max_patches = default_output_length * pooling_kernel_size * pooling_kernel_size;

        Self {
            patch_size,
            pooling_kernel_size,
            default_output_length,
            max_patches,
            supports_images,
        }
    }
}

impl Processor for Gemma4Processor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Gemma4ImageProcessor {
            patch_size: self.patch_size,
            pooling_kernel_size: self.pooling_kernel_size,
            default_output_length: self.default_output_length,
            max_patches: self.max_patches,
            supports_images: self.supports_images,
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[IMAGE_TOKEN, BOI_TOKEN, EOI_TOKEN]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

struct Gemma4ImageProcessor {
    patch_size: usize,
    pooling_kernel_size: usize,
    default_output_length: usize,
    max_patches: usize,
    supports_images: bool,
}

impl Gemma4ImageProcessor {
    fn output_tokens_for_size(&self, new_h: usize, new_w: usize) -> usize {
        let ph = new_h / self.patch_size;
        let pw = new_w / self.patch_size;
        let pool_area = self.pooling_kernel_size * self.pooling_kernel_size;
        (ph * pw) / pool_area
    }

    fn compute_resize_dims(&self, orig_h: usize, orig_w: usize) -> Result<(usize, usize)> {
        if orig_h == 0 || orig_w == 0 {
            candle_core::bail!(
                "Gemma4 image resize: input dimensions must be non-zero, got {orig_h}x{orig_w}"
            );
        }

        let target_px = self.max_patches * self.patch_size * self.patch_size;
        let grid_unit = self.pooling_kernel_size * self.patch_size;
        let pool_area = self.pooling_kernel_size * self.pooling_kernel_size;
        let max_side_length = (self.max_patches / pool_area) * grid_unit;

        let factor = (target_px as f64 / (orig_h as f64 * orig_w as f64)).sqrt();
        let ideal_h = orig_h as f64 * factor;
        let ideal_w = orig_w as f64 * factor;

        let mut new_h = (ideal_h / grid_unit as f64).floor() as usize * grid_unit;
        let mut new_w = (ideal_w / grid_unit as f64).floor() as usize * grid_unit;

        if new_h == 0 && new_w == 0 {
            candle_core::bail!(
                "Gemma4 image resize: both dimensions round to 0 for input {orig_h}x{orig_w}"
            );
        }

        if new_h == 0 {
            new_h = grid_unit;
            new_w = (((orig_w as f64 / orig_h as f64) * grid_unit as f64) as usize)
                .min(max_side_length);
            new_w = (new_w / grid_unit).max(1) * grid_unit;
        } else if new_w == 0 {
            new_w = grid_unit;
            new_h = (((orig_h as f64 / orig_w as f64) * grid_unit as f64) as usize)
                .min(max_side_length);
            new_h = (new_h / grid_unit).max(1) * grid_unit;
        }

        if new_h * new_w > target_px {
            candle_core::bail!(
                "Gemma4 image resize: {new_h}x{new_w} = {} pixels exceeds patch budget of {target_px} \
                 for input {orig_h}x{orig_w}",
                new_h * new_w
            );
        }

        Ok((new_h, new_w))
    }

    fn build_image_sequence(&self, num_tokens: usize) -> String {
        let image_tokens = vec![IMAGE_TOKEN.to_string(); num_tokens].join("");
        format!("{BOI_TOKEN}{image_tokens}{EOI_TOKEN}")
    }
}

fn cached_tokens_for_ranges(prefix_len: usize, ranges: &[(usize, usize)]) -> Vec<usize> {
    ranges
        .iter()
        .map(|&(offset, length)| prefix_len.saturating_sub(offset).min(length))
        .collect()
}

impl InputsProcessor for Gemma4ImageProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }

    fn process_inputs(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        is_xlora: bool,
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> anyhow::Result<InputProcessorOutput> {
        if is_xlora {
            return Err(anyhow::Error::msg(
                "Cannot make inputs for X-LoRA vision model.",
            ));
        }
        if no_kv_cache {
            return Err(anyhow::Error::msg("Vision model must have kv cache."));
        }
        let Some(tokenizer) = tokenizer else {
            return Err(anyhow::Error::msg(
                "Gemma4ImageProcessor requires a specified tokenizer.",
            ));
        };

        let config = other_config.expect("Need a PreProcessorConfig config.");
        let preprocessor_config: &PreProcessorConfig =
            config.downcast_ref().expect("Downcast failed.");

        let has_images = input_seqs.iter().any(|seq| seq.has_images());
        let mut image_hashes_accum = Vec::new();
        let mut image_cached_tokens_accum = Vec::new();
        let mut image_sizes_accum = Vec::new();

        let pixel_values = if has_images {
            if !self.supports_images {
                return Err(anyhow::Error::msg(
                    "This image processor does not support images.",
                ));
            }

            let mut pixel_values_accum = Vec::new();
            for seq in input_seqs.iter_mut() {
                let Some(images) = seq.take_images() else {
                    continue;
                };

                let per_image_dims = images
                    .iter()
                    .map(|img| {
                        let (w, h) = img.dimensions();
                        self.compute_resize_dims(h as usize, w as usize)
                    })
                    .collect::<Result<Vec<_>>>()?;

                let PreprocessedImages {
                    pixel_values,
                    pixel_attention_mask: _,
                    image_sizes: _,
                    num_img_tokens: _,
                    aspect_ratio_ids: _,
                    aspect_ratio_mask: _,
                    num_tiles: _,
                    image_grid_thw: _,
                    video_grid_thw: _,
                    rows: _,
                    cols: _,
                    pixel_values_list: _,
                    tgt_sizes: _,
                    image_sizes_all,
                    num_crops: _,
                } = self.preprocess(
                    images,
                    vec![],
                    preprocessor_config,
                    device,
                    (usize::MAX, usize::MAX),
                )?;

                if !seq.multimodal.has_changed_prompt {
                    let mut prompt = tokenizer
                        .decode(seq.get_toks(), false)
                        .expect("Detokenization failed!");
                    let positions: Vec<usize> = prompt
                        .match_indices(IMAGE_TOKEN)
                        .map(|(idx, _)| idx)
                        .collect();

                    for (i, &pos) in positions.iter().enumerate().rev() {
                        let (new_h, new_w) = per_image_dims.get(i).copied().unwrap_or_else(|| {
                            let grid_unit = self.pooling_kernel_size * self.patch_size;
                            (grid_unit, grid_unit)
                        });
                        let num_tokens = self.output_tokens_for_size(new_h, new_w);
                        let replacement = self.build_image_sequence(num_tokens);

                        prompt = format!(
                            "{}{}{}",
                            &prompt[..pos],
                            replacement,
                            &prompt[pos + IMAGE_TOKEN.len()..],
                        );
                    }

                    seq.set_initial_prompt(prompt.clone());
                    let toks = tokenizer
                        .encode_fast(prompt.as_str(), false)
                        .expect("Tokenization failed!");
                    seq.set_toks_and_reallocate(
                        toks.get_ids().to_vec(),
                        paged_attn_metadata.as_mut(),
                    );
                    seq.multimodal.has_changed_prompt = true;
                }

                let n_images = pixel_values.dim(0).unwrap_or(0);
                let image_ranges = find_image_placeholder_ranges(seq.get_toks(), IMAGE_TOKEN_ID);
                let cached_image_tokens =
                    cached_tokens_for_ranges(seq.prefix_cache_len(), &image_ranges);
                let seq_image_hashes = seq.image_hashes().unwrap_or(&[]);
                let image_sizes = image_sizes_all.unwrap_or_default();

                for idx in 0..n_images {
                    let total_tokens = image_ranges
                        .get(idx)
                        .map(|(_, length)| *length)
                        .unwrap_or_else(|| {
                            image_sizes
                                .get(idx)
                                .map(|&(h, w)| self.output_tokens_for_size(h as usize, w as usize))
                                .unwrap_or(self.default_output_length)
                        });
                    let cached_tokens = cached_image_tokens
                        .get(idx)
                        .copied()
                        .unwrap_or(0)
                        .min(total_tokens);
                    if cached_tokens >= total_tokens {
                        continue;
                    }

                    pixel_values_accum.push(pixel_values.get(idx)?.unsqueeze(0)?);
                    if let Some(&size) = image_sizes.get(idx) {
                        image_sizes_accum.push(size);
                    }
                    if let Some(&hash) = seq_image_hashes.get(idx) {
                        image_hashes_accum.push(hash);
                    }
                    image_cached_tokens_accum.push(cached_tokens);
                }
            }

            if pixel_values_accum.is_empty() {
                None
            } else {
                Some(Tensor::cat(&pixel_values_accum, 0)?)
            }
        } else {
            None
        };

        for seq in input_seqs.iter_mut() {
            if seq.mm_features().is_empty() {
                if let Some(hashes) = seq.image_hashes().map(|h| h.to_vec()) {
                    if !hashes.is_empty() {
                        let ranges = find_image_placeholder_ranges(seq.get_toks(), IMAGE_TOKEN_ID);
                        let mut features = build_mm_features_from_ranges(&ranges, &hashes, "img");
                        features.sort_by_key(|f| f.offset);
                        seq.set_mm_features(features);
                    }
                }
            }
        }

        let text_models_inputs_processor::InnerInputProcessorOutput {
            inputs:
                text_models_inputs_processor::InputMetadata {
                    input,
                    positions,
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta,
                },
            seq_indices,
        } = if is_prompt {
            get_prompt_input(
                input_seqs
                    .iter()
                    .map(|seq| seq.get_toks())
                    .collect::<Vec<_>>(),
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
            )?
        } else {
            get_completion_input(
                input_seqs
                    .iter()
                    .map(|seq| seq.get_toks())
                    .collect::<Vec<_>>(),
                input_seqs,
                device,
                no_kv_cache,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
            )?
        };

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values: if is_prompt { pixel_values } else { None },
            model_specific_args: Box::new(Gemma4SpecificArgs {
                audio_mel: None,
                audio_mel_mask: None,
                image_hashes: if is_prompt {
                    image_hashes_accum
                } else {
                    vec![]
                },
                image_cached_tokens: if is_prompt {
                    image_cached_tokens_accum
                } else {
                    vec![]
                },
                image_sizes: if is_prompt { image_sizes_accum } else { vec![] },
                audio_hashes: vec![],
                audio_cached_tokens: vec![],
                video_pixel_values: None,
                video_hashes: vec![],
                video_cached_tokens: vec![],
                video_sizes: vec![],
            }),
            paged_attn_meta,
            flash_meta,
        });

        Ok(InputProcessorOutput {
            inputs,
            seq_indices,
        })
    }
}

impl ImagePreProcessor for Gemma4ImageProcessor {
    const DEFAULT_MEAN: [f64; 3] = [0.0, 0.0, 0.0];
    const DEFAULT_STD: [f64; 3] = [1.0, 1.0, 1.0];

    fn preprocess(
        &self,
        mut images: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
        (_bs, _max_num_images): (usize, usize),
    ) -> Result<PreprocessedImages> {
        let _ = videos;

        let do_rescale = config.do_rescale.unwrap_or(true);
        let rescale_factor = config.rescale_factor.unwrap_or(1.0 / 255.0);
        let do_convert_rgb = config.do_convert_rgb.unwrap_or(true);
        let resample = config.resampling.to_filter()?;

        for image in &mut images {
            if do_convert_rgb {
                *image = DynamicImage::ImageRgb8(image.to_rgb8());
            }
        }

        let mut pixel_values = Vec::new();
        let mut image_sizes = Vec::new();

        for image in images {
            let (w, h) = image.dimensions();
            let (new_h, new_w) = self.compute_resize_dims(h as usize, w as usize)?;
            let resized = image.resize_exact(new_w as u32, new_h as u32, resample);

            let transforms = Transforms {
                input: &ToTensorNoNorm,
                inner_transforms: &[&do_rescale.then_some(Rescale {
                    factor: Some(rescale_factor),
                })],
            };

            pixel_values.push(resized.apply(transforms, device)?.unsqueeze(0)?);
            image_sizes.push((new_h as u32, new_w as u32));
        }

        let max_h = image_sizes.iter().map(|(h, _)| *h).max().unwrap_or(0) as usize;
        let max_w = image_sizes.iter().map(|(_, w)| *w).max().unwrap_or(0) as usize;

        let mut padded = Vec::new();
        for (pv, &(h, w)) in pixel_values.iter().zip(image_sizes.iter()) {
            let h = h as usize;
            let w = w as usize;
            if h < max_h || w < max_w {
                padded.push(
                    pv.pad_with_zeros(2, 0, max_h - h)?
                        .pad_with_zeros(3, 0, max_w - w)?,
                );
            } else {
                padded.push(pv.clone());
            }
        }

        Ok(PreprocessedImages {
            pixel_values: Tensor::cat(&padded, 0)?,
            pixel_attention_mask: None,
            image_sizes: None,
            num_img_tokens: None,
            aspect_ratio_ids: None,
            aspect_ratio_mask: None,
            num_tiles: None,
            image_grid_thw: None,
            video_grid_thw: None,
            rows: None,
            cols: None,
            pixel_values_list: None,
            tgt_sizes: None,
            image_sizes_all: Some(image_sizes),
            num_crops: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{cached_tokens_for_ranges, Gemma4Processor};
    use crate::vision_models::processor_config::ProcessorConfig;

    #[test]
    fn computes_default_patch_budget_from_output_length() {
        let processor = Gemma4Processor::new(ProcessorConfig::default(), 16, 3, 280, true);
        assert_eq!(processor.max_patches, 2520);
    }

    #[test]
    fn cached_tokens_for_ranges_handles_partial_overlap() {
        let ranges = vec![(5, 4), (12, 3), (20, 2)];

        assert_eq!(cached_tokens_for_ranges(0, &ranges), vec![0, 0, 0]);
        assert_eq!(cached_tokens_for_ranges(7, &ranges), vec![2, 0, 0]);
        assert_eq!(cached_tokens_for_ranges(13, &ranges), vec![4, 1, 0]);
        assert_eq!(cached_tokens_for_ranges(30, &ranges), vec![4, 3, 2]);
    }
}
