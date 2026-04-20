use std::collections::HashMap;

use candle_core::backend::BackendStorage;
#[cfg(feature = "cuda")]
use candle_core::cuda::cudarc::driver::{result, DevicePtr};
use candle_core::{DType, Device, Result, Storage, Tensor};
#[allow(unused_imports)]
use engine_paged_attn::{kv_scale_update, paged_attention, reshape_and_cache};
#[cfg(feature = "cuda")]
#[allow(unused_imports)]
use engine_paged_attn::turbo_rotate;

const KV_SCALE_UPDATE_ITERATION: i32 = 128;
use std::sync::atomic::{AtomicI32, Ordering};

use crate::{
    attention::SdpaParams,
    layers::Sdpa,
    paged_attention::PagedCacheType,
    pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
    turbo3_rotate_forward_in_place, turbo3_rotate_inverse_in_place,
    turbo3_score_from_packed, turbo3_value_from_packed_rotated, TurboQuantArtifacts,
    TurboQuantBits,
};

fn turboquant_debug_enabled() -> bool {
    std::env::var_os("ENGINE_TURBOQUANT_DEBUG").is_some()
}

fn turboquant_native_paged_enabled() -> bool {
    std::env::var_os("ENGINE_TURBOQUANT_DISABLE_NATIVE_PAGED").is_none()
}

fn turboquant_rotation_enabled() -> bool {
    std::env::var_os("ENGINE_TURBOQUANT_DISABLE_ROTATION").is_none()
}

fn turboquant_should_use_windowed_paged_kv_metadata(use_full: bool) -> bool {
    // `paged_kv_*` metadata is currently built only from the windowed block tables.
    // Full-attention layers in hybrid models (for example GPT-OSS) must therefore
    // fall back to `full_block_tables/full_context_lens` until a full-context
    // paged-KV metadata variant exists.
    !use_full
}

fn turboquant_debug_log(message: impl AsRef<str>) {
    if turboquant_debug_enabled() {
        tracing::info!(
            target: "engine_core::paged_attention::turboquant_debug",
            "{}",
            message.as_ref()
        );
    }
}

fn turboquant_rmse(lhs: &[f32], rhs: &[f32]) -> f32 {
    if lhs.is_empty() {
        return 0.0;
    }
    let sum = lhs
        .iter()
        .zip(rhs.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>();
    (sum / lhs.len() as f32).sqrt()
}

fn turboquant_max_abs_error(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn turboquant_l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|v| v * v).sum::<f32>().sqrt()
}

fn log_turboquant_roundtrip(
    bits: TurboQuantBits,
    head_dim: usize,
    token_idx: usize,
    head_idx: usize,
    key: &[f32],
    key_recon: &[f32],
    value: &[f32],
    value_recon: &[f32],
) {
    if !turboquant_debug_enabled() {
        return;
    }
    turboquant_debug_log(format!(
        "encode token={token_idx} head={head_idx} bits={} head_dim={head_dim} key_rmse={:.6} key_max_abs={:.6} key_l2={:.6} key_recon_l2={:.6} value_rmse={:.6} value_max_abs={:.6} value_l2={:.6} value_recon_l2={:.6}",
        bits.cache_type_name(),
        turboquant_rmse(key, key_recon),
        turboquant_max_abs_error(key, key_recon),
        turboquant_l2_norm(key),
        turboquant_l2_norm(key_recon),
        turboquant_rmse(value, value_recon),
        turboquant_max_abs_error(value, value_recon),
        turboquant_l2_norm(value),
        turboquant_l2_norm(value_recon),
    ));
}

fn log_turboquant_payload_roundtrip(
    bits: TurboQuantBits,
    head_dim: usize,
    token_idx: usize,
    head_idx: usize,
    kind: &str,
    expected: &[u8],
    observed: &[u8],
    original: &[f32],
    reconstructed: &[f32],
) {
    if !turboquant_debug_enabled() {
        return;
    }
    let byte_mismatches = expected
        .iter()
        .zip(observed.iter())
        .filter(|(lhs, rhs)| lhs != rhs)
        .count();
    turboquant_debug_log(format!(
        "cache-readback token={token_idx} head={head_idx} kind={kind} bits={} head_dim={head_dim} byte_mismatches={} payload_bytes={} rmse={:.6} max_abs={:.6}",
        bits.cache_type_name(),
        byte_mismatches,
        expected.len(),
        turboquant_rmse(original, reconstructed),
        turboquant_max_abs_error(original, reconstructed),
    ));
}

pub struct PagedAttention {
    alibi_slopes: Option<Tensor>,
    cache_type: Option<PagedCacheType>,
    k_scale: Option<Tensor>,
    v_scale: Option<Tensor>,
    kv_updated_times: AtomicI32,
}

impl PagedAttention {
    pub fn new(
        head_dim: usize,
        device: &Device,
        alibi_slopes: Option<Vec<f32>>,
        cache_type: Option<PagedCacheType>,
    ) -> Result<Self> {
        let alibi_slopes = if let Some(alibi_slopes) = alibi_slopes {
            assert_eq!(alibi_slopes.len(), head_dim);
            Some(Tensor::new(alibi_slopes, device)?)
        } else {
            None
        };
        Ok(Self {
            alibi_slopes,
            cache_type,
            k_scale: Some(Tensor::new(1f32, device)?),
            v_scale: Some(Tensor::new(1f32, device)?),
            kv_updated_times: AtomicI32::new(0),
        })
    }

    pub fn cache_type(&self) -> Option<PagedCacheType> {
        self.cache_type
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(unused_variables)]
    /// query: shape = [batch_size, seq_len, num_heads * head_size]
    /// key: shape = [batch_size, seq_len, num_kv_heads * head_size]
    /// value: shape = [batch_size, num_kv_heads * head_size]
    /// key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
    ///     block_size, x]
    /// value_cache: shape = [num_blocks, num_kv_heads, head_size,
    ///     block_size]
    /// input_metadata: metadata for paged attention.
    #[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
        mut key_cache: Option<Tensor>,
        mut value_cache: Option<Tensor>,
        input_metadata: &PagedAttentionInputMetadata,
        sdpa_params: &SdpaParams,
        flash_params: Option<&FlashParams>,
    ) -> Result<Tensor> {
        let turboquant_cache = key_cache
            .as_ref()
            .zip(value_cache.as_ref())
            .is_some_and(|(k, v)| k.dtype() == DType::U8 && v.dtype() == DType::U8);

        if let (Some(k_scale), Some(v_scale), Some(key_cache)) =
            (&self.k_scale, &self.v_scale, &key_cache)
        {
            if self.kv_updated_times.load(Ordering::Relaxed) < KV_SCALE_UPDATE_ITERATION
                && key_cache.dtype() == DType::F8E4M3
            {
                // scale update only used for fp8 kvcache
                kv_scale_update(key, value, k_scale, v_scale)?;
                self.kv_updated_times.fetch_add(1, Ordering::Relaxed);
            }
        }

        let slot_mapping = input_metadata
            .slot_mappings
            .get(&query.device().location())
            .unwrap();
        let dims = slot_mapping.dims();
        let slot_mapping = if dims.len() > 1 {
            &slot_mapping.flatten(0, dims.len())?
        } else {
            slot_mapping
        };

        // For models with per-layer sliding windows (GPT-OSS, Gemma2):
        // - Full-attention layers (sliding_window == None) use the full block tables.
        // - Sliding-window layers (sliding_window == Some) use the windowed block tables.
        // If full_block_tables is not populated, fall back to the regular block_tables.
        let use_full =
            sdpa_params.sliding_window.is_none() && input_metadata.full_block_tables.is_some();

        let block_tables = if use_full {
            input_metadata
                .full_block_tables
                .as_ref()
                .unwrap()
                .get(&query.device().location())
                .unwrap()
        } else {
            input_metadata
                .block_tables
                .as_ref()
                .unwrap()
                .get(&query.device().location())
                .unwrap()
        };
        let context_lens = if use_full {
            input_metadata
                .full_context_lens
                .as_ref()
                .unwrap()
                .get(&query.device().location())
                .unwrap()
        } else {
            input_metadata
                .context_lens
                .as_ref()
                .unwrap()
                .get(&query.device().location())
                .unwrap()
        };

        let alibi_slopes = if let Some(alibi_slopes) = self.alibi_slopes.as_ref() {
            Some(alibi_slopes.to_device(query.device())?)
        } else {
            None
        };

        let (batch_size, attention_heads, seq_len, head_size) = query.shape().dims4()?;
        let (_, key_value_heads, _, kv_head_size) = key.shape().dims4()?;
        let (_, _, _, value_head_size) = value.shape().dims4()?;
        if value_head_size != kv_head_size {
            candle_core::bail!(
                "turboquant key/value head dim mismatch: key={kv_head_size}, value={value_head_size}"
            );
        }
        let effective_cache_type = self.cache_type;
        if turboquant_cache && std::env::var_os("ENGINE_GEMMA4_CACHE_DEBUG").is_some() {
            if let (Some(kc), Some(vc)) = (key_cache.as_ref(), value_cache.as_ref()) {
                if let Ok((kb, kh, kp, ks, kx)) = kc.dims5() {
                    if let Ok((vb, vh, vp, vs, vx)) = vc.dims5() {
                        tracing::info!(
                            target: "engine_core::paged_attention::cache_debug",
                            "forward turboquant effective_cache_type={:?} kv_head_size={} cache_k=[{kb},{kh},{kp},{ks},{kx}] cache_v=[{vb},{vh},{vp},{vs},{vx}]",
                            effective_cache_type,
                            kv_head_size,
                        );
                    }
                }
            }
        }
        let turbo_bits = if turboquant_cache {
            Some(
                if let Some(bits) =
                    effective_cache_type.and_then(|cache_type| cache_type.turboquant_bits())
                {
                    bits
                } else {
                    turboquant_bits_for_cache(
                        key_cache.as_ref().unwrap(),
                        value_cache.as_ref().unwrap(),
                        kv_head_size,
                    )?
                },
            )
        } else {
            None
        };

        if turboquant_cache {
            turboquant_reshape_and_cache(
                key,
                value,
                key_cache.as_mut().unwrap(),
                value_cache.as_mut().unwrap(),
                slot_mapping,
                turbo_bits.unwrap(),
                kv_head_size,
            )
            .map_err(|e| {
                candle_core::Error::Msg(format!(
                    "turboquant encode/write failed (seq_len={}, kv_heads={}, kv_head_size={}): {e}",
                    seq_len, key_value_heads, kv_head_size
                ))
            })?;

            let turbo_rotated =
                turbo_bits.is_some_and(|bits| turboquant_uses_rotated_space(bits, kv_head_size));
            // The native Turbo3 paged decode expects the rotated-space cache layout.
            // Without that fastpath contract, fall back to gather+dequant for correctness.
            let turbo_decode_via_native_paged = turbo_bits
                .is_some_and(|bits| turboquant_native_rotated_cuda_supported(bits, kv_head_size))
                && turbo_rotated
                && attention_mask.is_none()
                && seq_len == 1
                && input_metadata.num_cached_tokens.is_none()
                && turboquant_native_paged_enabled();

            if !turbo_decode_via_native_paged {
                // Fused decode: compute attention scores directly from packed turbo3
                // bytes using online softmax, avoiding full KV decompression.
                let use_fused_decode = seq_len == 1
                    && turbo_rotated
                    && turbo_bits == Some(TurboQuantBits::Three)
                    && attention_mask.is_none()
                    && std::env::var_os("ENGINE_TURBOQUANT_DISABLE_FUSED_DECODE").is_none();

                if use_fused_decode {
                    return turboquant_fused_decode(
                        query,
                        key_cache.as_ref().unwrap(),
                        value_cache.as_ref().unwrap(),
                        TurboQuantBits::Three,
                        block_tables,
                        context_lens,
                        kv_head_size,
                        key_value_heads,
                        attention_heads,
                        sdpa_params.softmax_scale,
                    )
                    .map_err(|e| {
                        candle_core::Error::Msg(format!(
                            "turboquant fused decode failed (kv_heads={}, q_heads={}, kv_head_size={}): {e}",
                            key_value_heads, attention_heads, kv_head_size
                        ))
                    });
                }

                let paged_kv_metadata = turboquant_should_use_windowed_paged_kv_metadata(use_full);
                let (k_gathered, v_gathered, cu_kv, max_k) = turboquant_gather_kv_cache(
                    key_cache.as_ref().unwrap(),
                    value_cache.as_ref().unwrap(),
                    turbo_bits.unwrap(),
                    Some(block_tables),
                    Some(context_lens),
                    if paged_kv_metadata {
                        input_metadata
                            .paged_kv_indptr
                            .as_ref()
                            .and_then(|x| x.get(&query.device().location()))
                    } else {
                        None
                    },
                    if paged_kv_metadata {
                        input_metadata
                            .paged_kv_indices
                            .as_ref()
                            .and_then(|x| x.get(&query.device().location()))
                    } else {
                        None
                    },
                    if paged_kv_metadata {
                        input_metadata
                            .paged_kv_last_page_len
                            .as_ref()
                            .and_then(|x| x.get(&query.device().location()))
                    } else {
                        None
                    },
                    query.device(),
                    query.dtype(),
                    kv_head_size,
                )
                .map_err(|e| {
                    candle_core::Error::Msg(format!(
                        "turboquant gather/read failed (seq_len={}, kv_heads={}, kv_head_size={}): {e}",
                        seq_len, key_value_heads, kv_head_size
                    ))
                })?;
                let k_4d = k_gathered.unsqueeze(0)?.transpose(1, 2)?.contiguous()?;
                let v_4d = v_gathered.unsqueeze(0)?.transpose(1, 2)?.contiguous()?;
                let turbo_flash_params = build_turboquant_flash_params(
                    query,
                    flash_params,
                    input_metadata,
                    &cu_kv,
                    max_k,
                )?;
                let rotated_query = if turbo_rotated {
                    rotate_turbo3_attention_tensor(query, key_value_heads, true)?
                } else {
                    query.clone()
                };
                let attn_out = Sdpa
                    .run_attention(
                        &rotated_query,
                        &k_4d,
                        &v_4d,
                        attention_mask,
                        turbo_flash_params.as_ref(),
                        sdpa_params,
                    )
                    .map_err(|e| {
                        candle_core::Error::Msg(format!(
                            "turboquant attention failed (seq_len={}, kv_heads={}, kv_head_size={}, max_k={}, cu_kv_len={}): {e}",
                            seq_len,
                            key_value_heads,
                            kv_head_size,
                            max_k,
                            cu_kv.dims1().unwrap_or(0)
                        ))
                    })?;
                return if turbo_rotated {
                    rotate_turbo3_attention_tensor(&attn_out, key_value_heads, false)
                } else {
                    Ok(attn_out)
                };
            }
        }

        // === Prefix cache hit path ===
        if input_metadata.num_cached_tokens.is_some() && attention_mask.is_some() {
            // Write new tokens to cache for future decode steps
            if key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
                let k_flat = key
                    .transpose(1, 2)?
                    .reshape(((), key_value_heads, head_size))?;
                let v_flat = value
                    .transpose(1, 2)?
                    .reshape(((), key_value_heads, head_size))?;
                reshape_and_cache(
                    &k_flat,
                    &v_flat,
                    self.k_scale.as_ref(),
                    self.v_scale.as_ref(),
                    key_cache.as_mut().unwrap(),
                    value_cache.as_mut().unwrap(),
                    slot_mapping,
                )?;
            }

            assert!(
                alibi_slopes.is_none(),
                "alibi slopes not supported in prefix cache path"
            );

            let device = query.device();

            // Gather all K/V from paged cache into contiguous tensors.
            // The gather kernel handles x-unpacking for K, transpose for V,
            // and FP8 dequantization via k_scale/v_scale when applicable.
            let cu_kv = input_metadata
                .cu_seqlens_kv
                .as_ref()
                .expect("cu_seqlens_kv required for prefix cache path")
                .get(&device.location())
                .unwrap();
            let (k_gathered, v_gathered) = engine_paged_attn::gather_kv_cache(
                key_cache.as_ref().unwrap(),
                value_cache.as_ref().unwrap(),
                self.k_scale.as_ref(),
                self.v_scale.as_ref(),
                block_tables,
                cu_kv,
                query.dtype(),
            )?;

            // gathered: (total_kv, kv_heads, dim) -> (1, kv_heads, total_kv, dim)
            let k_4d = k_gathered.unsqueeze(0)?.transpose(1, 2)?;
            let v_4d = v_gathered.unsqueeze(0)?.transpose(1, 2)?;

            // Build a local FlashParams with packed K cu_seqlens from
            // cu_seqlens_kv (matching the gathered KV layout). The pipeline's
            // flash_params uses padded seqlens_k which doesn't match packed KV.
            // Q seqlens stay padded since Q is still in padded batch layout.
            let prefix_flash_params = flash_params.map(|fp| {
                let max_kv = input_metadata
                    .num_cached_tokens
                    .as_ref()
                    .unwrap()
                    .iter()
                    .zip(input_metadata.query_lens.as_ref().unwrap().iter())
                    .map(|(&nc, &ql)| (nc + ql) as u32)
                    .max()
                    .unwrap_or(0);
                FlashParams {
                    max_q: fp.max_q,
                    max_k: max_kv,
                    cumulative_seqlens_q: fp.cumulative_seqlens_q.clone(),
                    cumulative_seqlens_k: input_metadata.cu_seqlens_kv.as_ref().unwrap().clone(),
                    causal: fp.causal,
                }
            });

            return Sdpa.run_attention(
                query,
                &k_4d,
                &v_4d,
                attention_mask,
                prefix_flash_params.as_ref(),
                sdpa_params,
            );
        }

        #[allow(clippy::cast_possible_truncation)]
        let att = match attention_mask {
            None => None,
            Some(mask) => Some(Sdpa.run_attention(
                query,
                key,
                value,
                Some(mask),
                flash_params,
                sdpa_params,
            )?),
        };

        // paged-attn expects [batch_size, num_tokens, num_heads, head_size]
        let (query, key, value) = if seq_len > 1 {
            let q = query
                .transpose(1, 2)?
                .reshape(((), attention_heads, head_size))?;
            let k = key
                .transpose(1, 2)?
                .reshape(((), key_value_heads, head_size))?;
            let v = value
                .transpose(1, 2)?
                .reshape(((), key_value_heads, head_size))?;
            (q, k, v)
        } else {
            // avoid unnecessary transpose for decoding
            let q = query.reshape(((), attention_heads, head_size))?;
            let k = key.reshape(((), key_value_heads, head_size))?;
            let v = value.reshape(((), key_value_heads, head_size))?;
            (q, k, v)
        };

        // key: Tensor,              // [num_tokens, num_heads, head_size]
        // value: Tensor,            // [num_tokens, num_heads, head_size]
        // key_cache: &mut Tensor,   // [num_blocks, num_heads, head_size/x, block_size, x] 48,32,16,16,8
        // value_cache: &mut Tensor, // [num_blocks, num_heads, head_size, block_size] 48,32,128,16
        // slot_mapping: Tensor,     // [num_tokens]
        if !turboquant_cache && key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
            reshape_and_cache(
                &key,
                &value,
                self.k_scale.as_ref(),
                self.v_scale.as_ref(),
                key_cache.as_mut().unwrap(),
                value_cache.as_mut().unwrap(),
                slot_mapping,
            )?;
        }

        if let Some(att) = att {
            // Return result in prefill or first prefix chunk
            return Ok(att);
        }

        //  Args:
        //  output: shape = [num_generation_tokens, num_heads, head_size]
        //
        //  query: shape = [num_generation_tokens, num_heads, head_size]
        //
        //  key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
        //      block_size, x]
        //
        //  value_cache: shape = [num_blocks, num_kv_heads, head_size,
        //      block_size]
        //
        //  input_metadata: metadata for paged attention.
        //
        //  alibi_slopes: shape = [num_heads]
        #[allow(clippy::cast_possible_truncation)]
        let turbo_rotated =
            turbo_bits.is_some_and(|bits| turboquant_uses_rotated_space(bits, head_size));
        let paged_query = if turbo_rotated {
            rotate_turbo3_attention_tensor(&query, key_value_heads, true)?
        } else {
            query.clone()
        };
        let res = paged_attention(
            &paged_query,
            self.k_scale.as_ref(),
            self.v_scale.as_ref(),
            key_cache.as_ref().unwrap(),
            value_cache.as_ref().unwrap(),
            block_tables,
            context_lens,
            alibi_slopes.as_ref(),
            if use_full {
                input_metadata.full_max_context_len.unwrap()
            } else {
                input_metadata.max_context_len.unwrap()
            },
            sdpa_params.softmax_scale,
            sdpa_params.softcap.unwrap_or(1.0f32),
            sdpa_params.sinks.as_ref(),
        )?;

        if turbo_rotated {
            rotate_turbo3_attention_tensor(&res, key_value_heads, false)
        } else {
            Ok(res)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PagedAttention;
    use crate::paged_attention::PagedCacheType;

    #[test]
    fn paged_attention_preserves_explicit_cache_type() {
        let paged = PagedAttention::new(
            256,
            &candle_core::Device::Cpu,
            None,
            Some(PagedCacheType::TurboQuant3),
        )
        .unwrap();
        assert_eq!(paged.cache_type, Some(PagedCacheType::TurboQuant3));
    }

    #[test]
    fn paged_attention_keeps_auto_as_none() {
        let paged = PagedAttention::new(256, &candle_core::Device::Cpu, None, None).unwrap();
        assert_eq!(paged.cache_type, None);
    }
}

fn build_turboquant_flash_params(
    query: &Tensor,
    flash_params: Option<&FlashParams>,
    input_metadata: &PagedAttentionInputMetadata,
    cu_kv: &Tensor,
    max_k: u32,
) -> Result<Option<FlashParams>> {
    let device_location = query.device().location();
    let cumulative_seqlens_q = if let Some(fp) = flash_params {
        fp.cumulative_seqlens_q.clone()
    } else {
        let (batch_size, _, seq_len, _) = query.shape().dims4()?;
        let q_lens = if let Some(query_lens) = &input_metadata.query_lens {
            query_lens.iter().map(|&x| x as u32).collect::<Vec<_>>()
        } else {
            std::iter::once(0u32)
                .chain((1..=batch_size).map(|i| (i * seq_len) as u32))
                .collect::<Vec<_>>()
        };
        HashMap::from([(device_location, Tensor::new(q_lens, query.device())?)])
    };
    let max_q = if let Some(fp) = flash_params {
        fp.max_q
    } else if let Some(query_lens) = &input_metadata.query_lens {
        query_lens.iter().copied().max().unwrap_or(0) as u32
    } else {
        query.shape().dims4()?.2 as u32
    };
    let causal = flash_params.map(|fp| fp.causal).unwrap_or(true);
    Ok(Some(FlashParams {
        max_q,
        max_k,
        cumulative_seqlens_q,
        cumulative_seqlens_k: HashMap::from([(device_location, cu_kv.clone())]),
        causal,
    }))
}

fn infer_turboquant_cache_layout(
    key_cache: &Tensor,
    value_cache: &Tensor,
    head_dim_hint: usize,
) -> Result<(TurboQuantBits, usize)> {
    let (_, _, key_payload_bytes, _, x) = key_cache.dims5()?;
    let (_, _, value_payload_bytes, _, value_x) = value_cache.dims5()?;
    if x != 1 {
        candle_core::bail!("turboquant key cache expects x=1, got {x}");
    }
    if value_x != 1 {
        candle_core::bail!("turboquant value cache expects x=1, got {value_x}");
    }

    let mut matches = Vec::new();
    for bits in [
        TurboQuantBits::Two,
        TurboQuantBits::Three,
        TurboQuantBits::Four,
    ] {
        let dims: Vec<usize> = match bits {
            TurboQuantBits::Three => (32..=8192).step_by(32).collect(),
            TurboQuantBits::Two | TurboQuantBits::Four => (1..=8192).collect(),
        };
        for dim in dims {
            if key_payload_bytes == bits.packed_key_bytes_for_dim(dim)
                && value_payload_bytes == bits.packed_value_bytes_for_dim(dim)
            {
                matches.push((bits, dim));
            }
        }
    }

    if let Some((bits, dim)) = matches
        .iter()
        .copied()
        .find(|(_, dim)| *dim == head_dim_hint)
    {
        return Ok((bits, dim));
    }
    if matches.len() == 1 {
        return Ok(matches[0]);
    }
    candle_core::bail!(
        "unable to infer turboquant cache type for head_dim={head_dim_hint}, key_payload_bytes={key_payload_bytes}, value_payload_bytes={value_payload_bytes}"
    );
}

fn turboquant_bits_for_cache(
    key_cache: &Tensor,
    value_cache: &Tensor,
    head_dim: usize,
) -> Result<TurboQuantBits> {
    Ok(infer_turboquant_cache_layout(key_cache, value_cache, head_dim)?.0)
}

fn infer_turboquant_head_dim_for_bits(
    key_cache: &Tensor,
    value_cache: &Tensor,
    bits: TurboQuantBits,
    head_dim_hint: usize,
) -> Result<usize> {
    let (_, _, key_payload_bytes, _, x) = key_cache.dims5()?;
    let (_, _, value_payload_bytes, _, value_x) = value_cache.dims5()?;
    if x != 1 {
        candle_core::bail!("turboquant key cache expects x=1, got {x}");
    }
    if value_x != 1 {
        candle_core::bail!("turboquant value cache expects x=1, got {value_x}");
    }

    let dims: Vec<usize> = match bits {
        TurboQuantBits::Three => (32..=8192).step_by(32).collect(),
        TurboQuantBits::Two | TurboQuantBits::Four => (1..=8192).collect(),
    };
    let matches = dims
        .into_iter()
        .filter(|dim| {
            key_payload_bytes == bits.packed_key_bytes_for_dim(*dim)
                && value_payload_bytes == bits.packed_value_bytes_for_dim(*dim)
        })
        .collect::<Vec<_>>();

    if matches.contains(&head_dim_hint) {
        return Ok(head_dim_hint);
    }
    if matches.len() == 1 {
        return Ok(matches[0]);
    }
    candle_core::bail!(
        "unable to infer turboquant head dim for cache_type={} head_dim={head_dim_hint}, key_payload_bytes={key_payload_bytes}, value_payload_bytes={value_payload_bytes}",
        bits.cache_type_name()
    );
}

fn turboquant_uses_rotated_space(bits: TurboQuantBits, head_dim: usize) -> bool {
    if !turboquant_rotation_enabled() {
        return false;
    }
    match bits {
        TurboQuantBits::Three => bits.supports_rotation_head_dim(head_dim),
        _ => bits.supports_rotation_head_dim(head_dim),
    }
}

fn turboquant_native_rotated_cuda_supported(bits: TurboQuantBits, head_dim: usize) -> bool {
    bits == TurboQuantBits::Three && head_dim % 128 == 0 && matches!(head_dim, 128 | 256)
}

fn turboquant_reshape_and_cache(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
    bits: TurboQuantBits,
    head_dim: usize,
) -> Result<()> {
    let (_, key_value_heads, seq_len, key_head_dim) = key.dims4()?;
    let (_, value_heads, value_seq_len, value_head_dim) = value.dims4()?;
    if key_head_dim != head_dim || value_head_dim != head_dim {
        candle_core::bail!(
            "turboquant head dim mismatch: key_head_dim={key_head_dim}, value_head_dim={value_head_dim}, expected={head_dim}"
        );
    }
    if value_heads != key_value_heads || value_seq_len != seq_len {
        candle_core::bail!(
            "turboquant key/value shape mismatch: key_heads={key_value_heads}, value_heads={value_heads}, key_seq_len={seq_len}, value_seq_len={value_seq_len}"
        );
    }
    let key = if seq_len > 1 {
        key.transpose(1, 2)?
            .reshape(((), key_value_heads, head_dim))?
    } else {
        key.reshape(((), key_value_heads, head_dim))?
    };
    let value = if seq_len > 1 {
        value
            .transpose(1, 2)?
            .reshape(((), key_value_heads, head_dim))?
    } else {
        value.reshape(((), key_value_heads, head_dim))?
    };
    let key = key.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let value = value.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let keys = key.to_vec3::<f32>()?;
    let values = value.to_vec3::<f32>()?;
    let slots = tensor_to_i64_vec(slot_mapping)?;
    let (num_blocks, num_heads, key_bytes, block_size, x) = key_cache.dims5()?;
    let (_, _, value_bytes, value_block_size, value_x) = value_cache.dims5()?;
    if x != 1 {
        candle_core::bail!("turboquant key cache expects x=1, got {x}");
    }
    if value_x != 1 {
        candle_core::bail!("turboquant value cache expects x=1, got {value_x}");
    }
    if block_size != value_block_size {
        candle_core::bail!(
            "turboquant key/value block sizes differ: {block_size} vs {value_block_size}"
        );
    }
    let cache_head_dim =
        infer_turboquant_head_dim_for_bits(key_cache, value_cache, bits, head_dim)?;
    if cache_head_dim != head_dim {
        candle_core::bail!(
            "invalid turboquant payload sizes for head_dim={head_dim}: cache encodes head_dim={cache_head_dim}"
        );
    }
    if !bits.supports_storage_head_dim(head_dim) {
        candle_core::bail!(
            "{} cache requires head_dim to be a multiple of 32, got {head_dim}",
            bits.cache_type_name()
        );
    }
    let expected_key_bytes = bits.packed_key_bytes_for_dim(head_dim);
    let expected_value_bytes = bits.packed_value_bytes_for_dim(head_dim);
    if key_bytes != expected_key_bytes || value_bytes != expected_value_bytes {
        candle_core::bail!(
            "invalid turboquant payload sizes for head_dim={head_dim}: key_bytes={key_bytes} expected={expected_key_bytes}, value_bytes={value_bytes} expected={expected_value_bytes}"
        );
    }
    let artifacts = TurboQuantArtifacts::new(head_dim, bits, 0xC701_0ACE);
    let turbo_rotated = turboquant_uses_rotated_space(bits, head_dim);
    let flat_artifacts = (bits == TurboQuantBits::Three && turbo_rotated)
        .then(|| TurboQuantArtifacts::new(num_heads * head_dim, bits, 0xC701_0ACE));
    let key_block_bytes = num_heads * key_bytes * block_size;
    let value_block_bytes = num_heads * value_bytes * block_size;
    let mut touched_key_blocks = HashMap::new();
    let mut touched_value_blocks = HashMap::new();
    let mut first_payload_sample: Option<(
        usize,
        usize,
        usize,
        Vec<u8>,
        Vec<u8>,
        Vec<f32>,
        Vec<f32>,
    )> = None;

    for (token_idx, slot) in slots.into_iter().enumerate() {
        if slot < 0 {
            continue;
        }
        let slot = slot as usize;
        let block_number = slot / block_size;
        let block_offset = slot % block_size;
        if block_number >= num_blocks {
            candle_core::bail!("turboquant slot {slot} maps outside cache");
        }
        if !touched_key_blocks.contains_key(&block_number) {
            let block = read_cuda_block_u8(key_cache, block_number, key_block_bytes).map_err(|e| {
                candle_core::Error::Msg(format!(
                    "turboquant key block read failed (token_idx={token_idx}, block_number={block_number}, block_bytes={key_block_bytes}): {e}"
                ))
            })?;
            touched_key_blocks.insert(block_number, block);
        }
        if !touched_value_blocks.contains_key(&block_number) {
            let block =
                read_cuda_block_u8(value_cache, block_number, value_block_bytes).map_err(|e| {
                    candle_core::Error::Msg(format!(
                        "turboquant value block read failed (token_idx={token_idx}, block_number={block_number}, block_bytes={value_block_bytes}): {e}"
                    ))
                })?;
            touched_value_blocks.insert(block_number, block);
        }
        let key_block = touched_key_blocks.get_mut(&block_number).unwrap();
        let value_block = touched_value_blocks.get_mut(&block_number).unwrap();
        if bits == TurboQuantBits::Three && turbo_rotated {
            let flat_artifacts = flat_artifacts.as_ref().unwrap();
            let flat_key = keys[token_idx]
                .iter()
                .flat_map(|head| head.iter().copied())
                .collect::<Vec<_>>();
            let flat_value = values[token_idx]
                .iter()
                .flat_map(|head| head.iter().copied())
                .collect::<Vec<_>>();
            let compressed_key =
                flat_artifacts.compress_key_with_rotation(&flat_key, turbo_rotated);
            let compressed_value =
                flat_artifacts.compress_value_with_rotation(&flat_value, turbo_rotated);
            let flat_key_payload = flat_artifacts.pack_key(&compressed_key);
            let flat_value_payload = flat_artifacts.pack_value(&compressed_value);
            if flat_key_payload.len() != num_heads * key_bytes
                || flat_value_payload.len() != num_heads * value_bytes
            {
                candle_core::bail!(
                    "turboquant3 flattened payload size mismatch: key={} expected={}, value={} expected={}",
                    flat_key_payload.len(),
                    num_heads * key_bytes,
                    flat_value_payload.len(),
                    num_heads * value_bytes,
                );
            }
            if token_idx == 0 {
                let key_recon = if turbo_rotated {
                    flat_artifacts
                        .reconstruct_key_mse_rotated_with_rotation(&compressed_key, turbo_rotated)
                } else {
                    flat_artifacts.reconstruct_key_mse_with_rotation(&compressed_key, turbo_rotated)
                };
                let value_recon = if turbo_rotated {
                    flat_artifacts
                        .decompress_value_rotated_with_rotation(&compressed_value, turbo_rotated)
                } else {
                    flat_artifacts.decompress_value_with_rotation(&compressed_value, turbo_rotated)
                };
                log_turboquant_roundtrip(
                    bits,
                    head_dim,
                    token_idx,
                    0,
                    &keys[token_idx][0],
                    &key_recon[..head_dim],
                    &values[token_idx][0],
                    &value_recon[..head_dim],
                );
            }
            for head in 0..num_heads {
                let key_start = head * key_bytes;
                let value_start = head * value_bytes;
                let key_payload = &flat_key_payload[key_start..key_start + key_bytes];
                let value_payload = &flat_value_payload[value_start..value_start + value_bytes];
                for (byte_idx, &byte) in key_payload.iter().enumerate() {
                    let linear = ((head * key_bytes + byte_idx) * block_size) + block_offset;
                    key_block[linear] = byte;
                }
                for (byte_idx, &byte) in value_payload.iter().enumerate() {
                    let linear = ((head * value_bytes + byte_idx) * block_size) + block_offset;
                    value_block[linear] = byte;
                }
            }
        } else {
            for head in 0..num_heads {
                let compressed_key =
                    artifacts.compress_key_with_rotation(&keys[token_idx][head], turbo_rotated);
                let compressed_value =
                    artifacts.compress_value_with_rotation(&values[token_idx][head], turbo_rotated);
                if token_idx == 0 && head == 0 {
                    let key_recon = if turbo_rotated {
                        artifacts.reconstruct_key_mse_rotated_with_rotation(
                            &compressed_key,
                            turbo_rotated,
                        )
                    } else {
                        artifacts.reconstruct_key_mse_with_rotation(&compressed_key, turbo_rotated)
                    };
                    let value_recon = if turbo_rotated {
                        artifacts.decompress_value_rotated_with_rotation(
                            &compressed_value,
                            turbo_rotated,
                        )
                    } else {
                        artifacts.decompress_value_with_rotation(&compressed_value, turbo_rotated)
                    };
                    log_turboquant_roundtrip(
                        bits,
                        head_dim,
                        token_idx,
                        head,
                        &keys[token_idx][head],
                        &key_recon,
                        &values[token_idx][head],
                        &value_recon,
                    );
                }
                let key_payload = artifacts.pack_key(&compressed_key);
                let value_payload = artifacts.pack_value(&compressed_value);
                if token_idx == 0 && head == 0 {
                    first_payload_sample = Some((
                        block_number,
                        block_offset,
                        head,
                        key_payload.clone(),
                        value_payload.clone(),
                        keys[token_idx][head].clone(),
                        values[token_idx][head].clone(),
                    ));
                }
                for (byte_idx, byte) in key_payload.into_iter().enumerate() {
                    let linear = ((head * key_bytes + byte_idx) * block_size) + block_offset;
                    key_block[linear] = byte;
                }
                for (byte_idx, byte) in value_payload.into_iter().enumerate() {
                    let linear = ((head * value_bytes + byte_idx) * block_size) + block_offset;
                    value_block[linear] = byte;
                }
            }
        }
    }

    for (block_number, block) in touched_key_blocks {
        write_cuda_block_u8(key_cache, block_number, &block).map_err(|e| {
            candle_core::Error::Msg(format!(
                "turboquant key block write failed (block_number={block_number}, block_bytes={}): {e}",
                block.len()
            ))
        })?;
    }
    for (block_number, block) in touched_value_blocks {
        write_cuda_block_u8(value_cache, block_number, &block).map_err(|e| {
            candle_core::Error::Msg(format!(
                "turboquant value block write failed (block_number={block_number}, block_bytes={}): {e}",
                block.len()
            ))
        })?;
    }
    key_cache.device().synchronize()?;
    value_cache.device().synchronize()?;
    if bits != TurboQuantBits::Three {
        if let Some((
            block_number,
            block_offset,
            head,
            key_payload,
            value_payload,
            original_key,
            original_value,
        )) = first_payload_sample
        {
            let key_block = read_cuda_block_u8(key_cache, block_number, key_block_bytes)?;
            let value_block = read_cuda_block_u8(value_cache, block_number, value_block_bytes)?;
            let mut observed_key = vec![0u8; key_bytes];
            let mut observed_value = vec![0u8; value_bytes];
            for byte_idx in 0..key_bytes {
                let linear = ((head * key_bytes + byte_idx) * block_size) + block_offset;
                observed_key[byte_idx] = key_block[linear];
            }
            for byte_idx in 0..value_bytes {
                let linear = ((head * value_bytes + byte_idx) * block_size) + block_offset;
                observed_value[byte_idx] = value_block[linear];
            }
            let turbo_rotated = turboquant_uses_rotated_space(bits, head_dim);
            let reconstructed_key = if turbo_rotated {
                artifacts.reconstruct_key_mse_with_rotation(
                    &artifacts.unpack_key(&observed_key),
                    turbo_rotated,
                )
            } else {
                artifacts.reconstruct_key_mse_with_rotation(
                    &artifacts.unpack_key(&observed_key),
                    turbo_rotated,
                )
            };
            let reconstructed_value = artifacts.decompress_value_with_rotation(
                &artifacts.unpack_value(&observed_value),
                turbo_rotated,
            );
            log_turboquant_payload_roundtrip(
                bits,
                head_dim,
                0,
                head,
                "key",
                &key_payload,
                &observed_key,
                &original_key,
                &reconstructed_key,
            );
            log_turboquant_payload_roundtrip(
                bits,
                head_dim,
                0,
                head,
                "value",
                &value_payload,
                &observed_value,
                &original_value,
                &reconstructed_value,
            );
        }
    }
    Ok(())
}

fn turboquant_decode_flat_token(
    artifacts: &TurboQuantArtifacts,
    bits: TurboQuantBits,
    head_dim: usize,
    key_payload: &[u8],
    value_payload: &[u8],
    k_out: &mut Vec<f32>,
    v_out: &mut Vec<f32>,
) {
    let key = artifacts.unpack_key(key_payload);
    let value = artifacts.unpack_value(value_payload);
    let reconstructed_key = if turboquant_uses_rotated_space(bits, head_dim) {
        artifacts.reconstruct_key_mse_rotated_with_rotation(&key, true)
    } else {
        artifacts.reconstruct_key_mse_with_rotation(&key, false)
    };
    let reconstructed_value = if turboquant_uses_rotated_space(bits, head_dim) {
        artifacts.decompress_value_rotated_with_rotation(&value, true)
    } else {
        artifacts.decompress_value_with_rotation(&value, false)
    };
    for head in reconstructed_key.chunks_exact(head_dim) {
        k_out.extend_from_slice(head);
    }
    for head in reconstructed_value.chunks_exact(head_dim) {
        v_out.extend_from_slice(head);
    }
}

/// Fused decode attention: computes attention scores directly from packed
/// turbo3 key bytes without materializing full f32 KV tensors.
///
/// This replaces the gather+dequant+SDPA pipeline for single-token decode,
/// eliminating two large O(N × num_heads × head_dim) intermediate allocations.
///
/// Algorithm: online softmax (flash-attention style) over compressed KV cache.
fn turboquant_fused_decode(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    bits: TurboQuantBits,
    block_tables: &Tensor,
    context_lens: &Tensor,
    head_dim: usize,
    key_value_heads: usize,
    attention_heads: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    assert_eq!(bits, TurboQuantBits::Three, "fused decode only supports turbo3");
    let out_dtype = query.dtype();
    let out_device = query.device().clone();

    let (num_blocks, num_heads, key_bytes, block_size, x) = key_cache.dims5()?;
    let (_, _, value_bytes, _, _) = value_cache.dims5()?;
    let _ = num_blocks;
    assert_eq!(x, 1);
    assert_eq!(num_heads, key_value_heads);

    let key_block_bytes = num_heads * key_bytes * block_size;
    let value_block_bytes = num_heads * value_bytes * block_size;
    let packed_key_head_bytes = bits.packed_key_bytes_for_dim(head_dim);
    let packed_value_head_bytes = bits.packed_value_bytes_for_dim(head_dim);

    // Extract query to f32 on CPU: shape (1, attention_heads, 1, head_dim)
    let query_f32 = query.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let query_flat = query_f32.flatten_all()?.to_vec1::<f32>()?;
    // query_flat has attention_heads * head_dim elements

    let queries_per_kv = attention_heads / key_value_heads;

    // Rotate each query head forward (WHT).
    // For GQA: group queries that share the same KV head and rotate together.
    let mut query_rot = query_flat.clone();
    for chunk in query_rot.chunks_exact_mut(head_dim) {
        if head_dim % 128 == 0 {
            turbo3_rotate_forward_in_place(chunk);
        }
    }

    // Parse block tables and context lengths
    let block_tables_2d = tensor_to_u32_2d(block_tables)?;
    let raw_context_lens = tensor_to_u32_vec(context_lens)?;
    let context_lens_vec: Vec<u32> = if !block_tables_2d.is_empty()
        && raw_context_lens.len() > block_tables_2d.len()
        && raw_context_lens.len() % block_tables_2d.len() == 0
    {
        let row_width = raw_context_lens.len() / block_tables_2d.len();
        raw_context_lens
            .chunks(row_width)
            .map(|row| row.iter().copied().max().map(|x| x + 1).unwrap_or(0))
            .collect()
    } else {
        raw_context_lens
    };

    // Online softmax state per (sequence, attention_head)
    // For single-token decode, batch_size is typically 1
    let batch_size = block_tables_2d.len();
    let mut output = vec![0.0f32; batch_size * attention_heads * head_dim];

    let mut key_blocks: HashMap<usize, Vec<u8>> = HashMap::new();
    let mut value_blocks: HashMap<usize, Vec<u8>> = HashMap::new();

    for seq_idx in 0..batch_size {
        let context_len = context_lens_vec[seq_idx] as usize;
        if context_len == 0 {
            continue;
        }

        // Per-head online softmax accumulators
        let mut max_scores = vec![f32::NEG_INFINITY; attention_heads];
        let mut sum_exps = vec![0.0f32; attention_heads];
        let mut accumulators = vec![vec![0.0f32; head_dim]; attention_heads];

        for token_idx in 0..context_len {
            let block_number = block_tables_2d[seq_idx][token_idx / block_size] as usize;
            let block_offset = token_idx % block_size;

            // Read key block (cached)
            let key_block = key_blocks.entry(block_number).or_insert_with(|| {
                read_cuda_block_u8(key_cache, block_number, key_block_bytes)
                    .expect("failed to read turboquant key block")
            });

            // Read value block (cached alongside key block)
            let value_block = value_blocks.entry(block_number).or_insert_with(|| {
                read_cuda_block_u8(value_cache, block_number, value_block_bytes)
                    .expect("failed to read turboquant value block")
            });

            // Process each KV head: score all query heads, decompress value once
            for kv_head in 0..key_value_heads {
                // Gather packed key payload for this head
                let mut key_payload = vec![0u8; packed_key_head_bytes];
                for byte_idx in 0..key_bytes {
                    let linear = ((kv_head * key_bytes + byte_idx) * block_size) + block_offset;
                    key_payload[byte_idx] = key_block[linear];
                }

                // Decompress value ONCE per KV head (shared across all query heads)
                let mut value_payload = vec![0u8; packed_value_head_bytes];
                for byte_idx in 0..value_bytes {
                    let linear =
                        ((kv_head * value_bytes + byte_idx) * block_size) + block_offset;
                    value_payload[byte_idx] = value_block[linear];
                }
                let value_vec = turbo3_value_from_packed_rotated(&value_payload);

                // Score and accumulate for each query head sharing this KV head
                for q_local in 0..queries_per_kv {
                    let attn_head = kv_head * queries_per_kv + q_local;
                    let q_start = attn_head * head_dim;
                    let q_slice = &query_rot[q_start..q_start + head_dim];

                    // Direct score from packed bytes — no key decompression!
                    let score =
                        turbo3_score_from_packed(q_slice, &key_payload) * softmax_scale;

                    // Online softmax update
                    let old_max = max_scores[attn_head];
                    let new_max = old_max.max(score);
                    let correction = (old_max - new_max).exp();
                    let new_exp = (score - new_max).exp();

                    // Rescale existing accumulator
                    sum_exps[attn_head] = sum_exps[attn_head] * correction + new_exp;
                    for v in accumulators[attn_head].iter_mut() {
                        *v *= correction;
                    }
                    max_scores[attn_head] = new_max;

                    // Accumulate weighted value
                    for (acc_v, &val) in
                        accumulators[attn_head].iter_mut().zip(value_vec.iter())
                    {
                        *acc_v += new_exp * val;
                    }
                }
            }
        }

        // Normalize accumulators
        let out_base = seq_idx * attention_heads * head_dim;
        for attn_head in 0..attention_heads {
            let inv_sum = if sum_exps[attn_head] > 0.0 {
                1.0 / sum_exps[attn_head]
            } else {
                0.0
            };
            let head_out = &mut output
                [out_base + attn_head * head_dim..out_base + (attn_head + 1) * head_dim];
            for (o, &a) in head_out.iter_mut().zip(accumulators[attn_head].iter()) {
                *o = a * inv_sum;
            }
        }
    }

    // Build output tensor: (batch_size, attention_heads, 1, head_dim) to match query shape
    let out_tensor =
        Tensor::from_vec(output, (batch_size, attention_heads, head_dim), &Device::Cpu)?;

    // Rotate output back from rotated space
    let out_rotated = rotate_turbo3_attention_tensor(&out_tensor, key_value_heads, false)?;

    // Reshape to match expected (batch, heads, 1, head_dim) output
    let out_final = out_rotated
        .reshape((batch_size, attention_heads, 1, head_dim))?
        .to_dtype(out_dtype)?
        .to_device(&out_device)?;

    Ok(out_final)
}

fn turboquant_gather_kv_cache(
    key_cache: &Tensor,
    value_cache: &Tensor,
    bits: TurboQuantBits,
    block_tables: Option<&Tensor>,
    context_lens: Option<&Tensor>,
    paged_kv_indptr: Option<&Tensor>,
    paged_kv_indices: Option<&Tensor>,
    paged_kv_last_page_len: Option<&Tensor>,
    out_device: &Device,
    out_dtype: DType,
    head_dim: usize,
) -> Result<(Tensor, Tensor, Tensor, u32)> {
    let (num_blocks, num_heads, key_bytes, block_size, x) = key_cache.dims5()?;
    let (_, _, value_bytes, value_block_size, value_x) = value_cache.dims5()?;
    if x != 1 || value_x != 1 || block_size != value_block_size {
        candle_core::bail!("unsupported turboquant cache layout");
    }
    let _ = num_blocks;
    let head_dim = infer_turboquant_head_dim_for_bits(key_cache, value_cache, bits, head_dim)?;
    if !bits.supports_storage_head_dim(head_dim) {
        candle_core::bail!(
            "{} cache requires head_dim to be a multiple of 32, got {head_dim}",
            bits.cache_type_name()
        );
    }
    let artifacts = TurboQuantArtifacts::new(head_dim, bits, 0xC701_0ACE);
    let turbo_rotated = turboquant_uses_rotated_space(bits, head_dim);
    let flat_artifacts = (bits == TurboQuantBits::Three && turbo_rotated)
        .then(|| TurboQuantArtifacts::new(num_heads * head_dim, bits, 0xC701_0ACE));
    let key_block_bytes = num_heads * key_bytes * block_size;
    let value_block_bytes = num_heads * value_bytes * block_size;
    let mut cu_kv = Vec::new();
    cu_kv.push(0u32);
    let mut running = 0u32;
    let mut max_k = 0u32;
    let mut k_out = Vec::new();
    let mut v_out = Vec::new();
    let mut key_blocks = HashMap::new();
    let mut value_blocks = HashMap::new();

    if let (Some(indptr), Some(indices), Some(last_page_len)) =
        (paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len)
    {
        let indptr = tensor_to_i32_vec(indptr)?;
        let indices = tensor_to_i32_vec(indices)?;
        let last_page_len = tensor_to_i32_vec(last_page_len)?;
        for seq_idx in 0..last_page_len.len() {
            let page_start = indptr[seq_idx] as usize;
            let page_end = indptr[seq_idx + 1] as usize;
            let num_pages = page_end.saturating_sub(page_start);
            let context_len = if num_pages == 0 {
                0usize
            } else {
                (num_pages - 1) * block_size + last_page_len[seq_idx] as usize
            };
            max_k = max_k.max(context_len as u32);
            for token_idx in 0..context_len {
                let block_number = indices[page_start + (token_idx / block_size)] as usize;
                let block_offset = token_idx % block_size;
                let key_block = key_blocks.entry(block_number).or_insert_with(|| {
                    read_cuda_block_u8(key_cache, block_number, key_block_bytes)
                        .expect("failed to read turboquant key block")
                });
                let value_block = value_blocks.entry(block_number).or_insert_with(|| {
                    read_cuda_block_u8(value_cache, block_number, value_block_bytes)
                        .expect("failed to read turboquant value block")
                });
                if bits == TurboQuantBits::Three && turbo_rotated {
                    let flat_artifacts = flat_artifacts.as_ref().unwrap();
                    let mut flat_key_payload = vec![0u8; num_heads * key_bytes];
                    let mut flat_value_payload = vec![0u8; num_heads * value_bytes];
                    for head in 0..num_heads {
                        let key_start = head * key_bytes;
                        let value_start = head * value_bytes;
                        for byte_idx in 0..key_bytes {
                            let linear =
                                ((head * key_bytes + byte_idx) * block_size) + block_offset;
                            flat_key_payload[key_start + byte_idx] = key_block[linear];
                        }
                        for byte_idx in 0..value_bytes {
                            let linear =
                                ((head * value_bytes + byte_idx) * block_size) + block_offset;
                            flat_value_payload[value_start + byte_idx] = value_block[linear];
                        }
                    }
                    turboquant_decode_flat_token(
                        flat_artifacts,
                        bits,
                        head_dim,
                        &flat_key_payload,
                        &flat_value_payload,
                        &mut k_out,
                        &mut v_out,
                    );
                } else {
                    for head in 0..num_heads {
                        let mut key_payload = vec![0u8; key_bytes];
                        let mut value_payload = vec![0u8; value_bytes];
                        for byte_idx in 0..key_bytes {
                            let linear =
                                ((head * key_bytes + byte_idx) * block_size) + block_offset;
                            key_payload[byte_idx] = key_block[linear];
                        }
                        for byte_idx in 0..value_bytes {
                            let linear =
                                ((head * value_bytes + byte_idx) * block_size) + block_offset;
                            value_payload[byte_idx] = value_block[linear];
                        }
                        let key = artifacts.unpack_key(&key_payload);
                        let value = artifacts.unpack_value(&value_payload);
                        if turbo_rotated {
                            k_out.extend(
                                artifacts.reconstruct_key_mse_rotated_with_rotation(&key, true),
                            );
                            v_out.extend(
                                artifacts.decompress_value_rotated_with_rotation(&value, true),
                            );
                        } else {
                            k_out.extend(artifacts.reconstruct_key_mse_with_rotation(&key, false));
                            v_out.extend(artifacts.decompress_value_with_rotation(&value, false));
                        }
                    }
                }
            }
            running += context_len as u32;
            cu_kv.push(running);
        }
    } else {
        let block_tables_tensor = block_tables
            .ok_or_else(|| candle_core::Error::msg("missing turboquant block_tables"))?;
        let block_tables = tensor_to_u32_2d(block_tables_tensor)?;
        let context_lens_tensor = context_lens
            .ok_or_else(|| candle_core::Error::msg("missing turboquant context_lens"))?;
        let context_lens = if context_lens_tensor.dims().len() == 2 {
            tensor_to_u32_2d(context_lens_tensor)?
                .into_iter()
                .map(|row| row.into_iter().max().map(|x| x + 1).unwrap_or(0))
                .collect::<Vec<_>>()
        } else {
            let raw_context_lens = tensor_to_u32_vec(context_lens_tensor)?;
            if raw_context_lens.len() == block_tables.len() {
                raw_context_lens
            } else if !block_tables.is_empty()
                && raw_context_lens.len() > block_tables.len()
                && raw_context_lens.len() % block_tables.len() == 0
            {
                let row_width = raw_context_lens.len() / block_tables.len();
                raw_context_lens
                    .chunks(row_width)
                    .map(|row| row.iter().copied().max().map(|x| x + 1).unwrap_or(0))
                    .collect::<Vec<_>>()
            } else {
                raw_context_lens
            }
        };
        if turboquant_uses_rotated_space(bits, head_dim)
            && turboquant_native_rotated_cuda_supported(bits, head_dim)
        {
            let cu_kv_tensor = Tensor::new(cu_kv_from_context_lens(&context_lens), out_device)?;
            let (k, v) = engine_paged_attn::gather_kv_cache(
                key_cache,
                value_cache,
                None,
                None,
                block_tables_tensor,
                &cu_kv_tensor,
                out_dtype,
            )?;
            max_k = context_lens.iter().copied().max().unwrap_or(0);
            return Ok((k, v, cu_kv_tensor, max_k));
        }
        let artifacts = TurboQuantArtifacts::new(head_dim, bits, 0xC701_0ACE);
        for (seq_idx, &context_len) in context_lens.iter().enumerate() {
            max_k = max_k.max(context_len);
            for token_idx in 0..context_len as usize {
                let block_number = block_tables[seq_idx][token_idx / block_size] as usize;
                let block_offset = token_idx % block_size;
                let key_block = key_blocks.entry(block_number).or_insert_with(|| {
                    read_cuda_block_u8(key_cache, block_number, key_block_bytes)
                        .expect("failed to read turboquant key block")
                });
                let value_block = value_blocks.entry(block_number).or_insert_with(|| {
                    read_cuda_block_u8(value_cache, block_number, value_block_bytes)
                        .expect("failed to read turboquant value block")
                });
                if bits == TurboQuantBits::Three && turbo_rotated {
                    let flat_artifacts = flat_artifacts.as_ref().unwrap();
                    let mut flat_key_payload = vec![0u8; num_heads * key_bytes];
                    let mut flat_value_payload = vec![0u8; num_heads * value_bytes];
                    for head in 0..num_heads {
                        let key_start = head * key_bytes;
                        let value_start = head * value_bytes;
                        for byte_idx in 0..key_bytes {
                            let linear =
                                ((head * key_bytes + byte_idx) * block_size) + block_offset;
                            flat_key_payload[key_start + byte_idx] = key_block[linear];
                        }
                        for byte_idx in 0..value_bytes {
                            let linear =
                                ((head * value_bytes + byte_idx) * block_size) + block_offset;
                            flat_value_payload[value_start + byte_idx] = value_block[linear];
                        }
                    }
                    turboquant_decode_flat_token(
                        flat_artifacts,
                        bits,
                        head_dim,
                        &flat_key_payload,
                        &flat_value_payload,
                        &mut k_out,
                        &mut v_out,
                    );
                } else {
                    for head in 0..num_heads {
                        let mut key_payload = vec![0u8; key_bytes];
                        let mut value_payload = vec![0u8; value_bytes];
                        for byte_idx in 0..key_bytes {
                            let linear =
                                ((head * key_bytes + byte_idx) * block_size) + block_offset;
                            key_payload[byte_idx] = key_block[linear];
                        }
                        for byte_idx in 0..value_bytes {
                            let linear =
                                ((head * value_bytes + byte_idx) * block_size) + block_offset;
                            value_payload[byte_idx] = value_block[linear];
                        }
                        let key = artifacts.unpack_key(&key_payload);
                        let value = artifacts.unpack_value(&value_payload);
                        if turbo_rotated {
                            k_out.extend(
                                artifacts.reconstruct_key_mse_rotated_with_rotation(&key, true),
                            );
                            v_out.extend(
                                artifacts.decompress_value_rotated_with_rotation(&value, true),
                            );
                        } else {
                            k_out.extend(artifacts.reconstruct_key_mse_with_rotation(&key, false));
                            v_out.extend(artifacts.decompress_value_with_rotation(&value, false));
                        }
                    }
                }
            }
            running += context_len;
            cu_kv.push(running);
        }
    }

    let total_tokens = running as usize;
    let k = Tensor::from_vec(k_out, (total_tokens, num_heads, head_dim), &Device::Cpu)?
        .to_dtype(out_dtype)?
        .to_device(out_device)?;
    let v = Tensor::from_vec(v_out, (total_tokens, num_heads, head_dim), &Device::Cpu)?
        .to_dtype(out_dtype)?
        .to_device(out_device)?;
    let cu_kv = Tensor::new(cu_kv, out_device)?;
    Ok((k, v, cu_kv, max_k))
}

fn cu_kv_from_context_lens(context_lens: &[u32]) -> Vec<u32> {
    let mut cu_kv = Vec::with_capacity(context_lens.len() + 1);
    let mut running = 0u32;
    cu_kv.push(0);
    for &len in context_lens {
        running += len;
        cu_kv.push(running);
    }
    cu_kv
}

fn rotate_turbo3_tensor(t: &Tensor, forward: bool) -> Result<Tensor> {
    let orig_dtype = t.dtype();
    let orig_device = t.device().clone();
    match t.shape().dims() {
        [_, _, _] => {}
        [_, _, _, _] => {}
        dims => {
            candle_core::bail!(
                "turbo3 rotation expects rank-3 or rank-4 tensor, got rank {} ({dims:?})",
                dims.len()
            )
        }
    }
    let layout_aligned = if t.shape().dims().len() == 4 {
        t.transpose(1, 2)?.contiguous()?
    } else {
        t.clone()
    };
    #[cfg(feature = "cuda")]
    if layout_aligned.device().is_cuda() {
        let rotated = turbo_rotate(&layout_aligned, forward)?;
        return if t.shape().dims().len() == 4 {
            rotated.transpose(1, 2)
        } else {
            Ok(rotated)
        };
    }
    let aligned_shape = layout_aligned.shape().clone();
    let flat_tensor = layout_aligned
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .contiguous()?
        .flatten_all()?;
    let mut flat = flat_tensor.to_vec1::<f32>()?;
    if flat.len() % 128 != 0 {
        candle_core::bail!(
            "turbo3 rotation expects flattened tensor element count to be a multiple of 128, got {}",
            flat.len()
        );
    }
    if forward {
        turbo3_rotate_forward_in_place(&mut flat);
    } else {
        turbo3_rotate_inverse_in_place(&mut flat);
    }
    let rotated = Tensor::from_vec(flat, aligned_shape, &Device::Cpu)?
        .to_dtype(orig_dtype)?
        .to_device(&orig_device)?;
    if t.shape().dims().len() == 4 {
        rotated.transpose(1, 2)
    } else {
        Ok(rotated)
    }
}

fn rotate_turbo3_attention_tensor(
    t: &Tensor,
    key_value_heads: usize,
    forward: bool,
) -> Result<Tensor> {
    let (attention_heads, head_dim, rank4) = match t.shape().dims() {
        [_, heads, dim] => (*heads, *dim, false),
        [_, heads, _, dim] => (*heads, *dim, true),
        dims => {
            candle_core::bail!(
                "turbo3 attention rotation expects rank-3 or rank-4 tensor, got rank {} ({dims:?})",
                dims.len()
            )
        }
    };
    if attention_heads == key_value_heads || head_dim >= 128 {
        return rotate_turbo3_tensor(t, forward);
    }
    if attention_heads % key_value_heads != 0 {
        candle_core::bail!(
            "turbo3 attention rotation expects attention_heads divisible by key_value_heads, got {attention_heads} vs {key_value_heads}"
        );
    }
    if 128 % head_dim != 0 {
        candle_core::bail!(
            "turbo3 attention rotation expects head_dim to divide 128 for grouped GQA rotation, got {head_dim}"
        );
    }
    let group_heads = 128 / head_dim;
    if key_value_heads % group_heads != 0 {
        candle_core::bail!(
            "turbo3 attention rotation expects key_value_heads divisible by group_heads, got {key_value_heads} vs {group_heads}"
        );
    }

    let orig_dtype = t.dtype();
    let orig_device = t.device().clone();
    let layout_aligned = if rank4 {
        t.transpose(1, 2)?.contiguous()?
    } else {
        t.clone()
    };
    let aligned_shape = layout_aligned.shape().clone();
    let flat_tensor = layout_aligned
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .contiguous()?
        .flatten_all()?;
    let mut flat = flat_tensor.to_vec1::<f32>()?;
    let queries_per_kv = attention_heads / key_value_heads;
    let token_width = attention_heads * head_dim;
    if flat.len() % token_width != 0 {
        candle_core::bail!(
            "turbo3 attention rotation encountered invalid flattened token width: total={} token_width={token_width}",
            flat.len()
        );
    }
    for token in flat.chunks_exact_mut(token_width) {
        for kv_group in 0..(key_value_heads / group_heads) {
            for local_query in 0..queries_per_kv {
                let mut block = Vec::with_capacity(128);
                for gh in 0..group_heads {
                    let kv_head = kv_group * group_heads + gh;
                    let attn_head = kv_head * queries_per_kv + local_query;
                    let start = attn_head * head_dim;
                    block.extend_from_slice(&token[start..start + head_dim]);
                }
                if forward {
                    turbo3_rotate_forward_in_place(&mut block);
                } else {
                    turbo3_rotate_inverse_in_place(&mut block);
                }
                for gh in 0..group_heads {
                    let kv_head = kv_group * group_heads + gh;
                    let attn_head = kv_head * queries_per_kv + local_query;
                    let start = attn_head * head_dim;
                    let block_start = gh * head_dim;
                    token[start..start + head_dim]
                        .copy_from_slice(&block[block_start..block_start + head_dim]);
                }
            }
        }
    }

    let rotated = Tensor::from_vec(flat, aligned_shape, &Device::Cpu)?
        .to_dtype(orig_dtype)?
        .to_device(&orig_device)?;
    if rank4 {
        rotated.transpose(1, 2)
    } else {
        Ok(rotated)
    }
}

fn tensor_to_i64_vec(t: &Tensor) -> Result<Vec<i64>> {
    match t.dtype() {
        DType::I64 => t.to_device(&Device::Cpu)?.to_vec1::<i64>(),
        DType::I32 => Ok(t
            .to_device(&Device::Cpu)?
            .to_vec1::<i32>()?
            .into_iter()
            .map(i64::from)
            .collect()),
        DType::U32 => Ok(t
            .to_device(&Device::Cpu)?
            .to_vec1::<u32>()?
            .into_iter()
            .map(|x| x as i64)
            .collect()),
        other => candle_core::bail!("unsupported slot mapping dtype for turboquant: {other:?}"),
    }
}

fn tensor_to_i32_vec(t: &Tensor) -> Result<Vec<i32>> {
    match t.dtype() {
        DType::I32 => t.to_device(&Device::Cpu)?.to_vec1::<i32>(),
        DType::U32 => Ok(t
            .to_device(&Device::Cpu)?
            .to_vec1::<u32>()?
            .into_iter()
            .map(|x| x as i32)
            .collect()),
        other => candle_core::bail!("unsupported tensor dtype for turboquant i32 vec: {other:?}"),
    }
}

fn tensor_to_u32_vec(t: &Tensor) -> Result<Vec<u32>> {
    match t.dtype() {
        DType::U32 => t.to_device(&Device::Cpu)?.to_vec1::<u32>(),
        DType::I32 => Ok(t
            .to_device(&Device::Cpu)?
            .to_vec1::<i32>()?
            .into_iter()
            .map(|x| x as u32)
            .collect()),
        other => candle_core::bail!("unsupported tensor dtype for turboquant u32 vec: {other:?}"),
    }
}

fn tensor_to_u32_2d(t: &Tensor) -> Result<Vec<Vec<u32>>> {
    match t.dtype() {
        DType::U32 => t.to_device(&Device::Cpu)?.to_vec2::<u32>(),
        DType::I32 => Ok(t
            .to_device(&Device::Cpu)?
            .to_vec2::<i32>()?
            .into_iter()
            .map(|row| row.into_iter().map(|x| x as u32).collect())
            .collect()),
        other => {
            candle_core::bail!("unsupported tensor dtype for turboquant u32 matrix: {other:?}")
        }
    }
}

#[cfg(feature = "cuda")]
fn read_cuda_block_u8(t: &Tensor, block_number: usize, block_bytes: usize) -> Result<Vec<u8>> {
    let (storage, layout) = t.storage_and_layout();
    let storage = match &*storage {
        Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("turboquant cache currently requires cuda storage"),
    };
    let slice = storage.as_cuda_slice::<u8>()?;
    let start = layout.start_offset() + block_number * block_bytes;
    let view = slice.slice(start..start + block_bytes);
    let mut host = vec![0u8; block_bytes];
    storage.device().memcpy_dtoh(&view, &mut host)?;
    Ok(host)
}

#[cfg(not(feature = "cuda"))]
fn read_cuda_block_u8(_t: &Tensor, _block_number: usize, _block_bytes: usize) -> Result<Vec<u8>> {
    candle_core::bail!("turboquant paged-KV helpers are CUDA-only; build with --features cuda");
}

#[cfg(feature = "cuda")]
fn write_cuda_block_u8(t: &Tensor, block_number: usize, data: &[u8]) -> Result<()> {
    use candle_core::cuda_backend::WrapErr;

    let (storage, layout) = t.storage_and_layout();
    let storage = match &*storage {
        Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("turboquant cache currently requires cuda storage"),
    };
    let slice = storage.as_cuda_slice::<u8>()?;
    let start = layout.start_offset() + block_number * data.len();
    let stream = storage.device().cuda_stream();
    let view = slice.slice(start..start + data.len());
    let (dst_ptr, _guard) = view.device_ptr(&stream);
    unsafe { result::memcpy_htod_async(dst_ptr, data, stream.cu_stream()) }.w()?;
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn write_cuda_block_u8(_t: &Tensor, _block_number: usize, _data: &[u8]) -> Result<()> {
    candle_core::bail!("turboquant paged-KV helpers are CUDA-only; build with --features cuda");
}

#[cfg(test)]
mod tests {
    use super::{
        turboquant_native_rotated_cuda_supported, turboquant_should_use_windowed_paged_kv_metadata,
    };
    use crate::TurboQuantBits;

    #[test]
    fn turboquant_windowed_paged_kv_metadata_is_disabled_for_full_attention_layers() {
        assert!(!turboquant_should_use_windowed_paged_kv_metadata(true));
    }

    #[test]
    fn turboquant_windowed_paged_kv_metadata_is_enabled_for_sliding_layers() {
        assert!(turboquant_should_use_windowed_paged_kv_metadata(false));
    }

    #[test]
    fn turboquant_native_rotated_cuda_support_excludes_unsupported_head_dims() {
        assert!(turboquant_native_rotated_cuda_supported(
            TurboQuantBits::Three,
            128
        ));
        assert!(turboquant_native_rotated_cuda_supported(
            TurboQuantBits::Three,
            256
        ));
        assert!(!turboquant_native_rotated_cuda_supported(
            TurboQuantBits::Three,
            512
        ));
    }
}
