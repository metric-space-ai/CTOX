//! CPU/Metal implementation of indexed MoE forward for GGUF quantized weights.
//!
//! Uses per-expert quantized matmul via QMatMul::forward to avoid full
//! dequantization of all experts. Only the selected experts are processed,
//! and each uses candle's fused GGML kernels (SIMD vec_dot on CPU).

use candle_core::{
    quantized::{GgmlDType, QMatMul, QStorage, QTensor},
    Device, IndexOp, Module, Result, Shape, Tensor,
};
use candle_nn::Linear;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use crate::{QuantMethod, QuantMethodConfig, UnquantLinear};

fn normalize_indexed_moe_input(x: &Tensor) -> Result<Tensor> {
    match x.dims() {
        &[_num_tokens, _hidden_dim] => x.unsqueeze(1),
        _ => Ok(x.clone()),
    }
}

/// Extract a single expert's quantized data from the combined QTensor and
/// create a per-expert QTensor without dequantizing.
///
/// The combined QTensor has shape [num_experts, out_features, in_features].
/// Each expert's block-quantized data is contiguous in memory.
fn extract_expert_qtensor(
    all_data: &[u8],
    expert_idx: usize,
    out_features: usize,
    in_features: usize,
    dtype: GgmlDType,
    device: &Device,
) -> Result<Arc<QTensor>> {
    let block_size = dtype.block_size();
    let type_size = dtype.type_size();

    // Total quantized values per expert
    let values_per_expert = out_features * in_features;
    let blocks_per_expert = values_per_expert / block_size;
    let bytes_per_expert = blocks_per_expert * type_size;

    let start = expert_idx * bytes_per_expert;
    let end = start + bytes_per_expert;
    let expert_data = &all_data[start..end];

    let storage = QStorage::from_data(Cow::Owned(expert_data.to_vec()), device, dtype)?;
    let shape = Shape::from_dims(&[out_features, in_features]);
    Ok(Arc::new(QTensor::new(storage, shape)?))
}

/// Perform indexed MoE forward pass on a QTensor using per-expert quantized matmul.
///
/// Instead of dequantizing all experts to f32, this extracts only the selected
/// experts as QTensors and uses QMatMul::forward (which uses candle's fused
/// GGML SIMD kernels on CPU).
///
/// # Arguments
/// * `qtensor` - The quantized weight tensor [num_experts, n, k]
/// * `x` - Input tensor [batch, topk_or_1, k]
/// * `ids` - Expert indices tensor [batch, topk]
///
/// # Returns
/// Output tensor [batch, topk, n]
pub fn qtensor_indexed_moe_forward(
    qtensor: &Arc<QTensor>,
    x: &Tensor,
    ids: &Tensor,
) -> Result<Tensor> {
    let output_device = x.device().clone();
    let cpu = Device::Cpu;
    let x = normalize_indexed_moe_input(x)?;
    let x = if x.device().is_cpu() {
        x
    } else {
        x.to_device(&cpu)?
    };
    let ids = if ids.device().is_cpu() {
        ids.clone()
    } else {
        ids.to_device(&cpu)?
    };

    let shape = qtensor.shape();
    let (_num_experts, out_features, in_features) = shape.dims3()?;
    let (num_tokens, topk) = ids.dims2()?;
    let dtype = qtensor.dtype();

    // Get raw quantized data once
    let all_data = qtensor.data()?;

    // Collect unique expert indices to avoid redundant QTensor creation
    let ids_vec: Vec<u32> = ids.flatten_all()?.to_vec1()?;
    let mut expert_cache: HashMap<u32, Arc<QTensor>> = HashMap::new();

    for &eid in &ids_vec {
        if !expert_cache.contains_key(&eid) {
            let expert_q =
                extract_expert_qtensor(&all_data, eid as usize, out_features, in_features, dtype, &cpu)?;
            expert_cache.insert(eid, expert_q);
        }
    }

    // Compute per-token, per-expert outputs using quantized matmul
    let mut results = Vec::with_capacity(num_tokens * topk);

    for token_idx in 0..num_tokens {
        for slot_idx in 0..topk {
            let expert_idx = ids_vec[token_idx * topk + slot_idx];
            let expert_q = expert_cache.get(&expert_idx).unwrap();

            // x_slice: [1, in_features]
            let x_slice = x.i((token_idx, 0..1, ..))?.squeeze(0)?;

            // QMatMul::forward computes x @ W^T using fused GGML kernels
            let qmm = QMatMul::from_arc(expert_q.clone())?;
            let out = qmm.forward(&x_slice)?; // [1, out_features]
            results.push(out);
        }
    }

    // Stack results: [num_tokens * topk, out_features] -> [num_tokens, topk, out_features]
    let stacked = Tensor::cat(&results, 0)?;
    let result = stacked.reshape((num_tokens, topk, out_features))?;

    if output_device.is_cpu() {
        Ok(result)
    } else {
        result.to_device(&output_device)
    }
}

/// Perform indexed MoE forward pass on a QMatMul.
///
/// This is the main entry point for CPU/Metal GGUF quantized MoE forward.
///
/// # Arguments
/// * `qmatmul` - The quantized weight matrix
/// * `x` - Input tensor [batch, topk_or_1, k]
/// * `ids` - Expert indices tensor [batch, topk]
///
/// # Returns
/// Output tensor [batch, topk, n]
pub fn cpu_indexed_moe_forward(qmatmul: &QMatMul, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let output_device = x.device().clone();
    let cpu = Device::Cpu;
    let x = normalize_indexed_moe_input(x)?;
    let x = if x.device().is_cpu() {
        x
    } else {
        x.to_device(&cpu)?
    };
    let ids = if ids.device().is_cpu() {
        ids.clone()
    } else {
        ids.to_device(&cpu)?
    };
    let result = match qmatmul {
        QMatMul::QTensor(qtensor) => qtensor_indexed_moe_forward(qtensor, &x, &ids),
        QMatMul::Tensor(t) | QMatMul::TensorF16(t) => {
            // For non-quantized tensors, use UnquantLinear directly
            let weight = if t.device().is_cpu() {
                t.clone()
            } else {
                t.to_device(&cpu)?
            };
            let unquant =
                UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(weight, None)))?;
            unquant.gather_forward(&x, &ids)
        }
    }?;

    if output_device.is_cpu() {
        Ok(result)
    } else {
        result.to_device(&output_device)
    }
}
