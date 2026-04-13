//! CPU/Metal implementation of indexed MoE forward for GGUF quantized weights.
//!
//! This dequantizes the weights and delegates to UnquantLinear's gather_forward.

use candle_core::{
    quantized::{QMatMul, QTensor},
    Device, Result, Tensor,
};
use candle_nn::Linear;
use std::sync::Arc;

use crate::{QuantMethod, QuantMethodConfig, UnquantLinear};

fn normalize_indexed_moe_input(x: &Tensor) -> Result<Tensor> {
    match x.dims() {
        &[_num_tokens, _hidden_dim] => x.unsqueeze(1),
        _ => Ok(x.clone()),
    }
}

/// Perform indexed MoE forward pass on a QTensor by dequantizing and using UnquantLinear.
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

    // Dequantize all weights to f32
    let weights = qtensor.dequantize(&cpu)?;

    // Create an UnquantLinear and use its gather_forward
    let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(weights, None)))?;

    let result = unquant.gather_forward(&x, &ids)?;
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
