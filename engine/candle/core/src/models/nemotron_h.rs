#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Linear, Module};
use mistralrs_quant::{
    ColumnParallelLayer, ImmediateIsqMatch, MatMul, QuantMethod, QuantMethodConfig,
    QuantizedConfig, ReplicatedLayer, RowParallelLayer, ShardedVarBuilder, SumAllReduce, UnquantLinear,
};
use serde::{Deserialize, Serialize};

use crate::{
    amoe::{AnyMoeBaseModelMixin, MlpLayer},
    attention::SdpaParams,
    device_map::{DeviceMappedMask, DeviceMapper},
    kv_cache::{
        HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, KvCache,
        RecurrentLayerConfig,
    },
    layers::{embedding, CausalMasker, RmsNorm, RotaryEmbedding, Sdpa},
    layers_masker::PastKvLenCache,
    moe::shard,
    ops::TopKLastDimOp,
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata, NormalModel,
    },
    utils::unvarbuilder::UnVarBuilder,
};

fn default_rope_theta() -> f32 {
    10_000.0
}

fn default_layer_norm_epsilon() -> f64 {
    1e-5
}

fn default_true() -> bool {
    true
}

fn default_false() -> bool {
    false
}

fn default_silu() -> crate::layers::Activation {
    crate::layers::Activation::Silu
}

fn default_relu2() -> crate::layers::Activation {
    crate::layers::Activation::Relu2
}

fn default_ssm_state_size() -> usize {
    128
}

fn default_n_groups() -> usize {
    8
}

fn default_conv_kernel() -> usize {
    4
}

fn default_expand() -> usize {
    2
}

fn default_time_step_min() -> f64 {
    0.001
}

fn default_time_step_max() -> f64 {
    0.1
}

fn default_time_step_floor() -> f64 {
    1e-4
}

fn default_chunk_size() -> usize {
    128
}

fn default_num_experts_per_tok() -> usize {
    2
}

fn default_n_group() -> usize {
    1
}

fn default_topk_group() -> usize {
    1
}

fn default_routed_scaling_factor() -> f32 {
    1.0
}

fn default_default_layers_block_type() -> Vec<NemotronLayerType> {
    vec![
        NemotronLayerType::Mamba,
        NemotronLayerType::Moe,
        NemotronLayerType::Attention,
        NemotronLayerType::Moe,
    ]
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum NemotronLayerType {
    Mamba,
    Attention,
    Moe,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    #[serde(default)]
    pub layers_block_type: Option<Vec<NemotronLayerType>>,
    #[serde(default)]
    pub hybrid_override_pattern: Option<String>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_false")]
    pub attention_bias: bool,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f64,
    #[serde(default = "default_false")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_false")]
    pub use_bias: bool,
    #[serde(default = "default_false")]
    pub mlp_bias: bool,
    #[serde(default = "default_relu2")]
    pub mlp_hidden_act: crate::layers::Activation,
    pub intermediate_size: usize,
    pub mamba_num_heads: usize,
    pub mamba_head_dim: usize,
    #[serde(default = "default_silu")]
    pub mamba_hidden_act: crate::layers::Activation,
    #[serde(default = "default_ssm_state_size", alias = "mamba_d_state")]
    pub ssm_state_size: usize,
    #[serde(default = "default_n_groups", alias = "mamba_n_groups")]
    pub n_groups: usize,
    #[serde(default = "default_conv_kernel", alias = "mamba_d_conv")]
    pub conv_kernel: usize,
    #[serde(default = "default_expand", alias = "mamba_expand")]
    pub expand: usize,
    #[serde(default = "default_time_step_min", alias = "mamba_dt_min")]
    pub time_step_min: f64,
    #[serde(default = "default_time_step_max", alias = "mamba_dt_max")]
    pub time_step_max: f64,
    #[serde(default)]
    pub time_step_limit: Option<(f64, f64)>,
    #[serde(default = "default_time_step_floor", alias = "mamba_dt_init_floor")]
    pub time_step_floor: f64,
    #[serde(default = "default_true", alias = "mamba_conv_bias")]
    pub use_conv_bias: bool,
    #[serde(default = "default_chunk_size", alias = "mamba_chunk_size")]
    pub chunk_size: usize,
    #[serde(default = "default_false")]
    pub mamba_proj_bias: bool,
    pub n_routed_experts: usize,
    #[serde(default)]
    pub n_shared_experts: usize,
    pub moe_intermediate_size: usize,
    pub moe_shared_expert_intermediate_size: usize,
    #[serde(default)]
    pub moe_latent_size: Option<usize>,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default = "default_n_group")]
    pub n_group: usize,
    #[serde(default = "default_topk_group")]
    pub topk_group: usize,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
}

impl Config {
    pub fn layer_types(&self) -> Result<Vec<NemotronLayerType>> {
        if let Some(types) = &self.layers_block_type {
            return Ok(types.clone());
        }

        if let Some(pattern) = &self.hybrid_override_pattern {
            return pattern
                .chars()
                .map(|ch| match ch {
                    'M' => Ok(NemotronLayerType::Mamba),
                    'E' => Ok(NemotronLayerType::Moe),
                    '*' => Ok(NemotronLayerType::Attention),
                    other => candle_core::bail!(
                        "Unsupported Nemotron hybrid_override_pattern token `{other}`."
                    ),
                })
                .collect();
        }

        Ok(default_default_layers_block_type())
    }

    pub fn num_hidden_layers(&self) -> Result<usize> {
        Ok(self.layer_types()?.len())
    }

    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn mamba_intermediate_size(&self) -> usize {
        self.mamba_num_heads * self.mamba_head_dim
    }

    pub fn mamba_conv_dim(&self) -> usize {
        self.mamba_intermediate_size() + 2 * self.n_groups * self.ssm_state_size
    }
}

#[derive(Clone)]
struct SharedMlp {
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act_fn: crate::layers::Activation,
}

impl SharedMlp {
    fn new(
        vb: ShardedVarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        quant_config: &Option<QuantizedConfig>,
        act_fn: crate::layers::Activation,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let up_proj = ColumnParallelLayer::new(
            hidden_size,
            intermediate_size,
            quant_config,
            false,
            comm,
            vb.pp("up_proj"),
        )?;
        let down_proj = RowParallelLayer::new(
            intermediate_size,
            hidden_size,
            quant_config,
            false,
            comm,
            vb.pp("down_proj"),
        )?;
        Ok(Self {
            up_proj,
            down_proj,
            act_fn,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.up_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let up = MatMul.qmethod_matmul(&xs, &*self.up_proj)?;
        let activated = self.act_fn.forward(&up)?;
        let mut res = MatMul.qmethod_matmul(&activated, &*self.down_proj)?;
        if self.up_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        vec![&mut self.up_proj, &mut self.down_proj]
    }
}

#[derive(Clone)]
struct RoutedExperts {
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act_fn: crate::layers::Activation,
    all_reduce: SumAllReduce,
    world_size: usize,
}

impl RoutedExperts {
    fn apply_routed_expert_isq(
        layer: Arc<dyn QuantMethod>,
        vb: ShardedVarBuilder,
        target_device: &Device,
    ) -> Result<Arc<dyn QuantMethod>> {
        let Some(params) = mistralrs_quant::get_immediate_isq() else {
            return Ok(layer);
        };
        let Some(ImmediateIsqMatch { ty, device }) = mistralrs_quant::immediate_isq_match(&vb)
        else {
            return Ok(layer);
        };

        let device = device.unwrap_or_else(|| target_device.clone());
        layer.apply_isq(
            Some(ty),
            device,
            &std::sync::atomic::AtomicUsize::new(0),
            None,
            params.guard.clone(),
        )
    }

    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        target_device: &Device,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let input_dim = cfg.moe_latent_size.unwrap_or(cfg.hidden_size);
        let num_experts = cfg.n_routed_experts;
        let experts_vb = vb.pp("experts");
        let load_experts_vb =
            if mistralrs_quant::get_immediate_isq().is_some() && !experts_vb.device().is_cpu() {
                experts_vb.clone().set_device(Device::Cpu)
            } else {
                experts_vb.clone()
            };

        let mut up_proj_weights = Vec::with_capacity(num_experts);
        let mut down_proj_weights = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            let expert_vb = load_experts_vb.pp(i);
            let up = expert_vb.get_with_hints(
                (cfg.moe_intermediate_size, input_dim),
                "up_proj.weight",
                shard(0, comm.rank(), comm.world_size()),
            )?;
            let down = expert_vb.get_with_hints(
                (input_dim, cfg.moe_intermediate_size),
                "down_proj.weight",
                shard(1, comm.rank(), comm.world_size()),
            )?;
            up_proj_weights.push(up);
            down_proj_weights.push(down);
        }

        let mut up_proj: Arc<dyn QuantMethod> = Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(Linear::new(Tensor::stack(&up_proj_weights, 0)?, None)),
        )?);
        let mut down_proj: Arc<dyn QuantMethod> =
            Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                Linear::new(Tensor::stack(&down_proj_weights, 0)?, None),
            ))?);

        let expert0_vb = experts_vb.pp("0");
        up_proj = Self::apply_routed_expert_isq(up_proj, expert0_vb.pp("up_proj"), target_device)?;
        down_proj =
            Self::apply_routed_expert_isq(down_proj, expert0_vb.pp("down_proj"), target_device)?;

        Ok(Self {
            up_proj,
            down_proj,
            act_fn: cfg.mlp_hidden_act,
            all_reduce: SumAllReduce::new(comm),
            world_size: comm.world_size(),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        topk_indices: &Tensor,
        topk_weights: &Tensor,
    ) -> Result<Tensor> {
        let up = self
            .up_proj
            .gather_forward_autocast(hidden_states, topk_indices)?;
        let up = self.act_fn.forward(&up)?;
        let down = self.down_proj.gather_forward_autocast(&up, topk_indices)?;
        let weighted =
            down.broadcast_mul(&topk_weights.to_dtype(down.dtype())?.unsqueeze(D::Minus1)?)?;
        let mut out = weighted.sum(D::Minus2)?;
        if self.world_size > 1 {
            out = self.all_reduce.sum_all_reduce(&out)?;
        }
        Ok(out)
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        // Routed experts use indexed_moe kernels, so they must keep the dtype selected during
        // construction instead of flowing through the generic model-wide ISQ fallback chain.
        Vec::new()
    }
}

struct TopkRouter {
    weight: Tensor,
    correction_bias: Tensor,
    n_routed_experts: usize,
    n_group: usize,
    topk_group: usize,
    top_k: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
}

impl TopkRouter {
    fn new(cfg: &Config, vb: ShardedVarBuilder, layer_device: &Device) -> Result<Self> {
        let mut weight = vb.get((cfg.n_routed_experts, cfg.hidden_size), "weight")?;
        if !weight.device().same_device(layer_device) {
            weight = weight.to_device(layer_device)?;
        }
        let weight = weight.to_dtype(DType::F32)?;

        let mut correction_bias = vb.get(cfg.n_routed_experts, "e_score_correction_bias")?;
        if !correction_bias.device().same_device(layer_device) {
            correction_bias = correction_bias.to_device(layer_device)?;
        }
        let correction_bias = correction_bias.to_dtype(DType::F32)?;

        Ok(Self {
            weight,
            correction_bias,
            n_routed_experts: cfg.n_routed_experts,
            n_group: cfg.n_group,
            topk_group: cfg.topk_group,
            top_k: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
            routed_scaling_factor: cfg.routed_scaling_factor,
        })
    }

    fn route_tokens(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor)> {
        let (num_tokens, _) = hidden_states.dims2()?;
        let hidden_states = hidden_states.to_dtype(DType::F32)?;
        let router_logits = hidden_states.broadcast_matmul(&self.weight.t()?)?;
        let scores = candle_nn::ops::sigmoid(&router_logits)?;

        let scores_for_choice = scores
            .reshape((num_tokens, self.n_routed_experts))?
            .broadcast_add(&self.correction_bias.unsqueeze(0)?)?;
        let group_scores = scores_for_choice
            .reshape((num_tokens, self.n_group, self.n_routed_experts / self.n_group))?
            .topk(2)?
            .values
            .sum(D::Minus1)?;
        let group_idx = group_scores.topk(self.topk_group)?.indices;
        let mut group_mask = group_scores.zeros_like()?;
        group_mask = group_mask.scatter_add(
            &group_idx,
            &group_idx.ones_like()?.to_dtype(group_mask.dtype())?,
            1,
        )?;
        let score_mask = group_mask
            .unsqueeze(D::Minus1)?
            .expand((
                num_tokens,
                self.n_group,
                self.n_routed_experts / self.n_group,
            ))?
            .reshape((num_tokens, self.n_routed_experts))?;
        let masked_scores = scores_for_choice.broadcast_mul(&score_mask)?;
        let topk_indices = masked_scores.topk(self.top_k)?.indices;
        let mut topk_weights = scores.gather(&topk_indices, 1)?;

        if self.norm_topk_prob {
            let denom = (topk_weights.sum_keepdim(D::Minus1)? + 1e-20)?;
            topk_weights = topk_weights.broadcast_div(&denom)?;
        }

        topk_weights = (topk_weights * self.routed_scaling_factor as f64)?;
        Ok((topk_indices, topk_weights))
    }
}

struct MoeMixer {
    router: TopkRouter,
    experts: RoutedExperts,
    shared_experts: SharedMlp,
}

impl MoeMixer {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<mistralrs_quant::Comm>,
        real_device: Device,
    ) -> Result<Self> {
        if cfg.moe_latent_size.is_some() {
            candle_core::bail!("NemotronH `moe_latent_size` is not implemented yet.");
        }
        if cfg.n_shared_experts > 1 {
            candle_core::bail!("NemotronH only supports one shared expert in Candle for now.");
        }

        let layer_device = mapper
            .device_for(layer_idx, false)
            .cloned()
            .unwrap_or(real_device);

        let router = TopkRouter::new(cfg, vb.pp("gate"), &layer_device)?;
        let experts = RoutedExperts::new(cfg, vb.clone(), &layer_device, comm)?;
        let shared_experts = SharedMlp::new(
            mapper.set_device(layer_idx, vb.pp("shared_experts"), loading_isq),
            cfg.hidden_size,
            cfg.moe_shared_expert_intermediate_size,
            &cfg.quantization_config,
            cfg.mlp_hidden_act,
            comm,
        )?;

        Ok(Self {
            router,
            experts,
            shared_experts,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;
        let residual = hidden_states.clone();
        let flat = hidden_states.reshape((batch_size * seq_len, hidden_size))?;
        let (topk_indices, topk_weights) = self.router.route_tokens(&flat)?;
        let routed = self
            .experts
            .forward(&flat, &topk_indices, &topk_weights)?
            .reshape((batch_size, seq_len, hidden_size))?;
        let shared = self.shared_experts.forward(&residual)?;
        routed + shared
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        let mut layers = self.experts.get_isq_layers();
        layers.extend(self.shared_experts.get_isq_layers());
        layers
    }
}

#[derive(Debug)]
struct MambaLayerCache {
    conv_state: Tensor,
    ssm_state: Tensor,
    seqlen_offset: usize,
}

impl MambaLayerCache {
    fn new(
        batch_size: usize,
        conv_dim: usize,
        conv_width: usize,
        num_heads: usize,
        head_dim: usize,
        state_size: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            conv_state: Tensor::zeros((batch_size, conv_dim, conv_width), dtype, device)?,
            ssm_state: Tensor::zeros((batch_size, num_heads, head_dim, state_size), dtype, device)?,
            seqlen_offset: 0,
        })
    }

    fn reset(&mut self) -> Result<()> {
        self.conv_state = self.conv_state.zeros_like()?;
        self.ssm_state = self.ssm_state.zeros_like()?;
        self.seqlen_offset = 0;
        Ok(())
    }
}

impl Clone for MambaLayerCache {
    fn clone(&self) -> Self {
        Self {
            conv_state: self.conv_state.clone(),
            ssm_state: self.ssm_state.clone(),
            seqlen_offset: self.seqlen_offset,
        }
    }
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    (Tensor::ones_like(x)? + x.exp()?)?.log()
}

fn create_mamba_cache(
    batch_size: usize,
    cfg: &Config,
    dtype: DType,
    device: &Device,
) -> Result<MambaLayerCache> {
    MambaLayerCache::new(
        batch_size,
        cfg.mamba_conv_dim(),
        cfg.conv_kernel,
        cfg.mamba_num_heads,
        cfg.mamba_head_dim,
        cfg.ssm_state_size,
        dtype,
        device,
    )
}

struct RmsNormGated {
    weight: Tensor,
    eps: f64,
    group_size: usize,
}

impl RmsNormGated {
    fn new(
        hidden_size: usize,
        group_size: usize,
        eps: f64,
        vb: ShardedVarBuilder,
        isq_target_device: Option<&Device>,
    ) -> Result<Self> {
        let mut weight = vb.get((hidden_size,), "weight")?;
        if let Some(target_device) = isq_target_device {
            weight = weight.to_device(target_device)?;
        }
        Ok(Self {
            weight,
            eps,
            group_size,
        })
    }

    fn forward(&self, hidden_states: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let dtype = hidden_states.dtype();
        let gate = candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?;
        let hidden_states = hidden_states.to_dtype(DType::F32)?.broadcast_mul(&gate)?;
        let hidden_shape = hidden_states.shape().dims().to_vec();
        let hidden_size = *hidden_shape
            .last()
            .expect("RmsNormGated requires at least one hidden dimension");
        if hidden_size % self.group_size != 0 {
            candle_core::bail!(
                "Nemotron gated RMSNorm hidden size {hidden_size} is not divisible by group size {}.",
                self.group_size
            );
        }
        let mut grouped_shape = hidden_shape.clone();
        grouped_shape.pop();
        grouped_shape.push(hidden_size / self.group_size);
        grouped_shape.push(self.group_size);
        let grouped = hidden_states.reshape(grouped_shape.as_slice())?;
        let variance = grouped.sqr()?.mean_keepdim(D::Minus1)?;
        let hidden_states = grouped
            .broadcast_div(&(variance + self.eps)?.sqrt()?)?
            .reshape(hidden_shape.as_slice())?;
        hidden_states
            .to_dtype(dtype)?
            .broadcast_mul(&self.weight.to_dtype(dtype)?)
    }
}

struct MambaLayer {
    in_proj: Linear,
    conv1d_weight: Tensor,
    conv1d_bias: Option<Tensor>,
    dt_bias: Tensor,
    a_log: Tensor,
    d: Tensor,
    norm: RmsNormGated,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    ssm_state_size: usize,
    conv_kernel_size: usize,
    n_groups: usize,
    time_step_limit: Option<(f64, f64)>,
}

impl MambaLayer {
    fn load(
        vb: ShardedVarBuilder,
        cfg: &Config,
        isq_target_device: Option<&Device>,
    ) -> Result<Self> {
        let intermediate_size = cfg.mamba_intermediate_size();
        let conv_dim = cfg.mamba_conv_dim();
        let projection_size = intermediate_size + conv_dim + cfg.mamba_num_heads;

        let in_proj_vb = vb.pp("in_proj");
        let mut in_proj_weight = in_proj_vb.get((projection_size, cfg.hidden_size), "weight")?;
        let mut in_proj_bias = if cfg.mamba_proj_bias {
            Some(in_proj_vb.get(projection_size, "bias")?)
        } else {
            None
        };

        let mut conv1d_weight = vb
            .pp("conv1d")
            .get((conv_dim, 1, cfg.conv_kernel), "weight")?;
        let mut conv1d_bias = if cfg.use_conv_bias {
            Some(vb.pp("conv1d").get(conv_dim, "bias")?)
        } else {
            None
        };

        let mut dt_bias = vb.get(cfg.mamba_num_heads, "dt_bias")?;
        let mut a_log = vb.get(cfg.mamba_num_heads, "A_log")?;
        let mut d = vb.get(cfg.mamba_num_heads, "D")?;
        let norm = RmsNormGated::new(
            intermediate_size,
            intermediate_size / cfg.n_groups,
            cfg.layer_norm_epsilon,
            vb.pp("norm"),
            isq_target_device,
        )?;

        let out_proj_vb = vb.pp("out_proj");
        let mut out_proj_weight =
            out_proj_vb.get((cfg.hidden_size, intermediate_size), "weight")?;
        let mut out_proj_bias = if cfg.use_bias {
            Some(out_proj_vb.get(cfg.hidden_size, "bias")?)
        } else {
            None
        };

        if let Some(target_device) = isq_target_device {
            in_proj_weight = in_proj_weight.to_device(target_device)?;
            if let Some(ref bias) = in_proj_bias {
                in_proj_bias = Some(bias.to_device(target_device)?);
            }
            conv1d_weight = conv1d_weight.to_device(target_device)?;
            if let Some(ref bias) = conv1d_bias {
                conv1d_bias = Some(bias.to_device(target_device)?);
            }
            dt_bias = dt_bias.to_device(target_device)?;
            a_log = a_log.to_device(target_device)?;
            d = d.to_device(target_device)?;
            out_proj_weight = out_proj_weight.to_device(target_device)?;
            if let Some(ref bias) = out_proj_bias {
                out_proj_bias = Some(bias.to_device(target_device)?);
            }
        }

        Ok(Self {
            in_proj: Linear::new(in_proj_weight, in_proj_bias),
            conv1d_weight,
            conv1d_bias,
            dt_bias,
            a_log,
            d,
            norm,
            out_proj: Linear::new(out_proj_weight, out_proj_bias),
            num_heads: cfg.mamba_num_heads,
            head_dim: cfg.mamba_head_dim,
            intermediate_size,
            ssm_state_size: cfg.ssm_state_size,
            conv_kernel_size: cfg.conv_kernel,
            n_groups: cfg.n_groups,
            time_step_limit: cfg.time_step_limit,
        })
    }

    fn forward(&self, x: &Tensor, cache: &mut MambaLayerCache) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let dtype = x.dtype();
        let groups_time_state_size = self.n_groups * self.ssm_state_size;

        let projected = self.in_proj.forward(x)?;
        let gate = projected.narrow(D::Minus1, 0, self.intermediate_size)?;
        let hidden_states_b_c = projected.narrow(
            D::Minus1,
            self.intermediate_size,
            self.intermediate_size + 2 * groups_time_state_size,
        )?;
        let dt = projected.narrow(
            D::Minus1,
            self.intermediate_size + self.intermediate_size + 2 * groups_time_state_size,
            self.num_heads,
        )?;

        let y = if cache.seqlen_offset > 0 && seq_len == 1 {
            self.forward_cached(
                &hidden_states_b_c.squeeze(1)?,
                &dt.squeeze(1)?,
                cache,
                batch_size,
            )?
            .unsqueeze(1)?
        } else {
            self.forward_prefill(&hidden_states_b_c, &dt, cache, batch_size, seq_len)?
        };

        cache.seqlen_offset += seq_len;
        let scan_output = self.norm.forward(&y, &gate)?;
        self.out_proj.forward(&scan_output.to_dtype(dtype)?)
    }

    fn forward_cached(
        &self,
        hidden_states_b_c: &Tensor,
        dt: &Tensor,
        cache: &mut MambaLayerCache,
        batch_size: usize,
    ) -> Result<Tensor> {
        let conv_state = cache
            .conv_state
            .narrow(D::Minus1, 1, self.conv_kernel_size - 1)?;
        let new_col = hidden_states_b_c.contiguous()?.unsqueeze(D::Minus1)?;
        cache.conv_state = Tensor::cat(&[&conv_state, &new_col], D::Minus1)?.contiguous()?;

        let mut conv_out = cache
            .conv_state
            .broadcast_mul(&self.conv1d_weight.squeeze(1)?.unsqueeze(0)?)?
            .sum(D::Minus1)?;
        if let Some(ref bias) = self.conv1d_bias {
            conv_out = conv_out.broadcast_add(bias)?;
        }
        let conv_out = candle_nn::ops::silu(&conv_out)?;
        self.forward_ssm(&conv_out, dt, cache, batch_size)
    }

    fn forward_prefill(
        &self,
        hidden_states_b_c: &Tensor,
        dt: &Tensor,
        cache: &mut MambaLayerCache,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let hidden_states = hidden_states_b_c.transpose(1, 2)?;
        let mut conv_out = hidden_states.conv1d(
            &self.conv1d_weight,
            self.conv_kernel_size - 1,
            1,
            1,
            self.conv1d_weight.dim(0)?,
        )?;
        if let Some(ref bias) = self.conv1d_bias {
            conv_out = conv_out.broadcast_add(&bias.reshape((1, bias.dim(0)?, 1))?)?;
        }
        conv_out = candle_nn::ops::silu(&conv_out.narrow(D::Minus1, 0, seq_len)?)?;
        let conv_out = conv_out.transpose(1, 2)?;

        if seq_len >= self.conv_kernel_size {
            cache.conv_state = hidden_states.narrow(
                D::Minus1,
                seq_len - self.conv_kernel_size,
                self.conv_kernel_size,
            )?
            .contiguous()?;
        } else {
            let pad = Tensor::zeros(
                (
                    batch_size,
                    hidden_states.dim(1)?,
                    self.conv_kernel_size - seq_len,
                ),
                hidden_states.dtype(),
                hidden_states.device(),
            )?;
            cache.conv_state = Tensor::cat(&[&pad, &hidden_states], D::Minus1)?.contiguous()?;
        }

        self.forward_ssm_sequence(&conv_out, dt, cache, batch_size, seq_len)
    }

    fn forward_ssm(
        &self,
        conv_out: &Tensor,
        dt: &Tensor,
        cache: &mut MambaLayerCache,
        batch_size: usize,
    ) -> Result<Tensor> {
        let groups_time_state_size = self.n_groups * self.ssm_state_size;
        let hidden_states = conv_out.narrow(D::Minus1, 0, self.intermediate_size)?;
        let b = conv_out.narrow(D::Minus1, self.intermediate_size, groups_time_state_size)?;
        let c = conv_out.narrow(
            D::Minus1,
            self.intermediate_size + groups_time_state_size,
            groups_time_state_size,
        )?;

        let mut dt = softplus(&dt.broadcast_add(&self.dt_bias.to_dtype(dt.dtype())?)?)?;
        if let Some((min_dt, max_dt)) = self.time_step_limit {
            dt = dt.clamp(min_dt, max_dt)?;
        }
        let a = self.a_log.exp()?.neg()?;

        let hidden_states = hidden_states.reshape((batch_size, self.num_heads, self.head_dim))?;
        let dt = dt.reshape((batch_size, self.num_heads, 1))?.broadcast_as((
            batch_size,
            self.num_heads,
            self.head_dim,
        ))?;
        let a = a.reshape((1, self.num_heads, 1))?.broadcast_as((
            batch_size,
            self.num_heads,
            self.head_dim,
        ))?;
        let d = self.d.reshape((1, self.num_heads, 1))?.broadcast_as((
            batch_size,
            self.num_heads,
            self.head_dim,
        ))?;

        let b = b.reshape((batch_size, self.n_groups, self.ssm_state_size))?;
        let c = c.reshape((batch_size, self.n_groups, self.ssm_state_size))?;

        let b = if self.n_groups == self.num_heads {
            b
        } else {
            b.unsqueeze(2)?
                .expand((
                    batch_size,
                    self.n_groups,
                    self.num_heads / self.n_groups,
                    self.ssm_state_size,
                ))?
                .reshape((batch_size, self.num_heads, self.ssm_state_size))?
        };
        let c = if self.n_groups == self.num_heads {
            c
        } else {
            c.unsqueeze(2)?
                .expand((
                    batch_size,
                    self.n_groups,
                    self.num_heads / self.n_groups,
                    self.ssm_state_size,
                ))?
                .reshape((batch_size, self.num_heads, self.ssm_state_size))?
        };

        let a_dt = a.broadcast_mul(&dt)?;
        let delta_a = a_dt.exp()?;
        let delta_b = dt
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&b.unsqueeze(2)?.broadcast_as((
                batch_size,
                self.num_heads,
                self.head_dim,
                self.ssm_state_size,
            ))?)?;

        let hidden = hidden_states.unsqueeze(D::Minus1)?;
        cache.ssm_state = cache
            .ssm_state
            .broadcast_mul(&delta_a.unsqueeze(D::Minus1)?)?
            .broadcast_add(&delta_b.broadcast_mul(&hidden)?)?;

        let y = cache
            .ssm_state
            .broadcast_mul(&c.unsqueeze(2)?.broadcast_as((
                batch_size,
                self.num_heads,
                self.head_dim,
                self.ssm_state_size,
            ))?)?
            .sum(D::Minus1)?
            .broadcast_add(&hidden_states.broadcast_mul(&d)?)?;

        y.reshape((batch_size, self.intermediate_size))
    }

    fn forward_ssm_sequence(
        &self,
        conv_out: &Tensor,
        dt: &Tensor,
        cache: &mut MambaLayerCache,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let mut outputs = Vec::with_capacity(seq_len);
        for step in 0..seq_len {
            let step_conv = conv_out.i((.., step, ..))?;
            let step_dt = dt.i((.., step, ..))?;
            outputs.push(self.forward_ssm(&step_conv, &step_dt, cache, batch_size)?);
        }
        Tensor::stack(&outputs.iter().collect::<Vec<_>>(), 1)
    }
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    _rotary_emb: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn load(
        vb: ShardedVarBuilder,
        cfg: &Config,
        paged_attn: Option<PagedAttention>,
        rotary_emb: Arc<RotaryEmbedding>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let q_proj = ColumnParallelLayer::new(
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.head_dim,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            vb.pp("q_proj"),
        )?;
        let kv_size = cfg.num_key_value_heads() * cfg.head_dim;
        let kv_shard =
            mistralrs_quant::compute_kv_shard(cfg.num_key_value_heads(), cfg.head_dim, comm);
        let k_proj = ColumnParallelLayer::new_with_shard(
            cfg.hidden_size,
            kv_size,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            kv_shard,
            vb.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            cfg.hidden_size,
            kv_size,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            kv_shard,
            vb.pp("v_proj"),
        )?;
        let o_proj = RowParallelLayer::new(
            cfg.num_attention_heads * cfg.head_dim,
            cfg.hidden_size,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            vb.pp("o_proj"),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads / comm.world_size(),
            num_key_value_heads: (cfg.num_key_value_heads() / comm.world_size()).max(1),
            head_dim: cfg.head_dim,
            _rotary_emb: rotary_emb,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads(),
                    cfg.num_attention_heads,
                    comm,
                ),
                softcap: None,
                softmax_scale: 1.0 / (cfg.head_dim as f32).sqrt(),
                sliding_window: cfg.sliding_window,
                sinks: None,
            },
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &Option<Tensor>,
        _seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let original_dtype = x.dtype();
        let mut x = x.clone();
        if let Some(t) = self.q_proj.quantized_act_type() {
            x = x.to_dtype(t)?;
        }
        let mut q = MatMul.qmethod_matmul(&x, &*self.q_proj)?;
        let mut k = MatMul.qmethod_matmul(&x, &*self.k_proj)?;
        let mut v = MatMul.qmethod_matmul(&x, &*self.v_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            q = q.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

        let (q, k, v) = if seq_len != 1 {
            (
                q.reshape((batch_size, seq_len, self.num_attention_heads, self.head_dim))?
                    .transpose(1, 2)?,
                k.reshape((batch_size, seq_len, self.num_key_value_heads, self.head_dim))?
                    .transpose(1, 2)?,
                v.reshape((batch_size, seq_len, self.num_key_value_heads, self.head_dim))?
                    .transpose(1, 2)?,
            )
        } else {
            (
                q.reshape((batch_size, self.num_attention_heads, seq_len, self.head_dim))?,
                k.reshape((batch_size, self.num_key_value_heads, seq_len, self.head_dim))?,
                v.reshape((batch_size, self.num_key_value_heads, seq_len, self.head_dim))?,
            )
        };

        let mut y = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask.clone().as_ref(),
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    Some(flash_params),
                )?,
                None => {
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        attention_mask.clone().as_ref(),
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        Some(flash_params),
                    )?
                }
            },
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;
                Sdpa.run_attention(
                    &q,
                    &k,
                    &v,
                    attention_mask.clone().as_ref(),
                    Some(flash_params),
                    &self.sdpa_params,
                )?
            }
        };

        if let Some(t) = self.q_proj.quantized_act_type() {
            y = y.to_dtype(t)?;
        }
        y = if attention_mask.is_some() {
            y.transpose(1, 2)?.reshape((batch_size, seq_len, ()))?
        } else {
            y.reshape((batch_size, seq_len, ()))?
        };
        let mut out = MatMul.qmethod_matmul(&y, &*self.o_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            out = out.to_dtype(original_dtype)?;
        }
        Ok(out)
    }
}

struct AttentionBlock {
    norm: RmsNorm,
    mixer: Attention,
}

impl AttentionBlock {
    fn load(
        vb: ShardedVarBuilder,
        cfg: &Config,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        rotary_emb: Arc<RotaryEmbedding>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        Ok(Self {
            norm: RmsNorm::new(
                cfg.hidden_size,
                cfg.layer_norm_epsilon,
                mapper.set_device(layer_idx, vb.pp("norm"), false),
            )?,
            mixer: Attention::load(
                mapper.set_device(layer_idx, vb.pp("mixer"), loading_isq),
                cfg,
                paged_attn,
                rotary_emb,
                comm,
            )?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.norm.forward(x)?;
        let x = self.mixer.forward(
            &x,
            attention_mask,
            seqlen_offsets,
            kv_cache,
            metadata,
            flash_params,
        )?;
        x + residual
    }
}

struct MambaBlock {
    norm: RmsNorm,
    mixer: MambaLayer,
}

impl MambaBlock {
    fn load(
        vb: ShardedVarBuilder,
        cfg: &Config,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
    ) -> Result<Self> {
        let isq_target_device = if loading_isq {
            mapper.device_for(layer_idx, false)
        } else {
            None
        };
        Ok(Self {
            norm: RmsNorm::new(
                cfg.hidden_size,
                cfg.layer_norm_epsilon,
                mapper.set_device(layer_idx, vb.pp("norm"), false),
            )?,
            mixer: MambaLayer::load(
                mapper.set_device(layer_idx, vb.pp("mixer"), loading_isq),
                cfg,
                isq_target_device,
            )?,
        })
    }

    fn forward(&self, x: &Tensor, cache: &mut MambaLayerCache) -> Result<Tensor> {
        let residual = x;
        let x = self.norm.forward(x)?;
        let x = self.mixer.forward(&x, cache)?;
        x + residual
    }
}

struct MoeBlock {
    norm: RmsNorm,
    mixer: MoeMixer,
}

impl MoeBlock {
    fn load(
        vb: ShardedVarBuilder,
        cfg: &Config,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<mistralrs_quant::Comm>,
        real_device: Device,
    ) -> Result<Self> {
        Ok(Self {
            norm: RmsNorm::new(
                cfg.hidden_size,
                cfg.layer_norm_epsilon,
                mapper.set_device(layer_idx, vb.pp("norm"), false),
            )?,
            mixer: MoeMixer::new(
                cfg,
                mapper.set_device(layer_idx, vb.pp("mixer"), loading_isq),
                mapper,
                layer_idx,
                loading_isq,
                comm,
                real_device,
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.norm.forward(x)?;
        let x = self.mixer.forward(&x)?;
        x + residual
    }
}

enum Block {
    Attention(AttentionBlock),
    Mamba(MambaBlock),
    Moe(MoeBlock),
}

enum LayerCache {
    Attention(KvCache),
    Mamba(MambaLayerCache),
    Stateless,
}

struct InternalCache {
    caches: Vec<LayerCache>,
}

impl InternalCache {
    fn new(
        layer_types: &[NemotronLayerType],
        cfg: &Config,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut caches = Vec::with_capacity(layer_types.len());
        for layer_type in layer_types {
            match layer_type {
                NemotronLayerType::Attention => {
                    caches.push(LayerCache::Attention(KvCache::new_normal(
                        2,
                        cfg.max_position_embeddings,
                        HybridCache::CACHE_GROW_SIZE,
                    )))
                }
                NemotronLayerType::Mamba => caches.push(LayerCache::Mamba(create_mamba_cache(
                    1, cfg, dtype, device,
                )?)),
                NemotronLayerType::Moe => caches.push(LayerCache::Stateless),
            }
        }
        Ok(Self { caches })
    }
}

pub struct Model {
    embeddings: Embedding,
    layers: Vec<Block>,
    norm_f: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    internal_cache: Arc<Mutex<InternalCache>>,
    kv_cache: EitherCache,
    device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
    num_attention_heads: usize,
    max_seq_len: usize,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let layer_types = cfg.layer_types()?;
        let num_hidden_layers = layer_types.len();
        let vb_backbone = vb.pp("backbone");
        let mapper = normal_loading_metadata.mapper;

        let embeddings = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_backbone.pp("embeddings"), false),
            &cfg.quantization_config,
        )?;
        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        } else {
            ReplicatedLayer::from_linear(Linear::new(
                mapper
                    .cast_nm_device(embeddings.embeddings(), normal_loading_metadata.loading_isq)?,
                None,
            ))?
        };
        let norm_f = RmsNorm::new(
            cfg.hidden_size,
            cfg.layer_norm_epsilon,
            mapper.set_nm_device(vb_backbone.pp("norm_f"), false),
        )?;

        let mut ropes = HashMap::new();
        for (i, layer_type) in layer_types.iter().enumerate() {
            if matches!(layer_type, NemotronLayerType::Attention) {
                let device = mapper
                    .device_for(i, false)
                    .unwrap_or(&normal_loading_metadata.real_device);
                if let std::collections::hash_map::Entry::Vacant(entry) =
                    ropes.entry(device.location())
                {
                    entry.insert(Arc::new(RotaryEmbedding::new(
                        cfg.rope_theta,
                        cfg.head_dim,
                        cfg.max_position_embeddings,
                        device,
                        true,
                        vb_backbone.dtype(),
                    )?));
                }
            }
        }

        let vb_layers = vb_backbone.pp("layers");
        let mut layers = Vec::with_capacity(num_hidden_layers);
        for (i, layer_type) in layer_types.iter().enumerate() {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let comm = mapper.get_comm_for(i)?;
            let vb_layer = vb_layers.pp(i);
            let block = match layer_type {
                NemotronLayerType::Attention => {
                    let paged_attn = match &attention_mechanism {
                        AttentionImplementation::Eager => None,
                        AttentionImplementation::PagedAttention => {
                            Some(PagedAttention::new(cfg.head_dim, device, None)?)
                        }
                    };
                    Block::Attention(AttentionBlock::load(
                        vb_layer,
                        cfg,
                        &*mapper,
                        i,
                        normal_loading_metadata.loading_isq,
                        paged_attn,
                        ropes
                            .get(&device.location())
                            .expect("missing Nemotron attention rope")
                            .clone(),
                        &comm,
                    )?)
                }
                NemotronLayerType::Mamba => Block::Mamba(MambaBlock::load(
                    vb_layer,
                    cfg,
                    &*mapper,
                    i,
                    normal_loading_metadata.loading_isq,
                )?),
                NemotronLayerType::Moe => Block::Moe(MoeBlock::load(
                    vb_layer,
                    cfg,
                    &*mapper,
                    i,
                    normal_loading_metadata.loading_isq,
                    &comm,
                    normal_loading_metadata.real_device.clone(),
                )?),
            };
            layers.push(block);
        }

        let internal_cache = Arc::new(Mutex::new(InternalCache::new(
            &layer_types,
            cfg,
            &normal_loading_metadata.real_device,
            vb_backbone.dtype(),
        )?));

        let recurrent_layer_devices: Vec<Device> = layer_types
            .iter()
            .enumerate()
            .map(|(layer_idx, layer_type)| match layer_type {
                NemotronLayerType::Mamba | NemotronLayerType::Moe => mapper
                    .device_for(layer_idx, false)
                    .unwrap_or(&normal_loading_metadata.real_device)
                    .clone(),
                NemotronLayerType::Attention => normal_loading_metadata.real_device.clone(),
            })
            .collect();
        let pipeline_layer_types = layer_types
            .iter()
            .map(|layer_type| match layer_type {
                NemotronLayerType::Attention => HybridLayerType::Attention,
                NemotronLayerType::Mamba | NemotronLayerType::Moe => HybridLayerType::Recurrent,
            })
            .collect();
        let pipeline_cache = Arc::new(Mutex::new(HybridCache::new_mapped(
            HybridCacheConfig {
                layer_types: pipeline_layer_types,
                max_seq_len: cfg.max_position_embeddings,
                recurrent: RecurrentLayerConfig {
                    conv_dim: cfg.mamba_conv_dim(),
                    conv_width: cfg.conv_kernel,
                    state_dims: vec![cfg.mamba_num_heads, cfg.mamba_head_dim, cfg.ssm_state_size],
                },
            },
            vb_backbone.dtype(),
            &normal_loading_metadata.real_device,
            &recurrent_layer_devices,
        )?));

        let world_size = mapper.get_comm_for(0)?.world_size();
        let num_attention_heads = cfg.num_attention_heads / world_size;

        Ok(Self {
            embeddings,
            layers,
            norm_f,
            lm_head,
            internal_cache,
            kv_cache: EitherCache::Hybrid(pipeline_cache),
            device: normal_loading_metadata.real_device,
            mapper,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_key_value_heads() / world_size).max(1),
                num_attn_heads: num_attention_heads,
                sliding_window: cfg.sliding_window,
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            num_attention_heads,
            max_seq_len: cfg.max_position_embeddings,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut x = self.embeddings.forward(input_ids)?;
        let mut internal_cache = self.internal_cache.lock().unwrap();
        let mut hybrid_cache = self.kv_cache.hybrid();
        let state_indices = hybrid_cache.state_indices().cloned();
        let mask = CausalMasker.make_causal_mask_matrix(
            input_ids,
            metadata
                .as_ref()
                .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                .unwrap_or(&*hybrid_cache as &dyn PastKvLenCache),
            x.dtype(),
            self.num_attention_heads,
        )?;
        let mask = mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });
        let mask = DeviceMappedMask::new(mask, &*self.mapper)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = self.mapper.map(x, layer_idx)?;
            match layer {
                Block::Attention(block) => {
                    let mask_for_layer = mask.as_ref().map(|m| m.get(x.device()).clone());
                    if let Some(HybridLayerCache::Attention(kv_cache)) =
                        hybrid_cache.get_mut(layer_idx)
                    {
                        x = block.forward(
                            &x,
                            &mask_for_layer,
                            seqlen_offsets,
                            kv_cache,
                            metadata
                                .as_ref()
                                .map(|(kv_cache, meta)| (kv_cache[layer_idx].clone(), *meta)),
                            flash_params,
                        )?;
                    } else if let LayerCache::Attention(kv_cache) =
                        &mut internal_cache.caches[layer_idx]
                    {
                        x = block.forward(
                            &x,
                            &mask_for_layer,
                            seqlen_offsets,
                            kv_cache,
                            metadata
                                .as_ref()
                                .map(|(kv_cache, meta)| (kv_cache[layer_idx].clone(), *meta)),
                            flash_params,
                        )?;
                    } else {
                        candle_core::bail!(
                            "Nemotron attention cache mismatch at layer {layer_idx}."
                        );
                    }
                }
                Block::Mamba(block) => {
                    if let (Some(indices), Some(HybridLayerCache::Recurrent(pool))) =
                        (&state_indices, hybrid_cache.get_mut(layer_idx))
                    {
                        let indices_vec: Vec<u32> = indices.to_vec1()?;
                        if indices_vec.is_empty() {
                            candle_core::bail!("Nemotron recurrent state indices are empty.");
                        }
                        let first_offset = pool.get_seqlen_offset(indices_vec[0] as usize);
                        let conv_state = pool.gather_conv_state(indices)?;
                        let ssm_state = pool.gather_recurrent_state(indices)?;
                        let mut cache = MambaLayerCache {
                            conv_state,
                            ssm_state,
                            seqlen_offset: first_offset,
                        };
                        x = block.forward(&x, &mut cache)?;
                        pool.scatter_conv_state(indices, &cache.conv_state)?;
                        pool.scatter_recurrent_state(indices, &cache.ssm_state)?;
                        for idx in indices_vec {
                            pool.set_seqlen_offset(idx as usize, cache.seqlen_offset);
                        }
                    } else if let LayerCache::Mamba(cache) = &mut internal_cache.caches[layer_idx] {
                        if seqlen_offsets.first().copied() == Some(0) {
                            cache.reset()?;
                        }
                        x = block.forward(&x, cache)?;
                    } else {
                        candle_core::bail!("Nemotron mamba cache mismatch at layer {layer_idx}.");
                    }
                }
                Block::Moe(block) => {
                    x = block.forward(&x)?;
                }
            }
        }

        let x = x.to_device(&self.device)?;
        let x = self.norm_f.forward(&x)?;
        let mut x = extract_logits(&x, context_lens)?;
        if let Some(t) = self.lm_head.quantized_act_type() {
            x = x.to_dtype(t)?;
        }
        MatMul.qmethod_matmul(&x, &*self.lm_head)
    }
}

impl IsqModel for Model {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = vec![(&mut self.lm_head, None)];
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            match layer {
                Block::Attention(block) => {
                    tensors.push((&mut block.mixer.q_proj, Some(layer_idx)));
                    tensors.push((&mut block.mixer.k_proj, Some(layer_idx)));
                    tensors.push((&mut block.mixer.v_proj, Some(layer_idx)));
                    tensors.push((&mut block.mixer.o_proj, Some(layer_idx)));
                }
                Block::Mamba(_) => {}
                Block::Moe(block) => {
                    tensors.extend(
                        block
                            .mixer
                            .get_isq_layers()
                            .into_iter()
                            .map(|layer| (layer, Some(layer_idx))),
                    );
                }
            }
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_b = uvb.pp("backbone");
        uvb_b.pp("embeddings").add(&self.embeddings);
        uvb_b.pp("norm_f").add(&self.norm_f);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_b.pp("layers").pp(layer_idx);
            match layer {
                Block::Attention(block) => {
                    uvb_l.pp("norm").add(&block.norm);
                }
                Block::Mamba(block) => {
                    uvb_l.pp("norm").add(&block.norm);
                    uvb_l
                        .pp("mixer")
                        .pp("norm")
                        .add_tensor("weight", block.mixer.norm.weight.clone());
                    uvb_l
                        .pp("mixer")
                        .add_tensor("dt_bias", block.mixer.dt_bias.clone());
                    uvb_l
                        .pp("mixer")
                        .add_tensor("A_log", block.mixer.a_log.clone());
                    uvb_l.pp("mixer").add_tensor("D", block.mixer.d.clone());
                    uvb_l
                        .pp("mixer")
                        .add_tensor("conv1d.weight", block.mixer.conv1d_weight.clone());
                    if let Some(ref bias) = block.mixer.conv1d_bias {
                        uvb_l.pp("mixer").add_tensor("conv1d.bias", bias.clone());
                    }
                }
                Block::Moe(block) => {
                    uvb_l.pp("norm").add(&block.norm);
                    uvb_l.pp("mixer").pp("gate").add_tensor(
                        "e_score_correction_bias",
                        block.mixer.router.correction_bias.clone(),
                    );
                }
            }
        }
        uvb.to_safetensors()
    }
}

impl NormalModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }

    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        candle_core::bail!("NemotronH does not support X-LoRA forward")
    }

    fn is_xlora(&self) -> bool {
        false
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn cache(&self) -> &EitherCache {
        &self.kv_cache
    }

    fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.kv_cache
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl AnyMoeBaseModelMixin for Model {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        vec![]
    }

    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::{Config, NemotronLayerType};

    #[test]
    fn parses_hybrid_override_pattern() {
        let cfg: Config = serde_json::from_str(
            r#"{
                "vocab_size": 1,
                "hidden_size": 1,
                "hybrid_override_pattern": "ME*E",
                "num_attention_heads": 1,
                "head_dim": 1,
                "max_position_embeddings": 1,
                "intermediate_size": 1,
                "mamba_num_heads": 1,
                "mamba_head_dim": 1,
                "n_routed_experts": 2,
                "n_shared_experts": 1,
                "moe_intermediate_size": 1,
                "moe_shared_expert_intermediate_size": 1
            }"#,
        )
        .unwrap();

        assert_eq!(
            cfg.layer_types().unwrap(),
            vec![
                NemotronLayerType::Mamba,
                NemotronLayerType::Moe,
                NemotronLayerType::Attention,
                NemotronLayerType::Moe,
            ]
        );
    }
}
