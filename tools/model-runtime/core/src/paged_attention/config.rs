#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KvCacheLayout {
    Standard,
    Mla {
        kv_lora_rank: usize,
        kpe_head_dim: usize,
    },
}

pub trait ModelConfigLike {
    fn max_seq_len(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn num_attn_heads(&self) -> usize;
    fn k_head_dim(&self) -> usize;
    fn v_head_dim(&self) -> usize;
    fn num_kv_heads_for_layer(&self, _layer_idx: usize) -> usize {
        self.num_kv_heads()
    }
    fn k_head_dim_for_layer(&self, _layer_idx: usize) -> usize {
        self.k_head_dim()
    }
    fn v_head_dim_for_layer(&self, _layer_idx: usize) -> usize {
        self.v_head_dim()
    }
    fn kv_cache_layout(&self) -> KvCacheLayout {
        KvCacheLayout::Standard
    }
    fn kv_cache_elements_per_token(&self) -> usize {
        let num_layers = self.num_layers().max(1);
        let total: usize = (0..num_layers)
            .map(|i| {
                let kv_heads = self.num_kv_heads_for_layer(i);
                let k_dim = self.k_head_dim_for_layer(i);
                let v_dim = self.v_head_dim_for_layer(i);
                kv_heads * k_dim.max(v_dim) * 2
            })
            .sum();
        total / num_layers
    }
}

#[derive(Clone)]
pub struct ModelConfigMetadata {
    pub max_seq_len: usize,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_kv_heads: usize,
    pub num_attn_heads: usize,
    pub sliding_window: Option<usize>,
    pub k_head_dim: usize,
    pub v_head_dim: usize,
    pub kv_cache_layout: KvCacheLayout,
}

impl ModelConfigLike for ModelConfigMetadata {
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn num_attn_heads(&self) -> usize {
        self.num_attn_heads
    }
    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
    fn num_layers(&self) -> usize {
        self.num_layers
    }
    fn k_head_dim(&self) -> usize {
        self.k_head_dim
    }
    fn v_head_dim(&self) -> usize {
        self.v_head_dim
    }
    fn num_kv_heads_for_layer(&self, _layer_idx: usize) -> usize {
        self.num_kv_heads
    }
    fn k_head_dim_for_layer(&self, _layer_idx: usize) -> usize {
        self.k_head_dim
    }
    fn v_head_dim_for_layer(&self, _layer_idx: usize) -> usize {
        self.v_head_dim
    }
    fn kv_cache_layout(&self) -> KvCacheLayout {
        self.kv_cache_layout
    }
    fn kv_cache_elements_per_token(&self) -> usize {
        match self.kv_cache_layout {
            KvCacheLayout::Standard => 2 * self.num_kv_heads * self.k_head_dim.max(self.v_head_dim),
            KvCacheLayout::Mla {
                kv_lora_rank,
                kpe_head_dim,
            } => kv_lora_rank + kpe_head_dim,
        }
    }
}
