use std::{
    str::FromStr,
    sync::{Arc, Mutex, MutexGuard},
};

use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use super::config::{KvCacheLayout, ModelConfigLike};
use crate::TurboQuantBits;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Default)]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq, eq_int))]
pub enum PagedCacheType {
    #[default]
    Auto,
    F8E4M3,
    TurboQuant2,
    TurboQuant3,
    TurboQuant4,
}

impl PagedCacheType {
    pub fn to_dtype(&self, act_dtype: DType) -> DType {
        match self {
            PagedCacheType::F8E4M3 => DType::F8E4M3,
            PagedCacheType::TurboQuant2
            | PagedCacheType::TurboQuant3
            | PagedCacheType::TurboQuant4 => DType::U8,
            PagedCacheType::Auto => act_dtype,
        }
    }

    pub fn turboquant_bits(&self) -> Option<TurboQuantBits> {
        match self {
            Self::TurboQuant2 => Some(TurboQuantBits::Two),
            Self::TurboQuant3 => Some(TurboQuantBits::Three),
            Self::TurboQuant4 => Some(TurboQuantBits::Four),
            Self::Auto | Self::F8E4M3 => None,
        }
    }

    pub fn is_turboquant(&self) -> bool {
        self.turboquant_bits().is_some()
    }
}

impl FromStr for PagedCacheType {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "f8e4m3" => Ok(Self::F8E4M3),
            "turboquant2" => Ok(Self::TurboQuant2),
            "turboquant3" => Ok(Self::TurboQuant3),
            "turboquant4" => Ok(Self::TurboQuant4),
            other => Err(format!(
                "Unexpected `PagedCacheType`, got `{other}` but expected `auto`, `f8e4m3`, `turboquant2`, `turboquant3` or `turboquant4`."
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub block_size: usize,
    pub num_gpu_blocks: usize,
    pub cache_type: PagedCacheType,
}

pub type KVCache = (Tensor, Tensor);

pub struct CacheEngine {
    gpu_cache: Arc<Mutex<Vec<KVCache>>>,
}

impl CacheEngine {
    pub fn new(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        layer_devices: Vec<Option<Device>>,
    ) -> Result<Self> {
        let dtype = cache_config.cache_type.to_dtype(dtype);
        Ok(Self {
            gpu_cache: Arc::new(Mutex::new(Self::allocate_gpu_cache(
                model_config,
                cache_config,
                dtype,
                device,
                layer_devices,
            )?)),
        })
    }

    pub fn get_kv_cache(&self) -> MutexGuard<'_, Vec<KVCache>> {
        // Use blocking lock instead of busy-wait spin loop to avoid CPU waste
        // and potential thread starvation issues
        self.gpu_cache.lock().expect("KV cache mutex was poisoned")
    }

    fn allocate_gpu_cache(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        layer_devices: Vec<Option<Device>>,
    ) -> Result<Vec<KVCache>> {
        let kv_cache_layout = model_config.kv_cache_layout();
        let mut gpu_cache = Vec::new();

        for device in layer_devices
            .iter()
            .take(model_config.num_layers())
            .map(|x| x.as_ref().unwrap_or(device))
        {
            let (key_blocks, value_blocks) = match kv_cache_layout {
                KvCacheLayout::Standard => {
                    let key_block_shape = Self::calculate_key_block_shape(
                        model_config,
                        cache_config.cache_type,
                        dtype,
                        cache_config.block_size,
                    );
                    let value_block_shape = Self::calculate_value_block_shape(
                        model_config,
                        cache_config.cache_type,
                        cache_config.block_size,
                    );
                    #[allow(unused)]
                    let key_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count = cache_config.num_gpu_blocks
                                * key_block_shape.0
                                * key_block_shape.1
                                * key_block_shape.2
                                * key_block_shape.3;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "k_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    cache_config.num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                    key_block_shape.3,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    cache_config.num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                    key_block_shape.3,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    #[allow(unused)]
                    let value_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count = cache_config.num_gpu_blocks
                                * value_block_shape.0
                                * value_block_shape.1
                                * value_block_shape.2
                                * if cache_config.cache_type.is_turboquant() {
                                    1
                                } else {
                                    1
                                };
                            let buffer = dev.new_private_buffer(elem_count, dtype, "v_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            if cache_config.cache_type.is_turboquant() {
                                Tensor::from((
                                    storage,
                                    Shape::from_dims(&[
                                        cache_config.num_gpu_blocks,
                                        value_block_shape.0,
                                        value_block_shape.1,
                                        value_block_shape.2,
                                        1,
                                    ]),
                                ))
                            } else {
                                Tensor::from((
                                    storage,
                                    Shape::from_dims(&[
                                        cache_config.num_gpu_blocks,
                                        value_block_shape.0,
                                        value_block_shape.1,
                                        value_block_shape.2,
                                    ]),
                                ))
                            }
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            if cache_config.cache_type.is_turboquant() {
                                Tensor::empty(
                                    (
                                        cache_config.num_gpu_blocks,
                                        value_block_shape.0,
                                        value_block_shape.1,
                                        value_block_shape.2,
                                        1,
                                    ),
                                    dtype,
                                    device,
                                )?
                            } else {
                                Tensor::empty(
                                    (
                                        cache_config.num_gpu_blocks,
                                        value_block_shape.0,
                                        value_block_shape.1,
                                        value_block_shape.2,
                                    ),
                                    dtype,
                                    device,
                                )?
                            }
                        }
                    };
                    (key_blocks, value_blocks)
                }
                KvCacheLayout::Mla {
                    kv_lora_rank,
                    kpe_head_dim,
                } => {
                    #[allow(unused)]
                    let key_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count = cache_config.num_gpu_blocks
                                * cache_config.block_size
                                * kv_lora_rank;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "k_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    cache_config.num_gpu_blocks,
                                    cache_config.block_size,
                                    kv_lora_rank,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    cache_config.num_gpu_blocks,
                                    cache_config.block_size,
                                    kv_lora_rank,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    #[allow(unused)]
                    let value_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count = cache_config.num_gpu_blocks
                                * cache_config.block_size
                                * kpe_head_dim;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "v_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    cache_config.num_gpu_blocks,
                                    cache_config.block_size,
                                    kpe_head_dim,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    cache_config.num_gpu_blocks,
                                    cache_config.block_size,
                                    kpe_head_dim,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    (key_blocks, value_blocks)
                }
            };
            gpu_cache.push((key_blocks, value_blocks));
        }
        Ok(gpu_cache)
    }

    fn calculate_key_block_shape(
        model_config: &dyn ModelConfigLike,
        cache_type: PagedCacheType,
        dtype: DType,
        block_size: usize,
    ) -> (usize, usize, usize, usize) {
        let (head_bytes, x) = if let Some(bits) = cache_type.turboquant_bits() {
            (
                bits.packed_key_bytes_for_dim(model_config.k_head_dim()),
                1usize,
            )
        } else {
            let element_size = dtype.size_in_bytes();
            let x = 16 / element_size;
            (model_config.k_head_dim(), x)
        };
        (
            model_config.num_kv_heads(),
            head_bytes.div_ceil(x),
            block_size,
            x,
        )
    }

    fn calculate_value_block_shape(
        model_config: &dyn ModelConfigLike,
        cache_type: PagedCacheType,
        block_size: usize,
    ) -> (usize, usize, usize) {
        let value_bytes = if let Some(bits) = cache_type.turboquant_bits() {
            bits.packed_value_bytes_for_dim(model_config.v_head_dim())
        } else {
            model_config.v_head_dim()
        };
        (model_config.num_kv_heads(), value_bytes, block_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::KvCacheLayout;

    struct DummyConfig;

    impl ModelConfigLike for DummyConfig {
        fn max_seq_len(&self) -> usize {
            4096
        }

        fn num_layers(&self) -> usize {
            4
        }

        fn hidden_size(&self) -> usize {
            4096
        }

        fn num_kv_heads(&self) -> usize {
            8
        }

        fn num_attn_heads(&self) -> usize {
            32
        }

        fn k_head_dim(&self) -> usize {
            128
        }

        fn v_head_dim(&self) -> usize {
            128
        }

        fn kv_cache_layout(&self) -> KvCacheLayout {
            KvCacheLayout::Standard
        }
    }

    #[test]
    fn turboquant_block_shapes_include_packed_overhead() {
        let key_shape = CacheEngine::calculate_key_block_shape(
            &DummyConfig,
            PagedCacheType::TurboQuant3,
            DType::U8,
            32,
        );
        let value_shape =
            CacheEngine::calculate_value_block_shape(&DummyConfig, PagedCacheType::TurboQuant3, 32);

        assert_eq!(key_shape, (8, 56, 32, 1));
        assert_eq!(value_shape, (8, 56, 32));
    }

    #[test]
    fn turboquant4_key_shape_tracks_qjl_overhead() {
        let key_shape = CacheEngine::calculate_key_block_shape(
            &DummyConfig,
            PagedCacheType::TurboQuant4,
            DType::U8,
            32,
        );
        let value_shape =
            CacheEngine::calculate_value_block_shape(&DummyConfig, PagedCacheType::TurboQuant4, 32);

        assert_eq!(key_shape, (8, 68, 32, 1));
        assert_eq!(value_shape, (8, 66, 32));
    }

    #[test]
    fn dense_block_shapes_stay_unchanged() {
        let key_shape = CacheEngine::calculate_key_block_shape(
            &DummyConfig,
            PagedCacheType::Auto,
            DType::F16,
            32,
        );
        let value_shape =
            CacheEngine::calculate_value_block_shape(&DummyConfig, PagedCacheType::Auto, 32);

        assert_eq!(key_shape, (8, 16, 32, 8));
        assert_eq!(value_shape, (8, 128, 32));
    }
}
