pub mod cache;
mod experts;

use engine_quant::Shard;

pub use cache::{
    CacheConfig, CacheStats, ColdTier, ExpertBytes, ExpertTriple, MoEExpertCache, Slot, Topology,
};
pub use experts::{MoEExecutionPolicy, MoEExperts, MoEExpertsBackend, MoEExpertsConfig};

pub fn shard(dim: usize, rank: usize, world_size: usize) -> Shard {
    Shard::Simple {
        dim,
        rank,
        world_size,
    }
}
