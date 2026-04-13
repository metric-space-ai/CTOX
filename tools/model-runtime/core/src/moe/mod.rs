mod experts;

use engine_quant::Shard;

pub use experts::{MoEExecutionPolicy, MoEExperts, MoEExpertsBackend, MoEExpertsConfig};

pub fn shard(dim: usize, rank: usize, world_size: usize) -> Shard {
    Shard::Simple {
        dim,
        rank,
        world_size,
    }
}
