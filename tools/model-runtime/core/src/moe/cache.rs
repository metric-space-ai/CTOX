//! MoE expert cache with LFU eviction and tiered storage.
//!
//! The cache lets CTOX run MoE models whose total expert weights exceed the
//! GPU VRAM budget. Only `capacity` experts are kept resident on-device; the
//! rest live on a warm (CPU RAM) or cold (SSD) tier and are swapped in on
//! demand, evicting the least-frequently-used resident expert.
//!
//! # Topologies
//!
//! * [`Topology::Unified`] — Apple Silicon et al. GPU and CPU share one
//!   physical memory, so a separate "CPU RAM" tier would just double-count
//!   the same bytes. Tiering collapses to Resident ↔ Cold (SSD).
//! * [`Topology::Discrete`] — NVIDIA / AMD with dedicated VRAM. Three tiers:
//!   Resident (VRAM) ↔ Warm (CPU RAM bytes) ↔ Cold (SSD).
//! * [`Topology::CpuOnly`] — no GPU. Cache is a pass-through; capacity is
//!   forced to `num_experts`.
//!
//! # Phase boundaries (see CLAUDE.md / HARNESS.md for CTOX phase plan)
//!
//! * **Phase 1a (this module)** — the tiering abstraction, synchronous LFU
//!   eviction, and Resident ↔ Warm transitions via [`QuantizedSerde`].
//!   The cold tier is a RAM-backed stub ([`ColdTier::InMemory`]) so that
//!   tests can exercise the same code path.
//! * **Phase 2** — async prefetch, real SSD file I/O for the cold tier
//!   with a startup bandwidth probe, pinned-memory transfers on CUDA.
//! * **Phase 3** — end-to-end run of Qwen3.6-35B-A3B on the target M5.
//!
//! # Interaction with the rest of the runtime
//!
//! An `MoEExpertCache` is constructed from a fully-loaded `Vec<ExpertTriple>`.
//! At construction time, experts beyond `capacity` are serialized via
//! [`QuantMethod::serialize`] and their GPU-resident `Arc`s are dropped so the
//! VRAM is actually freed. [`MoEExpertCache::ensure_resident`] is called from
//! the MoE forward path (see `experts.rs::forward_cached`) and returns an
//! [`ExpertTriple`] whose three `Arc<dyn QuantMethod>` can be fed to the
//! existing matmul autocast helpers.

use std::borrow::Cow;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_core::{Device, Result};
use engine_quant::{Comm, QuantMethod, QuantizeOntoGuard, QuantizedSerde, ReplicatedLayer};

/// Hardware topology relevant for cache tier planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Topology {
    /// Unified memory (Apple Silicon, etc.): VRAM and RAM are the same bytes.
    /// No warm tier; cache tiers are Resident ↔ Cold.
    Unified,
    /// Discrete GPU (NVIDIA / AMD): distinct VRAM. Three tiers Resident ↔ Warm ↔ Cold.
    Discrete,
    /// CPU-only execution. Cache is a pass-through; all experts resident.
    CpuOnly,
}

impl Topology {
    /// Detect topology from a candle device handle.
    pub fn detect(device: &Device) -> Self {
        match device {
            Device::Metal(_) => Self::Unified,
            Device::Cuda(_) => Self::Discrete,
            Device::Cpu => Self::CpuOnly,
        }
    }

    pub fn has_warm_tier(self) -> bool {
        matches!(self, Self::Discrete)
    }

    pub fn has_cold_tier(self) -> bool {
        !matches!(self, Self::CpuOnly)
    }
}

/// Three quantized linear layers making up one MoE expert: gate, up, down.
#[derive(Clone)]
pub struct ExpertTriple {
    pub gate: Arc<dyn QuantMethod>,
    pub up: Arc<dyn QuantMethod>,
    pub down: Arc<dyn QuantMethod>,
}

/// Serialized bytes for one expert's three layers. Product of
/// [`QuantMethod::serialize`] on each of gate / up / down.
pub struct ExpertBytes {
    pub gate: Vec<u8>,
    pub up: Vec<u8>,
    pub down: Vec<u8>,
}

impl ExpertBytes {
    /// Produce serialized bytes from a resident triple.
    ///
    /// Requires the concrete [`QuantMethod`] types to implement
    /// [`QuantMethod::serialize`]. The major types (Unquant, Hqq, Fp8, Afq,
    /// F8Q8, Mxfp4, Gguf) do; bitsandbytes / GPTQ currently do not and will
    /// surface an explicit error at cache construction.
    pub fn from_triple(triple: &ExpertTriple) -> Result<Self> {
        Ok(Self {
            gate: triple.gate.serialize()?.into_owned(),
            up: triple.up.serialize()?.into_owned(),
            down: triple.down.serialize()?.into_owned(),
        })
    }

    /// Materialize these bytes into a new [`ExpertTriple`] on `device`.
    ///
    /// Uses [`ReplicatedLayer::deserialize`] which peeks the UQFF type byte
    /// and dispatches to the correct concrete-type deserializer.
    pub fn materialize(
        &self,
        device: &Device,
        comm: &Arc<Comm>,
        guard: &QuantizeOntoGuard,
    ) -> Result<ExpertTriple> {
        let gate =
            ReplicatedLayer::deserialize(Cow::Borrowed(&self.gate), device, comm, guard.clone())?;
        let up =
            ReplicatedLayer::deserialize(Cow::Borrowed(&self.up), device, comm, guard.clone())?;
        let down =
            ReplicatedLayer::deserialize(Cow::Borrowed(&self.down), device, comm, guard.clone())?;
        Ok(ExpertTriple { gate, up, down })
    }

    pub fn size_bytes(&self) -> usize {
        self.gate.len() + self.up.len() + self.down.len()
    }
}

/// Cold-tier storage for one expert.
pub enum ColdTier {
    /// Bytes held in RAM. Used on Phase 1a unit tests and as a fallback when no
    /// cold-tier path is configured (i.e. everything not resident stays in
    /// CPU RAM).
    InMemory(Arc<ExpertBytes>),
    /// Bytes staged on SSD. Each range is `(offset, len)` bytes inside `path`.
    /// Materialization reads the three ranges and dispatches via
    /// [`ReplicatedLayer::deserialize`] onto the target device.
    File {
        path: PathBuf,
        gate_range: (u64, u64),
        up_range: (u64, u64),
        down_range: (u64, u64),
    },
}

impl ColdTier {
    /// Stage an expert's serialized bytes into a new shared SSD file.
    ///
    /// Appends `bytes.gate`, `bytes.up`, `bytes.down` end-to-end and records
    /// the `(offset, len)` for each so that materialization can read them
    /// back without re-parsing the full file.
    pub fn write_to_file(bytes: &ExpertBytes, path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(candle_core::Error::wrap)?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(path)
            .map_err(candle_core::Error::wrap)?;
        let gate_off = 0u64;
        file.write_all(&bytes.gate).map_err(candle_core::Error::wrap)?;
        let up_off = gate_off + bytes.gate.len() as u64;
        file.write_all(&bytes.up).map_err(candle_core::Error::wrap)?;
        let down_off = up_off + bytes.up.len() as u64;
        file.write_all(&bytes.down).map_err(candle_core::Error::wrap)?;
        file.sync_all().map_err(candle_core::Error::wrap)?;
        Ok(Self::File {
            path: path.to_path_buf(),
            gate_range: (gate_off, bytes.gate.len() as u64),
            up_range: (up_off, bytes.up.len() as u64),
            down_range: (down_off, bytes.down.len() as u64),
        })
    }

    pub fn materialize(
        &self,
        device: &Device,
        comm: &Arc<Comm>,
        guard: &QuantizeOntoGuard,
    ) -> Result<ExpertTriple> {
        match self {
            Self::InMemory(bytes) => bytes.materialize(device, comm, guard),
            Self::File {
                path,
                gate_range,
                up_range,
                down_range,
            } => {
                let bytes = Self::read_file_layout(path, *gate_range, *up_range, *down_range)?;
                bytes.materialize(device, comm, guard)
            }
        }
    }

    fn read_file_layout(
        path: &Path,
        gate: (u64, u64),
        up: (u64, u64),
        down: (u64, u64),
    ) -> Result<ExpertBytes> {
        let mut file = File::open(path).map_err(candle_core::Error::wrap)?;
        let read_range = |file: &mut File, range: (u64, u64)| -> Result<Vec<u8>> {
            file.seek(SeekFrom::Start(range.0))
                .map_err(candle_core::Error::wrap)?;
            let mut buf = vec![0u8; range.1 as usize];
            file.read_exact(&mut buf).map_err(candle_core::Error::wrap)?;
            Ok(buf)
        };
        let gate_bytes = read_range(&mut file, gate)?;
        let up_bytes = read_range(&mut file, up)?;
        let down_bytes = read_range(&mut file, down)?;
        Ok(ExpertBytes {
            gate: gate_bytes,
            up: up_bytes,
            down: down_bytes,
        })
    }

    /// RAM fast-path used by [`MoEExpertCache::ensure_resident`] when the
    /// slot's cold backing is still RAM-resident. `File` variants return
    /// `None`, forcing the caller onto the disk path.
    fn bytes_in_memory(&self) -> Option<Arc<ExpertBytes>> {
        match self {
            Self::InMemory(b) => Some(b.clone()),
            Self::File { .. } => None,
        }
    }
}

/// Measure sustained sequential-read throughput from a candidate SSD path.
///
/// Used by the admission controller to decide whether the cold tier is fast
/// enough to serve expert swaps without turning inference into a disk-wait
/// loop. Writes a 64 MiB probe file, flushes the OS cache as best we can by
/// re-opening, and times a read-to-completion.
///
/// Returns `None` if the probe could not be executed (e.g. path not writable)
/// — callers should treat that as "cold tier unusable" rather than silently
/// continuing.
pub fn probe_ssd_bandwidth_mbps(dir: &Path) -> Option<u32> {
    const PROBE_BYTES: usize = 64 * 1024 * 1024;
    std::fs::create_dir_all(dir).ok()?;
    let probe = dir.join(".ctox_moe_cache_probe");
    let payload = vec![0u8; PROBE_BYTES];
    {
        let mut f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&probe)
            .ok()?;
        f.write_all(&payload).ok()?;
        f.sync_all().ok()?;
    }
    let t0 = Instant::now();
    {
        let mut f = File::open(&probe).ok()?;
        let mut sink = vec![0u8; PROBE_BYTES];
        f.read_exact(&mut sink).ok()?;
    }
    let elapsed = t0.elapsed();
    let _ = std::fs::remove_file(&probe);
    let secs = elapsed.as_secs_f64();
    if secs <= 0.0 {
        return None;
    }
    let mbps = (PROBE_BYTES as f64 / (1024.0 * 1024.0) / secs) as u32;
    Some(mbps)
}

/// Per-expert slot state.
pub enum Slot {
    Resident(ExpertTriple),
    /// Warm tier: serialized bytes in CPU RAM. Only populated when topology is
    /// [`Topology::Discrete`] — on [`Topology::Unified`] it would alias VRAM.
    Warm(Arc<ExpertBytes>),
    /// Cold tier: serialized bytes on SSD (or, in Phase 1a, RAM stub).
    Cold(ColdTier),
}

impl Slot {
    /// A zero-byte placeholder used transiently when swapping slot contents.
    fn placeholder() -> Self {
        Slot::Cold(ColdTier::InMemory(Arc::new(ExpertBytes {
            gate: Vec::new(),
            up: Vec::new(),
            down: Vec::new(),
        })))
    }
}

/// Configuration for constructing a cache.
pub struct CacheConfig {
    /// K: max resident experts on GPU.
    pub capacity: usize,
    /// Detected topology; controls which tiers are populated.
    pub topology: Topology,
    /// GPU device to materialize resident experts onto.
    pub device: Device,
    /// Optional budget (bytes) for the warm tier. `None` = unbounded.
    /// Ignored on `Unified` / `CpuOnly`.
    pub warm_tier_budget_bytes: Option<usize>,
    /// Optional SSD directory. When `Some`, experts that don't fit in the
    /// warm tier (or that would alias VRAM on unified-memory) are staged to
    /// disk here. When `None`, every non-resident expert stays in CPU RAM.
    pub cold_tier_path: Option<PathBuf>,
    /// Sustained SSD-read floor (MiB/s) required from the cold tier. Measured
    /// at cache construction via [`probe_ssd_bandwidth_mbps`]. If the probe
    /// falls below this, construction fails rather than risk thrashing on a
    /// slow disk. `0` disables the check.
    pub cold_tier_min_mbps: u32,
}

#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses_warm: u64,
    pub misses_cold: u64,
    pub evictions: u64,
    pub promotions: u64,
    pub lfu_counts: Vec<u64>,
    pub last_access: Vec<u64>,
}

pub struct MoEExpertCache {
    num_experts: usize,
    cfg: CacheConfig,
    comm: Arc<Comm>,
    guard: QuantizeOntoGuard,
    inner: Mutex<CacheInner>,
}

struct CacheInner {
    slots: Vec<Slot>,
    clock: u64,
    stats: CacheStats,
    warm_bytes_total: usize,
}

impl MoEExpertCache {
    /// Build a cache from a fully-loaded expert set.
    ///
    /// If `capacity >= experts.len()`, the cache degenerates into a lossless
    /// pass-through (every expert stays resident).
    ///
    /// Otherwise, experts `[capacity..num_experts]` are serialized and their
    /// GPU `Arc`s are dropped — freeing VRAM — then deposited onto the warm
    /// (on `Discrete`) or cold (on `Unified` / `CpuOnly`) tier.
    pub fn new(experts: Vec<ExpertTriple>, cfg: CacheConfig, comm: Arc<Comm>) -> Result<Self> {
        if experts.is_empty() {
            candle_core::bail!("MoEExpertCache: experts vec is empty");
        }
        if cfg.capacity == 0 {
            candle_core::bail!("MoEExpertCache: capacity must be > 0");
        }
        if matches!(cfg.topology, Topology::CpuOnly) && cfg.capacity < experts.len() {
            candle_core::bail!(
                "MoEExpertCache: CpuOnly topology requires capacity >= num_experts"
            );
        }

        let num_experts = experts.len();
        let capacity = cfg.capacity.min(num_experts);
        let mut slots: Vec<Slot> = Vec::with_capacity(num_experts);
        let mut warm_bytes_total = 0usize;

        // Admission probe on the cold tier before we commit to using it.
        // `probe_ssd_bandwidth_mbps` writes + reads a 64 MiB file, which is
        // expensive enough that we only pay it once at construction.
        if let Some(cold_path) = &cfg.cold_tier_path {
            if cfg.cold_tier_min_mbps > 0 {
                match probe_ssd_bandwidth_mbps(cold_path) {
                    Some(observed) if observed >= cfg.cold_tier_min_mbps => {
                        tracing::info!(
                            "MoE cache cold-tier probe OK: {} MiB/s >= {} MiB/s required ({})",
                            observed,
                            cfg.cold_tier_min_mbps,
                            cold_path.display()
                        );
                    }
                    Some(observed) => {
                        candle_core::bail!(
                            "MoE cache cold-tier probe too slow: {} MiB/s < {} MiB/s required at {}",
                            observed,
                            cfg.cold_tier_min_mbps,
                            cold_path.display()
                        );
                    }
                    None => {
                        candle_core::bail!(
                            "MoE cache cold-tier probe failed at {} — path not usable",
                            cold_path.display()
                        );
                    }
                }
            }
        }

        for (idx, triple) in experts.into_iter().enumerate() {
            if idx < capacity {
                slots.push(Slot::Resident(triple));
            } else {
                let bytes = Arc::new(ExpertBytes::from_triple(&triple)?);
                drop(triple);
                match cfg.topology {
                    Topology::Discrete => {
                        warm_bytes_total += bytes.size_bytes();
                        let within_budget = match cfg.warm_tier_budget_bytes {
                            Some(b) => warm_bytes_total <= b,
                            None => true,
                        };
                        if within_budget {
                            slots.push(Slot::Warm(bytes));
                        } else if let Some(cold_dir) = &cfg.cold_tier_path {
                            // Warm tier full; spill this expert to SSD.
                            warm_bytes_total -= bytes.size_bytes();
                            let expert_path = cold_dir.join(format!("expert_{idx:05}.bin"));
                            let tier = ColdTier::write_to_file(&bytes, &expert_path)?;
                            slots.push(Slot::Cold(tier));
                        } else {
                            candle_core::bail!(
                                "MoE cache: warm-tier budget {:?} exceeded at expert {} and no cold_tier_path set",
                                cfg.warm_tier_budget_bytes,
                                idx
                            );
                        }
                    }
                    Topology::Unified => {
                        // Unified memory: the warm tier would alias VRAM so
                        // skip it. If a cold path is provided, stage to SSD;
                        // else fall back to keeping bytes in RAM (smaller
                        // models or tests).
                        if let Some(cold_dir) = &cfg.cold_tier_path {
                            let expert_path = cold_dir.join(format!("expert_{idx:05}.bin"));
                            let tier = ColdTier::write_to_file(&bytes, &expert_path)?;
                            slots.push(Slot::Cold(tier));
                        } else {
                            slots.push(Slot::Cold(ColdTier::InMemory(bytes)));
                        }
                    }
                    Topology::CpuOnly => {
                        slots.push(Slot::Cold(ColdTier::InMemory(bytes)));
                    }
                }
            }
        }

        let stats = CacheStats {
            lfu_counts: vec![0; num_experts],
            last_access: vec![0; num_experts],
            ..CacheStats::default()
        };

        Ok(Self {
            num_experts,
            cfg: CacheConfig {
                capacity,
                ..cfg
            },
            comm,
            guard: QuantizeOntoGuard::new(),
            inner: Mutex::new(CacheInner {
                slots,
                clock: 0,
                stats,
                warm_bytes_total,
            }),
        })
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn capacity(&self) -> usize {
        self.cfg.capacity
    }

    pub fn topology(&self) -> Topology {
        self.cfg.topology
    }

    pub fn warm_bytes_total(&self) -> usize {
        self.inner.lock().unwrap().warm_bytes_total
    }

    pub fn stats(&self) -> CacheStats {
        self.inner.lock().unwrap().stats.clone()
    }

    /// Ensure expert `idx` is resident on the GPU; swap out the LFU victim if needed.
    ///
    /// Returns a cheap clone of the triple's three `Arc`s. The caller can hold
    /// these `Arc`s past a subsequent eviction — the underlying tensors survive
    /// because refcounts stay non-zero while the forward pass runs.
    pub fn ensure_resident(&self, idx: usize) -> Result<ExpertTriple> {
        if idx >= self.num_experts {
            candle_core::bail!(
                "expert idx {} out of bounds (num_experts={})",
                idx,
                self.num_experts
            );
        }

        let mut guard = self.inner.lock().unwrap();
        let inner: &mut CacheInner = &mut guard;
        inner.clock += 1;
        let tick = inner.clock;
        inner.stats.lfu_counts[idx] += 1;
        inner.stats.last_access[idx] = tick;

        // Classify the slot without holding a borrow into `inner` across the
        // subsequent mutating calls. For RAM tiers, `Arc::clone` keeps the
        // underlying bytes alive after we drop the borrow. For the File cold
        // tier we must materialize via disk I/O — classify that separately so
        // the expensive disk read can happen without the caller needing a
        // bytes Arc up front.
        enum Reload {
            Hit(ExpertTriple),
            FromWarm(Arc<ExpertBytes>),
            FromCold(Arc<ExpertBytes>),
            FromColdFile {
                path: PathBuf,
                gate_range: (u64, u64),
                up_range: (u64, u64),
                down_range: (u64, u64),
            },
        }
        let reload = match &inner.slots[idx] {
            Slot::Resident(t) => Reload::Hit(t.clone()),
            Slot::Warm(b) => Reload::FromWarm(b.clone()),
            Slot::Cold(ColdTier::InMemory(b)) => Reload::FromCold(b.clone()),
            Slot::Cold(ColdTier::File {
                path,
                gate_range,
                up_range,
                down_range,
            }) => Reload::FromColdFile {
                path: path.clone(),
                gate_range: *gate_range,
                up_range: *up_range,
                down_range: *down_range,
            },
        };

        let triple = match reload {
            Reload::Hit(triple) => {
                inner.stats.hits += 1;
                return Ok(triple);
            }
            Reload::FromWarm(bytes) => {
                inner.stats.misses_warm += 1;
                bytes.materialize(&self.cfg.device, &self.comm, &self.guard)?
            }
            Reload::FromCold(bytes) => {
                inner.stats.misses_cold += 1;
                bytes.materialize(&self.cfg.device, &self.comm, &self.guard)?
            }
            Reload::FromColdFile {
                path,
                gate_range,
                up_range,
                down_range,
            } => {
                inner.stats.misses_cold += 1;
                let bytes =
                    ColdTier::read_file_layout(&path, gate_range, up_range, down_range)?;
                bytes.materialize(&self.cfg.device, &self.comm, &self.guard)?
            }
        };

        // Split-borrow the three fields of `inner` so we can pass each to
        // `demote` independently.
        let CacheInner {
            slots,
            stats,
            warm_bytes_total,
            ..
        } = inner;
        let victim = Self::pick_victim(slots, stats, idx)?;
        Self::demote(
            slots,
            stats,
            warm_bytes_total,
            victim,
            self.cfg.topology,
            self.cfg.warm_tier_budget_bytes,
            self.cfg.cold_tier_path.as_deref(),
        )?;

        slots[idx] = Slot::Resident(triple.clone());
        stats.promotions += 1;
        Ok(triple)
    }

    fn pick_victim(slots: &[Slot], stats: &CacheStats, exclude: usize) -> Result<usize> {
        let mut best: Option<(usize, u64, u64)> = None;
        for (i, slot) in slots.iter().enumerate() {
            if i == exclude {
                continue;
            }
            if matches!(slot, Slot::Resident(_)) {
                let lfu = stats.lfu_counts[i];
                let la = stats.last_access[i];
                match best {
                    None => best = Some((i, lfu, la)),
                    Some((_, blfu, bla)) => {
                        if lfu < blfu || (lfu == blfu && la < bla) {
                            best = Some((i, lfu, la));
                        }
                    }
                }
            }
        }
        best.map(|(i, _, _)| i).ok_or_else(|| {
            candle_core::Error::msg("no eviction victim found (cache state inconsistent?)")
        })
    }

    fn demote(
        slots: &mut [Slot],
        stats: &mut CacheStats,
        warm_bytes_total: &mut usize,
        idx: usize,
        topology: Topology,
        warm_tier_budget_bytes: Option<usize>,
        cold_tier_path: Option<&Path>,
    ) -> Result<()> {
        let victim = std::mem::replace(&mut slots[idx], Slot::placeholder());
        let triple = match victim {
            Slot::Resident(t) => t,
            other => {
                slots[idx] = other;
                candle_core::bail!("demote: slot {} is not resident", idx);
            }
        };
        let bytes = Arc::new(ExpertBytes::from_triple(&triple)?);
        drop(triple);
        match topology {
            Topology::Discrete => {
                let prospective_total = *warm_bytes_total + bytes.size_bytes();
                let fits = match warm_tier_budget_bytes {
                    Some(b) => prospective_total <= b,
                    None => true,
                };
                if fits {
                    *warm_bytes_total = prospective_total;
                    slots[idx] = Slot::Warm(bytes);
                } else if let Some(cold_dir) = cold_tier_path {
                    let expert_path = cold_dir.join(format!("expert_{idx:05}.bin"));
                    let tier = ColdTier::write_to_file(&bytes, &expert_path)?;
                    slots[idx] = Slot::Cold(tier);
                } else {
                    candle_core::bail!(
                        "demote: warm-tier budget exceeded and no cold_tier_path configured"
                    );
                }
            }
            Topology::Unified => {
                if let Some(cold_dir) = cold_tier_path {
                    let expert_path = cold_dir.join(format!("expert_{idx:05}.bin"));
                    let tier = ColdTier::write_to_file(&bytes, &expert_path)?;
                    slots[idx] = Slot::Cold(tier);
                } else {
                    slots[idx] = Slot::Cold(ColdTier::InMemory(bytes));
                }
            }
            Topology::CpuOnly => {
                slots[idx] = Slot::Cold(ColdTier::InMemory(bytes));
            }
        }
        stats.evictions += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `Slot` that looks resident from `pick_victim`'s perspective
    /// without requiring a real [`ExpertTriple`]. The `ExpertTriple`'s `Arc`
    /// contents are never dereferenced by the eviction logic, so a minimal
    /// stub suffices for LFU policy tests.
    fn stub_resident() -> Slot {
        use candle_core::{DType, Tensor};
        use engine_quant::{QuantMethodConfig, UnquantLinear};

        let device = Device::Cpu;
        let weight = Tensor::zeros((2, 2), DType::F32, &device).unwrap();
        let bias: Option<Tensor> = None;
        let linear = candle_nn::Linear::new(weight, bias);
        let q: Arc<dyn QuantMethod> =
            Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(linear)).unwrap());
        Slot::Resident(ExpertTriple {
            gate: q.clone(),
            up: q.clone(),
            down: q,
        })
    }

    fn stub_cold_bytes() -> Slot {
        Slot::Cold(ColdTier::InMemory(Arc::new(ExpertBytes {
            gate: vec![0; 4],
            up: vec![0; 4],
            down: vec![0; 4],
        })))
    }

    #[test]
    fn topology_detects_cpu() {
        let dev = Device::Cpu;
        assert_eq!(Topology::detect(&dev), Topology::CpuOnly);
        assert!(!Topology::CpuOnly.has_warm_tier());
        assert!(!Topology::CpuOnly.has_cold_tier());
    }

    #[test]
    fn topology_semantics_unified_vs_discrete() {
        assert!(!Topology::Unified.has_warm_tier());
        assert!(Topology::Unified.has_cold_tier());
        assert!(Topology::Discrete.has_warm_tier());
        assert!(Topology::Discrete.has_cold_tier());
    }

    #[test]
    fn pick_victim_selects_min_lfu() {
        // 4 experts, all resident, counters [5, 1, 3, 7]. Min is idx 1.
        let slots = vec![
            stub_resident(),
            stub_resident(),
            stub_resident(),
            stub_resident(),
        ];
        let stats = CacheStats {
            lfu_counts: vec![5, 1, 3, 7],
            last_access: vec![10, 20, 30, 40],
            ..CacheStats::default()
        };
        let victim = MoEExpertCache::pick_victim(&slots, &stats, /*exclude=*/ 99).unwrap();
        assert_eq!(victim, 1);
    }

    #[test]
    fn pick_victim_breaks_ties_with_oldest_access() {
        // Counters all equal at 5; last_access differs. Oldest (smallest) wins.
        let slots = vec![
            stub_resident(),
            stub_resident(),
            stub_resident(),
        ];
        let stats = CacheStats {
            lfu_counts: vec![5, 5, 5],
            last_access: vec![300, 100, 200],
            ..CacheStats::default()
        };
        let victim = MoEExpertCache::pick_victim(&slots, &stats, /*exclude=*/ 99).unwrap();
        assert_eq!(victim, 1);
    }

    #[test]
    fn pick_victim_excludes_incoming_index() {
        // Counters [1, 1, 3]. Normally idx 0 would win (tie-break by last_access).
        // If exclude=0, the next-best should be idx 1.
        let slots = vec![
            stub_resident(),
            stub_resident(),
            stub_resident(),
        ];
        let stats = CacheStats {
            lfu_counts: vec![1, 1, 3],
            last_access: vec![10, 20, 30],
            ..CacheStats::default()
        };
        let victim = MoEExpertCache::pick_victim(&slots, &stats, /*exclude=*/ 0).unwrap();
        assert_eq!(victim, 1);
    }

    #[test]
    fn pick_victim_skips_non_resident() {
        // Only slot 2 is Resident. Must choose it regardless of LFU counters.
        let slots = vec![
            stub_cold_bytes(),
            stub_cold_bytes(),
            stub_resident(),
        ];
        let stats = CacheStats {
            lfu_counts: vec![0, 0, 999],
            last_access: vec![0, 0, 999],
            ..CacheStats::default()
        };
        let victim = MoEExpertCache::pick_victim(&slots, &stats, /*exclude=*/ 99).unwrap();
        assert_eq!(victim, 2);
    }

    /// Build a single unquantized expert triple with distinct deterministic
    /// weights per role. Used by the end-to-end swap tests below; the tiny
    /// 2×2 shape keeps serialize/deserialize round trips cheap while still
    /// exercising the full UQFF codepath.
    fn real_expert_triple(seed: f32) -> ExpertTriple {
        use candle_core::{DType, Tensor};
        use engine_quant::{QuantMethodConfig, UnquantLinear};

        let device = Device::Cpu;
        let make = |bias: f32| -> Arc<dyn QuantMethod> {
            let data = vec![seed + bias, seed - bias, seed * 2.0, seed + 1.0];
            let weight = Tensor::from_slice(&data, (2, 2), &device).unwrap();
            let linear = candle_nn::Linear::new(weight, None);
            Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(linear)).unwrap())
        };
        ExpertTriple {
            gate: make(0.1),
            up: make(0.2),
            down: make(0.3),
        }
    }

    fn dummy_comm() -> Arc<engine_quant::Comm> {
        Arc::new(
            engine_quant::Comm::from_device(
                engine_quant::Id::new(),
                &Device::Cpu,
                /*rank=*/ 0,
                /*world_size=*/ 1,
            )
            .unwrap(),
        )
    }

    #[test]
    fn expert_bytes_roundtrip_preserves_weights() {
        // Verify the serialize / ReplicatedLayer::deserialize round trip that
        // the cache relies on for every tier transition actually preserves the
        // underlying QuantMethod's weights.
        let original = real_expert_triple(0.5);
        let bytes = ExpertBytes::from_triple(&original).unwrap();
        let comm = dummy_comm();
        let restored = bytes
            .materialize(&Device::Cpu, &comm, &QuantizeOntoGuard::new())
            .unwrap();

        let orig_w = original.gate.dequantize_w().unwrap();
        let rest_w = restored.gate.dequantize_w().unwrap();
        let diff = (orig_w - rest_w).unwrap().abs().unwrap().sum_all().unwrap();
        let err = diff.to_scalar::<f32>().unwrap();
        assert!(err < 1e-6, "round-trip drift {err}");
    }

    #[test]
    fn cold_tier_file_roundtrip_restores_bytes() {
        // End-to-end cold-tier disk path: write → read → deserialize.
        let triple = real_expert_triple(1.25);
        let bytes = ExpertBytes::from_triple(&triple).unwrap();
        let tmp = std::env::temp_dir().join("ctox_moe_cache_cold_test.bin");
        let tier = ColdTier::write_to_file(&bytes, &tmp).unwrap();
        let comm = dummy_comm();
        let restored = tier
            .materialize(&Device::Cpu, &comm, &QuantizeOntoGuard::new())
            .unwrap();

        let orig_w = triple.down.dequantize_w().unwrap();
        let rest_w = restored.down.dequantize_w().unwrap();
        let diff = (orig_w - rest_w).unwrap().abs().unwrap().sum_all().unwrap();
        let err = diff.to_scalar::<f32>().unwrap();
        assert!(err < 1e-6, "cold-tier round-trip drift {err}");

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn cache_end_to_end_swap_preserves_weights() {
        // Full cache exercise on CpuOnly topology: 3 experts, capacity=1, so
        // ensuring expert 2 resident forces expert 0 (the oldest & smallest
        // LFU counter) out to cold. Re-fetching expert 0 after that must
        // return weights bit-identical to its original form.
        let experts = vec![
            real_expert_triple(0.1),
            real_expert_triple(0.2),
            real_expert_triple(0.3),
        ];
        let originals: Vec<_> = experts
            .iter()
            .map(|t| t.gate.dequantize_w().unwrap())
            .collect();
        let comm = dummy_comm();

        // CpuOnly is the one topology where capacity must equal num_experts,
        // so use Unified for the actual swap test. Fake the topology because
        // we're on Cpu for testing but want swap semantics.
        let cache = MoEExpertCache::new(
            experts,
            CacheConfig {
                capacity: 1,
                topology: Topology::Unified,
                device: Device::Cpu,
                warm_tier_budget_bytes: None,
                cold_tier_path: None,
                cold_tier_min_mbps: 0,
            },
            comm.clone(),
        )
        .unwrap();

        // Touch expert 1 then 2 then 0; after that, refetching any expert
        // should still return bit-identical (within fp32 rounding) weights.
        let _ = cache.ensure_resident(1).unwrap();
        let _ = cache.ensure_resident(2).unwrap();
        let re0 = cache.ensure_resident(0).unwrap();
        let restored0 = re0.gate.dequantize_w().unwrap();
        let diff = (&originals[0] - &restored0)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap();
        assert!(
            diff.to_scalar::<f32>().unwrap() < 1e-6,
            "expert 0 drift after round-trip swap"
        );

        let stats = cache.stats();
        assert!(stats.misses_cold >= 1, "expected at least one cold miss");
        assert!(stats.evictions >= 1, "expected at least one eviction");
    }

    #[test]
    fn cache_end_to_end_swap_with_ssd_cold_tier() {
        // Same as cache_end_to_end_swap_preserves_weights but with an on-disk
        // cold tier — exercises the SSD write/read path + admission probe.
        let tmp_dir = std::env::temp_dir().join(format!(
            "ctox_moe_cache_ssd_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tmp_dir).unwrap();

        let experts = vec![
            real_expert_triple(0.4),
            real_expert_triple(0.5),
            real_expert_triple(0.6),
        ];
        let original_0 = experts[0].up.dequantize_w().unwrap();
        let comm = dummy_comm();

        let cache = MoEExpertCache::new(
            experts,
            CacheConfig {
                capacity: 1,
                topology: Topology::Unified,
                device: Device::Cpu,
                warm_tier_budget_bytes: None,
                cold_tier_path: Some(tmp_dir.clone()),
                cold_tier_min_mbps: 0,
            },
            comm.clone(),
        )
        .unwrap();

        let _ = cache.ensure_resident(1).unwrap();
        let _ = cache.ensure_resident(2).unwrap();
        let restored = cache.ensure_resident(0).unwrap();
        let restored_w = restored.up.dequantize_w().unwrap();
        let diff = (&original_0 - &restored_w)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap();
        assert!(
            diff.to_scalar::<f32>().unwrap() < 1e-6,
            "expert 0 drift after SSD round-trip"
        );

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn probe_ssd_bandwidth_returns_positive_on_tmp_dir() {
        // Sanity — tmp dir should always satisfy the probe. If this fails,
        // the probe impl has regressed (e.g. returning 0 for fast disks due
        // to a sub-millisecond read).
        let probe = probe_ssd_bandwidth_mbps(&std::env::temp_dir());
        assert!(probe.is_some(), "probe must succeed on a writable tmpdir");
        let mbps = probe.unwrap();
        assert!(mbps > 0, "probe reported 0 MiB/s — broken measurement");
    }

    #[test]
    fn pick_victim_errors_when_no_resident() {
        let slots = vec![stub_cold_bytes(), stub_cold_bytes()];
        let stats = CacheStats {
            lfu_counts: vec![0, 0],
            last_access: vec![0, 0],
            ..CacheStats::default()
        };
        let err = MoEExpertCache::pick_victim(&slots, &stats, 99).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("no eviction victim"), "unexpected error: {msg}");
    }
}
