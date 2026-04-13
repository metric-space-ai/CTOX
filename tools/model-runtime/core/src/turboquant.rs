use std::f32::consts::{PI, SQRT_2};

use half::f16;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

const TURBO3_BLOCK_SIZE: usize = 32;
const TURBO3_GROUP_SIZE: usize = 128;
const TURBO3_BLOCK_BYTES: usize = 14;
const TURBO3_CENTROIDS: [f32; 8] = [
    -0.190685, -0.117832, -0.065717, -0.021460, 0.021460, 0.065717, 0.117832, 0.190685,
];
const TURBO3_MIDPOINTS: [f32; 7] = [
    -0.1542585, -0.0917745, -0.0435885, 0.0, 0.0435885, 0.0917745, 0.1542585,
];
const TURBO_WHT_SIGNS1: [f32; 128] = [
    -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0,
    1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0,
    -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0,
    -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0,
    1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
    1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
    -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
];
const TURBO_WHT_SIGNS2: [f32; 128] = [
    1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0,
    1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
    1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
    -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0,
    -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0,
    -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
    -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TurboQuantBits {
    Two,
    Three,
    Four,
}

impl TurboQuantBits {
    pub fn cache_type_name(self) -> &'static str {
        match self {
            Self::Two => "turboquant2",
            Self::Three => "turboquant3",
            Self::Four => "turboquant4",
        }
    }

    pub fn total_bits(self) -> usize {
        match self {
            Self::Two => 2,
            Self::Three => 3,
            Self::Four => 4,
        }
    }

    pub fn key_mse_bits(self) -> usize {
        match self {
            Self::Two => 2,
            Self::Three => 3,
            Self::Four => 3,
        }
    }

    pub fn value_bits(self) -> usize {
        self.total_bits()
    }

    pub fn qjl_enabled(self) -> bool {
        matches!(self, Self::Four)
    }

    pub fn key_bits_per_dim(self) -> f32 {
        let packed = self.packed_key_bytes_for_dim(128) as f32;
        packed * 8.0 / 128.0
    }

    pub fn packed_key_bytes_for_dim(self, dim: usize) -> usize {
        if matches!(self, Self::Three) {
            assert_eq!(dim % TURBO3_BLOCK_SIZE, 0);
            return (dim / TURBO3_BLOCK_SIZE) * TURBO3_BLOCK_BYTES;
        }
        let quant_bytes = (dim * self.key_mse_bits()).div_ceil(8);
        let qjl_bytes = if self.qjl_enabled() {
            dim.div_ceil(8)
        } else {
            0
        };
        let scale_bytes = 2 + if self.qjl_enabled() { 2 } else { 0 };
        scale_bytes + quant_bytes + qjl_bytes
    }

    pub fn packed_value_bytes_for_dim(self, dim: usize) -> usize {
        if matches!(self, Self::Three) {
            assert_eq!(dim % TURBO3_BLOCK_SIZE, 0);
            return (dim / TURBO3_BLOCK_SIZE) * TURBO3_BLOCK_BYTES;
        }
        2 + (dim * self.value_bits()).div_ceil(8)
    }

    pub fn supports_storage_head_dim(self, dim: usize) -> bool {
        match self {
            Self::Three => dim % TURBO3_BLOCK_SIZE == 0,
            Self::Two | Self::Four => true,
        }
    }

    pub fn supports_rotation_head_dim(self, dim: usize) -> bool {
        match self {
            Self::Three => dim % TURBO3_GROUP_SIZE == 0,
            Self::Two | Self::Four => true,
        }
    }

    pub fn supports_head_dim(self, dim: usize) -> bool {
        self.supports_storage_head_dim(dim)
    }
}

#[derive(Debug, Clone)]
pub struct TurboQuantArtifacts {
    dim: usize,
    bits: TurboQuantBits,
    centroids_key: Vec<f32>,
    centroids_value: Vec<f32>,
    rotation: Vec<f32>,
    qjl: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct CompressedTurboQuantKey {
    pub mse_indices: Vec<u8>,
    pub qjl_signs: Vec<i8>,
    pub residual_norm: f32,
    pub norm: f32,
    pub block_norms: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct CompressedTurboQuantValue {
    pub indices: Vec<u8>,
    pub norm: f32,
    pub block_norms: Vec<f32>,
}

impl TurboQuantArtifacts {
    pub fn new(dim: usize, bits: TurboQuantBits, seed: u64) -> Self {
        if bits == TurboQuantBits::Three {
            return Self {
                dim,
                bits,
                centroids_key: Vec::new(),
                centroids_value: Vec::new(),
                rotation: Vec::new(),
                qjl: Vec::new(),
            };
        }
        let rotation = random_orthogonal(dim, seed);
        let qjl = random_orthogonal(dim, seed.wrapping_add(0x9E37_79B9));
        let centroids_key = gaussian_lloyd_max_centroids(dim, bits.key_mse_bits());
        let centroids_value = gaussian_lloyd_max_centroids(dim, bits.value_bits());
        Self {
            dim,
            bits,
            centroids_key,
            centroids_value,
            rotation,
            qjl,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn bits(&self) -> TurboQuantBits {
        self.bits
    }

    pub fn compress_key(&self, input: &[f32]) -> CompressedTurboQuantKey {
        self.compress_key_with_rotation(input, true)
    }

    pub fn compress_key_with_rotation(
        &self,
        input: &[f32],
        rotate_turbo3: bool,
    ) -> CompressedTurboQuantKey {
        assert_eq!(input.len(), self.dim);
        if self.bits == TurboQuantBits::Three {
            let (mse_indices, block_norms) =
                turbo3_quantize_reference(input, rotate_turbo3, turbo3_plain_group_size(self.dim));
            return CompressedTurboQuantKey {
                mse_indices,
                qjl_signs: Vec::new(),
                residual_norm: 0.0,
                norm: 0.0,
                block_norms,
            };
        }
        let norm = l2_norm(input);
        let normalized = if norm > 0.0 {
            input.iter().map(|x| *x / norm).collect::<Vec<_>>()
        } else {
            vec![0.0; self.dim]
        };
        let rotated = matvec(&self.rotation, &normalized, self.dim);
        let mse_indices = rotated
            .iter()
            .map(|&x| nearest_centroid_index(x, &self.centroids_key) as u8)
            .collect::<Vec<_>>();
        let (qjl_signs, residual_norm) = if self.bits.qjl_enabled() {
            let rotated_hat = mse_indices
                .iter()
                .map(|&i| self.centroids_key[i as usize])
                .collect::<Vec<_>>();
            let mut mse_recon = matvec_transposed(&self.rotation, &rotated_hat, self.dim);
            for v in &mut mse_recon {
                *v *= norm;
            }
            let residual = input
                .iter()
                .zip(mse_recon.iter())
                .map(|(x, xhat)| x - xhat)
                .collect::<Vec<_>>();
            let residual_norm = l2_norm(&residual);
            let projected = matvec(&self.qjl, &residual, self.dim);
            let qjl_signs = projected
                .into_iter()
                .map(|v| if v >= 0.0 { 1 } else { -1 })
                .collect::<Vec<_>>();
            (qjl_signs, residual_norm)
        } else {
            (Vec::new(), 0.0)
        };
        CompressedTurboQuantKey {
            mse_indices,
            qjl_signs,
            residual_norm,
            norm,
            block_norms: Vec::new(),
        }
    }

    pub fn pack_key(&self, key: &CompressedTurboQuantKey) -> Vec<u8> {
        if self.bits == TurboQuantBits::Three {
            return turbo3_pack_blocks(&key.mse_indices, &key.block_norms);
        }
        let mut out = Vec::with_capacity(self.bits.packed_key_bytes_for_dim(self.dim));
        out.extend_from_slice(&f16::from_f32(key.norm).to_bits().to_le_bytes());
        if self.bits.qjl_enabled() {
            out.extend_from_slice(&f16::from_f32(key.residual_norm).to_bits().to_le_bytes());
        }
        out.extend_from_slice(&pack_bits(&key.mse_indices, self.bits.key_mse_bits()));
        if self.bits.qjl_enabled() {
            out.extend_from_slice(&pack_signs(&key.qjl_signs));
        }
        out
    }

    pub fn unpack_key(&self, payload: &[u8]) -> CompressedTurboQuantKey {
        assert_eq!(payload.len(), self.bits.packed_key_bytes_for_dim(self.dim));
        if self.bits == TurboQuantBits::Three {
            let (mse_indices, block_norms) = turbo3_unpack_blocks(payload);
            return CompressedTurboQuantKey {
                mse_indices,
                qjl_signs: Vec::new(),
                residual_norm: 0.0,
                norm: 0.0,
                block_norms,
            };
        }
        let mut offset = 0;
        let norm =
            f16::from_bits(u16::from_le_bytes([payload[offset], payload[offset + 1]])).to_f32();
        offset += 2;
        let residual_norm = if self.bits.qjl_enabled() {
            let value =
                f16::from_bits(u16::from_le_bytes([payload[offset], payload[offset + 1]])).to_f32();
            offset += 2;
            value
        } else {
            0.0
        };
        let mse_bytes = (self.dim * self.bits.key_mse_bits()).div_ceil(8);
        let mse_indices = unpack_bits(
            &payload[offset..offset + mse_bytes],
            self.dim,
            self.bits.key_mse_bits(),
        );
        offset += mse_bytes;
        let qjl_signs = if self.bits.qjl_enabled() {
            unpack_signs(&payload[offset..], self.dim)
        } else {
            Vec::new()
        };
        CompressedTurboQuantKey {
            mse_indices,
            qjl_signs,
            residual_norm,
            norm,
            block_norms: Vec::new(),
        }
    }

    pub fn reconstruct_key_mse(&self, key: &CompressedTurboQuantKey) -> Vec<f32> {
        self.reconstruct_key_mse_with_rotation(key, true)
    }

    pub fn reconstruct_key_mse_with_rotation(
        &self,
        key: &CompressedTurboQuantKey,
        rotate_turbo3: bool,
    ) -> Vec<f32> {
        if self.bits == TurboQuantBits::Three {
            return turbo3_inverse_reference(
                &key.mse_indices,
                &key.block_norms,
                rotate_turbo3,
                turbo3_plain_group_size(self.dim),
            );
        }
        let rotated_hat = key
            .mse_indices
            .iter()
            .map(|&i| self.centroids_key[i as usize])
            .collect::<Vec<_>>();
        let mut out = matvec_transposed(&self.rotation, &rotated_hat, self.dim);
        for v in &mut out {
            *v *= key.norm;
        }
        out
    }

    pub fn reconstruct_key_mse_rotated(&self, key: &CompressedTurboQuantKey) -> Vec<f32> {
        self.reconstruct_key_mse_rotated_with_rotation(key, true)
    }

    pub fn reconstruct_key_mse_rotated_with_rotation(
        &self,
        key: &CompressedTurboQuantKey,
        rotate_turbo3: bool,
    ) -> Vec<f32> {
        if self.bits == TurboQuantBits::Three {
            return if rotate_turbo3 {
                turbo3_rotated_reference(&key.mse_indices, &key.block_norms)
            } else {
                turbo3_inverse_blocks_reference_with_centroids(
                    &key.mse_indices,
                    &key.block_norms,
                    &turbo3_plain_centroids(turbo3_plain_group_size(self.dim)),
                )
            };
        }
        let mut rotated_hat = key
            .mse_indices
            .iter()
            .map(|&i| self.centroids_key[i as usize])
            .collect::<Vec<_>>();
        for v in &mut rotated_hat {
            *v *= key.norm;
        }
        rotated_hat
    }

    pub fn estimate_inner_product(&self, query: &[f32], key: &CompressedTurboQuantKey) -> f32 {
        assert_eq!(query.len(), self.dim);
        let mse_recon = self.reconstruct_key_mse(key);
        let term1 = dot(query, &mse_recon);
        if !self.bits.qjl_enabled() {
            return term1;
        }
        let q_projected = matvec(&self.qjl, query, self.dim);
        let qjl_ip = q_projected
            .iter()
            .zip(key.qjl_signs.iter())
            .map(|(q, s)| q * (*s as f32))
            .sum::<f32>();
        let correction_scale = (PI / 2.0).sqrt() / self.dim as f32;
        term1 + key.residual_norm * correction_scale * qjl_ip
    }

    pub fn compress_value(&self, input: &[f32]) -> CompressedTurboQuantValue {
        self.compress_value_with_rotation(input, true)
    }

    pub fn compress_value_with_rotation(
        &self,
        input: &[f32],
        rotate_turbo3: bool,
    ) -> CompressedTurboQuantValue {
        assert_eq!(input.len(), self.dim);
        if self.bits == TurboQuantBits::Three {
            let (indices, block_norms) =
                turbo3_quantize_reference(input, rotate_turbo3, turbo3_plain_group_size(self.dim));
            return CompressedTurboQuantValue {
                indices,
                norm: 0.0,
                block_norms,
            };
        }
        let norm = l2_norm(input);
        let normalized = if norm > 0.0 {
            input.iter().map(|x| *x / norm).collect::<Vec<_>>()
        } else {
            vec![0.0; self.dim]
        };
        let rotated = matvec(&self.rotation, &normalized, self.dim);
        let indices = rotated
            .iter()
            .map(|&x| nearest_centroid_index(x, &self.centroids_value) as u8)
            .collect::<Vec<_>>();
        CompressedTurboQuantValue {
            indices,
            norm,
            block_norms: Vec::new(),
        }
    }

    pub fn pack_value(&self, value: &CompressedTurboQuantValue) -> Vec<u8> {
        if self.bits == TurboQuantBits::Three {
            return turbo3_pack_blocks(&value.indices, &value.block_norms);
        }
        let mut out = Vec::with_capacity(self.bits.packed_value_bytes_for_dim(self.dim));
        out.extend_from_slice(&f16::from_f32(value.norm).to_bits().to_le_bytes());
        out.extend_from_slice(&pack_bits(&value.indices, self.bits.value_bits()));
        out
    }

    pub fn unpack_value(&self, payload: &[u8]) -> CompressedTurboQuantValue {
        assert_eq!(
            payload.len(),
            self.bits.packed_value_bytes_for_dim(self.dim)
        );
        if self.bits == TurboQuantBits::Three {
            let (indices, block_norms) = turbo3_unpack_blocks(payload);
            return CompressedTurboQuantValue {
                indices,
                norm: 0.0,
                block_norms,
            };
        }
        let norm = f16::from_bits(u16::from_le_bytes([payload[0], payload[1]])).to_f32();
        let indices = unpack_bits(&payload[2..], self.dim, self.bits.value_bits());
        CompressedTurboQuantValue {
            indices,
            norm,
            block_norms: Vec::new(),
        }
    }

    pub fn decompress_value(&self, value: &CompressedTurboQuantValue) -> Vec<f32> {
        self.decompress_value_with_rotation(value, true)
    }

    pub fn decompress_value_with_rotation(
        &self,
        value: &CompressedTurboQuantValue,
        rotate_turbo3: bool,
    ) -> Vec<f32> {
        if self.bits == TurboQuantBits::Three {
            return turbo3_inverse_reference(
                &value.indices,
                &value.block_norms,
                rotate_turbo3,
                turbo3_plain_group_size(self.dim),
            );
        }
        let rotated = value
            .indices
            .iter()
            .map(|&i| self.centroids_value[i as usize])
            .collect::<Vec<_>>();
        let mut out = matvec_transposed(&self.rotation, &rotated, self.dim);
        for v in &mut out {
            *v *= value.norm;
        }
        out
    }

    pub fn decompress_value_rotated(&self, value: &CompressedTurboQuantValue) -> Vec<f32> {
        self.decompress_value_rotated_with_rotation(value, true)
    }

    pub fn decompress_value_rotated_with_rotation(
        &self,
        value: &CompressedTurboQuantValue,
        rotate_turbo3: bool,
    ) -> Vec<f32> {
        if self.bits == TurboQuantBits::Three {
            return if rotate_turbo3 {
                turbo3_rotated_reference(&value.indices, &value.block_norms)
            } else {
                turbo3_inverse_blocks_reference_with_centroids(
                    &value.indices,
                    &value.block_norms,
                    &turbo3_plain_centroids(turbo3_plain_group_size(self.dim)),
                )
            };
        }
        let mut rotated = value
            .indices
            .iter()
            .map(|&i| self.centroids_value[i as usize])
            .collect::<Vec<_>>();
        for v in &mut rotated {
            *v *= value.norm;
        }
        rotated
    }

    pub fn estimated_key_storage_bytes_per_token(&self) -> usize {
        self.bits.packed_key_bytes_for_dim(self.dim)
    }

    pub fn estimated_value_storage_bytes_per_token(&self) -> usize {
        self.bits.packed_value_bytes_for_dim(self.dim)
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn turbo3_plain_group_size(dim: usize) -> usize {
    if dim % TURBO3_GROUP_SIZE == 0 {
        TURBO3_GROUP_SIZE
    } else {
        dim
    }
}

fn turbo3_plain_centroids(group_size: usize) -> Vec<f32> {
    gaussian_lloyd_max_centroids(group_size, 3)
}

fn turbo3_quantize_reference(
    input: &[f32],
    rotate: bool,
    plain_group_size: usize,
) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(input.len() % TURBO3_BLOCK_SIZE, 0);
    if rotate && input.len() % TURBO3_GROUP_SIZE == 0 {
        turbo3_quantize_rotated_norm_corrected_reference(input)
    } else {
        turbo3_quantize_norm_corrected_reference(
            input,
            plain_group_size,
            &turbo3_plain_centroids(plain_group_size),
        )
    }
}

fn turbo3_quantize_rotated_norm_corrected_reference(input: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let mut indices = Vec::with_capacity(input.len());
    let mut block_norms = Vec::with_capacity(input.len() / TURBO3_BLOCK_SIZE);
    for group in input.chunks(TURBO3_GROUP_SIZE) {
        let group_norm = l2_norm(group);
        let inv_group_norm = if group_norm > 1e-10 {
            1.0 / group_norm
        } else {
            0.0
        };
        let mut rotated = group
            .iter()
            .map(|&value| value * inv_group_norm)
            .collect::<Vec<_>>();
        turbo_rotate_forward_reference(&mut rotated);
        let group_indices = rotated
            .iter()
            .map(|&value| turbo3_find_nearest_index(value))
            .collect::<Vec<_>>();
        let recon_norm = group_indices
            .iter()
            .map(|&idx| {
                let c = TURBO3_CENTROIDS[idx as usize];
                c * c
            })
            .sum::<f32>()
            .sqrt();
        let corrected_norm = if recon_norm > 1e-10 {
            group_norm / recon_norm
        } else {
            group_norm
        };
        indices.extend_from_slice(&group_indices);
        block_norms.extend(std::iter::repeat_n(
            corrected_norm,
            group.len() / TURBO3_BLOCK_SIZE,
        ));
    }
    (indices, block_norms)
}

fn turbo3_quantize_norm_corrected_reference(
    input: &[f32],
    group_size: usize,
    centroids: &[f32],
) -> (Vec<u8>, Vec<f32>) {
    let mut indices = Vec::with_capacity(input.len());
    let mut block_norms = Vec::with_capacity(input.len() / TURBO3_BLOCK_SIZE);
    for group in input.chunks(group_size) {
        let group_norm = l2_norm(group);
        let inv_group_norm = if group_norm > 1e-10 {
            1.0 / group_norm
        } else {
            0.0
        };
        let group_indices = group
            .iter()
            .map(|&x| nearest_centroid_index(x * inv_group_norm, centroids) as u8)
            .collect::<Vec<_>>();
        let recon_norm = group_indices
            .iter()
            .map(|&idx| {
                let c = centroids[idx as usize];
                c * c
            })
            .sum::<f32>()
            .sqrt();
        let corrected_norm = if recon_norm > 1e-10 {
            group_norm / recon_norm
        } else {
            group_norm
        };
        indices.extend_from_slice(&group_indices);
        block_norms.extend(std::iter::repeat_n(
            corrected_norm,
            group.len() / TURBO3_BLOCK_SIZE,
        ));
    }
    (indices, block_norms)
}

fn turbo3_pack_blocks(indices: &[u8], block_norms: &[f32]) -> Vec<u8> {
    assert_eq!(indices.len() % TURBO3_BLOCK_SIZE, 0);
    assert_eq!(block_norms.len(), indices.len() / TURBO3_BLOCK_SIZE);
    let mut out = Vec::with_capacity(block_norms.len() * TURBO3_BLOCK_BYTES);
    for (block_idx, block) in indices.chunks(TURBO3_BLOCK_SIZE).enumerate() {
        out.extend_from_slice(
            &f16::from_f32(block_norms[block_idx])
                .to_bits()
                .to_le_bytes(),
        );
        let mut low = [0u8; TURBO3_BLOCK_SIZE / 4];
        let mut high = [0u8; TURBO3_BLOCK_SIZE / 8];
        for (j, &idx) in block.iter().enumerate() {
            low[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
            if (idx & 0x4) != 0 {
                high[j / 8] |= 1 << (j % 8);
            }
        }
        out.extend_from_slice(&low);
        out.extend_from_slice(&high);
    }
    out
}

fn turbo3_unpack_blocks(payload: &[u8]) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(payload.len() % TURBO3_BLOCK_BYTES, 0);
    let num_blocks = payload.len() / TURBO3_BLOCK_BYTES;
    let mut indices = Vec::with_capacity(num_blocks * TURBO3_BLOCK_SIZE);
    let mut block_norms = Vec::with_capacity(num_blocks);
    for block in payload.chunks_exact(TURBO3_BLOCK_BYTES) {
        let norm = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        block_norms.push(norm);
        let low = &block[2..10];
        let high = &block[10..14];
        for j in 0..TURBO3_BLOCK_SIZE {
            let low2 = (low[j / 4] >> ((j % 4) * 2)) & 0x3;
            let hi1 = (high[j / 8] >> (j % 8)) & 0x1;
            indices.push(low2 | (hi1 << 2));
        }
    }
    (indices, block_norms)
}

fn turbo3_inverse_reference(
    indices: &[u8],
    block_norms: &[f32],
    rotate: bool,
    plain_group_size: usize,
) -> Vec<f32> {
    assert_eq!(indices.len() % TURBO3_BLOCK_SIZE, 0);
    if rotate && indices.len() % TURBO3_GROUP_SIZE == 0 {
        turbo3_inverse_groups_reference(indices, block_norms)
    } else {
        turbo3_inverse_blocks_reference_with_centroids(
            indices,
            block_norms,
            &turbo3_plain_centroids(plain_group_size),
        )
    }
}

fn turbo3_inverse_groups_reference(indices: &[u8], block_norms: &[f32]) -> Vec<f32> {
    assert_eq!(block_norms.len(), indices.len() / TURBO3_BLOCK_SIZE);
    let mut out = Vec::with_capacity(indices.len());
    for (group_idx, group) in indices.chunks(TURBO3_GROUP_SIZE).enumerate() {
        let mut rotated = Vec::with_capacity(TURBO3_GROUP_SIZE);
        for (local_idx, &idx) in group.iter().enumerate() {
            let block_norm = block_norms[group_idx * (TURBO3_GROUP_SIZE / TURBO3_BLOCK_SIZE)
                + (local_idx / TURBO3_BLOCK_SIZE)];
            rotated.push(TURBO3_CENTROIDS[idx as usize] * block_norm);
        }
        turbo_rotate_inverse_reference(&mut rotated);
        out.extend(rotated);
    }
    out
}

fn turbo3_inverse_blocks_reference(indices: &[u8], block_norms: &[f32]) -> Vec<f32> {
    turbo3_inverse_blocks_reference_with_centroids(indices, block_norms, &TURBO3_CENTROIDS)
}

fn turbo3_inverse_blocks_reference_with_centroids(
    indices: &[u8],
    block_norms: &[f32],
    centroids: &[f32],
) -> Vec<f32> {
    assert_eq!(block_norms.len(), indices.len() / TURBO3_BLOCK_SIZE);
    let mut out = Vec::with_capacity(indices.len());
    for (block_idx, block) in indices.chunks(TURBO3_BLOCK_SIZE).enumerate() {
        let norm = block_norms[block_idx];
        for &idx in block {
            out.push(centroids[idx as usize] * norm);
        }
    }
    out
}

fn turbo3_rotated_reference(indices: &[u8], block_norms: &[f32]) -> Vec<f32> {
    assert_eq!(indices.len() % TURBO3_BLOCK_SIZE, 0);
    assert_eq!(block_norms.len(), indices.len() / TURBO3_BLOCK_SIZE);
    if indices.len() % TURBO3_GROUP_SIZE != 0 {
        return turbo3_inverse_blocks_reference(indices, block_norms);
    }
    let mut out = Vec::with_capacity(indices.len());
    for (group_idx, group) in indices.chunks(TURBO3_GROUP_SIZE).enumerate() {
        for (local_idx, &idx) in group.iter().enumerate() {
            let block_norm = block_norms[group_idx * (TURBO3_GROUP_SIZE / TURBO3_BLOCK_SIZE)
                + (local_idx / TURBO3_BLOCK_SIZE)];
            out.push(TURBO3_CENTROIDS[idx as usize] * block_norm);
        }
    }
    out
}

fn turbo3_find_nearest_index(value: f32) -> u8 {
    if value < TURBO3_MIDPOINTS[0] {
        0
    } else if value < TURBO3_MIDPOINTS[1] {
        1
    } else if value < TURBO3_MIDPOINTS[2] {
        2
    } else if value < TURBO3_MIDPOINTS[3] {
        3
    } else if value < TURBO3_MIDPOINTS[4] {
        4
    } else if value < TURBO3_MIDPOINTS[5] {
        5
    } else if value < TURBO3_MIDPOINTS[6] {
        6
    } else {
        7
    }
}

fn turbo_rotate_forward_reference(values: &mut [f32]) {
    assert_eq!(values.len(), TURBO3_GROUP_SIZE);
    for (value, sign) in values.iter_mut().zip(TURBO_WHT_SIGNS1.iter()) {
        *value *= *sign;
    }
    fwht_turbo3(values);
    for (value, sign) in values.iter_mut().zip(TURBO_WHT_SIGNS2.iter()) {
        *value *= *sign;
    }
}

fn turbo_rotate_inverse_reference(values: &mut [f32]) {
    assert_eq!(values.len(), TURBO3_GROUP_SIZE);
    for (value, sign) in values.iter_mut().zip(TURBO_WHT_SIGNS2.iter()) {
        *value *= *sign;
    }
    fwht_turbo3(values);
    for (value, sign) in values.iter_mut().zip(TURBO_WHT_SIGNS1.iter()) {
        *value *= *sign;
    }
}

pub fn turbo3_rotate_forward_in_place(values: &mut [f32]) {
    assert_eq!(values.len() % TURBO3_GROUP_SIZE, 0);
    for chunk in values.chunks_mut(TURBO3_GROUP_SIZE) {
        turbo_rotate_forward_reference(chunk);
    }
}

pub fn turbo3_rotate_inverse_in_place(values: &mut [f32]) {
    assert_eq!(values.len() % TURBO3_GROUP_SIZE, 0);
    for chunk in values.chunks_mut(TURBO3_GROUP_SIZE) {
        turbo_rotate_inverse_reference(chunk);
    }
}

fn fwht_turbo3(values: &mut [f32]) {
    assert_eq!(values.len(), TURBO3_GROUP_SIZE);
    let mut h = 1usize;
    while h < TURBO3_GROUP_SIZE {
        let step = h * 2;
        let mut i = 0usize;
        while i < TURBO3_GROUP_SIZE {
            for j in i..(i + h) {
                let a = values[j];
                let b = values[j + h];
                values[j] = a + b;
                values[j + h] = a - b;
            }
            i += step;
        }
        h *= 2;
    }
    let scale = 1.0 / (TURBO3_GROUP_SIZE as f32).sqrt();
    for value in values.iter_mut() {
        *value *= scale;
    }
}

fn matvec(matrix: &[f32], vector: &[f32], dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|row| {
            let start = row * dim;
            dot(&matrix[start..start + dim], vector)
        })
        .collect()
}

fn matvec_transposed(matrix: &[f32], vector: &[f32], dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|col| {
            (0..dim)
                .map(|row| matrix[row * dim + col] * vector[row])
                .sum::<f32>()
        })
        .collect()
}

fn nearest_centroid_index(value: f32, centroids: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_dist = f32::MAX;
    for (idx, &centroid) in centroids.iter().enumerate() {
        let dist = (value - centroid).abs();
        if dist < best_dist {
            best = idx;
            best_dist = dist;
        }
    }
    best
}

fn random_orthogonal(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(seed);
    let mut cols: Vec<Vec<f32>> = Vec::with_capacity(dim);
    for _ in 0..dim {
        let mut col = (0..dim)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect::<Vec<f32>>();
        for basis in &cols {
            let proj = dot(&col, basis);
            for (x, b) in col.iter_mut().zip(basis.iter()) {
                *x -= proj * *b;
            }
        }
        let norm = l2_norm(&col).max(1e-8);
        for x in &mut col {
            *x /= norm;
        }
        cols.push(col);
    }
    let mut out = vec![0.0f32; dim * dim];
    for row in 0..dim {
        for col in 0..dim {
            out[row * dim + col] = cols[col][row];
        }
    }
    out
}

fn gaussian_lloyd_max_centroids(dim: usize, bits: usize) -> Vec<f32> {
    let n_levels = 1usize << bits;
    let sigma = 1.0f32 / (dim as f32).sqrt();
    let lo = -3.5 * sigma;
    let hi = 3.5 * sigma;
    let mut centroids = (0..n_levels)
        .map(|i| lo + (hi - lo) * (i as f32 + 0.5) / n_levels as f32)
        .collect::<Vec<_>>();

    for _ in 0..64 {
        let mut boundaries = Vec::with_capacity(n_levels + 1);
        boundaries.push(lo * 3.0);
        for pair in centroids.windows(2) {
            boundaries.push((pair[0] + pair[1]) * 0.5);
        }
        boundaries.push(hi * 3.0);
        let mut updated = Vec::with_capacity(n_levels);
        for idx in 0..n_levels {
            let a = boundaries[idx];
            let b = boundaries[idx + 1];
            let numerator = integrate_gaussian_moment(a, b, sigma);
            let denominator = integrate_gaussian_mass(a, b, sigma).max(1e-9);
            updated.push(numerator / denominator);
        }
        let delta = updated
            .iter()
            .zip(centroids.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        centroids = updated;
        if delta < 1e-7 {
            break;
        }
    }
    centroids
}

fn integrate_gaussian_mass(a: f32, b: f32, sigma: f32) -> f32 {
    gaussian_cdf(b / sigma) - gaussian_cdf(a / sigma)
}

fn integrate_gaussian_moment(a: f32, b: f32, sigma: f32) -> f32 {
    let scale = sigma / (2.0 * PI).sqrt();
    let ea = (-0.5 * (a / sigma) * (a / sigma)).exp();
    let eb = (-0.5 * (b / sigma) * (b / sigma)).exp();
    scale * (ea - eb)
}

fn gaussian_cdf(x: f32) -> f32 {
    0.5 * (1.0 + statrs::function::erf::erf((x / SQRT_2) as f64) as f32)
}

fn pack_bits(indices: &[u8], bits_per_value: usize) -> Vec<u8> {
    let mut out = vec![0u8; (indices.len() * bits_per_value).div_ceil(8)];
    let mut bit_cursor = 0usize;
    for &index in indices {
        let value = index as usize;
        for bit in 0..bits_per_value {
            if ((value >> bit) & 1) != 0 {
                out[bit_cursor / 8] |= 1 << (bit_cursor % 8);
            }
            bit_cursor += 1;
        }
    }
    out
}

fn unpack_bits(payload: &[u8], values: usize, bits_per_value: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(values);
    let mut bit_cursor = 0usize;
    for _ in 0..values {
        let mut value = 0u8;
        for bit in 0..bits_per_value {
            if ((payload[bit_cursor / 8] >> (bit_cursor % 8)) & 1) != 0 {
                value |= 1 << bit;
            }
            bit_cursor += 1;
        }
        out.push(value);
    }
    out
}

fn pack_signs(signs: &[i8]) -> Vec<u8> {
    let mut out = vec![0u8; signs.len().div_ceil(8)];
    for (idx, sign) in signs.iter().enumerate() {
        if *sign >= 0 {
            out[idx / 8] |= 1 << (idx % 8);
        }
    }
    out
}

fn unpack_signs(payload: &[u8], values: usize) -> Vec<i8> {
    (0..values)
        .map(|idx| {
            if ((payload[idx / 8] >> (idx % 8)) & 1) != 0 {
                1
            } else {
                -1
            }
        })
        .collect()
}

/// Compute attention score directly from a packed turbo3 key payload
/// without decompressing to a full f32 vector.
///
/// `query_rot` must already be WHT-forward-rotated and have length == head_dim.
/// `packed_payload` is the packed key bytes for one head (14 bytes per 32-element block).
///
/// Returns `sum_j(query_rot[j] * TURBO3_CENTROIDS[idx[j]] * block_norm[j/32])`.
pub fn turbo3_score_from_packed(query_rot: &[f32], packed_payload: &[u8]) -> f32 {
    assert_eq!(packed_payload.len() % TURBO3_BLOCK_BYTES, 0);
    let num_blocks = packed_payload.len() / TURBO3_BLOCK_BYTES;
    assert_eq!(query_rot.len(), num_blocks * TURBO3_BLOCK_SIZE);

    let mut score = 0.0f32;
    for (block_idx, block) in packed_payload.chunks_exact(TURBO3_BLOCK_BYTES).enumerate() {
        let norm = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let low = &block[2..10];
        let high = &block[10..14];
        let base = block_idx * TURBO3_BLOCK_SIZE;
        let mut block_score = 0.0f32;
        for j in 0..TURBO3_BLOCK_SIZE {
            let low2 = (low[j / 4] >> ((j % 4) * 2)) & 0x3;
            let hi1 = (high[j / 8] >> (j % 8)) & 0x1;
            let idx = (low2 | (hi1 << 2)) as usize;
            block_score += query_rot[base + j] * TURBO3_CENTROIDS[idx];
        }
        score += block_score * norm;
    }
    score
}

/// Compute attention scores for multiple heads from a flat (all-heads) packed payload.
///
/// `query_rot` has length `num_heads * head_dim`, already WHT-forward-rotated.
/// `packed_payload` has length `num_heads * packed_key_bytes_per_head`.
/// Returns one score per head.
pub fn turbo3_score_from_packed_flat(
    query_rot: &[f32],
    packed_payload: &[u8],
    head_dim: usize,
) -> Vec<f32> {
    let packed_head_bytes = TurboQuantBits::Three.packed_key_bytes_for_dim(head_dim);
    let num_heads = packed_payload.len() / packed_head_bytes;
    assert_eq!(packed_payload.len(), num_heads * packed_head_bytes);
    assert_eq!(query_rot.len(), num_heads * head_dim);

    (0..num_heads)
        .map(|h| {
            let q_slice = &query_rot[h * head_dim..(h + 1) * head_dim];
            let p_slice = &packed_payload[h * packed_head_bytes..(h + 1) * packed_head_bytes];
            turbo3_score_from_packed(q_slice, p_slice)
        })
        .collect()
}

/// Decompress a turbo3 value from packed bytes directly to f32 values
/// in rotated space (no inverse rotation applied).
///
/// This is equivalent to `turbo3_rotated_reference(unpack(payload))` but
/// avoids intermediate allocations.
pub fn turbo3_value_from_packed_rotated(packed_payload: &[u8]) -> Vec<f32> {
    assert_eq!(packed_payload.len() % TURBO3_BLOCK_BYTES, 0);
    let num_blocks = packed_payload.len() / TURBO3_BLOCK_BYTES;
    let mut out = Vec::with_capacity(num_blocks * TURBO3_BLOCK_SIZE);

    for block in packed_payload.chunks_exact(TURBO3_BLOCK_BYTES) {
        let norm = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let low = &block[2..10];
        let high = &block[10..14];
        for j in 0..TURBO3_BLOCK_SIZE {
            let low2 = (low[j / 4] >> ((j % 4) * 2)) & 0x3;
            let hi1 = (high[j / 8] >> (j % 8)) & 0x1;
            let idx = (low2 | (hi1 << 2)) as usize;
            out.push(TURBO3_CENTROIDS[idx] * norm);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn turboquant_storage_is_smaller_than_fp16_kv() {
        let tq = TurboQuantArtifacts::new(128, TurboQuantBits::Three, 42);
        let compressed = tq.estimated_key_storage_bytes_per_token()
            + tq.estimated_value_storage_bytes_per_token();
        let fp16 = 128 * 2 * 2;
        assert!(compressed < fp16);
    }

    #[test]
    fn turboquant_key_ip_estimate_is_finite() {
        let tq = TurboQuantArtifacts::new(64, TurboQuantBits::Three, 7);
        let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(9);
        let x = (0..64)
            .map(|_| rng.random::<f32>() - 0.5)
            .collect::<Vec<_>>();
        let q = (0..64)
            .map(|_| rng.random::<f32>() - 0.5)
            .collect::<Vec<_>>();
        let key = tq.compress_key(&x);
        let ip = tq.estimate_inner_product(&q, &key);
        assert!(ip.is_finite());
    }

    #[test]
    fn turboquant_value_roundtrip_is_finite() {
        let tq = TurboQuantArtifacts::new(64, TurboQuantBits::Four, 11);
        let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(13);
        let x = (0..64)
            .map(|_| rng.random::<f32>() - 0.5)
            .collect::<Vec<_>>();
        let value = tq.compress_value(&x);
        let restored = tq.decompress_value(&value);
        assert_eq!(restored.len(), 64);
        assert!(restored.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn turboquant_pack_roundtrip_preserves_shapes() {
        let tq = TurboQuantArtifacts::new(32, TurboQuantBits::Three, 19);
        let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(23);
        let x = (0..32)
            .map(|_| rng.random::<f32>() - 0.5)
            .collect::<Vec<_>>();
        let key = tq.compress_key(&x);
        let key_packed = tq.pack_key(&key);
        let key_unpacked = tq.unpack_key(&key_packed);
        assert_eq!(key_unpacked.mse_indices.len(), 32);
        assert!(key_unpacked.qjl_signs.is_empty());

        let value = tq.compress_value(&x);
        let value_packed = tq.pack_value(&value);
        let value_unpacked = tq.unpack_value(&value_packed);
        assert_eq!(value_unpacked.indices.len(), 32);
    }

    #[test]
    fn turboquant4_pack_roundtrip_preserves_qjl_state() {
        let tq = TurboQuantArtifacts::new(128, TurboQuantBits::Four, 29);
        let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(31);
        let x = (0..128)
            .map(|_| rng.random::<f32>() - 0.5)
            .collect::<Vec<_>>();
        let key = tq.compress_key(&x);
        let packed = tq.pack_key(&key);
        assert_eq!(packed.len(), tq.estimated_key_storage_bytes_per_token());
        let unpacked = tq.unpack_key(&packed);
        assert_eq!(unpacked.mse_indices.len(), 128);
        assert_eq!(unpacked.qjl_signs.len(), 128);
        assert!(unpacked.residual_norm.is_finite());
    }

    #[test]
    fn turbo3_rotated_reconstruction_inverts_to_plain_decode() {
        let tq = TurboQuantArtifacts::new(256, TurboQuantBits::Three, 37);
        let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(41);
        let x = (0..256)
            .map(|_| rng.random::<f32>() - 0.5)
            .collect::<Vec<_>>();

        let key = tq.compress_key(&x);
        let expected = tq.reconstruct_key_mse(&key);
        let mut rotated = tq.reconstruct_key_mse_rotated(&key);
        turbo3_rotate_inverse_in_place(&mut rotated);

        assert_eq!(rotated.len(), expected.len());
        for (lhs, rhs) in rotated.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1e-5, "{lhs} != {rhs}");
        }
    }

    #[test]
    fn turbo3_rotated_value_inverts_to_plain_decode() {
        let tq = TurboQuantArtifacts::new(256, TurboQuantBits::Three, 43);
        let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(47);
        let x = (0..256)
            .map(|_| rng.random::<f32>() - 0.5)
            .collect::<Vec<_>>();

        let value = tq.compress_value(&x);
        let expected = tq.decompress_value(&value);
        let mut rotated = tq.decompress_value_rotated(&value);
        turbo3_rotate_inverse_in_place(&mut rotated);

        assert_eq!(rotated.len(), expected.len());
        for (lhs, rhs) in rotated.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1e-5, "{lhs} != {rhs}");
        }
    }

    #[test]
    fn turbo3_rotation_is_reversible() {
        let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(53);
        let mut x = (0..256)
            .map(|_| rng.random::<f32>() - 0.5)
            .collect::<Vec<_>>();
        let original = x.clone();

        turbo3_rotate_forward_in_place(&mut x);
        turbo3_rotate_inverse_in_place(&mut x);

        for (lhs, rhs) in x.iter().zip(original.iter()) {
            assert!((lhs - rhs).abs() < 1e-5, "{lhs} != {rhs}");
        }
    }

    #[test]
    fn turbo3_supports_32_wide_head_groups() {
        assert!(TurboQuantBits::Three.supports_storage_head_dim(64));
        assert!(TurboQuantBits::Three.supports_storage_head_dim(128));
        assert!(TurboQuantBits::Three.supports_storage_head_dim(256));
        assert!(!TurboQuantBits::Three.supports_rotation_head_dim(64));
        assert!(TurboQuantBits::Three.supports_rotation_head_dim(128));
        assert!(TurboQuantBits::Three.supports_rotation_head_dim(256));
    }

    #[test]
    fn turbo3_dim128_rotated_inverts_to_plain_decode() {
        let tq = TurboQuantArtifacts::new(128, TurboQuantBits::Three, 59);
        let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(61);
        let x = (0..128)
            .map(|_| rng.random::<f32>() - 0.5)
            .collect::<Vec<_>>();

        let key = tq.compress_key(&x);
        let key_plain = tq.reconstruct_key_mse(&key);
        let mut key_rotated = tq.reconstruct_key_mse_rotated(&key);
        turbo3_rotate_inverse_in_place(&mut key_rotated);
        for (lhs, rhs) in key_plain.iter().zip(key_rotated.iter()) {
            assert!((lhs - rhs).abs() < 1e-5, "{lhs} != {rhs}");
        }

        let value = tq.compress_value(&x);
        let value_plain = tq.decompress_value(&value);
        let mut value_rotated = tq.decompress_value_rotated(&value);
        turbo3_rotate_inverse_in_place(&mut value_rotated);
        for (lhs, rhs) in value_plain.iter().zip(value_rotated.iter()) {
            assert!((lhs - rhs).abs() < 1e-5, "{lhs} != {rhs}");
        }
    }

    #[test]
    fn turbo3_flattened_token_payload_can_be_split_back_into_heads() {
        let num_heads = 8;
        let head_dim = 64;
        let flat_dim = num_heads * head_dim;
        let flat = TurboQuantArtifacts::new(flat_dim, TurboQuantBits::Three, 67);
        let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(73);
        let x = (0..flat_dim)
            .map(|_| rng.random::<f32>() - 0.5)
            .collect::<Vec<_>>();

        let key = flat.compress_key(&x);
        let packed = flat.pack_key(&key);
        let reconstructed = flat.reconstruct_key_mse(&key);
        let head_bytes = TurboQuantBits::Three.packed_key_bytes_for_dim(head_dim);

        assert_eq!(packed.len(), num_heads * head_bytes);
        let reassembled = (0..num_heads)
            .flat_map(|head_idx| {
                packed[head_idx * head_bytes..(head_idx + 1) * head_bytes]
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(reassembled, packed);
        let unpacked = flat.unpack_key(&reassembled);
        let unpacked_reconstructed = flat.reconstruct_key_mse(&unpacked);
        for head_idx in 0..num_heads {
            let expected = &reconstructed[head_idx * head_dim..(head_idx + 1) * head_dim];
            let observed = &unpacked_reconstructed[head_idx * head_dim..(head_idx + 1) * head_dim];
            for (lhs, rhs) in observed.iter().zip(expected.iter()) {
                assert!((lhs - rhs).abs() < 5e-4, "{lhs} != {rhs}");
            }
        }
    }

    #[test]
    fn turbo3_score_from_packed_matches_reconstruct_dot() {
        let dim = 128;
        let tq = TurboQuantArtifacts::new(dim, TurboQuantBits::Three, 79);
        let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(83);

        let key_input = (0..dim)
            .map(|_| rng.random::<f32>() - 0.5)
            .collect::<Vec<_>>();
        let query = (0..dim)
            .map(|_| rng.random::<f32>() - 0.5)
            .collect::<Vec<_>>();

        // Compress and pack key
        let key = tq.compress_key(&key_input);
        let packed = tq.pack_key(&key);

        // Method A: existing path — reconstruct rotated key, dot with rotated query
        let key_rotated = tq.reconstruct_key_mse_rotated(&key);
        let mut query_rot = query.clone();
        turbo3_rotate_forward_in_place(&mut query_rot);
        let expected_score = dot(&query_rot, &key_rotated);

        // Method B: direct score from packed bytes
        let direct_score = turbo3_score_from_packed(&query_rot, &packed);

        assert!(
            (expected_score - direct_score).abs() < 1e-4,
            "expected {expected_score}, got {direct_score}"
        );
    }

    #[test]
    fn turbo3_score_from_packed_flat_matches_per_head() {
        let num_heads = 4;
        let head_dim = 128;
        let flat_dim = num_heads * head_dim;
        let tq = TurboQuantArtifacts::new(flat_dim, TurboQuantBits::Three, 89);
        let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(97);

        let key_input = (0..flat_dim)
            .map(|_| rng.random::<f32>() - 0.5)
            .collect::<Vec<_>>();
        let mut query_rot = (0..flat_dim)
            .map(|_| rng.random::<f32>() - 0.5)
            .collect::<Vec<_>>();
        turbo3_rotate_forward_in_place(&mut query_rot);

        let key = tq.compress_key(&key_input);
        let packed = tq.pack_key(&key);

        // Flat scoring
        let flat_scores = turbo3_score_from_packed_flat(&query_rot, &packed, head_dim);
        assert_eq!(flat_scores.len(), num_heads);

        // Per-head scoring
        let head_bytes = TurboQuantBits::Three.packed_key_bytes_for_dim(head_dim);
        for h in 0..num_heads {
            let q_slice = &query_rot[h * head_dim..(h + 1) * head_dim];
            let p_slice = &packed[h * head_bytes..(h + 1) * head_bytes];
            let per_head_score = turbo3_score_from_packed(q_slice, p_slice);
            assert!(
                (flat_scores[h] - per_head_score).abs() < 1e-6,
                "head {h}: flat={}, per_head={}",
                flat_scores[h],
                per_head_score
            );
        }
    }

    #[test]
    fn turbo3_value_from_packed_rotated_matches_existing() {
        let dim = 128;
        let tq = TurboQuantArtifacts::new(dim, TurboQuantBits::Three, 101);
        let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(103);

        let val_input = (0..dim)
            .map(|_| rng.random::<f32>() - 0.5)
            .collect::<Vec<_>>();

        let value = tq.compress_value(&val_input);
        let packed = tq.pack_value(&value);

        // Existing path
        let expected = tq.decompress_value_rotated(&value);

        // New direct path
        let direct = turbo3_value_from_packed_rotated(&packed);

        assert_eq!(expected.len(), direct.len());
        for (lhs, rhs) in expected.iter().zip(direct.iter()) {
            assert!((lhs - rhs).abs() < 1e-6, "{lhs} != {rhs}");
        }
    }
}
