use anyhow::{bail, Result};

pub type AudioCodeSequence = Vec<Vec<u32>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioCodeChunkPlan {
    pub requests: Vec<AudioCodeSequence>,
    pub chunks: Vec<AudioCodeSequence>,
    pub chunk_lengths: Vec<usize>,
    pub request_chunk_indices: Vec<Vec<usize>>,
}

pub fn parse_batched_audio_input(
    input_ids: &[u32],
    num_codebooks: usize,
) -> Result<(Vec<AudioCodeSequence>, Vec<usize>)> {
    if num_codebooks == 0 {
        bail!("num_codebooks must be greater than zero");
    }
    let mut offset = 0usize;
    let mut requests = Vec::new();
    let mut ctx_frames_all = Vec::new();
    while offset < input_ids.len() {
        if input_ids.len() - offset < 2 {
            bail!("batched audio input is truncated before ctx/header fields");
        }
        let ctx_frames = input_ids[offset] as usize;
        let context_length = input_ids[offset + 1] as usize;
        offset += 2;

        let frame_count = ctx_frames + context_length;
        let flat_len = frame_count
            .checked_mul(num_codebooks)
            .ok_or_else(|| anyhow::anyhow!("audio token length overflow"))?;
        if input_ids.len() - offset < flat_len {
            bail!(
                "batched audio input is truncated: need {} token ids for {} frames with {} codebooks, got {}",
                flat_len,
                frame_count,
                num_codebooks,
                input_ids.len() - offset
            );
        }
        let mut request = Vec::with_capacity(frame_count);
        for frame in input_ids[offset..offset + flat_len].chunks_exact(num_codebooks) {
            request.push(frame.to_vec());
        }
        offset += flat_len;
        requests.push(request);
        ctx_frames_all.push(ctx_frames);
    }
    Ok((requests, ctx_frames_all))
}

pub fn apply_ctx_frames_cutting(
    batch_audio_arrays: &[Vec<f32>],
    all_ctx_frames: &[usize],
    downsample_factor: usize,
) -> Result<Vec<Vec<f32>>> {
    if batch_audio_arrays.len() != all_ctx_frames.len() {
        bail!(
            "audio/context length mismatch: {} audio arrays vs {} ctx frame entries",
            batch_audio_arrays.len(),
            all_ctx_frames.len()
        );
    }
    let mut result = Vec::with_capacity(batch_audio_arrays.len());
    for (audio_array, ctx_frames) in batch_audio_arrays.iter().zip(all_ctx_frames) {
        let cut = downsample_factor
            .saturating_mul(*ctx_frames)
            .min(audio_array.len());
        result.push(audio_array[cut..].to_vec());
    }
    Ok(result)
}

fn truncate_end_of_audio_and_unshift(codes: &AudioCodeSequence) -> Result<AudioCodeSequence> {
    let cutting_point = codes
        .iter()
        .position(|frame| frame.first().copied() == Some(1))
        .unwrap_or(codes.len());
    let mut processed = Vec::with_capacity(cutting_point);
    for frame in &codes[..cutting_point] {
        let mut shifted = Vec::with_capacity(frame.len());
        for token in frame {
            let shifted_token = token.checked_sub(2).ok_or_else(|| {
                anyhow::anyhow!(
                    "audio code token {token} cannot be shifted by special-token offset"
                )
            })?;
            shifted.push(shifted_token);
        }
        processed.push(shifted);
    }
    Ok(processed)
}

pub fn prepare_decode_chunks(
    raw_requests: &[AudioCodeSequence],
    chunk_size: usize,
) -> Result<AudioCodeChunkPlan> {
    if chunk_size == 0 {
        bail!("chunk_size must be greater than zero");
    }
    let mut requests = Vec::with_capacity(raw_requests.len());
    let mut chunks = Vec::new();
    let mut chunk_lengths = Vec::new();
    let mut request_chunk_indices = Vec::with_capacity(raw_requests.len());

    for raw_request in raw_requests {
        let processed = truncate_end_of_audio_and_unshift(raw_request)?;
        let mut indices = Vec::new();
        for chunk in processed.chunks(chunk_size) {
            indices.push(chunks.len());
            chunk_lengths.push(chunk.len());
            chunks.push(chunk.to_vec());
        }
        requests.push(processed);
        request_chunk_indices.push(indices);
    }

    Ok(AudioCodeChunkPlan {
        requests,
        chunks,
        chunk_lengths,
        request_chunk_indices,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        apply_ctx_frames_cutting, parse_batched_audio_input, prepare_decode_chunks,
        AudioCodeChunkPlan,
    };

    #[test]
    fn parses_batched_audio_input_layout() {
        let (requests, ctx_frames) = parse_batched_audio_input(
            &[
                1, 2, 10, 11, 12, 13, 14, 15, //
                0, 1, 20, 21,
            ],
            2,
        )
        .unwrap();
        assert_eq!(ctx_frames, vec![1, 0]);
        assert_eq!(
            requests,
            vec![
                vec![vec![10, 11], vec![12, 13], vec![14, 15]],
                vec![vec![20, 21]]
            ]
        );
    }

    #[test]
    fn rejects_truncated_batched_audio_input() {
        let err = parse_batched_audio_input(&[1, 2, 10, 11], 2)
            .unwrap_err()
            .to_string();
        assert!(err.contains("truncated"));
    }

    #[test]
    fn cuts_context_samples() {
        let cut =
            apply_ctx_frames_cutting(&[vec![0.0, 1.0, 2.0, 3.0], vec![10.0, 11.0]], &[1, 0], 2)
                .unwrap();
        assert_eq!(cut, vec![vec![2.0, 3.0], vec![10.0, 11.0]]);
    }

    #[test]
    fn prepares_decode_chunks_like_reference() {
        let plan = prepare_decode_chunks(
            &[
                vec![
                    vec![2, 5, 6],
                    vec![3, 7, 8],
                    vec![1, 9, 10],
                    vec![4, 11, 12],
                ],
                vec![vec![1, 5, 6]],
                vec![vec![10, 11, 12], vec![13, 14, 15]],
            ],
            1,
        )
        .unwrap();
        assert_eq!(
            plan,
            AudioCodeChunkPlan {
                requests: vec![
                    vec![vec![0, 3, 4], vec![1, 5, 6]],
                    vec![],
                    vec![vec![8, 9, 10], vec![11, 12, 13]],
                ],
                chunks: vec![
                    vec![vec![0, 3, 4]],
                    vec![vec![1, 5, 6]],
                    vec![vec![8, 9, 10]],
                    vec![vec![11, 12, 13]],
                ],
                chunk_lengths: vec![1, 1, 1, 1],
                request_chunk_indices: vec![vec![0, 1], vec![], vec![2, 3]],
            }
        );
    }
}
