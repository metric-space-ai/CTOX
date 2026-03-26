use candle_core::{Result, Tensor};

pub fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        x.unsqueeze(2)?
            .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
            .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}
