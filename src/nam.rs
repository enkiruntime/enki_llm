use crate::f16::F16;
use crate::math::*;
use enki::{nam, Space};

// Nams (Kernels)
#[nam]
pub fn kernel_embed(space: &Space, table: &[F16], x: &mut f32, token: u32, dim: u32) {
    if !space.in_bounds_x() {
        return;
    }
    let tid = space.x;
    let offset = (token as usize) * (dim as usize);
    *x = table[offset + tid].to_f32();
}

#[nam]
pub fn kernel_rmsnorm(
    space: &Space,
    x: &[f32],
    weight: &[F16],
    out: &mut f32,
    dim: u32,
    weight_offset: u32,
) {
    if !space.in_bounds_x() {
        return;
    }
    let tid = space.x;
    let d = dim as usize;

    let mut sum_sq = 0.0f32;
    for i in 0..d {
        let v = x[i];
        sum_sq += v * v;
    }
    let scale = 1.0 / (sum_sq / (dim as f32) + 1e-5).sqrt();
    let w_idx = weight_offset as usize + tid;
    *out = weight[w_idx] * (x[tid] * scale);
}

#[nam]
pub fn kernel_gemv_splitk_pass1(
    space: &Space,
    weights_t: &[F16],
    input: &[f32],
    partial_out: &mut [f32],
    rows: u32,
    _cols: u32,
    chunk_size: u32,
    weight_offset: u32,
) {
    if !space.in_bounds_xy() {
        return;
    }
    let r = space.x;
    let chunk_idx = space.y;
    let m = rows as usize;
    let c_sz = chunk_size as usize;
    let c_start = chunk_idx * c_sz;
    let c_end = c_start + c_sz;
    let w_off = weight_offset as usize;

    let mut sum = 0.0f32;
    let mut c = c_start;
    let mut ptr = w_off + c_start * m + r;

    while c < c_end {
        sum += dot4_strided_f16(weights_t, ptr, m, input, c);
        ptr += m * 4;
        c += 4;
    }

    partial_out[chunk_idx * m + r] = sum;
}

#[nam]
pub fn kernel_gemv_splitk_pass2(space: &Space, partial_in: &[f32], output: &mut f32, rows: u32) {
    if !space.in_bounds_x() {
        return;
    }
    let r = space.x;
    let m = rows as usize;

    let sum0 = partial_in[r] + partial_in[m + r];
    let sum1 = partial_in[m * 2 + r] + partial_in[m * 3 + r];
    let sum2 = partial_in[m * 4 + r] + partial_in[m * 5 + r];
    let sum3 = partial_in[m * 6 + r] + partial_in[m * 7 + r];

    *output = (sum0 + sum1) + (sum2 + sum3);
}

#[nam]
pub fn kernel_gemv_splitk_pass2_accumulate(
    space: &Space,
    partial_in: &[f32],
    dest: &mut f32,
    rows: u32,
) {
    if !space.in_bounds_x() {
        return;
    }
    let r = space.x;
    let m = rows as usize;

    let sum0 = partial_in[r] + partial_in[m + r];
    let sum1 = partial_in[m * 2 + r] + partial_in[m * 3 + r];
    let sum2 = partial_in[m * 4 + r] + partial_in[m * 5 + r];
    let sum3 = partial_in[m * 6 + r] + partial_in[m * 7 + r];

    *dest += (sum0 + sum1) + (sum2 + sum3);
}

#[nam]
pub fn kernel_rope(
    space: &Space,
    q: &mut [f32],
    k: &mut [f32],
    pos: u32,
    head_size: u32,
    n_kv_heads: u32,
    rope_theta: f32,
) {
    if !space.in_bounds_x() {
        return;
    }
    let tid = space.x;
    let half = (head_size / 2) as usize;
    let h = tid / half;
    let i = tid % half;

    let freq = 1.0 / rope_theta.powf((i * 2) as f32 / head_size as f32);
    let val = (pos as f32) * freq;

    let base = h * (head_size as usize);
    let idx0 = base + i;
    let idx1 = base + i + half;

    let (zq_0, zq_1) = rope_rotate_2d(q[idx0], q[idx1], val);
    q[idx0] = zq_0;
    q[idx1] = zq_1;

    if h < (n_kv_heads as usize) {
        let (zk_0, zk_1) = rope_rotate_2d(k[idx0], k[idx1], val);
        k[idx0] = zk_0;
        k[idx1] = zk_1;
    }
}

#[nam]
pub fn kernel_cache_kv(
    space: &Space,
    k: &[f32],
    v: &[f32],
    key_cache: &mut [f32],
    value_cache: &mut [f32],
    cache_offset: u32,
) {
    if !space.in_bounds_x() {
        return;
    }
    let tid = space.x;
    let idx = (cache_offset as usize) + tid;
    key_cache[idx] = k[tid];
    value_cache[idx] = v[tid];
}

#[nam]
pub fn kernel_attention_scores_strided(
    space: &Space,
    q: &[f32],
    key_cache: &[f32],
    scores_out: &mut [f32],
    pos: u32,
    head_size: u32,
    layer_offset: u32,
    kv_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    max_seq_len: u32,
) {
    if !space.in_bounds_xy() {
        return;
    }
    let th = space.x;
    let h = space.y;
    let p = pos as usize;

    let hs = head_size as usize;
    let q_base = h * hs;
    let l_off = layer_offset as usize;
    let kd = kv_dim as usize;
    let max_seq = max_seq_len as usize;

    let kv_mul = (n_heads / n_kv_heads).max(1) as usize;
    let kv_h = h / kv_mul;
    let kv_head_offset = kv_h * hs;
    let scale = 1.0 / (head_size as f32).sqrt();

    let mut t = th;
    while t <= p {
        let k_base = l_off + t * kd + kv_head_offset;
        let mut score = 0.0f32;
        let mut i = 0;
        while i < hs {
            score += dot4_contiguous_f32(q, q_base + i, key_cache, k_base + i);
            i += 4;
        }
        scores_out[h * max_seq + t] = score * scale;
        t += 64;
    }
}

#[nam]
pub fn kernel_attention_softmax(space: &Space, scores: &mut [f32], pos: u32, max_seq_len: u32) {
    if !space.in_bounds_x() {
        return;
    }
    let h = space.x;
    let p = pos as usize;
    let max_seq = max_seq_len as usize;
    let base = h * max_seq;

    let mut max_val = f32::NEG_INFINITY;
    for t in 0..=p {
        let v = scores[base + t];
        if v > max_val {
            max_val = v;
        }
    }

    let mut sum_exp = 0.0f32;
    for t in 0..=p {
        let exp_v = (scores[base + t] - max_val).exp();
        scores[base + t] = exp_v;
        sum_exp += exp_v;
    }

    let inv_sum = 1.0 / sum_exp;
    for t in 0..=p {
        scores[base + t] *= inv_sum;
    }
}

#[nam]
pub fn kernel_attention_values(
    space: &Space,
    scores: &[f32],
    value_cache: &[f32],
    out_xb: &mut [f32],
    pos: u32,
    head_size: u32,
    layer_offset: u32,
    kv_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    max_seq_len: u32,
) {
    if !space.in_bounds_xy() {
        return;
    }
    let i = space.x;
    let h = space.y;
    let hs = head_size as usize;
    let q_base = h * hs;
    let l_off = layer_offset as usize;
    let kd = kv_dim as usize;
    let p = pos as usize;
    let max_seq = max_seq_len as usize;
    let score_base = h * max_seq;

    let kv_mul = (n_heads / n_kv_heads).max(1) as usize;
    let kv_h = h / kv_mul;
    let kv_head_offset = kv_h * hs;

    let mut acc = 0.0f32;
    for t in 0..=p {
        let s = scores[score_base + t];
        let v_idx = l_off + t * kd + kv_head_offset + i;
        acc += s * value_cache[v_idx];
    }

    out_xb[q_base + i] = acc;
}

#[nam]
pub fn kernel_swiglu(_space: &Space, hb: &mut f32, hb2: &f32) {
    *hb = swiglu(*hb, *hb2);
}
