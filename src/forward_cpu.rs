use crate::config::DynamicConfig;
use crate::f16::F16;
use crate::model::WeightsOffsets;
use crate::nam::*;
use crate::state::CpuState;
use enki::Space;
use rayon::prelude::*;

pub fn forward_cpu_step(
    token: usize,
    pos: usize,
    c: &DynamicConfig,
    w: &WeightsOffsets,
    weights: &[F16],
    cpu: &mut CpuState,
) -> Vec<f32> {
    let dim = c.dim;
    let hidden_dim = c.hidden_dim;
    let head_size = c.head_size();
    let kv_dim = c.kv_dim();
    let n_layers = c.n_layers;
    let n_heads = c.n_heads;
    let n_kv_heads = c.kv_heads();
    let vocab_size = c.vocab_size;
    let chunk_dim = (dim / 8).max(1);
    let chunk_hidden = (hidden_dim / 8).max(1);

    cpu.x.par_iter_mut().enumerate().for_each(|(tid, x_elem)| {
        kernel_embed(
            &Space::cpu_x(tid, dim),
            weights,
            x_elem,
            token as u32,
            dim as u32,
        );
    });

    let partial_ptr = cpu.partial_scratch.as_mut_ptr() as usize;
    let partial_len = cpu.partial_scratch.len();
    let key_cache_ptr = cpu.key_cache.as_mut_ptr() as usize;
    let key_cache_len = cpu.key_cache.len();
    let val_cache_ptr = cpu.value_cache.as_mut_ptr() as usize;
    let val_cache_len = cpu.value_cache.len();
    let attn_scores_ptr = cpu.attn_scores.as_mut_ptr() as usize;
    let attn_scores_len = cpu.attn_scores.len();
    let xb_ptr = cpu.xb.as_mut_ptr() as usize;
    let xb_len = cpu.xb.len();
    let q_ptr = cpu.q.as_mut_ptr() as usize;
    let q_len = cpu.q.len();
    let k_ptr = cpu.k.as_mut_ptr() as usize;
    let k_len = cpu.k.len();

    for l in 0..n_layers {
        let layer_mat_offset = l * dim * dim;
        let layer_kv_offset = l * dim * kv_dim;
        let layer_mlp_offset = l * dim * hidden_dim;
        let layer_down_offset = l * hidden_dim * dim;

        let rms_att_off = w.rms_att_weight + l * dim;
        cpu.xb
            .par_iter_mut()
            .enumerate()
            .for_each(|(tid, out_elem)| {
                kernel_rmsnorm(
                    &Space::cpu_x(tid, dim),
                    &cpu.x,
                    weights,
                    out_elem,
                    dim as u32,
                    rms_att_off as u32,
                );
            });

        (0..8usize).into_par_iter().for_each(|chunk_idx| {
            let partial =
                unsafe { std::slice::from_raw_parts_mut(partial_ptr as *mut f32, partial_len) };
            for r in 0..dim {
                kernel_gemv_splitk_pass1(
                    &Space::cpu_xy(r, chunk_idx, dim, 8),
                    weights,
                    &cpu.xb,
                    partial,
                    dim as u32,
                    dim as u32,
                    chunk_dim as u32,
                    (w.wq + layer_mat_offset) as u32,
                );
            }
        });
        cpu.q.par_iter_mut().enumerate().for_each(|(r, q_elem)| {
            kernel_gemv_splitk_pass2(
                &Space::cpu_x(r, dim),
                &cpu.partial_scratch,
                q_elem,
                dim as u32,
            );
        });

        (0..8usize).into_par_iter().for_each(|chunk_idx| {
            let partial =
                unsafe { std::slice::from_raw_parts_mut(partial_ptr as *mut f32, partial_len) };
            for r in 0..kv_dim {
                kernel_gemv_splitk_pass1(
                    &Space::cpu_xy(r, chunk_idx, kv_dim, 8),
                    weights,
                    &cpu.xb,
                    partial,
                    kv_dim as u32,
                    dim as u32,
                    chunk_dim as u32,
                    (w.wk + layer_kv_offset) as u32,
                );
            }
        });
        cpu.k.par_iter_mut().enumerate().for_each(|(r, k_elem)| {
            kernel_gemv_splitk_pass2(
                &Space::cpu_x(r, kv_dim),
                &cpu.partial_scratch,
                k_elem,
                kv_dim as u32,
            );
        });

        (0..8usize).into_par_iter().for_each(|chunk_idx| {
            let partial =
                unsafe { std::slice::from_raw_parts_mut(partial_ptr as *mut f32, partial_len) };
            for r in 0..kv_dim {
                kernel_gemv_splitk_pass1(
                    &Space::cpu_xy(r, chunk_idx, kv_dim, 8),
                    weights,
                    &cpu.xb,
                    partial,
                    kv_dim as u32,
                    dim as u32,
                    chunk_dim as u32,
                    (w.wv + layer_kv_offset) as u32,
                );
            }
        });
        cpu.v.par_iter_mut().enumerate().for_each(|(r, v_elem)| {
            kernel_gemv_splitk_pass2(
                &Space::cpu_x(r, kv_dim),
                &cpu.partial_scratch,
                v_elem,
                kv_dim as u32,
            );
        });

        let total_rope = n_heads * (head_size / 2);
        (0..total_rope).into_par_iter().for_each(|tid| {
            let q_slice = unsafe { std::slice::from_raw_parts_mut(q_ptr as *mut f32, q_len) };
            let k_slice = unsafe { std::slice::from_raw_parts_mut(k_ptr as *mut f32, k_len) };
            kernel_rope(
                &Space::cpu_x(tid, total_rope),
                q_slice,
                k_slice,
                pos as u32,
                head_size as u32,
                n_kv_heads as u32,
                c.rope_theta,
            );
        });

        let cache_offset = l * c.seq_len * kv_dim + pos * kv_dim;
        (0..kv_dim).into_par_iter().for_each(|tid| {
            let kc =
                unsafe { std::slice::from_raw_parts_mut(key_cache_ptr as *mut f32, key_cache_len) };
            let vc =
                unsafe { std::slice::from_raw_parts_mut(val_cache_ptr as *mut f32, val_cache_len) };
            kernel_cache_kv(
                &Space::cpu_x(tid, kv_dim),
                &cpu.k,
                &cpu.v,
                kc,
                vc,
                cache_offset as u32,
            );
        });

        let l_att_off = l * c.seq_len * kv_dim;
        (0..n_heads).into_par_iter().for_each(|h| {
            let s_slice = unsafe {
                std::slice::from_raw_parts_mut(attn_scores_ptr as *mut f32, attn_scores_len)
            };
            for th in 0..64 {
                kernel_attention_scores_strided(
                    &Space::cpu_xy(th, h, 64, n_heads),
                    &cpu.q,
                    &cpu.key_cache,
                    s_slice,
                    pos as u32,
                    head_size as u32,
                    l_att_off as u32,
                    kv_dim as u32,
                    n_heads as u32,
                    n_kv_heads as u32,
                    c.seq_len as u32,
                );
            }
        });

        (0..n_heads).into_par_iter().for_each(|h| {
            let s_slice = unsafe {
                std::slice::from_raw_parts_mut(attn_scores_ptr as *mut f32, attn_scores_len)
            };
            kernel_attention_softmax(
                &Space::cpu_x(h, n_heads),
                s_slice,
                pos as u32,
                c.seq_len as u32,
            );
        });

        (0..n_heads).into_par_iter().for_each(|h| {
            let xb_slice = unsafe { std::slice::from_raw_parts_mut(xb_ptr as *mut f32, xb_len) };
            for i in 0..head_size {
                kernel_attention_values(
                    &Space::cpu_xy(i, h, head_size, n_heads),
                    &cpu.attn_scores,
                    &cpu.value_cache,
                    xb_slice,
                    pos as u32,
                    head_size as u32,
                    l_att_off as u32,
                    kv_dim as u32,
                    n_heads as u32,
                    n_kv_heads as u32,
                    c.seq_len as u32,
                );
            }
        });

        (0..8usize).into_par_iter().for_each(|chunk_idx| {
            let partial =
                unsafe { std::slice::from_raw_parts_mut(partial_ptr as *mut f32, partial_len) };
            for r in 0..dim {
                kernel_gemv_splitk_pass1(
                    &Space::cpu_xy(r, chunk_idx, dim, 8),
                    weights,
                    &cpu.xb,
                    partial,
                    dim as u32,
                    dim as u32,
                    chunk_dim as u32,
                    (w.wo + layer_mat_offset) as u32,
                );
            }
        });
        cpu.x.par_iter_mut().enumerate().for_each(|(r, x_elem)| {
            kernel_gemv_splitk_pass2_accumulate(
                &Space::cpu_x(r, dim),
                &cpu.partial_scratch,
                x_elem,
                dim as u32,
            );
        });

        let rms_ffn_off = w.rms_ffn_weight + l * dim;
        cpu.xb
            .par_iter_mut()
            .enumerate()
            .for_each(|(tid, out_elem)| {
                kernel_rmsnorm(
                    &Space::cpu_x(tid, dim),
                    &cpu.x,
                    weights,
                    out_elem,
                    dim as u32,
                    rms_ffn_off as u32,
                );
            });

        (0..8usize).into_par_iter().for_each(|chunk_idx| {
            let partial =
                unsafe { std::slice::from_raw_parts_mut(partial_ptr as *mut f32, partial_len) };
            for r in 0..hidden_dim {
                kernel_gemv_splitk_pass1(
                    &Space::cpu_xy(r, chunk_idx, hidden_dim, 8),
                    weights,
                    &cpu.xb,
                    partial,
                    hidden_dim as u32,
                    dim as u32,
                    chunk_dim as u32,
                    (w.w1 + layer_mlp_offset) as u32,
                );
            }
        });
        cpu.hb.par_iter_mut().enumerate().for_each(|(r, hb_elem)| {
            kernel_gemv_splitk_pass2(
                &Space::cpu_x(r, hidden_dim),
                &cpu.partial_scratch,
                hb_elem,
                hidden_dim as u32,
            );
        });

        (0..8usize).into_par_iter().for_each(|chunk_idx| {
            let partial =
                unsafe { std::slice::from_raw_parts_mut(partial_ptr as *mut f32, partial_len) };
            for r in 0..hidden_dim {
                kernel_gemv_splitk_pass1(
                    &Space::cpu_xy(r, chunk_idx, hidden_dim, 8),
                    weights,
                    &cpu.xb,
                    partial,
                    hidden_dim as u32,
                    dim as u32,
                    chunk_dim as u32,
                    (w.w3 + layer_mlp_offset) as u32,
                );
            }
        });
        cpu.hb2
            .par_iter_mut()
            .enumerate()
            .for_each(|(r, hb2_elem)| {
                kernel_gemv_splitk_pass2(
                    &Space::cpu_x(r, hidden_dim),
                    &cpu.partial_scratch,
                    hb2_elem,
                    hidden_dim as u32,
                );
            });

        cpu.hb
            .par_iter_mut()
            .zip(&cpu.hb2)
            .enumerate()
            .for_each(|(i, (hb_elem, hb2_elem))| {
                kernel_swiglu(&Space::cpu_x(i, hidden_dim), hb_elem, hb2_elem);
            });

        (0..8usize).into_par_iter().for_each(|chunk_idx| {
            let partial =
                unsafe { std::slice::from_raw_parts_mut(partial_ptr as *mut f32, partial_len) };
            for r in 0..dim {
                kernel_gemv_splitk_pass1(
                    &Space::cpu_xy(r, chunk_idx, dim, 8),
                    weights,
                    &cpu.hb,
                    partial,
                    dim as u32,
                    hidden_dim as u32,
                    chunk_hidden as u32,
                    (w.w2 + layer_down_offset) as u32,
                );
            }
        });
        cpu.x.par_iter_mut().enumerate().for_each(|(r, x_elem)| {
            kernel_gemv_splitk_pass2_accumulate(
                &Space::cpu_x(r, dim),
                &cpu.partial_scratch,
                x_elem,
                dim as u32,
            );
        });
    }

    cpu.xb
        .par_iter_mut()
        .enumerate()
        .for_each(|(tid, out_elem)| {
            kernel_rmsnorm(
                &Space::cpu_x(tid, dim),
                &cpu.x,
                weights,
                out_elem,
                dim as u32,
                w.rms_final_weight as u32,
            );
        });

    (0..8usize).into_par_iter().for_each(|chunk_idx| {
        let partial =
            unsafe { std::slice::from_raw_parts_mut(partial_ptr as *mut f32, partial_len) };
        for r in 0..vocab_size {
            kernel_gemv_splitk_pass1(
                &Space::cpu_xy(r, chunk_idx, vocab_size, 8),
                weights,
                &cpu.xb,
                partial,
                vocab_size as u32,
                dim as u32,
                chunk_dim as u32,
                w.wcls as u32,
            );
        }
    });
    cpu.logits
        .par_iter_mut()
        .enumerate()
        .for_each(|(r, logit_elem)| {
            kernel_gemv_splitk_pass2(
                &Space::cpu_x(r, vocab_size),
                &cpu.partial_scratch,
                logit_elem,
                vocab_size as u32,
            );
        });

    cpu.logits.clone()
}
