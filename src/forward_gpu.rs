use crate::config::DynamicConfig;
use crate::model::WeightsOffsets;
use crate::nam::*;
use crate::state::GpuState;
use anyhow::Result;
use enki::*;

// Forward GPU Step
pub fn forward_gpu_step(
    token: usize,
    pos: usize,
    c: &DynamicConfig,
    w: &WeightsOffsets,
    gpu: &mut GpuState,
    enki: &Enki,
) -> Result<Vec<f32>> {
    let dim = c.dim as u32;
    let hidden_dim = c.hidden_dim as u32;
    let head_size = c.head_size() as u32;
    let kv_dim = c.kv_dim() as u32;
    let n_layers = c.n_layers;
    let n_heads = c.n_heads as u32;
    let n_kv_heads = c.kv_heads() as u32;
    let vocab_size = c.vocab_size as u32;

    let chunk_dim = (dim / 8).max(1);
    let chunk_hidden = (hidden_dim / 8).max(1);

    let space_dim = Space::gpu_x(dim as usize);
    let space_kv = Space::gpu_x(kv_dim as usize);
    let space_hidden = Space::gpu_x(hidden_dim as usize);
    let space_rope = Space::gpu_x((n_heads * (head_size / 2)) as usize);
    let space_vocab = Space::gpu_x(vocab_size as usize);

    let space_splitk_dim = Space::gpu_xy(dim as usize, 8).tile([32, 1]);
    let space_splitk_kv = Space::gpu_xy(kv_dim as usize, 8).tile([32, 1]);
    let space_splitk_hidden = Space::gpu_xy(hidden_dim as usize, 8).tile([32, 1]);
    let space_splitk_vocab = Space::gpu_xy(vocab_size as usize, 8).tile([32, 1]);

    let space_att_scores = Space::gpu_xy(64, n_heads as usize).tile([32, 1]);
    let space_att_softmax = Space::gpu_x(n_heads as usize);
    let space_att_values = Space::gpu_xy(head_size as usize, n_heads as usize).tile([32, 1]);

    let p_dim = GpuParam::new(dim);
    let p_hidden_dim = GpuParam::new(hidden_dim);
    let p_kv_dim = GpuParam::new(kv_dim);
    let p_vocab_size = GpuParam::new(vocab_size);
    let p_pos = GpuParam::new(pos as u32);
    let p_head_size = GpuParam::new(head_size);
    let p_n_heads = GpuParam::new(n_heads);
    let p_n_kv_heads = GpuParam::new(n_kv_heads);
    let p_max_seq_len = GpuParam::new(c.seq_len as u32);
    let p_rope_theta = GpuParam::new(c.rope_theta);
    let p_chunk_dim = GpuParam::new(chunk_dim);
    let p_chunk_hidden = GpuParam::new(chunk_hidden);

    enki.try_flow(|_| unsafe {
        kernel_embed.run_unchecked(
            &space_dim,
            &gpu.gpu_weights.as_slice(),
            &mut gpu.x,
            GpuParam::new(token as u32),
            p_dim,
        );

        for l in 0..n_layers {
            let layer_mat_offset = l * (dim as usize) * (dim as usize);
            let layer_kv_offset = l * (dim as usize) * (kv_dim as usize);
            let layer_mlp_offset = l * (dim as usize) * (hidden_dim as usize);
            let layer_down_offset = l * (hidden_dim as usize) * (dim as usize);

            let rms_att_off = w.rms_att_weight + l * (dim as usize);
            kernel_rmsnorm.run_unchecked(
                &space_dim,
                &gpu.x.as_slice(),
                &gpu.gpu_weights.as_slice(),
                &mut gpu.xb,
                p_dim,
                GpuParam::new(rms_att_off as u32),
            );

            kernel_gemv_splitk_pass1.run_unchecked(
                &space_splitk_dim,
                &gpu.gpu_weights.as_slice(),
                &gpu.xb.as_slice(),
                &mut gpu.partial_scratch.as_mut_slice(),
                p_dim,
                p_dim,
                p_chunk_dim,
                GpuParam::new((w.wq + layer_mat_offset) as u32),
            );
            kernel_gemv_splitk_pass2.run_unchecked(
                &space_dim,
                &gpu.partial_scratch.as_slice(),
                &mut gpu.q,
                p_dim,
            );

            kernel_gemv_splitk_pass1.run_unchecked(
                &space_splitk_kv,
                &gpu.gpu_weights.as_slice(),
                &gpu.xb.as_slice(),
                &mut gpu.partial_scratch.as_mut_slice(),
                p_kv_dim,
                p_dim,
                p_chunk_dim,
                GpuParam::new((w.wk + layer_kv_offset) as u32),
            );
            kernel_gemv_splitk_pass2.run_unchecked(
                &space_kv,
                &gpu.partial_scratch.as_slice(),
                &mut gpu.k,
                p_kv_dim,
            );

            kernel_gemv_splitk_pass1.run_unchecked(
                &space_splitk_kv,
                &gpu.gpu_weights.as_slice(),
                &gpu.xb.as_slice(),
                &mut gpu.partial_scratch.as_mut_slice(),
                p_kv_dim,
                p_dim,
                p_chunk_dim,
                GpuParam::new((w.wv + layer_kv_offset) as u32),
            );
            kernel_gemv_splitk_pass2.run_unchecked(
                &space_kv,
                &gpu.partial_scratch.as_slice(),
                &mut gpu.v,
                p_kv_dim,
            );

            kernel_rope.run_unchecked(
                &space_rope,
                &mut gpu.q.as_mut_slice(),
                &mut gpu.k.as_mut_slice(),
                p_pos,
                p_head_size,
                p_n_kv_heads,
                p_rope_theta,
            );

            let cache_offset = l * (c.seq_len) * (kv_dim as usize) + pos * (kv_dim as usize);
            kernel_cache_kv.run_unchecked(
                &space_kv,
                &gpu.k.as_slice(),
                &gpu.v.as_slice(),
                &mut gpu.key_cache.as_mut_slice(),
                &mut gpu.value_cache.as_mut_slice(),
                GpuParam::new(cache_offset as u32),
            );

            let l_att_off = l * (c.seq_len) * (kv_dim as usize);
            kernel_attention_scores_strided.run_unchecked(
                &space_att_scores,
                &gpu.q.as_slice(),
                &gpu.key_cache.as_slice(),
                &mut gpu.attn_scores.as_mut_slice(),
                p_pos,
                p_head_size,
                GpuParam::new(l_att_off as u32),
                p_kv_dim,
                p_n_heads,
                p_n_kv_heads,
                p_max_seq_len,
            );
            kernel_attention_softmax.run_unchecked(
                &space_att_softmax,
                &mut gpu.attn_scores.as_mut_slice(),
                p_pos,
                p_max_seq_len,
            );
            kernel_attention_values.run_unchecked(
                &space_att_values,
                &gpu.attn_scores.as_slice(),
                &gpu.value_cache.as_slice(),
                &mut gpu.xb.as_mut_slice(),
                p_pos,
                p_head_size,
                GpuParam::new(l_att_off as u32),
                p_kv_dim,
                p_n_heads,
                p_n_kv_heads,
                p_max_seq_len,
            );

            kernel_gemv_splitk_pass1.run_unchecked(
                &space_splitk_dim,
                &gpu.gpu_weights.as_slice(),
                &gpu.xb.as_slice(),
                &mut gpu.partial_scratch.as_mut_slice(),
                p_dim,
                p_dim,
                p_chunk_dim,
                GpuParam::new((w.wo + layer_mat_offset) as u32),
            );
            kernel_gemv_splitk_pass2_accumulate.run_unchecked(
                &space_dim,
                &gpu.partial_scratch.as_slice(),
                &mut gpu.x,
                p_dim,
            );

            let rms_ffn_off = w.rms_ffn_weight + l * (dim as usize);
            kernel_rmsnorm.run_unchecked(
                &space_dim,
                &gpu.x.as_slice(),
                &gpu.gpu_weights.as_slice(),
                &mut gpu.xb,
                p_dim,
                GpuParam::new(rms_ffn_off as u32),
            );

            kernel_gemv_splitk_pass1.run_unchecked(
                &space_splitk_hidden,
                &gpu.gpu_weights.as_slice(),
                &gpu.xb.as_slice(),
                &mut gpu.partial_scratch.as_mut_slice(),
                p_hidden_dim,
                p_dim,
                p_chunk_dim,
                GpuParam::new((w.w1 + layer_mlp_offset) as u32),
            );
            kernel_gemv_splitk_pass2.run_unchecked(
                &space_hidden,
                &gpu.partial_scratch.as_slice(),
                &mut gpu.hb,
                p_hidden_dim,
            );

            kernel_gemv_splitk_pass1.run_unchecked(
                &space_splitk_hidden,
                &gpu.gpu_weights.as_slice(),
                &gpu.xb.as_slice(),
                &mut gpu.partial_scratch.as_mut_slice(),
                p_hidden_dim,
                p_dim,
                p_chunk_dim,
                GpuParam::new((w.w3 + layer_mlp_offset) as u32),
            );
            kernel_gemv_splitk_pass2.run_unchecked(
                &space_hidden,
                &gpu.partial_scratch.as_slice(),
                &mut gpu.hb2,
                p_hidden_dim,
            );

            kernel_swiglu.run_unchecked(&space_hidden, &mut gpu.hb, &gpu.hb2);

            kernel_gemv_splitk_pass1.run_unchecked(
                &space_splitk_dim,
                &gpu.gpu_weights.as_slice(),
                &gpu.hb.as_slice(),
                &mut gpu.partial_scratch.as_mut_slice(),
                p_dim,
                p_hidden_dim,
                p_chunk_hidden,
                GpuParam::new((w.w2 + layer_down_offset) as u32),
            );
            kernel_gemv_splitk_pass2_accumulate.run_unchecked(
                &space_dim,
                &gpu.partial_scratch.as_slice(),
                &mut gpu.x,
                p_dim,
            );
        }

        kernel_rmsnorm.run_unchecked(
            &space_dim,
            &gpu.x.as_slice(),
            &gpu.gpu_weights.as_slice(),
            &mut gpu.xb,
            p_dim,
            GpuParam::new(w.rms_final_weight as u32),
        );

        kernel_gemv_splitk_pass1.run_unchecked(
            &space_splitk_vocab,
            &gpu.gpu_weights.as_slice(),
            &gpu.xb.as_slice(),
            &mut gpu.partial_scratch.as_mut_slice(),
            p_vocab_size,
            p_dim,
            p_chunk_dim,
            GpuParam::new(w.wcls as u32),
        );
        kernel_gemv_splitk_pass2.run_unchecked(
            &space_vocab,
            &gpu.partial_scratch.as_slice(),
            &mut gpu.logits,
            p_vocab_size,
        );
    })?;

    Ok(gpu.logits.to_vec())
}
