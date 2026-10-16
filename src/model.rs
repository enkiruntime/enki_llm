use crate::config::DynamicConfig;
use crate::f16::F16;
use crate::state::GpuState;
use anyhow::Result;
use enki::GpuVec;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::fs::File;
use std::path::Path;

// Model
pub struct WeightsOffsets {
    pub rms_att_weight: usize,
    pub wq: usize,
    pub wk: usize,
    pub wv: usize,
    pub wo: usize,
    pub rms_ffn_weight: usize,
    pub w1: usize,
    pub w2: usize,
    pub w3: usize,
    pub rms_final_weight: usize,
    pub wcls: usize,
}

pub struct Model {
    pub config: DynamicConfig,
    pub offsets: WeightsOffsets,
}

impl Model {
    pub fn load_to_gpu(config: &DynamicConfig, weights_path: &Path) -> Result<(Self, GpuState)> {
        let file = File::open(weights_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let tensors = SafeTensors::deserialize(&mmap)?;

        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        let kv_dim = config.kv_dim();
        let vocab_size = config.vocab_size;
        let layers = config.n_layers;
        let seq_len = config.seq_len.min(2048);

        let mut offset = 0;
        let token_embedding_off = offset;
        offset += vocab_size * dim;
        let rms_att_off = offset;
        offset += layers * dim;
        let wq_off = offset;
        offset += layers * dim * dim;
        let wk_off = offset;
        offset += layers * dim * kv_dim;
        let wv_off = offset;
        offset += layers * dim * kv_dim;
        let wo_off = offset;
        offset += layers * dim * dim;
        let rms_ffn_off = offset;
        offset += layers * dim;
        let w1_off = offset;
        offset += layers * dim * hidden_dim;
        let w2_off = offset;
        offset += layers * hidden_dim * dim;
        let w3_off = offset;
        offset += layers * dim * hidden_dim;
        let rms_final_off = offset;
        offset += dim;

        let wcls_off = offset;
        offset += vocab_size * dim;

        let total_weights_len = offset;
        println!(
            "- Allocating {:.1} MB in GPU VRAM (F16)...",
            (total_weights_len * 2) as f64 / (1024.0 * 1024.0)
        );

        let mut gpu_weights = GpuVec::with_capacity(total_weights_len);
        gpu_weights.resize(total_weights_len, F16::ZERO);

        let get_tensor_f16 = |name: &str| -> Result<Vec<F16>> {
            let t = tensors.tensor(name)?;
            let bytes = t.data();

            match t.dtype() {
                safetensors::Dtype::F16 => {
                    let u16_slice: &[u16] = bytemuck::cast_slice(bytes);
                    Ok(u16_slice.iter().map(|&bits| F16(bits)).collect())
                }
                safetensors::Dtype::BF16 => {
                    let u16_slice: &[u16] = bytemuck::cast_slice(bytes);
                    Ok(u16_slice
                        .iter()
                        .map(|&bits| {
                            let f = f32::from_bits((bits as u32) << 16);
                            F16::from_f32(f)
                        })
                        .collect())
                }
                safetensors::Dtype::F32 => {
                    let f32_slice: &[f32] = bytemuck::cast_slice(bytes);
                    Ok(f32_slice.iter().map(|&f| F16::from_f32(f)).collect())
                }
                other => anyhow::bail!("Unsupported dtype: {:?}", other),
            }
        };

        println!("- Streaming Embeddings & Global Norms to VRAM...");
        let embed = get_tensor_f16("model.embed_tokens.weight")?;
        gpu_weights.copy_from_slice_at(token_embedding_off, &embed);

        let final_norm = get_tensor_f16("model.norm.weight")?;
        gpu_weights.copy_from_slice_at(rms_final_off, &final_norm);

        println!("- Transposing & Streaming lm_head (wcls) to VRAM (100% Coalesced)...");
        let lm_head_raw = if config.tie_word_embeddings {
            embed.clone()
        } else {
            get_tensor_f16("lm_head.weight")?
        };

        let mut trans_wcls = vec![F16::ZERO; vocab_size * dim];
        for r in 0..vocab_size {
            for c in 0..dim {
                trans_wcls[c * vocab_size + r] = lm_head_raw[r * dim + c];
            }
        }
        gpu_weights.copy_from_slice_at(wcls_off, &trans_wcls);
        drop(trans_wcls);

        println!(
            "- Streaming & Transposing {} Layers chunk-by-chunk...",
            layers
        );
        let mut trans_buf = Vec::new();
        for l in 0..layers {
            let att_norm = get_tensor_f16(&format!("model.layers.{}.input_layernorm.weight", l))?;
            gpu_weights.copy_from_slice_at(rms_att_off + l * dim, &att_norm);

            let ffn_norm = get_tensor_f16(&format!(
                "model.layers.{}.post_attention_layernorm.weight",
                l
            ))?;
            gpu_weights.copy_from_slice_at(rms_ffn_off + l * dim, &ffn_norm);

            let mut upload_t = |name: &str, off: usize, rows: usize, cols: usize| -> Result<()> {
                let raw = get_tensor_f16(&format!("model.layers.{}.{}", l, name))?;
                trans_buf.resize(rows * cols, F16::ZERO);
                for r in 0..rows {
                    for c in 0..cols {
                        trans_buf[c * rows + r] = raw[r * cols + c];
                    }
                }
                gpu_weights.copy_from_slice_at(off, &trans_buf);
                Ok(())
            };

            upload_t("self_attn.q_proj.weight", wq_off + l * dim * dim, dim, dim)?;
            upload_t(
                "self_attn.k_proj.weight",
                wk_off + l * dim * kv_dim,
                kv_dim,
                dim,
            )?;
            upload_t(
                "self_attn.v_proj.weight",
                wv_off + l * dim * kv_dim,
                kv_dim,
                dim,
            )?;
            upload_t("self_attn.o_proj.weight", wo_off + l * dim * dim, dim, dim)?;

            upload_t(
                "mlp.gate_proj.weight",
                w1_off + l * dim * hidden_dim,
                hidden_dim,
                dim,
            )?;
            upload_t(
                "mlp.up_proj.weight",
                w3_off + l * dim * hidden_dim,
                hidden_dim,
                dim,
            )?;
            upload_t(
                "mlp.down_proj.weight",
                w2_off + l * hidden_dim * dim,
                dim,
                hidden_dim,
            )?;
        }

        println!(
            "- Pre-allocating KV-Cache & Scratchpads in VRAM (Context: {} tokens)...",
            seq_len
        );

        let gpu_state = GpuState::new(&config, gpu_weights);

        let offsets = WeightsOffsets {
            rms_att_weight: rms_att_off,
            wq: wq_off,
            wk: wk_off,
            wv: wv_off,
            wo: wo_off,
            rms_ffn_weight: rms_ffn_off,
            w1: w1_off,
            w2: w2_off,
            w3: w3_off,
            rms_final_weight: rms_final_off,
            wcls: wcls_off,
        };

        let mut effective_config = config.clone();
        effective_config.seq_len = seq_len;

        println!("Model successfully loaded and ready on GPU!");
        Ok((
            Model {
                config: effective_config,
                offsets,
            },
            gpu_state,
        ))
    }
}
