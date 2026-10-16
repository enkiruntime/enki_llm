use crate::config::DynamicConfig;
use crate::f16::F16;
use enki::{gpu_vec, GpuVec};

pub struct CpuState {
    pub x: Vec<f32>,
    pub xb: Vec<f32>,
    pub hb: Vec<f32>,
    pub hb2: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub logits: Vec<f32>,
    pub key_cache: Vec<f32>,
    pub value_cache: Vec<f32>,
    pub partial_scratch: Vec<f32>,
    pub attn_scores: Vec<f32>,
}

impl CpuState {
    pub fn new(c: &DynamicConfig) -> Self {
        let dim = c.dim;
        let hidden_dim = c.hidden_dim;
        let kv_dim = c.kv_dim();
        let vocab_size = c.vocab_size;
        let seq_len = c.seq_len;
        let kv_cache_len = c.n_layers * seq_len * kv_dim;
        let max_partial_len = (vocab_size * 8).max(hidden_dim * 8);

        Self {
            x: vec![0.0; dim],
            xb: vec![0.0; dim],
            hb: vec![0.0; hidden_dim],
            hb2: vec![0.0; hidden_dim],
            q: vec![0.0; dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            logits: vec![0.0; vocab_size],
            key_cache: vec![0.0; kv_cache_len],
            value_cache: vec![0.0; kv_cache_len],
            partial_scratch: vec![0.0; max_partial_len],
            attn_scores: vec![0.0; c.n_heads * seq_len],
        }
    }
}

pub struct GpuState {
    pub gpu_weights: GpuVec<F16>,
    pub x: GpuVec<f32>,
    pub xb: GpuVec<f32>,
    pub hb: GpuVec<f32>,
    pub hb2: GpuVec<f32>,
    pub q: GpuVec<f32>,
    pub k: GpuVec<f32>,
    pub v: GpuVec<f32>,
    pub logits: GpuVec<f32>,
    pub key_cache: GpuVec<f32>,
    pub value_cache: GpuVec<f32>,
    pub partial_scratch: GpuVec<f32>,
    pub attn_scores: GpuVec<f32>,
}

impl GpuState {
    pub fn new(c: &DynamicConfig, gpu_weights: GpuVec<F16>) -> Self {
        let dim = c.dim;
        let hidden_dim = c.hidden_dim;
        let kv_dim = c.kv_dim();
        let vocab_size = c.vocab_size;
        let seq_len = c.seq_len;
        let kv_cache_len = c.n_layers * seq_len * kv_dim;
        let max_partial_len = (vocab_size * 8).max(hidden_dim * 8);

        Self {
            gpu_weights,
            x: gpu_vec![0.0; dim],
            xb: gpu_vec![0.0; dim],
            hb: gpu_vec![0.0; hidden_dim],
            hb2: gpu_vec![0.0; hidden_dim],
            q: gpu_vec![0.0; dim],
            k: gpu_vec![0.0; kv_dim],
            v: gpu_vec![0.0; kv_dim],
            logits: gpu_vec![0.0; vocab_size],
            key_cache: gpu_vec![0.0; kv_cache_len],
            value_cache: gpu_vec![0.0; kv_cache_len],
            partial_scratch: gpu_vec![0.0; max_partial_len],
            attn_scores: gpu_vec![0.0; c.n_heads * seq_len],
        }
    }
}
