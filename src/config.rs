use anyhow::{Context, Result};
use hf_hub::api::sync::Api;
use serde::Deserialize;
use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

// Auto Downloader & config
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum EosIdField {
    Single(usize),
    Multiple(Vec<usize>),
}

#[derive(Debug, Deserialize, Clone)]
pub struct DynamicConfig {
    #[serde(alias = "hidden_size")]
    pub dim: usize,
    #[serde(alias = "intermediate_size")]
    pub hidden_dim: usize,
    #[serde(alias = "num_hidden_layers")]
    pub n_layers: usize,
    #[serde(alias = "num_attention_heads")]
    pub n_heads: usize,
    #[serde(alias = "num_key_value_heads")]
    pub n_kv_heads: Option<usize>,
    pub vocab_size: usize,
    #[serde(alias = "max_position_embeddings", default = "default_seq_len")]
    pub seq_len: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub eos_token_id: Option<EosIdField>,
}

fn default_seq_len() -> usize {
    2048
}
fn default_rope_theta() -> f32 {
    10000.0
}

impl DynamicConfig {
    pub fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }
    pub fn kv_heads(&self) -> usize {
        self.n_kv_heads.unwrap_or(self.n_heads)
    }
    pub fn kv_dim(&self) -> usize {
        (self.dim * self.kv_heads()) / self.n_heads
    }
}

pub struct ModelFiles {
    pub config: DynamicConfig,
    pub weights_path: PathBuf,
    pub tokenizer_path: PathBuf,
}

#[derive(Debug, Clone, Copy)]
pub enum ModelFamily {
    Llama3,
    ChatML,  // SmolLM2, Qwen
    Mistral, // Mistral, Llama 2
    Gemma,   // Gemma 1 & 2
    Raw,     // Base models
}

pub struct ChatEngine {
    pub family: ModelFamily,
    pub stop_tokens: HashSet<usize>,
}

impl ChatEngine {
    pub fn new(config: &DynamicConfig, tokenizer: &Tokenizer) -> Self {
        let family = if tokenizer.token_to_id("<|im_start|>").is_some() {
            ModelFamily::ChatML
        } else if tokenizer.token_to_id("<|start_header_id|>").is_some() {
            ModelFamily::Llama3
        } else if tokenizer.token_to_id("<start_of_turn>").is_some() {
            ModelFamily::Gemma
        } else if tokenizer.token_to_id("[INST]").is_some() {
            ModelFamily::Mistral
        } else {
            ModelFamily::Raw
        };

        let mut stop_tokens = HashSet::new();

        if let Some(eos) = &config.eos_token_id {
            match eos {
                EosIdField::Single(id) => {
                    stop_tokens.insert(*id);
                }
                EosIdField::Multiple(ids) => {
                    for &id in ids {
                        stop_tokens.insert(id);
                    }
                }
            }
        }

        let common_stop_strings = [
            "<|eot_id|>",
            "<|end_of_text|>",
            "<|im_end|>",
            "<|endoftext|>",
            "</s>",
            "<end_of_turn>",
        ];
        for s in common_stop_strings {
            if let Some(id) = tokenizer.token_to_id(s) {
                stop_tokens.insert(id as usize);
            }
        }

        println!("- Model Architecture Family detected: {:?}", family);
        println!("- Dynamic Stop Tokens: {:?}", stop_tokens);

        Self {
            family,
            stop_tokens,
        }
    }

    pub fn format_prompt(&self, user_msg: &str, is_first_turn: bool) -> String {
        match self.family {
            ModelFamily::ChatML => {
                if is_first_turn {
                    format!(
                        "<|im_start|>system\nYou are a helpful and concise AI assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                        user_msg
                    )
                } else {
                    format!(
                        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                        user_msg
                    )
                }
            }
            ModelFamily::Llama3 => {
                if is_first_turn {
                    format!(
                        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful and concise AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                        user_msg
                    )
                } else {
                    format!(
                        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                        user_msg
                    )
                }
            }
            ModelFamily::Mistral => {
                format!("[INST] {} [/INST]", user_msg)
            }
            ModelFamily::Gemma => {
                format!(
                    "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
                    user_msg
                )
            }
            ModelFamily::Raw => {
                format!("User: {}\nAssistant: ", user_msg)
            }
        }
    }
}

pub struct ModelDownloader;

impl ModelDownloader {
    pub fn resolve_model(target: &str) -> Result<ModelFiles> {
        let path = Path::new(target);

        if path.exists() && path.is_dir() {
            println!("- Loading model from local directory: {:?}", path);
            let config_path = path.join("config.json");
            let weights_path = path.join("model.safetensors");
            let tokenizer_path = path.join("tokenizer.json");

            let config_file = File::open(&config_path)?;
            let config: DynamicConfig = serde_json::from_reader(config_file)?;

            return Ok(ModelFiles {
                config,
                weights_path,
                tokenizer_path,
            });
        }

        println!("- Connecting to Hugging Face Hub for: '{}'...", target);
        let api = Api::new().context("Failed to initialize Hugging Face Hub API")?;
        let repo = api.model(target.to_string());

        println!("  ├─ Checking & fetching config.json...");
        let config_path = repo.get("config.json")?;
        let config_file = File::open(&config_path)?;
        let config: DynamicConfig = serde_json::from_reader(config_file)?;

        println!("  ├─ Checking & fetching tokenizer.json...");
        let tokenizer_path = repo.get("tokenizer.json")?;

        println!("  └─ Checking & downloading model.safetensors...");
        let weights_path = repo.get("model.safetensors")?;

        println!("Model files resolved and cached successfully!");
        Ok(ModelFiles {
            config,
            weights_path,
            tokenizer_path,
        })
    }
}
