#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle_core::{safetensors, Device, Tensor, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::parler_tts::{Config, Model};
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;
use std::fs::File;
use anyhow::Error as E;

pub struct ParlerInferenceModel {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
}

impl ParlerInferenceModel {
    pub fn load(
        model_name: &str,
        revision: &str,
    ) -> anyhow::Result<Self> {

        // Set device to GPU if available.
        let device = Device::new_cuda(0)?;

        // Use the Hugging Face Hub API to load the model files.
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            model_name.to_string(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        ));

        // Load the model, config, and tokenizer files.
        println!("Loading model...");
        let model_file = repo.get("model.safetensors")?;
        let config_file = repo.get("config.json")?;
        let tokenizer_file = repo.get("tokenizer.json")?;

        // Initialize the tokenizer.
        let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;

        let config: Config = serde_json::from_reader(File::open(config_file)?)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };

        let model = Model::new(&config, vb)?;

        Ok(Self {
            model,
            tokenizer,
            device
        })
    }

    // pub fn infer_sentence_embedding(&self, sentence: &str) -> anyhow::Result<Tensor> {
    //     let tokens = self
    //         .tokenizer
    //         .encode(sentence, true)
    //         .map_err(anyhow::Error::msg)?;
    //     let token_ids = Tensor::new(tokens.get_ids(), &self.device)?.unsqueeze(0)?;
    //     let token_type_ids = token_ids.zeros_like()?;
    //     let start = std::time::Instant::now();
    //     let embeddings = self.model.forward(&token_ids, &token_type_ids)?;
    //     println!("time taken for forward: {:?}", start.elapsed());
    //     println!("embeddings: {:?}", embeddings);
    //     let embeddings = Self::apply_max_pooling(&embeddings)?;
    //     println!("embeddings after pooling: {:?}", embeddings);
    //     let embeddings = Self::l2_normalize(&embeddings)?;
    //     Ok(embeddings)
    // }

}