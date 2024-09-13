// #[cfg(feature = "mkl")]
// extern crate intel_mkl_src;

// #[cfg(feature = "accelerate")]
// extern crate accelerate_src;

use candle_core::{Device, Tensor, DType, IndexOp};
use candle_nn::VarBuilder;
use candle_transformers::models::parler_tts::{Config, Model};
use tokenizers::Tokenizer;
use std::fs::File;
use std::path::PathBuf;
use anyhow::Error as E;

use axum::body::Bytes;
use candle_transformers::models::parler_tts::PlKVCache;
use candle_transformers::models::t5::T5PlKvCache;

pub struct ParlerInferenceModel {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
    config: Config,
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
            device,
            config,
        })
    }

    pub fn run_inference(
        &self,
        text: &str,
        prompt: &str,
        cache: &mut PlKVCache
    ) -> anyhow::Result<Bytes> {
        // Tokenize the text and prompt.
        let description_tokens = self.tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let description_tensor = Tensor::new(description_tokens, &self.device)?.unsqueeze(0)?;
    
        let prompt_tokens = self.tokenizer
            .encode(text, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let prompt_tensor = Tensor::new(prompt_tokens, &self.device)?.unsqueeze(0)?;
    
        // Set up the logits processor for generation.
        let lp = candle_transformers::generation::LogitsProcessor::new(0, Some(0.0), None);
    
        // Run the model to generate audio codes.
        println!("Generating...");

        let mut cache = PlKVCache::new(self.model.num_decoder_layers());
        let mut t5_cache = T5PlKvCache::new();
        
        let codes = self.model.generate(&prompt_tensor, &description_tensor, lp, 512, &mut cache, &mut t5_cache)?;
        
        let codes = codes.to_dtype(DType::I64)?;
    
        // Decode the generated audio codes into PCM audio.
        let codes = codes.unsqueeze(0)?;
        let pcm = self.model
            .audio_encoder
            .decode_codes(&codes.to_device(&self.device)?)?;
    
        // Normalize the audio for proper loudness.
        let pcm = pcm.i((0, 0))?;
        let pcm = candle_examples::audio::normalize_loudness(&pcm, 24_000, true)?;
        let pcm = pcm.to_vec1::<f32>()?;
    
        // Write the audio as a WAV file, using the sampling rate from the config.
        // let mut output = File::create("out.wav")?;
        // candle_examples::wav::write_pcm_as_wav(&mut output, &pcm, self.config.audio_encoder.sampling_rate)?;

        // Write audio as wave in memory
        let mut buffer = Vec::new();
        candle_examples::wav::write_pcm_as_wav(&mut buffer, &pcm, self.config.audio_encoder.sampling_rate)?;
    
        Ok(Bytes::from(buffer))
    }

    pub fn num_decoder_layers(&self) -> usize {
        self.model.decoder.num_layers()
    }

}