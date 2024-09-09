use axum::{
    extract::{Json, Path},
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tokio::task;
use tokio::sync::oneshot;
use tokio::io::AsyncWriteExt;
use std::fs::File;
use anyhow::Error as E;
use clap::{Parser, ValueEnum};

use candle_core::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::parler_tts::{Config, Model};
use tokenizers::Tokenizer;

#[derive(Deserialize)]
struct GenerateRequest {
    text: String,
    prompt: String,
}

#[derive(Serialize)]
struct GenerateResponse {
    message: String,
}

#[tokio::main]
async fn main() {
    // Initialize tracing (for debugging/logging).
    tracing_subscriber::fmt::init();

    // Build our application with a POST route for generating TTS.
    let app = Router::new().route("/generate", post(generate_tts));

    // Run the app with hyper, listening on port 3000.
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// TTS generation handler.
async fn generate_tts(Json(payload): Json<GenerateRequest>) -> impl IntoResponse {
    // Create a new channel for communicating between the async task and response.
    let (tx, rx) = oneshot::channel();

    // Spawn a blocking task to run inference.
    task::spawn_blocking(move || {
        // Run your TTS model with the provided `text` and `prompt`.
        match run_inference(payload.text, payload.prompt) {
            Ok(_) => {
                // Notify the main thread that the process is done.
                let _ = tx.send("TTS generation succeeded".to_string()); // Send an owned String
            }
            Err(e) => {
                let _ = tx.send(format!("TTS generation failed: {}", e)); // Send an owned String
            }
        }
    });

    // Wait for the result from the spawned task.
    let result = rx.await.unwrap_or_else(|_| "Unknown error".to_string());

    Json(GenerateResponse { message: result })
}


fn run_inference(text: String, prompt: String) -> anyhow::Result<()> {
    // Set up some parameters specific to the "mini-v1" model.
    let model_id = "parler-tts/parler-tts-mini-v1".to_string();
    let revision = "main".to_string();

    // Use the Hugging Face Hub API to load the model files.
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        model_id,
        hf_hub::RepoType::Model,
        revision,
    ));

    // Load the model, config, and tokenizer files.
    println!("Loading model and tokenizers...");
    let model_file = repo.get("model.safetensors")?;
    let config_file = repo.get("config.json")?;
    let tokenizer_file = repo.get("tokenizer.json")?;

    // Initialize the tokenizer.
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;

    // Create the device (use GPU if available).
    let device = candle_examples::device(false)?; // false means using GPU

    // Load the model from the safetensor files.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
    let config: Config = serde_json::from_reader(File::open(config_file)?)?;
    let mut model = Model::new(&config, vb)?;

    // Tokenize the text and prompt.
    let description_tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let description_tensor = Tensor::new(description_tokens, &device)?.unsqueeze(0)?;

    let prompt_tokens = tokenizer
        .encode(text, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let prompt_tensor = Tensor::new(prompt_tokens, &device)?.unsqueeze(0)?;

    // Set up the logits processor for generation.
    let lp = candle_transformers::generation::LogitsProcessor::new(0, Some(0.0), None);

    // Run the model to generate audio codes.
    println!("Generating...");
    let codes = model.generate(&prompt_tensor, &description_tensor, lp, 512)?;
    let codes = codes.to_dtype(DType::I64)?;

    // Decode the generated audio codes into PCM audio.
    let codes = codes.unsqueeze(0)?;
    let pcm = model
        .audio_encoder
        .decode_codes(&codes.to_device(&device)?)?;

    // Normalize the audio for proper loudness.
    let pcm = pcm.i((0, 0))?;
    let pcm = candle_examples::audio::normalize_loudness(&pcm, 24_000, true)?;
    let pcm = pcm.to_vec1::<f32>()?;

    // Write the audio as a WAV file.
    let mut output = File::create("out.wav")?;
    candle_examples::wav::write_pcm_as_wav(&mut output, &pcm, config.audio_encoder.sampling_rate)?;

    Ok(())
}