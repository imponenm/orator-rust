use axum::{
    extract::{Json, Path, State},
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

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::parler_tts::{Config, Model};
use tokenizers::Tokenizer;
use std::sync::{Arc, Mutex};

#[derive(Deserialize)]
struct GenerateRequest {
    text: String,
    prompt: String,
}

#[derive(Serialize)]
struct GenerateResponse {
    message: String,
}

struct AppState {
    model: Model,
    config: Config,  // Store the config here
    tokenizer: Tokenizer,
    device: Device,
}


#[tokio::main]
async fn main() {
    // Initialize tracing (for debugging/logging).
    tracing_subscriber::fmt::init();

    // Load the model once before starting the server.
    // let model = load_model().unwrap();
    // let shared_model = Arc::new(Mutex::new(model));
    let app_state = load_model().unwrap();
    let shared_state = Arc::new(Mutex::new(app_state));

    // Build our application with a POST route for generating TTS.
    let app = Router::new().route("/generate", post(generate_tts)).with_state(shared_state);

    // Run the app with hyper, listening on port 3000.
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

fn load_model() -> anyhow::Result<AppState> {
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
    println!("Loading model...");
    let model_file = repo.get("model.safetensors")?;
    let config_file = repo.get("config.json")?;
    let tokenizer_file = repo.get("tokenizer.json")?;

    // Initialize the tokenizer.
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;

    // Create the device (use GPU if available).
    let device = candle_examples::device(false)?; // false means using CPU

    // Load the model from the safetensor files.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
    let config: Config = serde_json::from_reader(File::open(config_file)?)?;
    let model = Model::new(&config, vb)?;

    // Return both the model and its configuration.
    Ok(AppState { model, config, tokenizer, device })
}


// TTS generation handler.
async fn generate_tts(
    State(app_state): State<Arc<Mutex<AppState>>>, // Access shared state.
    Json(payload): Json<GenerateRequest>,
) -> impl IntoResponse {
    // Clone the state to be used in the task.
    let state_clone = Arc::clone(&app_state);

    // Create a new channel for communicating between the async task and response.
    let (tx, rx) = oneshot::channel();

    // Spawn a blocking task to run inference.
    task::spawn_blocking(move || {
        let state = state_clone.lock().unwrap();
        let result = run_inference(&state, payload.text, payload.prompt);

        // Notify the main thread of the result.
        let _ = tx.send(
            match result {
                Ok(_) => "TTS generation succeeded".to_string(),
                Err(e) => format!("TTS generation failed: {}", e),
            }
        );
    });

    // Wait for the result from the spawned task.
    let result = rx.await.unwrap_or_else(|_| "Unknown error".to_string());

    Json(GenerateResponse { message: result })
}


fn run_inference(state: &AppState, text: String, prompt: String) -> anyhow::Result<()> {
    let tokenizer = &state.tokenizer;
    let device = &state.device;
    let mut model = &mut state.model;  // Borrow the model as mutable here.

    // Tokenize the text and prompt.
    let description_tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let description_tensor = Tensor::new(description_tokens, device)?.unsqueeze(0)?;

    let prompt_tokens = tokenizer
        .encode(text, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let prompt_tensor = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;

    // Set up the logits processor for generation.
    let lp = candle_transformers::generation::LogitsProcessor::new(0, Some(0.0), None);

    // Run the model to generate audio codes.
    println!("Generating...");
    let codes = model.generate(&prompt_tensor, &description_tensor, lp, 512)?;  // Mutably use model here
    let codes = codes.to_dtype(DType::I64)?;

    // Decode the generated audio codes into PCM audio.
    let codes = codes.unsqueeze(0)?;
    let pcm = model
        .audio_encoder
        .decode_codes(&codes.to_device(device)?)?;

    // Normalize the audio for proper loudness.
    let pcm = pcm.i((0, 0))?;
    let pcm = candle_examples::audio::normalize_loudness(&pcm, 24_000, true)?;
    let pcm = pcm.to_vec1::<f32>()?;

    // Write the audio as a WAV file, using the sampling rate from the config.
    let mut output = File::create("out.wav")?;
    candle_examples::wav::write_pcm_as_wav(&mut output, &pcm, state.config.audio_encoder.sampling_rate)?;

    Ok(())
}

