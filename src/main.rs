use axum::{
    extract::{Json, State},
    response::IntoResponse,
    routing::post,
    Router,
    http::StatusCode
};
use serde::Deserialize;
// use std::net::SocketAddr;
use tokio::task;
// use tokio::sync::oneshot;
// use tokio::io::AsyncWriteExt;
// use std::fs::File;
// use anyhow::Error as E;
// use clap::{Parser, ValueEnum};

// use candle_core::{DType, Device, IndexOp, Tensor};
// use candle_nn::VarBuilder;
// use candle_transformers::models::parler_tts::{Config, Model};
// use tokenizers::Tokenizer;
use std::sync::Arc;

mod parler;
// use candle_transformers::models::parler_tts::PlKVCache;

#[derive(Deserialize)]
struct ReqPayload {
    text: String,
    prompt: String,
}



#[tokio::main]
async fn main() -> anyhow::Result<()>{
    // Initialize tracing (for debugging/logging).
    tracing_subscriber::fmt::init();

    let model_id = "parler-tts/parler-tts-mini-v1";
    let revision = "main";
    let parler_model = parler::ParlerInferenceModel::load(model_id, revision)?;

    // Load the model once before starting the server.
    let shared_state = Arc::new(parler_model);

    // Build our application with a POST route for generating TTS.
    let app = Router::new().route("/generate", post(run_inference)).with_state(shared_state);

    // Run the app with hyper, listening on port 3000.
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    axum::serve(listener, app).await.unwrap();

    Ok(())
}


async fn run_inference (
    State(model_ctx): State<Arc<parler::ParlerInferenceModel>>, 
    Json(payload): Json<ReqPayload>,
) -> impl IntoResponse {
    let parler_model = Arc::clone(&model_ctx);
    let text = payload.text;
    let prompt = payload.prompt;

    // Spawn a blocking task for CPU-intensive work
    let result = task::spawn_blocking(move || {
        parler_model.run_inference(&text, &prompt)
    }).await.unwrap();  // Unwrap the JoinError

    match result {
        Ok(audio_data) => (
            StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, "audio/wav")],
            audio_data,
        ),
        Err(e) => {
            eprintln!("Error during inference: {:?}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                [(axum::http::header::CONTENT_TYPE, "text/plain")],
                format!("Failed to generate audio: {}", e).into(),
            )
        },
    }
}

