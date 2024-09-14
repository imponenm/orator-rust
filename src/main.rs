use axum::{
    extract::{Json, Path, State},
    response::IntoResponse,
    routing::{get, post},
    Router,
    http::StatusCode
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

mod parler;
use candle_transformers::models::parler_tts::PlKVCache;

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
    // TODO: Need to edit parler_tts to use separate KV caches, so that I don't have to make this mutable
    let parler_model = &*model_ctx;

    let text = payload.text;
    let prompt = payload.prompt;

    // let mut cache = PlKVCache::new(parler_model.num_decoder_layers());

    match parler_model.run_inference(&text, &prompt) {
        Ok(audio_data) => (
            StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, "audio/wav")],
            audio_data,
        ),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            [(axum::http::header::CONTENT_TYPE, "text/plain")],
            "Failed to generate audio.".into(),
        ),
    };

        // (StatusCode::INTERNAL_SERVER_ERROR,
        // [(axum::http::header::CONTENT_TYPE, "text/plain")],
        // "Server Error".into())
}