use anyhow::Error as E;
use clap::{Parser, ValueEnum};

use candle_core::{DType, IndexOp, Tensor, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::parler_tts::{Config, Model, Decoder};
use candle_transformers::models::t5;
use tokenizers::{tokenizer, Tokenizer};
use axum::response::{IntoResponse, Response};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::body::Body;
use axum::body::Bytes;
use axum::Router;
use axum::extract::State;
use tokio::fs::File;
use tokio_util::io::ReaderStream;
use axum::routing::post;
use std::sync::{Arc, Mutex};
use std::thread;
use std::io::Cursor;
use std::io::Write;


#[derive(Parser, Debug, Clone)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Display the token for the specified prompt.
    #[arg(long)]
    verbose_prompt: bool,

    #[arg(long, default_value = "Hey, how are you doing today?")]
    prompt: String,

    #[arg(
        long,
        default_value = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."
    )]
    description: String,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long, default_value = None)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    #[arg(long, default_value_t = 5000)]
    sample_len: usize,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(long, default_value = "parler-tts/parler-tts-mini-v1")]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: Option<String>,

    #[arg(long)]
    quantized: bool,

    /// Use f16 precision for all the computations rather than f32.
    #[arg(long)]
    f16: bool,

    #[arg(long)]
    model_file: Option<String>,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    #[arg(long, default_value_t = 512)]
    max_steps: usize,

    /// The output wav file.
    #[arg(long, default_value = "out.wav")]
    out_file: String,

    #[arg(long, default_value = "large-v1")]
    which: Which,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Which {
    #[value(name = "large-v1")]
    LargeV1,
    #[value(name = "mini-v1")]
    MiniV1,
}

struct AppState {
    model: Mutex<Model>,
    tokenizer: Tokenizer,
    device: Device,
    config: Config,
    args: Args,
    vb: VarBuilder<'static>,
    // Add any other shared resources here
}

#[tokio::main]
async fn main() {
    let args = Args {
        prompt: "Hey, how are you doing today?".to_string(),
        description: "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.".to_string(),
        temperature: 0.0,
        top_p: None,
        seed: 0,
        sample_len: 5000,
        repeat_penalty: 1.0,
        repeat_last_n: 64,
        model_id: Some("parler-tts/parler-tts-large-v1".to_string()),
        revision: Some("main".to_string()),
        quantized: false,
        f16: false,
        model_file: None,
        tokenizer_file: None,
        tracing: false,
        verbose_prompt: false,
        config_file: None,
        max_steps: 512,
        out_file: "out.wav".to_string(),
        which: Which::LargeV1,
        cpu: false,

    };
    println!("Args: {:?}", args);

    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature, args.repeat_penalty, args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let api = hf_hub::api::sync::Api::new().unwrap();
    let args_clone = args.clone();
    let model_id = match args.model_id {
        Some(model_id) => model_id.to_string(),
        None => match args.which {
            Which::LargeV1 => "parler-tts/parler-tts-large-v1".to_string(),
            Which::MiniV1 => "parler-tts/parler-tts-mini-v1".to_string(),
        },
    };
    let revision = match args.revision {
        Some(r) => r,
        None => "main".to_string(),
    };
    let repo = api.repo(hf_hub::Repo::with_revision(
        model_id,
        hf_hub::RepoType::Model,
        revision,
    ));
    let model_files = match args.model_file {
        Some(m) => vec![m.into()],
        None => match args.which {
            Which::MiniV1 => vec![repo.get("model.safetensors").unwrap()],
            Which::LargeV1 => {
                candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json").unwrap()
            }
        },
    };
    let config = match args.config_file {
        Some(m) => m.into(),
        None => repo.get("config.json").unwrap(),
    };
    let tokenizer = match args.tokenizer_file {
        Some(m) => m.into(),
        None => repo.get("tokenizer.json").unwrap(),
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg).unwrap();

    let start = std::time::Instant::now();
    let device = candle_examples::device(args.cpu).unwrap();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, DType::F32, &device).unwrap() };
    let config: Config = serde_json::from_reader(std::fs::File::open(config).unwrap()).unwrap();
    let mut model = Model::new(&config, vb.clone()).unwrap();
    let model_mutex = Mutex::new(model);
    println!("loaded the model in {:?}", start.elapsed());

    let app_state = Arc::new(AppState {
        model: model_mutex,
        tokenizer,
        device,
        config,
        args: args_clone,
        vb,
        // Initialize other shared resources if any
    });

    // let app = Router::new().route("/generate", post(generate_audio)).with_state(shared_state);
    let app = Router::new().route("/generate", post(generate_audio)).with_state(app_state);;

    // Run the app with hyper, listening on port 3000.
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn generate_audio(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let args = state.args.clone();
    let tokenizer = &state.tokenizer;
    let device = &state.device;
    let config = &state.config;
    let vb = &state.vb;

    let description_tokens = tokenizer
        .encode(args.description.to_string(), true)
        .map_err(E::msg).unwrap()
        .get_ids()
        .to_vec();
    let description_tokens = Tensor::new(description_tokens, &device).unwrap().unsqueeze(0).unwrap();
    let prompt_tokens = tokenizer
        .encode(args.prompt.to_string(), true)
        .map_err(E::msg).unwrap()
        .get_ids()
        .to_vec();
    let prompt_tokens = Tensor::new(prompt_tokens, &device).unwrap().unsqueeze(0).unwrap();
    let lp = candle_transformers::generation::LogitsProcessor::new(
        args.seed,
        Some(args.temperature),
        args.top_p,
    );
    println!("starting generation...");
    let start = std::time::Instant::now();
    
    // Run inference on separate thread
    // TODO: Do i need the reference & here?
    // let codes = &state.model.lock().unwrap().generate(&prompt_tokens, &description_tokens, lp, args.max_steps, &mut decoder, &mut text_encoder).unwrap();
    let cloned_state = Arc::clone(&state);
    let codes = tokio::task::spawn_blocking(move || {
        let mut decoder = Decoder::new(&cloned_state.config.decoder, cloned_state.vb.pp("decoder")).unwrap();
        let mut text_encoder = t5::T5EncoderModel::load(cloned_state.vb.pp("text_encoder"), &cloned_state.config.text_encoder).unwrap();
        cloned_state.model.lock().unwrap().generate(&prompt_tokens, &description_tokens, lp, args.max_steps, &mut decoder, &mut text_encoder)
    }).await.unwrap().unwrap();

    println!("generated codes\n{codes}");
    let codes = codes.to_dtype(DType::I64).unwrap();
    codes.save_safetensors("codes", "out.safetensors").unwrap();
    let codes = codes.unsqueeze(0).unwrap();

    // TODO: Do i need the reference & here?
    let pcm = &state.model.lock().unwrap()
        .audio_encoder
        .decode_codes(&codes.to_device(&device).unwrap()).unwrap();

    println!("{pcm}");
    let pcm = pcm.i((0, 0)).unwrap();
    let pcm = candle_examples::audio::normalize_loudness(&pcm, 24_000, true).unwrap();
    let pcm = pcm.to_vec1::<f32>().unwrap();

    // Generate WAV data in memory
    let wav_data = generate_wav_data(&pcm, state.config.audio_encoder.sampling_rate);
    let body = Body::from(wav_data);

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_static("audio/wav"));
    headers.insert("Content-Disposition", HeaderValue::from_static("attachment; filename=\"generated_audio.wav\""));
    println!("inference time: {:?}", start.elapsed());

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "audio/wav")
        .header("Content-Disposition", "attachment; filename=\"generated_audio.wav\"")
        .body(body)
        .unwrap()

    // let mut output = std::fs::File::create(&args.out_file).unwrap();
    // candle_examples::wav::write_pcm_as_wav(&mut output, &pcm, config.audio_encoder.sampling_rate).unwrap();
    // println!("inference time: {:?}", start.elapsed());
    
    // println!("Stream");
    // let file = File::open(&args.out_file).await.unwrap();
    // let stream = ReaderStream::new(file);
    
    // Body::from_stream(stream)
}

fn generate_wav_data(pcm: &[f32], sample_rate: u32) -> Bytes {
    let mut cursor = Cursor::new(Vec::new());
    
    // Write WAV header
    write_wav_header(&mut cursor, pcm.len() as u32, sample_rate).unwrap();
    
    // Write PCM data
    for &sample in pcm {
        let sample_i16 = (sample * 32767.0) as i16;
        cursor.write_all(&sample_i16.to_le_bytes()).unwrap();
    }

    Bytes::from(cursor.into_inner())
}

fn write_wav_header(writer: &mut impl std::io::Write, num_samples: u32, sample_rate: u32) -> std::io::Result<()> {
    let data_len = num_samples * 2; // 16-bit samples
    let total_header_len = 44;
    let total_len = total_header_len + data_len;

    writer.write_all(b"RIFF")?;
    writer.write_all(&(total_len - 8).to_le_bytes())?;
    writer.write_all(b"WAVE")?;
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?; // fmt chunk size
    writer.write_all(&1u16.to_le_bytes())?; // audio format (PCM)
    writer.write_all(&1u16.to_le_bytes())?; // num channels
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&(sample_rate * 2).to_le_bytes())?; // byte rate
    writer.write_all(&2u16.to_le_bytes())?; // block align
    writer.write_all(&16u16.to_le_bytes())?; // bits per sample
    writer.write_all(b"data")?;
    writer.write_all(&data_len.to_le_bytes())?;

    Ok(())
}
