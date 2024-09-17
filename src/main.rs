use anyhow::Error as E;
use clap::{Parser, ValueEnum};

use candle_core::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::parler_tts::{Config, Model};
use tokenizers::Tokenizer;
use axum::response::IntoResponse;
use axum::http::header;
use axum::body::Body;
use axum::Router;
use tokio::fs::File;
use tokio_util::io::ReaderStream;
use axum::routing::post;


#[derive(Parser, Debug)]
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

#[tokio::main]
async fn main() {
    if let Err(_) = tracing_subscriber::fmt::try_init() {
        println!("Tracing subscriber has already been initialized.");
    }

    // let app = Router::new().route("/generate", post(generate_audio)).with_state(shared_state);
    let app = Router::new().route("/generate", post(generate_audio));

    // Run the app with hyper, listening on port 3000.
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn generate_audio() -> impl IntoResponse {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    // let args = Args::parse();
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

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        if let Err(_) = tracing_subscriber::registry().with(chrome_layer).try_init() {
            println!("Tracing subscriber has already been initialized.");
        }
        Some(guard)
    } else {
        None
    };
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
    let mut model = Model::new(&config, vb).unwrap();
    println!("loaded the model in {:?}", start.elapsed());

    let description_tokens = tokenizer
        .encode(args.description, true)
        .map_err(E::msg).unwrap()
        .get_ids()
        .to_vec();
    let description_tokens = Tensor::new(description_tokens, &device).unwrap().unsqueeze(0).unwrap();
    let prompt_tokens = tokenizer
        .encode(args.prompt, true)
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
    let codes = model.generate(&prompt_tokens, &description_tokens, lp, args.max_steps).unwrap();
    println!("generated codes\n{codes}");
    let codes = codes.to_dtype(DType::I64).unwrap();
    codes.save_safetensors("codes", "out.safetensors").unwrap();
    let codes = codes.unsqueeze(0).unwrap();
    let pcm = model
        .audio_encoder
        .decode_codes(&codes.to_device(&device).unwrap()).unwrap();
    println!("{pcm}");
    let pcm = pcm.i((0, 0)).unwrap();
    let pcm = candle_examples::audio::normalize_loudness(&pcm, 24_000, true).unwrap();
    let pcm = pcm.to_vec1::<f32>().unwrap();
    let mut output = std::fs::File::create(&args.out_file).unwrap();
    candle_examples::wav::write_pcm_as_wav(&mut output, &pcm, config.audio_encoder.sampling_rate).unwrap();
    println!("inference time: {:?}", start.elapsed());
    
    println!("Stream");
    let header = [
        (header::CONTENT_TYPE, "audio/wav"),
        (header::CONTENT_LENGTH, ""),
        (header::TRANSFER_ENCODING, "chunked"),
        (header::CACHE_CONTROL, "no-cache, must-revalidate"),
        (header::PRAGMA, "no-cache"),

    ];
    let file = File::open(&args.out_file).await.unwrap();
    let stream = ReaderStream::new(file);
    
    Body::from_stream(stream)
    //Ok(())
}