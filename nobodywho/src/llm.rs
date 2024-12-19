use crate::chat_state;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::pin::pin;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, LazyLock};

// the longest token I've seen is llama3.2's token 119224, at 96 bytes
const MAX_TOKEN_STR_LEN: usize = 128;

static LLAMA_BACKEND: LazyLock<LlamaBackend> =
    LazyLock::new(|| LlamaBackend::init().expect("Failed to initialize llama backend"));

pub enum LLMOutput {
    Token(String),
    FatalErr(WorkerError),
    Done(String),
}

pub type Model = Arc<LlamaModel>;

pub fn has_discrete_gpu() -> bool {
    // Use the `wgpu` crate to enumerate GPUs,
    // label them as CPU, Integrated GPU, and Discrete GPU
    // and return true only if one of them is a discrete gpu.
    // TODO: how does this act on macos?
    wgpu::Instance::default()
        .enumerate_adapters(wgpu::Backends::all())
        .into_iter()
        .any(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
}

#[derive(Debug, thiserror::Error)]
pub enum LoadModelError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Invalid or unsupported GGUF model: {0}")]
    InvalidModel(String),
}

pub fn get_model(
    model_path: &str,
    use_gpu_if_available: bool,
) -> Result<Arc<LlamaModel>, LoadModelError> {
    // HACK: only offload anything to the gpu if we can find a dedicated GPU
    //       there seems to be a bug which results in garbage tokens if we over-allocate an integrated GPU
    //       while using the vulkan backend. See: https://github.com/nobodywho-ooo/nobodywho-rs/pull/14
    if !std::path::Path::new(model_path).exists() {
        return Err(LoadModelError::ModelNotFound(model_path.into()));
    }

    let model_params = LlamaModelParams::default().with_n_gpu_layers(
        if use_gpu_if_available && (has_discrete_gpu() || cfg!(target_os = "macos")) {
            1000000
        } else {
            0
        },
    );
    let model_params = pin!(model_params);
    let model =
        LlamaModel::load_from_file(&LLAMA_BACKEND, model_path, &model_params).map_err(|e| {
            LoadModelError::InvalidModel(format!(
                "Bad model path: {} - Llama.cpp error: {}",
                model_path, e
            ))
        })?;
    Ok(Arc::new(model))
}

// random candidates
pub struct SamplerConfig {
    pub seed: u32,
    pub temperature: f32,
    pub penalty_last_n: i32,
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
    pub penalize_nl: bool,
    pub ignore_eos: bool,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            seed: 1234,
            temperature: 0.8,
            penalty_last_n: -1,
            penalty_repeat: 0.0,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            penalize_nl: false,
            ignore_eos: false,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
        }
    }
}

fn make_sampler(model: &LlamaModel, config: SamplerConfig) -> LlamaSampler {
    // init mirostat sampler
    LlamaSampler::chain(
        [
            LlamaSampler::penalties(
                model.n_vocab(),
                model.token_eos().0,
                model.token_nl().0,
                config.penalty_last_n,
                config.penalty_repeat,
                config.penalty_freq,
                config.penalty_present,
                config.penalize_nl,
                config.ignore_eos,
            ),
            LlamaSampler::temp(config.temperature),
            //LlamaSampler::mirostat_v2(config.seed, config.mirostat_tau, config.mirostat_eta),
            LlamaSampler::dist(config.seed),
        ],
        true, // no pperf
    )
}

#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    #[error("Could not determine number of threads available: {0}")]
    ThreadCountError(#[from] std::io::Error),

    #[error("Could not create context: {0}")]
    CreateContextError(#[from] llama_cpp_2::LlamaContextLoadError),

    #[error("Could not tokenize string: {0}")]
    TokenizerError(#[from] llama_cpp_2::StringToTokenError),

    #[error("Could not detokenize string: {0}")]
    Detokenize(#[from] llama_cpp_2::TokenToStringError),

    #[error("Could not add token to batch: {0}")]
    BatchAddError(#[from] llama_cpp_2::llama_batch::BatchAddError),

    #[error("Llama.cpp failed decoding: {0}")]
    DecodeError(#[from] llama_cpp_2::DecodeError),

    #[error("Lama.cpp failed fetching chat template: {0}")]
    ChatTemplateError(#[from] llama_cpp_2::ChatTemplateError),

    #[error("Failed applying the jinja chat template: {0}")]
    ApplyTemplateError(#[from] minijinja::Error),

    #[error("Context exceeded maximum length")]
    ContextLengthExceededError,

    #[error("Could not send newly generated token out to the game engine.")]
    SendError, // this is actually a SendError<LLMOutput>, but that becomes recursive and weord.
}

/// Adds a sequence of tokens to the batch for processing.
///
/// # Arguments
/// * `batch` - The batch to add tokens to
/// * `tokens` - The sequence of tokens to add
/// * `pos` - The starting position in the context
/// * `seq_ids` - Sequence IDs for the tokens
///
/// # Returns
/// * `Ok(())` if successful
/// * `Err(WorkerError)` if batch addition fails
fn add_sequence(
    batch: &mut LlamaBatch,
    tokens: &[LlamaToken],
    pos: i32,
    seq_ids: &[i32],
) -> Result<(), WorkerError> {
    let n_tokens = tokens.len();

    for (i, token) in (0..).zip(tokens.iter()) {
        // Only compute logits for the last token to save computation
        let output_logits = i == n_tokens - 1;
        batch.add(*token, pos + i as i32, seq_ids, output_logits)?;
    }

    Ok(())
}

/// Main entry point for the completion worker thread.
/// Wraps the actual worker implementation to handle error reporting.
pub fn run_completion_worker(
    model: Arc<LlamaModel>,
    message_rx: Receiver<String>,
    completion_tx: Sender<LLMOutput>,
    sampler_config: SamplerConfig,
    n_ctx: u32,
    system_prompt: String,
) {
    if let Err(msg) = run_completion_worker_result(
        model,
        message_rx,
        &completion_tx,
        sampler_config,
        n_ctx,
        system_prompt,
    ) {
        // Forward fatal errors to the consumer
        completion_tx
            .send(LLMOutput::FatalErr(msg))
            .expect("Could not send llm worker fatal error back to consumer.");
    }
}

/// Core implementation of the completion worker.
///
/// # Arguments
/// * `model` - The LLaMA model to use for inference
/// * `message_rx` - Channel receiver for incoming user messages
/// * `completion_tx` - Channel sender for completion outputs
/// * `sampler_config` - Configuration for the token sampler
/// * `n_ctx` - Maximum context length
/// * `system_prompt` - System prompt to initialize the chat
///
/// # Returns
/// * `Ok(())` if the worker exits normally
/// * `Err(WorkerError)` on fatal errors
fn run_completion_worker_result(
    model: Arc<LlamaModel>,
    message_rx: Receiver<String>,
    completion_tx: &Sender<LLMOutput>,
    sampler_config: SamplerConfig,
    n_ctx: u32,
    system_prompt: String,
) -> Result<(), WorkerError> {
    // Initialize chat state with model's chat template
    let mut chat_state = chat_state::ChatState::new(
        model.get_chat_template(4_000)?,
        model.token_to_str(model.token_bos(), Special::Tokenize)?,
        model.token_to_str(model.token_eos(), Special::Tokenize)?,
    );
    chat_state.add_message("system".to_string(), system_prompt);

    // Set up context parameters using available parallelism
    let n_threads = std::thread::available_parallelism()?.get() as i32;
    let n_ctx = std::cmp::min(n_ctx, model.n_ctx_train());
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(std::num::NonZero::new(n_ctx))
        .with_n_threads(n_threads)
        .with_n_threads_batch(n_threads);

    // Create inference context and sampler
    let mut ctx = model.new_context(&LLAMA_BACKEND, ctx_params)?;
    let mut sampler = make_sampler(&model, sampler_config);

    let mut n_cur = 0; // Current position in context window
    let mut response = String::new();

    // Main message processing loop
    while let Ok(content) = message_rx.recv() {
        // Add user message to chat state
        chat_state.add_message("user".to_string(), content);

        // Get the new tokens to process since last update
        let diff = chat_state.render_diff()?;
        let tokens = ctx.model.str_to_token(&diff, AddBos::Always)?;

        // Create batch for processing tokens
        let mut batch = LlamaBatch::new(ctx.n_ctx() as usize, 1);
        add_sequence(&mut batch, &tokens, n_cur, &[0])?;

        // Token generation loop
        loop {
            // Check for context window overflow
            if n_cur + batch.n_tokens() >= ctx.n_ctx() as i32 {
                return Err(WorkerError::ContextLengthExceededError);
            }

            // Process current batch
            ctx.decode(&mut batch)?;
            n_cur += batch.n_tokens();

            // Sample next token
            let new_token: LlamaToken = sampler.sample(&ctx, -1);
            sampler.accept(new_token);

            // Check for end of generation
            if ctx.model.is_eog_token(new_token) {
                break;
            }

            // Convert token to text and stream to user
            let output_string = ctx.model.token_to_str_with_size(
                new_token,
                MAX_TOKEN_STR_LEN,
                Special::Tokenize,
            )?;
            response.push_str(&output_string);
            completion_tx
                .send(LLMOutput::Token(output_string))
                .map_err(|_| WorkerError::SendError)?;

            // Prepare batch for next token
            batch.clear();
            batch.add(new_token, n_cur as i32, &[0], true)?;
        }

        // Process final batch and update chat state
        ctx.decode(&mut batch)?;
        chat_state.add_message("assistant".to_string(), response.clone());

        // Send completion signal
        completion_tx
            .send(LLMOutput::Done(response.clone()))
            .map_err(|_| WorkerError::SendError)?;

        response.clear();
    }

    // This should be unreachable as the receiver loop only exits on error
    unreachable!();
}
pub enum EmbeddingsOutput {
    Embedding(Vec<f32>),
    FatalError(WorkerError),
}

pub fn run_embedding_worker(
    model: Arc<LlamaModel>,
    text_rx: Receiver<String>,
    embedding_tx: Sender<EmbeddingsOutput>,
) {
    // this function is a pretty thin wrapper to send back an `Err` if we get it
    if let Err(msg) = run_embedding_worker_result(model, text_rx, &embedding_tx) {
        embedding_tx
            .send(EmbeddingsOutput::FatalError(msg))
            .expect("Could not send llm worker fatal error back to consumer.");
    }
}

pub fn run_embedding_worker_result(
    model: Arc<LlamaModel>,
    text_rx: Receiver<String>,
    embedding_tx: &Sender<EmbeddingsOutput>,
) -> Result<(), WorkerError> {
    let n_threads = std::thread::available_parallelism()?.get() as i32;
    let ctx_params = LlamaContextParams::default()
        .with_n_threads(n_threads)
        .with_embeddings(true);

    let mut ctx = model.new_context(&LLAMA_BACKEND, ctx_params)?;

    while let Ok(text) = text_rx.recv() {
        let mut batch = LlamaBatch::new(ctx.n_ctx() as usize, 1);

        let tokens = ctx.model.str_to_token(&text, AddBos::Always)?;

        add_sequence(&mut batch, &tokens, 0, &[0]).expect("Failed to add sequence");

        ctx.clear_kv_cache();

        ctx.decode(&mut batch)?;

        let embedding = ctx.embeddings_seq_ith(0).unwrap().to_vec();
        embedding_tx
            .send(EmbeddingsOutput::Embedding(embedding))
            .map_err(|_| WorkerError::SendError)?;
    }
    Ok(())
}

fn dotproduct(a: &[f32], b: &[f32]) -> f32 {
    assert!(a.len() == b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let norm_a = dotproduct(a, a).sqrt();
    let norm_b = dotproduct(b, b).sqrt();
    if norm_a == 0. || norm_b == 0. {
        return f32::NAN;
    }
    dotproduct(a, b) / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test_model_path {
        () => {
            std::env::var("TEST_MODEL")
                .unwrap_or("model.gguf".to_string())
                .as_str()
        };
    }

    macro_rules! test_embeddings_model_path {
        () => {
            std::env::var("TEST_EMBEDDINGS_MODEL")
                .unwrap_or("embeddings.gguf".to_string())
                .as_str()
        };
    }

    #[test]
    fn test_chat_completion() {
        let model = get_model(test_model_path!(), true).unwrap();

        let (prompt_tx, prompt_rx) = std::sync::mpsc::channel();
        let (completion_tx, completion_rx) = std::sync::mpsc::channel();

        let system_prompt = "You are a helpful assistant. The user asks you a question, and you provide an answer. You take multiple turns to provide the answer. Be consice and only provide the answer".to_string();
        std::thread::spawn(|| {
            run_completion_worker(
                model,
                prompt_rx,
                completion_tx,
                SamplerConfig::default(),
                4096,
                system_prompt,
            )
        });

        prompt_tx
            .send("What is the capital of Denmark?".to_string())
            .unwrap();

        let result: String;
        loop {
            match completion_rx.recv() {
                Ok(LLMOutput::Token(_)) => {}
                Ok(LLMOutput::Done(response)) => {
                    result = response;
                    break;
                }
                _ => unreachable!(),
            }
        }
        assert!(
            result.contains("Copenhagen"),
            "Expected completion to contain 'Copenhagen', got: {result}"
        );

        prompt_tx
            .send("What language to they speak there?".to_string())
            .unwrap();
        let result: String;
        loop {
            match completion_rx.recv() {
                Ok(LLMOutput::Token(_)) => {}
                Ok(LLMOutput::Done(response)) => {
                    result = response;
                    break;
                }
                _ => unreachable!(),
            }
        }

        assert!(
            result.contains("Danish"),
            "Expected completion to contain 'Danish', got: {result}"
        );
    }

    #[test]
    fn test_embeddings() {
        let model = get_model(test_embeddings_model_path!(), true).unwrap();

        let (prompt_tx, prompt_rx) = std::sync::mpsc::channel();
        let (embedding_tx, embedding_rx) = std::sync::mpsc::channel();

        std::thread::spawn(|| run_embedding_worker(model, prompt_rx, embedding_tx));

        prompt_tx
            .send("Copenhagen is the capital of Denmark.".to_string())
            .unwrap();
        let copenhagen_embedding = match embedding_rx.recv() {
            Ok(EmbeddingsOutput::Embedding(vec)) => vec,
            _ => panic!(),
        };

        prompt_tx
            .send("Berlin is the capital of Germany.".to_string())
            .unwrap();
        let berlin_embedding = match embedding_rx.recv() {
            Ok(EmbeddingsOutput::Embedding(vec)) => vec,
            _ => panic!(),
        };

        prompt_tx
            .send("Your mother was a hamster and your father smelt of elderberries!".to_string())
            .unwrap();
        let insult_embedding = match embedding_rx.recv() {
            Ok(EmbeddingsOutput::Embedding(vec)) => vec,
            _ => panic!(),
        };

        assert!(
            insult_embedding.len() == berlin_embedding.len()
                && berlin_embedding.len() == copenhagen_embedding.len()
                && copenhagen_embedding.len() == insult_embedding.len(),
            "not all embedding lengths were equal"
        );

        // cosine similarity should not care about order
        assert_eq!(
            cosine_similarity(&copenhagen_embedding, &berlin_embedding),
            cosine_similarity(&berlin_embedding, &copenhagen_embedding)
        );

        // any vector should have cosine similarity 1 to itself
        // (tolerate small float error)
        assert!(
            (cosine_similarity(&copenhagen_embedding, &copenhagen_embedding) - 1.0).abs() < 0.001,
        );

        // the insult should have a lower similarity than the two geography sentences
        assert!(
            cosine_similarity(&copenhagen_embedding, &insult_embedding)
                < cosine_similarity(&copenhagen_embedding, &berlin_embedding)
        );
    }
}
