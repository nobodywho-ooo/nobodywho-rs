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

use crate::chat_state;

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

#[derive(Clone)]
pub struct SamplerConfig {
    pub method: SamplerMethod,
    pub penalty_last_n: i32,
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
    pub penalize_nl: bool,
    pub ignore_eos: bool,
}

impl Default for SamplerConfig {
    fn default () -> Self {
        Self {
            penalty_last_n: -1,
            penalty_repeat: 0.0,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            penalize_nl: false,
            ignore_eos: false,
            method : SamplerMethod::Temperature(TemperatureConfig::default()),
            // method : SamplerMethod::MirostatV2(MirostatV2Config {
            //     seed: 1234,
            //     temperature: 0.8,
            //     tau: 5.0,
            //     eta: 0.1,
            // }),
        }
    }
}


#[derive(Clone)]
pub enum SamplerMethod {
    MirostatV2(MirostatV2Config),
    Temperature(TemperatureConfig),
    TopK(TopKConfig),
    Greedy,
}

pub struct TopKConfig {
    pub top_k: i32,
    pub seed: u32
}

impl Default for TopKConfig {
    fn default() -> Self {
        Self { top_k : 40, seed : 1234 }
    }
}

#[derive(Clone)]
pub struct MirostatV2Config {
    pub seed: u32,
    pub temperature: f32,
    pub tau: f32,
    pub eta: f32,
}

impl Default for MirostatV2Config {
    fn default() -> Self {
        Self {
            seed: 1234,
            temperature: 0.8,
            tau: 5.0,
            eta: 0.1
        }
    }
}

#[derive(Clone)]
pub struct TemperatureConfig {
    pub seed: u32,
    pub temperature: f32,
}

impl Default for TemperatureConfig {
    fn default() -> Self {
        Self {
            seed: 1234,
            temperature: 0.8,
        }
    }
}

fn make_sampler(model: &LlamaModel, sampler_config: SamplerConfig) -> LlamaSampler {
    // init mirostat sampler
    let penalties = LlamaSampler::penalties(
        model.n_vocab(),
        model.token_eos().0,
        model.token_nl().0,
        sampler_config.penalty_last_n,
        sampler_config.penalty_repeat,
        sampler_config.penalty_freq,
        sampler_config.penalty_present,
        sampler_config.penalize_nl,
        sampler_config.ignore_eos,
    );
    let methodvec = match sampler_config.method {
        SamplerMethod::MirostatV2(conf) => {
            vec![
                penalties,
                LlamaSampler::temp(conf.temperature),
                LlamaSampler::mirostat_v2(conf.seed, conf.tau, conf.eta),
            ]
        },
        SamplerMethod::Temperature(conf) => {
            vec![
                penalties,
                LlamaSampler::temp(conf.temperature),
                LlamaSampler::dist(conf.seed),
            ]
        }
        SamplerMethod::Greedy => {
            vec![LlamaSampler::greedy()]
        }
        SamplerMethod::TopK(conf) => {
            vec![
                LlamaSampler::top_k(conf.top_k),
                LlamaSampler::dist(conf.seed)
            ]
        }
    };
    let chainvec = vec![penalties].into_iter().chain(methodvec).collect();
    LlamaSampler::chain(chainvec, true)
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

pub fn run_completion_worker(
    model: Arc<LlamaModel>,
    message_rx: Receiver<String>,
    completion_tx: Sender<LLMOutput>,
    sampler_config: SamplerConfig,
    n_ctx: u32,
    system_prompt: String,
) {
    // this function is a pretty thin wrapper to send back an `Err` if we get it
    if let Err(msg) = run_completion_worker_result(
        model,
        message_rx,
        &completion_tx,
        sampler_config,
        n_ctx,
        system_prompt,
    ) {
        completion_tx
            .send(LLMOutput::FatalErr(msg))
            .expect("Could not send llm worker fatal error back to consumer.");
    }
}

fn run_completion_worker_result(
    model: Arc<LlamaModel>,
    message_rx: Receiver<String>,
    completion_tx: &Sender<LLMOutput>,
    sampler_config: SamplerConfig,
    n_ctx: u32,
    system_prompt: String,
) -> Result<(), WorkerError> {
    // according to llama.cpp source code, the longest known template is about 1200bytes
    let chat_template = model.get_chat_template(4_000)?;
    let mut chat_state = chat_state::ChatState::new(
        chat_template,
        model.token_to_str(model.token_bos(), Special::Tokenize)?,
        model.token_to_str(model.token_eos(), Special::Tokenize)?,
    );
    chat_state.add_message("system".to_string(), system_prompt);

    let n_threads = std::thread::available_parallelism()?.get() as i32;
    let n_ctx: u32 = std::cmp::min(n_ctx, model.n_ctx_train());
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(std::num::NonZero::new(n_ctx))
        .with_n_threads(n_threads)
        .with_n_threads_batch(n_threads);

    let mut ctx = model.new_context(&LLAMA_BACKEND, ctx_params)?;

    let mut n_cur = 0;

    let mut sampler = make_sampler(&model, sampler_config);

    while let Ok(content) = message_rx.recv() {
        chat_state.add_message("user".to_string(), content);

        let diff = chat_state.render_diff()?;

        let tokens_list = ctx.model.str_to_token(&diff, AddBos::Always)?;

        let mut batch = LlamaBatch::new(ctx.n_ctx() as usize, 1);

        // put tokens in the batch
        let last_index = (tokens_list.len() - 1) as i32;

        for (i, token) in (0..).zip(tokens_list.into_iter()) {
            // llama_decode will output logits only for the last token of the prompt
            let is_last = i == last_index;
            batch.add(token, n_cur + i, &[0], is_last)?;
        }

        ctx.decode(&mut batch)?;

        // main loop
        n_cur += batch.n_tokens();
        if n_ctx as i64 <= n_cur as i64 {
            return Err(WorkerError::ContextLengthExceededError);
        }

        let mut response = String::new();

        loop {
            // sample the next token
            {
                // sample the next token
                let new_token_id: LlamaToken = sampler.sample(&ctx, batch.n_tokens() - 1);
                sampler.accept(new_token_id);

                // is it an end of stream?
                if new_token_id == ctx.model.token_eos() {
                    batch.clear();
                    batch.add(new_token_id, n_cur, &[0], true)?;

                    chat_state.add_message("assistant".to_string(), response.clone());

                    completion_tx
                        .send(LLMOutput::Done(response.clone()))
                        .map_err(|_| WorkerError::SendError)?;

                    response.clear();
                    break;
                }

                // the longest token I've seen is llama3.2's token 119224, at 96 bytes
                const MAX_TOKEN_STR_LEN: usize = 128;
                let output_string = ctx.model.token_to_str_with_size(
                    new_token_id,
                    MAX_TOKEN_STR_LEN,
                    Special::Tokenize,
                )?;

                response.push_str(&output_string);

                // send new token string back to user
                completion_tx
                    .send(LLMOutput::Token(output_string))
                    .map_err(|_| WorkerError::SendError)?;

                // prepare batch or the next decode
                batch.clear();

                batch.add(new_token_id, n_cur, &[0], true)?;
            }

            n_cur += 1;
            if n_ctx <= n_cur.try_into().expect("n_cur does not fit in u32") {
                return Err(WorkerError::ContextLengthExceededError);
            }

            ctx.decode(&mut batch)?;
        }
    }
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

        let tokens_list = ctx.model.str_to_token(&text, AddBos::Always)?;

        batch
            .add_sequence(&tokens_list, 0, false)
            .expect("Failed to add sequence");

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
                DEFAULT_SAMPLER_CONFIG,
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
