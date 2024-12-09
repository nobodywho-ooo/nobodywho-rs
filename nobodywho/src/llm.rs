use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::sampling::params::LlamaSamplerChainParams;
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
                model_path.to_string(),
                e.to_string()
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

// defaults here match the defaults read from `llama-cli --help`
pub const DEFAULT_SAMPLER_CONFIG: SamplerConfig = SamplerConfig {
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
};

fn make_sampler(
    model: &LlamaModel,
    config: SamplerConfig,
) -> Result<LlamaSampler, llama_cpp_2::LlamaSamplerError> {
    // init mirostat sampler
    let sampler_params = LlamaSamplerChainParams::default();
    let sampler = LlamaSampler::new(sampler_params)?
        .add_penalties(
            model.n_vocab(),
            model.token_eos().0,
            model.token_nl().0,
            config.penalty_last_n,
            config.penalty_repeat,
            config.penalty_freq,
            config.penalty_present,
            config.penalize_nl,
            config.ignore_eos,
        )
        .add_temp(config.temperature)
        .add_mirostat_v2(config.seed, config.mirostat_tau, config.mirostat_eta);

    Ok(sampler)
}

#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    #[error("Could not determine number of threads available: {0}")]
    ThreadCountError(#[from] std::io::Error),

    #[error("Could not create context: {0}")]
    CreateContextError(#[from] llama_cpp_2::LlamaContextLoadError),

    #[error("Could not create sampler: {0}")]
    CreateSamplerError(#[from] llama_cpp_2::LlamaSamplerError),

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

pub fn run_worker(
    model: Arc<LlamaModel>,
    message_rx: Receiver<(String, String)>,
    completion_tx: Sender<LLMOutput>,
    sampler_config: SamplerConfig,
    n_ctx: u32,
    system_prompt: String
) {
    // this function is a pretty thin wrapper to send back an `Err` if we get it
    if let Err(msg) = run_worker_result(model, message_rx, &completion_tx, sampler_config, n_ctx, system_prompt) {
        completion_tx
            .send(LLMOutput::FatalErr(msg))
            .expect("Could not send llm worker fatal error back to consumer.");
    }
}

fn run_worker_result(
    model: Arc<LlamaModel>,
    message_rx: Receiver<(String, String)>,
    completion_tx: &Sender<LLMOutput>,
    sampler_config: SamplerConfig,
    n_ctx: u32,
    system_prompt: String,
) -> Result<(), WorkerError> {
    // according to llama.cpp source code, the longest known template is about 1200bytes
    let chat_template = model.get_chat_template(4_000)?;
    let mut chat_state = chat_state::ChatState::new(chat_template);
    chat_state.add_message("system", &system_prompt);

    let n_threads = std::thread::available_parallelism()?.get() as i32;
    let n_ctx: u32 = std::cmp::min(n_ctx, model.n_ctx_train());
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(std::num::NonZero::new(n_ctx))
        .with_n_threads(n_threads)
        .with_n_threads_batch(n_threads);

    let mut ctx = model.new_context(&LLAMA_BACKEND, ctx_params)?;

    let mut n_cur = 0;

    let mut sampler = make_sampler(&model, sampler_config)?;

    while let Ok((role, content)) = message_rx.recv() {
        chat_state.add_message(&role, &content);

        let diff = chat_state.render_diff()?;
        println!("DIFF:{}", diff);

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

        // The `Decoder`
        let mut utf8decoder = encoding_rs::UTF_8.new_decoder();

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

                    chat_state.add_message("assistant", &response);

                    completion_tx
                        .send(LLMOutput::Done(response))
                        .map_err(|_| WorkerError::SendError)?;
                    break;
                }

                let output_bytes = ctx.model.token_to_bytes(new_token_id, Special::Tokenize)?;

                // use `Decoder.decode_to_string()` to avoid the intermediate buffer
                let mut output_string = String::with_capacity(32);
                let _decode_result =
                    utf8decoder.decode_to_string(&output_bytes, &mut output_string, false);

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

macro_rules! test_model_path {
    () => {
        std::env::var("TEST_MODEL")
            .unwrap_or("model.gguf".to_string())
            .as_str()
    };
}

#[cfg(test)]
mod tests {
    use llama_cpp_2::model::LlamaChatMessage;

    use super::*;

    #[test]
    fn test_completion() {
        let model = get_model(test_model_path!(), true).unwrap();

        let (prompt_tx, prompt_rx) = std::sync::mpsc::channel();
        let (completion_tx, completion_rx) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            run_worker(
                model,
                prompt_rx,
                completion_tx,
                DEFAULT_SAMPLER_CONFIG,
                4096,
            )
        });

        prompt_tx.send("Count to five: 1, 2, ".to_string()).unwrap();

        let mut result: String = "".to_string();

        for _ in 0..10 {
            match completion_rx.recv() {
                Ok(LLMOutput::Token(token)) => {
                    result += token.as_str();
                }
                Ok(LLMOutput::Done) => {
                    break;
                }
                _ => unreachable!(),
            }
        }
        let expected = "3, 4, 5";
        assert_eq!(&result[..expected.len()], expected);

        // Kill worker
    }

    #[test]
    fn test_chat_completion() {
        let model = get_model(test_model_path!(), true).unwrap();
        let model_copy = model.clone();

        let (prompt_tx, prompt_rx) = std::sync::mpsc::channel();
        let (completion_tx, completion_rx) = std::sync::mpsc::channel();

        std::thread::spawn(|| {
            run_worker(
                model,
                prompt_rx,
                completion_tx,
                DEFAULT_SAMPLER_CONFIG,
                4096,
            )
        });

        let chat: Vec<LlamaChatMessage> = vec![
            LlamaChatMessage::new(
                "system".to_string(),
                "You are a helpful assistant. The user asks you a question, and you provide an answer. You take multiple turns to provide the answer. Be consice and only provide the answer".to_string(),
            )
            .unwrap(),
            LlamaChatMessage::new(
                "user".to_string(),
                "What is the capital of Denmark?".to_string(),
            )
            .unwrap(),
        ];

        let prompt = model_copy.apply_chat_template(None, chat, true).unwrap();

        prompt_tx.send(prompt).unwrap();

        let mut result: String = "".to_string();

        loop {
            match completion_rx.recv() {
                Ok(LLMOutput::Token(token)) => {
                    result += token.as_str();
                }
                Ok(LLMOutput::Done) => {
                    break;
                }
                _ => unreachable!(),
            }
        }

        let chat: Vec<LlamaChatMessage> = vec![LlamaChatMessage::new(
            "user".to_string(),
            "What language to they speak there?".to_string(),
        )
        .unwrap()];

        let prompt = model_copy.apply_chat_template(None, chat, true).unwrap();

        prompt_tx.send(prompt).unwrap();

        let mut result: String = "".to_string();

        loop {
            match completion_rx.recv() {
                Ok(LLMOutput::Token(token)) => {
                    result += token.as_str();
                }
                Ok(LLMOutput::Done) => {
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
    fn test_initialize_default_sampler() {
        let model = get_model(test_model_path!(), true).expect("Failed loading model");
        let sampler = make_sampler(&model, DEFAULT_SAMPLER_CONFIG);
        assert!(sampler.is_ok(), "make_sampler returned an Err");
    }
}
