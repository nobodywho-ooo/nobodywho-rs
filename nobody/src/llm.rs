use std::pin::pin;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, LazyLock};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaChatMessage;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::token::data_array::LlamaTokenDataArray;

static LLAMA_BACKEND: LazyLock<LlamaBackend> =
    LazyLock::new(|| LlamaBackend::init().expect("Failed to initialize llama backend"));

pub enum LLMOutput {
    Token(String),
    Done,
}

pub type Model = Arc<LlamaModel>;

pub fn get_model(model_path: &str) -> Arc<LlamaModel> {
    // TODO: Set the number of GPU layers
    let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);
    let model_params = pin!(model_params);
    Arc::new(LlamaModel::load_from_file(&LLAMA_BACKEND, model_path, &model_params).unwrap())
}

pub fn apply_chat_template(model: Model, chat: Vec<(String, String)>) -> Result<String, String> {
    let chat_result: Result<Vec<LlamaChatMessage>, String> = chat
        .into_iter()
        .map(|t| LlamaChatMessage::new(t.0, t.1).map_err(|e| e.to_string()))
        .collect();
    let chat_string = model.apply_chat_template(None, chat_result?, true).map_err(|e| e.to_string())?;
    Ok(chat_string)
}

pub fn run_worker(
    model: Arc<LlamaModel>,
    prompt_rx: Receiver<String>,
    completion_tx: Sender<LLMOutput>,
    seed: u32,
) {
    let n_threads = num_cpus::get() as i32;
    let ctx_params = LlamaContextParams::default()
        .with_seed(seed)
        .with_n_threads(n_threads)
        .with_n_threads_batch(n_threads);

    let mut ctx = model.new_context(&LLAMA_BACKEND, ctx_params).unwrap();

    let mut n_cur = 0;

    while let Ok(prompt) = prompt_rx.recv() {
        let tokens_list = ctx.model.str_to_token(&prompt, AddBos::Always).unwrap();

        let mut batch = LlamaBatch::new(ctx.n_ctx() as usize, 1);

        // put tokens in the batch
        let last_index = (tokens_list.len() - 1) as i32;

        for (i, token) in (0..).zip(tokens_list.into_iter()) {
            // llama_decode will output logits only for the last token of the prompt
            let is_last = i == last_index;
            batch.add(token, n_cur + i, &[0], is_last).unwrap();
        }

        ctx.decode(&mut batch).unwrap();

        // main loop
        n_cur += batch.n_tokens();

        // The `Decoder`
        let mut utf8decoder = encoding_rs::UTF_8.new_decoder();

        loop {
            // sample the next token
            {
                let candidates = ctx.candidates_ith(batch.n_tokens() - 1);

                let candidates_p = LlamaTokenDataArray::from_iter(candidates, false);

                // sample the most likely token
                // TODO: parameterize sampler
                let new_token_id = ctx.sample_token_greedy(candidates_p);

                // is it an end of stream?
                if new_token_id == ctx.model.token_eos() {
                    batch.clear();
                    batch.add(new_token_id, n_cur, &[0], true).unwrap();
                    completion_tx.send(LLMOutput::Done).unwrap();
                    break;
                }

                let output_bytes = ctx
                    .model
                    .token_to_bytes(new_token_id, Special::Tokenize)
                    .unwrap();

                // use `Decoder.decode_to_string()` to avoid the intermediate buffer
                let mut output_string = String::with_capacity(32);
                let _decode_result =
                    utf8decoder.decode_to_string(&output_bytes, &mut output_string, false);

                // send new token string back to user
                completion_tx.send(LLMOutput::Token(output_string)).unwrap();

                // prepare batch or the next decode
                batch.clear();

                batch.add(new_token_id, n_cur, &[0], true).unwrap();
            }

            n_cur += 1;

            ctx.decode(&mut batch).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_2::model::LlamaChatMessage;

    use super::*;

    #[test]
    fn test_completion() {
        let model = get_model("model.bin");

        let (prompt_tx, prompt_rx) = std::sync::mpsc::channel();
        let (completion_tx, completion_rx) = std::sync::mpsc::channel();

        std::thread::spawn(move || run_worker(model, prompt_rx, completion_tx, 1234));

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
        let model = get_model("model.bin");
        let model_copy = model.clone();

        let (prompt_tx, prompt_rx) = std::sync::mpsc::channel();
        let (completion_tx, completion_rx) = std::sync::mpsc::channel();

        std::thread::spawn(|| run_worker(model, prompt_rx, completion_tx, 1234));

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
}
