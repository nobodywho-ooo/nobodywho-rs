use std::num::NonZeroU32;
use std::pin::pin;
use std::sync::mpsc::{Receiver, Sender};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel, Special};
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use std::sync::LazyLock;

static LLAMA_BACKEND: LazyLock<LlamaBackend> =
    LazyLock::new(|| LlamaBackend::init().expect("Failed to initialize llama backend"));

#[derive(Debug)]
enum ActorMessage {
    GetCompletion {
        prompt: String,
        respond_to: Sender<ModelOutput>,
    },
    GetChatResponse {
        text: String,
        respond_to: Sender<ModelOutput>,
    },
}

pub struct ModelActor {
    seed: u32,
    model_path: String,
    sender: Option<Sender<ActorMessage>>,
}

#[derive(Debug)]
pub enum ModelOutput {
    Token(String),
    Done,
}

impl ModelActor {
    pub fn from_model_path(model_path: String) -> Self {
        let seed = 1234; // default seed
        Self {
            seed,
            model_path,
            sender: None,
        }
    }

    pub fn with_seed(self, seed: u32) -> Self {
        //! set seed
        //! XXX: only works if `run` has not been called yet
        Self { seed, ..self }
    }

    pub fn run(&mut self) {
        // loads model and starts thread
        // TODO: can we give better errors here?
        let model_path = self.model_path.clone();
        let seed = self.seed;
        let (sender, receiver) = std::sync::mpsc::channel();
        self.sender = Some(sender);
        std::thread::spawn(move || model_worker(model_path, seed, receiver));
    }

    pub fn get_completion(&self, prompt: String, tx: Sender<ModelOutput>) {
        if let Some(sender) = &(self.sender) {
            sender
                .send(ActorMessage::GetCompletion {
                    prompt,
                    respond_to: tx,
                })
                .unwrap();
        } else {
            panic!("Model actor is not running. Call run() first.");
        }
    }

    pub fn get_chat_response(&self, text: String, tx: Sender<ModelOutput>) {
        if let Some(sender) = &self.sender {
            sender
                .send(ActorMessage::GetChatResponse {
                    text,
                    respond_to: tx,
                })
                .unwrap();
        } else {
            panic!("Model actor is not running. Call run() first.");
        }
    }
}

fn get_completion(
    ctx: &mut LlamaContext,
    model: &LlamaModel,
    prompt: &str,
    respond_to: &Sender<ModelOutput>,
) {
    let tokens_list = model.str_to_token(&prompt, AddBos::Always).unwrap();

    // create a llama_batch
    // we use this object to submit token data for decoding
    // TODO: how big should we make the batch?
    //       it was (arbitrarily) 512 tokens before
    //       now it is the maximum size
    //       what are the performance implications?
    let mut batch = LlamaBatch::new(ctx.n_ctx() as usize, 1);

    // put tokens in the batch
    let last_index: i32 = (tokens_list.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        // llama_decode will output logits only for the last token of the prompt
        let is_last = i == last_index;
        batch.add(token, i, &[0], is_last).unwrap();
    }

    // llm go brrrr
    ctx.decode(&mut batch).unwrap();

    // main loop
    let mut n_cur = batch.n_tokens();

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
            if new_token_id == model.token_eos() {
                // XXX: is there a better way to deal with error here?
                //      this will error if the other end of the respond_to channel is free'd
                //      I'm not sure there's much we can do, maybe print a warning?
                let _ = respond_to.send(ModelOutput::Done);
                batch.clear();
                batch.add(new_token_id, n_cur, &[0], true).unwrap();
                break;
            }

            // convert tokens to String
            let output_bytes = model
                .token_to_bytes(new_token_id, Special::Tokenize)
                .unwrap();
            // use `Decoder.decode_to_string()` to avoid the intermediate buffer
            let mut output_string = String::with_capacity(32);
            let _decode_result =
                utf8decoder.decode_to_string(&output_bytes, &mut output_string, false);

            // send new token string back to user
            respond_to.send(ModelOutput::Token(output_string)).unwrap();

            // prepare batch or the next decode
            batch.clear();
            batch.add(new_token_id, n_cur, &[0], true).unwrap();
        }

        n_cur += 1;

        ctx.decode(&mut batch).unwrap();
    }
}

fn model_worker(model_path: String, seed: u32, receiver: Receiver<ActorMessage>) {
    // Offload the model to the GPU
    let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);

    let model_params = pin!(model_params);

    let model = LlamaModel::load_from_file(&LLAMA_BACKEND, model_path, &model_params).unwrap();

    let n_threads = num_cpus::get() as i32;
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(2048))
        .with_seed(seed)
        .with_n_threads(n_threads)
        .with_n_threads_batch(n_threads);

    let mut ctx = model.new_context(&LLAMA_BACKEND, ctx_params).unwrap();

    loop {
        let got = receiver.recv();
        match got {
            Ok(ActorMessage::GetChatResponse { text, respond_to }) => {
                // XXX: this fails if either of the strings contain a null-byte
                let chat_msg = LlamaChatMessage::new("user".to_string(), text.clone()).unwrap();
                let chat: Vec<LlamaChatMessage> = vec![chat_msg];

                // from llama.h: "The recommended alloc size is 2 * (total number of characters of all messages"
                // XXX: this gets a BuffSizeError if the chat history is very short
                let expanded_text = model.apply_chat_template(None, chat, true).unwrap();
                get_completion(&mut ctx, &model, &expanded_text, &respond_to)
            }
            Ok(ActorMessage::GetCompletion { prompt, respond_to }) => {
                get_completion(&mut ctx, &model, &prompt, &respond_to)
            }
            Err(_) => {
                // receiver is no longer attached to a sender
                // this must mean that we can safely kill this worker
                // since there is no longer any way to give it new tasks
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completion() {
        let mut actor = ModelActor::from_model_path("./model.bin".to_string()).with_seed(1337);
        actor.run();
        let (tx, rx) = std::sync::mpsc::channel();
        actor.get_completion("Count to five: 1, 2, ".to_string(), tx.clone());
        let mut result: String = "".to_string();
        while let Ok(ModelOutput::Token(s)) = rx.recv() {
            result += s.as_str();
        }
        let expected = "3, 4, 5";
        assert_eq!(&result[..expected.len()], expected);
    }

    #[test]
    fn test_chat() {
        let mut actor = ModelActor::from_model_path("./model.bin".to_string());
        actor.run();
        let (tx, rx) = std::sync::mpsc::channel();
        actor.get_chat_response("I need you to respond with just the text 'Hello, world!' literally, without the quotes.".to_string(), tx);
        let mut result: String = "".to_string();
        while let Ok(ModelOutput::Token(s)) = rx.recv() {
            result += s.as_str();
        }
        let expected = "Hello, world!";
        assert_eq!(&result[..expected.len()], expected);
    }
}
