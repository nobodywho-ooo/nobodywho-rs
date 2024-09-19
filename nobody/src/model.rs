use std::num::NonZeroU32;
use std::pin::pin;
use std::sync::mpsc::{Receiver, Sender};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::token::data_array::LlamaTokenDataArray;

enum ActorMessage {
    GetCompletion {
        prompt: String,
        respond_to: Sender<String>,
    },
}

pub struct ModelActor {
    seed: u32,
    model_path: String,
    sender: Option<Sender<ActorMessage>>,
}


impl ModelActor {
    pub fn from_model_path(model_path: String) -> Self {
        let seed = 1234; // default seed
        Self { seed, model_path, sender: None }
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

    pub fn get_completion(&self, prompt: String, tx: Sender<String>) {
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
}

fn model_worker(model_path: String, seed: u32, receiver: Receiver<ActorMessage>) {
    let backend = LlamaBackend::init().expect("Failed to initialize LlamaBackend");

    // Offload the model to the GPU
    let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);

    let model_params = pin!(model_params);

    let model = LlamaModel::load_from_file(&backend, model_path, &model_params).unwrap();

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(2048))
        .with_seed(seed);

    let mut ctx = model.new_context(&backend, ctx_params).unwrap();

    while let Ok(ActorMessage::GetCompletion { prompt, respond_to }) = receiver.recv() {
        let tokens_list = model.str_to_token(&prompt, AddBos::Always).unwrap();

        // print the prompt token-by-token
        for token in &tokens_list {
            println!("{}", model.token_to_str(*token, Special::Tokenize).unwrap());
        }

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

                // sendb new token string back to user
                respond_to.send(output_string).unwrap();

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
    use super::*;

    #[test]
    fn test_generation() {
        println!("running test!");
        let mut actor = ModelActor::from_model_path("./model.bin".to_string()).with_seed(1337);
        actor.run();
        let (tx, rx) = std::sync::mpsc::channel();
        actor.get_completion("Count to five: 1, 2, ".to_string(), tx.clone());
        assert_eq!(rx.recv(), Ok("3".to_string()));
        assert_eq!(rx.recv(), Ok(",".to_string()));
        assert_eq!(rx.recv(), Ok(" ".to_string()));
        assert_eq!(rx.recv(), Ok("4".to_string()));
        assert_eq!(rx.recv(), Ok(",".to_string()));
        assert_eq!(rx.recv(), Ok(" ".to_string()));
        assert_eq!(rx.recv(), Ok("5".to_string()));
    }
}
