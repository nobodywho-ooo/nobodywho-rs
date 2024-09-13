use godot::classes::INode;
use godot::prelude::*;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::token::data_array::LlamaTokenDataArray;

use std::num::NonZeroU32;
use std::pin::pin;
use std::sync::mpsc::{Receiver, Sender};

struct NobodyExtension;

fn run_llama_worker(sender: Sender<String>, receiver: Receiver<String>) {
    let n_len = 128;

    let model_path = "model.bin";

    let backend = LlamaBackend::init().expect("Failed to initialize LlamaBackend");

    // Offload the model to the GPU
    let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);

    let model_params = pin!(model_params);

    let model = LlamaModel::load_from_file(&backend, model_path, &model_params).unwrap();

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(2048))
        .with_seed(1234);

    std::thread::spawn(move || {
        let mut ctx = model.new_context(&backend, ctx_params).unwrap();

        loop {
            let prompt = receiver.recv().unwrap();
            let tokens_list = model.str_to_token(&prompt, AddBos::Always).unwrap();

            let n_cxt = ctx.n_ctx() as i32;
            let n_kv_req = tokens_list.len() as i32 + (n_len - tokens_list.len() as i32);

            // make sure the KV cache is big enough to hold all the prompt and generated tokens
            if n_kv_req > n_cxt {
                panic!(
                    "n_kv_req > n_ctx, the required kv cache size is not big enough
     either reduce n_len or increase n_ctx"
                )
            }

            if tokens_list.len() >= n_len.try_into().unwrap() {
                panic!("the prompt is too long, it has more tokens than n_len")
            }

            // print the prompt token-by-token
            for token in &tokens_list {
                godot_print!("{}", model.token_to_str(*token, Special::Tokenize).unwrap());
            }

            // create a llama_batch with size 512
            // we use this object to submit token data for decoding
            let mut batch = LlamaBatch::new(512, 1);

            let last_index: i32 = (tokens_list.len() - 1) as i32;
            for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
                // llama_decode will output logits only for the last token of the prompt
                let is_last = i == last_index;
                batch.add(token, i, &[0], is_last).unwrap();
            }

            ctx.decode(&mut batch).unwrap();

            // main loop
            let mut n_cur = batch.n_tokens();

            // The `Decoder`
            let mut decoder = encoding_rs::UTF_8.new_decoder();

            while n_cur <= n_len {
                // sample the next token
                {
                    let candidates = ctx.candidates_ith(batch.n_tokens() - 1);

                    let candidates_p = LlamaTokenDataArray::from_iter(candidates, false);

                    // sample the most likely token
                    let new_token_id = ctx.sample_token_greedy(candidates_p);

                    // is it an end of stream?
                    if new_token_id == model.token_eos() {
                        godot_print!("");
                        break;
                    }

                    let output_bytes = model
                        .token_to_bytes(new_token_id, Special::Tokenize)
                        .unwrap();

                    // use `Decoder.decode_to_string()` to avoid the intermediate buffer
                    let mut output_string = String::with_capacity(32);
                    let _decode_result =
                        decoder.decode_to_string(&output_bytes, &mut output_string, false);

                    sender.send(output_string).unwrap();

                    batch.clear();
                    batch.add(new_token_id, n_cur, &[0], true).unwrap();
                }

                n_cur += 1;

                ctx.decode(&mut batch).unwrap();
            }
        }
    });
}

#[gdextension]
unsafe impl ExtensionLibrary for NobodyExtension {}

#[derive(GodotClass)]
#[class(base=Node)]
struct NobodyPrompt {
    sender: Sender<String>,
    receiver: Receiver<String>,
    base: Base<Node>,
}

#[godot_api]
impl INode for NobodyPrompt {
    fn init(base: Base<Node>) -> Self {
        let (tx1, rx1) = std::sync::mpsc::channel();
        let (tx2, rx2) = std::sync::mpsc::channel();

        run_llama_worker(tx1, rx2);

        NobodyPrompt {
            sender: tx2,
            receiver: rx1,
            base,
        }
    }

    fn physics_process(&mut self, _delta: f64) {
        if let Ok(token) = self.receiver.try_recv() {
            self.base_mut()
                .emit_signal("completion_updated".into(), &[Variant::from(token)]);
        }
    }
}

#[godot_api]
impl NobodyPrompt {
    #[func]
    fn prompt(&self, prompt: String) {
        self.sender.send(prompt).unwrap();
    }

    #[signal]
    fn completion_updated();

    #[signal]
    fn completion_finished();
}
