use std::num::NonZeroU32;

use godot::classes::INode;
use godot::prelude::*;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::token::data_array::LlamaTokenDataArray;

struct NobodyExtension;

#[gdextension]
unsafe impl ExtensionLibrary for NobodyExtension {}

#[derive(GodotClass)]
#[class(base=Node)]
struct NobodyModel {
    model_path: String,
    seed: u32,
}

#[godot_api]
impl INode for NobodyModel {
    fn init(_base: Base<Node>) -> Self {
        let model_path = "model.bin";
        let seed = 1234;

        Self {
            model_path: model_path.into(),
            seed,
        }
    }
}

fn model_worker(model_path: String, seed: u32) {
    let backend = LlamaBackend::init().unwrap();

    // Metal baby (all layers on GPU)
    let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);
    let model = LlamaModel::load_from_file(&backend, model_path.clone(), &model_params).unwrap();

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(2048).unwrap()))
        .with_seed(seed);

    let ctx = model.new_context(&backend, ctx_params).unwrap();

    // XXX: get text from client
    let prompt = "Hi there!";
    let tokens_list = model
        .str_to_token(&prompt.to_string(), AddBos::Always)
        .unwrap();

    if tokens_list.len() >= ctx.n_ctx() as usize {
        // TODO: don't die so hard
        panic!(
            "The given text is longer than the allowed context size ({} > {})",
            tokens_list.len(),
            ctx.n_ctx()
        );
    }

    // init
    let max_sequences = 1;
    let mut batch = LlamaBatch::new(tokens_list.len(), max_sequences);

    let last_index: i32 = (tokens_list.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        // llama_decode will output logits only for the last token of the prompt
        let is_last = i == last_index;
        batch.add(token, i, &[0], is_last).unwrap();
    }
}

impl NobodyModel {
    fn load_model(&self) {}
}

#[derive(GodotClass)]
#[class(base=Node)]
struct NobodyPrompt {
    #[export]
    model_node: Option<Gd<NobodyModel>>,
}

#[godot_api]
impl INode for NobodyPrompt {
    fn init(_base: Base<Node>) -> Self {
        godot_print!("Hello, world!"); // Prints to the Godot console

        let model_node = None;

        Self { model_node }
    }
}

#[godot_api]
impl NobodyPrompt {
    #[func]
    fn say(&self, prompt: GString) {
        // tokenize the prompt

        if tokens_list.len() >= usize::try_from(n_len).unwrap() {
            panic!("the prompt is too long, it has more tokens than n_len")
        }

        for token in &tokens_list {
            godot_print!(
                "{}",
                self.model.token_to_str(*token, Special::Tokenize).unwrap()
            );
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
        let mut n_decode = 0;

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
                if new_token_id == self.model.token_eos() {
                    godot_print!("End of stream");
                    break;
                }

                let output_bytes = self
                    .model
                    .token_to_bytes(new_token_id, Special::Tokenize)
                    .unwrap();

                // use `Decoder.decode_to_string()` to avoid the intermediate buffer
                let mut output_string = String::with_capacity(32);
                let _decode_result =
                    decoder.decode_to_string(&output_bytes, &mut output_string, false);
                godot_print!("{output_string}");

                batch.clear();
                batch.add(new_token_id, n_cur, &[0], true).unwrap();
            }

            n_cur += 1;

            ctx.decode(&mut batch).unwrap();

            n_decode += 1;
        }
    }
}
