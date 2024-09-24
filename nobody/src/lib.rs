use godot::classes::INode;
use godot::prelude::*;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use std::num::NonZeroU32;
use std::pin::pin;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, LazyLock};

static LLAMA_BACKEND: LazyLock<LlamaBackend> =
    LazyLock::new(|| LlamaBackend::init().expect("Failed to initialize llama backend"));

struct NobodyExtension;

#[gdextension]
unsafe impl ExtensionLibrary for NobodyExtension {}

#[derive(GodotClass)]
#[class(base=Node)]
struct NobodyModel {
    #[export(file)]
    model_path: GString,

    #[export]
    seed: u32,

    // TODO: Should be a Option
    model: Arc<LlamaModel>,
}

#[godot_api]
impl INode for NobodyModel {
    fn init(_base: Base<Node>) -> Self {
        // default values to show in godot editor
        let model_path: String = "model.bin".into();

        let seed = 1234;

        let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);

        let model_params = pin!(model_params);

        let model = Arc::new(
            LlamaModel::load_from_file(&LLAMA_BACKEND, model_path.clone(), &model_params).unwrap(),
        );

        Self {
            model_path: model_path.into(),
            seed,
            model,
        }
    }
}

#[derive(GodotClass)]
#[class(base=Node)]
struct NobodyPrompt {
    #[export]
    model_node: Option<Gd<NobodyModel>>,

    rx: Option<Receiver<String>>,
    tx: Option<Sender<String>>,

    base: Base<Node>,
}

#[godot_api]
impl INode for NobodyPrompt {
    fn init(base: Base<Node>) -> Self {
        Self {
            model_node: None,
            rx: None,
            tx: None,
            base,
        }
    }

    fn physics_process(&mut self, _delta: f64) {
        if let Some(rx) = self.rx.as_ref() {
            if let Ok(token) = rx.try_recv() {
                self.base_mut()
                    .emit_signal("completion_updated".into(), &[Variant::from(token)]);
            }
        }
    }
}

#[godot_api]
impl NobodyPrompt {
    #[func]
    fn run(&mut self) {
        if let Some(gd_model_node) = self.model_node.as_mut() {
            let nobody_model: GdRef<NobodyModel> = gd_model_node.bind();

            let model = nobody_model.model.clone();

            let (tx1, rx1) = std::sync::mpsc::channel::<String>();
            let (tx2, rx2) = std::sync::mpsc::channel::<String>();

            self.tx = Some(tx1);
            self.rx = Some(rx2);

            std::thread::spawn(move || {
                let ctx_params = LlamaContextParams::default()
                    .with_n_ctx(NonZeroU32::new(2048))
                    .with_seed(1234);

                let mut ctx = model.new_context(&LLAMA_BACKEND, ctx_params).unwrap();

                while let Ok(prompt) = rx1.recv() {
                    let tokens_list = model.str_to_token(&prompt, AddBos::Always).unwrap();

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
                            let _decode_result = utf8decoder.decode_to_string(
                                &output_bytes,
                                &mut output_string,
                                false,
                            );

                            // sendb new token string back to user
                            tx2.send(output_string).unwrap();

                            // prepare batch or the next decode
                            batch.clear();
                            batch.add(new_token_id, n_cur, &[0], true).unwrap();
                        }

                        n_cur += 1;

                        ctx.decode(&mut batch).unwrap();
                    }
                }
            });
        } else {
            godot_error!("Model node not set");
        }
    }

    #[func]
    fn prompt(&mut self, prompt: String) {
        if let Some(tx) = self.tx.as_ref() {
            tx.send(prompt).unwrap();
        } else {
            godot_error!("Model not initialized. Call `run` first");
        }
    }

    #[signal]
    fn completion_updated();

    #[signal]
    fn completion_finished();
}
