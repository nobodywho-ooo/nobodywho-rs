mod llm;

use godot::classes::INode;
use godot::prelude::*;
use llama_cpp_2::model::LlamaModel;
use llm::run_worker;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

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

    model: Option<Arc<LlamaModel>>,
}

#[godot_api]
impl INode for NobodyModel {
    fn init(_base: Base<Node>) -> Self {
        // default values to show in godot editor
        let model_path: String = "model.bin".into();

        let seed = 1234;

        Self {
            model_path: model_path.into(),
            model: None,
            seed,
        }
    }

    fn ready(&mut self) {
        let model_path_string: String = self.model_path.clone().into();
        self.model = Some(llm::get_model(model_path_string.as_str()));
    }
}

#[derive(GodotClass)]
#[class(base=Node)]
struct NobodyPrompt {
    #[export]
    model_node: Option<Gd<NobodyModel>>,

    completion_rx: Option<Receiver<llm::LLMOutput>>,
    prompt_tx: Option<Sender<String>>,

    base: Base<Node>,
}

#[godot_api]
impl INode for NobodyPrompt {
    fn init(base: Base<Node>) -> Self {
        Self {
            model_node: None,
            completion_rx: None,
            prompt_tx: None,
            base,
        }
    }

    fn physics_process(&mut self, _delta: f64) {
        loop {
            if let Some(rx) = self.completion_rx.as_ref() {
                match rx.try_recv() {
                    Ok(llm::LLMOutput::Token(token)) => {
                        self.base_mut()
                            .emit_signal("completion_updated".into(), &[Variant::from(token)]);
                    }
                    Ok(llm::LLMOutput::Done) => {
                        self.base_mut()
                            .emit_signal("completion_finished".into(), &[]);
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        break;
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        godot_error!("Unexpected: Model channel disconnected");
                    }
                }
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
            if let Some(model) = nobody_model.model.clone() {
                let (prompt_tx, prompt_rx) = std::sync::mpsc::channel::<String>();
                let (completion_tx, completion_rx) = std::sync::mpsc::channel::<llm::LLMOutput>();

                self.prompt_tx = Some(prompt_tx);
                self.completion_rx = Some(completion_rx);

                let seed = nobody_model.seed;
                std::thread::spawn(move || {
                    run_worker(model, prompt_rx, completion_tx, seed);
                });
            } else {
                godot_error!("Unexpected: Model node is not ready yet.");
            }
        } else {
            godot_error!("Model node not set");
        }
    }

    #[func]
    fn prompt(&mut self, prompt: String) {
        if let Some(tx) = self.prompt_tx.as_ref() {
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
