mod llm;

use godot::classes::INode;
use godot::prelude::*;
use llm::run_worker;
use std::sync::mpsc::{Receiver, Sender};

struct NobodyWhoExtension;

#[gdextension]
unsafe impl ExtensionLibrary for NobodyWhoExtension {}

#[derive(GodotClass)]
#[class(base=Node)]
struct NobodyWhoModel {
    #[export(file)]
    model_path: GString,

    #[export]
    seed: u32,

    model: Option<llm::Model>,
}

#[godot_api]
impl INode for NobodyWhoModel {
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

macro_rules! run_model {
    ($self:ident) => {
        {
            // simple closure that loads the model and returns a result
            // TODO: why does run_result need to be mutable?
            let mut run_result = || -> Result<(), String> {
                // get NobodyWhoModel
                let gd_model_node = $self.model_node.as_mut().ok_or("Model node is not set.")?;
                let nobody_model: GdRef<NobodyWhoModel> = gd_model_node.bind();
                let model: llm::Model = nobody_model.model.clone().ok_or("Could not access NobodyWhoModel.")?;

                // make and store channels for communicating with the llm worker thread
                let (prompt_tx, prompt_rx) = std::sync::mpsc::channel::<String>();
                let (completion_tx, completion_rx) = std::sync::mpsc::channel::<llm::LLMOutput>();
                $self.prompt_tx = Some(prompt_tx);
                $self.completion_rx = Some(completion_rx);

                // start the llm worker
                let seed = nobody_model.seed;
                std::thread::spawn(move || {
                    run_worker(model, prompt_rx, completion_tx, seed);
                });

                Ok(())
            };

            // run it and show error in godot if it fails
            if let Err(msg) = run_result() {
                godot_error!("Error running model: {}", msg);
            }
        }
    };
}

macro_rules! send_text {
    ($self:ident, $text:expr) => {
        if let Some(tx) = $self.prompt_tx.as_ref() {
            tx.send($text).unwrap();
        } else {
            godot_error!("Model not initialized. Call `run` first");
        }
    }
}

macro_rules! emit_tokens {
    ($self:ident) => {
        {
            loop {
                if let Some(rx) = $self.completion_rx.as_ref() {
                    match rx.try_recv() {
                        Ok(llm::LLMOutput::Token(token)) => {
                            $self.base_mut()
                                .emit_signal("completion_updated".into(), &[Variant::from(token)]);
                        }
                        Ok(llm::LLMOutput::Done) => {
                            $self.base_mut()
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
}

#[derive(GodotClass)]
#[class(base=Node)]
struct NobodyWhoPromptCompletion {
    #[export]
    model_node: Option<Gd<NobodyWhoModel>>,

    completion_rx: Option<Receiver<llm::LLMOutput>>,
    prompt_tx: Option<Sender<String>>,

    base: Base<Node>,
}

#[godot_api]
impl INode for NobodyWhoPromptCompletion {
    fn init(base: Base<Node>) -> Self {
        Self {
            model_node: None,
            completion_rx: None,
            prompt_tx: None,
            base,
        }
    }

    fn physics_process(&mut self, _delta: f64) { emit_tokens!(self) }
}

#[godot_api]
impl NobodyWhoPromptCompletion {
    #[func]
    fn run(&mut self) { run_model!(self) }

    #[func]
    fn prompt(&mut self, prompt: String) { send_text!(self, prompt) }

    #[signal]
    fn completion_updated();

    #[signal]
    fn completion_finished();
}

#[derive(GodotClass)]
#[class(base=Node)]
struct NobodyWhoPromptChat {
    #[export]
    model_node: Option<Gd<NobodyWhoModel>>,

    #[export]
    player_name: GString,

    #[export]
    npc_name: GString,

    #[export]
    #[var(hint = MULTILINE_TEXT)]
    prompt: GString,

    prompt_tx: Option<Sender<String>>,
    completion_rx: Option<Receiver<llm::LLMOutput>>,

    base: Base<Node>,
}

#[godot_api]
impl INode for NobodyWhoPromptChat {
    fn init(base: Base<Node>) -> Self {
        Self {
            model_node: None,
            player_name: "Player".into(),
            npc_name: "Character".into(),
            prompt: "".into(),
            prompt_tx: None,
            completion_rx: None,
            base,
        }
    }

    fn physics_process(&mut self, _delta: f64) { emit_tokens!(self) }
}

#[godot_api]
impl NobodyWhoPromptChat {
    #[func]
    fn run(&mut self) { run_model!(self) }

    #[func]
    fn say(&mut self, message: String) {
        // TODO: also send system prompt on first message

        // simple closure that returns Err(String) if something fails
        let say_result = || -> Result<(), String> {
            // get the model instance
            let gd_model_node = self.model_node.as_mut().ok_or("No model node provided. Remember to set a model node on NobodyWhoPromptChat.")?;
            let nobody_model: GdRef<NobodyWhoModel> = gd_model_node.bind();
            let model: llm::Model = nobody_model
                .model
                .clone()
                .ok_or("Could not access LlamaModel from model node.".to_string())?;

            // make a chat string
            let messages = vec![(self.player_name.to_string(), message)];
            let text: String = llm::apply_chat_template(model, messages)?;
            send_text!(self, text);
            Ok::<(), String>(())
        };

        // run it and show the error in godot if it fails
        if let Err(msg) = say_result() {
            godot_error!("Error sending chat message to model worker: {msg}");
        }
    }

    #[signal]
    fn completion_updated();

    #[signal]
    fn completion_finished();
}
