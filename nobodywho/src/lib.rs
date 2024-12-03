mod db;
mod llm;

use godot::classes::{INode, ProjectSettings};
use godot::prelude::*;
use llm::run_worker;
use std::sync::mpsc::{Receiver, Sender};

struct NobodyWhoExtension;

#[gdextension]
unsafe impl ExtensionLibrary for NobodyWhoExtension {}

#[derive(GodotClass)]
#[class(tool, base=Resource)]
struct NobodyWhoSampler {
    base: Base<Resource>,

    #[export]
    seed: u32,
    #[export]
    temperature: f32,
    #[export]
    penalty_last_n: i32,
    #[export]
    penalty_repeat: f32,
    #[export]
    penalty_freq: f32,
    #[export]
    penalty_present: f32,
    #[export]
    penalize_nl: bool,
    #[export]
    ignore_eos: bool,
    #[export]
    mirostat_tau: f32,
    #[export]
    mirostat_eta: f32,
}

#[godot_api]
impl IResource for NobodyWhoSampler {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            seed: llm::DEFAULT_SAMPLER_CONFIG.seed,
            temperature: llm::DEFAULT_SAMPLER_CONFIG.temperature,
            penalty_last_n: llm::DEFAULT_SAMPLER_CONFIG.penalty_last_n,
            penalty_repeat: llm::DEFAULT_SAMPLER_CONFIG.penalty_repeat,
            penalty_freq: llm::DEFAULT_SAMPLER_CONFIG.penalty_freq,
            penalty_present: llm::DEFAULT_SAMPLER_CONFIG.penalty_present,
            penalize_nl: llm::DEFAULT_SAMPLER_CONFIG.penalize_nl,
            ignore_eos: llm::DEFAULT_SAMPLER_CONFIG.ignore_eos,
            mirostat_tau: llm::DEFAULT_SAMPLER_CONFIG.mirostat_tau,
            mirostat_eta: llm::DEFAULT_SAMPLER_CONFIG.mirostat_eta,
        }
    }
}

impl NobodyWhoSampler {
    pub fn get_sampler_config(&self) -> llm::SamplerConfig {
        llm::SamplerConfig {
            seed: self.seed,
            temperature: self.temperature,
            penalty_last_n: self.penalty_last_n,
            penalty_repeat: self.penalty_repeat,
            penalty_freq: self.penalty_freq,
            penalty_present: self.penalty_present,
            penalize_nl: self.penalize_nl,
            ignore_eos: self.ignore_eos,
            mirostat_tau: self.mirostat_tau,
            mirostat_eta: self.mirostat_eta,
        }
    }
}

#[derive(GodotClass)]
#[class(base=Node)]
struct NobodyWhoModel {
    #[export(file = "*.gguf")]
    model_path: GString,

    model: Option<llm::Model>,
}

#[godot_api]
impl INode for NobodyWhoModel {
    fn init(_base: Base<Node>) -> Self {
        // default values to show in godot editor
        let model_path: String = "model.gguf".into();

        Self {
            model_path: model_path.into(),
            model: None,
        }
    }
}

impl NobodyWhoModel {
    // memoized model loader
    fn get_model(&mut self) -> Result<llm::Model, llm::LoadModelError> {
        if let Some(model) = &self.model {
            return Ok(model.clone());
        }

        let project_settings = ProjectSettings::singleton();
        let model_path_string: String = project_settings
            .globalize_path(&self.model_path.clone())
            .into();

        match llm::get_model(model_path_string.as_str()) {
            Ok(model) => {
                self.model = Some(model.clone());
                Ok(model.clone())
            }
            Err(err) => {
                godot_error!("Could not load model: {:?}", err.to_string());
                Err(err)
            }
        }
    }
}

macro_rules! run_model {
    ($self:ident) => {{
        // simple closure that loads the model and returns a result
        // TODO: why does run_result need to be mutable?
        let mut run_result = || -> Result<(), String> {
            // get NobodyWhoModel
            let gd_model_node = $self.model_node.as_mut().ok_or("Model node is not set.")?;
            let mut nobody_model = gd_model_node.bind_mut();
            let model: llm::Model = nobody_model.get_model().map_err(|e| e.to_string())?;

            // get NobodyWhoSampler
            let sampler_config: llm::SamplerConfig =
                if let Some(gd_sampler) = $self.sampler.as_mut() {
                    let nobody_sampler: GdRef<NobodyWhoSampler> = gd_sampler.bind();
                    nobody_sampler.get_sampler_config()
                } else {
                    llm::DEFAULT_SAMPLER_CONFIG
                };

            // make and store channels for communicating with the llm worker thread
            let (prompt_tx, prompt_rx) = std::sync::mpsc::channel::<String>();
            let (completion_tx, completion_rx) = std::sync::mpsc::channel::<llm::LLMOutput>();
            $self.prompt_tx = Some(prompt_tx);
            $self.completion_rx = Some(completion_rx);

            // start the llm worker
            let n_ctx = $self.context_length;
            std::thread::spawn(move || {
                run_worker(model, prompt_rx, completion_tx, sampler_config, n_ctx);
            });

            Ok(())
        };

        // run it and show error in godot if it fails
        if let Err(msg) = run_result() {
            godot_error!("Error running model: {}", msg);
        }
    }};
}

macro_rules! send_text {
    ($self:ident, $text:expr) => {
        if let Some(tx) = $self.prompt_tx.as_ref() {
            tx.send($text).unwrap();
        } else {
            godot_error!("Model not initialized. Call `run` first");
        }
    };
}

macro_rules! emit_tokens {
    ($self:ident) => {{
        loop {
            if let Some(rx) = $self.completion_rx.as_ref() {
                match rx.try_recv() {
                    Ok(llm::LLMOutput::Token(token)) => {
                        $self
                            .base_mut()
                            .emit_signal("completion_updated", &[Variant::from(token)]);
                    }
                    Ok(llm::LLMOutput::Done) => {
                        $self.base_mut().emit_signal("completion_finished", &[]);
                    }
                    Ok(llm::LLMOutput::FatalErr(msg)) => {
                        godot_error!("Model worker crashed: {msg}");
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        break;
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        godot_error!("Model output channel died. Did the LLM worker crash?");
                        // set hanging channel to None
                        // this prevents repeating the dead channel error message foreve
                        $self.completion_rx = None;
                    }
                }
            } else {
                break;
            }
        }
    }};
}

#[derive(GodotClass)]
#[class(base=Node)]
struct NobodyWhoPromptCompletion {
    #[export]
    model_node: Option<Gd<NobodyWhoModel>>,

    #[export]
    sampler: Option<Gd<NobodyWhoSampler>>,

    #[export]
    context_length: u32,

    completion_rx: Option<Receiver<llm::LLMOutput>>,
    prompt_tx: Option<Sender<String>>,

    base: Base<Node>,
}

#[godot_api]
impl INode for NobodyWhoPromptCompletion {
    fn init(base: Base<Node>) -> Self {
        Self {
            model_node: None,
            sampler: None,
            context_length: 4096,
            completion_rx: None,
            prompt_tx: None,
            base,
        }
    }

    fn physics_process(&mut self, _delta: f64) {
        emit_tokens!(self)
    }
}

#[godot_api]
impl NobodyWhoPromptCompletion {
    #[func]
    fn run(&mut self) {
        run_model!(self)
    }

    #[func]
    fn prompt(&mut self, prompt: String) {
        send_text!(self, prompt)
    }

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
    sampler: Option<Gd<NobodyWhoSampler>>,

    #[export]
    #[var(hint = MULTILINE_TEXT)]
    prompt: GString,
    sent_prompt: bool,

    #[export]
    context_length: u32,

    prompt_tx: Option<Sender<String>>,
    completion_rx: Option<Receiver<llm::LLMOutput>>,

    base: Base<Node>,
}

#[godot_api]
impl INode for NobodyWhoPromptChat {
    fn init(base: Base<Node>) -> Self {
        Self {
            model_node: None,
            sampler: None,
            prompt: "".into(),
            sent_prompt: false,
            context_length: 4096,
            prompt_tx: None,
            completion_rx: None,
            base,
        }
    }

    fn physics_process(&mut self, _delta: f64) {
        emit_tokens!(self)
    }
}

#[godot_api]
impl NobodyWhoPromptChat {
    #[func]
    fn run(&mut self) {
        run_model!(self)
    }

    #[func]
    fn say(&mut self, message: String) {
        // simple closure that returns Err(String) if something fails
        let say_result = || -> Result<(), String> {
            // get the model instance
            let gd_model_node = self.model_node.as_mut().ok_or(
                "No model node provided. Remember to set a model node on NobodyWhoPromptChat.",
            )?;
            let nobody_model: GdRef<NobodyWhoModel> = gd_model_node.bind();
            let model: llm::Model = nobody_model
                .model
                .clone()
                .ok_or("Could not access LlamaModel from model node.".to_string())?;

            // make a chat string
            let mut messages: Vec<(String, String)> = vec![];
            if !self.sent_prompt {
                messages.push(("system".into(), (&self.prompt).into()));
                self.sent_prompt = true;
            }
            messages.push(("user".to_string(), message));
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
