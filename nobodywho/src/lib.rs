mod chat_state;
mod db;
mod llm;

use godot::classes::{INode, ProjectSettings};
use godot::prelude::*;
use llm::{run_worker, SamplerConfig};
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

    #[export]
    use_gpu_if_available: bool,

    model: Option<llm::Model>,
}

#[godot_api]
impl INode for NobodyWhoModel {
    fn init(_base: Base<Node>) -> Self {
        // default values to show in godot editor
        let model_path: String = "model.gguf".into();

        Self {
            model_path: model_path.into(),
            use_gpu_if_available: true,
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

        match llm::get_model(model_path_string.as_str(), self.use_gpu_if_available) {
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

    #[export]
    context_length: u32,

    prompt_tx: Option<Sender<(String, String)>>,
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
            context_length: 4096,
            prompt_tx: None,
            completion_rx: None,
            base,
        }
    }

    fn physics_process(&mut self, _delta: f64) {
        loop {
            if let Some(rx) = self.completion_rx.as_ref() {
                match rx.try_recv() {
                    Ok(llm::LLMOutput::Token(token)) => {
                        self.base_mut()
                            .emit_signal("completion_updated", &[Variant::from(token)]);
                    }
                    Ok(llm::LLMOutput::Done(response)) => {
                        self.base_mut().emit_signal("completion_finished", &[Variant::from(response)]);
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
                        self.completion_rx = None;
                    }
                }
            } else {
                break;
            }
        }
    }
}

#[godot_api]
impl NobodyWhoPromptChat {
    fn get_model(&mut self) -> Option<llm::Model> {
        let gd_model_node = self.model_node.as_mut()?;
        let mut nobody_model = gd_model_node.bind_mut();
        let model: llm::Model = nobody_model.get_model().ok()?;

        Some(model)
    }

    fn get_sampler_config(&mut self) -> SamplerConfig {
        if let Some(gd_sampler) = self.sampler.as_mut() {
            let nobody_sampler: GdRef<NobodyWhoSampler> = gd_sampler.bind();
            nobody_sampler.get_sampler_config()
        } else {
            llm::DEFAULT_SAMPLER_CONFIG
        }
    }

    #[func]
    fn run(&mut self) {
        let mut result = || -> Result<(), String> {
            let model = self.get_model().ok_or("Model not loaded")?;
            let sampler_config = self.get_sampler_config();

            // make and store channels for communicating with the llm worker thread
            let (prompt_tx, prompt_rx) = std::sync::mpsc::channel();
            let (completion_tx, completion_rx) = std::sync::mpsc::channel();
            self.prompt_tx = Some(prompt_tx);
            self.completion_rx = Some(completion_rx);

            // start the llm worker
            let n_ctx = self.context_length;
            std::thread::spawn(move || {
                run_worker(model, prompt_rx, completion_tx, sampler_config, n_ctx);
            });

            // Send the system prompt to the worker
            self.send_message("system".into(), self.prompt.to_string());

            Ok(())
        };

        // run it and show error in godot if it fails
        if let Err(msg) = result() {
            godot_error!("Error running model: {}", msg);
        }
    }

    fn send_message(&mut self, role: String, content: String) {
        if let Some(tx) = self.prompt_tx.as_ref() {
            tx.send((role, content)).unwrap();
        } else {
            godot_error!("Model not initialized. Call `run` first");
        }
    }

    #[func]
    fn say(&mut self, message: String) {
        self.send_message("user".into(), message);
    }

    #[signal]
    fn completion_updated();

    #[signal]
    fn completion_finished();
}
