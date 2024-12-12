mod chat_state;
mod db;
mod llm;

use godot::classes::{INode, ProjectSettings};
use godot::prelude::*;
use llm::{run_completion_worker, run_embedding_worker, SamplerConfig};
use std::sync::mpsc::{Receiver, Sender};

struct NobodyWhoExtension;

#[gdextension]
unsafe impl ExtensionLibrary for NobodyWhoExtension {}

#[derive(GodotConvert, Var, Export, Debug, Clone, Copy)]
#[godot(via=GString)]
enum SamplerMethod {
    Greedy,
    Temperature,
    MirostatV2
}

#[derive(GodotClass)]
#[class(tool, base=Resource)]
struct NobodyWhoSampler {
    base: Base<Resource>,

    #[export]
    method: SamplerMethod,

    sampler_config: llm::SamplerConfig
}

#[godot_api]
impl IResource for NobodyWhoSampler {
    fn init(base: Base<Resource>) -> Self {
        match llm::DEFAULT_SAMPLER_CONFIG {
            llm::SamplerConfig::MirostatV2(config) => {
                Self {
                    method: SamplerMethod::MirostatV2,
                    sampler_config: llm::DEFAULT_SAMPLER_CONFIG,
                    base,
                }
            }
        }
    }

    fn get_property_list(&mut self) -> Vec<godot::meta::PropertyInfo> {
        let base_properties = vec![
        ];
        let penalty_properties = vec![
            godot::meta::PropertyInfo::new_export::<i32>("penalty_last_n"),
            godot::meta::PropertyInfo::new_export::<f32>("penalty_repeat"),
            godot::meta::PropertyInfo::new_export::<f32>("penalty_freq"),
            godot::meta::PropertyInfo::new_export::<f32>("penalty_present"),
            godot::meta::PropertyInfo::new_export::<bool>("penalize_nl"),
            godot::meta::PropertyInfo::new_export::<bool>("ignore_eos"),
        ];
        let method_properties = match self.method {
            SamplerMethod::Greedy => vec![],
            SamplerMethod::Temperature => vec![
                godot::meta::PropertyInfo::new_export::<u32>("seed"),
                godot::meta::PropertyInfo::new_export::<f32>("temperature")
            ],
            SamplerMethod::MirostatV2 => vec![
                godot::meta::PropertyInfo::new_export::<u32>("seed"),
                godot::meta::PropertyInfo::new_export::<f32>("tau"),
                godot::meta::PropertyInfo::new_export::<f32>("eta")
            ]
        };
        base_properties.into_iter().chain(penalty_properties).chain(method_properties).collect()
    }

    // fn get_property() {
    // }

    fn set_property(&mut self, property: StringName, value: Variant) -> bool {
        // let mut obj: Gd<Object> = self.base.to_gd().upcast::<Object>();
        // // this line doesn't work:
        // self.base.notify_property_list_changed();
        if property == "method".into() {
            let new_method = SamplerMethod::try_from_variant(&value).expect("Unexpected: Got invalid sampler method"); 
            self.method = new_method;
            self.base.to_gd().upcast::<Object>().notify_property_list_changed();
            return true;
        }
        true
    }
}

impl NobodyWhoSampler {
    pub fn get_sampler_config(&self) -> llm::SamplerConfig {
        llm::DEFAULT_SAMPLER_CONFIG
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
struct NobodyWhoChat {
    #[export]
    model_node: Option<Gd<NobodyWhoModel>>,

    #[export]
    sampler: Option<Gd<NobodyWhoSampler>>,

    #[export]
    #[var(hint = MULTILINE_TEXT)]
    system_prompt: GString,

    #[export]
    context_length: u32,

    prompt_tx: Option<Sender<String>>,
    completion_rx: Option<Receiver<llm::LLMOutput>>,

    base: Base<Node>,
}

#[godot_api]
impl INode for NobodyWhoChat {
    fn init(base: Base<Node>) -> Self {
        Self {
            model_node: None,
            sampler: None,
            system_prompt: "".into(),
            context_length: 4096,
            prompt_tx: None,
            completion_rx: None,
            base,
        }
    }

    fn physics_process(&mut self, _delta: f64) {
        while let Some(rx) = self.completion_rx.as_ref() {
            match rx.try_recv() {
                Ok(llm::LLMOutput::Token(token)) => {
                    self.base_mut()
                        .emit_signal("response_updated", &[Variant::from(token)]);
                }
                Ok(llm::LLMOutput::Done(response)) => {
                    self.base_mut()
                        .emit_signal("response_finished", &[Variant::from(response)]);
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
        }
    }
}

#[godot_api]
impl NobodyWhoChat {
    fn get_model(&mut self) -> Result<llm::Model, String> {
        let gd_model_node = self.model_node.as_mut().ok_or("Model node was not set")?;
        let mut nobody_model = gd_model_node.bind_mut();
        let model: llm::Model = nobody_model.get_model().map_err(|e| e.to_string())?;

        Ok(model)
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
    fn start_worker(&mut self) {
        let mut result = || -> Result<(), String> {
            let model = self.get_model()?;
            let sampler_config = self.get_sampler_config();

            // make and store channels for communicating with the llm worker thread
            let (prompt_tx, prompt_rx) = std::sync::mpsc::channel();
            let (completion_tx, completion_rx) = std::sync::mpsc::channel();
            self.prompt_tx = Some(prompt_tx);
            self.completion_rx = Some(completion_rx);

            // start the llm worker
            let n_ctx = self.context_length;
            let system_prompt = self.system_prompt.to_string();
            std::thread::spawn(move || {
                run_completion_worker(
                    model,
                    prompt_rx,
                    completion_tx,
                    sampler_config,
                    n_ctx,
                    system_prompt,
                );
            });

            Ok(())
        };

        // run it and show error in godot if it fails
        if let Err(msg) = result() {
            godot_error!("Error running model: {}", msg);
        }
    }

    fn send_message(&mut self, content: String) {
        if let Some(tx) = self.prompt_tx.as_ref() {
            tx.send(content).unwrap();
        } else {
            godot_warn!("Worker was not started yet, starting now... You may want to call `start_worker()` ahead of time to avoid waiting.");
            self.start_worker();
            self.send_message(content);
        }
    }

    #[func]
    fn say(&mut self, message: String) {
        self.send_message(message);
    }

    #[signal]
    fn response_updated(new_token: String);

    #[signal]
    fn response_finished(response: String);
}

#[derive(GodotClass)]
#[class(base=Node)]
struct NobodyWhoEmbedding {
    #[export]
    model_node: Option<Gd<NobodyWhoModel>>,

    text_tx: Option<Sender<String>>,
    embedding_rx: Option<Receiver<llm::EmbeddingsOutput>>,
    base: Base<Node>,
}

#[godot_api]
impl INode for NobodyWhoEmbedding {
    fn init(base: Base<Node>) -> Self {
        Self {
            model_node: None,
            text_tx: None,
            embedding_rx: None,
            base,
        }
    }

    fn physics_process(&mut self, _delta: f64) {
        while let Some(rx) = self.embedding_rx.as_ref() {
            match rx.try_recv() {
                Ok(llm::EmbeddingsOutput::FatalError(errmsg)) => {
                    godot_error!("Embeddings worker crashed: {errmsg}");
                    self.embedding_rx = None; // un-set here to avoid spamming error message
                }
                Ok(llm::EmbeddingsOutput::Embedding(embd)) => {
                    self.base_mut().emit_signal(
                        "embedding_finished",
                        &[PackedFloat32Array::from(embd).to_variant()],
                    );
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // got nothing yet - no worries
                    break;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    godot_error!("Unexpected: Embeddings worker channel disconnected");
                    self.embedding_rx = None; // un-set here to avoid spamming error message
                }
            }
        }
    }
}

#[godot_api]
impl NobodyWhoEmbedding {
    #[signal]
    fn embedding_finished(embedding: PackedFloat32Array);

    fn get_model(&mut self) -> Result<llm::Model, String> {
        let gd_model_node = self.model_node.as_mut().ok_or("Model node was not set")?;
        let mut nobody_model = gd_model_node.bind_mut();
        let model: llm::Model = nobody_model.get_model().map_err(|e| e.to_string())?;

        Ok(model)
    }

    #[func]
    fn start_worker(&mut self) {
        let mut result = || -> Result<(), String> {
            let model = self.get_model()?;

            // make and store channels for communicating with the llm worker thread
            let (embedding_tx, embedding_rx) = std::sync::mpsc::channel();
            let (text_tx, text_rx) = std::sync::mpsc::channel();
            self.embedding_rx = Some(embedding_rx);
            self.text_tx = Some(text_tx);

            // start the llm worker
            std::thread::spawn(move || {
                run_embedding_worker(model, text_rx, embedding_tx);
            });

            Ok(())
        };

        // run it and show error in godot if it fails
        if let Err(msg) = result() {
            godot_error!("Error running model: {}", msg);
        }
    }

    #[func]
    fn embed(&mut self, text: String) -> Signal {
        // returns signal, so that you can `var vec = await embed("Hello, world!")`
        if let Some(tx) = &self.text_tx {
            if tx.send(text).is_err() {
                godot_error!("Embedding worker died.");
            }
        } else {
            godot_warn!("Worker was not started yet, starting now... You may want to call `start_worker()` ahead of time to avoid waiting.");
            self.start_worker();
            return self.embed(text);
        };

        return godot::builtin::Signal::from_object_signal(&self.base_mut(), "embedding_finished");
    }

    #[func]
    fn cosine_similarity(a: PackedFloat32Array, b: PackedFloat32Array) -> f32 {
        llm::cosine_similarity(a.as_slice(), b.as_slice())
    }
}
