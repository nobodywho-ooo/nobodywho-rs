mod model;

use godot::classes::INode;
use godot::prelude::*;

use std::sync::mpsc::{Receiver, Sender};

use crate::model::ModelActor;

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

    model_actor: Option<ModelActor>,
}

#[godot_api]
impl INode for NobodyModel {
    fn init(_base: Base<Node>) -> Self {
        // default values to show in godot editor
        let model_path = "model.bin".into();
        let seed = 1234;
        Self {
            model_path,
            seed,
            model_actor: None,
        }
    }
}

#[godot_api]
impl NobodyModel {
    #[func]
    fn run(&mut self) {
        let mut model_actor =
            ModelActor::from_model_path(self.model_path.to_string()).with_seed(self.seed);
        model_actor.run();
        self.model_actor = Some(model_actor);
    }
}

#[derive(GodotClass)]
#[class(base=Node)]
struct NobodyPrompt {
    #[export]
    model_node: Option<Gd<NobodyModel>>,

    // channels for communicating with ModelActor
    rx: Receiver<String>,
    tx: Sender<String>,

    base: Base<Node>,
}

#[godot_api]
impl INode for NobodyPrompt {
    fn init(base: Base<Node>) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        Self {
            model_node: None,
            rx,
            tx,
            base,
        }
    }

    fn physics_process(&mut self, _delta: f64) {
        if let Ok(token) = self.rx.try_recv() {
            self.base_mut()
                .emit_signal("completion_updated".into(), &[Variant::from(token)]);
        }
    }
}

#[godot_api]
impl NobodyPrompt {
    #[func]
    fn prompt(&mut self, prompt: String) {
        match self.model_node {
            Some(ref mut gd_model_node) => {
                let mut nobody_model: GdMut<NobodyModel> = gd_model_node.bind_mut();
                let model_actor = match &nobody_model.model_actor {
                    Some(model_actor) => model_actor,
                    None => {
                        godot_warn!("Model was not loaded yet. Loading now... Call the .run() method on NobodyModel to control when to load the model.");
                        nobody_model.run();
                        if let Some(model_actor) = &nobody_model.model_actor {
                            model_actor
                        } else {
                            panic!("Unexpected: nobody_model.model_actor is still None after calling run()");
                        }
                    }
                };
                model_actor.get_completion(prompt, self.tx.clone());
            }
            None => {
                godot_error!("you must set a model node first");
            }
        }
    }

    #[signal]
    fn completion_updated();

    #[signal]
    fn completion_finished();
}
