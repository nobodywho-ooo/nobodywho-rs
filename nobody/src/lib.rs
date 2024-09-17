mod model;

use godot::classes::INode;
use godot::prelude::*;

use std::sync::mpsc::{Receiver, Sender};

use crate::model::ModelActorHandle;

struct NobodyExtension;

#[gdextension]
unsafe impl ExtensionLibrary for NobodyExtension {}

#[derive(GodotClass)]
#[class(base=Node)]
struct NobodyPrompt {
    model_actor: ModelActorHandle,

    rx: Receiver<String>,
    tx: Sender<String>,

    base: Base<Node>,
}

#[godot_api]
impl INode for NobodyPrompt {
    fn init(base: Base<Node>) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        let model_actor =
            ModelActorHandle::from_model_path_and_seed("./model.bin".to_string(), 1234);

        Self {
            model_actor,
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
    fn prompt(&self, prompt: String) {
        self.model_actor.get_completion(prompt, self.tx.clone());
    }

    #[signal]
    fn completion_updated();

    #[signal]
    fn completion_finished();
}
