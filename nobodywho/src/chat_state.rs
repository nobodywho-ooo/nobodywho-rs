use std::sync::LazyLock;

use minijinja::{context, Environment};
use serde::{self, Serialize};

static MINIJINJA_ENV: LazyLock<Environment> = LazyLock::new(|| {
    let mut env = Environment::new();
    env.add_function(
        "raise_exception",
        |msg: String| -> Result<(), minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    env
});

#[derive(Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

pub struct ChatState {
    messages: Vec<Message>,
    chat_template: String,
    length: usize,
}

impl ChatState {
    pub fn new(chat_template: String) -> Self {
        Self {
            messages: Vec::new(),
            chat_template,
            length: 0,
        }
    }

    pub fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push(Message {
            role: role.to_string(),
            content: content.to_string(),
        });
    }

    pub fn render_diff(&mut self) -> Result<String, minijinja::Error> {
        let tmpl = MINIJINJA_ENV.template_from_str(&self.chat_template)?;

        let ctx = context! {
            messages => &self.messages,
            add_generation_prompt => true,
        };

        let text = tmpl.render(ctx)?;

        let diff = text[self.length..].to_string();

        self.length = text.len();

        Ok(diff)
    }
}
