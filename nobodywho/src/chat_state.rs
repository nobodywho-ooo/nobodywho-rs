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
    env.add_function("strftime_now", strftime_now);
    env
});

fn strftime_now(format_str: &str) -> String {
    chrono::Local::now().format(format_str).to_string()
}

#[derive(Serialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

pub struct ChatState {
    messages: Vec<Message>,
    chat_template: String,
    length: usize,
    eos_token: String,
    bos_token: String
}

/// given a chat history where the first two messages are from system and user
/// return a history where the first message is from user, and contains the system prompt as well.
/// (this is what llama.cpp does for the gemma template too)
fn concat_system_and_first_user_messages(messages: &[Message]) -> Result<Vec<Message>, minijinja::Error> {
    if messages.len() < 2 || messages[0].role != "system" || messages[1].role != "user" {
        // HACK: this should probably be a custom ChatStateError, and nont a minijinja error
        //       but this was quick and easy rn, and we "abuse" the minijinja errors for
        //       `raise_exception` anyway...
        return Err(minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            "Cannot replace system prompt unless the first two messages are from system and user roles."
        ))
    }
    let new_first_message = Message {
        role: "user".to_string(),
        content: format!("{}\n\n{}", messages[0].content, messages[1].content)
    };
    let new_messages = vec![new_first_message]
        .into_iter()
        .chain(messages[2..].iter().cloned())
        .collect();
    Ok(new_messages)
}

impl ChatState {
    pub fn new(chat_template: String, bos_token: String, eos_token: String) -> Self {
        Self {
            messages: Vec::new(),
            chat_template,
            length: 0,
            eos_token,
            bos_token,
        }
    }

    pub fn add_message(&mut self, role: String, content: String) {
        self.messages.push(Message { role, content });
    }

    fn render(&mut self) -> Result<String, minijinja::Error> {
        let tmpl = MINIJINJA_ENV.template_from_str(&self.chat_template)?;

        let ctx = context! {
            messages => &self.messages,
            add_generation_prompt => true,
            eos_token => self.eos_token,
            bos_token => self.bos_token,
        };

        match tmpl.render(ctx) {
            Ok(rendered) => Ok(rendered),
            Err(err) => match err.kind() {
                minijinja::ErrorKind::InvalidOperation => {
                    if err.to_string().contains("System role not supported") {
                        // this is the error message we get when rendering the gemma2
                        // concat the first two messages and try again
                        self.messages = concat_system_and_first_user_messages(&self.messages)?;
                        self.render()
                    } else {
                        Err(err)
                    }
                },
                _ => Err(err)
            },
        }
    }

    pub fn render_diff(&mut self) -> Result<String, minijinja::Error> {
        // render the full template
        let text = self.render()?;

        // get the chars that are new since the last template render
        let diff = text[self.length..].to_string();

        // note the length of this template render
        self.length = text.len();

        Ok(diff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama31_template() {
        // test that llama 3.1 template renders
        let template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}";
        let mut chatstate = ChatState::new(template.into(), "<|bos|>".into(), "<|eos|>".into());
        chatstate.add_message("user".into(), "Hello, world!".into());
        let rendered = chatstate.render_diff().unwrap();
        let expected = "<|bos|><|start_header_id|>user<|end_header_id|>

Hello, world!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

";
        assert_eq!(rendered, expected)
    }

    #[test]
    fn test_strftime_now() {
        // huggingface chat template docs say that `strftime_now(format_str)` should be equivalent to `datetime.now().strftime(format_str)`
        // https://huggingface.co/docs/transformers/main/chat_templating#callable-functions

        let result = strftime_now("%Y-%m-%d");
        assert!(result.len() == 10, "Expected format YYYY-MM-DD to be 10 chars");

        let result = strftime_now("%H:%M:%S");
        assert!(result.len() == 8, "Expected format HH:MM:SS to be 8 chars");
    }
}
