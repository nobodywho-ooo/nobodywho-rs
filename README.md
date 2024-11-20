
![Nobody Who](./banner.png)

# Nobody Who‽

NobodyWho is a plugin for the Godot game engine that lets you interact with local LLMs for interactive storytelling.


## How to Install

You can install it from inside the Godot editor: In Godot 4.3+, go to AssetLib and search for "NobodyWho"

...or you can grab a specific version from our github releases page.


## Getting started

The plugin does not include a large language model (LLM). You need to provide an LLM in the GGUF file format. A good place to start is something like [Gemma 2 9B](https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf)

Once you have a GGUF model file, you can add a `NobodyWhoModel` node to your Godot scene. On this node, set the model file to the GGUF model you just downloaded.

`NobodyWhoModel` contains the weights of the model. The model takes up a lot of RAM, and can take a little while to initialize, so if you plan on having several characters/conversations, it's a big advantage to point to the same `NobodyWhoModel` node.

Now you can add a `NobodyWhoPromptChat` node to your scene. From the node inspector, show this chat node where to find the `NobodyWhoModel` node.
Also in the inspector, you can provide a prompt, which gives the LLM instructions on how to carry out the chat.

Now you can add a script to the `NobodyWhoPromptChat` node, to provide your chat interaction.

`NobodyWhoPromptChat` uses this programming interface:
    - `run()`: a function that starts the LLM worker. Must be called before doing anything else.
    - `say(text)`: a function that can be used to send text from the user to the LLM.
    - `completion_updated(text)`: a signal with a string parameter, that is emitted every time the LLM produces more text. Contains rougly one word per invocation.
    - `completion_finished()`: a signal which indicates that the LLM is done speaking.


## Example `NobodyWhoPromptChat` script

```gdscript
extends NobodyWhoPromptChat

func _ready():
    # initializes the LLM worker
    run()

func user_sent_some_text(text):
    # call this function whenever the user submits some new text
    # for example when hitting "enter" in a text input or something like that
    say(text)
    disable_user_input()

func _on_completion_updated(text):
    # attach the completion_updated signal to this function
    # it will be called every time the LLM produces some new text
    show_new_text_on_screen(text)

func _on_completion_finished():
    # attach the completion_finished signal to this function
    # it will be called when the LLM is done producing new text
    enable_user_input()

func show_new_text_on_screen(text):
    # ...omitted. Write your own chat ui code here.

func enable_user_input():
    # ...omitted. Write your own chat ui code here.

func disable_user_input():
    # ...omitted. Write your own chat ui code here.

```
