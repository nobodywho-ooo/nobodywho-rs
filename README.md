![Nobody Who](./assets/banner.png)

[![Discord](https://img.shields.io/discord/1308812521456799765?logo=discord&style=flat-square)](https://discord.gg/qhaMc2qCYB)
[![Matrix](https://img.shields.io/badge/Matrix-000?logo=matrix&logoColor=fff)](https://matrix.to/#/#nobodywho:matrix.org)
[![Mastodon](https://img.shields.io/badge/Mastodon-6364FF?logo=mastodon&logoColor=fff&style=flat-square)](https://mastodon.gamedev.place/@nobodywho)
[![Godot Engine](https://img.shields.io/badge/Godot-%23FFFFFF.svg?logo=godot-engine)](https://godotengine.org/asset-library/asset/2886)

NobodyWho is a plugin for the Godot game engine that lets you interact with local LLMs for interactive storytelling.


## How to Install

You can install it from inside the Godot editor: In Godot 4.3+, go to AssetLib and search for "NobodyWho".

...or you can grab a specific version from our [github releases page.](https://github.com/nobodywho-ooo/nobodywho/releases) You can install these zip files by going to the "AssetLib" tab in Godot and selecting "Import".


## Getting started

The plugin does not include a large language model (LLM). You need to provide an LLM in the GGUF file format. A good place to start is something like [Gemma 2 2B](https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf)

Once you have a GGUF model file, you can add a `NobodyWhoModel` node to your Godot scene. On this node, set the model file to the GGUF model you just downloaded.

`NobodyWhoModel` contains the weights of the model. The model takes up a lot of RAM, and can take a little while to initialize, so if you plan on having several characters/conversations, it's a big advantage to point to the same `NobodyWhoModel` node.

Now you can add a `NobodyWhoChat` node to your scene. From the node inspector, set the "Model Node" field, to show this chat node where to find the `NobodyWhoModel`.
Also in the inspector, you can provide a prompt, which gives the LLM instructions on how to carry out the chat.

Now you can add a script to the `NobodyWhoChat` node, to provide your chat interaction.

`NobodyWhoChat` uses this programming interface:

- `say(text: String)`: a function that can be used to send text from the user to the LLM.
- `response_updated(token: String)`: a signal that is emitted every time the LLM produces more text. Contains roughly one word per invocation.
- `response_finished(response: String)`: a signal which indicates that the LLM is done speaking.
- `start_worker()`: a function that starts the LLM worker. The LLM needs a few seconds to get ready before chatting, so you may want to call this ahead of time.


## Example `NobodyWhoChat` script

```gdscript
extends NobodyWhoChat

func _ready():
	# configure node
	model_node = get_node("../ChatModel")
	system_prompt = "You are an evil wizard. Always try to curse anyone who talks to you."

	# say something
	say("Hi there! Who are you?")

	# wait for the response
	var response = await response_finished
	print("Got response: " + response)

    # in this example we just use the `response_finished` signal to get the complete response
    # in real-world-use you definitely want to connect `response_updated`, which gives one word at a time
    # the whole interaction feels *much* smoother if you stream the response out word-by-word.
```


## Example `NobodyWhoEmbedding` script

```gdscript
extends NobodyWhoEmbedding

func _ready():
    # configure node
    self.model_node = get_node("../EmbeddingModel")

    # generate some embeddings
    embed("The dragon is on the hill.")
    var dragon_hill_embd = await self.embedding_finished

    embed("The dragon is hungry for humans.")
    var dragon_hungry_embd = await self.embedding_finished

    embed("This doesn't matter.")
    var irrelevant_embd = await self.embedding_finished

    # test similarity,
    # here we show that two embeddings will have high similarity, if they mean similar things
    var low_similarity = cosine_similarity(irrelevant_embd, dragon_hill_embd)
    var high_similarity = cosine_similarity(dragon_hill_embd, dragon_hungry_embd) 
    assert(low_similarity < high_similarity)
```
