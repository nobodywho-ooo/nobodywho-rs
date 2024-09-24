extends NobodyPrompt


# Called when the node enters the scene tree for the first time.
func _ready():
	run()
	prompt("Hello, my name is")

func _on_completion_updated(text):
	print("Got new text from godot signal: " + text)
