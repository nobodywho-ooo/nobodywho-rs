extends NobodyPrompt


# Called when the node enters the scene tree for the first time.
func _ready():
	self.run()
	prompt("Hello, my name is")
	pass # Replace with function body.


func _on_completion_updated(text):
	print(text)
