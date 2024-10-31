extends NobodyWhoPromptChat

func _ready() -> void:
	run()
	say("Hi there! Who are you?")

func _on_completion_updated(text) -> void:
	print(text)

func _on_completion_finished() -> void:
	print("MODEL FINISHED")
	say("Interesting... Tell me more.")
