extends NobodyWhoChat

func _ready() -> void:
	say("Hi there! Who are you?")

func _on_response_updated(text: String) -> void:
	print(text)

func _on_response_finished(response: String) -> void:
	print("MODEL FINISHED")
	say("Interesting... Tell me more.")
