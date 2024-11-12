extends NobodyWhoEmbedding

func _ready():
	run()
	embed("The quick brown fox jumps over the lazy dog")

func _on_embedding_finished(embedding):
	print(embedding)
