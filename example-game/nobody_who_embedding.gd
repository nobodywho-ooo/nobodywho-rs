extends NobodyWhoEmbedding

func _ready():
	run()
	var embedding_vector = await embed("The quick brown fox jumps over the lazy dog")
	print(embedding_vector)
