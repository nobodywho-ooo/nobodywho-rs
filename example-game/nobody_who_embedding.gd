extends NobodyWhoEmbedding

func _ready():
	print("doing embed")
	var copenhagen_embd = await embed("Copenhagen is the capital of Denmark.")
	var berlin_embd = await embed("Berlin is the capital of Germany.")
	var insult_embd = await embed("Your mother was a hamster, and your father smelt of elderberries.")
	assert(cosine_similarity(copenhagen_embd, berlin_embd) > cosine_similarity(copenhagen_embd, insult_embd))
	print("embed works")
