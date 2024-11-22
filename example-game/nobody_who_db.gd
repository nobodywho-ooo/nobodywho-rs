extends NobodyWhoDB


# Called when the node enters the scene tree for the first time.
func _ready():
	open_db()

	print(query("SELECT sqlite_version(), vec_version();", []))

	execute("CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[4]);", []);

	var items = [
		[1, PackedFloat32Array([0.1, 0.1, 0.1, 0.1])],
		[2, PackedFloat32Array([0.2, 0.2, 0.2, 0.2])],
		[3, PackedFloat32Array([0.3, 0.3, 0.3, 0.3])],
		[4, PackedFloat32Array([0.4, 0.4, 0.4, 0.4])],
		[5, PackedFloat32Array([0.5, 0.5, 0.5, 0.5])],
	]

	for item in items:
		execute("INSERT INTO vec_items(rowid, embedding) VALUES (?, ?);", item);

	var knn_query = """
		SELECT rowid, distance 
		FROM vec_items 
		WHERE embedding MATCH ?1 
		ORDER BY distance 
		LIMIT 3;
	"""
	var search_vector = PackedFloat32Array([0.3, 0.3, 0.3, 0.3]);

	print(query(knn_query, [search_vector]))

	self.close_db();
