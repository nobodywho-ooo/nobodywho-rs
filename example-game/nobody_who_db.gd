extends NobodyWhoDB


# Called when the node enters the scene tree for the first time.
func _ready():
    self.open_db();

    print(self.query("SELECT sqlite_version(), vec_version();", []))

    print(self.execute("CREATE TABLE IF NOT EXISTS texts (id INTEGER PRIMARY KEY, name TEXT);", []));
    print(self.execute("INSERT INTO texts (name) VALUES (\"Hello, World!\"), (\"Goodbye, World!\");", []));

    print(self.query("SELECT * FROM texts;", []));

    self.close_db();
