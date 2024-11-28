use godot::prelude::*;
use rusqlite::{
    ffi::sqlite3_auto_extension,
    params_from_iter,
    types::{ToSql, ToSqlOutput, Value},
    Connection, Error, Result,
};
use sqlite_vec::sqlite3_vec_init;

pub struct SqlVariant(pub Variant);

impl ToSql for SqlVariant {
    fn to_sql(&self) -> Result<ToSqlOutput<'_>, Error> {
        match self.0.get_type() {
            VariantType::NIL => Ok(ToSqlOutput::Owned(Value::Null)),
            VariantType::BOOL => Ok(ToSqlOutput::from(self.0.to::<bool>())),
            VariantType::INT => Ok(ToSqlOutput::from(self.0.to::<i64>())),
            VariantType::FLOAT => Ok(ToSqlOutput::from(self.0.to::<f64>())),
            VariantType::STRING => Ok(ToSqlOutput::from(self.0.to::<String>())),
            VariantType::PACKED_BYTE_ARRAY => {
                Ok(ToSqlOutput::from(self.0.to::<PackedByteArray>().to_vec()))
            }
            VariantType::PACKED_FLOAT32_ARRAY => Ok(ToSqlOutput::from(
                self.0.to::<PackedFloat32Array>().to_byte_array().to_vec(),
            )),

            _ => Err(Error::ToSqlConversionFailure(Box::new(
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Unsupported Variant type for SQL conversion",
                ),
            ))),
        }
    }
}

#[derive(GodotClass)]
#[class(base=Node)]
pub struct NobodyWhoDB {
    #[export(file = "*.db")]
    db_path: GString,
    conn: Option<Connection>,
    base: Base<Node>,
}

#[godot_api]
impl INode for NobodyWhoDB {
    fn init(base: Base<Node>) -> Self {
        Self {
            db_path: ":memory:".into(), // Default value in Godot editor
            conn: None,
            base,
        }
    }
}

#[godot_api]
impl NobodyWhoDB {
    #[func]
    fn open_db(&mut self) {
        unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())));
        }

        if let Ok(conn) = Connection::open(self.db_path.to_string()) {
            self.conn = Some(conn);
        }
    }

    #[func]
    fn close_db(&mut self) {
        self.conn = None;
    }

    #[func]
    fn execute(&self, query: String, params: VariantArray) -> i64 {
        let mut stmt = self
            .conn
            .as_ref()
            .expect("Database connection is not open. Call open_db() first.")
            .prepare(&query)
            .expect("Failed to prepare query.");

        let params: Vec<SqlVariant> = params.iter_shared().map(SqlVariant).collect();

        let result = stmt
            .execute(params_from_iter(params))
            .expect("Failed to execute query.");

        result as i64
    }

    #[func]
    fn query(&self, query: String, params: VariantArray) -> VariantArray {
        let mut stmt = self
            .conn
            .as_ref()
            .expect("Database connection is not open. Call open_db() first.")
            .prepare(&query)
            .expect("Failed to prepare query.");

        let params: Vec<SqlVariant> = params.iter_shared().map(SqlVariant).collect();

        let column_count = stmt.column_count();
        let mut rows = stmt
            .query(params_from_iter(params))
            .expect("Failed to execute query.");

        let mut result = VariantArray::new();
        while let Some(row) = rows.next().expect("Failed to fetch row.") {
            let mut row_data = VariantArray::new();

            for i in 0..column_count {
                let value = match row.get_unwrap(i) {
                    Value::Null => Variant::nil(),
                    Value::Integer(i) => i.to_variant(),
                    Value::Real(f) => f.to_variant(),
                    Value::Text(s) => s.to_variant(),
                    Value::Blob(b) => PackedByteArray::from(b).to_variant(),
                };
                row_data.push(&value);
            }

            result.push(&row_data.to_variant());
        }

        result
    }
}
