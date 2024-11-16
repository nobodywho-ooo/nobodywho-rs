use godot::prelude::*;
use rusqlite::{
    types::{ToSql, ToSqlOutput, Value},
    Connection, Error,
};

pub struct SqlVariant(pub Variant);

impl ToSql for SqlVariant {
    fn to_sql(&self) -> Result<ToSqlOutput<'_>, Error> {
        match self.0.get_type() {
            VariantType::NIL => Ok(ToSqlOutput::Owned(Value::Null)),
            VariantType::BOOL => {
                let val: bool = self.0.to();
                Ok(ToSqlOutput::from(val))
            }
            VariantType::INT => {
                let val: i64 = self.0.to();
                Ok(ToSqlOutput::from(val))
            }
            VariantType::FLOAT => {
                let val: f64 = self.0.to();
                Ok(ToSqlOutput::from(val))
            }
            VariantType::STRING => {
                let val: String = self.0.to();
                Ok(ToSqlOutput::from(val))
            }
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
