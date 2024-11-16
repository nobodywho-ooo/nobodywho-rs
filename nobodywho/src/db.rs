use godot::prelude::{Variant, VariantType};
use rusqlite::{
    types::{ToSql, ToSqlOutput, Value},
    Error,
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
