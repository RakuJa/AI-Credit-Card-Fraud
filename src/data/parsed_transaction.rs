use std::fmt::Display;

use uuid::Uuid;

use crate::data::transaction::Transaction;

#[derive(Debug, Clone)]
pub struct ParsedTransaction {
    pub uuid: Uuid,
    pub amount: f32,
    pub time: f32,
    pub is_fraud: bool,
    pub certainty: f32,
    pub count: i64,
}

impl ParsedTransaction {
    pub fn default() -> Self {
        Self {
            uuid: Uuid::default(),
            amount: 0.0,
            time: 0.0,
            is_fraud: false,
            certainty: 0.,
            count: 0,
        }
    }
}

impl Display for ParsedTransaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "UUID: {}\nAmount: {}\nTime: {}\nCertainty: {:.3}% \nCount: {}",
            self.uuid, self.amount, self.time, self.certainty, self.count
        )
    }
}

impl From<(Transaction, f32, i64)> for ParsedTransaction {
    fn from(t: (Transaction, f32, i64)) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            amount: t.0.amount,
            time: t.0.time,
            is_fraud: t.1 > 0.90,
            certainty: t.1 * 100.,
            count: t.2,
        }
    }
}
