use std::fmt::Display;

use uuid::Uuid;

use crate::data::transaction::Transaction;

#[derive(Debug, Clone)]
pub struct ParsedTransaction {
    pub uuid: Uuid,
    pub amount: f64,
    pub time: f64,
    pub is_fraud: bool,
    pub certainty: f64,
}

impl ParsedTransaction {
    pub fn default() -> Self {
        Self {
            uuid: Uuid::default(),
            amount: 0.0,
            time: 0.0,
            is_fraud: false,
            certainty: 0.,
        }
    }
}

impl Display for ParsedTransaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "UUID: {}\nAmount: {}\nTime: {}\nCertainty: {:.3}%",
            self.uuid, self.amount, self.time, self.certainty
        )
    }
}

impl From<(Transaction, f64)> for ParsedTransaction {
    fn from(t: (Transaction, f64)) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            amount: t.0.amount,
            time: t.0.time,
            is_fraud: t.1 > 0.95,
            certainty: t.1,
        }
    }
}
