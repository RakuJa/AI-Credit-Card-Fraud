use crate::data::transaction::Transaction;
use uuid::Uuid;

#[derive(Debug)]
pub struct ParsedTransaction {
    pub uuid: Uuid,
    pub amount: f64,
    pub time: f64,
    pub fraud: bool,
}

impl From<Transaction> for ParsedTransaction {
    fn from(t: Transaction) -> Self {
        Self::from((t, false))
    }
}

impl From<(Transaction, bool)> for ParsedTransaction {
    fn from(t: (Transaction, bool)) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            amount: t.0.amount,
            time: t.0.time,
            fraud: t.1,
        }
    }
}
