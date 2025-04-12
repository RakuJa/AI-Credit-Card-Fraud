use std::collections::HashMap;

use crate::data::parsed_transaction::ParsedTransaction;

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum TransactionLabel {
    FRAUD,
    LEGIT,
}

impl From<TransactionLabel> for String {
    fn from(val: TransactionLabel) -> Self {
        Self::from(match val {
            TransactionLabel::FRAUD => "FRAUD",
            TransactionLabel::LEGIT => "LEGIT",
        })
    }
}

pub struct SharedState {
    pub current_transaction: ParsedTransaction,
    pub transaction_history: HashMap<TransactionLabel, Vec<ParsedTransaction>>,
}

impl Default for SharedState {
    fn default() -> Self {
        Self {
            current_transaction: ParsedTransaction::default(),
            transaction_history: maplit::hashmap! {
                TransactionLabel::LEGIT => Vec::new(),
                TransactionLabel::FRAUD => Vec::new(),
            },
        }
    }
}
