use crate::comms::command::Command;
use crate::data::parsed_transaction::ParsedTransaction;
use crate::data::transaction::Transaction;
use fake::{Fake, Faker};
use log::debug;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;

pub fn predict(
    model_path: &str,
    tx_transaction: &Sender<ParsedTransaction>,
    rx_command: &Receiver<Command>,
) {
    use lightgbm3::Booster;

    let bst = Booster::from_file(model_path).unwrap();
    debug!("Starting prediction");
    let mut create_new_data = true;
    loop {
        if create_new_data {
            let created_transaction: Transaction = Faker.fake();
            let features: Vec<f64> = created_transaction.clone().into();
            let n_features = features.len();

            let y_pred = bst
                .predict_with_params(&features, n_features as i32, true, "num_threads=1")
                .unwrap()[0];
            debug!("Sent: {y_pred}");
            let transaction = if y_pred > 0.9 {
                ParsedTransaction::from((created_transaction, true))
            } else {
                ParsedTransaction::from((created_transaction, false))
            };
            tx_transaction.send(transaction).unwrap();
            match rx_command.recv().unwrap() {
                Command::STOP | Command::PAUSE => {
                    create_new_data = false;
                }
                Command::START | Command::RESUME => {
                    create_new_data = true;
                }
            }
        }
    }
}
