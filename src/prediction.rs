use fake::{Fake, Faker};
use flume::{Receiver, Sender};
use log::debug;

use crate::{
    comms::command::Command,
    data::{parsed_transaction::ParsedTransaction, transaction::Transaction},
};

pub fn predict(
    model_path: &str,
    tx_transaction: &Sender<ParsedTransaction>,
    rx_command: &Receiver<Command>,
) {
    use lightgbm3::Booster;

    let bst = Booster::from_file(model_path).unwrap();
    debug!("Starting prediction");
    let mut is_predict_running = true;
    loop {
        if is_predict_running {
            let created_transaction: Transaction = Faker.fake();
            let features: Vec<f64> = created_transaction.clone().into();
            let n_features = i32::try_from(features.len()).unwrap();

            let y_pred = bst
                .predict_with_params(&features, n_features, true, "num_threads=1")
                .unwrap()[0];

            let transaction = if y_pred > 0.9 {
                ParsedTransaction::from((created_transaction, true))
            } else {
                ParsedTransaction::from((created_transaction, false))
            };
            debug!("Sent: {transaction:?}");
            tx_transaction.send(transaction).unwrap();
        }
        let x = rx_command.recv().unwrap();
        is_predict_running = match x {
            Command::STOP | Command::PAUSE => false,
            Command::START | Command::RESUME => true,
        };
    }
}
