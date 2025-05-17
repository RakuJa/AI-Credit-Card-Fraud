use crate::{
    comms::command::Command,
    data::{parsed_transaction::ParsedTransaction, transaction::Transaction},
};
use fake::{Fake, Faker};
use flume::{Receiver, Sender};
use log::debug;
#[cfg(not(feature = "lightgbm"))]
use ndarray::Array;
#[cfg(not(feature = "lightgbm"))]
use ort::session::Session;

#[cfg(feature = "lightgbm")]
pub fn predict(
    model_path: &str,
    tx_transaction: &Sender<ParsedTransaction>,
    rx_command: &Receiver<Command>,
) {
    use lightgbm3::Booster;
    let mut file_path = model_path.to_owned();
    file_path.push_str("lgb.txt");
    let bst = Booster::from_file(file_path.as_str()).unwrap();
    println!("Loaded prediction model from {}", model_path);
    debug!("Starting prediction");
    let mut is_predict_running = true;
    let mut count = 1;
    loop {
        if is_predict_running {
            let created_transaction: Transaction = Faker.fake();
            let features: Vec<f32> = created_transaction.clone().into();
            let n_features = i32::try_from(features.len()).unwrap();
            let y_pred = bst
                .predict_with_params(&features, n_features, true, "num_threads=1")
                .unwrap()[0] as f32;
            let transaction = ParsedTransaction::from((created_transaction, y_pred, count));
            count += 1;
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

#[cfg(not(feature = "lightgbm"))]
pub fn predict(
    model_path: &str,
    tx_transaction: &Sender<ParsedTransaction>,
    rx_command: &Receiver<Command>,
) {
    let mut file_path = model_path.to_owned();
    file_path.push_str("model.onnx");
    let session = Session::builder()
        .unwrap()
        .commit_from_file(file_path)
        .unwrap();

    debug!("Starting prediction");
    let mut is_predict_running = true;
    let mut count = 1;
    loop {
        if is_predict_running {
            let created_transaction: Transaction = Faker.fake();
            let features: Vec<f32> = created_transaction.clone().into();

            let input = Array::from_shape_vec((1, 30), features).unwrap();
            let ort_input = ort::inputs! {
                "input" => input
            }
            .unwrap();
            let outputs = session.run(ort_input).unwrap();
            let prob_view = outputs[1].try_extract_tensor::<f32>().unwrap();
            let class_view = outputs[0].try_extract_tensor::<i64>().unwrap();
            let probabilities =   // Extract as tensor
                prob_view.view()
                .into_shape_with_order(2).unwrap(); // Reshape to known size

            let classes =   // Extract as tensor
                class_view.view()
                    .into_shape_with_order(1).unwrap(); // Reshape to known size
            let class_pred = classes[0];
            let fraud_prob = if class_pred == 1 {
                0.5 + probabilities[0]
            } else {
                1.0 - probabilities[0]
            };
            let transaction = ParsedTransaction::from((created_transaction, fraud_prob, count));
            count += 1;
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
