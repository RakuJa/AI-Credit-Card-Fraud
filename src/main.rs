use std::{env, thread};

use eframe::egui;

use crate::{data::parsed_transaction::ParsedTransaction, ui::app::MainApp};

pub mod comms;
mod data;
mod prediction;
mod ui;

fn main() -> eframe::Result {
    dotenvy::dotenv().unwrap();
    let model_path = env::var("MODEL_PATH").expect("Error fetching model path");
    env_logger::init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([320.0, 240.0]),
        ..Default::default()
    };

    let (tx_command, rx_command) = flume::bounded(1);
    let (tx_transaction, rx_transaction) = flume::unbounded::<ParsedTransaction>();
    let app = MainApp::new(&tx_command, rx_transaction);

    thread::spawn(move || prediction::predict(model_path.as_str(), &tx_transaction, &rx_command));

    eframe::run_native(
        "MyApp",
        options,
        Box::new(|cc| {
            // This gives us image support:
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Ok(Box::new(app))
        }),
    )
}
