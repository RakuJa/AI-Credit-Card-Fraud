use crate::comms::command::Command;
use crate::comms::shared_state::SharedState;
use crate::data::parsed_transaction::ParsedTransaction;
use log::debug;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

pub struct MainApp {
    shared_state: Arc<Mutex<SharedState>>,
    command_sender: Sender<Command>,
    update_receiver: Receiver<ParsedTransaction>,
}

impl MainApp {
    pub fn new(tx: Sender<Command>, rx: Receiver<ParsedTransaction>) -> Self {
        Self {
            shared_state: Arc::new(Mutex::new(SharedState::default())),
            command_sender: tx,
            update_receiver: rx,
        }
    }
}

impl eframe::App for MainApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(msg) = self.update_receiver.try_recv() {
            let mut state = self.shared_state.lock().unwrap();
            debug!("Received: {}", msg.amount);
            state.uuid = msg.uuid;
            state.amount = msg.amount;
            state.time = msg.time;
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Card Fraud Detector");

            let state = self.shared_state.lock().unwrap();
            debug!("Parsed: {}", state.amount);
            ui.label(format!("Transaction ID: '{}'", state.uuid));
            ui.label(format!("Transaction Amount: '{}'", state.amount));
            ui.label(format!("Transaction Time: '{}'", state.time));
            if ui.button("Stop").clicked() {
                let _ = self.command_sender.send(Command::STOP);
            } else {
                let _ = self.command_sender.send(Command::STOP);
            }

            ui.image(egui::include_image!("../../ui/icons/rustacean-banner.png"));
        });
    }
}
