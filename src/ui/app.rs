use crate::comms::command::Command;
use crate::comms::shared_state::SharedState;
use crate::data::parsed_transaction::ParsedTransaction;
use flume::{Receiver, Sender};
use log::debug;
use std::sync::{Arc, Mutex};

pub struct MainApp {
    shared_state: Arc<Mutex<SharedState>>,
    command_sender: Sender<Command>,
    is_predict_running: bool,
    update_receiver: Receiver<ParsedTransaction>,
}

impl MainApp {
    pub fn new(tx: &Sender<Command>, rx: Receiver<ParsedTransaction>) -> Self {
        let x = Self {
            shared_state: Arc::new(Mutex::new(SharedState::default())),
            command_sender: tx.clone(),
            is_predict_running: true,
            update_receiver: rx,
        };
        tx.send(Command::START).unwrap();
        x
    }
}

impl eframe::App for MainApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(msg) = self.update_receiver.try_recv() {
            let mut state = self.shared_state.lock().unwrap();
            debug!("Received: {msg:?}");
            state.uuid = msg.uuid;
            state.amount = msg.amount;
            state.time = msg.time;
        }
        ctx.request_repaint();
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Card Fraud Detector");

            let state = self.shared_state.lock().unwrap();
            debug!("Parsed: {}", state.amount);
            ui.label(format!("Transaction ID: '{}'", state.uuid));
            ui.label(format!("Transaction Amount: '{}'", state.amount));
            ui.label(format!("Transaction Time: '{}'", state.time));
            if !self.is_predict_running && ui.button("Continue").clicked() {
                let _ = self.command_sender.send(Command::RESUME);
                self.is_predict_running = true;
            }
            if self.is_predict_running {
                if ui.button("Pause").clicked() {
                    let _ = self.command_sender.send(Command::PAUSE);
                    self.is_predict_running = false;
                } else {
                    let _ = self.command_sender.send(Command::RESUME);
                    self.is_predict_running = true;
                }
            }
            ui.image(egui::include_image!("../../ui/icons/rustacean-banner.png"));
        });
    }
}
