use std::sync::{Arc, Mutex};

use egui::{CentralPanel, Frame, Margin, ScrollArea};
use egui_virtual_list::VirtualList;
use flume::{Receiver, Sender};
use log::debug;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
    comms::{
        command::Command,
        shared_state::{SharedState, TransactionLabel},
    },
    data::parsed_transaction::ParsedTransaction,
};

pub struct MainApp {
    shared_state: Arc<Mutex<SharedState>>,
    command_sender: Sender<Command>,
    is_predict_running: bool,
    update_receiver: Receiver<ParsedTransaction>,
    virtual_list: VirtualList,
}

impl MainApp {
    pub fn new(tx: &Sender<Command>, rx: Receiver<ParsedTransaction>) -> Self {
        let x = Self {
            shared_state: Arc::new(Mutex::new(SharedState::default())),
            command_sender: tx.clone(),
            is_predict_running: true,
            update_receiver: rx,
            virtual_list: VirtualList::default(),
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
            state.current_transaction.uuid = msg.uuid;
            state.current_transaction.amount = msg.amount;
            state.current_transaction.time = msg.time;
            state.current_transaction.fraud = msg.fraud;
        }
        ctx.request_repaint();
        CentralPanel::default().show(ctx, |ui| {
            ui.heading("Card Fraud Detector");

            let mut state = self.shared_state.lock().unwrap();

            if state.current_transaction.fraud {
                let new_item = state.current_transaction.clone();
                state
                    .transaction_history
                    .entry(TransactionLabel::FRAUD)
                    .and_modify(|x| x.push(new_item));
            }

            debug!("Parsed: {}", state.current_transaction.amount);
            ui.label(format!(
                "Transaction ID: '{}'",
                state.current_transaction.uuid
            ));
            ui.label(format!(
                "Transaction Amount: '{}'",
                state.current_transaction.amount
            ));
            ui.label(format!(
                "Transaction Time: '{}'",
                state.current_transaction.time
            ));
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
            let items = state
                .transaction_history
                .clone()
                .get(&TransactionLabel::FRAUD)
                .unwrap()
                .clone();
            drop(state);
            ScrollArea::vertical().show(ui, |ui| {
                ui.set_width(ui.available_width());
                self.virtual_list
                    .ui_custom_layout(ui, items.len(), |ui, start_index| {
                        let item = &items[start_index];

                        // For the sake of the example we generate a random height based on the item
                        // index but if your row height e.g. depends on some
                        // text with varying rows this would also work.
                        let mut rng = StdRng::seed_from_u64(100);
                        let height = rng.random_range(0..=100);

                        Frame::canvas(ui.style())
                            .inner_margin(Margin::symmetric(16, 8 + height / 2))
                            .show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                ui.label(format!("Item {item}"));
                            });

                        // Return the amount of items that were rendered this row,
                        // so you could vary the amount of items per row
                        1
                    });
            });

            ui.image(egui::include_image!("../../ui/icons/rustacean-banner.png"));
        });
    }
}
