use std::sync::{Arc, Mutex};

use egui::{
    Button, CentralPanel, Color32, FontFamily, FontId, Frame, Margin, RichText, ScrollArea,
};
use egui_virtual_list::VirtualList;
use flume::{Receiver, Sender};
use log::debug;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::ui::font_handler::load_fonts;
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
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        tx: &Sender<Command>,
        rx: Receiver<ParsedTransaction>,
    ) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.
        load_fonts(&cc.egui_ctx);

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
        load_fonts(ctx);
        while let Ok(msg) = self.update_receiver.try_recv() {
            let mut state = self.shared_state.lock().unwrap();
            debug!("Received: {msg:?}");
            state.current_transaction.uuid = msg.uuid;
            state.current_transaction.amount = msg.amount;
            state.current_transaction.time = msg.time;
            state.current_transaction.is_fraud = msg.is_fraud;
            state.current_transaction.certainty = msg.certainty;
        }
        ctx.request_repaint();
        CentralPanel::default().show(ctx, |ui| {
            ctx.set_pixels_per_point(1.5);
            credits_panel(ui);
            ui.separator();
            let mut state = self.shared_state.lock().unwrap();
            current_transaction_title_panel(ui, state.current_transaction.is_fraud);
            if state.current_transaction.is_fraud {
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


            ui.separator();
            ui.horizontal(|ui| {
                if !self.is_predict_running {
                    let btn = Button::new(
                        RichText::new("Continue").color(Color32::BLACK)
                    ).fill(Color32::from_rgb(146, 130, 103));
                    if ui.add(btn).clicked() {
                        let _ = self.command_sender.send(Command::RESUME);
                        self.is_predict_running = true;
                    }
                }
                if self.is_predict_running {
                    let btn = Button::new(
                        RichText::new("Pause").color(Color32::BLACK)
                    ).fill(Color32::from_rgb(146, 130, 103));
                    if ui.add(btn).clicked() {
                        let _ = self.command_sender.send(Command::PAUSE);
                        self.is_predict_running = false;
                    } else {
                        let _ = self.command_sender.send(Command::RESUME);
                        self.is_predict_running = true;
                    }
                }
                ui.label(RichText::new("?").color(Color32::WHITE).size(15.).underline())
                    .on_hover_text("This app shows a continuous flow of transactions with only the most important values shown for visual clarity.")
                    .on_hover_text("Right below the current transaction, you can find a list of all the fraudulent ones identified.")
                    .on_hover_text("Click Pause if you want to stop the transaction flow. After pausing you can resume by clicking Continue");
            });


            ui.separator();
            ui.heading(
                RichText::new("List of fraudulent transactions")
                    .color(Color32::from_rgb(178, 161, 123))
            );
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
                        let height = rng.random_range(0..=50);

                        Frame::canvas(ui.style())
                            .inner_margin(Margin::symmetric(16, 8 + height / 2))
                            .show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                ui.label(format!("{item}"));
                            });

                        // Return the amount of items that were rendered this row,
                        // so you could vary the amount of items per row
                        1
                    });
            });
        });
    }
}

fn credits_panel(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.add(
                egui::Image::new(egui::include_image!("../../ui/icons/ferris.png"))
                    .maintain_aspect_ratio(true)
                    .max_width(100.0)
                    .corner_radius(10),
            );
        });
        ui.vertical(|ui| {
            ui.heading(
                RichText::new("Credit Card Fraud")
                    .color(Color32::BLACK)
                    .font(FontId {
                        size: 15.0,
                        family: FontFamily::Monospace, //FontFamily::Name("GoodTimesRg".into())
                    }),
            );
            ui.heading(
                RichText::new("Code By RakuJa")
                    .color(Color32::from_rgb(102, 0, 51))
                    .font(FontId {
                        size: 15.0,
                        family: FontFamily::Name("Pixelify".into()),
                    }),
            );
            ui.heading(
                RichText::new("Design By Meru")
                    .color(Color32::PURPLE)
                    .font(FontId {
                        size: 15.0,
                        family: FontFamily::Name("Pixelify".into()),
                    }),
            );
        })
    });
}

fn current_transaction_title_panel(ui: &mut egui::Ui, is_fraud: bool) {
    ui.horizontal(|ui| {
        ui.heading(
            RichText::new("Flow of current transactions".to_uppercase())
                .color(Color32::from_rgb(178, 161, 123)),
        );

        // Create the rectangle shape
        let mut rect = ui.available_rect_before_wrap();
        rect.set_width(10.0);
        rect.set_height(10.0);
        let shape = egui::Shape::rect_filled(
            rect,
            egui::CornerRadius::same(100),
            if is_fraud {
                Color32::DARK_RED
            } else {
                Color32::from_rgb(178, 161, 123)
            },
        );
        // Draw the shape
        ui.painter().add(shape);
    });
}
