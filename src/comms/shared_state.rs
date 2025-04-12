use uuid::Uuid;

pub struct SharedState {
    pub uuid: Uuid,
    pub amount: f64,
    pub time: f64,
}

impl Default for SharedState {
    fn default() -> Self {
        Self {
            uuid: Uuid::default(),
            amount: 0.0,
            time: 0.0,
        }
    }
}
