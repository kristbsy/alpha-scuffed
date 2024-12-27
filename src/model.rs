use crate::{
    dataset::Dataset,
    game::{Game, Policy},
};
use anyhow::{Ok, Result};

pub trait TrainableModel<const N: usize, const I: usize> {
    fn new() -> Result<Self>
    where
        Self: Sized;
    fn train(&mut self, dataset: Dataset<N, I>) -> Result<()>;
    fn predict(&self, state: [f32; I]) -> Result<([f32; N], f32)>;
    fn predict_moves(&self, state: [f32; I]) -> Result<[f32; N]>;
    fn predict_score(&self, state: [f32; I]) -> Result<f32>;
}

pub struct AiPolicy<const N: usize, const I: usize, M: TrainableModel<N, I>> {
    pub model: M,
}

impl<const N: usize, const I: usize, T: Game<N, I>, M: TrainableModel<N, I>> Policy<N, I, T>
    for AiPolicy<N, I, M>
{
    fn select_move(&self, game: &T) -> anyhow::Result<usize> {
        let state = game.get_game_state_slice();
        let move_mask: [f32; N] = game
            .available_moves()
            .map(|el| if el { 1.0 } else { 0.0 } as f32);
        let visits = self.model.predict_moves(state)?;
        let masked_visits: [f32; N] = visits
            .iter()
            .zip(move_mask)
            .map(|(a, b)| a * b)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let next_move = masked_visits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .expect("NaN value encountered")
            .0;
        Ok(next_move)
    }

    fn select_moves_batch(&self, games: Vec<&T>) -> anyhow::Result<Vec<usize>> {
        // TODO: use actual batching
        Ok(games
            .iter()
            .map(|game| self.select_move(*game))
            .collect::<Result<Vec<_>>>()?)
    }

    fn predict_score(&self, game: &T) -> anyhow::Result<f32> {
        let state = game.get_game_state_slice();
        let score = self.model.predict_score(state)?;
        Ok(score)
    }

    fn can_predict_score(&self) -> bool {
        true
    }
}
