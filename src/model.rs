use crate::{
    dataset::Dataset,
    game::{Game, Policy},
};
use anyhow::Result;

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
        Ok(games
            .iter()
            .map(|game| self.select_move(*game))
            .collect::<Result<Vec<_>>>()?)
        /*
        let states: Vec<_> = games
            .iter()
            .map(|game| game.get_game_state_slice())
            .flatten()
            .collect();
        let tensor = Tensor::from_vec(states.to_vec(), (games.len(), I), &DEVICE)?;
        let masks = games
            .iter()
            .map(|game| {
                game.available_moves()
                    .map(|el| if el { 1.0 } else { 0.0 } as f32)
                    .to_vec()
            })
            .flatten()
            .collect();
        let move_mask = Tensor::from_vec(masks, (games.len(), N), &DEVICE)?;
        let pred = self.model.forward(&tensor)? * move_mask;
        let pred = pred?;
        let next_move = pred.argmax(0)?.to_vec1::<u32>()?;
        Ok(next_move.iter().map(|el| *el as usize).collect())
        */
    }
}
