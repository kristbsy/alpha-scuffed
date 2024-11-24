use std::{fmt::Display, fs};

use serde::{Deserialize, Serialize};

use crate::{
    candle_ai::softmax,
    game::{Game, Policy},
    mcts,
};

#[derive(Clone)]
pub struct Dataset<const N: usize, const I: usize> {
    pub game_states: Vec<[f32; I]>,
    pub visit_stats: Vec<[f32; N]>,
    pub scores: Vec<f32>,
}

// TODO: remove Display requirement
pub fn create_dataset<
    const N: usize,
    const I: usize,
    T: Game<N, I> + Display,
    U: Policy<N, I, T>,
>(
    num_games: usize,
    policy: U,
) -> anyhow::Result<Dataset<N, I>> {
    let mut game_states: Vec<[f32; I]> = Vec::new();
    let mut scores: Vec<f32> = Vec::new();
    let mut visit_stats: Vec<[f32; N]> = Vec::new();
    for i in 0..num_games {
        let mut game = T::new();
        let mut flipped = false;
        if i == num_games - 1 {
            println!("{}", game);
        }
        while !game.game_ended() {
            let stats = mcts::<N, I, T, U>(&game, &policy)?;
            let visit_stat = stats.node_visits;
            let state = stats.game_state;
            game.perform_move(stats.best_move_index);
            game.flip_board();
            game_states.push(state);
            scores.push(stats.score);
            visit_stats.push(visit_stat);
            flipped = !flipped;
        }
        if i % 10 == 0 {
            println!("Simulated {} games", i);
        }
        if flipped {
            game.flip_board();
        }
        println!("{}", game);
    }
    visit_stats = softmax(visit_stats)?;
    Ok(Dataset {
        game_states,
        scores,
        visit_stats,
    })
}

impl<const N: usize, const I: usize> From<SerializableDataset<N, I>> for Dataset<N, I> {
    fn from(value: SerializableDataset<N, I>) -> Self {
        let mut x: Vec<[f32; I]> = Vec::new();
        let mut y: Vec<[f32; N]> = Vec::new();

        assert!(
            value.states_width == I,
            "wrong x-dimension on loaded dataset, expected {}, got {}",
            I,
            value.states_width
        );
        assert!(
            value.visits_width == N,
            "wrong x-dimension on loaded dataset, expected {}, got {}",
            N,
            value.visits_width
        );

        for chunk in value.game_states.chunks_exact(I) {
            let mut next = [0f32; I];
            next[..I].copy_from_slice(&chunk[..I]);
            x.push(next);
        }
        for chunk in value.node_visits.chunks_exact(N) {
            let mut next = [0f32; N];
            next[..N].copy_from_slice(&chunk[..N]);
            y.push(next);
        }

        Dataset {
            game_states: x,
            visit_stats: y,
            scores: value.scores,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct SerializableDataset<const N: usize, const I: usize> {
    game_states: Vec<f32>,
    node_visits: Vec<f32>,
    scores: Vec<f32>,
    states_width: usize,
    visits_width: usize,
}

impl<const N: usize, const I: usize> From<Dataset<N, I>> for SerializableDataset<N, I> {
    fn from(value: Dataset<N, I>) -> Self {
        let flat_x = value.game_states.iter().cloned().flatten().collect();
        let flat_y = value.visit_stats.iter().cloned().flatten().collect();
        SerializableDataset {
            game_states: flat_x,
            node_visits: flat_y,
            scores: value.scores,
            states_width: I,
            visits_width: N,
        }
    }
}

pub fn save_dataset<const N: usize, const I: usize>(
    data: &SerializableDataset<N, I>,
    name: String,
) {
    let data_json = serde_json::to_string_pretty(&data).unwrap();
    fs::write(format!("./{}.json", name), data_json).unwrap();
}
