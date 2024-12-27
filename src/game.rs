use std::any;

use anyhow::{ensure, Result};
use rand::seq::IteratorRandom;

use crate::mcts::GameStats;

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum SimpleBoardState {
    Empty,
    Player,
    Opponent,
}

impl SimpleBoardState {
    pub fn simple_state(&self) -> [f32; 2] {
        if *self == Self::Empty {
            [0.0, 0.0]
        } else if *self == Self::Player {
            [1.0, 0.0]
        } else {
            [0.0, 1.0]
        }
    }

    /// Swaps player and opponent, empty stays the same
    pub fn swap(&self) -> Self {
        match &self {
            SimpleBoardState::Empty => SimpleBoardState::Empty,
            SimpleBoardState::Player => Self::Opponent,
            SimpleBoardState::Opponent => Self::Player,
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum Players {
    Player,
    Opponent,
}

impl Players {
    pub fn swap(&self) -> Self {
        match self {
            Players::Player => Players::Opponent,
            Players::Opponent => Players::Player,
        }
    }
}

#[derive(PartialEq, Eq)]
pub enum GameResult {
    Win,
    Loss,
    Tie,
}

impl GameResult {
    pub fn points(&self) -> f32 {
        match self {
            GameResult::Win => 1.0,
            GameResult::Loss => -1.0,
            GameResult::Tie => 0.0,
        }
    }
}

pub fn move_indices<const N: usize, const I: usize, T: Game<N, I>>(game: &T) -> Vec<usize> {
    return game
        .available_moves()
        .iter()
        .enumerate()
        .filter(|(_, x)| **x)
        .map(|x| x.0)
        .collect();
}

impl TryFrom<SimpleBoardState> for Players {
    type Error = anyhow::Error;

    fn try_from(value: SimpleBoardState) -> Result<Self> {
        ensure!(
            value != SimpleBoardState::Empty,
            "Cannot get player from empty space"
        );
        if value == SimpleBoardState::Player {
            return Ok(Players::Player);
        }
        Ok(Players::Opponent)
    }
}

impl From<Players> for SimpleBoardState {
    fn from(value: Players) -> Self {
        if value == Players::Opponent {
            return SimpleBoardState::Opponent;
        }
        SimpleBoardState::Player
    }
}

pub trait Game<const N: usize, const I: usize>: Clone {
    fn winning_player(&self) -> Option<Players>;
    fn available_moves(&self) -> [bool; N];
    fn perform_move(&mut self, space: usize);
    fn new() -> Self;
    fn game_ended(&self) -> bool;
    fn current_player(&self) -> Players;
    fn flip_board(&mut self);
    fn get_game_state_slice(&self) -> [f32; I];
    fn get_game_variations(stats: &GameStats<N, I>) -> Vec<GameStats<N, I>>;
}

pub trait Policy<const N: usize, const I: usize, T: Game<N, I>> {
    fn select_move(&self, game: &T) -> anyhow::Result<usize>;
    fn select_moves_batch(&self, games: Vec<&T>) -> anyhow::Result<Vec<usize>>;
    fn predict_score(&self, game: &T) -> anyhow::Result<f32>;
    fn can_predict_score(&self) -> bool;
}

pub struct RandomPolicy {}

impl<const N: usize, const I: usize, T: Game<N, I>> Policy<N, I, T> for RandomPolicy {
    fn select_move(&self, game: &T) -> anyhow::Result<usize> {
        let next_move = game
            .available_moves()
            .iter()
            .enumerate()
            .filter(|(_, available)| **available)
            .choose(&mut rand::thread_rng())
            .unwrap()
            .0;
        Ok(next_move)
    }

    fn select_moves_batch(&self, games: Vec<&T>) -> anyhow::Result<Vec<usize>> {
        games.iter().map(|game| self.select_move(*game)).collect()
    }

    fn predict_score(&self, game: &T) -> Result<f32> {
        todo!()
    }

    fn can_predict_score(&self) -> bool {
        false
    }
}
