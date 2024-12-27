use std::fmt::Display;

use anyhow::Ok;
use rand::seq::IteratorRandom;

use crate::{
    game::{Game, Players, SimpleBoardState},
    mcts::GameStats,
};

impl Checkers {
    pub fn print(&self) {
        let mut display_board = Vec::with_capacity(9);
        for space in &self.board {
            display_board.push(match space {
                SimpleBoardState::Empty => " ",
                SimpleBoardState::Player => "X",
                SimpleBoardState::Opponent => "O",
            });
        }
        let one = display_board[0];
        let two = display_board[1];
        let three = display_board[2];
        let four = display_board[3];
        let five = display_board[4];
        let six = display_board[5];
        let seven = display_board[6];
        let eight = display_board[7];
        let nine = display_board[8];
        println!("╔═╦═╦═╗");
        println!("║{one}║{two}║{three}║");
        println!("╠═╬═╬═╣");
        println!("║{four}║{five}║{six}║");
        println!("╠═╬═╬═╣");
        println!("║{seven}║{eight}║{nine}║");
        println!("╚═╩═╩═╝");
        let next_player = match self.current_player {
            Players::Player => "X",
            Players::Opponent => "O",
        };
        println!("Next player: {}", next_player);
    }

    fn validate_board_state(&self) {
        let player_pieces = self
            .board
            .iter()
            .filter(|square| **square == SimpleBoardState::Player)
            .count();
        let opponent_pieces = self
            .board
            .iter()
            .filter(|square| **square == SimpleBoardState::Opponent)
            .count();

        assert!(player_pieces.abs_diff(opponent_pieces) < 2);
    }
}

impl Display for Checkers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut display_board = Vec::with_capacity(9);
        for space in &self.board {
            display_board.push(match space {
                SimpleBoardState::Empty => " ",
                SimpleBoardState::Player => "X",
                SimpleBoardState::Opponent => "O",
            });
        }
        let one = display_board[0];
        let two = display_board[1];
        let three = display_board[2];
        let four = display_board[3];
        let five = display_board[4];
        let six = display_board[5];
        let seven = display_board[6];
        let eight = display_board[7];
        let nine = display_board[8];
        writeln!(f, "╔═╦═╦═╗")?;
        writeln!(f, "║{one}║{two}║{three}║")?;
        writeln!(f, "╠═╬═╬═╣")?;
        writeln!(f, "║{four}║{five}║{six}║")?;
        writeln!(f, "╠═╬═╬═╣")?;
        writeln!(f, "║{seven}║{eight}║{nine}║")?;
        writeln!(f, "╚═╩═╩═╝")?;
        let next_player = match self.current_player {
            Players::Player => "X",
            Players::Opponent => "O",
        };
        writeln!(f, "Next player: {}", next_player)
    }
}

impl Game<9, 18> for Checkers {
    fn winning_player(&self) -> Option<Players> {
        for i in 0..=2 {
            // Check rows
            let offset = i * 3;
            if self.board[offset] != SimpleBoardState::Empty
                && self.board[offset] == self.board[1 + offset]
                && self.board[1 + offset] == self.board[2 + offset]
            {
                return Some(self.board[offset].try_into().unwrap());
            }
            // Check columns
            if self.board[i] != SimpleBoardState::Empty
                && self.board[i] == self.board[3 + i]
                && self.board[3 + i] == self.board[6 + i]
            {
                return Some(self.board[i].try_into().unwrap());
            }
        }
        if self.board[0] != SimpleBoardState::Empty
            && self.board[0] == self.board[4]
            && self.board[4] == self.board[8]
        {
            return Some(self.board[0].try_into().unwrap());
        }
        if self.board[2] != SimpleBoardState::Empty
            && self.board[2] == self.board[4]
            && self.board[4] == self.board[6]
        {
            return Some(self.board[2].try_into().unwrap());
        }
        None
    }

    fn available_moves(&self) -> [bool; 9] {
        let mut moves: [bool; 9] = [false; 9];
        for i in 0..self.board.len() {
            moves[i] = self.board[i] == SimpleBoardState::Empty
        }
        moves
    }

    fn perform_move(&mut self, space: usize) {
        assert!(self.board[space] == SimpleBoardState::Empty);
        self.board[space] = self.current_player.into();
        self.current_player = match self.current_player {
            Players::Player => Players::Opponent,
            Players::Opponent => Players::Player,
        };
    }

    fn new() -> Self {
        Self {
            board: [SimpleBoardState::Empty; 9],
            current_player: Players::Player,
        }
    }

    fn game_ended(&self) -> bool {
        let no_moves_available = !self.available_moves().iter().any(|x| *x);
        let winner = self.winning_player().is_some();
        winner || no_moves_available
    }

    fn current_player(&self) -> Players {
        self.current_player
    }

    fn flip_board(&mut self) {
        let flipped_board = self.board.map(|square| square.swap());
        self.board = flipped_board;
        self.current_player = self.current_player.swap();
    }

    fn get_game_state_slice(&self) -> [f32; 18] {
        let mut out_slice = [0.0; 18];
        for i in 0..self.board.len() {
            out_slice[i] = match self.board[i] {
                SimpleBoardState::Player => 1.0,
                _ => 0.0,
            }
        }
        for i in 0..self.board.len() {
            out_slice[self.board.len() + i] = match self.board[i] {
                SimpleBoardState::Opponent => 1.0,
                _ => 0.0,
            }
        }
        out_slice
    }

    fn get_game_variations(stats: &GameStats<9, 18>) -> Vec<GameStats<9, 18>> {
        vec![stats.clone()]
    }
}

#[allow(unused)]
fn run_random_checkers() {
    let mut game = Checkers {
        board: [SimpleBoardState::Empty; 9],
        current_player: Players::Player,
    };
    while !game.game_ended() {
        let next_move = game
            .available_moves()
            .iter()
            .enumerate()
            .filter(|(_, available)| **available)
            .choose(&mut rand::thread_rng())
            .unwrap()
            .0;
        game.perform_move(next_move);
        game.validate_board_state();
    }
}

#[derive(Debug, Clone)]
pub struct Checkers {
    // 3x3 board
    // Indices:
    // 0 1 2
    // 3 4 5
    // 6 7 8
    board: [SimpleBoardState; 9],
    current_player: Players,
}
