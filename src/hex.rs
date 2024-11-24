use std::{default, fmt::Display};

use itertools::Itertools;
use tinyvec::ArrayVec;

use crate::game::{Game, Players, SimpleBoardState};

#[derive(Clone, Copy)]
pub struct Hex<const T: usize, const U: usize> {
    // note that T is the total squares, not the width due to constraints in const generics
    // The board is hexagonal, which can be represented as a skewed square
    // Determining which parts are connected is not trivial
    board: [SimpleBoardState; T],
    current_player: Players,
    side_length: usize,
    winning_player: Option<Players>,
    game_ended: bool,
}

impl<const T: usize, const U: usize> Hex<T, U> {
    fn check_connection(&self, index: isize) -> Option<u8> {
        if index >= 0 && index < T as isize {
            Some(index as u8)
        } else {
            None
        }
    }

    fn get_connections(&self, index: usize) -> ArrayVec<[u8; 6]> {
        let mut out = ArrayVec::<[u8; 6]>::default();
        let coords = self.coordinates(index);
        let index = index as isize;
        let width = self.side_length as isize;
        // false negative
        let upper_left_wall = coords.0 == 0;
        let lower_left_wall = coords.1 == self.side_length - 1;
        let left_wall = upper_left_wall || lower_left_wall;
        //false positive
        let upper_right_wall = coords.1 == 0;
        let lower_right_wall = coords.0 == self.side_length - 1;
        let right_wall = upper_right_wall || lower_right_wall;

        //upper left connection
        if !upper_left_wall {
            if let Some(connection) = self.check_connection(index - 1) {
                out.push(connection);
            }
        };
        //upper right connection
        if !upper_right_wall {
            if let Some(connection) = self.check_connection(index - width) {
                out.push(connection);
            }
        };
        //left connection
        if !left_wall {
            if let Some(connection) = self.check_connection(index + width - 1) {
                out.push(connection);
            }
        };
        //lower left connection
        if !lower_left_wall {
            if let Some(connection) = self.check_connection(index + width) {
                out.push(connection);
            }
        };
        //lower right connection
        if !lower_right_wall {
            if let Some(connection) = self.check_connection(index + 1) {
                out.push(connection);
            }
        };
        // right connection
        if !right_wall {
            if let Some(connection) = self.check_connection(index - width + 1) {
                out.push(connection);
            }
        };

        out
    }

    fn check_winning_player(&mut self) {
        // Find all placed pieces that touches player side, expand until touching opposing side or exhausted
        //     _
        //    /0\
        //   /3 1\
        //  /6 4 2\
        //   \7 5/
        //    \8/

        // Connections:
        //   \ /
        //  - O -
        //   / \
        // values: i - index for connections, w - board width
        //  Upper left: i - 1
        // Upper right: i - w
        //        left: i + w - 1
        //  Lower left: i + w
        // Lower right: i + 1
        //       right: i - w + 1
        /*
                if self
                    .board
                    .iter()
                    .filter(|square| **square != SimpleBoardState::Empty)
                    .count()
                    < self.side_length * 2 - 1
                {
                    return;
                }
        */
        // Player upper left and lower right
        let players = [Players::Player, Players::Opponent];
        //println!();
        //let mut queue: Vec<usize> = Vec::new();
        let mut queue: ArrayVec<[u8; 64]> = ArrayVec::default();
        for player in players {
            //eprintln!("player = {:#?}", player);
            let mut initial_squares: ArrayVec<[u8; 64]> = if player == Players::Player {
                (0..self.side_length)
                    .map(|index| index * self.side_length)
                    .filter(|index| self.board[*index] == player.into())
                    .map(|index| index as u8)
                    .collect()
            } else {
                (0..self.side_length)
                    .filter(|index| self.board[*index] == player.into())
                    .map(|el| el as u8)
                    .collect()
            };
            //eprintln!("initial_squares = {:#?}", initial_squares);
            let target_squares: ArrayVec<[u8; 8]> = if player == Players::Player {
                (0..self.side_length)
                    .map(|index| (index * self.side_length + self.side_length - 1) as u8)
                    .collect()
            } else {
                (T - self.side_length..T).map(|i| i as u8).collect()
            };
            queue.append(&mut initial_squares);
            //eprintln!("target_squares = {:#?}", target_squares);

            let mut i = 0;
            while queue.len() > i {
                //eprintln!("queue = {:#?}", queue);
                let index = queue[i];
                let connections = self.get_connections(index as usize);
                let mut to_check: ArrayVec<_> = connections
                    .iter()
                    .copied()
                    .filter(|i| !queue.contains(i) && self.board[*i as usize] == player.into())
                    .collect();
                //eprintln!("index = {:#?}", index);
                //eprintln!("connections = {:#?}", connections);
                //eprintln!("to_check = {:#?}", to_check);

                for target_square in target_squares.iter().copied() {
                    if to_check.contains(&target_square)
                        && self.board[target_square as usize] == player.into()
                    {
                        self.winning_player = Some(player);
                        self.game_ended = true;
                        return;
                    }
                }
                queue.append(&mut to_check);
                i += 1;
            }
            queue.clear();
        }

        self.game_ended = false;
        self.winning_player = None;
    }
    fn coordinates(&self, index: usize) -> (usize, usize) {
        let x = index % self.side_length;
        let y = index / self.side_length;
        (x, y)
    }
    fn index(&self, x: usize, y: usize) -> usize {
        x + y * self.side_length
    }
}

impl<const T: usize, const U: usize> Game<T, U> for Hex<T, U> {
    fn winning_player(&self) -> Option<Players> {
        self.winning_player
    }

    fn available_moves(&self) -> [bool; T] {
        self.board
            .iter()
            .cloned()
            .map(|state| state == SimpleBoardState::Empty)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn perform_move(&mut self, space: usize) {
        assert!(
            self.board[space] == SimpleBoardState::Empty,
            "Tried to make move on occupied hex"
        );
        self.board[space] = self.current_player.into();
        self.current_player = self.current_player.swap();
        self.check_winning_player();
    }

    fn new() -> Self {
        let sqrt = (T as f64).sqrt() as usize;
        assert!(
            T * 2 == U,
            "Bad dimensions on hex generics, U has to equal T*2"
        );
        assert_eq!(sqrt * sqrt, T, "T must be a perfect square");
        Self {
            board: [SimpleBoardState::Empty; T],
            current_player: Players::Player,
            side_length: sqrt,
            winning_player: None,
            game_ended: false,
        }
    }

    fn game_ended(&self) -> bool {
        self.game_ended
    }

    fn current_player(&self) -> Players {
        self.current_player
    }

    fn flip_board(&mut self) {
        //     _
        //    /0\
        //   /3 1\
        //  /6 4 2\
        //   \7 5/
        //    \8/
        //     _
        //    /0\
        //   /1 3\
        //  /2 4 6\
        //   \5 7/
        //    \8/
        let width = self.side_length;
        let mut out: [SimpleBoardState; T] = [SimpleBoardState::Empty; T];
        for i in 0..width {
            // in chunk index
            for j in 0..width {
                //chunk index
                out[i * width + j] = self.board[j * width + i];
            }
        }
        out = out.map(|el| el.swap());
        self.board = out;
        self.current_player = self.current_player.swap();
    }

    fn get_game_state_slice(&self) -> [f32; U] {
        // This should never fail since U == T * 2
        self.board
            .iter()
            .cloned()
            .map(|space| space.simple_state())
            .flatten()
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}

impl<const T: usize, const U: usize> Display for Hex<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        //          (0,0)                sum 0
        //       (0,1) (1,0)             sum 1
        //    (0,2) (1,1) (2,0)          sum 2
        // (0,3) (1,2) (2,1) (3,0)       sum 3
        //    (1,3) (2,2) (3,1)          sum 4
        //       (2,3) (3,2)             sum 5
        //          (3,3)                sum 6
        // The sum of x and y equals height, generate all number combinations that create the sum
        // variables: h, side_length
        // widest at h == side_length - 1 (h starts at 0)
        // amount = side_length - abs(h + 1 - side_length)
        // start = (0, h) if h < side_length else (h - side_length + 1, 3)
        let height = self.side_length * 2 - 1;
        let stride = self.side_length - 1;
        for h in 0..height {
            let start_index = if h < self.side_length {
                h * self.side_length
            } else {
                self.side_length * self.side_length - self.side_length + h - (self.side_length - 1)
            };
            let middle_distance = (h as isize + 1 - self.side_length as isize).abs();
            let amount: usize = (self.side_length as isize - middle_distance)
                .try_into()
                .unwrap();
            let mut indices: Vec<usize> = Vec::with_capacity(amount);
            for i in 0..amount {
                indices.push(start_index - stride * i);
            }
            let states: Vec<_> = indices.iter().map(|index| self.board[*index]).collect();
            let chars: Vec<_> = states
                .iter()
                .map(|state| match state {
                    SimpleBoardState::Empty => " ",
                    SimpleBoardState::Player => "X",
                    SimpleBoardState::Opponent => "O",
                })
                .collect();
            let mut row_string = String::from(if h <= height / 2 { "/" } else { "\\" });
            let padding = String::from(" ".repeat(middle_distance.try_into().unwrap()));
            for i in 0..chars.len() {
                row_string.push_str(chars[i]);
                if i < chars.len() - 1 {
                    row_string.push_str(" ");
                } else {
                    if h <= height / 2 {
                        row_string.push_str("\\");
                    } else {
                        row_string.push_str("/");
                    }
                }
            }
            writeln!(f, "{padding}{row_string}")?;
        }
        Ok(())
    }
}
