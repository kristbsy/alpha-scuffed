use anyhow::{ensure, Result};
use ego_tree::{iter::Children, NodeId, NodeMut, NodeRef, Tree};
use itertools::Itertools;
use ordered_float::NotNan;
use rand::seq::IteratorRandom;
use rand::seq::SliceRandom;

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum SimpleBoardState {
    Empty,
    Player,
    Opponent,
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum Players {
    Player,
    Opponent,
}

#[derive(PartialEq, Eq)]
pub enum GameResult {
    Win,
    Loss,
    Tie,
}

impl TryFrom<SimpleBoardState> for Players {
    type Error = anyhow::Error;

    fn try_from(value: SimpleBoardState) -> Result<Self> {
        ensure!(
            value != SimpleBoardState::Empty,
            "Cannot get player from empty space"
        );
        if value == SimpleBoardState::Player {
            return Ok(Players::Opponent);
        }
        Ok(Players::Player)
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

pub trait Game: Clone {
    fn winning_player(&self) -> Option<Players>;
    fn available_moves(&self) -> Vec<bool>;
    fn perform_move(&mut self, space: usize);
    fn new() -> Self;
    fn game_ended(&self) -> bool;
    fn next_player(&self) -> Players;
}

#[derive(Debug, Clone)]
pub struct Checkers {
    // 3x3 board
    // Indices:
    // 0 1 2
    // 3 4 5
    // 6 7 8
    board: Vec<SimpleBoardState>,
    next_player: Players,
}

impl Checkers {
    fn print(&self) {
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
        let next_player = match self.next_player {
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

impl Game for Checkers {
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

    fn available_moves(&self) -> Vec<bool> {
        self.board
            .iter()
            .map(|square| *square == SimpleBoardState::Empty)
            .collect()
    }

    fn perform_move(&mut self, space: usize) {
        assert!(self.board[space] == SimpleBoardState::Empty);
        self.board[space] = self.next_player.into();
        if self.next_player == Players::Opponent {
            self.next_player = Players::Player;
        } else {
            self.next_player = Players::Opponent;
        }
    }

    fn new() -> Self {
        Self {
            board: vec![SimpleBoardState::Empty; 9],
            next_player: Players::Player,
        }
    }

    fn game_ended(&self) -> bool {
        let no_moves_available = !self.available_moves().iter().any(|x| *x);
        let winner = self.winning_player().is_some();
        winner || no_moves_available
    }

    fn next_player(&self) -> Players {
        self.next_player
    }
}

#[allow(unused)]
fn run_random_checkers() {
    let mut game = Checkers {
        board: vec![SimpleBoardState::Empty; 9],
        next_player: Players::Player,
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

struct MCTSData<T: Game> {
    game: T,
    visits: usize,
    score: f32,
    source_move: Option<usize>,
}

impl<T: Game> MCTSData<T> {
    fn new(game: T) -> Self {
        Self {
            game,
            visits: 0,
            score: 0.,
            source_move: None,
        }
    }
}

fn move_indices<T: Game>(game: &T) -> Vec<usize> {
    return game
        .available_moves()
        .iter()
        .enumerate()
        .filter(|(_, x)| **x)
        .map(|x| x.0)
        .collect();
}

fn expand<T: Game>(mut node: NodeMut<'_, MCTSData<T>>) {
    let game = &node.value().game.clone();
    let moves = move_indices(game);
    for mv in moves {
        let mut new_game = game.clone();
        new_game.perform_move(mv);
        let data = MCTSData::<T> {
            game: new_game,
            visits: 0,
            score: 0.,
            source_move: Some(mv),
        };
        node.append(data);
    }
}

fn backprop<T: Game>(mut node: NodeMut<'_, MCTSData<T>>, points: f32) {
    const DECAY: f32 = 0.9;
    node.value().visits += 1;
    // TODO: figure out why flipped
    node.value().score -= points;
    let is_root = node.parent().is_none();
    if !is_root {
        backprop(node.parent().unwrap(), points * DECAY);
    }
}

fn ucb<T: Game>(node: NodeRef<'_, MCTSData<T>>) -> f32 {
    if node.value().visits == 0 {
        return f32::MAX;
    }
    const EXPLORATION_WEIGHT: f32 = 1.;
    let exploration_score = f32::sqrt(
        f32::ln(node.parent().unwrap().value().visits as f32) / node.value().visits as f32,
    ) * EXPLORATION_WEIGHT;
    return node.value().score / node.value().visits as f32 + exploration_score;
}

fn select_child<T: Game>(children: Children<MCTSData<T>>) -> NodeId {
    children
        .into_iter()
        .map(|children| (children.id(), children))
        .max_set_by_key(|(_, x)| NotNan::new(ucb(*x)).unwrap())
        .choose(&mut rand::thread_rng())
        .unwrap()
        .0
}

fn mcts<T: Game>(root_game: &T) -> usize {
    const SIMULATIONS: usize = 10000;
    let mut mcts_tree: Tree<MCTSData<T>> = Tree::new(MCTSData::new(root_game.clone()));

    for _ in 0..SIMULATIONS {
        let mut cur_node = mcts_tree.root();
        while cur_node.has_children() {
            let next_node_id = select_child(cur_node.children());
            cur_node = mcts_tree.get(next_node_id).unwrap();
        }

        let game = cur_node.value().game.clone();
        if game.game_ended() {
            let result = game.winning_player();
            let points = match result {
                Some(Players::Player) => 1.0,
                Some(Players::Opponent) => -1.0,
                None => 0.0,
            };
            let cur_node = mcts_tree.get_mut(cur_node.id()).unwrap();
            backprop(cur_node, points);
            continue;
        }

        let result = random_simulation(cur_node.value().game.clone());
        let points = match result {
            GameResult::Win => 1.,
            GameResult::Loss => -1.,
            GameResult::Tie => 0.,
        };

        let id = cur_node.id();
        let cur_node = mcts_tree.get_mut(id).unwrap();
        backprop(cur_node, points);
        let cur_node = mcts_tree.get_mut(id).unwrap();
        expand(cur_node);
    }

    for child in mcts_tree.root().children() {
        println!(
            "source: {}, visits: {}",
            child.value().source_move.unwrap(),
            child.value().visits
        )
    }

    mcts_tree
        .root()
        .children()
        .map(|thing| thing.value())
        .max_by_key(|x| x.visits)
        .unwrap()
        .source_move
        .unwrap()
}

fn random_simulation<T: Game + Clone>(game: T) -> GameResult {
    let mut game = game.clone();
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
    }
    match game.winning_player() {
        Some(Players::Player) => GameResult::Win,
        Some(Players::Opponent) => GameResult::Loss,
        None => GameResult::Tie,
    }
}

fn main() {
    let mut game = Checkers::new();
    while !game.game_ended() {
        let next_move = mcts(&game);
        game.perform_move(next_move);
        game.print();
        if game.game_ended() {
            break;
        }
        let random_move = game
            .available_moves()
            .iter()
            .enumerate()
            .filter(|x| *x.1)
            .choose(&mut rand::thread_rng())
            .unwrap()
            .0;
        game.perform_move(random_move);
        game.print();
    }
}
