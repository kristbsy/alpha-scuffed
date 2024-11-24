use ego_tree::{iter::Children, NodeId, NodeMut, NodeRef, Tree};
use itertools::Itertools;
use ordered_float::NotNan;
use rand::seq::SliceRandom;

use crate::game::{move_indices, Game, GameResult, Players, Policy};

struct MCTSData<const N: usize, const I: usize, T: Game<N, I>> {
    game: T,
    visits: usize,
    score: f32,
    source_move: Option<usize>,
}

impl<const N: usize, const I: usize, T: Game<N, I>> MCTSData<N, I, T> {
    fn new(game: T) -> Self {
        Self {
            game,
            visits: 0,
            score: 0.,
            source_move: None,
        }
    }
}

fn expand<const N: usize, const I: usize, T: Game<N, I>>(
    node: &mut NodeMut<'_, MCTSData<N, I, T>>,
) {
    let game = node.value().game.clone();
    let moves = move_indices(&game);
    for mv in moves {
        let mut new_game = game.clone();
        new_game.perform_move(mv);
        let data = MCTSData::<N, I, T> {
            game: new_game,
            visits: 0,
            score: 0.,
            source_move: Some(mv),
        };
        node.append(data);
    }
}

fn backprop<const N: usize, const I: usize, T: Game<N, I>>(
    node: &mut NodeMut<'_, MCTSData<N, I, T>>,
    points: f32,
) {
    const DECAY: f32 = 0.9;
    node.value().visits += 1;
    node.value().score += points;
    if node.parent().is_some() {
        backprop(&mut node.parent().unwrap(), points * DECAY);
    }
}

fn ucb<const N: usize, const I: usize, T: Game<N, I>>(
    node: NodeRef<'_, MCTSData<N, I, T>>,
) -> NotNan<f32> {
    if node.value().visits == 0 {
        return NotNan::new(f32::MAX).unwrap();
    }
    const EXPLORATION_WEIGHT: f32 = 10.;
    let exploration_score = f32::sqrt(
        f32::sqrt(node.parent().unwrap().value().visits as f32)
            / (node.value().visits as f32 + 1.0),
    ) * EXPLORATION_WEIGHT;
    let exploitation_score = node.value().score / node.value().visits as f32;
    return NotNan::new(exploitation_score + exploration_score).unwrap();
}

// Selects the child with the highest ucb score, random tie break
fn select_child<const N: usize, const I: usize, T: Game<N, I>>(
    children: Children<MCTSData<N, I, T>>,
) -> NodeId {
    children
        .into_iter()
        .map(|children| (children.id(), children))
        .max_set_by_key(|(_, x)| ucb(*x))
        .choose(&mut rand::thread_rng())
        .unwrap()
        .0
}

fn select_leaf<const N: usize, const I: usize, T: Game<N, I>>(
    tree: &Tree<MCTSData<N, I, T>>,
    node_id: NodeId,
) -> NodeId {
    let mut node = tree.get(node_id).unwrap();
    while node.has_children() {
        let next_node_id = select_child(node.children());
        node = tree.get(next_node_id).unwrap()
    }
    node.id()
}

pub fn mcts<const N: usize, const I: usize, T: Game<N, I>, U: Policy<N, I, T>>(
    root_game: &T,
    policy: &U,
) -> anyhow::Result<GameStats<N, I>> {
    const SIMULATIONS: usize = 1000;
    let mut mcts_tree: Tree<MCTSData<N, I, T>> = Tree::new(MCTSData::new(root_game.clone()));

    for _ in 0..SIMULATIONS {
        let mut cur_node = mcts_tree
            .get_mut(select_leaf(&mcts_tree, mcts_tree.root().id()))
            .unwrap();
        let game = &cur_node.value().game;

        if game.game_ended() {
            let result = game.winning_player();
            let points = match result {
                Some(Players::Player) => 1.0,
                Some(Players::Opponent) => -1.0,
                None => 0.0,
            };
            backprop(&mut cur_node, points);
            continue;
        }

        let result = simulate::<N, I, T, U>(game, policy, Players::Player)?;
        let points = result.points();

        expand(&mut cur_node);
        backprop(&mut cur_node, points);
    }
    Ok(get_tree_stats(&mcts_tree))
}

pub struct GameStats<const N: usize, const I: usize> {
    pub best_move_index: usize,
    pub game_state: [f32; I],
    pub node_visits: [f32; N],
    pub score: f32,
}

fn get_tree_stats<const N: usize, const I: usize, T: Game<N, I>>(
    tree: &Tree<MCTSData<N, I, T>>,
) -> GameStats<N, I> {
    let child_datas: Vec<_> = tree.root().children().map(|thing| thing.value()).collect();
    let score = tree.root().value().score;
    let mut visit_stats = [0.0_f32; N];
    for data in &child_datas {
        // Soundness: Only the root node is none, so source_move here should always be Some
        visit_stats[data.source_move.unwrap()] = data.visits as f32;
    }
    let best_move_index = child_datas
        .iter()
        .max_by_key(|x| x.visits)
        .unwrap()
        .source_move
        .unwrap();
    GameStats {
        best_move_index,
        node_visits: visit_stats,
        game_state: tree.root().value().game.get_game_state_slice(),
        score,
    }
}

pub fn simulate<const N: usize, const I: usize, T: Game<N, I>, U: Policy<N, I, T>>(
    game: &T,
    policy: &U,
    simulated_player: Players,
) -> anyhow::Result<GameResult> {
    let mut game = game.clone();
    while !game.game_ended() {
        let next_move = policy.select_move(&game)?;
        game.perform_move(next_move);
    }
    let winner = game.winning_player();
    if let Some(player) = winner {
        if player == simulated_player {
            Ok(GameResult::Win)
        } else {
            Ok(GameResult::Loss)
        }
    } else {
        Ok(GameResult::Tie)
    }
}
